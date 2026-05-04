import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def _default_dino_repo() -> str:
    env_repo = os.environ.get("DINOV2_REPO", "").strip()
    if env_repo:
        return env_repo
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "dinov2",
        here.parents[2] / "dinov2",
        here.parents[3] / "dinov2" if len(here.parents) > 3 else None,
        Path.cwd() / "dinov2",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return str(candidate)
    return ""


def _clean_state_dict(raw_state):
    if isinstance(raw_state, dict):
        for key in ("model", "state_dict", "teacher", "student"):
            value = raw_state.get(key)
            if isinstance(value, dict):
                raw_state = value
                break
    if not isinstance(raw_state, dict):
        return raw_state

    cleaned = {}
    prefixes = (
        "module.",
        "backbone.",
        "encoder.",
        "teacher.",
        "student.",
    )
    for key, value in raw_state.items():
        new_key = str(key)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        cleaned[new_key] = value
    return cleaned


class _OnTheFlyDINOExtractor:
    def __init__(
        self,
        weight_path: Path,
        model_name: str = "dinov2_vitl14",
        repo_path: Optional[str] = None,
        device: str = "auto",
        max_side: int = 700,
    ):
        import torch

        self.torch = torch
        self.device = self._resolve_device(device)
        self.max_side = int(max(0, max_side))
        self.model_name = str(model_name or "dinov2_vitl14")
        self.repo_path = str(repo_path or "").strip()
        self.patch_size = 14 if "14" in self.model_name else 16
        self.model = self._load_model(Path(weight_path))
        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, device: str) -> str:
        torch = self.torch
        device = str(device or "auto").strip()
        if not device or device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return device

    def _load_model(self, weight_path: Path):
        torch = self.torch
        if self.repo_path and Path(self.repo_path).exists():
            model = torch.hub.load(self.repo_path, self.model_name, source="local", pretrained=False)
        else:
            model = torch.hub.load("facebookresearch/dinov2", self.model_name, pretrained=False)

        state = torch.load(str(weight_path), map_location="cpu")
        state = _clean_state_dict(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            print("Warn: DINO checkpoint missing {} keys when loading {}.".format(len(missing), weight_path))
        if len(unexpected) > 0:
            print("Warn: DINO checkpoint has {} unexpected keys when loading {}.".format(len(unexpected), weight_path))
        return model

    @staticmethod
    def _resize_for_dino(image_rgb: np.ndarray, max_side: int, patch_size: int) -> np.ndarray:
        h, w = image_rgb.shape[:2]
        scale = 1.0
        if max_side > 0 and max(h, w) > max_side:
            scale = float(max_side) / float(max(h, w))
        resized_h = max(patch_size, int(round(h * scale)))
        resized_w = max(patch_size, int(round(w * scale)))
        resized_h = max(patch_size, (resized_h // patch_size) * patch_size)
        resized_w = max(patch_size, (resized_w // patch_size) * patch_size)
        if resized_h == h and resized_w == w:
            return image_rgb
        return cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    def _image_to_tensor(self, image_rgb: np.ndarray):
        torch = self.torch
        image = np.asarray(image_rgb, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device, non_blocking=True)

    def _extract_feature_tensor(self, tensor):
        torch = self.torch
        with torch.no_grad():
            if hasattr(self.model, "get_intermediate_layers"):
                features = self.model.get_intermediate_layers(
                    tensor,
                    n=1,
                    reshape=True,
                    return_class_token=False,
                )[0]
            elif hasattr(self.model, "forward_features"):
                output = self.model.forward_features(tensor)
                if isinstance(output, dict) and "x_norm_patchtokens" in output:
                    patch_tokens = output["x_norm_patchtokens"]
                elif isinstance(output, dict) and "x_prenorm" in output:
                    patch_tokens = output["x_prenorm"][:, 1:]
                else:
                    raise RuntimeError("DINO forward_features output does not contain patch tokens")
                batch, token_count, channel_count = patch_tokens.shape
                feat_h = max(1, tensor.shape[-2] // self.patch_size)
                feat_w = max(1, token_count // feat_h)
                features = patch_tokens[:, : feat_h * feat_w, :].transpose(1, 2).reshape(batch, channel_count, feat_h, feat_w)
            else:
                raise RuntimeError("DINO model does not expose dense feature extraction APIs")
        return features

    def extract(self, image_rgb: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        import torch.nn.functional as F

        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_rgb must be HxWx3")
        resized = self._resize_for_dino(image, self.max_side, self.patch_size)
        tensor = self._image_to_tensor(resized)
        features = self._extract_feature_tensor(tensor)
        features = F.interpolate(
            features,
            size=(int(target_shape[0]), int(target_shape[1])),
            mode="bilinear",
            align_corners=False,
        )
        feature_map = features[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(feature_map, axis=2, keepdims=True)
        return feature_map / np.maximum(norms, 1e-6)
