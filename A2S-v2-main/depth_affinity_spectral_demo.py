import argparse
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.segmentation import slic


DEPTH_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PALETTE = [
    (255, 99, 71),
    (255, 215, 0),
    (0, 191, 255),
    (50, 205, 50),
    (255, 105, 180),
    (138, 43, 226),
    (255, 140, 0),
    (64, 224, 208),
]


def _candidate_local_dino_repos() -> List[Path]:
    project_root = Path(__file__).resolve().parent.parent
    home = Path.home()
    torch_home = Path(os.environ.get("TORCH_HOME", "")).expanduser() if os.environ.get("TORCH_HOME") else None
    xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", "")).expanduser() if os.environ.get("XDG_CACHE_HOME") else None
    candidates = [
        project_root / "dinov2",
        project_root / "facebookresearch_dinov2_main",
        project_root / "Depth-Anything-V2-main",
        project_root / ".." / "dinov2",
        project_root / ".." / "facebookresearch_dinov2_main",
        home / ".cache" / "torch" / "hub" / "facebookresearch_dinov2_main",
    ]
    if torch_home is not None:
        candidates.append(torch_home / "hub" / "facebookresearch_dinov2_main")
    if xdg_cache_home is not None:
        candidates.append(xdg_cache_home / "torch" / "hub" / "facebookresearch_dinov2_main")

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if str(candidate) in seen:
            continue
        seen.add(str(candidate))
        if (candidate / "hubconf.py").exists():
            unique.append(candidate)

    hub_search_roots = [home / ".cache" / "torch" / "hub"]
    if torch_home is not None:
        hub_search_roots.append(torch_home / "hub")
    if xdg_cache_home is not None:
        hub_search_roots.append(xdg_cache_home / "torch" / "hub")
    for root in hub_search_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob("*dinov2*")):
            candidate = candidate.resolve()
            if str(candidate) in seen:
                continue
            seen.add(str(candidate))
            if (candidate / "hubconf.py").exists():
                unique.append(candidate)
    return unique


def _default_dino_repo() -> str:
    repos = _candidate_local_dino_repos()
    return str(repos[0]) if repos else ""


def _suppress_optional_dino_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*xFormers is not available.*",
        category=UserWarning,
    )


@contextmanager
def _temporarily_hide_xformers_imports():
    sentinel_names = ["xformers", "xformers.ops", "xformers._C"]
    saved: Dict[str, object] = {}
    try:
        for name in sentinel_names:
            if name in sys.modules:
                saved[name] = sys.modules[name]
            sys.modules[name] = None
        yield
    finally:
        for name in sentinel_names:
            if name in saved:
                sys.modules[name] = saved[name]
            else:
                sys.modules.pop(name, None)


def _is_xformers_runtime_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return (
        "memory_efficient_attention" in lowered
        or ("no operator found" in lowered and "xformers" in lowered)
        or "xformers wasn't built with cuda support" in lowered
    )


def _normalize_gray(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[..., 0]
    min_value = float(image.min()) if image.size > 0 else 0.0
    max_value = float(image.max()) if image.size > 0 else 0.0
    if max_value - min_value <= 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return ((image - min_value) / (max_value - min_value)).astype(np.float32)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.round(_normalize_gray(image) * 255.0), 0.0, 255.0).astype(np.uint8)


def _resize_feature_map(feature_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    feature_map = np.asarray(feature_map, dtype=np.float32)
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    if feature_map.ndim == 2:
        feature_map = feature_map[..., None]
    if feature_map.shape[:2] == (target_h, target_w):
        return feature_map.astype(np.float32)
    resized_channels = [
        cv2.resize(feature_map[..., channel_idx], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        for channel_idx in range(feature_map.shape[2])
    ]
    return np.stack(resized_channels, axis=2).astype(np.float32)


def _prepare_feature_map(feature_map: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    feature_map = np.asarray(feature_map, dtype=np.float32)
    if feature_map.ndim == 2:
        feature_map = feature_map[..., None]
    elif feature_map.ndim == 3:
        if feature_map.shape[0] <= 32 and feature_map.shape[1] > 32 and feature_map.shape[2] > 32:
            feature_map = np.transpose(feature_map, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported feature-map shape: {feature_map.shape}")
    feature_map = _resize_feature_map(feature_map, target_shape)
    feature_map = np.nan_to_num(feature_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    denom = np.linalg.norm(feature_map, axis=2, keepdims=True)
    return feature_map / np.maximum(denom, 1e-6)


class _OnTheFlyDINOExtractor:
    def __init__(
        self,
        weight_path: Optional[Path],
        model_name: str,
        repo_path: str,
        device: str,
        max_side: int,
    ) -> None:
        try:
            import torch
        except Exception as exc:
            raise RuntimeError(f"PyTorch is required for DINO feature extraction: {exc}")
        self.torch = torch
        self.weight_path = Path(weight_path) if weight_path else None
        self.model_name = str(model_name)
        self.repo_path = str(repo_path).strip() or _default_dino_repo()
        self.device = self._resolve_device(device)
        self.max_side = max(0, int(max_side))
        self._xformers_forced_off = False
        self.model = self._load_model()
        self.model.eval().to(self.device)
        self.patch_size = self._infer_patch_size(self.model)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

    def _resolve_device(self, device: str):
        device = str(device).strip().lower()
        if device in {"", "auto"}:
            return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        return self.torch.device(device)

    def _load_model(self):
        source = "local" if self.repo_path else "github"
        repo = self.repo_path if self.repo_path else "facebookresearch/dinov2"
        try:
            with warnings.catch_warnings(), _temporarily_hide_xformers_imports():
                _suppress_optional_dino_warnings()
                model = self.torch.hub.load(repo, self.model_name, source=source, pretrained=False)
        except Exception as exc:
            if source == "github":
                local_repo_hint = _default_dino_repo()
                hint_suffix = (
                    f" Detected local fallback candidate: {local_repo_hint}. Try --dino-repo {local_repo_hint}"
                    if local_repo_hint
                    else ""
                )
                raise RuntimeError(
                    "Failed to build DINO model from GitHub. Provide --dino-repo pointing to a local repo clone, "
                    f"or make sure the environment can reach github.{hint_suffix} Details: {exc}"
                )
            raise RuntimeError(f"Failed to build DINO model from local repo {repo}: {exc}")

        if self.weight_path is not None:
            if not self.weight_path.exists():
                raise FileNotFoundError(f"DINO weight file not found: {self.weight_path}")
            payload = self.torch.load(str(self.weight_path), map_location="cpu")
            state_dict = self._extract_state_dict(payload)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warn: missing keys when loading DINO weights ({len(missing)} keys).")
            if unexpected:
                print(f"Warn: unexpected keys when loading DINO weights ({len(unexpected)} keys).")
        self._force_disable_xformers_runtime(model=model)
        return model

    def _extract_state_dict(self, payload: object) -> Dict[str, object]:
        if isinstance(payload, dict):
            for key in ["state_dict", "model", "teacher", "student", "network", "backbone"]:
                value = payload.get(key)
                if isinstance(value, dict) and value:
                    payload = value
                    break
        if not isinstance(payload, dict):
            raise RuntimeError("Unsupported checkpoint format for DINO weights.")
        cleaned: Dict[str, object] = {}
        for raw_key, value in payload.items():
            if not isinstance(raw_key, str):
                continue
            key = raw_key
            for prefix in ["module.", "model.", "backbone.", "teacher.", "student.", "encoder."]:
                if key.startswith(prefix):
                    key = key[len(prefix):]
            cleaned[key] = value
        return cleaned

    def _infer_patch_size(self, model) -> int:
        value = getattr(model, "patch_size", None)
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, tuple) and len(value) > 0 and int(value[0]) > 0:
            return int(value[0])
        patch_embed = getattr(model, "patch_embed", None)
        if patch_embed is not None:
            value = getattr(patch_embed, "patch_size", None)
            if isinstance(value, int) and value > 0:
                return int(value)
            if isinstance(value, tuple) and len(value) > 0 and int(value[0]) > 0:
                return int(value[0])
        return 14

    def _rounded_multiple(self, value: int, divisor: int) -> int:
        value = max(1, int(value))
        divisor = max(1, int(divisor))
        return max(divisor, int(round(float(value) / float(divisor))) * divisor)

    def _compute_scaled_shape(self, height: int, width: int) -> Tuple[int, int]:
        scale = min(1.0, float(self.max_side) / float(max(height, width))) if self.max_side > 0 else 1.0
        scaled_h = self._rounded_multiple(int(round(height * scale)), self.patch_size)
        scaled_w = self._rounded_multiple(int(round(width * scale)), self.patch_size)
        return scaled_h, scaled_w

    def get_batch_key(self, rgb_shape: Tuple[int, int]) -> Tuple[int, int]:
        return self._compute_scaled_shape(int(rgb_shape[0]), int(rgb_shape[1]))

    def _prepare_sample_tensor(self, rgb: np.ndarray):
        rgb = np.asarray(rgb, dtype=np.uint8)
        height, width = rgb.shape[:2]
        scaled_h, scaled_w = self._compute_scaled_shape(height, width)
        rgb_resized = cv2.resize(rgb, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        tensor = self.torch.from_numpy(rgb_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
        tensor = tensor.to(self.device)
        tensor = (tensor - self.mean[0]) / self.std[0]
        return tensor, (scaled_h, scaled_w)

    def _extract_tokens(self, outputs, grid_h: int, grid_w: int):
        torch = self.torch
        if isinstance(outputs, dict):
            for key in ["x_norm_patchtokens", "patch_tokens", "x_patchtokens", "patchtokens"]:
                value = outputs.get(key)
                if torch.is_tensor(value):
                    outputs = value
                    break
        if torch.is_tensor(outputs):
            if outputs.ndim == 4:
                return outputs
            if outputs.ndim == 3:
                if outputs.shape[1] == 1 + grid_h * grid_w:
                    outputs = outputs[:, 1:, :]
                if outputs.shape[1] == grid_h * grid_w:
                    return outputs.reshape(outputs.shape[0], grid_h, grid_w, outputs.shape[2]).permute(0, 3, 1, 2)
        if isinstance(outputs, (list, tuple)) and outputs:
            for value in outputs:
                tokens = self._extract_tokens(value, grid_h=grid_h, grid_w=grid_w)
                if tokens is not None:
                    return tokens
        return None

    def _force_disable_xformers_runtime(self, model=None) -> None:
        if self._xformers_forced_off:
            return

        torch = self.torch
        target_model = model if model is not None else getattr(self, "model", None)
        if target_model is None:
            return
        for module_name, module in list(sys.modules.items()):
            if module is None:
                continue
            file_path = str(getattr(module, "__file__", "") or "").replace("\\", "/")
            is_dino_attention = module_name.endswith("dinov2.layers.attention") or file_path.endswith("/dinov2/layers/attention.py")
            is_dino_block = module_name.endswith("dinov2.layers.block") or file_path.endswith("/dinov2/layers/block.py")
            is_dino_swiglu = module_name.endswith("dinov2.layers.swiglu_ffn") or file_path.endswith("/dinov2/layers/swiglu_ffn.py")
            if not (is_dino_attention or is_dino_block or is_dino_swiglu):
                continue
            if hasattr(module, "XFORMERS_AVAILABLE"):
                setattr(module, "XFORMERS_AVAILABLE", False)
            if is_dino_attention and hasattr(module, "memory_efficient_attention"):
                setattr(module, "memory_efficient_attention", None)

        def _fallback_mem_eff_attention(this, x, attn_bias=None):
            def _apply_dropout(value, dropout_obj):
                if callable(dropout_obj):
                    return dropout_obj(value)
                drop_prob = float(dropout_obj) if dropout_obj is not None else 0.0
                if drop_prob <= 0.0:
                    return value
                return torch.nn.functional.dropout(value, p=drop_prob, training=bool(this.training))

            batch, num_tokens, dim = x.shape
            qkv = this.qkv(x).reshape(batch, num_tokens, 3, this.num_heads, dim // this.num_heads).permute(2, 0, 3, 1, 4)
            q = qkv[0] * this.scale
            k = qkv[1]
            v = qkv[2]
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = _apply_dropout(attn, getattr(this, "attn_drop", 0.0))
            x = (attn @ v).transpose(1, 2).reshape(batch, num_tokens, dim)
            x = this.proj(x)
            x = _apply_dropout(x, getattr(this, "proj_drop", 0.0))
            return x

        def _fallback_swiglu(this, x):
            x12 = this.w12(x)
            x1, x2 = x12.chunk(2, dim=-1)
            x = torch.nn.functional.silu(x1) * x2
            return this.w3(x)

        for module in target_model.modules():
            class_name = module.__class__.__name__
            if class_name == "MemEffAttention" and not getattr(module, "_codex_xformers_patched", False):
                module.forward = MethodType(_fallback_mem_eff_attention, module)
                module._codex_xformers_patched = True
            elif (
                "SwiGLU" in class_name
                and hasattr(module, "w12")
                and hasattr(module, "w3")
                and not getattr(module, "_codex_xformers_patched", False)
            ):
                module.forward = MethodType(_fallback_swiglu, module)
                module._codex_xformers_patched = True

        self._xformers_forced_off = True

    def extract(self, rgb: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        if rgb is None:
            raise RuntimeError("RGB image is required for DINO feature extraction.")
        try:
            return self._extract_impl_batch([rgb], [target_shape])[0]
        except Exception as exc:
            if not _is_xformers_runtime_error(exc):
                raise
            print("Warn: incompatible xFormers runtime detected; retrying with PyTorch attention fallback.")
            self._force_disable_xformers_runtime()
            return self._extract_impl_batch([rgb], [target_shape])[0]

    def extract_batch(self, rgbs: Sequence[np.ndarray], target_shapes: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
        if len(rgbs) != len(target_shapes):
            raise ValueError("rgbs and target_shapes must have the same length.")
        if not rgbs:
            return []
        try:
            return self._extract_impl_batch(rgbs, target_shapes)
        except Exception as exc:
            if not _is_xformers_runtime_error(exc):
                raise
            print("Warn: incompatible xFormers runtime detected during batched extraction; retrying with PyTorch attention fallback.")
            self._force_disable_xformers_runtime()
            return self._extract_impl_batch(rgbs, target_shapes)

    def _extract_impl_batch(self, rgbs: Sequence[np.ndarray], target_shapes: Sequence[Tuple[int, int]]) -> List[np.ndarray]:
        prepared_tensors: List[object] = []
        scaled_shape: Optional[Tuple[int, int]] = None
        for rgb in rgbs:
            x, current_scaled_shape = self._prepare_sample_tensor(rgb)
            if scaled_shape is None:
                scaled_shape = current_scaled_shape
            elif current_scaled_shape != scaled_shape:
                raise ValueError("All images in one batch must share the same prepared shape.")
            prepared_tensors.append(x)
        if scaled_shape is None:
            return []
        x = self.torch.stack(prepared_tensors, dim=0)
        scaled_h, scaled_w = scaled_shape
        grid_h = max(1, scaled_h // self.patch_size)
        grid_w = max(1, scaled_w // self.patch_size)
        with self.torch.no_grad():
            tokens = None
            if hasattr(self.model, "forward_features"):
                try:
                    outputs = self.model.forward_features(x)
                    tokens = self._extract_tokens(outputs, grid_h=grid_h, grid_w=grid_w)
                except Exception:
                    tokens = None
            if tokens is None and hasattr(self.model, "get_intermediate_layers"):
                outputs = self.model.get_intermediate_layers(x, n=1, reshape=False)
                tokens = self._extract_tokens(outputs, grid_h=grid_h, grid_w=grid_w)
            if tokens is None:
                outputs = self.model(x)
                tokens = self._extract_tokens(outputs, grid_h=grid_h, grid_w=grid_w)
            if tokens is None:
                raise RuntimeError("Unable to parse patch tokens from the DINO model output.")
            tokens = tokens.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.float32)
        return [_prepare_feature_map(tokens[idx], target_shape=target_shapes[idx]) for idx in range(len(target_shapes))]


def _ensure_odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _filter_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= int(min_area):
            filtered[labels == label_idx] = 1
    return filtered


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: List[Tuple[int, np.ndarray]] = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        component = (labels == label_idx).astype(np.uint8)
        components.append((area, component))
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components]


def _iter_depth_paths(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for current_root, _, file_names in os.walk(root):
        for file_name in sorted(file_names):
            if Path(file_name).suffix.lower() in DEPTH_EXTENSIONS:
                yield Path(current_root) / file_name


def _build_stem_index(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for current_root, _, file_names in os.walk(root):
        for file_name in sorted(file_names):
            if Path(file_name).suffix.lower() not in DEPTH_EXTENSIONS:
                continue
            stem = Path(file_name).stem
            if stem not in index:
                index[stem] = Path(current_root) / file_name
    return index


def _read_rgb_image(path: Path) -> Optional[np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _mask_mean_depth(mask: np.ndarray, depth: np.ndarray) -> float:
    values = np.asarray(depth, dtype=np.float32)[np.asarray(mask) > 0]
    if values.size == 0:
        return 0.0
    return float(values.mean())


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(np.asarray(mask) > 0)
    if len(xs) == 0:
        return np.zeros(2, dtype=np.float32)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def _merge_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    return np.maximum((np.asarray(mask_a) > 0).astype(np.uint8), (np.asarray(mask_b) > 0).astype(np.uint8))


def _select_best_target(fragment_mask: np.ndarray, target_masks: Sequence[np.ndarray], depth_map: np.ndarray) -> int:
    if not target_masks:
        return -1
    fragment_center = _mask_centroid(fragment_mask)
    fragment_depth = _mask_mean_depth(fragment_mask, depth_map)
    diag = float(np.hypot(fragment_mask.shape[0], fragment_mask.shape[1]))
    best_idx = -1
    best_score = None
    for target_idx, target_mask in enumerate(target_masks):
        target_center = _mask_centroid(target_mask)
        target_depth = _mask_mean_depth(target_mask, depth_map)
        spatial_dist = float(np.linalg.norm(fragment_center - target_center)) / max(diag, 1.0)
        depth_gap = abs(fragment_depth - target_depth)
        score = depth_gap + 0.2 * spatial_dist
        if best_score is None or score < best_score:
            best_score = score
            best_idx = target_idx
    return best_idx


def _restore_mask_coverage(reference_mask: np.ndarray, region_masks: List[np.ndarray], depth_map: np.ndarray) -> List[np.ndarray]:
    reference_mask = (np.asarray(reference_mask) > 0).astype(np.uint8)
    if not region_masks:
        return [reference_mask] if int(reference_mask.sum()) > 0 else []

    union = np.zeros_like(reference_mask, dtype=np.uint8)
    valid_regions: List[np.ndarray] = []
    for region in region_masks:
        region = (np.asarray(region) > 0).astype(np.uint8)
        if int(region.sum()) <= 0:
            continue
        valid_regions.append(region)
        union = np.maximum(union, region)
    if not valid_regions:
        return [reference_mask] if int(reference_mask.sum()) > 0 else []

    leftover = ((reference_mask > 0) & (union == 0)).astype(np.uint8)
    if int(leftover.sum()) <= 0:
        return valid_regions

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(leftover, connectivity=8)
    for label_idx in range(1, num_labels):
        fragment = (labels == label_idx).astype(np.uint8)
        target_idx = _select_best_target(fragment, valid_regions, depth_map)
        if target_idx >= 0:
            valid_regions[target_idx] = _merge_masks(valid_regions[target_idx], fragment)
    return valid_regions


def _build_support_mask(depth: np.ndarray, gt_mask: np.ndarray, min_area: int) -> np.ndarray:
    gt_mask = (np.asarray(gt_mask) > 0).astype(np.uint8)
    if int(gt_mask.sum()) <= 0:
        return np.zeros_like(gt_mask, dtype=np.uint8)
    depth_valid = np.isfinite(np.asarray(depth, dtype=np.float32)).astype(np.uint8)
    support = ((gt_mask > 0) & (depth_valid > 0)).astype(np.uint8)
    return _filter_components(support, min_area=min_area)


def _preprocess_depth(
    depth: np.ndarray,
    median_ksize: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
) -> np.ndarray:
    depth = _normalize_gray(depth)
    median_ksize = _ensure_odd(median_ksize)
    if median_ksize > 1:
        depth = cv2.medianBlur(depth.astype(np.float32), median_ksize)
    if int(bilateral_d) > 1:
        depth = cv2.bilateralFilter(
            depth.astype(np.float32),
            d=int(bilateral_d),
            sigmaColor=float(bilateral_sigma_color),
            sigmaSpace=float(bilateral_sigma_space),
        )
    return _normalize_gray(depth)


def _compute_depth_discontinuity(depth: np.ndarray) -> np.ndarray:
    depth = _normalize_gray(depth)
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return _normalize_gray(cv2.magnitude(grad_x, grad_y))


def _labels_to_rgb(label_map: np.ndarray) -> np.ndarray:
    label_map = np.asarray(label_map, dtype=np.int32)
    label_rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label_idx in sorted(int(v) for v in np.unique(label_map) if int(v) > 0):
        label_rgb[label_map == label_idx] = np.array(PALETTE[(label_idx - 1) % len(PALETTE)], dtype=np.uint8)
    return label_rgb


def _draw_boundaries(base_gray: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(_to_uint8(base_gray), cv2.COLOR_GRAY2BGR)
    for label_idx in sorted(int(v) for v in np.unique(label_map) if int(v) > 0):
        mask = (label_map == label_idx).astype(np.uint8)
        color = PALETTE[(label_idx - 1) % len(PALETTE)]
        eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        canvas[(mask > 0) & (eroded == 0)] = np.array(color, dtype=np.uint8)
    return canvas


def _resize_rgb_if_needed(rgb: Optional[np.ndarray], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if rgb is None:
        return None
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[:2] == tuple(target_shape):
        return rgb
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _build_full_image_superpixels(
    depth: np.ndarray,
    rgb: Optional[np.ndarray],
    superpixel_count: int,
    slic_compactness: float,
    slic_sigma: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    n_segments = max(1, int(superpixel_count))
    debug_info: Dict[str, object] = {
        "requested_segments": int(n_segments),
        "slic_input_mode": "rgb",
    }
    if n_segments <= 1:
        return np.ones_like(depth, dtype=np.int32), debug_info

    image_rgb = _resize_rgb_if_needed(rgb, depth.shape)
    if image_rgb is None:
        depth_rgb = np.clip(_normalize_gray(depth) * 255.0, 0.0, 255.0).astype(np.uint8)
        image_rgb = np.repeat(depth_rgb[..., None], 3, axis=2)
        debug_info["slic_input_mode"] = "depth_fallback_rgb"
    try:
        labels = slic(
            image_rgb,
            n_segments=int(n_segments),
            compactness=float(slic_compactness),
            sigma=float(slic_sigma),
            start_label=0,
            channel_axis=-1,
        ).astype(np.int32)
    except Exception:
        labels = np.ones_like(depth, dtype=np.int32)
        debug_info["slic_error"] = True
    return labels, debug_info


def _clip_full_image_superpixels_to_component(
    component_crop: np.ndarray,
    slic_labels_crop: np.ndarray,
    depth_crop: np.ndarray,
    requested_segments: int,
    min_superpixel_area: int,
    slic_input_mode: str = "rgbd",
) -> Tuple[np.ndarray, Dict[str, object]]:
    component_crop = (np.asarray(component_crop) > 0).astype(np.uint8)
    label_crop = np.zeros_like(component_crop, dtype=np.int32)
    debug_info: Dict[str, object] = {
        "requested_segments": int(max(1, requested_segments)),
        "raw_superpixels": 0,
        "kept_superpixels": 0,
        "slic_input_mode": str(slic_input_mode).strip().lower() or "rgbd",
    }
    if int(component_crop.sum()) <= 0:
        return label_crop, debug_info

    slic_labels_crop = np.asarray(slic_labels_crop, dtype=np.int32)
    unique_labels = np.unique(slic_labels_crop)
    if unique_labels.size <= 1:
        label_crop[component_crop > 0] = 1
        debug_info["raw_superpixels"] = 1
        debug_info["kept_superpixels"] = 1
        return label_crop, debug_info

    raw_masks: List[np.ndarray] = []
    kept_masks: List[np.ndarray] = []
    for local_label in sorted(int(value) for value in unique_labels):
        masked_region = ((slic_labels_crop == local_label) & (component_crop > 0)).astype(np.uint8)
        for region_mask in _connected_components(masked_region):
            if int(region_mask.sum()) <= 0:
                continue
            raw_masks.append(region_mask)
            if int(region_mask.sum()) >= int(min_superpixel_area):
                kept_masks.append(region_mask)
    debug_info["raw_superpixels"] = int(len(raw_masks)) if raw_masks else 1
    if not kept_masks:
        kept_masks = raw_masks if raw_masks else [component_crop.copy()]

    kept_masks = _restore_mask_coverage(component_crop, kept_masks, depth_crop)
    for label_idx, region_mask in enumerate(kept_masks, start=1):
        label_crop[np.asarray(region_mask) > 0] = int(label_idx)
    debug_info["kept_superpixels"] = int(len(kept_masks))
    return label_crop, debug_info


def _pca_reduce_features(features: np.ndarray, output_dim: int) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] <= 1 or output_dim <= 0 or features.shape[1] <= output_dim:
        return features
    centered = features - features.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return features
    basis = vt[: int(output_dim)]
    reduced = centered @ basis.T
    return reduced.astype(np.float32)


def _normalize_feature_rows(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.size <= 0:
        return features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-6)
    return (features / norms).astype(np.float32)


def _extract_superpixel_features(
    superpixel_labels: np.ndarray,
    depth_crop: np.ndarray,
    appearance_crop: Optional[np.ndarray],
    dino_pca_dim: int,
) -> Dict[str, np.ndarray]:
    superpixel_labels = np.asarray(superpixel_labels, dtype=np.int32)
    depth_crop = np.asarray(depth_crop, dtype=np.float32)
    node_count = int(superpixel_labels.max())
    dino_dim = 0
    if appearance_crop is not None and np.asarray(appearance_crop).ndim == 3:
        dino_dim = int(np.asarray(appearance_crop).shape[2])

    areas = np.zeros(node_count, dtype=np.float32)
    centroids = np.zeros((node_count, 2), dtype=np.float32)
    depth_mean = np.zeros(node_count, dtype=np.float32)
    dino_mean = np.zeros((node_count, dino_dim), dtype=np.float32)

    for node_idx in range(1, node_count + 1):
        mask = superpixel_labels == node_idx
        if not np.any(mask):
            continue
        ys, xs = np.where(mask)
        areas[node_idx - 1] = float(len(xs))
        centroids[node_idx - 1] = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        depth_values = depth_crop[mask]
        depth_mean[node_idx - 1] = float(depth_values.mean()) if depth_values.size > 0 else 0.0
        if dino_dim > 0 and appearance_crop is not None:
            dino_values = np.asarray(appearance_crop, dtype=np.float32)[mask]
            if dino_values.size > 0:
                dino_mean[node_idx - 1] = dino_values.mean(axis=0).astype(np.float32)

    if dino_dim > 0:
        dino_mean = _pca_reduce_features(dino_mean, output_dim=int(dino_pca_dim))
        dino_mean = _normalize_feature_rows(dino_mean)

    return {
        "areas": areas,
        "centroids": centroids,
        "depth_mean": depth_mean,
        "dino_mean": dino_mean,
    }


def _line_coordinates(x0: int, y0: int, x1: int, y1: int) -> Tuple[np.ndarray, np.ndarray]:
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    step_x = 1 if x0 < x1 else -1
    step_y = 1 if y0 < y1 else -1
    err = dx + dy
    xs: List[int] = []
    ys: List[int] = []
    while True:
        xs.append(int(x0))
        ys.append(int(y0))
        if x0 == x1 and y0 == y1:
            break
        err2 = 2 * err
        if err2 >= dy:
            err += dy
            x0 += step_x
        if err2 <= dx:
            err += dx
            y0 += step_y
    return np.asarray(ys, dtype=np.int32), np.asarray(xs, dtype=np.int32)


def _max_line_edge_response(
    edge_map: np.ndarray,
    support_mask: np.ndarray,
    point_a: Tuple[float, float],
    point_b: Tuple[float, float],
) -> float:
    edge_map = np.asarray(edge_map, dtype=np.float32)
    support_mask = (np.asarray(support_mask) > 0)
    y_coords, x_coords = _line_coordinates(
        x0=int(round(point_a[0])),
        y0=int(round(point_a[1])),
        x1=int(round(point_b[0])),
        y1=int(round(point_b[1])),
    )
    y_coords = np.clip(y_coords, 0, edge_map.shape[0] - 1)
    x_coords = np.clip(x_coords, 0, edge_map.shape[1] - 1)
    values = edge_map[y_coords, x_coords]
    valid = support_mask[y_coords, x_coords]
    if not np.any(valid):
        return 1.0
    max_edge = float(values[valid].max()) if np.any(valid) else 1.0
    outside_ratio = 1.0 - float(valid.mean())
    return max(max_edge, outside_ratio)


def _adjacent_superpixel_pairs(superpixel_labels: np.ndarray) -> List[Tuple[int, int]]:
    superpixel_labels = np.asarray(superpixel_labels, dtype=np.int32)
    if superpixel_labels.ndim != 2 or superpixel_labels.size <= 0:
        return []

    height, width = superpixel_labels.shape
    pair_set = set()
    offsets = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1),
    ]
    for dy, dx in offsets:
        src_y0 = max(0, -dy)
        src_y1 = height - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = width - max(0, dx)
        dst_y0 = max(0, dy)
        dst_y1 = height - max(0, -dy)
        dst_x0 = max(0, dx)
        dst_x1 = width - max(0, -dx)
        src = superpixel_labels[src_y0:src_y1, src_x0:src_x1]
        dst = superpixel_labels[dst_y0:dst_y1, dst_x0:dst_x1]
        valid = (src > 0) & (dst > 0) & (src != dst)
        if not np.any(valid):
            continue
        pairs = np.stack([src[valid], dst[valid]], axis=1).astype(np.int32)
        pairs.sort(axis=1)
        for left_label, right_label in np.unique(pairs, axis=0):
            pair_set.add((int(left_label) - 1, int(right_label) - 1))
    return sorted(pair_set)


def _local_knn_superpixel_pairs(
    superpixel_features: Dict[str, np.ndarray],
    base_pairs: Sequence[Tuple[int, int]],
    extra_knn_neighbors: int,
) -> List[Tuple[int, int]]:
    extra_knn_neighbors = int(max(0, extra_knn_neighbors))
    if extra_knn_neighbors <= 0:
        return []

    centroids = np.asarray(superpixel_features["centroids"], dtype=np.float32)
    areas = np.asarray(superpixel_features["areas"], dtype=np.float32)
    node_count = int(len(areas))
    if node_count <= 1:
        return []

    base_pair_set = {tuple(sorted((int(left_idx), int(right_idx)))) for left_idx, right_idx in base_pairs}
    adjacency: List[set] = [set() for _ in range(node_count)]
    for left_idx, right_idx in base_pair_set:
        adjacency[left_idx].add(right_idx)
        adjacency[right_idx].add(left_idx)

    extra_pair_set = set()
    for node_idx in range(node_count):
        if float(areas[node_idx]) <= 0.0:
            continue

        offsets = centroids - centroids[node_idx]
        distances = np.sum(offsets * offsets, axis=1).astype(np.float32)
        invalid = np.zeros(node_count, dtype=bool)
        invalid[node_idx] = True
        if adjacency[node_idx]:
            invalid[np.asarray(sorted(adjacency[node_idx]), dtype=np.int32)] = True
        invalid[areas <= 0.0] = True
        distances[invalid] = np.inf

        candidate_count = int(np.count_nonzero(np.isfinite(distances)))
        if candidate_count <= 0:
            continue
        local_k = min(extra_knn_neighbors, candidate_count)
        neighbor_indices = np.argpartition(distances, kth=local_k - 1)[:local_k]
        neighbor_indices = neighbor_indices[np.argsort(distances[neighbor_indices])]
        for neighbor_idx in neighbor_indices:
            if not np.isfinite(float(distances[neighbor_idx])):
                continue
            left_idx, right_idx = sorted((int(node_idx), int(neighbor_idx)))
            pair = (left_idx, right_idx)
            if pair in base_pair_set:
                continue
            extra_pair_set.add(pair)

    return sorted(extra_pair_set)


def _build_affinity_matrix(
    superpixel_labels: np.ndarray,
    superpixel_features: Dict[str, np.ndarray],
    edge_map: np.ndarray,
    support_mask: np.ndarray,
    spatial_diag: float,
    sigma_sem: float,
    sigma_dep: float,
    sigma_spatial: float,
    sigma_edge: float,
    min_affinity: float,
    extra_knn_neighbors: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    areas = np.asarray(superpixel_features["areas"], dtype=np.float32)
    centroids = np.asarray(superpixel_features["centroids"], dtype=np.float32)
    depth_mean = np.asarray(superpixel_features["depth_mean"], dtype=np.float32)
    dino_mean = np.asarray(superpixel_features["dino_mean"], dtype=np.float32)
    node_count = int(len(areas))
    affinity = np.zeros((node_count, node_count), dtype=np.float32)
    pair_count = 0
    affinity_sum = 0.0

    adjacent_pairs = _adjacent_superpixel_pairs(superpixel_labels)
    knn_pairs = _local_knn_superpixel_pairs(
        superpixel_features=superpixel_features,
        base_pairs=adjacent_pairs,
        extra_knn_neighbors=extra_knn_neighbors,
    )
    graph_pairs = sorted(set(adjacent_pairs) | set(knn_pairs))
    for left_idx, right_idx in graph_pairs:
        sem_affinity = 1.0
        if dino_mean.ndim == 2 and dino_mean.shape[1] > 0:
            cosine = float(np.clip(np.dot(dino_mean[left_idx], dino_mean[right_idx]), -1.0, 1.0))
            sem_affinity = float(np.exp(-(1.0 - cosine) / max(float(sigma_sem), 1e-6)))
        depth_gap = float(depth_mean[left_idx] - depth_mean[right_idx])
        depth_affinity = float(np.exp(-(depth_gap * depth_gap) / max(float(sigma_dep), 1e-6)))
        spatial_gap = float(np.linalg.norm(centroids[left_idx] - centroids[right_idx])) / max(float(spatial_diag), 1e-6)
        spatial_affinity = float(np.exp(-(spatial_gap * spatial_gap) / max(float(sigma_spatial), 1e-6)))
        edge_value = _max_line_edge_response(
            edge_map=edge_map,
            support_mask=support_mask,
            point_a=(float(centroids[left_idx, 0]), float(centroids[left_idx, 1])),
            point_b=(float(centroids[right_idx, 0]), float(centroids[right_idx, 1])),
        )
        edge_penalty = float(np.exp(-edge_value / max(float(sigma_edge), 1e-6)))
        weight = float(sem_affinity * depth_affinity * spatial_affinity * edge_penalty)
        if weight < float(min_affinity):
            weight = 0.0
        affinity[left_idx, right_idx] = weight
        affinity[right_idx, left_idx] = weight
        pair_count += 1
        affinity_sum += weight

    debug_info = {
        "node_count": int(node_count),
        "adjacent_pair_count": int(len(adjacent_pairs)),
        "knn_pair_count": int(len(knn_pairs)),
        "pair_count": int(pair_count),
        "mean_affinity": float(affinity_sum / max(pair_count, 1)),
        "max_affinity": float(affinity.max()) if affinity.size > 0 else 0.0,
    }
    return affinity, debug_info


def _candidate_split_thresholds(values: np.ndarray) -> List[float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size <= 1:
        return []
    unique_values = np.unique(np.round(values, 6))
    if unique_values.size <= 1:
        return []
    if unique_values.size <= 16:
        return [float(0.5 * (unique_values[idx] + unique_values[idx + 1])) for idx in range(unique_values.size - 1)]
    percentiles = [15, 25, 35, 45, 50, 55, 65, 75, 85]
    return sorted(set(float(np.percentile(values, percentile)) for percentile in percentiles))


def _normalized_cut_score(
    affinity: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> float:
    cut_value = float(affinity[np.ix_(left_indices, right_indices)].sum())
    assoc_left = float(affinity[left_indices, :].sum())
    assoc_right = float(affinity[right_indices, :].sum())
    if assoc_left <= 1e-8 or assoc_right <= 1e-8:
        return float("inf")
    return cut_value / assoc_left + cut_value / assoc_right


def _best_ncut_split(
    affinity: np.ndarray,
    node_areas: np.ndarray,
    min_cluster_regions: int,
    min_instance_area: int,
) -> Optional[Dict[str, object]]:
    affinity = np.asarray(affinity, dtype=np.float64)
    node_areas = np.asarray(node_areas, dtype=np.float32)
    node_count = int(affinity.shape[0])
    if node_count < max(2, int(min_cluster_regions) * 2):
        return None
    degrees = affinity.sum(axis=1)
    if np.count_nonzero(degrees > 1e-8) < 2:
        return None

    inv_sqrt = 1.0 / np.sqrt(np.maximum(degrees, 1e-8))
    laplacian = np.eye(node_count, dtype=np.float64) - inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
    laplacian = 0.5 * (laplacian + laplacian.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:
        return None
    if eigenvectors.shape[1] < 2:
        return None

    fiedler_vector = eigenvectors[:, 1].astype(np.float32)
    best_split = None
    for threshold in _candidate_split_thresholds(fiedler_vector):
        left_mask = fiedler_vector <= float(threshold)
        right_mask = ~left_mask
        if int(left_mask.sum()) < int(min_cluster_regions) or int(right_mask.sum()) < int(min_cluster_regions):
            continue
        left_indices = np.where(left_mask)[0].astype(np.int32)
        right_indices = np.where(right_mask)[0].astype(np.int32)
        if float(node_areas[left_indices].sum()) < float(min_instance_area):
            continue
        if float(node_areas[right_indices].sum()) < float(min_instance_area):
            continue
        ncut_value = _normalized_cut_score(affinity, left_indices, right_indices)
        if not np.isfinite(ncut_value):
            continue
        if best_split is None or float(ncut_value) < float(best_split["ncut"]):
            best_split = {
                "left_indices": left_indices,
                "right_indices": right_indices,
                "ncut": float(ncut_value),
                "fiedler_range": float(fiedler_vector.max() - fiedler_vector.min()),
                "eigengap": float(eigenvalues[2] - eigenvalues[1]) if len(eigenvalues) > 2 else 0.0,
            }
    return best_split


def _recursive_ncut(
    affinity: np.ndarray,
    node_areas: np.ndarray,
    min_cluster_regions: int,
    min_instance_area: int,
    ncut_threshold: float,
    max_recursion_depth: int,
) -> Tuple[List[np.ndarray], List[float]]:
    affinity = np.asarray(affinity, dtype=np.float32)
    node_areas = np.asarray(node_areas, dtype=np.float32)
    clusters: List[np.ndarray] = []
    accepted_scores: List[float] = []

    def _recurse(node_indices: np.ndarray, depth: int) -> None:
        node_indices = np.asarray(node_indices, dtype=np.int32)
        if node_indices.size <= 0:
            return
        if depth >= int(max_recursion_depth):
            clusters.append(node_indices)
            return
        if int(node_indices.size) < int(min_cluster_regions) * 2:
            clusters.append(node_indices)
            return
        if float(node_areas[node_indices].sum()) < float(min_instance_area) * 2.0:
            clusters.append(node_indices)
            return

        sub_affinity = affinity[np.ix_(node_indices, node_indices)]
        best_split = _best_ncut_split(
            affinity=sub_affinity,
            node_areas=node_areas[node_indices],
            min_cluster_regions=min_cluster_regions,
            min_instance_area=min_instance_area,
        )
        if best_split is None or float(best_split["ncut"]) > float(ncut_threshold):
            clusters.append(node_indices)
            return

        accepted_scores.append(float(best_split["ncut"]))
        _recurse(node_indices[np.asarray(best_split["left_indices"], dtype=np.int32)], depth=depth + 1)
        _recurse(node_indices[np.asarray(best_split["right_indices"], dtype=np.int32)], depth=depth + 1)

    _recurse(np.arange(len(node_areas), dtype=np.int32), depth=0)
    return clusters, accepted_scores


def _cluster_superpixels(
    component_crop: np.ndarray,
    superpixel_labels: np.ndarray,
    affinity: np.ndarray,
    superpixel_features: Dict[str, np.ndarray],
    min_cluster_regions: int,
    min_instance_area: int,
    ncut_threshold: float,
    max_recursion_depth: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    component_crop = (np.asarray(component_crop) > 0).astype(np.uint8)
    superpixel_labels = np.asarray(superpixel_labels, dtype=np.int32)
    node_count = int(superpixel_labels.max())
    cluster_labels = np.zeros_like(superpixel_labels, dtype=np.int32)
    if node_count <= 1:
        cluster_labels[component_crop > 0] = 1
        return cluster_labels, {"cluster_count": 1, "accepted_splits": 0, "best_ncut": None}

    clusters, accepted_scores = _recursive_ncut(
        affinity=affinity,
        node_areas=np.asarray(superpixel_features["areas"], dtype=np.float32),
        min_cluster_regions=min_cluster_regions,
        min_instance_area=min_instance_area,
        ncut_threshold=ncut_threshold,
        max_recursion_depth=max_recursion_depth,
    )

    if not clusters:
        clusters = [np.arange(node_count, dtype=np.int32)]
    for cluster_idx, cluster_nodes in enumerate(clusters, start=1):
        node_labels = cluster_nodes.astype(np.int32) + 1
        cluster_labels[np.isin(superpixel_labels, node_labels)] = int(cluster_idx)

    return cluster_labels, {
        "cluster_count": int(len(clusters)),
        "accepted_splits": int(len(accepted_scores)),
        "best_ncut": float(min(accepted_scores)) if accepted_scores else None,
    }


def split_depth_instances_affinity_spectral(
    depth: np.ndarray,
    gt_mask: np.ndarray,
    rgb: Optional[np.ndarray],
    appearance_map: Optional[np.ndarray],
    min_component_area: int,
    min_instance_area: int,
    median_ksize: int,
    bilateral_d: int,
    bilateral_sigma_color: float,
    bilateral_sigma_space: float,
    superpixel_count: int,
    min_superpixel_area: int,
    slic_compactness: float,
    slic_sigma: float,
    slic_depth_scale: float,
    dino_pca_dim: int,
    sigma_sem: float,
    sigma_dep: float,
    sigma_spatial: float,
    sigma_edge: float,
    min_affinity: float,
    min_cluster_regions: int,
    ncut_threshold: float,
    max_recursion_depth: int,
    extra_knn_neighbors: int = 8,
    slic_input_mode: str = "rgbd",
) -> Dict[str, object]:
    depth = _normalize_gray(depth)
    support = _build_support_mask(depth=depth, gt_mask=gt_mask, min_area=min_component_area)
    zeros_f32 = np.zeros_like(depth, dtype=np.float32)
    zeros_i32 = np.zeros_like(support, dtype=np.int32)
    zeros_u8 = np.zeros_like(support, dtype=np.uint8)
    if int(support.sum()) <= 0:
        return {
            "label_map": zeros_i32.copy(),
            "superpixel_label_map": zeros_i32.copy(),
            "depth_smooth": zeros_f32.copy(),
            "depth_edge": zeros_f32.copy(),
            "support_mask": zeros_u8.copy(),
            "debug_lines": [],
        }

    depth_smooth = _preprocess_depth(
        depth=depth,
        median_ksize=median_ksize,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )
    depth_edge = _compute_depth_discontinuity(depth_smooth)
    full_image_superpixels, full_slic_debug = _build_full_image_superpixels(
        depth=depth_smooth,
        rgb=rgb,
        superpixel_count=superpixel_count,
        slic_compactness=slic_compactness,
        slic_sigma=slic_sigma,
    )
    label_map = np.zeros_like(support, dtype=np.int32)
    superpixel_label_map = np.zeros_like(support, dtype=np.int32)
    debug_lines: List[str] = []
    next_label = 1
    next_superpixel_label = 1

    component_idx = 0
    for component in _connected_components(support):
        component_idx += 1
        component = (np.asarray(component) > 0).astype(np.uint8)
        ys, xs = np.where(component > 0)
        if len(xs) <= 0:
            continue
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        crop_slice = np.s_[y0:y1 + 1, x0:x1 + 1]
        component_crop = component[crop_slice]
        depth_crop = np.asarray(depth_smooth, dtype=np.float32)[crop_slice]
        edge_crop = np.asarray(depth_edge, dtype=np.float32)[crop_slice]
        appearance_crop = None
        if appearance_map is not None and np.asarray(appearance_map).ndim == 3 and np.asarray(appearance_map).shape[:2] == support.shape:
            appearance_crop = np.asarray(appearance_map, dtype=np.float32)[crop_slice]

        superpixel_crop, superpixel_debug = _clip_full_image_superpixels_to_component(
            component_crop=component_crop,
            slic_labels_crop=full_image_superpixels[crop_slice],
            depth_crop=depth_crop,
            requested_segments=int(full_slic_debug.get("requested_segments", superpixel_count)),
            min_superpixel_area=min_superpixel_area,
            slic_input_mode=str(full_slic_debug.get("slic_input_mode", "rgb")),
        )
        superpixel_features = _extract_superpixel_features(
            superpixel_labels=superpixel_crop,
            depth_crop=depth_crop,
            appearance_crop=appearance_crop,
            dino_pca_dim=dino_pca_dim,
        )
        spatial_diag = float(np.hypot(component_crop.shape[0], component_crop.shape[1]))
        affinity, affinity_debug = _build_affinity_matrix(
            superpixel_labels=superpixel_crop,
            superpixel_features=superpixel_features,
            edge_map=edge_crop,
            support_mask=component_crop,
            spatial_diag=spatial_diag,
            sigma_sem=sigma_sem,
            sigma_dep=sigma_dep,
            sigma_spatial=sigma_spatial,
            sigma_edge=sigma_edge,
            min_affinity=min_affinity,
            extra_knn_neighbors=extra_knn_neighbors,
        )
        cluster_crop, cluster_debug = _cluster_superpixels(
            component_crop=component_crop,
            superpixel_labels=superpixel_crop,
            affinity=affinity,
            superpixel_features=superpixel_features,
            min_cluster_regions=min_cluster_regions,
            min_instance_area=min_instance_area,
            ncut_threshold=ncut_threshold,
            max_recursion_depth=max_recursion_depth,
        )

        superpixel_view = superpixel_label_map[crop_slice]
        for local_label in sorted(int(value) for value in np.unique(superpixel_crop) if int(value) > 0):
            superpixel_view[superpixel_crop == local_label] = int(next_superpixel_label)
            next_superpixel_label += 1

        label_view = label_map[crop_slice]
        for local_label in sorted(int(value) for value in np.unique(cluster_crop) if int(value) > 0):
            label_view[cluster_crop == local_label] = int(next_label)
            next_label += 1

        debug_lines.append(
            "component={} area={} slic_input_mode={} requested_segments={} raw_superpixels={} kept_superpixels={} adj_pairs={} knn_pairs={} graph_pairs={} mean_affinity={:.6f} clusters={} accepted_splits={} best_ncut={}".format(
                int(component_idx),
                int(component_crop.sum()),
                str(superpixel_debug.get("slic_input_mode", "rgbd")),
                int(superpixel_debug.get("requested_segments", 0)),
                int(superpixel_debug.get("raw_superpixels", 0)),
                int(superpixel_debug.get("kept_superpixels", 0)),
                int(affinity_debug.get("adjacent_pair_count", 0)),
                int(affinity_debug.get("knn_pair_count", 0)),
                int(affinity_debug.get("pair_count", 0)),
                float(affinity_debug.get("mean_affinity", 0.0)),
                int(cluster_debug.get("cluster_count", 0)),
                int(cluster_debug.get("accepted_splits", 0)),
                "None" if cluster_debug.get("best_ncut") is None else "{:.4f}".format(float(cluster_debug["best_ncut"])),
            )
        )

    return {
        "label_map": label_map,
        "superpixel_label_map": superpixel_label_map,
        "depth_smooth": depth_smooth,
        "depth_edge": depth_edge,
        "support_mask": support.astype(np.uint8),
        "debug_lines": debug_lines,
    }


def _save_outputs(
    depth_path: Path,
    output_root: Path,
    relative_path: Path,
    rgb: Optional[np.ndarray],
    depth_smooth: np.ndarray,
    depth_edge: np.ndarray,
    support_mask: np.ndarray,
    superpixel_label_map: np.ndarray,
    gt_mask: np.ndarray,
    label_map: np.ndarray,
    debug_lines: Sequence[str],
) -> None:
    stem = relative_path.with_suffix("")
    boundary_path = output_root / "boundary" / f"{stem.as_posix()}.png"
    rgb_boundary_path = output_root / "rgb_boundary" / f"{stem.as_posix()}.png"
    labels_path = output_root / "labels" / f"{stem.as_posix()}.png"
    union_path = output_root / "union" / f"{stem.as_posix()}.png"
    gt_path = output_root / "gt" / f"{stem.as_posix()}.png"
    depth_smooth_path = output_root / "depth_smooth" / f"{stem.as_posix()}.png"
    depth_edge_path = output_root / "depth_edge" / f"{stem.as_posix()}.png"
    support_mask_path = output_root / "support_mask" / f"{stem.as_posix()}.png"
    superpixel_boundary_path = output_root / "superpixel_boundary" / f"{stem.as_posix()}.png"
    superpixel_labels_path = output_root / "superpixel_labels" / f"{stem.as_posix()}.png"
    debug_stats_path = output_root / "debug_stats" / f"{stem.as_posix()}.txt"

    for path in [
        boundary_path,
        rgb_boundary_path,
        labels_path,
        union_path,
        gt_path,
        depth_smooth_path,
        depth_edge_path,
        support_mask_path,
        superpixel_boundary_path,
        superpixel_labels_path,
        debug_stats_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    boundary_rgb = _draw_boundaries(depth_smooth, label_map)
    superpixel_boundary_rgb = _draw_boundaries(depth_smooth, superpixel_label_map)
    labels_rgb = _labels_to_rgb(label_map)
    superpixel_labels_rgb = _labels_to_rgb(superpixel_label_map)
    union_u8 = ((label_map > 0).astype(np.uint8) * 255)
    gt_u8 = ((np.asarray(gt_mask) > 0).astype(np.uint8) * 255)
    superpixel_label_overlay_rgb = superpixel_labels_rgb.copy()
    if rgb is not None:
        superpixel_mask = np.asarray(superpixel_label_map) > 0
        superpixel_label_overlay_rgb = np.asarray(rgb, dtype=np.uint8).copy()
        if np.any(superpixel_mask):
            overlay = superpixel_label_overlay_rgb.astype(np.float32)
            overlay[superpixel_mask] = (
                0.55 * overlay[superpixel_mask]
                + 0.45 * superpixel_labels_rgb[superpixel_mask].astype(np.float32)
            )
            superpixel_label_overlay_rgb = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    cv2.imwrite(str(boundary_path), cv2.cvtColor(boundary_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(superpixel_boundary_path), cv2.cvtColor(superpixel_boundary_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(labels_path), cv2.cvtColor(labels_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(superpixel_labels_path), cv2.cvtColor(superpixel_label_overlay_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(union_path), union_u8)
    cv2.imwrite(str(gt_path), gt_u8)
    cv2.imwrite(str(depth_smooth_path), _to_uint8(depth_smooth))
    cv2.imwrite(str(depth_edge_path), _to_uint8(depth_edge))
    cv2.imwrite(str(support_mask_path), (np.asarray(support_mask) > 0).astype(np.uint8) * 255)
    debug_stats_path.write_text("\n".join(debug_lines) + ("\n" if debug_lines else ""), encoding="utf-8")

    if rgb is not None:
        rgb_boundary = np.asarray(rgb, dtype=np.uint8).copy()
        for label_idx in sorted(int(value) for value in np.unique(label_map) if int(value) > 0):
            mask = (label_map == label_idx).astype(np.uint8)
            eroded = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
            inner_boundary = (mask > 0) & (eroded == 0)
            rgb_boundary[inner_boundary] = np.array(PALETTE[(label_idx - 1) % len(PALETTE)], dtype=np.uint8)
        cv2.imwrite(str(rgb_boundary_path), cv2.cvtColor(rgb_boundary, cv2.COLOR_RGB2BGR))

    print(
        "Saved {} instances for {} -> {}".format(
            int(len(np.unique(label_map[label_map > 0]))),
            str(depth_path),
            str(boundary_path),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split GT-constrained depth regions with SLIC superpixels and explicit multimodal affinity spectral clustering."
    )
    parser.add_argument("--depth", required=True, help="Depth image file or directory.")
    parser.add_argument("--output", required=True, help="Directory to save visualizations.")
    default_gt_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "dataset",
            "DUTS-TR",
            "segmentations",
        )
    )
    default_rgb_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "dataset",
            "DUTS-TR",
            "images",
        )
    )
    parser.add_argument("--gt-root", default=default_gt_root, help="Directory containing GT masks matched by basename.")
    parser.add_argument("--rgb-root", default=default_rgb_root, help="Optional RGB directory used for SLIC and DINO features.")
    parser.add_argument("--invert", action="store_true", help="Invert depth if near objects are darker instead of brighter.")
    parser.add_argument("--min-component-area", default=128, type=int, help="Minimum GT-support connected-component area.")
    parser.add_argument("--min-instance-area", default=64, type=int, help="Minimum kept instance area during recursive NCut.")
    parser.add_argument("--median-ksize", default=5, type=int, help="Median-blur kernel size for depth preprocessing.")
    parser.add_argument("--bilateral-d", default=7, type=int, help="Bilateral-filter pixel diameter.")
    parser.add_argument("--bilateral-sigma-color", default=25.0, type=float, help="Bilateral sigmaColor.")
    parser.add_argument("--bilateral-sigma-space", default=25.0, type=float, help="Bilateral sigmaSpace.")
    parser.add_argument("--dino-weight", default="", help="Optional local DINOv2 checkpoint used to generate dense semantic features.")
    parser.add_argument("--dino-model", default="dinov2_vitl14", help="Hub model name, e.g. dinov2_vitl14.")
    parser.add_argument("--dino-repo", default=_default_dino_repo(), help="Optional local clone path of facebookresearch/dinov2.")
    parser.add_argument("--dino-device", default="auto", help="Device for DINO inference: auto/cpu/cuda:0 ...")
    parser.add_argument("--dino-max-side", default=700, type=int, help="Maximum long-side resolution fed into the DINO encoder. 0 means full image size.")
    parser.add_argument("--dino-pca-dim", default=64, type=int, help="Optional PCA output dimension applied to superpixel-level DINO descriptors.")
    parser.add_argument("--superpixel-count", default=200, type=int, help="Requested SLIC superpixel count over the full image before support-mask clipping.")
    parser.add_argument("--min-superpixel-area", default=40, type=int, help="Minimum kept superpixel area before coverage restoration.")
    parser.add_argument("--slic-compactness", default=10.0, type=float, help="Compactness used by SLIC.")
    parser.add_argument("--slic-sigma", default=1.0, type=float, help="Gaussian smoothing sigma used by SLIC.")
    parser.add_argument("--slic-depth-scale", default=0.5, type=float, help="Legacy compatibility option; RGB SLIC no longer uses depth channels.")
    parser.add_argument("--slic-input-mode", default="rgb", choices=["rgb", "depth", "rgbd"], help="Legacy compatibility option; RGB SLIC is always used when RGB is available.")
    parser.add_argument("--sigma-sem", default=0.20, type=float, help="Sigma used in semantic affinity exp(-(1-cos)/sigma_sem).")
    parser.add_argument("--sigma-dep", default=0.02, type=float, help="Sigma used in depth affinity exp(-(d_i-d_j)^2/sigma_dep).")
    parser.add_argument("--sigma-spatial", default=0.12, type=float, help="Sigma used in spatial affinity exp(-dist^2/sigma_spatial). Distances are normalized by the component diagonal.")
    parser.add_argument("--sigma-edge", default=0.20, type=float, help="Sigma used in edge penalty exp(-max_edge/sigma_edge).")
    parser.add_argument("--min-affinity", default=1e-6, type=float, help="Clamp affinities below this value to zero.")
    parser.add_argument("--min-cluster-regions", default=2, type=int, help="Minimum number of superpixels kept on each side of a recursive split.")
    parser.add_argument("--ncut-threshold", default=0.10, type=float, help="Maximum normalized-cut score accepted by a recursive split.")
    parser.add_argument("--max-recursion-depth", default=8, type=int, help="Maximum recursive NCut depth inside one GT connected component.")
    parser.add_argument("--extra-knn-neighbors", default=8, type=int, help="Extra non-adjacent spatial-nearest graph edges added per node beyond direct adjacency. Set 4 or 8, 0 disables.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    depth_root = Path(args.depth)
    output_root = Path(args.output)
    gt_root = Path(args.gt_root)
    rgb_root = Path(args.rgb_root)

    if not depth_root.exists():
        raise FileNotFoundError(f"Depth path not found: {depth_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    input_paths = list(_iter_depth_paths(depth_root))
    if not input_paths:
        raise FileNotFoundError(f"No depth images found under {depth_root}")
    gt_index = _build_stem_index(gt_root)
    if not gt_index:
        raise FileNotFoundError(f"No GT masks found under {gt_root}")
    rgb_index = _build_stem_index(rgb_root) if rgb_root.exists() else {}

    dino_extractor = None
    if str(args.dino_weight).strip():
        if str(args.dino_repo).strip():
            print(f"Info: using local DINO repo: {args.dino_repo}")
        dino_extractor = _OnTheFlyDINOExtractor(
            weight_path=Path(args.dino_weight),
            model_name=args.dino_model,
            repo_path=args.dino_repo,
            device=args.dino_device,
            max_side=args.dino_max_side,
        )
    else:
        print("Warn: no --dino-weight provided; semantic affinity will stay neutral and the graph will use depth/spatial/edge terms only.")

    base_root = depth_root if depth_root.is_dir() else depth_root.parent
    for depth_path in input_paths:
        depth_gray = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
        if depth_gray is None:
            print(f"Skip unreadable depth image: {depth_path}")
            continue

        gt_path = gt_index.get(depth_path.stem)
        if gt_path is None:
            print(f"Skip depth image without matching GT mask: {depth_path}")
            continue

        gt_gray = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt_gray is None:
            print(f"Skip unreadable GT mask: {gt_path}")
            continue
        if gt_gray.shape != depth_gray.shape:
            print(f"Skip {depth_path} because GT shape {gt_gray.shape} does not match depth shape {depth_gray.shape}.")
            continue
        gt_mask = (gt_gray > 0).astype(np.uint8)

        rgb = None
        rgb_path = rgb_index.get(depth_path.stem)
        if rgb_path is not None:
            rgb = _read_rgb_image(rgb_path)
            rgb = _resize_rgb_if_needed(rgb, target_shape=depth_gray.shape[:2])

        appearance_map = None
        if dino_extractor is not None:
            if rgb is None:
                print(f"Warn: skip DINO features for {depth_path} because no matching RGB image was found.")
            else:
                try:
                    appearance_map = dino_extractor.extract(rgb, target_shape=rgb.shape[:2])
                except Exception as exc:
                    print(f"Warn: failed to extract DINO features for {depth_path}: {exc}")

        depth = _normalize_gray(depth_gray)
        if args.invert:
            depth = 1.0 - depth

        results = split_depth_instances_affinity_spectral(
            depth=depth,
            gt_mask=gt_mask,
            rgb=rgb,
            appearance_map=appearance_map,
            min_component_area=args.min_component_area,
            min_instance_area=args.min_instance_area,
            median_ksize=args.median_ksize,
            bilateral_d=args.bilateral_d,
            bilateral_sigma_color=args.bilateral_sigma_color,
            bilateral_sigma_space=args.bilateral_sigma_space,
            superpixel_count=args.superpixel_count,
            min_superpixel_area=args.min_superpixel_area,
            slic_compactness=args.slic_compactness,
            slic_sigma=args.slic_sigma,
            slic_depth_scale=args.slic_depth_scale,
            slic_input_mode=args.slic_input_mode,
            dino_pca_dim=args.dino_pca_dim,
            sigma_sem=args.sigma_sem,
            sigma_dep=args.sigma_dep,
            sigma_spatial=args.sigma_spatial,
            sigma_edge=args.sigma_edge,
            min_affinity=args.min_affinity,
            min_cluster_regions=args.min_cluster_regions,
            ncut_threshold=args.ncut_threshold,
            max_recursion_depth=args.max_recursion_depth,
            extra_knn_neighbors=args.extra_knn_neighbors,
        )

        relative_path = depth_path.relative_to(base_root)
        _save_outputs(
            depth_path=depth_path,
            output_root=output_root,
            relative_path=relative_path,
            rgb=rgb,
            depth_smooth=results["depth_smooth"],
            depth_edge=results["depth_edge"],
            support_mask=results["support_mask"],
            superpixel_label_map=results["superpixel_label_map"],
            gt_mask=gt_mask,
            label_map=results["label_map"],
            debug_lines=results["debug_lines"],
        )


if __name__ == "__main__":
    main()
