import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tools.ai.demo_utils import crf_inference_label

def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    return image

def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    mask = mask > 0
    return mask.astype(np.uint8)

def extract_red_mask_from_overlay(
    overlay_rgb: np.ndarray,
    red_min: int = 140,
    green_max: int = 170,
    blue_max: int = 170,
) -> np.ndarray:
    image = _ensure_uint8_rgb(overlay_rgb)
    red = image[..., 0] >= red_min
    green = image[..., 1] <= green_max
    blue = image[..., 2] <= blue_max
    return (red & green & blue).astype(np.uint8)

def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_idx).astype(np.uint8)

def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    mask = _ensure_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return []

    components: List[Tuple[int, np.ndarray]] = []
    for label_idx in range(1, num_labels):
        component = (labels == label_idx).astype(np.uint8)
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        components.append((area, component))

    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components]

def _normalize_gray_map(gray_map: np.ndarray) -> np.ndarray:
    gray_map = np.asarray(gray_map, dtype=np.float32)
    if gray_map.ndim == 3:
        gray_map = gray_map[..., 0]
    min_value = float(gray_map.min()) if gray_map.size > 0 else 0.0
    max_value = float(gray_map.max()) if gray_map.size > 0 else 0.0
    if max_value - min_value <= 1e-6:
        return np.zeros_like(gray_map, dtype=np.float32)
    return ((gray_map - min_value) / (max_value - min_value)).astype(np.float32)

def _build_stem_path_index(root: Optional[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return index
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for current_root, _, file_names in os.walk(root):
        for file_name in file_names:
            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in valid_exts:
                continue
            if stem not in index:
                index[stem] = os.path.join(current_root, file_name)
    return index

def _filter_components(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    if int(mask.sum()) <= 0:
        return mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        if int(stats[label_idx, cv2.CC_STAT_AREA]) >= int(min_area):
            filtered[labels == label_idx] = 1
    return filtered

def _label_map_to_components(label_map: np.ndarray, min_area: int = 16) -> List[np.ndarray]:
    label_map = np.asarray(label_map, dtype=np.int32)
    components: List[Tuple[int, np.ndarray]] = []
    for label_idx in sorted(int(value) for value in np.unique(label_map) if int(value) > 0):
        label_mask = (label_map == label_idx).astype(np.uint8)
        for component in _connected_components(label_mask):
            component = _filter_components(component, min_area=max(1, int(min_area)))
            if int(component.sum()) <= 0:
                continue
            components.append((int(component.sum()), component))
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components]

def _read_rgb_image(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

def _read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    return (mask > 0).astype(np.uint8)

def _write_rgb(path: str, image_rgb: np.ndarray) -> None:
    cv2.imwrite(path, cv2.cvtColor(_ensure_uint8_rgb(image_rgb), cv2.COLOR_RGB2BGR))

def _tensor_to_uint8_rgb(image_3chw: torch.Tensor) -> np.ndarray:
    image = image_3chw.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    return image.permute(1, 2, 0).numpy()

def _resize_to_short_edge(image_rgb: np.ndarray, short_edge: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    image_rgb = _ensure_uint8_rgb(image_rgb)
    if short_edge <= 0:
        return image_rgb, image_rgb.shape[:2]
    h, w = image_rgb.shape[:2]
    min_edge = min(h, w)
    if min_edge <= 0:
        return image_rgb, (h, w)
    scale = float(short_edge) / float(min_edge)
    if abs(scale - 1.0) < 1e-4:
        return image_rgb, (h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, (new_h, new_w)

def _resize_mask(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    mask = _ensure_binary_mask(mask).astype(np.uint8)
    if mask.shape == (target_h, target_w):
        return mask
    resized = cv2.resize(mask.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 0.5).astype(np.uint8)

def _resize_score_map(score_map: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    score_map = np.asarray(score_map, dtype=np.float32)
    if score_map.ndim == 3:
        score_map = score_map.squeeze()
    if score_map.shape == (target_h, target_w):
        return np.clip(score_map, 0.0, 1.0).astype(np.float32)
    resized = cv2.resize(score_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)

def _logits_to_prob_map(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim == 3:
        logits = logits.squeeze()
    logits = np.clip(logits, -32.0, 32.0)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

def _threshold_prob_map(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    prob_map = np.asarray(prob_map, dtype=np.float32)
    if prob_map.ndim == 3:
        prob_map = prob_map.squeeze()
    return (prob_map >= float(threshold)).astype(np.uint8)

def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = _ensure_binary_mask(mask_a).astype(bool)
    mask_b = _ensure_binary_mask(mask_b).astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)

def _refine_mask_with_crf(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image_rgb = _ensure_uint8_rgb(image_rgb)
    mask_u8 = (_ensure_binary_mask(mask) * 255).astype(np.uint8)
    _, mask_u8 = cv2.threshold(mask_u8, 0, 255, cv2.THRESH_OTSU)
    labels = (mask_u8 > 0).astype(np.uint8)

    try:
        refined = crf_inference_label(image_rgb, labels, n_labels=2)
    except ModuleNotFoundError:
        refined = labels

    refined = cv2.medianBlur(refined.astype(np.uint8), 7)
    return _ensure_binary_mask(refined)

def _sample_meta_value(sample_meta: Optional[Dict[str, object]], key: str, idx: int, default: int) -> int:
    if sample_meta is None or key not in sample_meta:
        return int(default)

    value = sample_meta[key]
    if isinstance(value, torch.Tensor):
        return int(value[idx].item())
    if isinstance(value, np.ndarray):
        return int(value[idx])
    if isinstance(value, (list, tuple)):
        return int(value[idx])
    return int(value)
