import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


DEPTH_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PALETTE = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 190),
    (0, 128, 128),
    (230, 190, 255),
]


def _normalize_gray(gray: np.ndarray) -> np.ndarray:
    gray = np.asarray(gray, dtype=np.float32)
    if gray.ndim == 3:
        gray = gray[..., 0]
    if gray.size == 0:
        return gray.astype(np.float32)
    min_value = float(np.nanmin(gray))
    max_value = float(np.nanmax(gray))
    if max_value - min_value <= 1e-6:
        return np.zeros_like(gray, dtype=np.float32)
    return ((gray - min_value) / (max_value - min_value)).astype(np.float32)


def _to_uint8(gray: np.ndarray) -> np.ndarray:
    return np.clip(_normalize_gray(gray) * 255.0, 0, 255).astype(np.uint8)


def _read_rgb_image(path: os.PathLike) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Unreadable image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _iter_depth_paths(root: os.PathLike) -> Iterable[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in DEPTH_EXTENSIONS:
        yield root
        return
    for current_root, _, file_names in os.walk(root):
        for file_name in sorted(file_names):
            path = Path(current_root) / file_name
            if path.suffix.lower() in DEPTH_EXTENSIONS:
                yield path


def _build_stem_index(root: os.PathLike) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not root or not Path(root).exists():
        return index
    for path in _iter_depth_paths(root):
        index.setdefault(path.stem, path)
    return index


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    if int(mask.sum()) <= 0:
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        components.append((area, (labels == label_idx).astype(np.uint8)))
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components]


def _build_support_mask(depth: np.ndarray, gt_mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    support = ((np.asarray(gt_mask) > 0) & np.isfinite(np.asarray(depth))).astype(np.uint8)
    if int(support.sum()) <= 0:
        return support
    kept = np.zeros_like(support, dtype=np.uint8)
    for component in _connected_components(support):
        if int(component.sum()) >= int(max(1, min_area)):
            kept[component > 0] = 1
    return kept


def _preprocess_depth(
    depth: np.ndarray,
    median_ksize: int = 5,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 25.0,
    bilateral_sigma_space: float = 25.0,
) -> np.ndarray:
    depth_u8 = _to_uint8(depth)
    median_ksize = int(max(1, median_ksize))
    if median_ksize % 2 == 0:
        median_ksize += 1
    if median_ksize > 1:
        depth_u8 = cv2.medianBlur(depth_u8, median_ksize)
    if int(bilateral_d) > 0:
        depth_u8 = cv2.bilateralFilter(
            depth_u8,
            int(bilateral_d),
            float(bilateral_sigma_color),
            float(bilateral_sigma_space),
        )
    return _normalize_gray(depth_u8)


def _compute_depth_discontinuity(depth: np.ndarray) -> np.ndarray:
    depth = _normalize_gray(depth)
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    return _normalize_gray(np.sqrt(grad_x * grad_x + grad_y * grad_y))


def _restore_mask_coverage(component_mask: np.ndarray, kept_masks: List[np.ndarray], depth_crop: Optional[np.ndarray] = None) -> List[np.ndarray]:
    component = (np.asarray(component_mask) > 0)
    if not kept_masks:
        return [component.astype(np.uint8)]
    restored = [(np.asarray(mask) > 0).astype(np.uint8) for mask in kept_masks]
    union = np.logical_or.reduce([mask.astype(bool) for mask in restored])
    missing = component & ~union
    if not np.any(missing):
        return restored

    centroids = []
    for mask in restored:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            centroids.append(np.array([0.0, 0.0], dtype=np.float32))
        else:
            centroids.append(np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32))
    for missing_component in _connected_components(missing.astype(np.uint8)):
        ys, xs = np.where(missing_component > 0)
        center = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        nearest_idx = int(np.argmin([np.linalg.norm(center - centroid) for centroid in centroids]))
        restored[nearest_idx][missing_component > 0] = 1
    return restored


def _labels_to_rgb(label_map: np.ndarray) -> np.ndarray:
    labels = np.asarray(label_map, dtype=np.int32)
    rgb = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    for label_idx in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        rgb[labels == label_idx] = np.array(PALETTE[(label_idx - 1) % len(PALETTE)], dtype=np.uint8)
    return rgb


def _draw_boundaries(base: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    base_u8 = _to_uint8(base)
    rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)
    labels = np.asarray(label_map, dtype=np.int32)
    for label_idx in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
        mask = (labels == label_idx).astype(np.uint8)
        boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
        rgb[boundary] = np.array(PALETTE[(label_idx - 1) % len(PALETTE)], dtype=np.uint8)
    return rgb
