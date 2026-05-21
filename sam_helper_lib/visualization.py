from typing import List, Tuple

import cv2
import numpy as np

from .image_ops import _ensure_binary_mask, _ensure_uint8_rgb

def _draw_label_grid_overlay(
    image_rgb: np.ndarray,
    label_map: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    image = _ensure_uint8_rgb(image_rgb).copy()
    labels = np.asarray(label_map, dtype=np.int32)
    if labels.shape != image.shape[:2]:
        labels = cv2.resize(labels, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    valid = labels > 0
    boundary = np.zeros(labels.shape, dtype=np.uint8)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & (valid[:, 1:] | valid[:, :-1])
    vertical = (labels[1:, :] != labels[:-1, :]) & (valid[1:, :] | valid[:-1, :])
    boundary[:, 1:][horizontal] = 1
    boundary[:, :-1][horizontal] = 1
    boundary[1:, :][vertical] = 1
    boundary[:-1, :][vertical] = 1
    if int(thickness) > 1:
        kernel = np.ones((int(thickness), int(thickness)), dtype=np.uint8)
        boundary = cv2.dilate(boundary, kernel, iterations=1)
    image[boundary > 0] = np.array(color, dtype=np.uint8)
    return image

def _seed_masks_to_union(seed_masks: List[np.ndarray], out_shape: Tuple[int, int]) -> np.ndarray:
    union_mask = np.zeros(out_shape, dtype=np.uint8)
    for seed_mask in seed_masks:
        union_mask = np.maximum(union_mask, _ensure_binary_mask(seed_mask))
    return union_mask

def _seed_masks_to_label_rgb(seed_masks: List[np.ndarray], out_shape: Tuple[int, int]) -> np.ndarray:
    palette = [
        (255, 99, 71),
        (255, 215, 0),
        (0, 191, 255),
        (50, 205, 50),
        (255, 105, 180),
        (138, 43, 226),
        (255, 140, 0),
        (64, 224, 208),
    ]
    label_rgb = np.zeros((out_shape[0], out_shape[1], 3), dtype=np.uint8)
    for seed_idx, seed_mask in enumerate(seed_masks):
        color = np.array(palette[seed_idx % len(palette)], dtype=np.uint8)
        label_rgb[_ensure_binary_mask(seed_mask) > 0] = color
    return label_rgb

def _seed_masks_overlay(image_rgb: np.ndarray, seed_masks: List[np.ndarray], alpha: float = 0.45) -> np.ndarray:
    image_rgb = _ensure_uint8_rgb(image_rgb)
    label_rgb = _seed_masks_to_label_rgb(seed_masks, image_rgb.shape[:2])
    union_mask = _seed_masks_to_union(seed_masks, image_rgb.shape[:2]).astype(bool)
    overlay = image_rgb.astype(np.float32).copy()
    if np.any(union_mask):
        overlay[union_mask] = (
            (1.0 - alpha) * overlay[union_mask] + alpha * label_rgb[union_mask].astype(np.float32)
        )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)

def _draw_heatmap_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    image_rgb = _ensure_uint8_rgb(image_rgb)
    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.ndim == 3:
        heat = heat.squeeze()
    heat = np.clip(heat, 0.0, 1.0)
    heat_u8 = (heat * 255.0).astype(np.uint8)
    heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * image_rgb.astype(np.float32) + alpha * heat_rgb.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)
