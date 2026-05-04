import os
import json
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

SEGMENT_ANYTHING_ROOT = os.path.join(os.path.dirname(__file__), "segment-anything-main")
if os.path.isdir(SEGMENT_ANYTHING_ROOT) and SEGMENT_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, SEGMENT_ANYTHING_ROOT)
REFER_ROOT = os.path.join(os.path.dirname(__file__), "A2S-v2-main", "refer")
if os.path.isdir(REFER_ROOT) and REFER_ROOT not in sys.path:
    sys.path.insert(0, REFER_ROOT)

from segment_anything import SamPredictor, sam_model_registry
from tools.ai.demo_utils import crf_inference_label


@dataclass
class SAMCandidate:
    mask: np.ndarray
    score: float
    point_idx: int
    fg_iou: float = 0.0
    heat_iou: float = 0.0
    bg_iou: float = 0.0
    filter_heat_iou: float = 0.0
    filter_bg_iou: float = 0.0
    logits: Optional[np.ndarray] = None
    mask_orig: Optional[np.ndarray] = None
    prob_orig: Optional[np.ndarray] = None


@dataclass
class SAMResult:
    mask: Optional[np.ndarray]
    points_xy: np.ndarray
    candidates: List[SAMCandidate]
    success: bool


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


def _nearest_mask_point(mask: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return float(x), float(y)
    points = np.stack([xs, ys], axis=1).astype(np.float32)
    target = np.array([x, y], dtype=np.float32)
    idx = int(np.argmin(np.sum((points - target) ** 2, axis=1)))
    return float(points[idx, 0]), float(points[idx, 1])


def _extreme_point(mask: np.ndarray, mode: str) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0

    if mode == "left":
        value = xs.min()
        sel = np.where(xs == value)[0]
        y = np.median(ys[sel])
        x = value
    elif mode == "right":
        value = xs.max()
        sel = np.where(xs == value)[0]
        y = np.median(ys[sel])
        x = value
    elif mode == "top":
        value = ys.min()
        sel = np.where(ys == value)[0]
        x = np.median(xs[sel])
        y = value
    elif mode == "bottom":
        value = ys.max()
        sel = np.where(ys == value)[0]
        x = np.median(xs[sel])
        y = value
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return _nearest_mask_point(mask, x, y)


def _center_point(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    return _nearest_mask_point(mask, float(xs.mean()), float(ys.mean()))


def _nearest_distinct_mask_point(
    mask: np.ndarray,
    x: float,
    y: float,
    exclude_points: Optional[List[Tuple[float, float]]] = None,
) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    target = np.array([x, y], dtype=np.float32)
    dists = np.sum((points - target) ** 2, axis=1)
    order = np.argsort(dists)

    excluded = [np.array(point, dtype=np.float32) for point in (exclude_points or [])]
    for idx in order:
        candidate = points[int(idx)]
        if any(np.allclose(candidate, point) for point in excluded):
            continue
        return float(candidate[0]), float(candidate[1])
    return None


def _region_center_point(
    mask: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    region_sel = (xs >= x_min) & (xs < x_max) & (ys >= y_min) & (ys < y_max)
    if not np.any(region_sel):
        return None

    region_xs = xs[region_sel].astype(np.float32)
    region_ys = ys[region_sel].astype(np.float32)
    return _nearest_mask_point(mask, float(region_xs.mean()), float(region_ys.mean()))


def _greedy_fill_points(mask: np.ndarray, points: List[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return points

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    selected = [np.array(p, dtype=np.float32) for p in points]

    while len(selected) < max_points:
        if not selected:
            idx = len(coords) // 2
            selected.append(coords[idx])
            continue
        dists = []
        for coord in coords:
            min_dist = min(np.sum((coord - pt) ** 2) for pt in selected)
            dists.append(min_dist)
        best_idx = int(np.argmax(np.array(dists)))
        candidate = coords[best_idx]
        if any(np.allclose(candidate, pt) for pt in selected):
            break
        selected.append(candidate)

    return [(float(pt[0]), float(pt[1])) for pt in selected]


def _component_region_points(component: np.ndarray) -> List[Tuple[float, float]]:
    ys, xs = np.where(component > 0)
    if len(xs) == 0:
        return []

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0

    ordered = [
        _region_center_point(component, x_min, x_mid + 1, y_min, y_mid + 1),
        _region_center_point(component, x_mid, x_max + 1, y_min, y_mid + 1),
        _region_center_point(component, x_min, x_mid + 1, y_mid, y_max + 1),
        _region_center_point(component, x_mid, x_max + 1, y_mid, y_max + 1),
        _center_point(component),
    ]

    unique_points: List[Tuple[float, float]] = []
    for point in ordered:
        if point is None:
            continue
        if point not in unique_points:
            unique_points.append(point)
    return unique_points


def select_five_prompt_points(mask: np.ndarray, max_points: int = 5) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    components = _connected_components(mask)
    if not components:
        components = [mask]

    unique_points: List[Tuple[float, float]] = []

    # First guarantee at least one center point for each connected component.
    for component in components:
        center = _center_point(component)
        if center not in unique_points:
            unique_points.append(center)
        if len(unique_points) >= max_points:
            return np.array(unique_points[:max_points], dtype=np.float32)

    # Then add richer regional points, prioritizing larger components first.
    for component in components:
        for point in _component_region_points(component):
            if point not in unique_points:
                unique_points.append(point)
            if len(unique_points) >= max_points:
                return np.array(unique_points[:max_points], dtype=np.float32)

    unique_points = _greedy_fill_points(mask, unique_points, max_points)
    return np.array(unique_points[:max_points], dtype=np.float32)


def _select_instance_positive_points(mask: np.ndarray, max_points: int = 3) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    max_points = max(1, int(max_points))
    if int(mask.sum()) <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    selected_points: List[Tuple[float, float]] = []
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    best_value = float(distance.max()) if distance.size > 0 else 0.0
    if best_value > 0.0:
        ys, xs = np.where(distance >= (best_value - 1e-6))
        center_x, center_y = _center_point(mask)
        coords = np.stack([xs, ys], axis=1).astype(np.float32)
        target = np.array([center_x, center_y], dtype=np.float32)
        point_idx = int(np.argmin(np.sum((coords - target) ** 2, axis=1)))
        selected_points.append((float(coords[point_idx, 0]), float(coords[point_idx, 1])))
    else:
        selected_points.append(_center_point(mask))

    for point in _component_region_points(mask):
        if point not in selected_points:
            selected_points.append(point)
        if len(selected_points) >= max_points:
            break

    selected_points = _greedy_fill_points(mask, selected_points, max_points)
    return np.array(selected_points[:max_points], dtype=np.float32)


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


class SAMHelper:
    def __init__(
        self,
        checkpoint: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        multimask_output: bool = True,
        max_masks: int = 3,
        overlap_thresh: float = 0.9,
        area_limit: float = 0.85,
    ):
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.multimask_output = multimask_output
        self.max_masks = max_masks
        self.overlap_thresh = overlap_thresh
        self.area_limit = area_limit

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam = sam.to(device)
        self.predictor = SamPredictor(sam)

    def _run_sam(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> SAMResult:
        image_rgb = _ensure_uint8_rgb(image_rgb)
        self.predictor.set_image(np.ascontiguousarray(image_rgb))

        all_candidates: List[SAMCandidate] = []
        for idx in range(points_xy.shape[0]):
            point = points_xy[idx:idx + 1].astype(np.float32)
            labels = np.ones((1,), dtype=np.int32)
            masks, scores, _ = self.predictor.predict(
                point_coords=point,
                point_labels=labels,
                multimask_output=self.multimask_output,
            )
            for j in range(len(scores)):
                mask = np.asarray(masks[j]).astype(np.uint8)
                score = float(np.asarray(scores)[j])
                all_candidates.append(SAMCandidate(mask=mask, score=score, point_idx=idx))

        all_candidates.sort(key=lambda item: item.score, reverse=True)

        kept_masks: List[np.ndarray] = []
        kept_candidates: List[SAMCandidate] = []
        for cand in all_candidates:
            mask_bool = cand.mask.astype(bool)
            area_ratio = float(mask_bool.mean())
            if area_ratio > self.area_limit:
                continue

            skip = False
            for kept in kept_masks:
                inter = np.logical_and(mask_bool, kept).sum()
                union = np.logical_or(mask_bool, kept).sum()
                if union > 0 and inter / union > self.overlap_thresh:
                    skip = True
                    break
            if skip:
                continue

            kept_masks.append(mask_bool)
            kept_candidates.append(cand)
            if len(kept_masks) >= self.max_masks:
                break

        if not kept_masks:
            return SAMResult(mask=None, points_xy=points_xy, candidates=all_candidates, success=False)

        union_mask = np.logical_or.reduce(kept_masks).astype(np.uint8)
        return SAMResult(mask=union_mask, points_xy=points_xy, candidates=kept_candidates, success=True)

    def predict_from_red_region(self, image_rgb: np.ndarray, red_mask: np.ndarray) -> SAMResult:
        points_xy = select_five_prompt_points(red_mask, max_points=5)
        if points_xy.shape[0] == 0:
            return SAMResult(mask=None, points_xy=points_xy, candidates=[], success=False)
        return self._run_sam(image_rgb, points_xy)

    def predict_from_red_overlay(self, image_rgb: np.ndarray, overlay_rgb: np.ndarray) -> SAMResult:
        red_mask = extract_red_mask_from_overlay(overlay_rgb)
        return self.predict_from_red_region(image_rgb, red_mask)

    @staticmethod
    def draw_points(image_rgb: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        image = _ensure_uint8_rgb(image_rgb).copy()
        for x, y in np.atleast_2d(points_xy):
            cv2.circle(image, (int(round(x)), int(round(y))), 5, (0, 255, 0), -1)
        return image

    @staticmethod
    def draw_box(image_rgb: np.ndarray, box_xyxy: Optional[np.ndarray]) -> np.ndarray:
        image = _ensure_uint8_rgb(image_rgb).copy()
        if box_xyxy is None or np.asarray(box_xyxy).size != 4:
            return image
        x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32).tolist()
        cv2.rectangle(
            image,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            (255, 255, 0),
            2,
        )
        return image

    @staticmethod
    def draw_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        image = _ensure_uint8_rgb(image_rgb).astype(np.float32) / 255.0
        mask = _ensure_binary_mask(mask).astype(bool)
        color = np.zeros_like(image, dtype=np.float32)
        color[mask] = [1.0, 0.0, 1.0]
        overlay = (1 - alpha) * image + alpha * color
        return np.clip(overlay * 255.0, 0, 255).astype(np.uint8)


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


def _mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    mask = _ensure_binary_mask(mask)
    # For disconnected tiny regions, only box the dominant foreground component.
    mask = _largest_component(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _select_box_refined_points(mask: np.ndarray, image_hw: Tuple[int, int]) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    component = _largest_component(mask)
    if component.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    selected_points: List[Tuple[float, float]] = []

    component_center = _center_point(component)
    selected_points.append(component_center)

    image_h, image_w = image_hw
    image_center_x = (float(image_w) - 1.0) / 2.0
    image_center_y = (float(image_h) - 1.0) / 2.0
    near_image_center = _nearest_distinct_mask_point(
        component,
        image_center_x,
        image_center_y,
        exclude_points=selected_points,
    )
    if near_image_center is not None:
        selected_points.append(near_image_center)

    return np.array(selected_points[:2], dtype=np.float32)


def _scale_points(points_xy: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = points_xy.astype(np.float32).copy()
    if src_w > 0 and dst_w != src_w:
        scaled[:, 0] *= float(dst_w) / float(src_w)
    if src_h > 0 and dst_h != src_h:
        scaled[:, 1] *= float(dst_h) / float(src_h)
    return scaled


def _scale_box(box_xyxy: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = box_xyxy.astype(np.float32).copy()
    if src_w > 0 and dst_w != src_w:
        scaled[[0, 2]] *= float(dst_w) / float(src_w)
    if src_h > 0 and dst_h != src_h:
        scaled[[1, 3]] *= float(dst_h) / float(src_h)
    return scaled


def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = _ensure_binary_mask(mask_a).astype(bool)
    mask_b = _ensure_binary_mask(mask_b).astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


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


class SAMTrainHelper:
    def __init__(
        self,
        checkpoint: str,
        *,
        save_root: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        multimask_output: bool = True,
        max_masks: int = 3,
        overlap_thresh: float = 0.9,
        area_limit: float = 0.85,
        score_thresh: float = 0.6,
        heat_iou_thresh: float = 0.1,
        large_target_heat_iou_thresh: float = 0.15,
        bg_iou_thresh: float = 0.15,
        large_target_bg_iou_thresh: float = 0.30,
        large_area_thresh: float = 0.06,
        large_points: int = 5,
        small_points: int = 1,
        large_uncertain_area_thresh: float = 0.30,
        rule_a_heat_iou_thresh: float = 0.4,
        rule_b_heat_iou_delta: float = 0.05,
        heat_bin_thr: float = 0.5,
        resize_short_edge: int = 640,
        use_crf: bool = False,
        small_fg_box_thresh: float = 0.04,
        use_affinity_split: bool = False,
        depth_root: str = "",
        seed_points_per_instance: int = 3,
        affinity_min_component_area: int = 128,
        affinity_min_instance_area: int = 64,
        affinity_superpixel_size: int = 20,
        affinity_min_superpixel_area: int = 48,
        affinity_slic_compactness: float = 6.0,
        affinity_slic_sigma: float = 0.0,
        affinity_slic_depth_scale: float = 0.5,
        affinity_sigma_sem: float = 0.20,
        affinity_sigma_dep: float = 0.04,
        affinity_sigma_spatial: float = 0.12,
        affinity_sigma_edge: float = 0.30,
        affinity_min_affinity: float = 1e-6,
        affinity_min_cluster_regions: int = 2,
        affinity_ncut_threshold: float = 0.12,
        affinity_max_recursion_depth: int = 8,
        dino_weight: str = "",
        dino_model: str = "dinov2_vitl14",
        dino_repo: str = "",
        dino_device: str = "",
        dino_max_side: int = 700,
        dino_pca_dim: int = 64,
    ):
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
        self.predictor = SamPredictor(sam_model_registry[model_type](checkpoint=checkpoint).to(device))
        self.save_root = save_root
        self.multimask_output = multimask_output
        self.max_masks = max_masks
        self.overlap_thresh = overlap_thresh
        self.area_limit = area_limit
        self.score_thresh = score_thresh
        self.heat_iou_thresh = heat_iou_thresh
        self.large_target_heat_iou_thresh = large_target_heat_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.large_target_bg_iou_thresh = large_target_bg_iou_thresh
        self.large_area_thresh = large_area_thresh
        self.large_points = large_points
        self.small_points = small_points
        self.large_uncertain_area_thresh = large_uncertain_area_thresh
        self.rule_a_heat_iou_thresh = rule_a_heat_iou_thresh
        self.rule_b_heat_iou_delta = rule_b_heat_iou_delta
        self.heat_bin_thr = heat_bin_thr
        self.resize_short_edge = resize_short_edge
        self.use_crf = use_crf
        self.small_fg_box_thresh = small_fg_box_thresh
        self.use_affinity_split = bool(use_affinity_split)
        self.depth_root = depth_root
        self.depth_index = _build_stem_path_index(depth_root)
        self.seed_points_per_instance = max(1, int(seed_points_per_instance))
        self.affinity_min_component_area = int(max(1, affinity_min_component_area))
        self.affinity_min_instance_area = int(max(1, affinity_min_instance_area))
        self.affinity_superpixel_size = int(max(8, affinity_superpixel_size))
        self.affinity_min_superpixel_area = int(max(1, affinity_min_superpixel_area))
        self.affinity_slic_compactness = float(max(0.0, affinity_slic_compactness))
        self.affinity_slic_sigma = float(max(0.0, affinity_slic_sigma))
        self.affinity_slic_depth_scale = float(max(0.0, affinity_slic_depth_scale))
        self.affinity_sigma_sem = float(max(1e-6, affinity_sigma_sem))
        self.affinity_sigma_dep = float(max(1e-6, affinity_sigma_dep))
        self.affinity_sigma_spatial = float(max(1e-6, affinity_sigma_spatial))
        self.affinity_sigma_edge = float(max(1e-6, affinity_sigma_edge))
        self.affinity_min_affinity = float(max(0.0, affinity_min_affinity))
        self.affinity_min_cluster_regions = int(max(1, affinity_min_cluster_regions))
        self.affinity_ncut_threshold = float(max(0.0, affinity_ncut_threshold))
        self.affinity_max_recursion_depth = int(max(1, affinity_max_recursion_depth))
        self.dino_weight = str(dino_weight or "").strip()
        self.dino_model = str(dino_model or "dinov2_vitl14").strip() or "dinov2_vitl14"
        self.dino_repo = str(dino_repo or "").strip()
        self.dino_device = str(dino_device or device).strip() or str(device)
        self.dino_max_side = int(max(0, dino_max_side))
        self.dino_pca_dim = int(max(0, dino_pca_dim))
        self._dino_extractor = None
        self._dino_checked = False
        self.mask_logit_threshold = float(self.predictor.model.mask_threshold)
        self.mask_prob_threshold = float(1.0 / (1.0 + np.exp(-self.mask_logit_threshold)))
        self.failed_names: Dict[int, List[str]] = {}
        if self.use_affinity_split:
            if self.depth_index:
                print(
                    "Using CCAM prompt affinity split with depth maps from {} ({} maps).".format(
                        os.path.abspath(str(depth_root)),
                        len(self.depth_index),
                    )
                )
            else:
                print("Using CCAM prompt affinity split with RGB-only fallback.")
            if self.dino_weight:
                print("DINO semantic affinity requested from {}.".format(os.path.abspath(self.dino_weight)))

    def _make_out_dir(self, prefix: str, epoch: int) -> str:
        out_dir = os.path.join(self.save_root, prefix, f"epoch{epoch + 1}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _select_points(self, region_mask: np.ndarray) -> np.ndarray:
        region_mask = _ensure_binary_mask(region_mask)
        if region_mask.sum() == 0:
            return np.zeros((0, 2), dtype=np.float32)

        components = _connected_components(region_mask)
        if not components:
            components = [region_mask]

        selected_points: List[Tuple[float, float]] = []
        for component in components:
            component_area_ratio = float(component.mean())
            if component_area_ratio >= self.large_area_thresh:
                component_points = _component_region_points(component)
                component_points = _greedy_fill_points(component, component_points, self.large_points)
                component_points = component_points[:self.large_points]
            else:
                component_points = _greedy_fill_points(component, [_center_point(component)], self.small_points)
                component_points = component_points[:self.small_points]

            for point in component_points:
                if point not in selected_points:
                    selected_points.append(point)

        return np.array(selected_points, dtype=np.float32)

    def _load_depth_map(self, sample_name: str, out_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self.depth_index:
            return None
        stem = os.path.splitext(os.path.basename(sample_name))[0]
        depth_path = self.depth_index.get(stem)
        if depth_path is None:
            return None
        depth_gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_gray is None:
            return None
        depth_map = _normalize_gray_map(depth_gray)
        if depth_map.shape != tuple(out_shape):
            depth_map = cv2.resize(depth_map, out_shape[::-1], interpolation=cv2.INTER_LINEAR)
            depth_map = _normalize_gray_map(depth_map)
        return depth_map.astype(np.float32)

    def _get_dino_extractor(self):
        if self._dino_checked:
            return self._dino_extractor
        self._dino_checked = True
        if not self.dino_weight:
            return None
        if not os.path.exists(self.dino_weight):
            print("Warn: DINO semantic affinity checkpoint not found: {}. Using depth/RGB affinity only.".format(self.dino_weight))
            return None
        try:
            from dinov2_feature_viz import _OnTheFlyDINOExtractor, _default_dino_repo

            self._dino_extractor = _OnTheFlyDINOExtractor(
                weight_path=Path(self.dino_weight),
                model_name=self.dino_model,
                repo_path=self.dino_repo or _default_dino_repo(),
                device=self.dino_device,
                max_side=self.dino_max_side,
            )
            print("Using DINO semantic affinity from {}.".format(os.path.abspath(self.dino_weight)))
        except Exception as exc:
            self._dino_extractor = None
            print("Warn: failed to initialize DINO semantic affinity extractor: {}. Using depth/RGB affinity only.".format(exc))
        return self._dino_extractor

    def _extract_dino_appearance_map(
        self,
        image_rgb: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        extractor = self._get_dino_extractor()
        if extractor is None:
            return None
        try:
            return extractor.extract(image_rgb, target_shape=target_shape)
        except Exception as exc:
            print("Warn: failed to extract DINO semantic affinity features: {}. Using depth/RGB affinity only for this sample.".format(exc))
            return None

    def _split_prompt_mask_with_local_affinity(
        self,
        prompt_mask: np.ndarray,
        image_rgb: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        prompt_mask = _ensure_binary_mask(prompt_mask)
        if int(prompt_mask.sum()) <= 0:
            return []
        try:
            from skimage.segmentation import slic
        except Exception:
            return _connected_components(prompt_mask)

        depth = (
            _normalize_gray_map(depth_map)
            if depth_map is not None
            else np.zeros(prompt_mask.shape, dtype=np.float32)
        )
        rgb = _ensure_uint8_rgb(image_rgb)
        if rgb.shape[:2] != prompt_mask.shape:
            rgb = cv2.resize(rgb, prompt_mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
        slic_input = np.concatenate(
            [lab, (self.affinity_slic_depth_scale * depth)[..., None].astype(np.float32)],
            axis=2,
        )

        label_map = np.zeros_like(prompt_mask, dtype=np.int32)
        next_label = 1
        for component in _connected_components(prompt_mask):
            component = _ensure_binary_mask(component)
            area = int(component.sum())
            if area < self.affinity_min_component_area:
                label_map[component > 0] = next_label
                next_label += 1
                continue
            n_segments = max(1, int(round(float(area) / float(self.affinity_superpixel_size * self.affinity_superpixel_size))))
            if n_segments <= 1:
                label_map[component > 0] = next_label
                next_label += 1
                continue
            try:
                labels = slic(
                    slic_input,
                    n_segments=n_segments,
                    compactness=self.affinity_slic_compactness,
                    sigma=self.affinity_slic_sigma,
                    start_label=1,
                    mask=(component > 0),
                    convert2lab=False,
                    enforce_connectivity=True,
                    min_size_factor=0.4,
                    max_size_factor=3.0,
                    channel_axis=-1,
                ).astype(np.int32)
            except Exception:
                label_map[component > 0] = next_label
                next_label += 1
                continue

            accepted = 0
            for local_label in sorted(int(value) for value in np.unique(labels) if int(value) > 0):
                region = ((labels == local_label) & (component > 0)).astype(np.uint8)
                region = _filter_components(region, min_area=self.affinity_min_superpixel_area)
                if int(region.sum()) < self.affinity_min_instance_area:
                    continue
                label_map[region > 0] = next_label
                next_label += 1
                accepted += 1
            if accepted == 0:
                label_map[component > 0] = next_label
                next_label += 1

        return _label_map_to_components(label_map, min_area=max(16, int(self.affinity_min_instance_area // 2)))

    def _split_prompt_mask_into_affinity_instances(
        self,
        prompt_mask: np.ndarray,
        image_rgb: np.ndarray,
        depth_map: Optional[np.ndarray],
    ) -> List[np.ndarray]:
        prompt_mask = _ensure_binary_mask(prompt_mask)
        if int(prompt_mask.sum()) <= 0:
            return []
        depth = (
            _normalize_gray_map(depth_map)
            if depth_map is not None
            else np.zeros(prompt_mask.shape, dtype=np.float32)
        )
        appearance_map = self._extract_dino_appearance_map(image_rgb, target_shape=prompt_mask.shape)
        try:
            from depth_affinity_spectral_demo import split_depth_instances_affinity_spectral

            results = split_depth_instances_affinity_spectral(
                depth=depth,
                gt_mask=prompt_mask,
                rgb=image_rgb,
                appearance_map=appearance_map,
                min_component_area=self.affinity_min_component_area,
                min_instance_area=self.affinity_min_instance_area,
                median_ksize=5,
                bilateral_d=7,
                bilateral_sigma_color=25.0,
                bilateral_sigma_space=25.0,
                superpixel_size=self.affinity_superpixel_size,
                min_superpixel_area=self.affinity_min_superpixel_area,
                slic_compactness=self.affinity_slic_compactness,
                slic_sigma=self.affinity_slic_sigma,
                slic_depth_scale=self.affinity_slic_depth_scale,
                dino_pca_dim=self.dino_pca_dim,
                sigma_sem=self.affinity_sigma_sem,
                sigma_dep=self.affinity_sigma_dep,
                sigma_spatial=self.affinity_sigma_spatial,
                sigma_edge=self.affinity_sigma_edge,
                min_affinity=self.affinity_min_affinity,
                min_cluster_regions=self.affinity_min_cluster_regions,
                ncut_threshold=self.affinity_ncut_threshold,
                max_recursion_depth=self.affinity_max_recursion_depth,
            )
            components = _label_map_to_components(
                results.get("label_map", np.zeros_like(prompt_mask, dtype=np.int32)),
                min_area=max(16, int(self.affinity_min_instance_area // 2)),
            )
            if components:
                return components
        except Exception as exc:
            if not hasattr(self, "_affinity_import_warned"):
                self._affinity_import_warned = True
                print(
                    "Warn: depth_affinity_spectral_demo is unavailable ({}). "
                    "Using local RGB/depth SLIC split fallback.".format(exc)
                )

        components = self._split_prompt_mask_with_local_affinity(prompt_mask, image_rgb, depth_map)
        return components if components else _connected_components(prompt_mask)

    def _run_sam_single(self, image_rgb: np.ndarray, points_xy: np.ndarray) -> List[SAMCandidate]:
        self.predictor.set_image(np.ascontiguousarray(_ensure_uint8_rgb(image_rgb)))
        candidates: List[SAMCandidate] = []
        for point_idx in range(points_xy.shape[0]):
            point = points_xy[point_idx:point_idx + 1].astype(np.float32)
            point_labels = np.ones((1,), dtype=np.int32)
            mask_logits, scores, _ = self.predictor.predict(
                point_coords=point,
                point_labels=point_labels,
                multimask_output=self.multimask_output,
                return_logits=True,
            )
            for logit_mask, score in zip(np.asarray(mask_logits), np.asarray(scores).reshape(-1)):
                prob_map = _logits_to_prob_map(logit_mask)
                candidates.append(
                    SAMCandidate(
                        mask=_threshold_prob_map(prob_map, self.mask_prob_threshold),
                        score=float(score),
                        point_idx=point_idx,
                        logits=np.asarray(logit_mask, dtype=np.float32),
                    )
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _run_sam_box(self, image_rgb: np.ndarray, box_xyxy: np.ndarray) -> List[SAMCandidate]:
        if box_xyxy.size != 4:
            return []

        self.predictor.set_image(np.ascontiguousarray(_ensure_uint8_rgb(image_rgb)))
        mask_logits, scores, _ = self.predictor.predict(
            box=box_xyxy.astype(np.float32),
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        candidates: List[SAMCandidate] = []
        for logit_mask, score in zip(np.asarray(mask_logits), np.asarray(scores).reshape(-1)):
            prob_map = _logits_to_prob_map(logit_mask)
            candidates.append(
                SAMCandidate(
                    mask=_threshold_prob_map(prob_map, self.mask_prob_threshold),
                    score=float(score),
                    point_idx=-1,
                    logits=np.asarray(logit_mask, dtype=np.float32),
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _run_sam_multi_positive(self, image_rgb: np.ndarray, points_xy: np.ndarray, point_idx: int = -1) -> List[SAMCandidate]:
        if points_xy.shape[0] == 0:
            return []

        self.predictor.set_image(np.ascontiguousarray(_ensure_uint8_rgb(image_rgb)))
        point_labels = np.ones((points_xy.shape[0],), dtype=np.int32)
        mask_logits, scores, _ = self.predictor.predict(
            point_coords=points_xy.astype(np.float32),
            point_labels=point_labels,
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        candidates: List[SAMCandidate] = []
        for logit_mask, score in zip(np.asarray(mask_logits), np.asarray(scores).reshape(-1)):
            prob_map = _logits_to_prob_map(logit_mask)
            candidates.append(
                SAMCandidate(
                    mask=_threshold_prob_map(prob_map, self.mask_prob_threshold),
                    score=float(score),
                    point_idx=int(point_idx),
                    logits=np.asarray(logit_mask, dtype=np.float32),
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def _prepare_candidate_outputs(self, candidate: SAMCandidate, target_hw: Tuple[int, int]) -> None:
        if candidate.logits is not None:
            candidate.prob_orig = _resize_score_map(_logits_to_prob_map(candidate.logits), target_hw)
            candidate.mask_orig = _threshold_prob_map(candidate.prob_orig, self.mask_prob_threshold)
        else:
            candidate.mask_orig = _resize_mask(candidate.mask, target_hw)
            candidate.prob_orig = candidate.mask_orig.astype(np.float32)

    def _ensure_candidate_masks(self, candidates: List[SAMCandidate], target_hw: Tuple[int, int]) -> None:
        for candidate in candidates:
            self._prepare_candidate_outputs(candidate, target_hw)

    @staticmethod
    def _candidate_area_ratio(candidate: SAMCandidate) -> float:
        return float(_ensure_binary_mask(candidate.mask_orig).astype(bool).mean())

    def _prepare_candidates_by_point(
        self,
        candidates: List[SAMCandidate],
        heat_iou_mask: np.ndarray,
        bg_mask: np.ndarray,
        fg_iou_masks_by_point: Optional[Dict[int, np.ndarray]] = None,
        metric_heat_iou_mask: Optional[np.ndarray] = None,
        metric_bg_iou_mask: Optional[np.ndarray] = None,
    ) -> Dict[int, List[SAMCandidate]]:
        target_hw = heat_iou_mask.shape
        metric_heat_iou_mask = heat_iou_mask if metric_heat_iou_mask is None else metric_heat_iou_mask
        metric_bg_iou_mask = bg_mask if metric_bg_iou_mask is None else metric_bg_iou_mask
        candidates_by_point: Dict[int, List[SAMCandidate]] = {}

        for candidate in candidates:
            self._prepare_candidate_outputs(candidate, target_hw)
            if fg_iou_masks_by_point is not None and candidate.point_idx in fg_iou_masks_by_point:
                candidate.fg_iou = _compute_iou(candidate.mask_orig, fg_iou_masks_by_point[candidate.point_idx])
            else:
                candidate.fg_iou = 0.0
            candidate.heat_iou = _compute_iou(candidate.mask_orig, metric_heat_iou_mask)
            candidate.bg_iou = _compute_iou(candidate.mask_orig, metric_bg_iou_mask)
            candidate.filter_heat_iou = _compute_iou(candidate.mask_orig, heat_iou_mask)
            candidate.filter_bg_iou = _compute_iou(candidate.mask_orig, bg_mask)
            candidates_by_point.setdefault(candidate.point_idx, []).append(candidate)

        return candidates_by_point

    def _get_valid_candidates_by_point(
        self,
        candidates_by_point: Dict[int, List[SAMCandidate]],
        heat_iou_thresh: Optional[float] = None,
        bg_iou_thresh: Optional[float] = None,
    ) -> Dict[int, List[SAMCandidate]]:
        effective_heat_iou_thresh = self.heat_iou_thresh if heat_iou_thresh is None else float(heat_iou_thresh)
        effective_bg_iou_thresh = self.bg_iou_thresh if bg_iou_thresh is None else float(bg_iou_thresh)
        valid_candidates_by_point: Dict[int, List[SAMCandidate]] = {}

        for point_idx in sorted(candidates_by_point.keys()):
            valid_candidates = [
                candidate for candidate in candidates_by_point[point_idx]
                if candidate.score >= self.score_thresh
                and candidate.filter_heat_iou >= effective_heat_iou_thresh
                and candidate.filter_bg_iou <= effective_bg_iou_thresh
                and self._candidate_area_ratio(candidate) <= self.area_limit
            ]
            if valid_candidates:
                valid_candidates_by_point[point_idx] = valid_candidates

        return valid_candidates_by_point

    def _select_best_candidates(
        self,
        valid_candidates_by_point: Dict[int, List[SAMCandidate]],
    ) -> List[SAMCandidate]:
        selected_candidates: List[SAMCandidate] = []
        for point_idx in sorted(valid_candidates_by_point.keys()):
            point_candidates = valid_candidates_by_point[point_idx]

            best_heat_candidate = max(
                point_candidates,
                key=lambda item: (float(item.filter_heat_iou), float(item.score)),
            )
            selected_candidates.append(best_heat_candidate)

            best_score_candidate = max(
                point_candidates,
                key=lambda item: (float(item.score), float(item.filter_heat_iou)),
            )
            if id(best_score_candidate) != id(best_heat_candidate):
                selected_candidates.append(best_score_candidate)
        return selected_candidates

    def _select_rule_a_candidates(
        self,
        valid_candidates_by_point: Dict[int, List[SAMCandidate]],
    ) -> List[SAMCandidate]:
        selected_candidates: List[SAMCandidate] = []
        for point_idx in sorted(valid_candidates_by_point.keys()):
            for candidate in valid_candidates_by_point[point_idx]:
                if candidate.filter_heat_iou >= self.rule_a_heat_iou_thresh:
                    selected_candidates.append(candidate)
        return selected_candidates

    def _select_rule_ab_candidates(
        self,
        valid_candidates_by_point: Dict[int, List[SAMCandidate]],
    ) -> List[SAMCandidate]:
        selected_candidates: List[SAMCandidate] = []
        seen_candidate_ids = set()

        for point_idx in sorted(valid_candidates_by_point.keys()):
            point_candidates = valid_candidates_by_point[point_idx]
            point_best_heat_iou = max(float(candidate.filter_heat_iou) for candidate in point_candidates)

            if point_best_heat_iou >= self.rule_a_heat_iou_thresh:
                point_selected = [
                    candidate for candidate in point_candidates
                    if candidate.filter_heat_iou >= self.rule_a_heat_iou_thresh
                ]
            else:
                point_selected = [
                    candidate for candidate in point_candidates
                    if point_best_heat_iou - float(candidate.filter_heat_iou) <= self.rule_b_heat_iou_delta
                ]

            for candidate in point_selected:
                candidate_id = id(candidate)
                if candidate_id not in seen_candidate_ids:
                    seen_candidate_ids.add(candidate_id)
                    selected_candidates.append(candidate)

        return selected_candidates

    def _select_rule_b_candidates(
        self,
        valid_candidates_by_point: Dict[int, List[SAMCandidate]],
    ) -> List[SAMCandidate]:
        selected_candidates: List[SAMCandidate] = []
        seen_candidate_ids = set()

        for point_idx in sorted(valid_candidates_by_point.keys()):
            point_candidates = valid_candidates_by_point[point_idx]
            best_candidate = max(
                point_candidates,
                key=lambda item: (float(item.filter_heat_iou), float(item.score)),
            )
            point_best_heat_iou = float(best_candidate.filter_heat_iou)

            if point_best_heat_iou < self.rule_a_heat_iou_thresh:
                point_selected = [
                    candidate for candidate in point_candidates
                    if point_best_heat_iou - float(candidate.filter_heat_iou) <= self.rule_b_heat_iou_delta
                ]
            else:
                point_selected = [best_candidate]

            for candidate in point_selected:
                candidate_id = id(candidate)
                if candidate_id not in seen_candidate_ids:
                    seen_candidate_ids.add(candidate_id)
                    selected_candidates.append(candidate)

        return selected_candidates

    @staticmethod
    def _merge_candidate_masks(
        candidates: List[SAMCandidate],
        fallback_mask: np.ndarray,
    ) -> np.ndarray:
        if candidates:
            return np.logical_or.reduce(
                [_ensure_binary_mask(candidate.mask_orig).astype(bool) for candidate in candidates]
            ).astype(np.uint8)
        return _ensure_binary_mask(fallback_mask).copy()

    @staticmethod
    def _merge_candidate_prob_maps(
        candidates: List[SAMCandidate],
        fallback_mask: np.ndarray,
    ) -> np.ndarray:
        if candidates:
            prob_maps: List[np.ndarray] = []
            for candidate in candidates:
                if candidate.prob_orig is not None:
                    prob_maps.append(np.clip(candidate.prob_orig, 0.0, 1.0).astype(np.float32))
                elif candidate.mask_orig is not None:
                    prob_maps.append(_ensure_binary_mask(candidate.mask_orig).astype(np.float32))
                else:
                    prob_maps.append(_ensure_binary_mask(candidate.mask).astype(np.float32))
            return np.maximum.reduce(prob_maps).astype(np.float32)
        return _ensure_binary_mask(fallback_mask).astype(np.float32)

    def _save_points_heatmap_overlay(
        self,
        image_rgb: np.ndarray,
        attn_prob: np.ndarray,
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        filename: str,
        epoch: int,
    ) -> None:
        out_dir = self._make_out_dir("points_attn_heatmap", epoch)
        raw_dir = os.path.join(out_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        heat_rgb = _draw_heatmap_overlay(np.zeros_like(image_rgb), attn_prob, alpha=1.0)
        overlay = _draw_heatmap_overlay(image_rgb, attn_prob, alpha=0.4)
        if box_xyxy is not None:
            overlay = SAMHelper.draw_box(overlay, box_xyxy)
        overlay = SAMHelper.draw_points(overlay, points_xy)

        stem = os.path.splitext(filename)[0]
        _write_rgb(os.path.join(out_dir, f"{stem}.png"), overlay)
        _write_rgb(os.path.join(raw_dir, f"{stem}.png"), heat_rgb)

    def _save_points_seg_overlay(
        self,
        image_rgb: np.ndarray,
        mask_orig: Optional[np.ndarray],
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        filename: str,
        epoch: int,
    ) -> None:
        if mask_orig is None:
            return
        out_dir = self._make_out_dir("points_sam_seg", epoch)
        stem = os.path.splitext(filename)[0]
        overlay_orig = SAMHelper.draw_mask_overlay(image_rgb, mask_orig)
        if box_xyxy is not None:
            overlay_orig = SAMHelper.draw_box(overlay_orig, box_xyxy)
        overlay_orig = SAMHelper.draw_points(overlay_orig, points_xy)
        _write_rgb(os.path.join(out_dir, f"{stem}.png"), overlay_orig)

    def _save_affinity_split_result(
        self,
        image_rgb: np.ndarray,
        prompt_mask: np.ndarray,
        seed_masks: List[np.ndarray],
        points_xy: np.ndarray,
        filename: str,
        epoch: int,
    ) -> None:
        if not self.use_affinity_split:
            return
        stem = os.path.splitext(filename)[0]
        prompt_dir = self._make_out_dir("affinity_prompt_mask", epoch)
        overlay_dir = self._make_out_dir("affinity_split_overlay", epoch)
        label_dir = self._make_out_dir("affinity_split_labels", epoch)
        union_dir = self._make_out_dir("affinity_split_union", epoch)

        prompt_mask = _ensure_binary_mask(prompt_mask)
        seed_masks = [_ensure_binary_mask(seed_mask) for seed_mask in seed_masks if int(_ensure_binary_mask(seed_mask).sum()) > 0]
        label_rgb = _seed_masks_to_label_rgb(seed_masks, image_rgb.shape[:2])
        overlay = _seed_masks_overlay(image_rgb, seed_masks, alpha=0.45)
        overlay = SAMHelper.draw_points(overlay, points_xy)
        union_mask = _seed_masks_to_union(seed_masks, image_rgb.shape[:2])

        cv2.imwrite(os.path.join(prompt_dir, f"{stem}.png"), prompt_mask.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(union_dir, f"{stem}.png"), union_mask.astype(np.uint8) * 255)
        _write_rgb(os.path.join(label_dir, f"{stem}.png"), label_rgb)
        _write_rgb(os.path.join(overlay_dir, f"{stem}.png"), overlay)

    def _save_per_point_candidates(
        self,
        image_rgb: np.ndarray,
        raw_candidates: List[SAMCandidate],
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        heat_iou_mask: np.ndarray,
        bg_mask: np.ndarray,
        uncertain_area_ratio: float,
        filename: str,
        epoch: int,
        output_prefixes: Optional[List[str]] = None,
        fg_iou_masks_by_point: Optional[Dict[int, np.ndarray]] = None,
        metric_heat_iou_mask: Optional[np.ndarray] = None,
        metric_bg_iou_mask: Optional[np.ndarray] = None,
    ) -> None:
        if not raw_candidates:
            return
        stem = os.path.splitext(filename)[0]
        if output_prefixes is None:
            output_prefixes = ["point_candidates"]
        metric_heat_iou_mask = heat_iou_mask if metric_heat_iou_mask is None else metric_heat_iou_mask
        metric_bg_iou_mask = bg_mask if metric_bg_iou_mask is None else metric_bg_iou_mask

        candidates_by_point: Dict[int, List[SAMCandidate]] = {}
        for candidate in raw_candidates:
            mask_orig = candidate.mask_orig if candidate.mask_orig is not None else _resize_mask(candidate.mask, heat_iou_mask.shape)
            candidate.mask_orig = mask_orig
            if fg_iou_masks_by_point is not None and candidate.point_idx in fg_iou_masks_by_point:
                candidate.fg_iou = _compute_iou(mask_orig, fg_iou_masks_by_point[candidate.point_idx])
            else:
                candidate.fg_iou = 0.0
            candidate.heat_iou = _compute_iou(mask_orig, metric_heat_iou_mask)
            candidate.bg_iou = _compute_iou(mask_orig, metric_bg_iou_mask)
            candidate.filter_heat_iou = _compute_iou(mask_orig, heat_iou_mask)
            candidate.filter_bg_iou = _compute_iou(mask_orig, bg_mask)
            candidates_by_point.setdefault(candidate.point_idx, []).append(candidate)

        for output_prefix in output_prefixes:
            out_dir = self._make_out_dir(output_prefix, epoch)
            for point_idx, cand_list in candidates_by_point.items():
                prompt_tag = "box" if point_idx < 0 else f"{point_idx + 1}"
                point_dir = os.path.join(out_dir, f"{stem}_{prompt_tag}")
                os.makedirs(point_dir, exist_ok=True)

                for cand_idx, candidate in enumerate(cand_list, start=1):
                    mask_orig = candidate.mask_orig if candidate.mask_orig is not None else _resize_mask(candidate.mask, heat_iou_mask.shape)
                    overlay = SAMHelper.draw_mask_overlay(image_rgb, mask_orig)
                    if point_idx < 0 and box_xyxy is not None:
                        overlay = SAMHelper.draw_box(overlay, box_xyxy)
                    elif point_idx < len(points_xy):
                        overlay = SAMHelper.draw_points(
                            overlay,
                            np.array([points_xy[point_idx]], dtype=np.float32),
                        )
                    save_name = (
                        f"{stem}_{'box' if point_idx < 0 else f'p{point_idx + 1}'}_c{cand_idx}"
                        f"_samiou{candidate.score:.2f}"
                        f"_fg_iou{candidate.fg_iou:.2f}"
                        f"_heat_iou{candidate.heat_iou:.2f}"
                        f"_bg_iou{candidate.bg_iou:.2f}"
                        f"_uncertain_area{uncertain_area_ratio:.2f}.png"
                    )
                    _write_rgb(os.path.join(point_dir, save_name), overlay)

    def _save_final_candidates(
        self,
        image_rgb: np.ndarray,
        candidates: List[SAMCandidate],
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        uncertain_area_ratio: float,
        filename: str,
        epoch: int,
    ) -> None:
        if not candidates:
            return
        stem = os.path.splitext(filename)[0]
        sample_dir = os.path.join(self._make_out_dir("sam_final_candidates", epoch), stem)
        os.makedirs(sample_dir, exist_ok=True)

        for candidate in candidates:
            mask_orig = candidate.mask_orig if candidate.mask_orig is not None else candidate.mask
            overlay = SAMHelper.draw_mask_overlay(image_rgb, mask_orig)
            if candidate.point_idx < 0 and box_xyxy is not None:
                overlay = SAMHelper.draw_box(overlay, box_xyxy)
                prompt_tag = "box"
            else:
                point = points_xy[candidate.point_idx] if candidate.point_idx < len(points_xy) else None
                if point is not None:
                    overlay = SAMHelper.draw_points(overlay, np.array([point], dtype=np.float32))
                prompt_tag = f"p{candidate.point_idx + 1}"
            save_name = (
                f"{stem}_{prompt_tag}"
                f"_samiou{candidate.score:.2f}"
                f"_fg_iou{candidate.fg_iou:.2f}"
                f"_heat_iou{candidate.heat_iou:.2f}"
                f"_bg_iou{candidate.bg_iou:.2f}"
                f"_uncertain_area{uncertain_area_ratio:.2f}.png"
            )
            _write_rgb(os.path.join(sample_dir, save_name), overlay)

    def _save_pseudo_label_binary(
        self,
        image_rgb: np.ndarray,
        prob_map: np.ndarray,
        filename: str,
        sample_meta: Optional[Dict[str, object]],
        sample_idx: int,
        epoch: int,
        prefix: str = "pseudo_labels",
    ) -> None:
        out_dir = self._make_out_dir(prefix, epoch)
        stem = os.path.splitext(filename)[0]
        prob_map = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)

        flipped = _sample_meta_value(sample_meta, "flipped", sample_idx, 0)
        orig_h = _sample_meta_value(sample_meta, "orig_h", sample_idx, prob_map.shape[0])
        orig_w = _sample_meta_value(sample_meta, "orig_w", sample_idx, prob_map.shape[1])

        if flipped:
            prob_map = np.ascontiguousarray(prob_map[:, ::-1])
        if prob_map.shape != (orig_h, orig_w):
            prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        map_u8 = (prob_map * 255.0).round().astype(np.uint8)
        mask_u8 = (map_u8 >= int(round(self.mask_prob_threshold * 255.0))).astype(np.uint8) * 255
        if self.use_crf:
            mask_u8 = (_refine_mask_with_crf(image_rgb, mask_u8) * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), mask_u8)

    def _save_pseudo_label_grayscale(
        self,
        prob_map: np.ndarray,
        filename: str,
        sample_meta: Optional[Dict[str, object]],
        sample_idx: int,
        epoch: int,
        prefix: str = "pseudo_labels",
    ) -> None:
        out_dir = self._make_out_dir(prefix, epoch)
        stem = os.path.splitext(filename)[0]
        prob_map = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)

        flipped = _sample_meta_value(sample_meta, "flipped", sample_idx, 0)
        orig_h = _sample_meta_value(sample_meta, "orig_h", sample_idx, prob_map.shape[0])
        orig_w = _sample_meta_value(sample_meta, "orig_w", sample_idx, prob_map.shape[1])

        if flipped:
            prob_map = np.ascontiguousarray(prob_map[:, ::-1])
        if prob_map.shape != (orig_h, orig_w):
            prob_map = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        map_u8 = (prob_map * 255.0).round().astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), map_u8)

    def _record_failure(self, filename: str, epoch: int) -> None:
        self.failed_names.setdefault(epoch, []).append(filename)

    def finalize_epoch(self, epoch: int) -> None:
        fail_dir = self._make_out_dir("sam_failures", epoch)
        log_path = os.path.join(fail_dir, "failures.txt")
        with open(log_path, "w", encoding="utf-8") as handle:
            for name in self.failed_names.get(epoch, []):
                handle.write(f"{name}\n")

    def process_batch(
        self,
        images_01: torch.Tensor,
        attn_prob: torch.Tensor,
        region_masks: torch.Tensor,
        bg_region_masks: torch.Tensor,
        sample_names: List[str],
        sample_meta: Optional[Dict[str, object]],
        epoch: int,
    ) -> None:
        images_01 = images_01.detach()
        attn_prob = attn_prob.detach().clamp(0, 1)
        region_masks = region_masks.detach()
        bg_region_masks = bg_region_masks.detach()

        for idx, sample_name in enumerate(sample_names):
            image_rgb = _tensor_to_uint8_rgb(images_01[idx])
            image_rgb_aug, aug_hw = _resize_to_short_edge(image_rgb, self.resize_short_edge)
            prob_np = attn_prob[idx, 0].cpu().numpy()
            prompt_mask = (region_masks[idx, 0].cpu().numpy() > 0.5).astype(np.uint8)
            fg_area_ratio = float(prompt_mask.mean())
            full_background_mask = (bg_region_masks[idx, 0].cpu().numpy() > 0.5).astype(np.uint8)
            heat_iou_mask = (prob_np >= self.heat_bin_thr).astype(np.uint8)
            uncertain_with_fg_mask = (1 - full_background_mask).astype(np.uint8)
            uncertain_area_ratio = float(uncertain_with_fg_mask.mean())
            is_large_target = float(uncertain_with_fg_mask.mean()) > self.large_uncertain_area_thresh
            heat_iou_ref_mask = uncertain_with_fg_mask if is_large_target else heat_iou_mask
            heat_iou_thresh = self.large_target_heat_iou_thresh if is_large_target else self.heat_iou_thresh
            bg_iou_thresh = self.large_target_bg_iou_thresh if is_large_target else self.bg_iou_thresh
            if prompt_mask.sum() == 0:
                self._record_failure(sample_name, epoch)
                continue

            box_xyxy = None
            points_xy = np.zeros((0, 2), dtype=np.float32)
            candidate_heat_iou_ref_mask = heat_iou_ref_mask
            candidate_metric_heat_iou_mask = candidate_heat_iou_ref_mask
            candidate_fg_iou_masks_by_point: Optional[Dict[int, np.ndarray]] = None
            fallback_mask = prompt_mask.copy()
            box_raw_candidates: List[SAMCandidate] = []
            point_candidate_prefixes = ["point_candidates"]
            if self.use_affinity_split:
                candidate_metric_heat_iou_mask = prompt_mask.copy()
                candidate_fg_iou_masks_by_point = {}
                depth_map = self._load_depth_map(sample_name, prompt_mask.shape)
                components = self._split_prompt_mask_into_affinity_instances(
                    prompt_mask=prompt_mask,
                    image_rgb=image_rgb,
                    depth_map=depth_map,
                )
                if not components:
                    self._record_failure(sample_name, epoch)
                    continue

                raw_candidates = []
                all_points: List[np.ndarray] = []
                point_group_idx = 0
                for component in components:
                    seed_mask = _ensure_binary_mask(component)
                    if int(seed_mask.sum()) <= 0:
                        continue
                    component_points = _select_instance_positive_points(
                        seed_mask,
                        max_points=self.seed_points_per_instance,
                    )
                    if component_points.shape[0] == 0:
                        continue
                    point_idx = point_group_idx
                    point_group_idx += int(component_points.shape[0])
                    candidate_fg_iou_masks_by_point[point_idx] = seed_mask
                    all_points.append(component_points.astype(np.float32))
                    component_points_aug = _scale_points(component_points, prompt_mask.shape, aug_hw)
                    raw_candidates.extend(
                        self._run_sam_multi_positive(
                            image_rgb_aug,
                            component_points_aug,
                            point_idx=point_idx,
                        )
                    )

                if not raw_candidates or not all_points:
                    self._record_failure(sample_name, epoch)
                    continue
                points_xy = np.concatenate(all_points, axis=0).astype(np.float32)
                fallback_mask = prompt_mask.copy()
                self._save_affinity_split_result(
                    image_rgb,
                    prompt_mask,
                    components,
                    points_xy,
                    sample_name,
                    epoch,
                )
            elif fg_area_ratio < self.small_fg_box_thresh:
                box_xyxy = _mask_to_box(prompt_mask)
                if box_xyxy is None:
                    self._record_failure(sample_name, epoch)
                    continue
                box_xyxy_aug = _scale_box(box_xyxy, prompt_mask.shape, aug_hw)
                box_raw_candidates = self._run_sam_box(image_rgb_aug, box_xyxy_aug)
                box_candidates_by_point = self._prepare_candidates_by_point(box_raw_candidates, heat_iou_ref_mask, full_background_mask)
                valid_box_candidates_by_point = self._get_valid_candidates_by_point(
                    box_candidates_by_point,
                    heat_iou_thresh=heat_iou_thresh,
                    bg_iou_thresh=bg_iou_thresh,
                )
                kept_box_candidates = self._select_best_candidates(valid_box_candidates_by_point)
                if kept_box_candidates:
                    box_mask = self._merge_candidate_masks(kept_box_candidates, prompt_mask)
                elif box_raw_candidates:
                    best_box_candidate = max(box_raw_candidates, key=lambda item: float(item.score))
                    if best_box_candidate.mask_orig is None:
                        self._prepare_candidate_outputs(best_box_candidate, prompt_mask.shape)
                    box_mask = _ensure_binary_mask(best_box_candidate.mask_orig).copy()
                else:
                    box_mask = prompt_mask.copy()

                points_xy = _select_box_refined_points(box_mask, prompt_mask.shape)
                if points_xy.shape[0] == 0:
                    self._record_failure(sample_name, epoch)
                    continue
                points_xy_aug = _scale_points(points_xy, prompt_mask.shape, aug_hw)
                raw_candidates = self._run_sam_single(image_rgb_aug, points_xy_aug)
                candidate_heat_iou_ref_mask = np.logical_or(
                    _ensure_binary_mask(prompt_mask).astype(bool),
                    _ensure_binary_mask(box_mask).astype(bool),
                ).astype(np.uint8)
                candidate_metric_heat_iou_mask = candidate_heat_iou_ref_mask
                fallback_mask = box_mask
                point_candidate_prefixes = ["point_candidates", "box_candidate"]
            else:
                points_xy = self._select_points(prompt_mask)
                if points_xy.shape[0] == 0:
                    self._record_failure(sample_name, epoch)
                    continue
                points_xy_aug = _scale_points(points_xy, prompt_mask.shape, aug_hw)
                raw_candidates = self._run_sam_single(image_rgb_aug, points_xy_aug)
            candidates_by_point = self._prepare_candidates_by_point(
                raw_candidates,
                candidate_heat_iou_ref_mask,
                full_background_mask,
                fg_iou_masks_by_point=candidate_fg_iou_masks_by_point,
                metric_heat_iou_mask=candidate_metric_heat_iou_mask,
            )
            valid_candidates_by_point = self._get_valid_candidates_by_point(
                candidates_by_point,
                heat_iou_thresh=heat_iou_thresh,
                bg_iou_thresh=bg_iou_thresh,
            )
            kept_candidates = self._select_best_candidates(valid_candidates_by_point)
            if kept_candidates:
                final_mask = self._merge_candidate_masks(kept_candidates, fallback_mask)
            else:
                self._record_failure(sample_name, epoch)
                final_mask = fallback_mask.copy()

            rule_a_candidates = self._select_rule_a_candidates(valid_candidates_by_point)
            rule_b_candidates = self._select_rule_b_candidates(valid_candidates_by_point)
            rule_ab_candidates = self._select_rule_ab_candidates(valid_candidates_by_point)

            rule_a_mask = self._merge_candidate_masks(rule_a_candidates, final_mask)
            rule_b_mask = self._merge_candidate_masks(rule_b_candidates, final_mask)
            rule_ab_mask = self._merge_candidate_masks(rule_ab_candidates, final_mask)
            final_prob = self._merge_candidate_prob_maps(kept_candidates, fallback_mask)
            rule_a_prob = self._merge_candidate_prob_maps(rule_a_candidates, final_mask)
            rule_b_prob = self._merge_candidate_prob_maps(rule_b_candidates, final_mask)
            rule_ab_prob = self._merge_candidate_prob_maps(rule_ab_candidates, final_mask)

            self._save_points_heatmap_overlay(image_rgb, prob_np, points_xy, box_xyxy, sample_name, epoch)
            self._save_points_seg_overlay(
                image_rgb,
                final_mask,
                points_xy,
                box_xyxy,
                sample_name,
                epoch,
            )
            self._save_per_point_candidates(
                image_rgb,
                box_raw_candidates,
                np.zeros((0, 2), dtype=np.float32),
                box_xyxy,
                heat_iou_ref_mask,
                full_background_mask,
                uncertain_area_ratio,
                sample_name,
                epoch,
                output_prefixes=["box_candidate"],
            )
            self._save_per_point_candidates(
                image_rgb,
                raw_candidates,
                points_xy,
                box_xyxy,
                candidate_heat_iou_ref_mask,
                full_background_mask,
                uncertain_area_ratio,
                sample_name,
                epoch,
                output_prefixes=point_candidate_prefixes,
                fg_iou_masks_by_point=candidate_fg_iou_masks_by_point,
                metric_heat_iou_mask=candidate_metric_heat_iou_mask,
            )
            self._save_pseudo_label_grayscale(
                final_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
            )
            self._save_pseudo_label_grayscale(
                rule_a_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
                prefix="pseudo_labels_rule_a",
            )
            self._save_pseudo_label_grayscale(
                rule_b_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
                prefix="pseudo_labels_rule_b",
            )
            self._save_pseudo_label_grayscale(
                rule_ab_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
                prefix="pseudo_labels_rule_ab",
            )
            self._save_pseudo_label_binary(
                image_rgb,
                final_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
                prefix="pseudo_labels_binary",
            )
            self._save_pseudo_label_binary(
                image_rgb,
                rule_a_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
                prefix="pseudo_labels_rule_a_binary",
            )
            self._save_pseudo_label_binary(
                image_rgb,
                rule_b_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
                prefix="pseudo_labels_rule_b_binary",
            )
            self._save_pseudo_label_binary(
                image_rgb,
                rule_ab_prob,
                sample_name,
                sample_meta,
                idx,
                epoch,
                prefix="pseudo_labels_rule_ab_binary",
            )
            self._save_final_candidates(
                image_rgb,
                kept_candidates,
                points_xy,
                box_xyxy,
                uncertain_area_ratio,
                sample_name,
                epoch,
            )
def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def _list_image_files(path: str) -> List[str]:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Not a directory: {path}")
    return [
        os.path.join(path, name)
        for name in sorted(os.listdir(path))
        if _is_image_file(name) and os.path.isfile(os.path.join(path, name))
    ]


def _build_stem_index(path: str) -> Dict[str, str]:
    return {
        os.path.splitext(os.path.basename(file_path))[0]: file_path
        for file_path in _list_image_files(path)
    }


def _save_result(
    result: SAMResult,
    image_rgb: np.ndarray,
    region_mask: np.ndarray,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    prompt_overlay = SAMHelper.draw_points(image_rgb, result.points_xy)
    _write_rgb(os.path.join(out_dir, "prompt_overlay.png"), prompt_overlay)

    region_mask_u8 = (_ensure_binary_mask(region_mask) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(out_dir, "region_mask.png"), region_mask_u8)

    if result.mask is not None:
        sam_mask_u8 = (_ensure_binary_mask(result.mask) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, "sam_mask.png"), sam_mask_u8)
        sam_overlay = SAMHelper.draw_mask_overlay(image_rgb, result.mask)
        sam_overlay = SAMHelper.draw_points(sam_overlay, result.points_xy)
        _write_rgb(os.path.join(out_dir, "sam_overlay.png"), sam_overlay)

    meta = {
        "success": bool(result.success),
        "points_xy": result.points_xy.tolist(),
        "num_candidates": len(result.candidates),
        "candidate_scores": [float(candidate.score) for candidate in result.candidates],
        "candidate_point_indices": [int(candidate.point_idx) for candidate in result.candidates],
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SAM from a red overlay region or binary mask.")
    parser.add_argument("--image", required=True, help="Path to the original RGB image.")
    parser.add_argument("--overlay", default=None, help="Path to the red overlay image.")
    parser.add_argument("--mask", default=None, help="Path to the binary mask image.")
    parser.add_argument("--checkpoint", required=True, help="Path to the SAM checkpoint.")
    parser.add_argument("--model-type", default="vit_h", help="SAM model type.")
    parser.add_argument("--device", default="cuda", help="Device for SAM, e.g. cuda or cpu.")
    parser.add_argument("--out-dir", required=True, help="Directory to save outputs.")
    parser.add_argument("--multimask-output", action="store_true", help="Enable SAM multimask output.")
    parser.add_argument("--max-masks", type=int, default=3, help="Maximum masks kept after merging.")
    args = parser.parse_args()

    if args.overlay is None and args.mask is None:
        parser.error("one of --overlay or --mask is required")
    return args


def main():
    args = parse_args()

    helper = SAMHelper(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        multimask_output=args.multimask_output,
        max_masks=args.max_masks,
    )

    if os.path.isdir(args.image):
        region_source = args.mask if args.mask is not None else args.overlay
        if region_source is None or not os.path.isdir(region_source):
            raise ValueError("when --image is a directory, --mask or --overlay must also be a directory")

        image_index = _build_stem_index(args.image)
        region_index = _build_stem_index(region_source)
        common_stems = sorted(set(image_index.keys()) & set(region_index.keys()))
        if not common_stems:
            raise FileNotFoundError("No matching image/mask stems found between the two directories")

        os.makedirs(args.out_dir, exist_ok=True)
        for stem in common_stems:
            image_rgb = _read_rgb_image(image_index[stem])
            if args.mask is not None:
                region_mask = _read_mask(region_index[stem])
            else:
                overlay_rgb = _read_rgb_image(region_index[stem])
                region_mask = extract_red_mask_from_overlay(overlay_rgb)

            result = helper.predict_from_red_region(image_rgb, region_mask)
            sample_out_dir = os.path.join(args.out_dir, stem)
            _save_result(result, image_rgb, region_mask, sample_out_dir)
            print(f"{stem}: success={result.success}, points={result.points_xy.tolist()}")

        print(f"saved_to={args.out_dir}")
        return

    image_rgb = _read_rgb_image(args.image)
    if args.mask is not None:
        region_mask = _read_mask(args.mask)
    else:
        overlay_rgb = _read_rgb_image(args.overlay)
        region_mask = extract_red_mask_from_overlay(overlay_rgb)

    result = helper.predict_from_red_region(image_rgb, region_mask)
    _save_result(result, image_rgb, region_mask, args.out_dir)

    print(f"success={result.success}")
    print(f"points={result.points_xy.tolist()}")
    print(f"saved_to={args.out_dir}")


if __name__ == "__main__":
    main()
