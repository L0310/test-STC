from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .image_ops import _connected_components, _ensure_binary_mask, _largest_component

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
    if dst_w > 0:
        scaled[:, 0] = np.clip(scaled[:, 0], 0.0, float(dst_w - 1))
    if dst_h > 0:
        scaled[:, 1] = np.clip(scaled[:, 1], 0.0, float(dst_h - 1))
    return scaled

def _merge_prompt_points(points_by_key: Optional[Dict[int, np.ndarray]]) -> np.ndarray:
    if not points_by_key:
        return np.zeros((0, 2), dtype=np.float32)
    merged: List[Tuple[float, float]] = []
    for key in sorted(points_by_key.keys()):
        points = np.asarray(points_by_key[key], dtype=np.float32).reshape(-1, 2)
        for x, y in points:
            point = (float(x), float(y))
            if point not in merged:
                merged.append(point)
    if not merged:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(merged, dtype=np.float32)

def _scale_box(box_xyxy: np.ndarray, src_hw: Tuple[int, int], dst_hw: Tuple[int, int]) -> np.ndarray:
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    scaled = box_xyxy.astype(np.float32).copy()
    if src_w > 0 and dst_w != src_w:
        scaled[[0, 2]] *= float(dst_w) / float(src_w)
    if src_h > 0 and dst_h != src_h:
        scaled[[1, 3]] *= float(dst_h) / float(src_h)
    return scaled
