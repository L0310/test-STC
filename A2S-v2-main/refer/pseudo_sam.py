import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PARENT_ROOT = os.path.dirname(PROJECT_ROOT)
SEGMENT_ANYTHING_ROOT = os.path.join(PARENT_ROOT, "segment-anything-main")
if os.path.isdir(SEGMENT_ANYTHING_ROOT) and SEGMENT_ANYTHING_ROOT not in sys.path:
    sys.path.insert(0, SEGMENT_ANYTHING_ROOT)

from segment_anything import SamPredictor, sam_model_registry


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
DEPTH_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_SEED_PALETTE = [
    (255, 99, 71),
    (255, 215, 0),
    (0, 191, 255),
    (50, 205, 50),
    (255, 105, 180),
    (138, 43, 226),
    (255, 140, 0),
    (64, 224, 208),
]


@dataclass
class SAMMaskCandidate:
    mask: np.ndarray
    score: float
    metrics: Tuple[float, float, float]
    better_count: int
    improvement_sum: float
    prob: Optional[np.ndarray] = None
    seed_recall: float = float("nan")
    seed_iou: float = float("nan")
    seed_precision: float = float("nan")
    fg_iou: float = float("nan")
    bg_iou: float = float("nan")
    heat_iou: float = float("nan")


@dataclass
class PseudoUpdateSummary:
    prompt_mode: str
    evaluated: int = 0
    replaced: int = 0
    ua_sum: float = 0.0
    ur_sum: float = 0.0
    ud_sum: float = 0.0

    def add_replacement(self, metrics: Tuple[float, float, float]) -> None:
        self.replaced += 1
        self.ua_sum += float(metrics[0])
        self.ur_sum += float(metrics[1])
        self.ud_sum += float(metrics[2])

    def merge(self, other: "PseudoUpdateSummary") -> None:
        self.evaluated += int(other.evaluated)
        self.replaced += int(other.replaced)
        self.ua_sum += float(other.ua_sum)
        self.ur_sum += float(other.ur_sum)
        self.ud_sum += float(other.ud_sum)

    def mean_metrics(self) -> Tuple[float, float, float]:
        if self.replaced <= 0:
            return float("nan"), float("nan"), float("nan")
        denom = float(self.replaced)
        return self.ua_sum / denom, self.ur_sum / denom, self.ud_sum / denom


def _ensure_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    return image


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def _tensor_to_uint8_rgb(image_3chw: torch.Tensor) -> np.ndarray:
    image = image_3chw.detach().cpu().numpy().transpose(1, 2, 0)
    image = ((image * STD) + MEAN) * 255.0
    return _ensure_uint8_rgb(image)


def _load_uint8_rgb(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _load_prob_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {path}")
    mask = mask.astype(np.float32)
    return (mask - mask.min()) / (mask.max() - mask.min() + 1e-5)


def _normalize_gray_map(gray_map: np.ndarray) -> np.ndarray:
    gray_map = np.asarray(gray_map, dtype=np.float32)
    if gray_map.ndim == 3:
        gray_map = gray_map[..., 0]
    min_value = float(gray_map.min()) if gray_map.size > 0 else 0.0
    max_value = float(gray_map.max()) if gray_map.size > 0 else 0.0
    if max_value - min_value <= 1e-6:
        return np.zeros_like(gray_map, dtype=np.float32)
    return ((gray_map - min_value) / (max_value - min_value)).astype(np.float32)


def _depth_gradient_map(depth_map: np.ndarray) -> np.ndarray:
    depth_map = _normalize_gray_map(depth_map)
    if depth_map.size == 0:
        return depth_map.astype(np.float32)
    depth_blur = cv2.GaussianBlur(depth_map, (5, 5), 0)
    grad_x = cv2.Sobel(depth_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    return _normalize_gray_map(grad_mag)


def _build_stem_path_index(root: Optional[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not root or not os.path.isdir(root):
        return index

    for current_root, _, file_names in os.walk(root):
        for file_name in file_names:
            stem, ext = os.path.splitext(file_name)
            if ext.lower() not in DEPTH_EXTENSIONS:
                continue
            full_path = os.path.join(current_root, file_name)
            if stem not in index:
                index[stem] = full_path
    return index


def _default_affinity_dino_weight() -> str:
    candidate = Path(PARENT_ROOT) / "PretrainModel" / "dinov2_vitl14_pretrain.pth"
    return str(candidate) if candidate.exists() else ""


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


def _select_instance_positive_point(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    mask = _ensure_binary_mask(mask)
    if int(mask.sum()) <= 0:
        return None

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    best_value = float(distance.max()) if distance.size > 0 else 0.0
    if best_value <= 0.0:
        return _center_point(mask)

    ys, xs = np.where(distance >= (best_value - 1e-6))
    if len(xs) <= 0:
        return _center_point(mask)

    center_x, center_y = _center_point(mask)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    target = np.array([center_x, center_y], dtype=np.float32)
    idx = int(np.argmin(np.sum((coords - target) ** 2, axis=1)))
    return float(coords[idx, 0]), float(coords[idx, 1])


def _select_instance_positive_points(mask: np.ndarray, max_points: int = 3) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    if int(mask.sum()) <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(_select_component_points(mask, max(1, int(max_points))), dtype=np.float32)


def _test_style_prediction(prob_map: np.ndarray, out_shape: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(np.asarray(prob_map, dtype=np.float32), out_shape[::-1])
    return np.clip(np.round(resized * 255.0) / 255.0, 0.0, 1.0).astype(np.float32)


def _flip_horizontal(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(image[:, ::-1])


def _flip_points_horizontal(points_xy: np.ndarray, width: int) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy.astype(np.float32)
    flipped = points_xy.astype(np.float32).copy()
    flipped[:, 0] = (width - 1) - flipped[:, 0]
    return flipped


def _flip_box_horizontal(box_xyxy: Optional[np.ndarray], width: int) -> Optional[np.ndarray]:
    if box_xyxy is None:
        return None
    box_xyxy = box_xyxy.astype(np.float32).copy()
    x0, y0, x1, y1 = box_xyxy.tolist()
    return np.array([(width - 1) - x1, y0, (width - 1) - x0, y1], dtype=np.float32)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_idx).astype(np.uint8)


def _filter_components(mask: np.ndarray, min_area: int = 1) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask if int(mask.sum()) >= int(min_area) else np.zeros_like(mask, dtype=np.uint8)

    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < int(min_area):
            continue
        filtered[labels == label_idx] = 1
    return filtered


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    mask = _ensure_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return []

    components = []
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        component = (labels == label_idx).astype(np.uint8)
        components.append((area, component))

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


def _center_point(mask: np.ndarray) -> Tuple[float, float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0, 0.0
    return _nearest_mask_point(mask, float(xs.mean()), float(ys.mean()))


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


def _greedy_fill_points(mask: np.ndarray, points: List[Tuple[float, float]], max_points: int) -> List[Tuple[float, float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return points

    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    selected = [np.array(point, dtype=np.float32) for point in points]

    while len(selected) < max_points:
        if not selected:
            selected.append(coords[len(coords) // 2])
            continue

        best_idx = 0
        best_dist = -1.0
        for idx, coord in enumerate(coords):
            min_dist = min(float(np.sum((coord - pt) ** 2)) for pt in selected)
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = idx

        candidate = coords[best_idx]
        if any(np.allclose(candidate, pt) for pt in selected):
            break
        selected.append(candidate)

    return [(float(pt[0]), float(pt[1])) for pt in selected[:max_points]]


def _select_component_points(component: np.ndarray, max_points: int) -> List[Tuple[float, float]]:
    component = _ensure_binary_mask(component)
    if component.sum() == 0:
        return []

    points = _component_region_points(component)
    if not points:
        points = [_center_point(component)]
    points = _greedy_fill_points(component, points, max_points)

    unique_points: List[Tuple[float, float]] = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
        if len(unique_points) >= max_points:
            break
    return unique_points


def select_prompt_points(mask: np.ndarray, max_points: int = 5) -> np.ndarray:
    mask = _ensure_binary_mask(mask)
    if mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    components = _connected_components(mask)
    if not components:
        components = [mask]

    unique_points: List[Tuple[float, float]] = []
    for component in components:
        component_points = _select_component_points(component, max_points)
        for point in component_points:
            if point not in unique_points:
                unique_points.append(point)
    return np.array(unique_points, dtype=np.float32)


def _distance_from_foreground(prompt_mask: np.ndarray) -> np.ndarray:
    prompt_mask = _ensure_binary_mask(prompt_mask)
    background = (prompt_mask == 0).astype(np.uint8)
    if background.sum() == 0:
        return np.zeros_like(prompt_mask, dtype=np.float32)
    return cv2.distanceTransform(background, cv2.DIST_L2, 5)


def _select_negative_point_from_region(
    coarse_mask: np.ndarray,
    distance_map: np.ndarray,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> Optional[Tuple[float, float]]:
    if x_max <= x_min or y_max <= y_min:
        return None

    coarse_region = np.asarray(coarse_mask[y_min:y_max, x_min:x_max], dtype=np.float32)
    dist_region = np.asarray(distance_map[y_min:y_max, x_min:x_max], dtype=np.float32)
    if coarse_region.size == 0 or dist_region.size == 0:
        return None

    bg_mask = coarse_region <= 1e-6
    if not np.any(bg_mask):
        return None

    masked_dist = np.where(bg_mask, dist_region, -1.0)
    best_value = float(masked_dist.max())
    if best_value < 0:
        return None

    ys, xs = np.where(masked_dist == best_value)
    if len(xs) == 0:
        return None

    center_x = (x_min + x_max - 1) / 2.0
    center_y = (y_min + y_max - 1) / 2.0
    coords = np.stack([xs + x_min, ys + y_min], axis=1).astype(np.float32)
    target = np.array([center_x, center_y], dtype=np.float32)
    idx = int(np.argmin(np.sum((coords - target) ** 2, axis=1)))
    return float(coords[idx, 0]), float(coords[idx, 1])


def _quadrant_regions(
    x0_i: int,
    y0_i: int,
    x1_i: int,
    y1_i: int,
) -> List[Tuple[int, int, int, int]]:
    x_mid = int(np.floor((x0_i + x1_i) / 2.0))
    y_mid = int(np.floor((y0_i + y1_i) / 2.0))
    return [
        (x0_i, x_mid + 1, y0_i, y_mid + 1),
        (x_mid + 1, x1_i + 1, y0_i, y_mid + 1),
        (x0_i, x_mid + 1, y_mid + 1, y1_i + 1),
        (x_mid + 1, x1_i + 1, y_mid + 1, y1_i + 1),
    ]


def _expanded_quadrant_regions(
    width: int,
    height: int,
    x0_i: int,
    y0_i: int,
    x1_i: int,
    y1_i: int,
) -> List[Tuple[int, int, int, int]]:
    box_w = max(1, x1_i - x0_i + 1)
    box_h = max(1, y1_i - y0_i + 1)
    x_mid = int(np.floor((x0_i + x1_i) / 2.0))
    y_mid = int(np.floor((y0_i + y1_i) / 2.0))
    return [
        (max(0, x0_i - box_w), x_mid + 1, max(0, y0_i - box_h), y_mid + 1),
        (x_mid + 1, min(width, x1_i + 1 + box_w), max(0, y0_i - box_h), y_mid + 1),
        (max(0, x0_i - box_w), x_mid + 1, y_mid + 1, min(height, y1_i + 1 + box_h)),
        (x_mid + 1, min(width, x1_i + 1 + box_w), y_mid + 1, min(height, y1_i + 1 + box_h)),
    ]


def _select_component_negative_points(
    component: np.ndarray,
    prompt_mask: np.ndarray,
    coarse_mask: np.ndarray,
    num_negative_points: int = 4,
) -> List[Tuple[float, float]]:
    component = _ensure_binary_mask(component)
    prompt_mask = _ensure_binary_mask(prompt_mask)
    coarse_mask = np.asarray(coarse_mask, dtype=np.float32)
    if component.sum() == 0:
        return []

    box = mask_to_box(component)
    if box is None:
        return []

    height, width = prompt_mask.shape
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float32).tolist()
    x0_i = max(0, min(width - 1, int(np.floor(x0))))
    y0_i = max(0, min(height - 1, int(np.floor(y0))))
    x1_i = max(0, min(width - 1, int(np.ceil(x1))))
    y1_i = max(0, min(height - 1, int(np.ceil(y1))))

    distance_map = _distance_from_foreground(prompt_mask)
    inner_regions = _quadrant_regions(x0_i, y0_i, x1_i, y1_i)
    outer_regions = _expanded_quadrant_regions(width, height, x0_i, y0_i, x1_i, y1_i)

    negative_points: List[Tuple[float, float]] = []
    for inner_region, outer_region in zip(
        inner_regions[:num_negative_points],
        outer_regions[:num_negative_points],
    ):
        point = _select_negative_point_from_region(coarse_mask, distance_map, *inner_region)
        if point is None:
            point = _select_negative_point_from_region(coarse_mask, distance_map, *outer_region)
        if point is None:
            continue
        if point not in negative_points:
            negative_points.append(point)
    return negative_points


def select_prompt_points_and_labels(
    prompt_mask: np.ndarray,
    coarse_mask: np.ndarray,
    max_positive_points: int = 5,
    max_negative_points: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    prompt_mask = _ensure_binary_mask(prompt_mask)
    if prompt_mask.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    components = _connected_components(prompt_mask)
    if not components:
        components = [prompt_mask]

    points_xy: List[Tuple[float, float]] = []
    point_labels: List[int] = []
    seen = set()

    for component in components:
        for point in _select_component_points(component, max_positive_points):
            key = (round(point[0], 4), round(point[1], 4), 1)
            if key in seen:
                continue
            seen.add(key)
            points_xy.append(point)
            point_labels.append(1)

        for point in _select_component_negative_points(
            component,
            prompt_mask=prompt_mask,
            coarse_mask=coarse_mask,
            num_negative_points=max_negative_points,
        ):
            key = (round(point[0], 4), round(point[1], 4), 0)
            if key in seen:
                continue
            seen.add(key)
            points_xy.append(point)
            point_labels.append(0)

    if not points_xy:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return np.array(points_xy, dtype=np.float32), np.array(point_labels, dtype=np.int32)


def mask_to_box(mask: np.ndarray) -> Optional[np.ndarray]:
    mask = _ensure_binary_mask(mask)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def logits_to_mask_prompt(logits_2d: np.ndarray, out_size: int = 256) -> np.ndarray:
    logits_2d = np.asarray(logits_2d, dtype=np.float32)
    resized = cv2.resize(logits_2d, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return resized[None, :, :]


def prob_to_binary_mask(
    prob_map: np.ndarray,
    thresh: float = 0.5,
    min_area: int = 16,
    max_area_ratio: float = 0.5,
) -> np.ndarray:
    prob_map = np.asarray(prob_map, dtype=np.float32)
    _ = max_area_ratio
    binary_mask = (prob_map >= thresh).astype(np.uint8)
    return _filter_components(binary_mask, min_area=min_area)


def binary_entropy(prob_map: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    prob_map = np.clip(np.asarray(prob_map, dtype=np.float32), eps, 1.0 - eps)
    entropy = -(prob_map * np.log(prob_map) + (1.0 - prob_map) * np.log(1.0 - prob_map))
    return entropy / np.log(2.0)


def sam_output_to_prob(mask_output: np.ndarray) -> np.ndarray:
    mask_output = np.asarray(mask_output, dtype=np.float32)
    if mask_output.size == 0:
        return mask_output
    if float(mask_output.min()) >= 0.0 and float(mask_output.max()) <= 1.0:
        return np.clip(mask_output, 0.0, 1.0)
    mask_output = np.clip(mask_output, -32.0, 32.0)
    return 1.0 / (1.0 + np.exp(-mask_output))


def compute_uncertainty_metrics(
    pseudo_prob: np.ndarray,
    student_prob: np.ndarray,
    entropy_thresh: float,
) -> Tuple[float, float, float]:
    pseudo_prob = np.clip(np.asarray(pseudo_prob, dtype=np.float32), 0.0, 1.0)
    student_prob = np.clip(np.asarray(student_prob, dtype=np.float32), 0.0, 1.0)

    entropy_map = binary_entropy(pseudo_prob)
    high_uncertain = entropy_map > entropy_thresh
    low_uncertain = entropy_map <= entropy_thresh

    residual = np.abs(pseudo_prob - student_prob)
    residual_entropy = binary_entropy(residual)

    ua = float(high_uncertain.mean())
    ur = float(high_uncertain.sum() / (float(low_uncertain.sum()) + 1e-10))
    ud = float(residual_entropy.mean())
    return ua, ur, ud


def _normalize_rel_path(gt_path: str) -> str:
    norm_path = os.path.normpath(gt_path)
    parts = norm_path.split(os.sep)
    if "pseudo" in parts:
        rel_parts = parts[parts.index("pseudo") + 1 :]
        if rel_parts:
            return os.path.join(*rel_parts)
    return os.path.basename(norm_path)


def _mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
    color_rgb: Tuple[float, float, float] = (1.0, 0.0, 1.0),
) -> np.ndarray:
    image = _ensure_uint8_rgb(image_rgb).astype(np.float32) / 255.0
    mask = _ensure_binary_mask(mask).astype(bool)
    color = np.zeros_like(image, dtype=np.float32)
    color[mask] = np.array(color_rgb, dtype=np.float32)
    overlay = (1.0 - alpha) * image + alpha * color
    return np.clip(overlay * 255.0, 0.0, 255.0).astype(np.uint8)


def _draw_points_and_box(
    image_rgb: np.ndarray,
    points_xy: np.ndarray,
    box_xyxy: Optional[np.ndarray],
    point_labels: Optional[np.ndarray] = None,
) -> np.ndarray:
    canvas = _ensure_uint8_rgb(image_rgb).copy()
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.size > 0:
        if point_labels is None:
            point_labels = np.ones((points_xy.shape[0],), dtype=np.int32)
        point_labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
        for idx, (x, y) in enumerate(np.atleast_2d(points_xy)):
            label = int(point_labels[idx]) if idx < len(point_labels) else 1
            color = (0, 255, 0) if label > 0 else (255, 0, 0)
            cv2.circle(canvas, (int(round(x)), int(round(y))), 5, color, -1)
    if box_xyxy is not None:
        boxes = np.asarray(box_xyxy, dtype=np.float32)
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        for box in boxes:
            x0, y0, x1, y1 = np.asarray(box, dtype=np.float32).tolist()
            cv2.rectangle(
                canvas,
                (int(round(x0)), int(round(y0))),
                (int(round(x1)), int(round(y1))),
                (255, 255, 0),
                2,
            )
    return canvas


def _prob_to_uint8(prob_map: np.ndarray) -> np.ndarray:
    prob_map = np.clip(np.asarray(prob_map, dtype=np.float32), 0.0, 1.0)
    return np.clip(np.round(prob_map * 255.0), 0.0, 255.0).astype(np.uint8)


def prob_to_coarse_mask(prob_map: np.ndarray) -> np.ndarray:
    prob_u8 = _prob_to_uint8(prob_map)
    return (prob_u8.astype(np.float32) / 255.0)


def coarse_mask_to_prompt_mask(
    coarse_mask: np.ndarray,
    min_area: int = 150,
    fg_thresh: float = 0.5,
    strict_greater: bool = False,
) -> np.ndarray:
    coarse_mask = np.asarray(coarse_mask, dtype=np.float32)
    if coarse_mask.ndim == 3:
        coarse_mask = coarse_mask[..., 0]
    if bool(strict_greater):
        binary_mask = (coarse_mask > float(fg_thresh)).astype(np.uint8)
    else:
        binary_mask = (coarse_mask >= float(fg_thresh)).astype(np.uint8)
    return _filter_components(binary_mask, min_area=min_area)


def _split_component_by_centers(
    component: np.ndarray,
    centers_xy: np.ndarray,
) -> List[np.ndarray]:
    component = _ensure_binary_mask(component)
    centers_xy = np.asarray(centers_xy, dtype=np.float32).reshape(-1, 2)
    if component.sum() == 0:
        return []
    if centers_xy.shape[0] <= 1:
        return [component]

    ys, xs = np.where(component > 0)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    dists = np.sum((coords[:, None, :] - centers_xy[None, :, :]) ** 2, axis=2)
    assign = np.argmin(dists, axis=1)

    split_components: List[np.ndarray] = []
    for center_idx in range(centers_xy.shape[0]):
        split = np.zeros_like(component, dtype=np.uint8)
        sel = assign == center_idx
        if np.any(sel):
            split[coords[sel, 1].astype(np.int32), coords[sel, 0].astype(np.int32)] = 1
        split_components.append(split)
    return split_components


def _split_component_by_depth_watershed(
    component: np.ndarray,
    coarse_mask: np.ndarray,
    depth_map: np.ndarray,
    core_mask: np.ndarray,
    core_min_area: int,
    depth_split_weight: float,
) -> List[np.ndarray]:
    component = _ensure_binary_mask(component)
    coarse_mask = np.asarray(coarse_mask, dtype=np.float32)
    depth_map = _normalize_gray_map(depth_map)
    core_mask = _ensure_binary_mask(core_mask)
    if component.sum() == 0 or depth_map.size == 0:
        return []

    box = mask_to_box(component)
    if box is None:
        return []

    height, width = component.shape
    x0, y0, x1, y1 = np.asarray(box, dtype=np.float32).tolist()
    x0_i = max(0, min(width - 1, int(np.floor(x0))))
    y0_i = max(0, min(height - 1, int(np.floor(y0))))
    x1_i = max(0, min(width - 1, int(np.ceil(x1))))
    y1_i = max(0, min(height - 1, int(np.ceil(y1))))

    component_crop = component[y0_i:y1_i + 1, x0_i:x1_i + 1]
    coarse_crop = coarse_mask[y0_i:y1_i + 1, x0_i:x1_i + 1]
    depth_crop = depth_map[y0_i:y1_i + 1, x0_i:x1_i + 1]
    core_crop = core_mask[y0_i:y1_i + 1, x0_i:x1_i + 1]
    if component_crop.size == 0 or depth_crop.size == 0:
        return []

    num_labels, core_labels = cv2.connectedComponents(core_crop, connectivity=8)
    if num_labels <= 2:
        return []

    markers = np.ones(component_crop.shape, dtype=np.int32)
    markers[component_crop > 0] = 0
    seed_label_values: List[int] = []
    for label_idx in range(1, num_labels):
        watershed_label = label_idx + 1
        markers[core_labels == label_idx] = watershed_label
        seed_label_values.append(watershed_label)
    if len(seed_label_values) <= 1:
        return []

    coarse_term = 1.0 - np.clip(coarse_crop, 0.0, 1.0)
    depth_term = _depth_gradient_map(depth_crop)
    topo = (1.0 - float(depth_split_weight)) * coarse_term + float(depth_split_weight) * depth_term
    topo = _normalize_gray_map(topo)
    topo_u8 = np.clip(np.round(topo * 255.0), 0.0, 255.0).astype(np.uint8)
    watershed_image = cv2.cvtColor(topo_u8, cv2.COLOR_GRAY2BGR)

    markers_ws = markers.copy()
    try:
        cv2.watershed(watershed_image, markers_ws)
    except cv2.error:
        return []

    seed_crops: List[np.ndarray] = []
    for watershed_label in seed_label_values:
        seed_crop = ((markers_ws == watershed_label) & (component_crop > 0)).astype(np.uint8)
        seed_crop = _filter_components(seed_crop, min_area=core_min_area)
        if int(seed_crop.sum()) <= 0:
            continue
        seed_crops.append(seed_crop)
    if len(seed_crops) <= 1:
        return []

    union_crop = np.zeros_like(component_crop, dtype=np.uint8)
    for seed_crop in seed_crops:
        union_crop = np.maximum(union_crop, seed_crop)
    leftover_crop = ((component_crop > 0) & (union_crop == 0)).astype(np.uint8)
    if int(leftover_crop.sum()) > 0:
        centers_xy = np.array([_center_point(seed_crop) for seed_crop in seed_crops], dtype=np.float32)
        leftover_splits = _split_component_by_centers(leftover_crop, centers_xy)
        for seed_idx, leftover_split in enumerate(leftover_splits):
            if seed_idx >= len(seed_crops):
                break
            seed_crops[seed_idx] = np.maximum(seed_crops[seed_idx], _ensure_binary_mask(leftover_split))

    seed_masks: List[np.ndarray] = []
    for seed_crop in seed_crops:
        seed_crop = _filter_components(seed_crop, min_area=core_min_area)
        if int(seed_crop.sum()) <= 0:
            continue
        seed_mask = np.zeros_like(component, dtype=np.uint8)
        seed_mask[y0_i:y1_i + 1, x0_i:x1_i + 1] = seed_crop
        seed_masks.append(seed_mask)
    return seed_masks


def _split_component_into_seed_masks(
    component: np.ndarray,
    coarse_mask: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    core_thresh: float = 0.8,
    core_min_area: int = 16,
    depth_split_weight: float = 0.35,
) -> List[np.ndarray]:
    component = _ensure_binary_mask(component)
    coarse_mask = np.asarray(coarse_mask, dtype=np.float32)
    if component.sum() == 0:
        return []

    core_mask = ((coarse_mask >= float(core_thresh)).astype(np.uint8) * component).astype(np.uint8)
    core_mask = _filter_components(core_mask, min_area=core_min_area)
    core_components = _connected_components(core_mask)
    if len(core_components) <= 1:
        return [component]

    if depth_map is not None:
        depth_seed_masks = _split_component_by_depth_watershed(
            component,
            coarse_mask=coarse_mask,
            depth_map=depth_map,
            core_mask=core_mask,
            core_min_area=core_min_area,
            depth_split_weight=depth_split_weight,
        )
        if len(depth_seed_masks) > 1:
            return depth_seed_masks

    centers_xy = np.array([_center_point(core) for core in core_components], dtype=np.float32)
    split_components = _split_component_by_centers(component, centers_xy)

    seed_masks: List[np.ndarray] = []
    for split_component, core_component in zip(split_components, core_components):
        split_component = _ensure_binary_mask(split_component)
        split_component[core_component > 0] = 1
        if int(split_component.sum()) > 0:
            seed_masks.append(split_component)

    return seed_masks if seed_masks else [component]


def split_prompt_mask_into_seeds(
    prompt_mask: np.ndarray,
    coarse_mask: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    core_thresh: float = 0.8,
    core_min_area: int = 16,
    depth_split_weight: float = 0.35,
) -> List[np.ndarray]:
    prompt_mask = _ensure_binary_mask(prompt_mask)
    coarse_mask = np.asarray(coarse_mask, dtype=np.float32)
    if prompt_mask.sum() == 0:
        return []

    components = _connected_components(prompt_mask)
    if not components:
        components = [prompt_mask]

    seed_components: List[np.ndarray] = []
    for component in components:
        seed_components.extend(
            _split_component_into_seed_masks(
                component,
                coarse_mask=coarse_mask,
                depth_map=depth_map,
                core_thresh=core_thresh,
                core_min_area=core_min_area,
                depth_split_weight=depth_split_weight,
            )
        )
    return [seed for seed in seed_components if int(np.asarray(seed).sum()) > 0]


def _uncertain_ratio(region: np.ndarray, eps: float = 1e-6) -> float:
    region = np.asarray(region, dtype=np.float32)
    if region.size == 0:
        return 0.0
    uncertain = (region > eps) & (region < 1.0 - eps)
    return float(uncertain.mean())


def _expand_box_prompt_from_coarse_mask(
    coarse_mask: np.ndarray,
    base_box_xyxy: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if base_box_xyxy is None:
        return None

    coarse_mask = np.asarray(coarse_mask, dtype=np.float32)
    if coarse_mask.ndim == 3:
        coarse_mask = coarse_mask[..., 0]
    height, width = coarse_mask.shape

    x0, y0, x1, y1 = np.asarray(base_box_xyxy, dtype=np.float32).tolist()
    x0_i = max(0, min(width - 1, int(np.floor(x0))))
    y0_i = max(0, min(height - 1, int(np.floor(y0))))
    x1_i = max(0, min(width - 1, int(np.ceil(x1))))
    y1_i = max(0, min(height - 1, int(np.ceil(y1))))

    box_w = max(1, x1_i - x0_i + 1)
    box_h = max(1, y1_i - y0_i + 1)

    left_start = max(0, x0_i - box_w)
    right_end = min(width, x1_i + 1 + box_w)
    top_start = max(0, y0_i - box_h)
    bottom_end = min(height, y1_i + 1 + box_h)

    left_region = coarse_mask[top_start:bottom_end, left_start:x0_i]
    right_region = coarse_mask[top_start:bottom_end, x1_i + 1:right_end]
    up_region = coarse_mask[top_start:y0_i, left_start:right_end]
    down_region = coarse_mask[y1_i + 1:bottom_end, left_start:right_end]

    left_width = max(0, x0_i - left_start)
    right_width = max(0, right_end - (x1_i + 1))
    up_height = max(0, y0_i - top_start)
    down_height = max(0, bottom_end - (y1_i + 1))

    expand_left = _uncertain_ratio(left_region) * float(left_width)
    expand_right = _uncertain_ratio(right_region) * float(right_width)
    expand_up = _uncertain_ratio(up_region) * float(up_height)
    expand_down = _uncertain_ratio(down_region) * float(down_height)

    expanded = np.array(
        [
            max(0.0, x0_i - expand_left),
            max(0.0, y0_i - expand_up),
            min(float(width - 1), x1_i + expand_right),
            min(float(height - 1), y1_i + expand_down),
        ],
        dtype=np.float32,
    )
    return expanded


def coarse_mask_to_binary_mask_prompt(
    coarse_mask: np.ndarray,
    out_size: int = 256,
    fg_value: float = 1.0,
    bg_value: float = -1.0,
) -> np.ndarray:
    coarse_mask = np.asarray(coarse_mask, dtype=np.float32)
    if coarse_mask.ndim == 3:
        coarse_mask = coarse_mask[..., 0]
    binary_mask = (coarse_mask >= 1.0 - 1e-6).astype(np.float32)
    prompt = np.where(binary_mask > 0, fg_value, bg_value).astype(np.float32)
    prompt = cv2.resize(prompt, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return prompt[None, :, :]


def _prob_heatmap_overlay(image_rgb: np.ndarray, prob_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    image_rgb = _ensure_uint8_rgb(image_rgb).astype(np.float32)
    heat_u8 = _prob_to_uint8(prob_map)
    heat_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    overlay = (1.0 - alpha) * image_rgb + alpha * heat_rgb
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def _binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = _ensure_binary_mask(mask_a).astype(bool)
    mask_b = _ensure_binary_mask(mask_b).astype(bool)
    union = float(np.logical_or(mask_a, mask_b).sum())
    if union <= 0.0:
        return 1.0
    inter = float(np.logical_and(mask_a, mask_b).sum())
    return inter / union


def _binary_bg_iou(mask_a: np.ndarray, background_mask: np.ndarray) -> float:
    mask_a = _ensure_binary_mask(mask_a).astype(bool)
    background_mask = _ensure_binary_mask(background_mask).astype(bool)
    union = float(np.logical_or(mask_a, background_mask).sum())
    if union <= 0.0:
        return 1.0
    inter = float(np.logical_and(mask_a, background_mask).sum())
    return inter / union


def _append_metric_to_rel_path(rel_path: str, metric_name: str, metric_value: float) -> str:
    base_name = os.path.basename(rel_path)
    stem, ext = os.path.splitext(base_name)
    metric_suffix = "__{}_{}".format(str(metric_name).strip().replace(" ", "_"), format(float(metric_value), ".4f"))
    new_name = "{}{}{}".format(stem, metric_suffix, ext)
    parent = os.path.dirname(rel_path)
    if not parent:
        return new_name
    return os.path.join(parent, new_name)


def _append_tag_to_rel_path(rel_path: str, tag: str) -> str:
    base_name = os.path.basename(rel_path)
    stem, ext = os.path.splitext(base_name)
    safe_tag = str(tag).strip().replace(" ", "_")
    new_name = "{}__{}{}".format(stem, safe_tag, ext)
    parent = os.path.dirname(rel_path)
    if not parent:
        return new_name
    return os.path.join(parent, new_name)


def _seed_masks_to_union(seed_masks: List[np.ndarray], out_shape: Tuple[int, int]) -> np.ndarray:
    union_mask = np.zeros(out_shape, dtype=np.uint8)
    for seed_mask in seed_masks:
        union_mask = np.maximum(union_mask, _ensure_binary_mask(seed_mask))
    return union_mask


def _seed_masks_to_label_rgb(seed_masks: List[np.ndarray], out_shape: Tuple[int, int]) -> np.ndarray:
    label_rgb = np.zeros((out_shape[0], out_shape[1], 3), dtype=np.uint8)
    for seed_idx, seed_mask in enumerate(seed_masks):
        color = np.array(_SEED_PALETTE[seed_idx % len(_SEED_PALETTE)], dtype=np.uint8)
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


class SAMPseudoLabelUpdater:
    def __init__(
        self,
        checkpoint: str,
        *,
        model_type: str = "vit_h",
        device: str = "cuda",
        max_points: int = 5,
        pred_thresh: float = 0.2,
        core_thresh: float = 0.8,
        entropy_thresh: float = 0.9,
        heat_iou_thresh: float = 0.85,
        depth_root: Optional[str] = None,
        depth_split_weight: float = 0.35,
        split_only: bool = False,
        multimask_output: bool = True,
        save_root: Optional[str] = None,
        prompt_max_area_ratio: float = 0.5,
        prompt_fg_thresh: float = 0.1,
        instance_prompt_mode: str = "affinity_points",
        seed_recall_thresh: float = 0.6,
        seed_fg_iou_thresh: float = 0.1,
        seed_heat_iou_thresh: float = 0.6,
        seed_score_thresh: float = 0.9,
        bg_prob_thresh: float = 0.0,
        seed_points_per_instance: int = 3,
        seed_bg_iou_thresh: float = 0.10,
        affinity_dino_weight: Optional[str] = None,
        affinity_dino_model: str = "dinov2_vitl14",
        affinity_dino_repo: Optional[str] = None,
        affinity_dino_device: Optional[str] = None,
        affinity_dino_max_side: int = 700,
        affinity_dino_pca_dim: int = 64,
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
    ):
        self.split_only = bool(split_only)
        self.predictor: Optional[SamPredictor] = None
        if not self.split_only:
            if not checkpoint or not os.path.exists(checkpoint):
                raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam = sam.to(device)
            self.predictor = SamPredictor(sam)
        self.max_points = int(max_points)
        self.pred_thresh = float(pred_thresh)
        self.core_thresh = float(core_thresh)
        self.entropy_thresh = float(entropy_thresh)
        self.heat_iou_thresh = float(np.clip(heat_iou_thresh, 0.0, 1.0))
        self.depth_root = depth_root
        self.depth_split_weight = float(np.clip(depth_split_weight, 0.0, 1.0))
        self.depth_index = _build_stem_path_index(depth_root)
        self.multimask_output = bool(multimask_output)
        self.save_root = save_root
        self.prompt_max_area_ratio = float(prompt_max_area_ratio)
        self.prompt_fg_thresh = float(np.clip(prompt_fg_thresh, 0.0, 1.0))
        self.instance_prompt_mode = str(instance_prompt_mode).strip() or "affinity_points"
        self.seed_recall_thresh = float(np.clip(seed_recall_thresh, 0.0, 1.0))
        self.seed_fg_iou_thresh = float(np.clip(seed_fg_iou_thresh, 0.0, 1.0))
        self.seed_heat_iou_thresh = float(np.clip(seed_heat_iou_thresh, 0.0, 1.0))
        self.seed_score_thresh = float(np.clip(seed_score_thresh, 0.0, 1.0))
        self.bg_prob_thresh = float(np.clip(bg_prob_thresh, 0.0, 1.0))
        self.seed_points_per_instance = int(max(1, seed_points_per_instance))
        self.seed_bg_iou_thresh = float(np.clip(seed_bg_iou_thresh, 0.0, 1.0))
        self.affinity_dino_weight = str(affinity_dino_weight or _default_affinity_dino_weight()).strip()
        self.affinity_dino_model = str(affinity_dino_model).strip() or "dinov2_vitl14"
        self.affinity_dino_repo = str(affinity_dino_repo or "").strip()
        self.affinity_dino_device = str(affinity_dino_device or device).strip() or str(device)
        self.affinity_dino_max_side = int(max(0, affinity_dino_max_side))
        self.affinity_dino_pca_dim = int(max(0, affinity_dino_pca_dim))
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
        self._affinity_dino_extractor = None
        self._affinity_dino_checked = False
        if self.depth_index:
            print(
                "Using depth-guided prompt splitting from {} ({} maps).".format(
                    os.path.abspath(str(depth_root)),
                    len(self.depth_index),
                )
            )
        elif depth_root:
            print(
                "Depth-guided prompt splitting disabled because no depth maps were found under {}.".format(
                    os.path.abspath(str(depth_root))
                )
            )
        if self.split_only:
            print("Running epoch-2 prompt splitting in visualization-only mode without SAM updates.")

    @staticmethod
    def should_update(epoch: int) -> bool:
        return int(epoch) == 2

    def prompt_mode(self, epoch: int) -> str:
        if self.should_update(epoch):
            if self.split_only:
                return "split_only"
            return self.instance_prompt_mode
        return "disabled"

    def _load_depth_map(
        self,
        image_path: Optional[str],
        out_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        if not image_path or not self.depth_index:
            return None

        stem = os.path.splitext(os.path.basename(image_path))[0]
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

    def _get_affinity_dino_extractor(self):
        if self._affinity_dino_checked:
            return self._affinity_dino_extractor
        self._affinity_dino_checked = True
        if not self.affinity_dino_weight or not os.path.exists(self.affinity_dino_weight):
            return None
        try:
            from dinov2_feature_viz import _OnTheFlyDINOExtractor, _default_dino_repo

            self._affinity_dino_extractor = _OnTheFlyDINOExtractor(
                weight_path=Path(self.affinity_dino_weight),
                model_name=self.affinity_dino_model,
                repo_path=self.affinity_dino_repo or _default_dino_repo(),
                device=self.affinity_dino_device,
                max_side=self.affinity_dino_max_side,
            )
            print(
                "Using optional DINO semantic affinity for pseudo-label refresh from {}.".format(
                    os.path.abspath(self.affinity_dino_weight)
                )
            )
        except Exception as exc:
            self._affinity_dino_extractor = None
            print("Warn: failed to initialize DINO semantic affinity extractor: {}. Falling back to depth-only affinity.".format(exc))
        return self._affinity_dino_extractor

    def _extract_affinity_appearance_map(
        self,
        image_rgb: np.ndarray,
        target_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        extractor = self._get_affinity_dino_extractor()
        if extractor is None:
            return None
        try:
            return extractor.extract(image_rgb, target_shape=target_shape)
        except Exception as exc:
            print("Warn: failed to extract DINO semantic affinity features: {}. Falling back to depth-only affinity.".format(exc))
            return None

    def _split_prompt_mask_into_affinity_instances(
        self,
        *,
        prompt_mask: np.ndarray,
        depth_map: Optional[np.ndarray],
        image_rgb: np.ndarray,
    ) -> List[np.ndarray]:
        prompt_mask = _ensure_binary_mask(prompt_mask)
        if int(prompt_mask.sum()) <= 0:
            return []
        try:
            from depth_affinity_spectral_demo import split_depth_instances_affinity_spectral
        except Exception as exc:
            print("Warn: failed to import depth_affinity_spectral_demo: {}. Falling back to legacy depth split.".format(exc))
            return []

        split_depth = (
            _normalize_gray_map(depth_map)
            if depth_map is not None
            else np.zeros(prompt_mask.shape, dtype=np.float32)
        )
        appearance_map = self._extract_affinity_appearance_map(image_rgb, target_shape=prompt_mask.shape)
        try:
            results = split_depth_instances_affinity_spectral(
                depth=split_depth,
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
                dino_pca_dim=self.affinity_dino_pca_dim,
                sigma_sem=self.affinity_sigma_sem,
                sigma_dep=self.affinity_sigma_dep,
                sigma_spatial=self.affinity_sigma_spatial,
                sigma_edge=self.affinity_sigma_edge,
                min_affinity=self.affinity_min_affinity,
                min_cluster_regions=self.affinity_min_cluster_regions,
                ncut_threshold=self.affinity_ncut_threshold,
                max_recursion_depth=self.affinity_max_recursion_depth,
            )
        except Exception as exc:
            print("Warn: depth affinity instance partition failed: {}. Falling back to legacy depth split.".format(exc))
            return []

        return _label_map_to_components(
            results.get("label_map", np.zeros_like(prompt_mask, dtype=np.int32)),
            min_area=max(16, int(self.affinity_min_instance_area // 2)),
        )

    def _predict_candidates(
        self,
        image_rgb: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        points_xy: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        mask_prompt: Optional[np.ndarray],
        mode: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        image_rgb = _ensure_uint8_rgb(image_rgb)
        if self.predictor is None:
            raise RuntimeError("SAM predictor is unavailable in split-only mode.")
        self.predictor.set_image(np.ascontiguousarray(image_rgb))

        kwargs = {
            "multimask_output": self.multimask_output,
            "return_logits": True,
        }
        _ = mask_prompt
        if points_xy is not None and np.asarray(points_xy).size > 0:
            kwargs["point_coords"] = np.asarray(points_xy, dtype=np.float32)
            kwargs["point_labels"] = np.asarray(point_labels, dtype=np.int32)
        if mode == "box" and box_xyxy is not None:
            kwargs["box"] = box_xyxy.astype(np.float32)

        masks, scores, _ = self.predictor.predict(**kwargs)
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)
        if masks.ndim >= 1 and masks.shape[0] > 3:
            masks = masks[:3]
            scores = scores[:3]
        return masks, scores

    def _select_candidate(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        current_prob: np.ndarray,
        student_prob: np.ndarray,
    ) -> Optional[SAMMaskCandidate]:
        if masks.size == 0:
            return None

        current_metrics = compute_uncertainty_metrics(current_prob, student_prob, self.entropy_thresh)
        best_candidate: Optional[SAMMaskCandidate] = None

        for idx in range(masks.shape[0]):
            candidate_prob = sam_output_to_prob(masks[idx])
            mask = (candidate_prob >= self.pred_thresh).astype(np.uint8)
            metrics = compute_uncertainty_metrics(candidate_prob, student_prob, self.entropy_thresh)
            improvements = [old - new for old, new in zip(current_metrics, metrics)]
            better_count = sum(delta > 1e-6 for delta in improvements)
            candidate = SAMMaskCandidate(
                mask=mask,
                score=float(scores[idx]),
                metrics=metrics,
                better_count=int(better_count),
                improvement_sum=float(sum(improvements)),
            )

            if best_candidate is None:
                best_candidate = candidate
                continue

            if candidate.better_count > best_candidate.better_count:
                best_candidate = candidate
                continue
            if candidate.better_count == best_candidate.better_count and candidate.improvement_sum > best_candidate.improvement_sum:
                best_candidate = candidate
                continue
            if (
                candidate.better_count == best_candidate.better_count
                and abs(candidate.improvement_sum - best_candidate.improvement_sum) <= 1e-6
                and candidate.score > best_candidate.score
            ):
                best_candidate = candidate

        return best_candidate

    def _select_seed_candidates_for_merge(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        seed_mask: np.ndarray,
        full_background_mask: np.ndarray,
        prompt_mask: np.ndarray,
        seed_recall_thresh: Optional[float] = None,
    ) -> List[SAMMaskCandidate]:
        if masks.size == 0:
            return []

        seed_mask = _ensure_binary_mask(seed_mask)
        full_background_mask = _ensure_binary_mask(full_background_mask)
        prompt_mask = _ensure_binary_mask(prompt_mask)
        seed_area = float(seed_mask.sum()) + 1e-10
        valid_candidates: List[SAMMaskCandidate] = []

        for idx in range(masks.shape[0]):
            candidate_prob = sam_output_to_prob(masks[idx])
            candidate_mask = (candidate_prob >= self.pred_thresh).astype(np.uint8)
            candidate_score = float(scores[idx])
            candidate_area = float(candidate_mask.sum()) + 1e-10
            inter = float((candidate_mask * seed_mask).sum())
            union = candidate_area + seed_area - inter
            recall = inter / seed_area
            precision = inter / candidate_area
            iou = inter / (union + 1e-10)
            fg_iou = _binary_iou(candidate_mask, seed_mask)
            bg_iou = _binary_bg_iou(candidate_mask, full_background_mask)
            heat_iou = _binary_iou(candidate_mask, prompt_mask)
            if float(candidate_score) <= float(self.seed_score_thresh):
                continue
            if float(fg_iou) <= float(self.seed_fg_iou_thresh):
                continue
            if float(bg_iou) > float(self.seed_bg_iou_thresh):
                continue
            if seed_recall_thresh is not None and float(recall) < float(seed_recall_thresh):
                continue

            candidate = SAMMaskCandidate(
                mask=candidate_mask,
                score=float(candidate_score),
                metrics=(float("nan"), float("nan"), float("nan")),
                better_count=0,
                improvement_sum=0.0,
                prob=candidate_prob,
                seed_recall=float(recall),
                seed_iou=float(iou),
                seed_precision=float(precision),
                fg_iou=float(fg_iou),
                bg_iou=float(bg_iou),
                heat_iou=float(heat_iou),
            )
            valid_candidates.append(candidate)

        if not valid_candidates:
            return []

        best_fg_candidate = max(
            valid_candidates,
            key=lambda candidate: (
                float(candidate.fg_iou),
                float(candidate.heat_iou),
                float(candidate.seed_recall),
                float(candidate.score),
            ),
        )
        best_heat_candidate = max(
            valid_candidates,
            key=lambda candidate: (
                float(candidate.heat_iou),
                float(candidate.fg_iou),
                float(candidate.seed_recall),
                float(candidate.score),
            ),
        )

        selected_candidates = [best_fg_candidate]
        if best_heat_candidate is not best_fg_candidate:
            selected_candidates.append(best_heat_candidate)
        return selected_candidates

    def _build_candidate_summary(
        self,
        candidate_prob: np.ndarray,
        score: float,
        current_prob: np.ndarray,
        student_prob: np.ndarray,
    ) -> SAMMaskCandidate:
        candidate_prob = np.clip(np.asarray(candidate_prob, dtype=np.float32), 0.0, 1.0)
        current_metrics = compute_uncertainty_metrics(current_prob, student_prob, self.entropy_thresh)
        metrics = compute_uncertainty_metrics(candidate_prob, student_prob, self.entropy_thresh)
        improvements = [old - new for old, new in zip(current_metrics, metrics)]
        better_count = sum(delta > 1e-6 for delta in improvements)
        return SAMMaskCandidate(
            mask=(candidate_prob >= self.pred_thresh).astype(np.uint8),
            score=float(score),
            metrics=metrics,
            better_count=int(better_count),
            improvement_sum=float(sum(improvements)),
            prob=candidate_prob,
        )

    def _save_epoch_result(
        self,
        *,
        epoch: int,
        gt_path: str,
        image_rgb: np.ndarray,
        student_prob: np.ndarray,
        points_xy: np.ndarray,
        point_labels: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        mask: np.ndarray,
        heat_iou: float,
        flip_value: bool,
    ) -> None:
        if not self.save_root:
            return

        rel_path = _normalize_rel_path(gt_path)
        overlay_rel_path = _append_metric_to_rel_path(rel_path, "heat_iou", heat_iou)
        overlay_path = os.path.join(
            self.save_root,
            "epoch_{:03d}".format(epoch),
            "points_sam_seg",
            overlay_rel_path,
        )
        low_iou_overlay_path = os.path.join(
            self.save_root,
            "epoch_{:03d}".format(epoch),
            "points_sam_seg_low_iou",
            overlay_rel_path,
        )
        pseudo_path = os.path.join(
            self.save_root,
            "epoch_{:03d}".format(epoch),
            "pseudo_labels",
            rel_path,
        )
        student_pred_path = os.path.join(
            self.save_root,
            "epoch_{:03d}".format(epoch),
            "student_pred",
            rel_path,
        )
        prompt_points_path = os.path.join(
            self.save_root,
            "epoch_{:03d}".format(epoch),
            "prompt_points",
            rel_path,
        )
        os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
        os.makedirs(os.path.dirname(low_iou_overlay_path), exist_ok=True)
        os.makedirs(os.path.dirname(pseudo_path), exist_ok=True)
        os.makedirs(os.path.dirname(student_pred_path), exist_ok=True)
        os.makedirs(os.path.dirname(prompt_points_path), exist_ok=True)

        save_image = image_rgb.copy()
        save_student_pred = _prob_heatmap_overlay(save_image, student_prob, alpha=0.45)
        save_points = points_xy.astype(np.float32).copy()
        save_labels = point_labels.astype(np.int32).copy()
        save_box = None if box_xyxy is None else box_xyxy.astype(np.float32).copy()
        save_mask = _ensure_binary_mask(mask)
        _ = flip_value

        save_student_pred = _draw_points_and_box(save_student_pred, save_points, save_box, save_labels)
        points_rgb = _draw_points_and_box(save_image, save_points, save_box, save_labels)
        overlay_rgb = _mask_overlay(save_image, save_mask)
        overlay_rgb = _draw_points_and_box(overlay_rgb, save_points, save_box, save_labels)
        pseudo_u8 = save_mask.astype(np.uint8) * 255

        if float(heat_iou) > float(self.seed_heat_iou_thresh):
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        if float(heat_iou) < self.heat_iou_thresh:
            cv2.imwrite(low_iou_overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(pseudo_path, pseudo_u8)
        cv2.imwrite(student_pred_path, cv2.cvtColor(save_student_pred, cv2.COLOR_RGB2BGR))
        cv2.imwrite(prompt_points_path, cv2.cvtColor(points_rgb, cv2.COLOR_RGB2BGR))

    def _save_seed_candidate_visualizations(
        self,
        *,
        epoch: int,
        gt_path: str,
        image_rgb: np.ndarray,
        points_xy: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box_xyxy: Optional[np.ndarray],
        seed_mask: np.ndarray,
        full_background_mask: np.ndarray,
        prompt_mask: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        component_idx: int,
    ) -> None:
        if not self.save_root or masks.size == 0:
            return

        rel_path = _normalize_rel_path(gt_path)
        epoch_root = os.path.join(self.save_root, "epoch_{:03d}".format(epoch))
        points_xy = np.asarray(points_xy if points_xy is not None else np.zeros((0, 2), dtype=np.float32), dtype=np.float32)
        point_labels = np.asarray(point_labels if point_labels is not None else np.zeros((0,), dtype=np.int32), dtype=np.int32)
        seed_mask = _ensure_binary_mask(seed_mask)
        full_background_mask = _ensure_binary_mask(full_background_mask)
        prompt_mask = _ensure_binary_mask(prompt_mask)

        for candidate_idx in range(int(masks.shape[0])):
            candidate_prob = sam_output_to_prob(masks[candidate_idx])
            candidate_mask = (candidate_prob >= self.pred_thresh).astype(np.uint8)
            fg_iou = _binary_iou(candidate_mask, seed_mask)
            bg_iou = _binary_bg_iou(candidate_mask, full_background_mask)
            heat_iou = _binary_iou(candidate_mask, prompt_mask)
            tagged_rel_path = _append_tag_to_rel_path(
                rel_path,
                "seed{:02d}_point{:02d}_cand{:02d}".format(
                    int(component_idx),
                    int(component_idx),
                    int(candidate_idx + 1),
                ),
            )
            overlay_rel_path = _append_metric_to_rel_path(tagged_rel_path, "fg_iou", fg_iou)
            overlay_rel_path = _append_metric_to_rel_path(overlay_rel_path, "bg_iou", bg_iou)
            overlay_path = os.path.join(epoch_root, "sam_seed_candidates", overlay_rel_path)
            low_iou_overlay_path = os.path.join(epoch_root, "sam_seed_candidates_low_iou", overlay_rel_path)
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            os.makedirs(os.path.dirname(low_iou_overlay_path), exist_ok=True)

            overlay_rgb = _mask_overlay(image_rgb, candidate_mask)
            overlay_rgb = _draw_points_and_box(overlay_rgb, points_xy, box_xyxy, point_labels)
            cv2.putText(
                overlay_rgb,
                "score={:.3f} fg_iou={:.4f} bg_iou={:.4f} heat_iou={:.4f}".format(
                    float(scores[candidate_idx]),
                    float(fg_iou),
                    float(bg_iou),
                    float(heat_iou),
                ),
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
            if float(fg_iou) < self.heat_iou_thresh:
                cv2.imwrite(low_iou_overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    def _save_split_result(
        self,
        *,
        epoch: int,
        gt_path: str,
        image_rgb: np.ndarray,
        student_prob: np.ndarray,
        prompt_mask: np.ndarray,
        seed_masks: List[np.ndarray],
        depth_map: Optional[np.ndarray],
    ) -> None:
        if not self.save_root:
            return

        rel_path = _normalize_rel_path(gt_path)
        epoch_root = os.path.join(self.save_root, "epoch_{:03d}".format(epoch))
        split_overlay_path = os.path.join(epoch_root, "depth_split_overlay", rel_path)
        split_label_path = os.path.join(epoch_root, "depth_split_labels", rel_path)
        split_union_path = os.path.join(epoch_root, "depth_split_union", rel_path)
        split_box_path = os.path.join(epoch_root, "depth_split_boxes", rel_path)
        prompt_mask_path = os.path.join(epoch_root, "depth_split_prompt_mask", rel_path)
        student_pred_path = os.path.join(epoch_root, "depth_split_student_pred", rel_path)
        depth_vis_path = os.path.join(epoch_root, "depth_split_depth", rel_path)

        os.makedirs(os.path.dirname(split_overlay_path), exist_ok=True)
        os.makedirs(os.path.dirname(split_label_path), exist_ok=True)
        os.makedirs(os.path.dirname(split_union_path), exist_ok=True)
        os.makedirs(os.path.dirname(split_box_path), exist_ok=True)
        os.makedirs(os.path.dirname(prompt_mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(student_pred_path), exist_ok=True)
        os.makedirs(os.path.dirname(depth_vis_path), exist_ok=True)

        seed_boxes = []
        for seed_mask in seed_masks:
            box_xyxy = mask_to_box(seed_mask)
            if box_xyxy is not None:
                seed_boxes.append(np.asarray(box_xyxy, dtype=np.float32))
        stacked_boxes = np.stack(seed_boxes, axis=0) if seed_boxes else None
        empty_points = np.zeros((0, 2), dtype=np.float32)
        empty_labels = np.zeros((0,), dtype=np.int32)

        split_overlay = _seed_masks_overlay(image_rgb, seed_masks, alpha=0.45)
        split_overlay = _draw_points_and_box(split_overlay, empty_points, stacked_boxes, empty_labels)
        split_boxes = _draw_points_and_box(image_rgb.copy(), empty_points, stacked_boxes, empty_labels)
        split_label_rgb = _seed_masks_to_label_rgb(seed_masks, image_rgb.shape[:2])
        split_union = _seed_masks_to_union(seed_masks, image_rgb.shape[:2]).astype(np.uint8) * 255
        prompt_mask_u8 = _ensure_binary_mask(prompt_mask).astype(np.uint8) * 255
        student_pred_overlay = _prob_heatmap_overlay(image_rgb, student_prob, alpha=0.45)
        student_pred_overlay = _draw_points_and_box(student_pred_overlay, empty_points, stacked_boxes, empty_labels)

        cv2.imwrite(split_overlay_path, cv2.cvtColor(split_overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(split_label_path, cv2.cvtColor(split_label_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(split_union_path, split_union)
        cv2.imwrite(split_box_path, cv2.cvtColor(split_boxes, cv2.COLOR_RGB2BGR))
        cv2.imwrite(prompt_mask_path, prompt_mask_u8)
        cv2.imwrite(student_pred_path, cv2.cvtColor(student_pred_overlay, cv2.COLOR_RGB2BGR))
        if depth_map is not None:
            depth_u8 = np.clip(np.round(_normalize_gray_map(depth_map) * 255.0), 0.0, 255.0).astype(np.uint8)
            cv2.imwrite(depth_vis_path, depth_u8)

    def update_batch(
        self,
        images: torch.Tensor,
        pred_logits: torch.Tensor,
        current_gts: torch.Tensor,
        gt_paths: List[str],
        image_paths: Optional[List[str]],
        flips: List[bool],
        epoch: int,
    ) -> PseudoUpdateSummary:
        mode = self.prompt_mode(int(epoch))
        pred_probs = torch.sigmoid(pred_logits.detach()).cpu().numpy()
        current_gts_np = current_gts.detach().cpu().numpy()
        summary = PseudoUpdateSummary(prompt_mode=mode)
        if mode == "disabled":
            return summary

        for idx, gt_path in enumerate(gt_paths):
            summary.evaluated += 1
            student_prob = pred_probs[idx, 0]
            current_prob = np.clip(current_gts_np[idx, 0].astype(np.float32), 0.0, 1.0)
            image_rgb = _tensor_to_uint8_rgb(images[idx])
            image_path = None
            depth_map = None

            if image_paths is not None:
                image_path = image_paths[idx]
                try:
                    image_rgb = _load_uint8_rgb(image_path)
                    current_prob = _load_prob_mask(gt_path)
                except FileNotFoundError:
                    pass

            student_prob = _test_style_prediction(student_prob, current_prob.shape)
            if image_path is not None:
                depth_map = self._load_depth_map(image_path, current_prob.shape)
            coarse_mask = prob_to_coarse_mask(student_prob)
            prompt_mask = coarse_mask_to_prompt_mask(
                coarse_mask,
                min_area=150,
                fg_thresh=self.prompt_fg_thresh,
                strict_greater=True,
            )
            if prompt_mask.sum() == 0:
                continue
            full_background_mask = (student_prob <= self.bg_prob_thresh).astype(np.uint8)

            components: List[np.ndarray] = []
            if mode in {"affinity_points", "split_only"}:
                components = self._split_prompt_mask_into_affinity_instances(
                    prompt_mask=prompt_mask,
                    depth_map=depth_map,
                    image_rgb=image_rgb,
                )
            if not components:
                components = split_prompt_mask_into_seeds(
                    prompt_mask,
                    coarse_mask=coarse_mask,
                    depth_map=depth_map,
                    core_thresh=self.core_thresh,
                    core_min_area=16,
                    depth_split_weight=self.depth_split_weight,
                )
            if not components:
                continue
            components = sorted(
                components,
                key=lambda component: int(np.asarray(component, dtype=np.uint8).sum()),
                reverse=True,
            )
            self._save_split_result(
                epoch=epoch,
                gt_path=gt_path,
                image_rgb=image_rgb,
                student_prob=student_prob,
                prompt_mask=prompt_mask,
                seed_masks=components,
                depth_map=depth_map,
            )
            if mode == "split_only":
                continue

            all_points_xy: List[np.ndarray] = []
            all_point_labels: List[np.ndarray] = []
            all_boxes_xyxy: List[np.ndarray] = []
            combined_prob = np.zeros_like(student_prob, dtype=np.float32)
            seed_scores: List[float] = []

            for component_idx, component in enumerate(components):
                seed_mask = _ensure_binary_mask(component)
                if seed_mask.sum() == 0:
                    continue

                box_xyxy = None
                points_xy = None
                point_labels = None
                if mode == "box":
                    base_box_xyxy = mask_to_box(seed_mask)
                    if base_box_xyxy is None:
                        continue
                    seed_coarse_mask = coarse_mask * seed_mask.astype(np.float32)
                    box_xyxy = _expand_box_prompt_from_coarse_mask(seed_coarse_mask, base_box_xyxy)
                elif mode == "affinity_points":
                    points_xy = _select_instance_positive_points(
                        seed_mask,
                        max_points=self.seed_points_per_instance,
                    )
                    if points_xy.size <= 0:
                        continue
                    point_labels = np.ones((points_xy.shape[0],), dtype=np.int32)

                mask_prompt = None

                masks, scores = self._predict_candidates(
                    image_rgb=image_rgb,
                    box_xyxy=box_xyxy,
                    points_xy=points_xy,
                    point_labels=point_labels,
                    mask_prompt=mask_prompt,
                    mode=mode,
                )
                self._save_seed_candidate_visualizations(
                    epoch=epoch,
                    gt_path=gt_path,
                    image_rgb=image_rgb,
                    points_xy=points_xy,
                    point_labels=point_labels,
                    box_xyxy=box_xyxy,
                    seed_mask=seed_mask,
                    full_background_mask=full_background_mask,
                    prompt_mask=prompt_mask,
                    masks=masks,
                    scores=scores,
                    component_idx=component_idx + 1,
                )
                seed_candidates = self._select_seed_candidates_for_merge(
                    masks=masks,
                    scores=scores,
                    seed_mask=seed_mask,
                    full_background_mask=full_background_mask,
                    prompt_mask=prompt_mask,
                    seed_recall_thresh=self.seed_recall_thresh if mode == "affinity_points" else None,
                )
                if not seed_candidates:
                    continue

                for seed_candidate in seed_candidates:
                    if seed_candidate.prob is None:
                        continue
                    combined_prob = np.maximum(combined_prob, seed_candidate.prob.astype(np.float32))
                    seed_scores.append(float(seed_candidate.score))
                if points_xy is not None and point_labels is not None:
                    all_points_xy.append(points_xy.astype(np.float32))
                    all_point_labels.append(point_labels.astype(np.int32))
                if mode == "box" and box_xyxy is not None:
                    all_boxes_xyxy.append(np.asarray(box_xyxy, dtype=np.float32))

            if not seed_scores:
                continue

            points_xy = np.concatenate(all_points_xy, axis=0) if all_points_xy else np.zeros((0, 2), dtype=np.float32)
            point_labels = np.concatenate(all_point_labels, axis=0) if all_point_labels else np.zeros((0,), dtype=np.int32)
            box_xyxy = np.stack(all_boxes_xyxy, axis=0) if all_boxes_xyxy else None

            candidate = self._build_candidate_summary(
                candidate_prob=combined_prob,
                score=float(np.mean(seed_scores)),
                current_prob=current_prob,
                student_prob=student_prob,
            )
            heat_iou = _binary_iou(candidate.mask, prompt_mask)

            flip_value = flips[idx]
            if isinstance(flip_value, torch.Tensor):
                flip_value = bool(flip_value.item())
            else:
                flip_value = bool(flip_value)
            self._save_epoch_result(
                epoch=epoch,
                gt_path=gt_path,
                image_rgb=image_rgb,
                student_prob=student_prob,
                points_xy=points_xy,
                point_labels=point_labels,
                box_xyxy=box_xyxy if mode == "box" else None,
                mask=candidate.mask,
                heat_iou=heat_iou,
                flip_value=flip_value,
            )

            if candidate.better_count < 2:
                continue

            new_mask = candidate.mask.astype(np.uint8) * 255
            if flip_value:
                new_mask = np.ascontiguousarray(new_mask[:, ::-1])
            cv2.imwrite(gt_path, new_mask)
            summary.add_replacement(candidate.metrics)

        return summary
