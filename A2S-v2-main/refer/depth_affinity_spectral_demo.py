import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from skimage.segmentation import slic

from depth_discontinuity_demo import (
    PALETTE,
    _build_stem_index,
    _build_support_mask,
    _compute_depth_discontinuity,
    _connected_components,
    _draw_boundaries,
    _iter_depth_paths,
    _labels_to_rgb,
    _normalize_gray,
    _preprocess_depth,
    _read_rgb_image,
    _restore_mask_coverage,
    _to_uint8,
)
from dinov2_feature_viz import _OnTheFlyDINOExtractor, _default_dino_repo


def _lab_image(rgb: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    if rgb is None:
        return np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[:2] != tuple(shape):
        return np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0


def _resize_rgb_if_needed(rgb: Optional[np.ndarray], target_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if rgb is None:
        return None
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape[:2] == tuple(target_shape):
        return rgb
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    return cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _build_slic_input_image(
    depth_crop: np.ndarray,
    rgb_crop: Optional[np.ndarray],
    depth_scale: float,
    input_mode: str = "rgbd",
) -> np.ndarray:
    depth_crop = _normalize_gray(depth_crop).astype(np.float32)
    input_mode = str(input_mode).strip().lower()
    if input_mode not in {"rgb", "depth", "rgbd"}:
        input_mode = "rgbd"

    if input_mode == "depth":
        return depth_crop[..., None]

    if rgb_crop is None:
        return depth_crop[..., None]

    lab_crop = _lab_image(rgb_crop, depth_crop.shape)
    if input_mode == "rgb":
        return lab_crop.astype(np.float32)

    return np.concatenate(
        [
            lab_crop.astype(np.float32),
            (float(depth_scale) * depth_crop)[..., None].astype(np.float32),
        ],
        axis=2,
    )


def _oversegment_component(
    component_crop: np.ndarray,
    depth_crop: np.ndarray,
    rgb_crop: Optional[np.ndarray],
    superpixel_size: int,
    min_superpixel_area: int,
    slic_compactness: float,
    slic_sigma: float,
    slic_depth_scale: float,
    slic_input_mode: str = "rgbd",
) -> Tuple[np.ndarray, Dict[str, object]]:
    component_crop = (np.asarray(component_crop) > 0).astype(np.uint8)
    label_crop = np.zeros_like(component_crop, dtype=np.int32)
    debug_info: Dict[str, object] = {
        "requested_segments": 0,
        "raw_superpixels": 0,
        "kept_superpixels": 0,
        "slic_input_mode": str(slic_input_mode).strip().lower() or "rgbd",
    }
    if int(component_crop.sum()) <= 0:
        return label_crop, debug_info

    component_area = int(component_crop.sum())
    approx_size = max(8, int(superpixel_size))
    n_segments = max(1, int(round(float(component_area) / float(approx_size * approx_size))))
    debug_info["requested_segments"] = int(n_segments)
    if n_segments <= 1:
        label_crop[component_crop > 0] = 1
        debug_info["raw_superpixels"] = 1
        debug_info["kept_superpixels"] = 1
        return label_crop, debug_info

    slic_input = _build_slic_input_image(
        depth_crop=depth_crop,
        rgb_crop=rgb_crop,
        depth_scale=slic_depth_scale,
        input_mode=slic_input_mode,
    )
    try:
        slic_labels = slic(
            slic_input,
            n_segments=int(n_segments),
            compactness=float(slic_compactness),
            sigma=float(slic_sigma),
            start_label=1,
            mask=(component_crop > 0),
            convert2lab=False,
            enforce_connectivity=True,
            min_size_factor=0.4,
            max_size_factor=3.0,
            channel_axis=-1,
        ).astype(np.int32)
    except Exception:
        label_crop[component_crop > 0] = 1
        debug_info["raw_superpixels"] = 1
        debug_info["kept_superpixels"] = 1
        debug_info["slic_error"] = True
        return label_crop, debug_info

    raw_masks: List[np.ndarray] = []
    kept_masks: List[np.ndarray] = []
    for local_label in sorted(int(value) for value in np.unique(slic_labels) if int(value) > 0):
        region_mask = ((slic_labels == local_label) & (component_crop > 0)).astype(np.uint8)
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
    superpixel_size: int,
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
        rgb_crop = None if rgb is None else np.asarray(rgb, dtype=np.uint8)[crop_slice]
        appearance_crop = None
        if appearance_map is not None and np.asarray(appearance_map).ndim == 3 and np.asarray(appearance_map).shape[:2] == support.shape:
            appearance_crop = np.asarray(appearance_map, dtype=np.float32)[crop_slice]

        superpixel_crop, superpixel_debug = _oversegment_component(
            component_crop=component_crop,
            depth_crop=depth_crop,
            rgb_crop=rgb_crop,
            superpixel_size=superpixel_size,
            min_superpixel_area=min_superpixel_area,
            slic_compactness=slic_compactness,
            slic_sigma=slic_sigma,
            slic_depth_scale=slic_depth_scale,
            slic_input_mode=slic_input_mode,
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
    parser.add_argument("--min-component-area", default=800, type=int, help="Minimum GT-support connected-component area.")
    parser.add_argument("--min-instance-area", default=400, type=int, help="Minimum kept instance area during recursive NCut.")
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
    parser.add_argument("--superpixel-size", default=24, type=int, help="Approximate SLIC superpixel width/height inside the GT region.")
    parser.add_argument("--min-superpixel-area", default=48, type=int, help="Minimum kept superpixel area before coverage restoration.")
    parser.add_argument("--slic-compactness", default=8.0, type=float, help="Compactness used by SLIC.")
    parser.add_argument("--slic-sigma", default=1.0, type=float, help="Gaussian smoothing sigma used by SLIC.")
    parser.add_argument("--slic-depth-scale", default=0.35, type=float, help="Relative depth-channel scale appended to Lab color before SLIC.")
    parser.add_argument("--slic-input-mode", default="rgbd", choices=["rgb", "depth", "rgbd"], help="Input modality used by SLIC: RGB only, depth only, or concatenated RGB-D.")
    parser.add_argument("--sigma-sem", default=0.20, type=float, help="Sigma used in semantic affinity exp(-(1-cos)/sigma_sem).")
    parser.add_argument("--sigma-dep", default=0.02, type=float, help="Sigma used in depth affinity exp(-(d_i-d_j)^2/sigma_dep).")
    parser.add_argument("--sigma-spatial", default=0.12, type=float, help="Sigma used in spatial affinity exp(-dist^2/sigma_spatial). Distances are normalized by the component diagonal.")
    parser.add_argument("--sigma-edge", default=0.20, type=float, help="Sigma used in edge penalty exp(-max_edge/sigma_edge).")
    parser.add_argument("--min-affinity", default=1e-6, type=float, help="Clamp affinities below this value to zero.")
    parser.add_argument("--min-cluster-regions", default=2, type=int, help="Minimum number of superpixels kept on each side of a recursive split.")
    parser.add_argument("--ncut-threshold", default=0.18, type=float, help="Maximum normalized-cut score accepted by a recursive split.")
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
            superpixel_size=args.superpixel_size,
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
