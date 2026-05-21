import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry

from .affinity_split import SAMTrainAffinityMixin
from .candidate_selection import SAMTrainCandidateSelectionMixin
from .image_ops import (
    _build_stem_path_index,
    _connected_components,
    _ensure_binary_mask,
    _ensure_uint8_rgb,
    _filter_components,
    _normalize_gray_map,
    _resize_to_short_edge,
    _tensor_to_uint8_rgb,
    _threshold_prob_map,
)
from .output_saver import SAMTrainOutputMixin
from .prompt_points import (
    _center_point,
    _component_region_points,
    _greedy_fill_points,
    _mask_to_box,
    _scale_box,
    _scale_points,
    _select_box_refined_points,
    _select_instance_positive_points,
)
from .sam_runner import SAMTrainRunnerMixin
from .types import SAMCandidate


class SAMTrainHelper(SAMTrainAffinityMixin, SAMTrainRunnerMixin, SAMTrainCandidateSelectionMixin, SAMTrainOutputMixin):
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
        heat_bin_thr: float = 0.5,
        resize_short_edge: int = 640,
        use_crf: bool = False,
        small_fg_box_thresh: float = 0.04,
        use_affinity_split: bool = False,
        depth_root: str = "",
        seed_points_per_instance: int = 3,
        affinity_min_component_area: int = 128,
        affinity_min_instance_area: int = 64,
        affinity_superpixel_count: int = 200,
        affinity_min_superpixel_area: int = 40,
        affinity_slic_compactness: float = 10.0,
        affinity_slic_sigma: float = 1.0,
        affinity_slic_depth_scale: float = 0.5,
        affinity_sigma_sem: float = 0.20,
        affinity_sigma_dep: float = 0.02,
        affinity_sigma_spatial: float = 0.12,
        affinity_sigma_edge: float = 0.20,
        affinity_min_affinity: float = 1e-6,
        affinity_min_cluster_regions: int = 2,
        affinity_ncut_threshold: float = 0.10,
        affinity_max_recursion_depth: int = 8,
        affinity_use_mask_prompt: bool = False,
        use_negative_prompt: bool = True,
        neg_ccam_thresh: float = 0.25,
        neg_bg_thresh: float = 0.05,
        neg_box_expand: float = 0.15,
        neg_margin: int = 8,
        neg_points_per_component: int = 1,
        mask_prompt_fg_logit: float = 3.0,
        refine_missing_ratio_thresh: float = 0.25,
        save_prefixes: Optional[List[str]] = None,
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
        self.affinity_superpixel_count = int(max(1, affinity_superpixel_count))
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
        self.affinity_use_mask_prompt = bool(affinity_use_mask_prompt)
        self.use_negative_prompt = bool(use_negative_prompt)
        self.neg_ccam_thresh = float(np.clip(neg_ccam_thresh, 0.0, 1.0))
        self.neg_bg_thresh = float(np.clip(neg_bg_thresh, 0.0, 1.0))
        self.neg_box_expand = float(max(0.0, neg_box_expand))
        self.neg_margin = int(max(0, neg_margin))
        self.neg_points_per_component = int(max(1, neg_points_per_component))
        self.mask_prompt_fg_logit = float(mask_prompt_fg_logit)
        self.refine_missing_ratio_thresh = float(np.clip(refine_missing_ratio_thresh, 0.0, 1.0))
        self.save_prefixes = None if save_prefixes is None else set(str(prefix) for prefix in save_prefixes)
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
            if self.affinity_use_mask_prompt:
                if self.save_prefixes is None:
                    print(
                        "Saving affinity point-only, instance-mask+point, "
                        "and whole-mask+group-point pseudo labels."
                    )
                else:
                    print("Saving compact affinity pseudo outputs: {}.".format(", ".join(sorted(self.save_prefixes))))

    def _should_save_prefix(self, prefix: str) -> bool:
        return self.save_prefixes is None or str(prefix) in self.save_prefixes

    def _should_save_any_prefix(self, prefixes: List[str]) -> bool:
        if self.save_prefixes is None:
            return True
        return any(str(prefix) in self.save_prefixes for prefix in prefixes)

    @staticmethod
    def _empty_negative_points() -> np.ndarray:
        return np.zeros((0, 2), dtype=np.float32)

    def _build_ccam_negative_points(
        self,
        cam_prob: np.ndarray,
        full_background_mask: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        cam_prob = np.clip(np.asarray(cam_prob, dtype=np.float32), 0.0, 1.0)
        full_background_mask = _ensure_binary_mask(full_background_mask)
        if cam_prob.shape != full_background_mask.shape:
            cam_prob = cv2.resize(
                cam_prob,
                full_background_mask.shape[::-1],
                interpolation=cv2.INTER_LINEAR,
            )

        ccam_mask = (cam_prob >= self.neg_ccam_thresh).astype(np.uint8)
        num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(ccam_mask, connectivity=8)
        label_map = np.asarray(label_map, dtype=np.int32)
        negative_points_by_label: Dict[int, np.ndarray] = {}
        if num_labels <= 1:
            return label_map, negative_points_by_label

        h, w = cam_prob.shape
        all_ccam = (label_map > 0).astype(np.uint8)
        if self.neg_margin > 0:
            kernel_size = int(self.neg_margin) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            unsafe_ccam = cv2.dilate(all_ccam, kernel, iterations=1).astype(bool)
        else:
            unsafe_ccam = all_ccam.astype(bool)

        low_activation_bg = cam_prob < self.neg_bg_thresh
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            bw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            expand_x = int(round(float(bw) * self.neg_box_expand))
            expand_y = int(round(float(bh) * self.neg_box_expand))
            x0 = max(0, x - expand_x)
            y0 = max(0, y - expand_y)
            x1 = min(w, x + bw + expand_x)
            y1 = min(h, y + bh + expand_y)
            if x1 <= x0 or y1 <= y0:
                continue

            box_mask = np.zeros((h, w), dtype=bool)
            box_mask[y0:y1, x0:x1] = True
            box_low_activation = box_mask & low_activation_bg
            neg_region = (
                box_low_activation
                & (full_background_mask > 0)
                & (~unsafe_ccam)
            )
            if not np.any(neg_region):
                neg_region = box_low_activation & (~unsafe_ccam)
            if not np.any(neg_region):
                continue

            distance = cv2.distanceTransform(neg_region.astype(np.uint8), cv2.DIST_L2, 5)
            ys, xs = np.where(neg_region)
            if len(xs) == 0:
                continue
            center = np.array([x + bw / 2.0, y + bh / 2.0], dtype=np.float32)
            coords = np.stack([xs, ys], axis=1).astype(np.float32)
            distance_scores = distance[ys, xs].astype(np.float32)
            if float(distance_scores.max()) <= 0.0:
                distance_scores = np.ones_like(distance_scores, dtype=np.float32)
            angles = np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0])
            order = np.argsort(-distance_scores)
            selected: List[np.ndarray] = []
            selected_bins: set = set()
            for coord_idx in order:
                bin_idx = int(np.floor((float(angles[coord_idx]) + np.pi) / (2.0 * np.pi / 8.0))) % 8
                if bin_idx in selected_bins and len(selected_bins) < 8:
                    continue
                selected.append(coords[coord_idx])
                selected_bins.add(bin_idx)
                if len(selected) >= self.neg_points_per_component:
                    break
            if len(selected) < self.neg_points_per_component:
                selected_arr = np.stack(selected, axis=0) if selected else np.zeros((0, 2), dtype=np.float32)
                for coord_idx in order:
                    coord = coords[coord_idx]
                    if selected_arr.size > 0 and np.any(np.all(np.isclose(selected_arr, coord), axis=1)):
                        continue
                    selected.append(coord)
                    selected_arr = np.stack(selected, axis=0)
                    if len(selected) >= self.neg_points_per_component:
                        break
            negative_points_by_label[int(label_idx)] = np.stack(selected, axis=0).astype(np.float32)

        return label_map, negative_points_by_label

    @staticmethod
    def _component_ccam_label(seed_mask: np.ndarray, ccam_label_map: np.ndarray) -> Optional[int]:
        seed_mask = _ensure_binary_mask(seed_mask)
        if int(seed_mask.sum()) <= 0 or ccam_label_map.size == 0:
            return None
        if seed_mask.shape != ccam_label_map.shape:
            ccam_label_map = cv2.resize(
                ccam_label_map.astype(np.int32),
                seed_mask.shape[::-1],
                interpolation=cv2.INTER_NEAREST,
            )
        labels = ccam_label_map[seed_mask > 0]
        labels = labels[labels > 0]
        if labels.size == 0:
            return None
        counts = np.bincount(labels.astype(np.int32))
        if counts.size <= 1:
            return None
        return int(np.argmax(counts[1:]) + 1)

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
            extra_heat_iou_mask = heat_iou_mask if is_large_target else None
            extra_heat_iou_thresh = self.heat_iou_thresh if is_large_target else None
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
            point_neg_raw_candidates: List[SAMCandidate] = []
            instance_mask_prompt_raw_candidates: List[SAMCandidate] = []
            instance_mask_prompt_neg_raw_candidates: List[SAMCandidate] = []
            whole_mask_prompt_raw_candidates: List[SAMCandidate] = []
            negative_points_by_point: Dict[int, np.ndarray] = {}
            save_point_negative_prompt = self.use_negative_prompt and self._should_save_any_prefix([
                "point_candidates_neg",
                "points_sam_seg_neg",
                "points_sam_seg_neg_rule_a",
                "points_sam_seg_neg_rule_ab",
                "pseudo_labels_neg",
                "pseudo_labels_neg_binary",
                "pseudo_labels_neg_rule_a",
                "pseudo_labels_neg_rule_a_binary",
                "pseudo_labels_neg_rule_ab",
                "pseudo_labels_neg_rule_ab_binary",
                "sam_final_candidates_neg",
                "sam_final_candidates_neg_rule_a",
                "sam_final_candidates_neg_rule_ab",
            ])
            save_instance_mask_prompt = self.affinity_use_mask_prompt and self._should_save_any_prefix([
                "mask_point_candidates",
                "mask_point_sam_seg",
                "mask_point_sam_seg_rule_a",
                "mask_point_sam_seg_rule_ab",
                "pseudo_labels_mask_point",
                "pseudo_labels_mask_point_binary",
                "pseudo_labels_mask_point_rule_a",
                "pseudo_labels_mask_point_rule_a_binary",
                "pseudo_labels_mask_point_rule_ab",
                "pseudo_labels_mask_point_rule_ab_binary",
                "sam_mask_point_final_candidates",
                "sam_mask_point_final_candidates_rule_a",
                "sam_mask_point_final_candidates_rule_ab",
            ])
            save_instance_mask_prompt_neg = self.use_negative_prompt and self.affinity_use_mask_prompt and self._should_save_any_prefix([
                "mask_point_candidates_neg",
                "mask_point_sam_seg_neg",
                "mask_point_sam_seg_neg_rule_a",
                "mask_point_sam_seg_neg_rule_ab",
                "pseudo_labels_mask_point_neg",
                "pseudo_labels_mask_point_neg_binary",
                "pseudo_labels_mask_point_neg_rule_a",
                "pseudo_labels_mask_point_neg_rule_a_binary",
                "pseudo_labels_mask_point_neg_rule_ab",
                "pseudo_labels_mask_point_neg_rule_ab_binary",
                "sam_mask_point_final_candidates_neg",
                "sam_mask_point_final_candidates_neg_rule_a",
                "sam_mask_point_final_candidates_neg_rule_ab",
            ])
            save_whole_mask_prompt = self.affinity_use_mask_prompt and self._should_save_any_prefix([
                "whole_mask_point_candidates",
                "whole_mask_point_sam_seg",
                "whole_mask_point_sam_seg_rule_a",
                "whole_mask_point_sam_seg_rule_ab",
                "pseudo_labels_whole_mask_point",
                "pseudo_labels_whole_mask_point_binary",
                "pseudo_labels_whole_mask_point_rule_a",
                "pseudo_labels_whole_mask_point_rule_a_binary",
                "pseudo_labels_whole_mask_point_rule_ab",
                "pseudo_labels_whole_mask_point_rule_ab_binary",
                "sam_whole_mask_point_final_candidates",
                "sam_whole_mask_point_final_candidates_rule_a",
                "sam_whole_mask_point_final_candidates_rule_ab",
            ])
            save_point_refine = self.use_affinity_split and self._should_save_any_prefix([
                "point_refine_candidates",
                "point_refine_sam_seg",
                "point_refine_sam_seg_rule_a",
                "point_refine_sam_seg_rule_ab",
                "point_refine_case_a",
                "point_refine_case_b",
                "point_refine_case_b_after",
                "point_refine_case_b_heat_iou_drop",
                "point_refine_case_b_before_candidates",
                "point_refine_case_b_after_candidates",
                "pseudo_labels_point_refine",
                "pseudo_labels_point_refine_binary",
                "pseudo_labels_point_refine_rule_a",
                "pseudo_labels_point_refine_rule_a_binary",
                "pseudo_labels_point_refine_rule_ab",
                "pseudo_labels_point_refine_rule_ab_binary",
                "sam_point_refine_final_candidates",
                "sam_point_refine_final_candidates_rule_a",
                "sam_point_refine_final_candidates_rule_ab",
            ])
            save_point_refine_neg = self.use_negative_prompt and self.use_affinity_split and self._should_save_any_prefix([
                "point_refine_candidates_neg",
                "point_refine_sam_seg_neg",
                "point_refine_sam_seg_neg_rule_a",
                "point_refine_sam_seg_neg_rule_ab",
                "pseudo_labels_point_refine_neg",
                "pseudo_labels_point_refine_neg_binary",
                "pseudo_labels_point_refine_neg_rule_a",
                "pseudo_labels_point_refine_neg_rule_a_binary",
                "pseudo_labels_point_refine_neg_rule_ab",
                "pseudo_labels_point_refine_neg_rule_ab_binary",
                "sam_point_refine_final_candidates_neg",
                "sam_point_refine_final_candidates_neg_rule_a",
                "sam_point_refine_final_candidates_neg_rule_ab",
            ])
            if self.use_affinity_split:
                candidate_metric_heat_iou_mask = prompt_mask.copy()
                candidate_fg_iou_masks_by_point = {}
                depth_map = self._load_depth_map(sample_name, prompt_mask.shape)
                components, superpixel_label_map = self._split_prompt_mask_into_affinity_instances(
                    prompt_mask=prompt_mask,
                    image_rgb=image_rgb,
                    depth_map=depth_map,
                )
                if not components:
                    self._record_failure(sample_name, epoch)
                    continue

                ccam_label_map = np.zeros_like(prompt_mask, dtype=np.int32)
                negative_points_by_ccam_label: Dict[int, np.ndarray] = {}
                if save_point_negative_prompt or save_instance_mask_prompt_neg:
                    ccam_label_map, negative_points_by_ccam_label = self._build_ccam_negative_points(
                        prob_np,
                        full_background_mask,
                    )

                self.predictor.set_image(np.ascontiguousarray(_ensure_uint8_rgb(image_rgb_aug)))
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
                    component_neg_points = self._empty_negative_points()
                    if save_point_negative_prompt or save_instance_mask_prompt_neg:
                        ccam_label = self._component_ccam_label(seed_mask, ccam_label_map)
                        if ccam_label is not None and ccam_label in negative_points_by_ccam_label:
                            component_neg_points = negative_points_by_ccam_label[ccam_label].astype(np.float32)
                            negative_points_by_point[point_idx] = component_neg_points
                    component_neg_points_aug = _scale_points(component_neg_points, prompt_mask.shape, aug_hw)
                    raw_candidates.extend(
                        self._run_sam_multi_positive(
                            image_rgb_aug,
                            component_points_aug,
                            point_idx=point_idx,
                            set_image=False,
                        )
                    )
                    if save_point_negative_prompt:
                        point_neg_raw_candidates.extend(
                            self._run_sam_multi_positive(
                                image_rgb_aug,
                                component_points_aug,
                                negative_points_xy=component_neg_points_aug,
                                point_idx=point_idx,
                                set_image=False,
                            )
                        )
                    if save_instance_mask_prompt:
                        instance_mask_prompt_raw_candidates.extend(
                            self._run_sam_multi_positive_with_mask_prompt(
                                image_rgb_aug,
                                component_points_aug,
                                seed_mask,
                                point_idx=point_idx,
                                set_image=False,
                            )
                        )
                    if save_instance_mask_prompt_neg:
                        instance_mask_prompt_neg_raw_candidates.extend(
                            self._run_sam_multi_positive_with_mask_prompt(
                                image_rgb_aug,
                                component_points_aug,
                                seed_mask,
                                negative_points_xy=component_neg_points_aug,
                                point_idx=point_idx,
                                set_image=False,
                            )
                        )
                    if save_whole_mask_prompt:
                        whole_mask_prompt_raw_candidates.extend(
                            self._run_sam_multi_positive_with_mask_prompt(
                                image_rgb_aug,
                                component_points_aug,
                                prompt_mask,
                                point_idx=point_idx,
                                set_image=False,
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
                    superpixel_label_map,
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
            fg_candidates_by_point = candidates_by_point if candidate_fg_iou_masks_by_point is not None else None
            kept_candidates = self._select_best_candidates(
                valid_candidates_by_point,
            )
            rule_a_candidates = self._select_best_candidates(
                valid_candidates_by_point,
                fg_candidates_by_point=fg_candidates_by_point,
                fg_score_thresh=0.9,
                fg_bg_iou_thresh=bg_iou_thresh,
                include_fg_iou=True,
            )
            rule_ab_candidates = self._select_best_candidates(
                valid_candidates_by_point,
                fg_candidates_by_point=fg_candidates_by_point,
                fg_score_thresh=0.9,
                fg_iou_thresh=0.15,
                fg_bg_iou_thresh=bg_iou_thresh,
                include_fg_iou=True,
            )
            if extra_heat_iou_mask is not None:
                extra_heat_candidates = self._select_best_heat_iou_candidates(
                    candidates_by_point,
                    extra_heat_iou_mask,
                    full_background_mask,
                    heat_iou_thresh=extra_heat_iou_thresh,
                    bg_iou_thresh=bg_iou_thresh,
                )
                kept_candidates = self._append_unique_candidates(kept_candidates, extra_heat_candidates)
                rule_a_candidates = self._append_unique_candidates(rule_a_candidates, extra_heat_candidates)
                rule_ab_candidates = self._append_unique_candidates(rule_ab_candidates, extra_heat_candidates)
            if kept_candidates:
                final_mask = self._merge_candidate_masks(kept_candidates, fallback_mask)
            else:
                self._record_failure(sample_name, epoch)
                final_mask = fallback_mask.copy()

            final_prob = self._merge_candidate_prob_maps(kept_candidates, fallback_mask)
            rule_a_prob = self._merge_candidate_prob_maps(rule_a_candidates, fallback_mask)
            rule_ab_prob = self._merge_candidate_prob_maps(rule_ab_candidates, fallback_mask)
            rule_a_mask = _threshold_prob_map(rule_a_prob, self.mask_prob_threshold)
            rule_ab_mask = _threshold_prob_map(rule_ab_prob, self.mask_prob_threshold)

            self._save_points_heatmap_overlay(image_rgb, prob_np, points_xy, box_xyxy, sample_name, epoch)
            self._save_points_seg_overlay(
                image_rgb,
                final_mask,
                points_xy,
                box_xyxy,
                sample_name,
                epoch,
            )
            self._save_points_seg_overlay(
                image_rgb,
                rule_a_mask,
                points_xy,
                box_xyxy,
                sample_name,
                epoch,
                prefix="points_sam_seg_rule_a",
            )
            self._save_points_seg_overlay(
                image_rgb,
                rule_ab_mask,
                points_xy,
                box_xyxy,
                sample_name,
                epoch,
                prefix="points_sam_seg_rule_ab",
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
            self._save_background_point_candidates(
                image_rgb,
                raw_candidates,
                points_xy,
                box_xyxy,
                candidate_heat_iou_ref_mask,
                full_background_mask,
                heat_iou_thresh,
                bg_iou_thresh,
                uncertain_area_ratio,
                sample_name,
                epoch,
                fg_iou_masks_by_point=candidate_fg_iou_masks_by_point,
            )
            if is_large_target:
                self._save_large_target_outputs(
                    image_rgb,
                    final_prob,
                    uncertain_with_fg_mask,
                    points_xy,
                    box_xyxy,
                    uncertain_area_ratio,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
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
                point_group_starts=sorted(candidate_fg_iou_masks_by_point.keys()) if candidate_fg_iou_masks_by_point else None,
            )
            self._save_final_candidates(
                image_rgb,
                rule_a_candidates,
                points_xy,
                box_xyxy,
                uncertain_area_ratio,
                sample_name,
                epoch,
                prefix="sam_final_candidates_rule_a",
                point_group_starts=sorted(candidate_fg_iou_masks_by_point.keys()) if candidate_fg_iou_masks_by_point else None,
            )
            self._save_final_candidates(
                image_rgb,
                rule_ab_candidates,
                points_xy,
                box_xyxy,
                uncertain_area_ratio,
                sample_name,
                epoch,
                prefix="sam_final_candidates_rule_ab",
                point_group_starts=sorted(candidate_fg_iou_masks_by_point.keys()) if candidate_fg_iou_masks_by_point else None,
            )
            point_group_starts = sorted(candidate_fg_iou_masks_by_point.keys()) if candidate_fg_iou_masks_by_point else []
            point_refine_sources = kept_candidates + rule_ab_candidates
            point_refine_missing_mask = (
                _ensure_binary_mask(prompt_mask).astype(bool)
                & (~_ensure_binary_mask(final_mask).astype(bool))
            ).astype(np.uint8)
            point_refine_missing_ratio = float(point_refine_missing_mask.sum()) / float(max(1, int(_ensure_binary_mask(prompt_mask).sum())))
            point_refine_is_case_b = point_refine_missing_ratio >= self.refine_missing_ratio_thresh
            point_refine_output_prob = final_prob
            point_refine_rule_a_output_prob = rule_a_prob
            point_refine_rule_ab_output_prob = rule_ab_prob
            if self.use_affinity_split and point_refine_sources and save_point_refine:
                self._save_point_refine_cases(
                    image_rgb,
                    final_mask,
                    points_xy,
                    prompt_mask,
                    point_refine_missing_ratio,
                    sample_name,
                    epoch,
                )
            if self.use_affinity_split and point_refine_is_case_b and point_refine_sources and (save_point_refine or save_point_refine_neg):
                self.predictor.set_image(np.ascontiguousarray(_ensure_uint8_rgb(image_rgb_aug)))
            if self.use_affinity_split and point_refine_is_case_b and point_refine_sources and save_point_refine:
                point_refine_before_candidates = self._append_unique_candidates([], point_refine_sources)
                self._save_final_candidates(
                    image_rgb,
                    point_refine_before_candidates,
                    points_xy,
                    box_xyxy,
                    uncertain_area_ratio,
                    sample_name,
                    epoch,
                    prefix="point_refine_case_b_before_candidates",
                    point_group_starts=point_group_starts,
                )
                (
                    point_refine_raw_candidates,
                    point_refine_points_xy,
                    point_refine_fg_masks_by_point,
                    _,
                    _,
                ) = self._run_point_refine_candidates(
                    image_rgb_aug,
                    point_refine_sources,
                    points_xy,
                    prompt_mask,
                    point_group_starts,
                    global_missing_mask=point_refine_missing_mask,
                    global_missing_ratio=point_refine_missing_ratio,
                    use_negative=False,
                )
                (
                    _,
                    point_refine_kept_candidates,
                    _,
                    _,
                    point_refine_final_prob,
                    point_refine_rule_a_prob,
                    point_refine_rule_ab_prob,
                    _,
                    _,
                    _,
                ) = self._build_selected_candidate_outputs(
                    point_refine_raw_candidates,
                    fallback_mask,
                    candidate_heat_iou_ref_mask,
                    full_background_mask,
                    candidate_metric_heat_iou_mask,
                    point_refine_fg_masks_by_point,
                    heat_iou_thresh,
                    bg_iou_thresh,
                    extra_heat_iou_mask=extra_heat_iou_mask,
                    extra_heat_iou_thresh=extra_heat_iou_thresh,
                )
                point_refine_output_prob = point_refine_final_prob
                point_refine_rule_a_output_prob = point_refine_rule_a_prob
                point_refine_rule_ab_output_prob = point_refine_rule_ab_prob
                self._save_final_candidates(
                    image_rgb,
                    point_refine_kept_candidates,
                    point_refine_points_xy,
                    box_xyxy,
                    uncertain_area_ratio,
                    sample_name,
                    epoch,
                    prefix="point_refine_case_b_after_candidates",
                    point_group_starts=sorted(point_refine_fg_masks_by_point.keys()),
                )
                self._save_point_refine_case_after(
                    image_rgb,
                    point_refine_final_prob,
                    point_refine_points_xy,
                    prompt_mask,
                    point_refine_missing_ratio,
                    sample_name,
                    epoch,
                )
                self._save_point_refine_heat_iou_drop(
                    image_rgb,
                    final_prob,
                    point_refine_final_prob,
                    candidate_metric_heat_iou_mask,
                    points_xy,
                    point_refine_points_xy,
                    prompt_mask,
                    point_refine_missing_ratio,
                    sample_name,
                    epoch,
                    result_name="base",
                )
                self._save_point_refine_heat_iou_drop(
                    image_rgb,
                    rule_ab_prob,
                    point_refine_rule_ab_prob,
                    candidate_metric_heat_iou_mask,
                    points_xy,
                    point_refine_points_xy,
                    prompt_mask,
                    point_refine_missing_ratio,
                    sample_name,
                    epoch,
                    result_name="rule_ab",
                )
                if point_refine_kept_candidates:
                    self._save_point_refine_case_binary_pair(
                        image_rgb,
                        final_prob,
                        point_refine_final_prob,
                        point_refine_missing_ratio,
                        sample_name,
                        sample_meta,
                        idx,
                        epoch,
                    )
                self._save_mask_prompt_outputs(
                    image_rgb,
                    point_refine_raw_candidates,
                    point_refine_points_xy,
                    box_xyxy,
                    fallback_mask,
                    candidate_heat_iou_ref_mask,
                    full_background_mask,
                    candidate_metric_heat_iou_mask,
                    point_refine_fg_masks_by_point,
                    heat_iou_thresh,
                    bg_iou_thresh,
                    uncertain_area_ratio,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    seg_prefix="point_refine_sam_seg",
                    candidate_prefix="point_refine_candidates",
                    pseudo_prefix="pseudo_labels_point_refine",
                    final_candidates_prefix="sam_point_refine_final_candidates",
                    extra_heat_iou_mask=extra_heat_iou_mask,
                    extra_heat_iou_thresh=extra_heat_iou_thresh,
                )
            if self.use_affinity_split and save_point_refine:
                self._save_pseudo_label_grayscale(
                    point_refine_output_prob,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    prefix="pseudo_labels_point_refine",
                )
                self._save_pseudo_label_grayscale(
                    point_refine_rule_a_output_prob,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    prefix="pseudo_labels_point_refine_rule_a",
                )
                self._save_pseudo_label_grayscale(
                    point_refine_rule_ab_output_prob,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    prefix="pseudo_labels_point_refine_rule_ab",
                )
                self._save_pseudo_label_binary(
                    image_rgb,
                    point_refine_output_prob,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    prefix="pseudo_labels_point_refine_binary",
                )
                self._save_pseudo_label_binary(
                    image_rgb,
                    point_refine_rule_a_output_prob,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    prefix="pseudo_labels_point_refine_rule_a_binary",
                )
                self._save_pseudo_label_binary(
                    image_rgb,
                    point_refine_rule_ab_output_prob,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    prefix="pseudo_labels_point_refine_rule_ab_binary",
                )
            if self.use_affinity_split and point_refine_is_case_b and point_refine_sources and save_point_refine_neg:
                (
                    point_refine_neg_raw_candidates,
                    point_refine_neg_points_xy,
                    point_refine_neg_fg_masks_by_point,
                    point_refine_negative_points_by_point,
                    _,
                ) = self._run_point_refine_candidates(
                    image_rgb_aug,
                    point_refine_sources,
                    points_xy,
                    prompt_mask,
                    point_group_starts,
                    global_missing_mask=point_refine_missing_mask,
                    global_missing_ratio=point_refine_missing_ratio,
                    negative_points_by_point=negative_points_by_point,
                    use_negative=True,
                )
                self._save_mask_prompt_outputs(
                    image_rgb,
                    point_refine_neg_raw_candidates,
                    point_refine_neg_points_xy,
                    box_xyxy,
                    fallback_mask,
                    candidate_heat_iou_ref_mask,
                    full_background_mask,
                    candidate_metric_heat_iou_mask,
                    point_refine_neg_fg_masks_by_point,
                    heat_iou_thresh,
                    bg_iou_thresh,
                    uncertain_area_ratio,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    seg_prefix="point_refine_sam_seg_neg",
                    candidate_prefix="point_refine_candidates_neg",
                    pseudo_prefix="pseudo_labels_point_refine_neg",
                    final_candidates_prefix="sam_point_refine_final_candidates_neg",
                    negative_points_by_point=point_refine_negative_points_by_point,
                    extra_heat_iou_mask=extra_heat_iou_mask,
                    extra_heat_iou_thresh=extra_heat_iou_thresh,
                )
            if self.use_affinity_split and save_point_negative_prompt:
                self._save_mask_prompt_outputs(
                    image_rgb,
                    point_neg_raw_candidates,
                    points_xy,
                    box_xyxy,
                    fallback_mask,
                    candidate_heat_iou_ref_mask,
                    full_background_mask,
                    candidate_metric_heat_iou_mask,
                    candidate_fg_iou_masks_by_point,
                    heat_iou_thresh,
                    bg_iou_thresh,
                    uncertain_area_ratio,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    seg_prefix="points_sam_seg_neg",
                    candidate_prefix="point_candidates_neg",
                    pseudo_prefix="pseudo_labels_neg",
                    final_candidates_prefix="sam_final_candidates_neg",
                    negative_points_by_point=negative_points_by_point,
                    extra_heat_iou_mask=extra_heat_iou_mask,
                    extra_heat_iou_thresh=extra_heat_iou_thresh,
                )
            if self.use_affinity_split and save_instance_mask_prompt:
                self._save_mask_prompt_outputs(
                    image_rgb,
                    instance_mask_prompt_raw_candidates,
                    points_xy,
                    box_xyxy,
                    fallback_mask,
                    candidate_heat_iou_ref_mask,
                    full_background_mask,
                    candidate_metric_heat_iou_mask,
                    candidate_fg_iou_masks_by_point,
                    heat_iou_thresh,
                    bg_iou_thresh,
                    uncertain_area_ratio,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    seg_prefix="mask_point_sam_seg",
                    candidate_prefix="mask_point_candidates",
                    pseudo_prefix="pseudo_labels_mask_point",
                    final_candidates_prefix="sam_mask_point_final_candidates",
                    extra_heat_iou_mask=extra_heat_iou_mask,
                    extra_heat_iou_thresh=extra_heat_iou_thresh,
                )
            if self.use_affinity_split and save_instance_mask_prompt_neg:
                self._save_mask_prompt_outputs(
                    image_rgb,
                    instance_mask_prompt_neg_raw_candidates,
                    points_xy,
                    box_xyxy,
                    fallback_mask,
                    candidate_heat_iou_ref_mask,
                    full_background_mask,
                    candidate_metric_heat_iou_mask,
                    candidate_fg_iou_masks_by_point,
                    heat_iou_thresh,
                    bg_iou_thresh,
                    uncertain_area_ratio,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    seg_prefix="mask_point_sam_seg_neg",
                    candidate_prefix="mask_point_candidates_neg",
                    pseudo_prefix="pseudo_labels_mask_point_neg",
                    final_candidates_prefix="sam_mask_point_final_candidates_neg",
                    negative_points_by_point=negative_points_by_point,
                    extra_heat_iou_mask=extra_heat_iou_mask,
                    extra_heat_iou_thresh=extra_heat_iou_thresh,
                )
            if self.use_affinity_split and save_whole_mask_prompt:
                self._save_mask_prompt_outputs(
                    image_rgb,
                    whole_mask_prompt_raw_candidates,
                    points_xy,
                    box_xyxy,
                    fallback_mask,
                    candidate_heat_iou_ref_mask,
                    full_background_mask,
                    candidate_metric_heat_iou_mask,
                    candidate_fg_iou_masks_by_point,
                    heat_iou_thresh,
                    bg_iou_thresh,
                    uncertain_area_ratio,
                    sample_name,
                    sample_meta,
                    idx,
                    epoch,
                    seg_prefix="whole_mask_point_sam_seg",
                    candidate_prefix="whole_mask_point_candidates",
                    pseudo_prefix="pseudo_labels_whole_mask_point",
                    final_candidates_prefix="sam_whole_mask_point_final_candidates",
                    extra_heat_iou_mask=extra_heat_iou_mask,
                    extra_heat_iou_thresh=extra_heat_iou_thresh,
                )
