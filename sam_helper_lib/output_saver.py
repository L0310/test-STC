import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .image_ops import (
    _compute_iou,
    _ensure_binary_mask,
    _refine_mask_with_crf,
    _resize_mask,
    _sample_meta_value,
    _threshold_prob_map,
    _write_rgb,
)
from .prompt_points import _merge_prompt_points
from .sam_predictor import SAMHelper
from .types import SAMCandidate
from .visualization import (
    _draw_heatmap_overlay,
    _draw_label_grid_overlay,
    _seed_masks_overlay,
    _seed_masks_to_label_rgb,
    _seed_masks_to_union,
)


class SAMTrainOutputMixin:
    def _make_out_dir(self, prefix: str, epoch: int) -> Optional[str]:
        if not self._should_save_prefix(prefix):
            return None
        out_dir = os.path.join(self.save_root, prefix, f"epoch{epoch + 1}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_point_refine_cases(
        self,
        image_rgb: np.ndarray,
        first_round_mask: np.ndarray,
        points_xy: np.ndarray,
        prompt_mask: np.ndarray,
        missing_ratio: float,
        filename: str,
        epoch: int,
    ) -> None:
        case_name = "b" if float(missing_ratio) >= self.refine_missing_ratio_thresh else "a"
        out_dir = self._make_out_dir(f"point_refine_case_{case_name}", epoch)
        if out_dir is None:
            return
        stem = os.path.splitext(filename)[0]
        overlay = SAMHelper.draw_mask_overlay(image_rgb, first_round_mask)
        if case_name == "b":
            overlay = _seed_masks_overlay(overlay, [_ensure_binary_mask(prompt_mask)], alpha=0.35)
        overlay = SAMHelper.draw_points(overlay, points_xy)
        save_name = f"{stem}_case{case_name}_missing{float(missing_ratio):.2f}.png"
        _write_rgb(os.path.join(out_dir, save_name), overlay)

    def _save_point_refine_case_binary_pair(
        self,
        image_rgb: np.ndarray,
        before_prob: np.ndarray,
        after_prob: np.ndarray,
        missing_ratio: float,
        filename: str,
        sample_meta: Optional[Dict[str, object]],
        sample_idx: int,
        epoch: int,
    ) -> None:
        case_name = "b" if float(missing_ratio) >= self.refine_missing_ratio_thresh else "a"
        self._save_pseudo_label_binary(
            image_rgb,
            before_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=f"pseudo_labels_point_refine_case_{case_name}_before_binary",
        )
        self._save_pseudo_label_binary(
            image_rgb,
            after_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=f"pseudo_labels_point_refine_case_{case_name}_after_binary",
        )

    def _save_point_refine_case_after(
        self,
        image_rgb: np.ndarray,
        after_prob: np.ndarray,
        points_xy: np.ndarray,
        prompt_mask: np.ndarray,
        missing_ratio: float,
        filename: str,
        epoch: int,
    ) -> None:
        if float(missing_ratio) < self.refine_missing_ratio_thresh:
            return
        out_dir = self._make_out_dir("point_refine_case_b_after", epoch)
        if out_dir is None:
            return
        stem = os.path.splitext(filename)[0]
        after_mask = _threshold_prob_map(after_prob, self.mask_prob_threshold)
        overlay = SAMHelper.draw_mask_overlay(image_rgb, after_mask)
        overlay = _seed_masks_overlay(overlay, [_ensure_binary_mask(prompt_mask)], alpha=0.35)
        overlay = SAMHelper.draw_points(overlay, points_xy)
        save_name = f"{stem}_caseb_after_missing{float(missing_ratio):.2f}.png"
        _write_rgb(os.path.join(out_dir, save_name), overlay)

    def _save_point_refine_heat_iou_drop(
        self,
        image_rgb: np.ndarray,
        before_prob: np.ndarray,
        after_prob: np.ndarray,
        heat_iou_mask: np.ndarray,
        before_points_xy: np.ndarray,
        after_points_xy: np.ndarray,
        prompt_mask: np.ndarray,
        missing_ratio: float,
        filename: str,
        epoch: int,
        result_name: str = "base",
    ) -> None:
        if float(missing_ratio) < self.refine_missing_ratio_thresh:
            return
        before_mask = _threshold_prob_map(before_prob, self.mask_prob_threshold)
        after_mask = _threshold_prob_map(after_prob, self.mask_prob_threshold)
        heat_iou_mask = _ensure_binary_mask(heat_iou_mask)
        before_heat_iou = _compute_iou(before_mask, heat_iou_mask)
        after_heat_iou = _compute_iou(after_mask, heat_iou_mask)
        if after_heat_iou >= before_heat_iou:
            return
        out_dir = self._make_out_dir("point_refine_case_b_heat_iou_drop", epoch)
        if out_dir is None:
            return

        stem = os.path.splitext(filename)[0]
        before_overlay = SAMHelper.draw_mask_overlay(image_rgb, before_mask)
        before_overlay = _seed_masks_overlay(before_overlay, [_ensure_binary_mask(prompt_mask)], alpha=0.35)
        before_overlay = SAMHelper.draw_points(before_overlay, before_points_xy)
        after_overlay = SAMHelper.draw_mask_overlay(image_rgb, after_mask)
        after_overlay = _seed_masks_overlay(after_overlay, [_ensure_binary_mask(prompt_mask)], alpha=0.35)
        after_overlay = SAMHelper.draw_points(after_overlay, after_points_xy)
        comparison = np.concatenate([before_overlay, after_overlay], axis=1)
        save_name = (
            f"{stem}_caseb_heatdrop"
            f"_{result_name}"
            f"_before{before_heat_iou:.2f}"
            f"_after{after_heat_iou:.2f}"
            f"_missing{float(missing_ratio):.2f}.png"
        )
        _write_rgb(os.path.join(out_dir, save_name), comparison)

    def _save_points_heatmap_overlay(
        self,
        image_rgb: np.ndarray,
        attn_prob: np.ndarray,
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        filename: str,
        epoch: int,
        prefix: str = "points_attn_heatmap",
    ) -> None:
        out_dir = self._make_out_dir(prefix, epoch)
        if out_dir is None:
            return
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
        prefix: str = "points_sam_seg",
        negative_points_xy: Optional[np.ndarray] = None,
    ) -> None:
        if mask_orig is None:
            return
        out_dir = self._make_out_dir(prefix, epoch)
        if out_dir is None:
            return
        stem = os.path.splitext(filename)[0]
        overlay_orig = SAMHelper.draw_mask_overlay(image_rgb, mask_orig)
        if box_xyxy is not None:
            overlay_orig = SAMHelper.draw_box(overlay_orig, box_xyxy)
        overlay_orig = SAMHelper.draw_points(overlay_orig, points_xy)
        overlay_orig = SAMHelper.draw_negative_points(overlay_orig, negative_points_xy)
        _write_rgb(os.path.join(out_dir, f"{stem}.png"), overlay_orig)

    def _save_affinity_split_result(
        self,
        image_rgb: np.ndarray,
        prompt_mask: np.ndarray,
        seed_masks: List[np.ndarray],
        points_xy: np.ndarray,
        superpixel_label_map: Optional[np.ndarray],
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
        superpixel_overlay_dir = self._make_out_dir("affinity_superpixel_overlay", epoch)

        prompt_mask = _ensure_binary_mask(prompt_mask)
        seed_masks = [_ensure_binary_mask(seed_mask) for seed_mask in seed_masks if int(_ensure_binary_mask(seed_mask).sum()) > 0]
        label_rgb = _seed_masks_to_label_rgb(seed_masks, image_rgb.shape[:2])
        overlay = _seed_masks_overlay(image_rgb, seed_masks, alpha=0.45)
        overlay = SAMHelper.draw_points(overlay, points_xy)
        union_mask = _seed_masks_to_union(seed_masks, image_rgb.shape[:2])

        if prompt_dir is not None:
            cv2.imwrite(os.path.join(prompt_dir, f"{stem}.png"), prompt_mask.astype(np.uint8) * 255)
        if union_dir is not None:
            cv2.imwrite(os.path.join(union_dir, f"{stem}.png"), union_mask.astype(np.uint8) * 255)
        if label_dir is not None:
            _write_rgb(os.path.join(label_dir, f"{stem}.png"), label_rgb)
        if overlay_dir is not None:
            _write_rgb(os.path.join(overlay_dir, f"{stem}.png"), overlay)
        if (
            superpixel_overlay_dir is not None
            and superpixel_label_map is not None
            and int((np.asarray(superpixel_label_map) > 0).sum()) > 0
        ):
            superpixel_overlay = _draw_label_grid_overlay(
                image_rgb,
                np.asarray(superpixel_label_map, dtype=np.int32),
                color=(0, 255, 0),
                thickness=1,
            )
            _write_rgb(os.path.join(superpixel_overlay_dir, f"{stem}.png"), superpixel_overlay)

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
        negative_points_by_point: Optional[Dict[int, np.ndarray]] = None,
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

        positive_group_starts = sorted(point_idx for point_idx in candidates_by_point.keys() if point_idx >= 0)

        def _prompt_group_bounds(point_idx: int) -> Tuple[int, int]:
            if point_idx < 0 or point_idx >= len(points_xy):
                return point_idx, point_idx
            next_starts = [start for start in positive_group_starts if start > point_idx]
            group_end = next_starts[0] if next_starts else len(points_xy)
            group_end = max(point_idx + 1, min(group_end, len(points_xy)))
            return point_idx, group_end

        def _prompt_group_tag(point_idx: int) -> str:
            if point_idx < 0:
                return "box"
            group_start, group_end = _prompt_group_bounds(point_idx)
            if group_end <= group_start + 1:
                return f"p{point_idx + 1}"
            return f"p{group_start + 1}-p{group_end}"

        for output_prefix in output_prefixes:
            out_dir = self._make_out_dir(output_prefix, epoch)
            if out_dir is None:
                continue
            for point_idx, cand_list in candidates_by_point.items():
                prompt_tag = _prompt_group_tag(point_idx)
                point_dir = os.path.join(out_dir, f"{stem}_{prompt_tag}")
                os.makedirs(point_dir, exist_ok=True)

                for cand_idx, candidate in enumerate(cand_list, start=1):
                    mask_orig = candidate.mask_orig if candidate.mask_orig is not None else _resize_mask(candidate.mask, heat_iou_mask.shape)
                    overlay = SAMHelper.draw_mask_overlay(image_rgb, mask_orig)
                    if point_idx < 0 and box_xyxy is not None:
                        overlay = SAMHelper.draw_box(overlay, box_xyxy)
                    elif point_idx < len(points_xy):
                        group_start, group_end = _prompt_group_bounds(point_idx)
                        overlay = SAMHelper.draw_points(
                            overlay,
                            points_xy[group_start:group_end].astype(np.float32),
                        )
                        if negative_points_by_point is not None and point_idx in negative_points_by_point:
                            overlay = SAMHelper.draw_negative_points(overlay, negative_points_by_point[point_idx])
                    save_name = (
                        f"{stem}_{prompt_tag}_c{cand_idx}"
                        f"_samiou{candidate.score:.2f}"
                        f"_fg_iou{candidate.fg_iou:.2f}"
                        f"_heat_iou{candidate.heat_iou:.2f}"
                        f"_bg_iou{candidate.bg_iou:.2f}"
                        f"_uncertain_area{uncertain_area_ratio:.2f}.png"
                    )
                    _write_rgb(os.path.join(point_dir, save_name), overlay)

    def _save_background_point_candidates(
        self,
        image_rgb: np.ndarray,
        raw_candidates: List[SAMCandidate],
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        heat_iou_mask: np.ndarray,
        bg_mask: np.ndarray,
        heat_iou_thresh: float,
        bg_iou_thresh: float,
        uncertain_area_ratio: float,
        filename: str,
        epoch: int,
        output_prefix: str = "point_bg_candidates",
        score_thresh: float = 0.7,
        fg_iou_masks_by_point: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        if not raw_candidates:
            return
        out_dir = self._make_out_dir(output_prefix, epoch)
        if out_dir is None:
            return

        stem = os.path.splitext(filename)[0]
        heat_iou_mask = _ensure_binary_mask(heat_iou_mask)
        bg_mask = _ensure_binary_mask(bg_mask)
        all_candidates_by_point: Dict[int, List[SAMCandidate]] = {}
        for candidate in raw_candidates:
            mask_orig = candidate.mask_orig if candidate.mask_orig is not None else _resize_mask(candidate.mask, bg_mask.shape)
            candidate.mask_orig = mask_orig
            candidate.filter_heat_iou = _compute_iou(mask_orig, heat_iou_mask)
            candidate.filter_bg_iou = _compute_iou(mask_orig, bg_mask)
            if fg_iou_masks_by_point is not None and candidate.point_idx in fg_iou_masks_by_point:
                candidate.fg_iou = _compute_iou(mask_orig, fg_iou_masks_by_point[candidate.point_idx])
            all_candidates_by_point.setdefault(candidate.point_idx, []).append(candidate)

        candidates_by_point: Dict[int, List[SAMCandidate]] = {}
        for point_idx, cand_list in all_candidates_by_point.items():
            has_background_like_candidate = any(
                float(candidate.score) > float(score_thresh)
                and float(candidate.filter_bg_iou) > float(bg_iou_thresh)
                and float(candidate.filter_heat_iou) < float(heat_iou_thresh)
                for candidate in cand_list
            )
            if has_background_like_candidate:
                candidates_by_point[point_idx] = cand_list
        if not candidates_by_point:
            return

        positive_group_starts = sorted(point_idx for point_idx in all_candidates_by_point.keys() if point_idx >= 0)

        def _prompt_group_bounds(point_idx: int) -> Tuple[int, int]:
            if point_idx < 0 or point_idx >= len(points_xy):
                return point_idx, point_idx
            next_starts = [start for start in positive_group_starts if start > point_idx]
            group_end = next_starts[0] if next_starts else len(points_xy)
            group_end = max(point_idx + 1, min(group_end, len(points_xy)))
            return point_idx, group_end

        def _prompt_group_tag(point_idx: int) -> str:
            if point_idx < 0:
                return "box"
            group_start, group_end = _prompt_group_bounds(point_idx)
            if group_end <= group_start + 1:
                return f"p{point_idx + 1}"
            return f"p{group_start + 1}-p{group_end}"

        for point_idx, cand_list in candidates_by_point.items():
            prompt_tag = _prompt_group_tag(point_idx)
            point_dir = os.path.join(out_dir, f"{stem}_{prompt_tag}")
            os.makedirs(point_dir, exist_ok=True)
            for cand_idx, candidate in enumerate(cand_list, start=1):
                mask_orig = candidate.mask_orig if candidate.mask_orig is not None else _resize_mask(candidate.mask, bg_mask.shape)
                overlay = SAMHelper.draw_mask_overlay(image_rgb, mask_orig)
                if point_idx < 0 and box_xyxy is not None:
                    overlay = SAMHelper.draw_box(overlay, box_xyxy)
                elif point_idx < len(points_xy):
                    group_start, group_end = _prompt_group_bounds(point_idx)
                    overlay = SAMHelper.draw_points(overlay, points_xy[group_start:group_end].astype(np.float32))
                save_name = (
                    f"{stem}_{prompt_tag}_c{cand_idx}"
                    f"_samiou{candidate.score:.2f}"
                    f"_fg_iou{candidate.fg_iou:.2f}"
                    f"_heat_iou{candidate.heat_iou:.2f}"
                    f"_bg_iou{candidate.bg_iou:.2f}"
                    f"_filter_heat_iou{candidate.filter_heat_iou:.2f}"
                    f"_filter_bg_iou{candidate.filter_bg_iou:.2f}"
                    f"_uncertain_area{uncertain_area_ratio:.2f}.png"
                )
                _write_rgb(os.path.join(point_dir, save_name), overlay)

    def _save_debug_binary_mask(
        self,
        mask: np.ndarray,
        filename: str,
        sample_meta: Optional[Dict[str, object]],
        sample_idx: int,
        epoch: int,
        prefix: str,
    ) -> None:
        out_dir = self._make_out_dir(prefix, epoch)
        if out_dir is None:
            return
        stem = os.path.splitext(filename)[0]
        mask = _ensure_binary_mask(mask)

        flipped = _sample_meta_value(sample_meta, "flipped", sample_idx, 0)
        orig_h = _sample_meta_value(sample_meta, "orig_h", sample_idx, mask.shape[0])
        orig_w = _sample_meta_value(sample_meta, "orig_w", sample_idx, mask.shape[1])

        if flipped:
            mask = np.ascontiguousarray(mask[:, ::-1])
        if mask.shape != (orig_h, orig_w):
            mask = _resize_mask(mask, (orig_h, orig_w))

        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), mask.astype(np.uint8) * 255)

    def _save_large_target_outputs(
        self,
        image_rgb: np.ndarray,
        final_prob: np.ndarray,
        uncertain_mask: np.ndarray,
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        uncertain_area_ratio: float,
        filename: str,
        sample_meta: Optional[Dict[str, object]],
        sample_idx: int,
        epoch: int,
    ) -> None:
        final_mask = _threshold_prob_map(final_prob, self.mask_prob_threshold)
        stem = os.path.splitext(filename)[0]

        seg_dir = self._make_out_dir("large_target_sam_seg", epoch)
        if seg_dir is not None:
            overlay = SAMHelper.draw_mask_overlay(image_rgb, final_mask)
            if box_xyxy is not None:
                overlay = SAMHelper.draw_box(overlay, box_xyxy)
            overlay = SAMHelper.draw_points(overlay, points_xy)
            save_name = f"{stem}_uncertain_area{float(uncertain_area_ratio):.2f}.png"
            _write_rgb(os.path.join(seg_dir, save_name), overlay)

        uncertain_overlay_dir = self._make_out_dir("large_target_uncertain_overlay", epoch)
        if uncertain_overlay_dir is not None:
            overlay = _seed_masks_overlay(image_rgb, [_ensure_binary_mask(uncertain_mask)], alpha=0.35)
            if box_xyxy is not None:
                overlay = SAMHelper.draw_box(overlay, box_xyxy)
            overlay = SAMHelper.draw_points(overlay, points_xy)
            save_name = f"{stem}_uncertain_area{float(uncertain_area_ratio):.2f}.png"
            _write_rgb(os.path.join(uncertain_overlay_dir, save_name), overlay)

        self._save_pseudo_label_binary(
            image_rgb,
            final_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix="pseudo_labels_large_target_binary",
        )
        self._save_debug_binary_mask(
            uncertain_mask,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix="large_target_uncertain_binary",
        )

    def _save_final_candidates(
        self,
        image_rgb: np.ndarray,
        candidates: List[SAMCandidate],
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        uncertain_area_ratio: float,
        filename: str,
        epoch: int,
        prefix: str = "sam_final_candidates",
        point_group_starts: Optional[List[int]] = None,
        negative_points_by_point: Optional[Dict[int, np.ndarray]] = None,
    ) -> None:
        if not candidates:
            return
        stem = os.path.splitext(filename)[0]
        out_dir = self._make_out_dir(prefix, epoch)
        if out_dir is None:
            return
        sample_dir = os.path.join(out_dir, stem)
        os.makedirs(sample_dir, exist_ok=True)

        if point_group_starts is None:
            positive_group_starts = sorted({candidate.point_idx for candidate in candidates if candidate.point_idx >= 0})
        else:
            positive_group_starts = sorted(int(value) for value in point_group_starts if int(value) >= 0)

        def _prompt_group_bounds(point_idx: int) -> Tuple[int, int]:
            if point_idx < 0 or point_idx >= len(points_xy):
                return point_idx, point_idx
            next_starts = [start for start in positive_group_starts if start > point_idx]
            group_end = next_starts[0] if next_starts else len(points_xy)
            group_end = max(point_idx + 1, min(group_end, len(points_xy)))
            return point_idx, group_end

        for candidate in candidates:
            mask_orig = candidate.mask_orig if candidate.mask_orig is not None else candidate.mask
            overlay = SAMHelper.draw_mask_overlay(image_rgb, mask_orig)
            if candidate.point_idx < 0 and box_xyxy is not None:
                overlay = SAMHelper.draw_box(overlay, box_xyxy)
                prompt_tag = "box"
            else:
                group_start, group_end = _prompt_group_bounds(candidate.point_idx)
                if 0 <= group_start < group_end <= len(points_xy):
                    overlay = SAMHelper.draw_points(overlay, points_xy[group_start:group_end].astype(np.float32))
                    if negative_points_by_point is not None and candidate.point_idx in negative_points_by_point:
                        overlay = SAMHelper.draw_negative_points(overlay, negative_points_by_point[candidate.point_idx])
                    if group_end <= group_start + 1:
                        prompt_tag = f"p{group_start + 1}"
                    else:
                        prompt_tag = f"p{group_start + 1}-p{group_end}"
                else:
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
        if out_dir is None:
            return
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
        if out_dir is None:
            return
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

    def _save_mask_prompt_outputs(
        self,
        image_rgb: np.ndarray,
        raw_candidates: List[SAMCandidate],
        points_xy: np.ndarray,
        box_xyxy: Optional[np.ndarray],
        fallback_mask: np.ndarray,
        heat_iou_mask: np.ndarray,
        bg_mask: np.ndarray,
        metric_heat_iou_mask: np.ndarray,
        fg_iou_masks_by_point: Optional[Dict[int, np.ndarray]],
        heat_iou_thresh: float,
        bg_iou_thresh: float,
        uncertain_area_ratio: float,
        filename: str,
        sample_meta: Optional[Dict[str, object]],
        sample_idx: int,
        epoch: int,
        seg_prefix: str,
        candidate_prefix: str,
        pseudo_prefix: str,
        final_candidates_prefix: str,
        negative_points_by_point: Optional[Dict[int, np.ndarray]] = None,
        extra_heat_iou_mask: Optional[np.ndarray] = None,
        extra_heat_iou_thresh: Optional[float] = None,
    ) -> None:
        if not raw_candidates:
            return

        (
            candidates_by_point,
            kept_candidates,
            rule_a_candidates,
            rule_ab_candidates,
            final_prob,
            rule_a_prob,
            rule_ab_prob,
            final_mask,
            rule_a_mask,
            rule_ab_mask,
        ) = self._build_selected_candidate_outputs(
            raw_candidates,
            fallback_mask,
            heat_iou_mask,
            bg_mask,
            metric_heat_iou_mask,
            fg_iou_masks_by_point,
            heat_iou_thresh,
            bg_iou_thresh,
            extra_heat_iou_mask=extra_heat_iou_mask,
            extra_heat_iou_thresh=extra_heat_iou_thresh,
        )
        point_group_starts = sorted(fg_iou_masks_by_point.keys()) if fg_iou_masks_by_point else None

        self._save_points_seg_overlay(
            image_rgb,
            final_mask,
            points_xy,
            box_xyxy,
            filename,
            epoch,
            prefix=seg_prefix,
            negative_points_xy=_merge_prompt_points(negative_points_by_point),
        )
        self._save_points_seg_overlay(
            image_rgb,
            rule_a_mask,
            points_xy,
            box_xyxy,
            filename,
            epoch,
            prefix=f"{seg_prefix}_rule_a",
            negative_points_xy=_merge_prompt_points(negative_points_by_point),
        )
        self._save_points_seg_overlay(
            image_rgb,
            rule_ab_mask,
            points_xy,
            box_xyxy,
            filename,
            epoch,
            prefix=f"{seg_prefix}_rule_ab",
            negative_points_xy=_merge_prompt_points(negative_points_by_point),
        )
        self._save_per_point_candidates(
            image_rgb,
            raw_candidates,
            points_xy,
            box_xyxy,
            heat_iou_mask,
            bg_mask,
            uncertain_area_ratio,
            filename,
            epoch,
            output_prefixes=[candidate_prefix],
            fg_iou_masks_by_point=fg_iou_masks_by_point,
            negative_points_by_point=negative_points_by_point,
            metric_heat_iou_mask=metric_heat_iou_mask,
        )
        self._save_pseudo_label_grayscale(
            final_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=pseudo_prefix,
        )
        self._save_pseudo_label_grayscale(
            rule_a_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=f"{pseudo_prefix}_rule_a",
        )
        self._save_pseudo_label_grayscale(
            rule_ab_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=f"{pseudo_prefix}_rule_ab",
        )
        self._save_pseudo_label_binary(
            image_rgb,
            final_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=f"{pseudo_prefix}_binary",
        )
        self._save_pseudo_label_binary(
            image_rgb,
            rule_a_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=f"{pseudo_prefix}_rule_a_binary",
        )
        self._save_pseudo_label_binary(
            image_rgb,
            rule_ab_prob,
            filename,
            sample_meta,
            sample_idx,
            epoch,
            prefix=f"{pseudo_prefix}_rule_ab_binary",
        )
        self._save_final_candidates(
            image_rgb,
            kept_candidates,
            points_xy,
            box_xyxy,
            uncertain_area_ratio,
            filename,
            epoch,
            prefix=final_candidates_prefix,
            point_group_starts=point_group_starts,
            negative_points_by_point=negative_points_by_point,
        )
        self._save_final_candidates(
            image_rgb,
            rule_a_candidates,
            points_xy,
            box_xyxy,
            uncertain_area_ratio,
            filename,
            epoch,
            prefix=f"{final_candidates_prefix}_rule_a",
            point_group_starts=point_group_starts,
            negative_points_by_point=negative_points_by_point,
        )
        self._save_final_candidates(
            image_rgb,
            rule_ab_candidates,
            points_xy,
            box_xyxy,
            uncertain_area_ratio,
            filename,
            epoch,
            prefix=f"{final_candidates_prefix}_rule_ab",
            point_group_starts=point_group_starts,
            negative_points_by_point=negative_points_by_point,
        )

    def _record_failure(self, filename: str, epoch: int) -> None:
        self.failed_names.setdefault(epoch, []).append(filename)

    def finalize_epoch(self, epoch: int) -> None:
        fail_dir = self._make_out_dir("sam_failures", epoch)
        if fail_dir is None:
            return
        log_path = os.path.join(fail_dir, "failures.txt")
        with open(log_path, "w", encoding="utf-8") as handle:
            for name in self.failed_names.get(epoch, []):
                handle.write(f"{name}\n")
