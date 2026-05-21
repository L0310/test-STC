from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .image_ops import (
    _ensure_binary_mask,
    _ensure_uint8_rgb,
    _logits_to_prob_map,
    _resize_mask,
    _threshold_prob_map,
)
from .prompt_points import _scale_points
from .types import SAMCandidate


class SAMTrainRunnerMixin:
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

    def _run_sam_multi_positive(
        self,
        image_rgb: np.ndarray,
        points_xy: np.ndarray,
        negative_points_xy: Optional[np.ndarray] = None,
        point_idx: int = -1,
        set_image: bool = True,
    ) -> List[SAMCandidate]:
        if points_xy.shape[0] == 0:
            return []

        if set_image:
            self.predictor.set_image(np.ascontiguousarray(_ensure_uint8_rgb(image_rgb)))
        prompt_points = points_xy.astype(np.float32)
        point_labels = np.ones((prompt_points.shape[0],), dtype=np.int32)
        if negative_points_xy is not None and np.asarray(negative_points_xy).size > 0:
            neg_points = np.asarray(negative_points_xy, dtype=np.float32).reshape(-1, 2)
            prompt_points = np.concatenate([prompt_points, neg_points], axis=0)
            point_labels = np.concatenate([point_labels, np.zeros((neg_points.shape[0],), dtype=np.int32)], axis=0)
        mask_logits, scores, _ = self.predictor.predict(
            point_coords=prompt_points,
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

    def _build_soft_mask_prompt(self, seed_mask: np.ndarray, mask_size: int = 256) -> np.ndarray:
        seed_mask = _ensure_binary_mask(seed_mask).astype(np.uint8)
        if int(seed_mask.sum()) <= 0:
            return np.zeros((1, int(mask_size), int(mask_size)), dtype=np.float32)

        original_size = getattr(self.predictor, "original_size", seed_mask.shape)
        input_size = getattr(self.predictor, "input_size", None)
        image_size = int(getattr(self.predictor.model.image_encoder, "img_size", 1024))

        original_h, original_w = int(original_size[0]), int(original_size[1])
        if seed_mask.shape != (original_h, original_w):
            seed_mask = cv2.resize(seed_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        seed_mask = _ensure_binary_mask(seed_mask)
        distance = cv2.distanceTransform(seed_mask.astype(np.uint8), cv2.DIST_L2, 5)
        max_distance = float(distance.max()) if distance.size > 0 else 0.0
        if max_distance > 0.0:
            normalized_distance = np.clip(distance / max_distance, 0.0, 1.0)
            soft_mask = np.where(
                seed_mask > 0,
                1.0 + (self.mask_prompt_fg_logit - 1.0) * normalized_distance,
                0.0,
            )
        else:
            soft_mask = seed_mask.astype(np.float32) * self.mask_prompt_fg_logit

        if input_size is not None:
            input_h, input_w = int(input_size[0]), int(input_size[1])
            transformed_mask = cv2.resize(soft_mask, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            padded_mask = np.zeros((image_size, image_size), dtype=np.float32)
            padded_mask[:input_h, :input_w] = transformed_mask
            mask_prompt = cv2.resize(
                padded_mask,
                (int(mask_size), int(mask_size)),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            mask_prompt = cv2.resize(
                soft_mask,
                (int(mask_size), int(mask_size)),
                interpolation=cv2.INTER_LINEAR,
            )
        mask_prompt = np.clip(mask_prompt, 0.0, self.mask_prompt_fg_logit).astype(np.float32)
        return mask_prompt[None, :, :]

    def _run_sam_multi_positive_with_mask_prompt(
        self,
        image_rgb: np.ndarray,
        points_xy: np.ndarray,
        seed_mask: np.ndarray,
        negative_points_xy: Optional[np.ndarray] = None,
        point_idx: int = -1,
        set_image: bool = True,
    ) -> List[SAMCandidate]:
        if points_xy.shape[0] == 0:
            return []

        if set_image:
            self.predictor.set_image(np.ascontiguousarray(_ensure_uint8_rgb(image_rgb)))
        prompt_points = points_xy.astype(np.float32)
        point_labels = np.ones((prompt_points.shape[0],), dtype=np.int32)
        if negative_points_xy is not None and np.asarray(negative_points_xy).size > 0:
            neg_points = np.asarray(negative_points_xy, dtype=np.float32).reshape(-1, 2)
            prompt_points = np.concatenate([prompt_points, neg_points], axis=0)
            point_labels = np.concatenate([point_labels, np.zeros((neg_points.shape[0],), dtype=np.int32)], axis=0)
        mask_input = self._build_soft_mask_prompt(seed_mask)
        mask_logits, scores, _ = self.predictor.predict(
            point_coords=prompt_points,
            point_labels=point_labels,
            mask_input=mask_input,
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

    @staticmethod
    def _candidate_group_bounds(
        points_xy: np.ndarray,
        point_group_starts: List[int],
        point_idx: int,
    ) -> Tuple[int, int]:
        if point_idx < 0 or point_idx >= len(points_xy):
            return point_idx, point_idx
        starts = sorted(int(value) for value in point_group_starts if int(value) >= 0)
        next_starts = [start for start in starts if start > point_idx]
        group_end = next_starts[0] if next_starts else len(points_xy)
        group_end = max(point_idx + 1, min(group_end, len(points_xy)))
        return point_idx, group_end

    def _run_point_refine_candidates(
        self,
        image_rgb_aug: np.ndarray,
        source_candidates: List[SAMCandidate],
        points_xy: np.ndarray,
        prompt_mask: np.ndarray,
        point_group_starts: List[int],
        global_missing_mask: Optional[np.ndarray] = None,
        global_missing_ratio: Optional[float] = None,
        negative_points_by_point: Optional[Dict[int, np.ndarray]] = None,
        use_negative: bool = False,
    ) -> Tuple[List[SAMCandidate], np.ndarray, Dict[int, np.ndarray], Dict[int, np.ndarray], List[Tuple[SAMCandidate, str, float]]]:
        refined_candidates: List[SAMCandidate] = []
        refine_points_groups: List[np.ndarray] = []
        refine_fg_masks_by_point: Dict[int, np.ndarray] = {}
        refine_negative_points_by_point: Dict[int, np.ndarray] = {}
        case_records: List[Tuple[SAMCandidate, str, float]] = []
        seen_source_ids = set()
        prompt_mask = _ensure_binary_mask(prompt_mask)
        if global_missing_mask is None:
            global_missing_mask = np.zeros_like(prompt_mask, dtype=np.uint8)
        else:
            global_missing_mask = _ensure_binary_mask(global_missing_mask)
            if global_missing_mask.shape != prompt_mask.shape:
                global_missing_mask = _resize_mask(global_missing_mask, prompt_mask.shape)
        if global_missing_ratio is None:
            global_missing_ratio = float(global_missing_mask.sum()) / float(max(1, int(prompt_mask.sum())))
        is_case_b = float(global_missing_ratio) >= self.refine_missing_ratio_thresh
        if not is_case_b:
            return refined_candidates, np.zeros((0, 2), dtype=np.float32), refine_fg_masks_by_point, refine_negative_points_by_point, case_records

        for source_candidate in source_candidates:
            source_id = id(source_candidate)
            if source_id in seen_source_ids:
                continue
            seen_source_ids.add(source_id)
            if source_candidate.mask_orig is None:
                continue
            group_start, group_end = self._candidate_group_bounds(points_xy, point_group_starts, source_candidate.point_idx)
            if not (0 <= group_start < group_end <= len(points_xy)):
                continue

            refine_seed_mask = prompt_mask.copy()
            refine_points_xy = points_xy[group_start:group_end].astype(np.float32)
            refine_point_idx = sum(group.shape[0] for group in refine_points_groups)
            refine_points_groups.append(refine_points_xy)
            refine_fg_masks_by_point[refine_point_idx] = refine_seed_mask

            negative_points = np.zeros((0, 2), dtype=np.float32)
            if use_negative and negative_points_by_point is not None and source_candidate.point_idx in negative_points_by_point:
                negative_points = np.asarray(negative_points_by_point[source_candidate.point_idx], dtype=np.float32).reshape(-1, 2)
                if negative_points.shape[0] > self.neg_points_per_component:
                    negative_points = negative_points[:self.neg_points_per_component]
                if negative_points.size > 0:
                    refine_negative_points_by_point[refine_point_idx] = negative_points

            refine_points_aug = _scale_points(refine_points_xy, prompt_mask.shape, image_rgb_aug.shape[:2])
            negative_points_aug = _scale_points(negative_points, prompt_mask.shape, image_rgb_aug.shape[:2])
            refined_candidates.extend(
                self._run_sam_multi_positive_with_mask_prompt(
                    image_rgb_aug,
                    refine_points_aug,
                    refine_seed_mask,
                    negative_points_xy=negative_points_aug if use_negative else None,
                    point_idx=refine_point_idx,
                    set_image=False,
                )
            )
            case_records.append((source_candidate, "b", float(global_missing_ratio)))

        if not refine_points_groups:
            return refined_candidates, np.zeros((0, 2), dtype=np.float32), refine_fg_masks_by_point, refine_negative_points_by_point, case_records
        refine_points_all = np.concatenate(refine_points_groups, axis=0).astype(np.float32)
        return refined_candidates, refine_points_all, refine_fg_masks_by_point, refine_negative_points_by_point, case_records
