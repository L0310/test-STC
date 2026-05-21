from typing import Dict, List, Optional, Tuple

import numpy as np

from .image_ops import (
    _compute_iou,
    _ensure_binary_mask,
    _logits_to_prob_map,
    _resize_mask,
    _resize_score_map,
    _threshold_prob_map,
)
from .types import SAMCandidate


class SAMTrainCandidateSelectionMixin:
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
        fg_candidates_by_point: Optional[Dict[int, List[SAMCandidate]]] = None,
        fg_score_thresh: float = 0.9,
        fg_iou_thresh: Optional[float] = None,
        fg_bg_iou_thresh: float = 0.15,
        include_fg_iou: bool = False,
    ) -> List[SAMCandidate]:
        selected_candidates: List[SAMCandidate] = []
        seen_candidate_ids = set()

        def _add_candidate(candidate: SAMCandidate) -> None:
            candidate_id = id(candidate)
            if candidate_id in seen_candidate_ids:
                return
            seen_candidate_ids.add(candidate_id)
            selected_candidates.append(candidate)

        point_indices = set(valid_candidates_by_point.keys())
        if include_fg_iou and fg_candidates_by_point is not None:
            point_indices.update(fg_candidates_by_point.keys())

        for point_idx in sorted(point_indices):
            point_candidates = valid_candidates_by_point.get(point_idx, [])
            if point_candidates:
                best_heat_candidate = max(
                    point_candidates,
                    key=lambda item: (float(item.filter_heat_iou), float(item.score)),
                )
                _add_candidate(best_heat_candidate)

                best_score_candidate = max(
                    point_candidates,
                    key=lambda item: (float(item.score), float(item.filter_heat_iou)),
                )
                _add_candidate(best_score_candidate)

            if not include_fg_iou or fg_candidates_by_point is None or point_idx not in fg_candidates_by_point:
                continue
            fg_candidates = fg_candidates_by_point[point_idx]
            if not fg_candidates:
                continue
            valid_fg_candidates = [
                candidate for candidate in fg_candidates
                if float(candidate.score) > float(fg_score_thresh)
                and float(candidate.filter_bg_iou) <= float(fg_bg_iou_thresh)
                and (fg_iou_thresh is None or float(candidate.fg_iou) > float(fg_iou_thresh))
            ]
            if not valid_fg_candidates:
                continue
            best_fg_candidate = max(
                valid_fg_candidates,
                key=lambda item: (float(item.fg_iou), float(item.score)),
            )
            _add_candidate(best_fg_candidate)

        return selected_candidates

    def _select_best_heat_iou_candidates(
        self,
        candidates_by_point: Dict[int, List[SAMCandidate]],
        heat_iou_mask: np.ndarray,
        bg_mask: np.ndarray,
        heat_iou_thresh: float,
        bg_iou_thresh: float,
    ) -> List[SAMCandidate]:
        heat_iou_mask = _ensure_binary_mask(heat_iou_mask)
        bg_mask = _ensure_binary_mask(bg_mask)
        selected_candidates: List[SAMCandidate] = []

        for point_idx in sorted(candidates_by_point.keys()):
            valid_candidates: List[Tuple[float, SAMCandidate]] = []
            for candidate in candidates_by_point[point_idx]:
                if candidate.mask_orig is None:
                    continue
                candidate_heat_iou = _compute_iou(candidate.mask_orig, heat_iou_mask)
                candidate_bg_iou = _compute_iou(candidate.mask_orig, bg_mask)
                if (
                    candidate.score >= self.score_thresh
                    and candidate_heat_iou >= float(heat_iou_thresh)
                    and candidate_bg_iou <= float(bg_iou_thresh)
                    and self._candidate_area_ratio(candidate) <= self.area_limit
                ):
                    valid_candidates.append((float(candidate_heat_iou), candidate))
            if not valid_candidates:
                continue
            _, best_candidate = max(
                valid_candidates,
                key=lambda item: (item[0], float(item[1].score)),
            )
            selected_candidates.append(best_candidate)

        return selected_candidates

    @staticmethod
    def _append_unique_candidates(
        base_candidates: List[SAMCandidate],
        extra_candidates: List[SAMCandidate],
    ) -> List[SAMCandidate]:
        if not extra_candidates:
            return base_candidates
        merged_candidates = list(base_candidates)
        seen_candidate_ids = {id(candidate) for candidate in merged_candidates}
        for candidate in extra_candidates:
            candidate_id = id(candidate)
            if candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate_id)
            merged_candidates.append(candidate)
        return merged_candidates

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

    def _build_selected_candidate_outputs(
        self,
        raw_candidates: List[SAMCandidate],
        fallback_mask: np.ndarray,
        heat_iou_mask: np.ndarray,
        bg_mask: np.ndarray,
        metric_heat_iou_mask: np.ndarray,
        fg_iou_masks_by_point: Optional[Dict[int, np.ndarray]],
        heat_iou_thresh: float,
        bg_iou_thresh: float,
        extra_heat_iou_mask: Optional[np.ndarray] = None,
        extra_heat_iou_thresh: Optional[float] = None,
    ) -> Tuple[
        Dict[int, List[SAMCandidate]],
        List[SAMCandidate],
        List[SAMCandidate],
        List[SAMCandidate],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        candidates_by_point = self._prepare_candidates_by_point(
            raw_candidates,
            heat_iou_mask,
            bg_mask,
            fg_iou_masks_by_point=fg_iou_masks_by_point,
            metric_heat_iou_mask=metric_heat_iou_mask,
        )
        valid_candidates_by_point = self._get_valid_candidates_by_point(
            candidates_by_point,
            heat_iou_thresh=heat_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
        )
        kept_candidates = self._select_best_candidates(valid_candidates_by_point)
        rule_a_candidates = self._select_best_candidates(
            valid_candidates_by_point,
            fg_candidates_by_point=candidates_by_point,
            fg_score_thresh=0.9,
            fg_bg_iou_thresh=bg_iou_thresh,
            include_fg_iou=True,
        )
        rule_ab_candidates = self._select_best_candidates(
            valid_candidates_by_point,
            fg_candidates_by_point=candidates_by_point,
            fg_score_thresh=0.9,
            fg_iou_thresh=0.15,
            fg_bg_iou_thresh=bg_iou_thresh,
            include_fg_iou=True,
        )
        if extra_heat_iou_mask is not None:
            extra_heat_candidates = self._select_best_heat_iou_candidates(
                candidates_by_point,
                extra_heat_iou_mask,
                bg_mask,
                heat_iou_thresh=self.heat_iou_thresh if extra_heat_iou_thresh is None else float(extra_heat_iou_thresh),
                bg_iou_thresh=bg_iou_thresh,
            )
            kept_candidates = self._append_unique_candidates(kept_candidates, extra_heat_candidates)
            rule_a_candidates = self._append_unique_candidates(rule_a_candidates, extra_heat_candidates)
            rule_ab_candidates = self._append_unique_candidates(rule_ab_candidates, extra_heat_candidates)
        final_prob = self._merge_candidate_prob_maps(kept_candidates, fallback_mask)
        rule_a_prob = self._merge_candidate_prob_maps(rule_a_candidates, fallback_mask)
        rule_ab_prob = self._merge_candidate_prob_maps(rule_ab_candidates, fallback_mask)
        final_mask = _threshold_prob_map(final_prob, self.mask_prob_threshold)
        rule_a_mask = _threshold_prob_map(rule_a_prob, self.mask_prob_threshold)
        rule_ab_mask = _threshold_prob_map(rule_ab_prob, self.mask_prob_threshold)
        return (
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
        )
