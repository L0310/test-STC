import os
from typing import List, Optional

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

from .image_ops import _ensure_binary_mask, _ensure_uint8_rgb, extract_red_mask_from_overlay
from .prompt_points import select_five_prompt_points
from .types import SAMCandidate, SAMResult

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
    def draw_negative_points(image_rgb: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
        image = _ensure_uint8_rgb(image_rgb).copy()
        if points_xy is None or np.asarray(points_xy).size == 0:
            return image
        for x, y in np.atleast_2d(points_xy):
            cv2.circle(image, (int(round(x)), int(round(y))), 5, (255, 0, 0), -1)
            cv2.circle(image, (int(round(x)), int(round(y))), 7, (255, 255, 255), 1)
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
