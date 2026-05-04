import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from segment_anything import SamPredictor, sam_model_registry


def _min_max_norm(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    tensor_min = tensor.amin(dim=(-2, -1), keepdim=True)
    tensor_max = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - tensor_min) / (tensor_max - tensor_min + eps)


def _resize_to_short_edge(
    img_1chw: torch.Tensor, short_edge: int
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize a single image so that the shorter edge equals ``short_edge``."""
    assert img_1chw.dim() == 4 and img_1chw.size(0) == 1, "expect [1,C,H,W]"
    _, _, h, w = img_1chw.shape
    if short_edge <= 0:
        return img_1chw, (h, w)
    min_edge = min(h, w)
    if min_edge == 0:
        return img_1chw, (h, w)
    scale = float(short_edge) / float(min_edge)
    if math.isclose(scale, 1.0, rel_tol=1e-4):
        return img_1chw, (h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = F.interpolate(img_1chw, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return resized, (new_h, new_w)


def _tensor_to_uint8(img_3chw: torch.Tensor) -> np.ndarray:
    assert img_3chw.dim() == 3 and img_3chw.size(0) == 3, "expect [3,H,W]"
    img = img_3chw.detach().cpu().clamp(0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img.permute(1, 2, 0).numpy()


def select_prompt_points(
    prob_map_1hw: torch.Tensor,
    *,
    percentile: float = 0.9,
    max_points: int = 3,
    smooth_kernel: int = 5,
    min_area_frac: float = 0.001,
    max_area_frac: float = 0.3,
) -> np.ndarray:
    """Pick multiple prompt points from an attention map."""
    assert prob_map_1hw.dim() == 3 and prob_map_1hw.size(0) == 1
    prob = prob_map_1hw.unsqueeze(0)  # [1,1,H,W]
    if smooth_kernel > 1:
        pad = smooth_kernel // 2
        prob = F.avg_pool2d(prob, kernel_size=smooth_kernel, stride=1, padding=pad)
    prob = prob.squeeze(0)  # [1,H,W]
    prob_np = prob.squeeze(0).detach().cpu().numpy()

    thr = np.quantile(prob_np, percentile)
    mask_high = prob_np >= thr
    h, w = prob_np.shape
    total_pixels = float(h * w)

    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_high.astype(np.uint8), connectivity=8
    )

    candidates: List[Tuple[float, float, float]] = []
    for idx in range(1, num_labels):
        area = stats[idx, cv2.CC_STAT_AREA]
        area_ratio = area / total_pixels
        if area_ratio < min_area_frac or area_ratio > max_area_frac:
            continue
        region_mask = labels == idx
        region_vals = prob_np * region_mask
        max_idx = int(np.argmax(region_vals))
        ry, rx = divmod(max_idx, w)
        candidates.append((float(prob_np[ry, rx]), float(rx), float(ry)))

    if not candidates:
        fallback_idx = int(np.argmax(prob_np))
        fy, fx = divmod(fallback_idx, w)
        return np.array([[float(fx), float(fy)]], dtype=np.float32)

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = np.array([[c[1], c[2]] for c in candidates[: max(1, max_points)]], dtype=np.float32)
    return selected


def remove_small_regions(mask: np.ndarray, area_thresh: float, mode: str) -> Tuple[np.ndarray, bool]:
    """Remove tiny holes or islands from a binary mask."""
    import cv2

    assert mode in {"holes", "islands"}
    area_thresh = max(0, int(area_thresh))
    if area_thresh == 0:
        return mask.astype(bool), False

    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(working_mask, connectivity=8)
    sizes = stats[:, -1][1:]
    small_regions = [i + 1 for i, size in enumerate(sizes) if size < area_thresh]
    if not small_regions:
        return mask.astype(bool), False

    fill_labels = [0] + small_regions
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        if not fill_labels and len(sizes) > 0:
            fill_labels = [int(np.argmax(sizes)) + 1]

    pruned = np.isin(regions, fill_labels)
    return pruned.astype(bool), True


@dataclass
class SAMCandidate:
    mask: np.ndarray
    iou: float
    point_idx: int
    heat_iou: float = 0.0


@dataclass
class SAMSampleResult:
    mask: Optional[np.ndarray]
    success: bool
    candidates: List[SAMCandidate]
    points_xy: np.ndarray


@dataclass
class SAMPointCandidate:
    mask: np.ndarray
    iou: float


@dataclass
class SAMVizRecord:
    image_uint8: np.ndarray
    aug_image_uint8: np.ndarray
    attn_prob: np.ndarray
    sam_mask: Optional[np.ndarray]
    points_xy: np.ndarray
    points_xy_aug: np.ndarray
    success: bool
    filename: str = ""
    per_point_candidates: List[List[SAMPointCandidate]] = field(default_factory=list)
    selected_candidates: List[SAMCandidate] = field(default_factory=list)


class SAMManager:
    """Utility wrapper that mirrors the behaviour of UnSAM but uses SAM predictor."""

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "use_sam_seed", True))
        self.device_type = str(getattr(cfg, "sam_device", "cuda"))
        self.model_type = str(getattr(cfg, "sam_model_type", "vit_h"))
        self.checkpoint = str(getattr(cfg, "sam_checkpoint", ""))

        self.short_edge = int(getattr(cfg, "sam_resize_short_edge", getattr(cfg, "unsam_resize_short_edge", 640)))
        self.prompt_percentile = float(getattr(cfg, "sam_prompt_percentile", getattr(cfg, "unsam_prompt_percentile", 0.9)))
        self.max_points = int(getattr(cfg, "sam_max_points", getattr(cfg, "unsam_max_points", 3)))
        self.min_area_frac = float(getattr(cfg, "sam_min_area_frac", getattr(cfg, "unsam_min_area_frac", 0.001)))
        self.max_area_frac = float(getattr(cfg, "sam_max_area_frac", getattr(cfg, "unsam_max_area_frac", 0.3)))
        self.smooth_kernel = int(getattr(cfg, "sam_smooth_kernel", getattr(cfg, "unsam_smooth_kernel", 5)))

        self.iou_thresh = float(getattr(cfg, "sam_score_thresh", getattr(cfg, "sam_score_thresh", 0.9)))
        self.candidate_iou_thresh = float(getattr(cfg, "sam_candidate_iou_thresh", getattr(cfg, "unsam_candidate_iou_thresh", 0.5)))
        self.overlap_ratio = float(getattr(cfg, "sam_overlap_ratio", getattr(cfg, "unsam_overlap_ratio", 0.9)))
        self.area_limit = float(getattr(cfg, "sam_area_limit", getattr(cfg, "unsam_area_limit", 0.85)))
        self.hole_scale = int(getattr(cfg, "sam_hole_scale", getattr(cfg, "unsam_hole_scale", 100)))
        self.island_scale = int(getattr(cfg, "sam_island_scale", getattr(cfg, "unsam_island_scale", 100)))
        self.max_masks = int(getattr(cfg, "sam_max_masks", getattr(cfg, "unsam_max_masks", 3)))
        self.fallback_iou = float(getattr(cfg, "sam_fallback_iou", getattr(cfg, "unsam_fallback_iou", 0.5)))
        self.multimask_output = bool(getattr(cfg, "sam_multimask_output", True))

        self.warmup_frac = float(getattr(cfg, "sam_warmup_frac", getattr(cfg, "unsam_warmup_frac", 0.1)))
        self.total_iters = int(getattr(cfg, "tot_iter", 13200))
        self.steps_per_epoch = int(getattr(cfg, "steps_per_epoch", 660))

        self.denorm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.denorm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        self.save_viz = bool(getattr(cfg, "sam_save_viz", getattr(cfg, "unsam_save_viz", False)))
        self.viz_root = str(getattr(cfg, "sam_viz_root", "./viz/SAM_point"))
        self.viz_prefix_attn = str(getattr(cfg, "sam_viz_prefix_attn", getattr(cfg, "unsam_viz_prefix_attn", "points_attn_heatmap")))
        self.viz_prefix_seg = str(getattr(cfg, "sam_viz_prefix_seg", getattr(cfg, "unsam_viz_prefix_seg", "points_sam_seg")))
        self.viz_prefix_per_point = str(getattr(cfg, "sam_viz_prefix_per_point", getattr(cfg, "unsam_viz_prefix_per_point", "point_candidates")))
        self.viz_prefix_pseudo = str(getattr(cfg, "sam_viz_prefix_pseudo", getattr(cfg, "unsam_viz_prefix_pseudo", "sam_pseudo")))
        self.viz_prefix_final = str(getattr(cfg, "sam_viz_prefix_final", getattr(cfg, "unsam_viz_prefix_final", "sam_final_candidates")))
        self.viz_prefix_fail = str(getattr(cfg, "sam_viz_prefix_fail", getattr(cfg, "unsam_viz_prefix_fail", "sam_failures")))
        self.viz_alpha = float(getattr(cfg, "sam_viz_alpha", getattr(cfg, "unsam_viz_alpha", 0.5)))
        self.viz_seg_thr = float(getattr(cfg, "sam_bin_thr", getattr(cfg, "unsam_bin_thr", 0.5)))
        self.point_save_iou_thr = float(getattr(cfg, "sam_point_save_iou_thr", getattr(cfg, "unsam_point_save_iou_thr", 0.6)))
        self.heat_iou_thresh = float(getattr(cfg, "sam_heat_iou_thresh", getattr(cfg, "unsam_heat_iou_thresh", 0.15)))
        self.attn_bin_mode = str(getattr(cfg, "sam_attn_bin_mode", getattr(cfg, "attn_bin_mode", "percentile")))
        self.attn_bin_thr = float(getattr(cfg, "sam_attn_bin_thr", getattr(cfg, "attn_bin_thr", 0.5)))
        self.attn_bin_q = float(getattr(cfg, "sam_attn_bin_q", getattr(cfg, "attn_bin_q", 0.9)))

        self._sam: Optional[SamPredictor] = None
        self._sam_device: Optional[str] = None

    def should_use(self, process: float) -> bool:
        if not self.enabled:
            return False
        warmup_steps = max(0, int(self.warmup_frac * self.total_iters))
        current_step = int(process * self.total_iters)
        return current_step >= warmup_steps

    def ensure_predictor(self):
        if self._sam is not None:
            return
        if not self.checkpoint:
            raise ValueError("sam_checkpoint must be set in cfg to use SAMManager.")

        model = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        if self.device_type != "cpu":
            model = model.to("cuda")
            self._sam_device = "cuda"
        else:
            model = model.to("cpu")
            self._sam_device = "cpu"
        self._sam = SamPredictor(model)
        logging.info("[SAM] loaded %s checkpoint from %s", self.model_type, self.checkpoint)

    def _heat_binary(self, attn_prob: np.ndarray) -> np.ndarray:
        heat = np.asarray(attn_prob)
        if heat.ndim == 3:
            heat = heat.squeeze()
        heat = np.clip(heat, 0.0, 1.0)
        if self.attn_bin_mode == "percentile":
            thr = float(np.quantile(heat, self.attn_bin_q))
            return heat >= thr
        return heat > self.attn_bin_thr

    def _epoch_from_step(self, step: int) -> int:
        step = max(0, min(step, self.total_iters - 1))
        epoch_max = int(math.ceil(self.total_iters / self.steps_per_epoch))
        return min(step // self.steps_per_epoch + 1, epoch_max)

    def _make_out_dir(self, prefix: str, process: float) -> str:
        step = int(min(max(process * self.total_iters, 0), self.total_iters - 1))
        epoch_idx = self._epoch_from_step(step)
        out_dir = os.path.join(self.viz_root, prefix, f"epoch{epoch_idx}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    @staticmethod
    def _compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        if union == 0:
            return 0.0
        return float(inter / union)

    @staticmethod
    def _draw_points(image: np.ndarray, points: np.ndarray) -> np.ndarray:
        import cv2

        overlay = image.copy()
        if points is None:
            return overlay
        pts = np.atleast_2d(points)
        for (x, y) in pts:
            cv2.circle(overlay, (int(round(x)), int(round(y))), 5, (0, 255, 0), -1)
        return overlay

    @staticmethod
    def _draw_single_point(image: np.ndarray, point: Optional[np.ndarray]) -> np.ndarray:
        import cv2

        overlay = image.copy()
        if point is None or len(point) < 2:
            return overlay
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)
        return overlay

    def _run_sam_single(
        self, img_uint8: np.ndarray, points_xy: np.ndarray
    ) -> Tuple[SAMSampleResult, List[List[SAMPointCandidate]]]:
        self.ensure_predictor()
        assert self._sam is not None
        predictor = self._sam
        predictor.set_image(np.ascontiguousarray(img_uint8))

        if points_xy.ndim == 1:
            points_xy = points_xy[None, :]

        per_point_candidates: List[List[SAMPointCandidate]] = []
        flat_candidates: List[Tuple[int, SAMPointCandidate]] = []

        for idx in range(points_xy.shape[0]):
            point = points_xy[idx : idx + 1].astype(np.float32)
            labels = np.ones((1,), dtype=np.int32)

            masks, scores, _ = predictor.predict(
                point_coords=point,
                point_labels=labels,
                multimask_output=self.multimask_output,
            )

            masks_np = np.asarray(masks)
            scores_np = np.asarray(scores).reshape(-1)

            cand_list: List[SAMPointCandidate] = []
            for j in range(scores_np.shape[0]):
                mask_np = masks_np[j]
                if mask_np.ndim == 3:
                    mask_np = mask_np.squeeze()
                mask_bin = (mask_np > 0).astype(np.float32)
                cand = SAMPointCandidate(mask=mask_bin, iou=float(scores_np[j]))
                cand_list.append(cand)
                flat_candidates.append((idx, cand))
            per_point_candidates.append(cand_list)

        if not flat_candidates:
            empty = SAMSampleResult(mask=None, success=False, candidates=[], points_xy=points_xy)
            return empty, per_point_candidates

        flat_candidates.sort(key=lambda item: item[1].iou, reverse=True)

        kept_masks: List[np.ndarray] = []
        selected_candidates: List[SAMCandidate] = []

        for point_idx, cand_point in flat_candidates:
            mask_bool = cand_point.mask.astype(bool)
            area_ratio = float(mask_bool.mean())
            if cand_point.iou < self.iou_thresh or area_ratio > self.area_limit:
                continue

            mask_clean, _ = remove_small_regions(mask_bool, self.hole_scale, mode="holes")
            mask_clean, _ = remove_small_regions(mask_clean, self.island_scale, mode="islands")

            skip = False
            for kept in kept_masks:
                inter = np.logical_and(mask_clean, kept).sum()
                union = np.logical_or(mask_clean, kept).sum()
                if union > 0 and inter / union > self.overlap_ratio:
                    skip = True
                    break

            if skip:
                continue

            kept_masks.append(mask_clean)
            selected_candidates.append(
                SAMCandidate(mask=mask_clean.astype(np.float32), iou=cand_point.iou, point_idx=point_idx)
            )
            if len(kept_masks) >= self.max_masks:
                break

        if not kept_masks:
            fallback_mask = np.zeros(img_uint8.shape[:2], dtype=np.float32)
            empty = SAMSampleResult(mask=fallback_mask, success=False, candidates=selected_candidates, points_xy=points_xy)
            return empty, per_point_candidates

        union = np.logical_or.reduce(kept_masks).astype(np.float32)
        area_union = float(union.mean())
        best_iou = flat_candidates[0][1].iou
        success_flag = not (area_union > self.area_limit or best_iou < self.fallback_iou)

        result = SAMSampleResult(
            mask=union,
            success=success_flag,
            candidates=selected_candidates,
            points_xy=points_xy,
        )
        return result, per_point_candidates

    def _filter_final_candidates(
        self,
        candidates: List[SAMCandidate],
        heat_mask: np.ndarray,
    ) -> Tuple[List[SAMCandidate], List[np.ndarray]]:
        import cv2

        filtered: List[SAMCandidate] = []
        filtered_masks: List[np.ndarray] = []
        for cand in candidates:
            mask = np.asarray(cand.mask)
            if mask.ndim == 3:
                mask = mask.squeeze()
            if mask.shape != heat_mask.shape:
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (heat_mask.shape[1], heat_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                mask_resized = mask.astype(np.float32)
            mask_bool = mask_resized > self.viz_seg_thr
            heat_iou = self._compute_iou(mask_bool, heat_mask)
            cand.heat_iou = heat_iou
            if cand.iou >= self.iou_thresh and heat_iou >= self.heat_iou_thresh:
                filtered.append(cand)
                filtered_masks.append(mask_bool)
        return filtered, filtered_masks

    def generate_pseudo(
        self,
        imgs: torch.Tensor,
        attn_pos: torch.Tensor,
        *,
        process: float,
        filenames: Optional[Sequence[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        List[SAMSampleResult],
        List[SAMVizRecord],
        List[str],
    ]:
        if not self.should_use(process):
            empty_success = torch.zeros(imgs.size(0), dtype=torch.bool, device=imgs.device)
            return None, empty_success, [], [], []

        b, _, h, w = imgs.shape
        attn_up = F.interpolate(attn_pos.detach(), size=(h, w), mode="bilinear", align_corners=False)
        attn_up = _min_max_norm(attn_up)

        pseudo = torch.zeros(b, 1, h, w, device=imgs.device, dtype=imgs.dtype)
        success_mask = torch.zeros(b, dtype=torch.bool, device=imgs.device)
        results: List[SAMSampleResult] = []
        viz_records: List[SAMVizRecord] = []

        mean = self.denorm_mean.to(imgs.device, imgs.dtype)
        std = self.denorm_std.to(imgs.device, imgs.dtype)
        imgs_denorm = (imgs * std + mean).clamp(0.0, 1.0)

        filenames = filenames or []
        filenames_used: List[str] = []

        for idx in range(b):
            prob_map = attn_up[idx : idx + 1]
            pts_xy_orig = select_prompt_points(
                prob_map.squeeze(0),
                percentile=self.prompt_percentile,
                max_points=self.max_points,
                smooth_kernel=self.smooth_kernel,
                min_area_frac=self.min_area_frac,
                max_area_frac=self.max_area_frac,
            )

            resized_img, (res_h, res_w) = _resize_to_short_edge(
                imgs_denorm[idx : idx + 1], self.short_edge
            )
            img_uint8 = _tensor_to_uint8(resized_img[0])

            pts_xy_scaled = pts_xy_orig.copy()
            orig_h = float(imgs_denorm[idx].shape[-2])
            orig_w = float(imgs_denorm[idx].shape[-1])
            if res_h > 0 and res_w > 0 and (res_h != orig_h or res_w != orig_w):
                scale_x = res_w / orig_w
                scale_y = res_h / orig_h
                pts_xy_scaled[:, 0] *= scale_x
                pts_xy_scaled[:, 1] *= scale_y

            attn_np = prob_map.detach().cpu().numpy().squeeze()
            img_uint8_full = _tensor_to_uint8(imgs_denorm[idx])
            img_uint8_aug = img_uint8.copy()
            fname = filenames[idx] if idx < len(filenames) else f"sample_{idx}"
            filenames_used.append(fname)

            sample_res, per_point_cands = self._run_sam_single(img_uint8, pts_xy_scaled)
            results.append(sample_res)

            heat_mask = self._heat_binary(attn_np)
            if heat_mask.shape != (img_uint8_full.shape[0], img_uint8_full.shape[1]):
                import cv2

                heat_mask = cv2.resize(
                    heat_mask.astype(np.uint8),
                    (img_uint8_full.shape[1], img_uint8_full.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                heat_mask = heat_mask.astype(bool)

            filtered_candidates, filtered_masks = self._filter_final_candidates(
                sample_res.candidates, heat_mask
            )
            sample_res.candidates = filtered_candidates

            if filtered_candidates:
                union = np.logical_or.reduce(filtered_masks).astype(np.float32)
                area_union = float(union.mean())
                best_iou = max(c.iou for c in filtered_candidates)
                success_flag = not (area_union > self.area_limit or best_iou < self.fallback_iou)
                if success_flag:
                    sample_res.mask = union
                    sample_res.success = True
                else:
                    sample_res.mask = np.zeros_like(heat_mask, dtype=np.float32)
                    sample_res.success = False
            else:
                sample_res.mask = np.zeros_like(heat_mask, dtype=np.float32)
                sample_res.success = False

            if not sample_res.success:
                viz_records.append(
                    SAMVizRecord(
                        image_uint8=img_uint8_full,
                        aug_image_uint8=img_uint8_aug,
                        attn_prob=attn_np,
                        sam_mask=None,
                        points_xy=pts_xy_orig,
                        points_xy_aug=pts_xy_scaled,
                        success=False,
                        filename=fname,
                        per_point_candidates=per_point_cands,
                        selected_candidates=sample_res.candidates,
                    )
                )
                continue

            if sample_res.mask is None:
                viz_records.append(
                    SAMVizRecord(
                        image_uint8=img_uint8_full,
                        aug_image_uint8=img_uint8_aug,
                        attn_prob=attn_np,
                        sam_mask=None,
                        points_xy=pts_xy_orig,
                        points_xy_aug=pts_xy_scaled,
                        success=False,
                        filename=fname,
                        per_point_candidates=per_point_cands,
                        selected_candidates=sample_res.candidates,
                    )
                )
                continue

            mask_tensor = torch.from_numpy(sample_res.mask).to(device=imgs.device, dtype=imgs.dtype)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
            mask_tensor = F.interpolate(mask_tensor, size=(h, w), mode="nearest")
            pseudo[idx : idx + 1] = mask_tensor
            success_mask[idx] = True
            mask_np = mask_tensor.detach().cpu().numpy().squeeze()
            viz_records.append(
                SAMVizRecord(
                    image_uint8=img_uint8_full,
                    aug_image_uint8=img_uint8_aug,
                    attn_prob=attn_np,
                    sam_mask=mask_np,
                    points_xy=pts_xy_orig,
                    points_xy_aug=pts_xy_scaled,
                    success=True,
                    filename=fname,
                    per_point_candidates=per_point_cands,
                    selected_candidates=sample_res.candidates,
                )
            )

        if success_mask.any():
            return pseudo, success_mask, results, viz_records, filenames_used
        return None, success_mask, results, viz_records, filenames_used

    def save_pseudo_masks(
        self,
        pseudo_masks: torch.Tensor,
        filenames: Sequence[str],
        process: float,
    ) -> None:
        # 与框提示版保持一致，暂时不单独落地伪标签二值图
        return

    def save_visualizations(
        self,
        records: List[SAMVizRecord],
        *,
        process: float,
    ) -> None:
        if not self.save_viz or not records:
            return
        self._save_points_heatmap_overlay(records, process)
        self._save_points_seg_overlay(records, process)
        self._save_per_point_candidates(records, process)
        self._save_final_candidates(records, process)
        self._write_failure_log(records, process)

    def _save_points_heatmap_overlay(self, records: List[SAMVizRecord], process: float) -> None:
        import cv2

        out_dir = self._make_out_dir(self.viz_prefix_attn, process)
        raw_dir = os.path.join(out_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        for record in records:
            img_uint8 = record.image_uint8
            heat = np.asarray(record.attn_prob, dtype=np.float32)
            if heat.ndim == 3 and heat.shape[0] == 1:
                heat = heat[0]
            heat = np.clip(heat, 0.0, 1.0)
            heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
            heat_u8 = (heat * 255.0).astype(np.uint8)
            heat_u8 = np.ascontiguousarray(heat_u8)
            if heat_u8.ndim == 3 and heat_u8.shape[-1] == 1:
                heat_u8 = heat_u8[:, :, 0]
            heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

            overlay = (1 - self.viz_alpha) * img_uint8.astype(np.float32) + self.viz_alpha * heat_rgb.astype(
                np.float32
            )
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            overlay = self._draw_points(overlay, record.points_xy)
            fname = os.path.splitext(record.filename)[0]
            save_path = os.path.join(out_dir, f"{fname}.png")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            raw_path = os.path.join(raw_dir, f"{fname}.png")
            cv2.imwrite(raw_path, cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR))

    def _save_points_seg_overlay(self, records: List[SAMVizRecord], process: float) -> None:
        import cv2

        out_dir = self._make_out_dir(self.viz_prefix_seg, process)
        for record in records:
            if record.sam_mask is None:
                continue
            img_norm = record.image_uint8.astype(np.float32) / 255.0
            seg_mask = np.asarray(record.sam_mask)
            if seg_mask.ndim == 3 and seg_mask.shape[0] == 1:
                seg_mask = seg_mask[0]
            if seg_mask.shape != img_norm.shape[:2]:
                seg_mask = cv2.resize(
                    seg_mask.astype(np.float32),
                    (img_norm.shape[1], img_norm.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            seg_mask = seg_mask > self.viz_seg_thr
            color = np.zeros((*seg_mask.shape, 3), dtype=np.float32)
            color[seg_mask] = [1.0, 0.0, 1.0]
            overlay = (1 - self.viz_alpha) * img_norm + self.viz_alpha * color
            overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
            overlay = self._draw_points(overlay, record.points_xy)
            fname = os.path.splitext(record.filename)[0]
            save_path = os.path.join(out_dir, f"{fname}.png")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def _save_per_point_candidates(self, records: List[SAMVizRecord], process: float) -> None:
        import cv2

        out_dir = self._make_out_dir(self.viz_prefix_per_point, process)
        for record in records:
            if not record.per_point_candidates:
                continue
            fname = os.path.splitext(record.filename)[0]
            orig_base = record.image_uint8.astype(np.float32) / 255.0
            aug_base = record.aug_image_uint8.astype(np.float32) / 255.0
            orig_h, orig_w, _ = record.image_uint8.shape
            aug_h, aug_w, _ = record.aug_image_uint8.shape
            heat_mask = self._heat_binary(record.attn_prob)
            if heat_mask.shape != (orig_h, orig_w):
                heat_mask = cv2.resize(heat_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                heat_mask = heat_mask.astype(bool)
            for idx, cand_list in enumerate(record.per_point_candidates):
                point_dir = os.path.join(out_dir, f"{fname}_{idx + 1}")
                os.makedirs(point_dir, exist_ok=True)
                for cand in cand_list:
                    if cand.iou < self.point_save_iou_thr:
                        continue
                    mask_eval = np.asarray(cand.mask, dtype=np.float32)
                    if mask_eval.ndim == 3:
                        mask_eval = mask_eval.squeeze()
                    mask_eval_bool = mask_eval > self.viz_seg_thr

                    color_aug = np.zeros((aug_h, aug_w, 3), dtype=np.float32)
                    color_aug[mask_eval_bool] = [1.0, 0.0, 1.0]
                    overlay_aug = (1 - self.viz_alpha) * aug_base + self.viz_alpha * color_aug
                    overlay_aug = np.clip(overlay_aug * 255.0, 0, 255).astype(np.uint8)
                    overlay_aug = cv2.resize(overlay_aug, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                    point_aug = record.points_xy[idx] if idx < len(record.points_xy) else None
                    overlay_aug = self._draw_single_point(overlay_aug, point_aug)
                    mask_orig = cv2.resize(mask_eval.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > self.viz_seg_thr
                    heat_iou = self._compute_iou(mask_orig, heat_mask)
                    save_path_aug = os.path.join(point_dir, f"{fname}_{idx + 1}_{cand.iou:.2f}_{heat_iou:.2f}_aug.png")
                    cv2.imwrite(save_path_aug, cv2.cvtColor(overlay_aug, cv2.COLOR_RGB2BGR))

                    color_orig = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
                    color_orig[mask_orig] = [1.0, 0.0, 1.0]
                    overlay_orig = (1 - self.viz_alpha) * orig_base + self.viz_alpha * color_orig
                    overlay_orig = np.clip(overlay_orig * 255.0, 0, 255).astype(np.uint8)
                    point_orig = record.points_xy[idx] if idx < len(record.points_xy) else None
                    overlay_orig = self._draw_single_point(overlay_orig, point_orig)
                    save_path_orig = os.path.join(point_dir, f"{fname}_{idx + 1}_{cand.iou:.2f}_{heat_iou:.2f}.png")
                    cv2.imwrite(save_path_orig, cv2.cvtColor(overlay_orig, cv2.COLOR_RGB2BGR))

    def _save_final_candidates(self, records: List[SAMVizRecord], process: float) -> None:
        import cv2

        out_dir = self._make_out_dir(self.viz_prefix_final, process)
        for record in records:
            if not record.selected_candidates:
                continue
            fname = os.path.splitext(record.filename)[0]
            sample_dir = os.path.join(out_dir, fname)
            os.makedirs(sample_dir, exist_ok=True)
            orig_base = record.image_uint8.astype(np.float32) / 255.0
            aug_base = record.aug_image_uint8.astype(np.float32) / 255.0
            orig_h, orig_w, _ = record.image_uint8.shape
            aug_h, aug_w, _ = record.aug_image_uint8.shape
            heat_mask = self._heat_binary(record.attn_prob)
            if heat_mask.shape != (orig_h, orig_w):
                heat_mask = cv2.resize(heat_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                heat_mask = heat_mask.astype(bool)

            for cand in record.selected_candidates:
                mask_eval = np.asarray(cand.mask, dtype=np.float32)
                mask_eval_bool = mask_eval > self.viz_seg_thr
                color_aug = np.zeros((aug_h, aug_w, 3), dtype=np.float32)
                color_aug[mask_eval_bool] = [1.0, 0.0, 1.0]
                overlay_aug = (1 - self.viz_alpha) * aug_base + self.viz_alpha * color_aug
                overlay_aug = np.clip(overlay_aug * 255.0, 0, 255).astype(np.uint8)
                overlay_aug = cv2.resize(overlay_aug, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
                point_orig = record.points_xy[cand.point_idx] if cand.point_idx < len(record.points_xy) else None
                overlay_aug = self._draw_single_point(overlay_aug, point_orig)

                mask_orig = cv2.resize(mask_eval.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > self.viz_seg_thr
                heat_iou = getattr(cand, "heat_iou", None)
                if heat_iou is None:
                    heat_iou = self._compute_iou(mask_orig, heat_mask)
                cand.heat_iou = heat_iou
                save_path_aug = os.path.join(sample_dir, f"{fname}_{cand.point_idx + 1}_{cand.iou:.2f}_{heat_iou:.2f}_aug.png")
                cv2.imwrite(save_path_aug, cv2.cvtColor(overlay_aug, cv2.COLOR_RGB2BGR))

                color_orig = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
                color_orig[mask_orig] = [1.0, 0.0, 1.0]
                overlay_orig = (1 - self.viz_alpha) * orig_base + self.viz_alpha * color_orig
                overlay_orig = np.clip(overlay_orig * 255.0, 0, 255).astype(np.uint8)
                overlay_orig = self._draw_single_point(overlay_orig, point_orig)
                save_path_orig = os.path.join(sample_dir, f"{fname}_{cand.point_idx + 1}_{cand.iou:.2f}_{heat_iou:.2f}.png")
                cv2.imwrite(save_path_orig, cv2.cvtColor(overlay_orig, cv2.COLOR_RGB2BGR))

    def _write_failure_log(self, records: List[SAMVizRecord], process: float) -> None:
        fails = [rec.filename for rec in records if not rec.success]
        if not fails:
            return
        fail_dir = self._make_out_dir(self.viz_prefix_fail, process)
        log_path = os.path.join(fail_dir, "failures.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            for name in fails:
                f.write(f"{name}\n")
