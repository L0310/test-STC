import argparse
import json
import os
from typing import Dict, List

import cv2
import numpy as np

from .image_ops import _ensure_binary_mask, _read_mask, _read_rgb_image, _write_rgb, extract_red_mask_from_overlay
from .sam_predictor import SAMHelper
from .types import SAMResult

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
