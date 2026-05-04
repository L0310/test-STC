# Usage:
#   python eval_pseudo_sod.py \
#     --pred_dir datasets/SOD/pseudo_masks \
#     --gt_dir datasets/sod/DUTS-TR/segmentations \
#     --device cuda:0
#
# Notes:
# - This script matches evaluateSOD/evaluator.py metric formulas and
#   dataset-level aggregation style.
# - If GT matching by relative path fails, add --match_by_basename.
# - If preds are probability maps, add --pred_is_prob.
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SOD pseudo labels with MAE, mean F-measure, mean E-measure, and S-measure."
    )
    parser.add_argument("--pred_dir", required=True, help="Directory of predicted masks/heatmaps.")
    parser.add_argument("--gt_dir", required=True, help="Directory of GT masks.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu")
    parser.add_argument("--beta2", type=float, default=0.3, help="beta^2 for F-beta")
    parser.add_argument(
        "--threshold_steps",
        type=int,
        default=255,
        help="Threshold count. Use 255 to match evaluateSOD/evaluator.py exactly.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="Compatibility arg. Not used by the current implementation.",
    )
    parser.add_argument("--pred_is_prob", action="store_true", help="Use bilinear resize for prob maps.")
    parser.add_argument(
        "--match_by_basename",
        action="store_true",
        help="Match GT by filename stem if relative path match fails.",
    )
    parser.add_argument(
        "--per_sample",
        action="store_true",
        help="Record per-sample metrics as JSONL.",
    )
    parser.add_argument(
        "--per_sample_out",
        default="per_sample_metrics.jsonl",
        help="Path to save per-sample metrics JSONL (use empty string to disable).",
    )
    parser.add_argument(
        "--topk_worst",
        type=int,
        default=0,
        help="Print worst-K samples by --worst_metric (requires --per_sample).",
    )
    parser.add_argument(
        "--worst_metric",
        choices=[
            "mae",
            "mean_fmeasure",
            "mean_emeasure",
            "s_measure",
        ],
        default="mae",
        help="Metric used to rank worst samples.",
    )
    parser.add_argument("--save_json", default="", help="Optional path to save metrics JSON.")
    return parser.parse_args()


def _index_gt_by_stem(gt_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in gt_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            index[path.stem] = path
    return index


def _resolve_gt(
    pred_path: Path, pred_root: Path, gt_root: Path, gt_by_stem: Optional[Dict[str, Path]]
) -> Optional[Path]:
    rel = pred_path.relative_to(pred_root)
    rel_no_mask = rel
    if rel.stem.endswith("_mask"):
        rel_no_mask = rel.with_name(rel.stem[:-5] + ".png")
    candidates = [
        gt_root / rel,
        (gt_root / rel).with_suffix(".png"),
        gt_root / rel_no_mask,
        (gt_root / rel_no_mask).with_suffix(".png"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    if gt_by_stem is not None:
        pred_stem = pred_path.stem
        if pred_stem.endswith("_mask"):
            pred_stem = pred_stem[:-5]
        return gt_by_stem.get(pred_stem)
    return None


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read: {path}")
    return img


def _to_unit_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32)).to(device)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _eval_mae(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - gt).mean()


def _eval_pr(pred: torch.Tensor, gt: torch.Tensor, num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    thlist = torch.linspace(0, 1 - 1e-10, num, device=pred.device)
    pred_th = (pred.unsqueeze(0) >= thlist.view(-1, 1, 1)).float()
    tp = (pred_th * gt.unsqueeze(0)).sum(dim=(1, 2))
    prec = tp / (pred_th.sum(dim=(1, 2)) + 1e-20)
    recall = tp / (gt.sum() + 1e-20)
    return prec, recall


def _eval_e(pred: torch.Tensor, gt: torch.Tensor, num: int) -> torch.Tensor:
    thlist = torch.linspace(0, 1 - 1e-10, num, device=pred.device)
    pred_th = (pred.unsqueeze(0) >= thlist.view(-1, 1, 1)).float()
    fm = pred_th - pred_th.mean(dim=(1, 2), keepdim=True)
    gt_centered = gt - gt.mean()
    align_matrix = 2 * gt_centered.unsqueeze(0) * fm / (
        gt_centered.unsqueeze(0) * gt_centered.unsqueeze(0) + fm * fm + 1e-20
    )
    enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
    return enhanced.sum(dim=(1, 2)) / (gt.numel() - 1 + 1e-20)


def _object(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    return 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)


def _s_object(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1 - gt)
    u = gt.mean()
    return u * o_fg + (1 - u) * o_bg


def _centroid(gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = gt.shape[-2:]
    gt = gt.view(rows, cols)
    if gt.sum() == 0:
        x = gt.new_tensor(round(cols / 2), dtype=torch.float32)
        y = gt.new_tensor(round(rows / 2), dtype=torch.float32)
    else:
        total = gt.sum()
        i = torch.arange(0, cols, device=gt.device, dtype=torch.float32)
        j = torch.arange(0, rows, device=gt.device, dtype=torch.float32)
        x = torch.round((gt.sum(dim=0) * i).sum() / total)
        y = torch.round((gt.sum(dim=1) * j).sum() / total)
    return x.long(), y.long()


def _divide_gt(
    gt: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = gt.shape[-2:]
    area = h * w
    gt = gt.view(h, w)
    lt = gt[:y, :x]
    rt = gt[:y, x:w]
    lb = gt[y:h, :x]
    rb = gt[y:h, x:w]
    x_float = x.float()
    y_float = y.float()
    w1 = x_float * y_float / area
    w2 = (w - x_float) * y_float / area
    w3 = x_float * (h - y_float) / area
    w4 = 1 - w1 - w2 - w3
    return lt, rt, lb, rb, w1, w2, w3, w4


def _divide_prediction(
    pred: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = pred.shape[-2:]
    pred = pred.view(h, w)
    lt = pred[:y, :x]
    rt = pred[:y, x:w]
    lb = pred[y:h, :x]
    rb = pred[y:h, x:w]
    return lt, rt, lb, rb


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    gt = gt.float()
    h, w = pred.shape[-2:]
    n = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (n - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (n - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (n - 1 + 1e-20)

    alpha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        return alpha / (beta + 1e-20)
    if alpha == 0 and beta == 0:
        return pred.new_tensor(1.0)
    return pred.new_tensor(0.0)


def _s_region(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    x, y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divide_gt(gt, x, y)
    p1, p2, p3, p4 = _divide_prediction(pred, x, y)
    q1 = _ssim(p1, gt1)
    q2 = _ssim(p2, gt2)
    q3 = _ssim(p3, gt3)
    q4 = _ssim(p4, gt4)
    return w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4


def _eval_smeasure(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    alpha = 0.5
    y = gt.mean()
    if y == 0:
        return 1.0 - pred.mean()
    if y == 1:
        return pred.mean()

    gt_bin = gt.clone()
    gt_bin[gt_bin >= 0.5] = 1
    gt_bin[gt_bin < 0.5] = 0
    q = alpha * _s_object(pred, gt_bin) + (1 - alpha) * _s_region(pred, gt_bin)
    if q.item() < 0:
        return pred.new_tensor(0.0)
    return q


def _rank_records(records, metric):
    reverse = metric == "mae"
    return sorted(records, key=lambda item: item[metric], reverse=reverse)


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred_dir)
    gt_root = Path(args.gt_dir)
    if not pred_root.exists():
        raise FileNotFoundError(f"pred_dir not found: {pred_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"gt_dir not found: {gt_root}")

    gt_by_stem = _index_gt_by_stem(gt_root) if args.match_by_basename else None
    pred_paths = sorted([p for p in pred_root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS])
    if not pred_paths:
        raise RuntimeError(f"No prediction files found under {pred_root}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    interp = cv2.INTER_LINEAR if args.pred_is_prob else cv2.INTER_NEAREST

    sample_count = 0
    missing = 0
    mae_sum = 0.0
    s_sum = 0.0
    f_curve_sum = None
    e_curve_sum = None
    per_sample_records = []

    with torch.no_grad():
        for pred_path in tqdm(pred_paths, desc="Eval pseudo"):
            gt_path = _resolve_gt(pred_path, pred_root, gt_root, gt_by_stem)
            if gt_path is None:
                missing += 1
                continue

            pred = _load_gray(pred_path).astype(np.float32)
            gt = _load_gray(gt_path).astype(np.float32)

            if pred.max() > 1.0:
                pred = pred / 255.0
            if gt.max() > 1.0:
                gt = gt / 255.0
            gt = (gt > 0.5).astype(np.float32)

            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=interp)

            pred_t = _to_unit_tensor(pred, device)
            gt_t = _to_unit_tensor(gt, device)
            gt_t = (gt_t > 0.5).float()

            mae = _eval_mae(pred_t, gt_t)
            prec, recall = _eval_pr(pred_t, gt_t, args.threshold_steps)
            f_curve = (1 + args.beta2) * prec * recall / (args.beta2 * prec + recall + 1e-20)
            f_curve[f_curve != f_curve] = 0
            e_curve = _eval_e(pred_t, gt_t, args.threshold_steps)
            s_measure = _eval_smeasure(pred_t, gt_t)

            mae_sum += float(mae.item())
            s_sum += float(s_measure.item())
            f_curve_sum = f_curve if f_curve_sum is None else f_curve_sum + f_curve
            e_curve_sum = e_curve if e_curve_sum is None else e_curve_sum + e_curve
            sample_count += 1

            if args.per_sample:
                per_sample_records.append(
                    {
                        "name": pred_path.name,
                        "mae": float(mae.item()),
                        "mean_fmeasure": float(f_curve.mean().item()),
                        "mean_emeasure": float(e_curve.mean().item()),
                        "s_measure": float(s_measure.item()),
                    }
                )

    if sample_count == 0:
        results = {
            "MAE": 0.0,
            "meanFmeasure": 0.0,
            "meanEmeasure": 0.0,
            "Smeasure": 0.0,
            "NumSamples": 0,
            "MissingGT": int(missing),
        }
    else:
        mean_f_curve = f_curve_sum / sample_count
        mean_e_curve = e_curve_sum / sample_count
        results = {
            "MAE": mae_sum / sample_count,
            "meanFmeasure": float(mean_f_curve.mean().item()),
            "meanEmeasure": float(mean_e_curve.mean().item()),
            "Smeasure": s_sum / sample_count,
            "NumSamples": int(sample_count),
            "MissingGT": int(missing),
        }

    print(results)
    print(
        "mae={MAE:.6f}, mean-fmeasure={meanFmeasure:.6f}, "
        "mean-Emeasure={meanEmeasure:.6f}, S-measure={Smeasure:.6f}".format(**results)
    )

    if args.per_sample:
        if args.per_sample_out:
            os.makedirs(os.path.dirname(args.per_sample_out) or ".", exist_ok=True)
            with open(args.per_sample_out, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"type": "summary", **results}, ensure_ascii=False) + "\n")
                for rec in _rank_records(per_sample_records, "mae"):
                    handle.write(json.dumps({"type": "sample", **rec}, ensure_ascii=False) + "\n")
            print(f"Per-sample metrics saved to: {args.per_sample_out}")

        if args.topk_worst > 0 and per_sample_records:
            worst = _rank_records(per_sample_records, args.worst_metric)[: args.topk_worst]
            print(f"Worst {len(worst)} by {args.worst_metric}:")
            for rec in worst:
                print(
                    "name={name} mae={mae:.6f} mean_f={mean_fmeasure:.6f} "
                    "mean_e={mean_emeasure:.6f} s={s_measure:.6f}".format(**rec)
                )

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
