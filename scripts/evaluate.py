#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


TARGET_LABELS = (8, 9, 10, 13)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    structure = ndi.generate_binary_structure(mask.ndim, 1)
    return mask & ~ndi.binary_erosion(mask, structure=structure, border_value=0)


def crop_to_union(
    reference: np.ndarray,
    prediction: np.ndarray,
    padding: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop metric inputs to their union while retaining a background border."""
    coordinates = np.nonzero(reference | prediction)
    slices = tuple(
        slice(max(0, int(axis.min()) - padding), min(size, int(axis.max()) + padding + 1))
        for axis, size in zip(coordinates, reference.shape)
    )
    return reference[slices], prediction[slices]


def boundary_metrics(reference: np.ndarray, prediction: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    if not reference.any() or not prediction.any():
        return {"hd95_mm": math.nan, "assd_mm": math.nan}
    reference, prediction = crop_to_union(reference, prediction)
    ref_surface = surface(reference)
    pred_surface = surface(prediction)
    if not ref_surface.any() or not pred_surface.any():
        return {"hd95_mm": math.nan, "assd_mm": math.nan}
    pred_to_ref = ndi.distance_transform_edt(~ref_surface, sampling=spacing)[pred_surface]
    ref_to_pred = ndi.distance_transform_edt(~pred_surface, sampling=spacing)[ref_surface]
    distances = np.concatenate([pred_to_ref, ref_to_pred])
    return {
        "hd95_mm": float(np.percentile(distances, 95)),
        "assd_mm": float((pred_to_ref.mean() + ref_to_pred.mean()) / 2.0),
    }


def cldice(reference: np.ndarray, prediction: np.ndarray) -> float:
    if not reference.any():
        return math.nan
    if not prediction.any():
        return 0.0
    reference, prediction = crop_to_union(reference, prediction)
    ref_skel = skeletonize(reference)
    pred_skel = skeletonize(prediction)
    if not ref_skel.any() or not pred_skel.any():
        return math.nan
    tprec = safe_divide(float(np.count_nonzero(pred_skel & reference)), float(np.count_nonzero(pred_skel)))
    tsens = safe_divide(float(np.count_nonzero(ref_skel & prediction)), float(np.count_nonzero(ref_skel)))
    return safe_divide(2.0 * tprec * tsens, tprec + tsens)


def voxel_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    tp = int(np.count_nonzero(reference & prediction))
    fp = int(np.count_nonzero(~reference & prediction))
    fn = int(np.count_nonzero(reference & ~prediction))
    ref_voxels = int(np.count_nonzero(reference))
    pred_voxels = int(np.count_nonzero(prediction))
    if not ref_voxels and not pred_voxels:
        dice = math.nan
        iou = math.nan
    else:
        dice = safe_divide(2 * tp, 2 * tp + fp + fn)
        iou = safe_divide(tp, tp + fp + fn)
    return {
        "reference_present": bool(ref_voxels),
        "prediction_present": bool(pred_voxels),
        "prediction_empty_on_positive_reference": bool(ref_voxels and not pred_voxels),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_ref": ref_voxels,
        "n_pred": pred_voxels,
        "dice": dice,
        "iou": iou,
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
    }


def label_names(dataset_json: Path) -> dict[int, str]:
    content = json.loads(dataset_json.read_text(encoding="utf-8"))
    return {int(value): name for name, value in content["labels"].items() if int(value) > 0}


def expected_cases(raw_root: Path, fold: int, case_prefix: str) -> list[str]:
    splits = json.loads((raw_root / "splits_final.json").read_text(encoding="utf-8"))
    cases = [str(case) for case in splits[fold]["val"]]
    return sorted(case for case in cases if case.startswith(case_prefix))


def mean_std(values: list[float]) -> tuple[float, float]:
    clean = np.asarray([value for value in values if not math.isnan(float(value))], dtype=np.float64)
    if clean.size == 0:
        return math.nan, math.nan
    return float(clean.mean()), float(clean.std(ddof=0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_class_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["label"]), str(row["class_name"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (label, class_name), items in sorted(grouped.items()):
        positive_items = [row for row in items if bool(row["reference_present"])]
        record: dict[str, Any] = {
            "label": label,
            "class_name": class_name,
            "cases_total": len(items),
            "positive_reference_cases": len(positive_items),
            "prediction_empty_positive_cases": sum(
                bool(row["prediction_empty_on_positive_reference"]) for row in positive_items
            ),
        }
        for metric in ("dice", "iou", "precision", "recall", "hd95_mm", "assd_mm", "cldice"):
            record[f"{metric}_mean"], record[f"{metric}_std"] = mean_std(
                [float(row[metric]) for row in positive_items]
            )
        summary.append(record)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--case-prefix", default="topcow_mr_")
    parser.add_argument("--exclude-case", action="append", default=[], help="Case ID to exclude; repeat as needed.")
    parser.add_argument("--experiment-name", default="prediction subset evaluation")
    parser.add_argument(
        "--boundary-labels",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13",
        help="Comma-separated labels for HD95/ASSD/clDice. Dice is still computed for all labels.",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    pred_dir = args.pred_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    names = label_names(raw_root / "dataset.json")
    labels = sorted(names)
    boundary_labels = {int(value) for value in args.boundary_labels.split(",") if value.strip()}
    cases = expected_cases(raw_root, args.fold, args.case_prefix)
    excluded_cases = set(args.exclude_case)
    cases = [case for case in cases if case not in excluded_cases]
    if not cases:
        raise RuntimeError(f"No validation cases with prefix {args.case_prefix!r}")

    class_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for case in cases:
        pred_path = pred_dir / f"{case}.nii.gz"
        ref_path = raw_root / "labelsTr" / f"{case}.nii.gz"
        if not pred_path.is_file():
            raise FileNotFoundError(pred_path)
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)

        pred_img = nib.load(str(pred_path))
        ref_img = nib.load(str(ref_path))
        pred = np.asarray(pred_img.dataobj).astype(np.int16, copy=False)
        ref = np.asarray(ref_img.dataobj).astype(np.int16, copy=False)
        if pred.shape != ref.shape:
            raise RuntimeError(f"Shape mismatch for {case}: {pred.shape} vs {ref.shape}")
        if not np.allclose(pred_img.affine, ref_img.affine, atol=1e-4):
            raise RuntimeError(f"Physical geometry mismatch for {case}")
        spacing = tuple(float(value) for value in ref_img.header.get_zooms()[:3])

        fg = voxel_metrics(ref > 0, pred > 0)
        positive_dice: list[float] = []
        target_dice: list[float] = []
        for label in labels:
            ref_mask = ref == label
            pred_mask = pred == label
            metrics = voxel_metrics(ref_mask, pred_mask)
            row: dict[str, Any] = {
                "fold": args.fold,
                "case": case,
                "label": label,
                "class_name": names[label],
                **metrics,
                "hd95_mm": math.nan,
                "assd_mm": math.nan,
                "cldice": math.nan,
            }
            if metrics["reference_present"]:
                if label in boundary_labels:
                    row.update(boundary_metrics(ref_mask, pred_mask, spacing))
                    row["cldice"] = cldice(ref_mask, pred_mask)
                if not math.isnan(float(metrics["dice"])):
                    positive_dice.append(float(metrics["dice"]))
                if label in TARGET_LABELS and not math.isnan(float(metrics["dice"])):
                    target_dice.append(float(metrics["dice"]))
            class_rows.append(row)

        case_rows.append(
            {
                "fold": args.fold,
                "case": case,
                "foreground_dice": fg["dice"],
                "foreground_iou": fg["iou"],
                "macro_dice_positive_reference": float(np.mean(positive_dice)) if positive_dice else math.nan,
                "target_dice_8_9_10_13_positive_reference": float(np.mean(target_dice)) if target_dice else math.nan,
            }
        )

    class_summary = summarize_class_rows(class_rows)
    macro_class_dice, macro_class_dice_std = mean_std([float(row["dice_mean"]) for row in class_summary])
    target_summary = [row for row in class_summary if int(row["label"]) in TARGET_LABELS]
    target_dice, target_dice_std = mean_std([float(row["dice_mean"]) for row in target_summary])
    foreground_dice, foreground_dice_std = mean_std([float(row["foreground_dice"]) for row in case_rows])

    write_csv(out_dir / "case_class_metrics.csv", class_rows)
    write_csv(out_dir / "class_summary.csv", class_summary)
    write_csv(out_dir / "case_summary.csv", case_rows)
    payload = {
        "experiment_name": args.experiment_name,
        "raw_root": str(raw_root),
        "pred_dir": str(pred_dir),
        "fold": args.fold,
        "case_prefix": args.case_prefix,
        "excluded_cases": sorted(excluded_cases),
        "case_count": len(cases),
        "foreground_dice_case_mean": foreground_dice,
        "foreground_dice_case_std": foreground_dice_std,
        "macro_class_dice_positive_reference": macro_class_dice,
        "macro_class_dice_positive_reference_std": macro_class_dice_std,
        "target_dice_8_9_10_13": target_dice,
        "target_dice_8_9_10_13_std": target_dice_std,
        "class_summary": class_summary,
    }
    (out_dir / "evaluation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# {args.experiment_name}",
        "",
        "## 验收对象",
        "",
        f"- 数据集：`{raw_root.name}`",
        f"- 预测目录：`{pred_dir}`",
        f"- fold：`{args.fold}`",
        f"- 病例筛选：`{args.case_prefix}*`",
        f"- 排除病例：`{sorted(excluded_cases)}`",
        f"- 病例数：`{len(cases)}`",
        "",
        "## 总体结果",
        "",
        f"- Foreground Dice（病例平均）：`{foreground_dice:.6f} +/- {foreground_dice_std:.6f}`",
        f"- Macro class Dice（仅统计真实存在类别）：`{macro_class_dice:.6f} +/- {macro_class_dice_std:.6f}`",
        f"- 重点类别 8/9/10/13 Dice：`{target_dice:.6f} +/- {target_dice_std:.6f}`",
        "",
        "## 分类别结果",
        "",
        "| Label | 类别 | 阳性病例 | 阳性中空预测 | Dice | HD95 mm | ASSD mm | clDice |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in class_summary:
        lines.append(
            f"| {row['label']} | {row['class_name']} | {row['positive_reference_cases']} | "
            f"{row['prediction_empty_positive_cases']} | {row['dice_mean']:.6f} | "
            f"{row['hd95_mm_mean']:.6f} | {row['assd_mm_mean']:.6f} | {row['cldice_mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 统计口径",
            "",
            "- 分类别 Dice/HD95/ASSD/clDice 只在真实标签中存在该血管的病例上取平均。",
            "- 这样不会把“真实缺失且预测缺失”的变异血管误算成完美分割。",
            "- Foreground Dice 是整例前景合并后计算，主要用于观察整体血管区域重叠。",
        ]
    )
    (out_dir / "RESULT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "RESULT_REPORT.md")


if __name__ == "__main__":
    main()
