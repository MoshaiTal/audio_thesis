from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


ID_COLUMNS = {"key", "alpha", "reference", "hypothesis", "pred_path", "lower_path", "upper_path", "text_path", "clean_path"}
TARGET_COLUMNS = {"pred_wer", "word_errors"}
CONTROL_COLUMNS = {"ref_words", "true_frames"}
TARGET_DERIVED_COLUMNS = {"word_errors_per_frame", "wer_times_ref_words"}
PRED_CLEAN_PREFIX = "pred_clean_"


def parse_float(value: object) -> object:
    if value in {"", None}:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def load_csv(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = [{key: parse_float(value) for key, value in row.items()} for row in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    if "key" not in rows[0]:
        raise RuntimeError(f"CSV must contain a key column: {path}")
    return rows


def merge_by_key(cp_rows: Sequence[Dict[str, object]], pred_clean_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    pred_clean_by_key = {str(row["key"]): row for row in pred_clean_rows}
    merged: List[Dict[str, object]] = []
    for cp_row in cp_rows:
        key = str(cp_row["key"])
        pred_clean_row = pred_clean_by_key.get(key)
        if pred_clean_row is None:
            continue
        out = dict(cp_row)
        for col, value in pred_clean_row.items():
            if col.startswith(PRED_CLEAN_PREFIX) or col == "clean_path":
                out[col] = value
            elif col in {"pred_wer", "word_errors", "ref_words", "true_frames"} and col not in out:
                out[col] = value
        merged.append(out)
    if not merged:
        raise RuntimeError("No matching keys between CP feature CSV and pred-clean CSV.")
    return merged


def numeric_array(rows: Sequence[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return float("nan")
    return pearsonr(rankdata(x), rankdata(y))


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    y_valid = y[mask]
    controls_valid = controls[mask]
    design = np.column_stack([np.ones(len(controls_valid)), controls_valid])
    beta, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
    resid = np.full_like(y, np.nan, dtype=np.float64)
    resid[mask] = y_valid - design @ beta
    return resid


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    return pearsonr(residualize(x, controls), residualize(y, controls))


def numeric_feature_names(rows: Sequence[Dict[str, object]]) -> List[str]:
    names: List[str] = []
    for key, value in rows[0].items():
        if key in ID_COLUMNS or key in TARGET_COLUMNS or key in CONTROL_COLUMNS or key in TARGET_DERIVED_COLUMNS:
            continue
        if key.startswith(PRED_CLEAN_PREFIX):
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            names.append(key)
    return names


def control_matrix(rows: Sequence[Dict[str, object]], controls: Sequence[str]) -> np.ndarray:
    return np.column_stack([numeric_array(rows, control) for control in controls])


def report_rows(rows: Sequence[Dict[str, object]], features: Sequence[str], pred_clean_controls: Sequence[str]) -> List[Dict[str, object]]:
    pred_wer = numeric_array(rows, "pred_wer")
    word_errors = numeric_array(rows, "word_errors")

    base_controls = ["ref_words", "true_frames"]
    controls_with_pred_clean = base_controls + list(pred_clean_controls)
    base_matrix = control_matrix(rows, base_controls)
    full_matrix = control_matrix(rows, controls_with_pred_clean)

    report: List[Dict[str, object]] = []
    for feature in features:
        x = numeric_array(rows, feature)
        raw_wer = pearsonr(x, pred_wer)
        base_wer = partial_corr(x, pred_wer, base_matrix)
        full_wer = partial_corr(x, pred_wer, full_matrix)
        raw_errors = pearsonr(x, word_errors)
        base_errors = partial_corr(x, word_errors, base_matrix)
        full_errors = partial_corr(x, word_errors, full_matrix)
        report.append(
            {
                "feature": feature,
                "pred_wer_pearson": raw_wer,
                "pred_wer_spearman": spearmanr(x, pred_wer),
                "pred_wer_partial_ref_words+true_frames": base_wer,
                f"pred_wer_partial_ref_words+true_frames+{'+'.join(pred_clean_controls)}": full_wer,
                "pred_wer_partial_drop_abs": abs(base_wer) - abs(full_wer),
                "word_errors_pearson": raw_errors,
                "word_errors_spearman": spearmanr(x, word_errors),
                "word_errors_partial_ref_words+true_frames": base_errors,
                f"word_errors_partial_ref_words+true_frames+{'+'.join(pred_clean_controls)}": full_errors,
                "word_errors_partial_drop_abs": abs(base_errors) - abs(full_errors),
            }
        )

    full_col = f"pred_wer_partial_ref_words+true_frames+{'+'.join(pred_clean_controls)}"
    report.sort(key=lambda row: abs(float(row[full_col])), reverse=True)
    return report


def decile_table(rows: Sequence[Dict[str, object]], feature: str) -> List[Dict[str, float]]:
    sorted_rows = sorted(rows, key=lambda row: float(row[feature]))
    bins: List[Dict[str, float]] = []
    for idx, chunk in enumerate(np.array_split(sorted_rows, min(10, len(sorted_rows))), start=1):
        chunk_rows = list(chunk)
        bins.append(
            {
                "bin": float(idx),
                "n": float(len(chunk_rows)),
                f"{feature}_min": float(min(float(row[feature]) for row in chunk_rows)),
                f"{feature}_max": float(max(float(row[feature]) for row in chunk_rows)),
                "mean_pred_wer": float(np.mean([float(row["pred_wer"]) for row in chunk_rows])),
                "mean_word_errors": float(np.mean([float(row["word_errors"]) for row in chunk_rows])),
                "mean_ref_words": float(np.mean([float(row["ref_words"]) for row in chunk_rows])),
                "mean_true_frames": float(np.mean([float(row["true_frames"]) for row in chunk_rows])),
                "mean_pred_clean_l1_p90": float(np.mean([float(row["pred_clean_l1_p90"]) for row in chunk_rows]))
                if "pred_clean_l1_p90" in chunk_rows[0]
                else float("nan"),
                "mean_pred_clean_rmse": float(np.mean([float(row["pred_clean_rmse"]) for row in chunk_rows]))
                if "pred_clean_rmse" in chunk_rows[0]
                else float("nan"),
            }
        )
    return bins


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether CP features still predict Whisper WER after controlling for pred-clean distance."
    )
    parser.add_argument("--cp-features-csv", type=Path, required=True)
    parser.add_argument("--pred-clean-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--pred-clean-controls",
        nargs="+",
        default=["pred_clean_l1_p90"],
        help="Pred-clean distance features to include as controls.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cp_rows = load_csv(args.cp_features_csv)
    pred_clean_rows = load_csv(args.pred_clean_csv)
    merged = merge_by_key(cp_rows, pred_clean_rows)

    missing_controls = [control for control in args.pred_clean_controls if control not in merged[0]]
    if missing_controls:
        raise RuntimeError(f"Missing pred-clean control columns: {missing_controls}")

    features = numeric_feature_names(merged)
    report = report_rows(merged, features, args.pred_clean_controls)
    full_col = f"pred_wer_partial_ref_words+true_frames+{'+'.join(args.pred_clean_controls)}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "merged_cp_pred_clean_features.csv", merged)
    write_csv(args.out_dir / "cp_after_pred_clean_control_report.csv", report)

    top = report[: args.top_k]
    summary = {
        "n": len(merged),
        "cp_features_csv": str(args.cp_features_csv),
        "pred_clean_csv": str(args.pred_clean_csv),
        "pred_clean_controls": args.pred_clean_controls,
        "top_features_after_pred_clean_control": top,
        "deciles": {str(row["feature"]): decile_table(merged, str(row["feature"])) for row in top[:5]},
    }
    (args.out_dir / "cp_after_pred_clean_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[SAVE] {args.out_dir / 'merged_cp_pred_clean_features.csv'}")
    print(f"[SAVE] {args.out_dir / 'cp_after_pred_clean_control_report.csv'}")
    print(f"[SAVE] {args.out_dir / 'cp_after_pred_clean_control_summary.json'}")
    print(f"\nTop CP features after controlling for ref_words + true_frames + {' + '.join(args.pred_clean_controls)}:")
    for row in top[: min(10, len(top))]:
        print(
            f"{row['feature']}: "
            f"before={float(row['pred_wer_partial_ref_words+true_frames']):.4f}, "
            f"after={float(row[full_col]):.4f}, "
            f"drop_abs={float(row['pred_wer_partial_drop_abs']):.4f}"
        )


if __name__ == "__main__":
    main()
