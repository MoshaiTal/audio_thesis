from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


NUMERIC_INPUT_COLUMNS = {
    "pred_wer",
    "word_errors",
    "ref_words",
    "width_mean",
    "width_median",
    "width_p90",
    "width_p95",
    "true_frames",
}

LEN_RE = re.compile(r"\[len=(\d+)\]")


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, object] = dict(row)
            for key in NUMERIC_INPUT_COLUMNS:
                if key in parsed and parsed[key] != "":
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    required = {"pred_path", "lower_path", "upper_path", "pred_wer", "word_errors", "ref_words"}
    missing = required - set(rows[0])
    if missing:
        raise RuntimeError(f"Input CSV is missing required columns: {sorted(missing)}")
    return rows


def apply_path_rewrites(rows: List[Dict[str, object]], rewrites: Sequence[str]) -> None:
    parsed: List[Tuple[str, str]] = []
    for rewrite in rewrites:
        if "=" not in rewrite:
            raise ValueError(f"Path rewrite must use OLD=NEW form, got: {rewrite}")
        old, new = rewrite.split("=", 1)
        parsed.append((old, new))

    for row in rows:
        for key in ["pred_path", "lower_path", "upper_path", "text_path"]:
            if key not in row:
                continue
            value = str(row[key])
            for old, new in parsed:
                if value.startswith(old):
                    value = new + value[len(old):]
            row[key] = value


def load_2d_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Array path from CSV does not exist: {path}. "
            "Run this on the machine that has the arrays, or use --path-rewrite OLD=NEW."
        )
    arr = np.squeeze(np.load(path).astype(np.float32))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}: {path}")
    return arr


def align_arrays(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    min_f = min(arr.shape[0] for arr in arrays)
    min_t = min(arr.shape[1] for arr in arrays)
    return tuple(arr[:min_f, :min_t] for arr in arrays)


def true_frames_from_row(row: Dict[str, object], fallback: int) -> int:
    if "true_frames" in row and row["true_frames"] not in {"", None}:
        try:
            return min(int(float(row["true_frames"])), fallback)
        except ValueError:
            pass
    for key in ["pred_path", "lower_path", "upper_path", "text_path"]:
        match = LEN_RE.search(str(row.get(key, "")))
        if match:
            return min(int(match.group(1)), fallback)
    return fallback


def crop_time(arr: np.ndarray, true_t: int) -> np.ndarray:
    return arr[:, : min(true_t, arr.shape[1])]


def orient_like_width(pred: np.ndarray, width: np.ndarray) -> np.ndarray:
    if pred.shape == width.shape:
        return pred
    if pred.T.shape == width.shape:
        return pred.T
    pred, _ = align_arrays(pred, width)
    return pred


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def frame_profile_stats(width: np.ndarray) -> Dict[str, float]:
    frame_mean = width.mean(axis=0)
    frame_max = width.max(axis=0)
    return {
        "frame_width_mean_mean": float(frame_mean.mean()),
        "frame_width_mean_std": float(frame_mean.std()),
        "frame_width_mean_p90": float(np.percentile(frame_mean, 90.0)),
        "frame_width_max_mean": float(frame_max.mean()),
        "frame_width_max_p90": float(np.percentile(frame_max, 90.0)),
    }


def band_features(width: np.ndarray) -> Dict[str, float]:
    bands = np.array_split(np.arange(width.shape[0]), 3)
    names = ["low", "mid", "high"]
    return {f"width_{name}_band_mean": safe_mean(width[idx, :]) for name, idx in zip(names, bands)}


def activity_features(pred: np.ndarray, width: np.ndarray, activity_percentile: float) -> Dict[str, float]:
    pred = orient_like_width(pred, width)
    pred, width = align_arrays(pred, width)

    energy = pred.mean(axis=0)
    threshold = float(np.percentile(energy, activity_percentile))
    active_frames = energy >= threshold
    inactive_frames = ~active_frames

    return {
        "active_frame_fraction": float(active_frames.mean()),
        "width_active_mean": safe_mean(width[:, active_frames]),
        "width_inactive_mean": safe_mean(width[:, inactive_frames]),
        "width_active_minus_inactive": safe_mean(width[:, active_frames]) - safe_mean(width[:, inactive_frames]),
    }


def base_width_features(width: np.ndarray, high_thresholds: Dict[str, float]) -> Dict[str, float]:
    values: Dict[str, float] = {
        "n_freq_bins": float(width.shape[0]),
        "n_frames": float(width.shape[1]),
        "width_mean_recomputed": float(width.mean()),
        "width_std": float(width.std()),
        "width_cv": float(width.std() / max(width.mean(), 1e-8)),
        "width_median_recomputed": float(np.median(width)),
        "width_p75": float(np.percentile(width, 75.0)),
        "width_p90_recomputed": float(np.percentile(width, 90.0)),
        "width_p95_recomputed": float(np.percentile(width, 95.0)),
        "width_max": float(width.max()),
    }
    for label, threshold in high_thresholds.items():
        values[f"width_frac_above_global_{label}"] = float(np.mean(width > threshold))
    values.update(frame_profile_stats(width))
    values.update(band_features(width))
    return values


def collect_global_width_thresholds(rows: Sequence[Dict[str, object]], percentiles: Sequence[float]) -> Dict[str, float]:
    all_widths: List[np.ndarray] = []
    for row in tqdm(rows, desc="Loading widths for global thresholds"):
        lower, upper = align_arrays(load_2d_array(row["lower_path"]), load_2d_array(row["upper_path"]))
        true_t = true_frames_from_row(row, lower.shape[1])
        lower = crop_time(lower, true_t)
        upper = crop_time(upper, true_t)
        all_widths.append(np.abs(upper - lower).reshape(-1))
    merged = np.concatenate(all_widths)
    return {f"q{int(p)}": float(np.percentile(merged, p)) for p in percentiles}


def add_feature_columns(
    rows: List[Dict[str, object]],
    high_thresholds: Dict[str, float],
    activity_percentile: float,
) -> List[Dict[str, object]]:
    enriched: List[Dict[str, object]] = []
    for row in tqdm(rows, desc="Computing CP width features"):
        lower, upper = align_arrays(load_2d_array(row["lower_path"]), load_2d_array(row["upper_path"]))
        true_t = true_frames_from_row(row, lower.shape[1])
        lower = crop_time(lower, true_t)
        upper = crop_time(upper, true_t)
        width = np.abs(upper - lower)
        pred = crop_time(load_2d_array(row["pred_path"]), true_t)

        out = dict(row)
        out["true_frames"] = float(true_t)
        out.update(base_width_features(width, high_thresholds))
        out.update(activity_features(pred, width, activity_percentile))
        out["word_errors_per_frame"] = float(out["word_errors"]) / max(float(out["n_frames"]), 1.0)
        out["wer_times_ref_words"] = float(out["pred_wer"]) * float(out["ref_words"])
        enriched.append(out)
    return enriched


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


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


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


def numeric_array(rows: Sequence[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def candidate_feature_names(rows: Sequence[Dict[str, object]]) -> List[str]:
    excluded = {
        "pred_wer",
        "word_errors",
        "ref_words",
        "word_errors_per_frame",
        "wer_times_ref_words",
        "n_frames",
        "true_frames",
        "n_freq_bins",
    }
    names: List[str] = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            names.append(key)
    return names


def correlation_report(rows: Sequence[Dict[str, object]], features: Sequence[str]) -> List[Dict[str, object]]:
    targets = {
        "pred_wer": numeric_array(rows, "pred_wer"),
        "word_errors": numeric_array(rows, "word_errors"),
        "word_errors_per_frame": numeric_array(rows, "word_errors_per_frame"),
    }
    controls = {
        "ref_words": numeric_array(rows, "ref_words")[:, None],
        "n_frames": numeric_array(rows, "n_frames")[:, None],
        "true_frames": numeric_array(rows, "true_frames")[:, None],
        "ref_words+n_frames": np.column_stack([numeric_array(rows, "ref_words"), numeric_array(rows, "n_frames")]),
        "ref_words+true_frames": np.column_stack([numeric_array(rows, "ref_words"), numeric_array(rows, "true_frames")]),
    }

    report: List[Dict[str, object]] = []
    for feature in features:
        x = numeric_array(rows, feature)
        row: Dict[str, object] = {"feature": feature}
        for target_name, y in targets.items():
            row[f"{target_name}_pearson"] = pearsonr(x, y)
            row[f"{target_name}_spearman"] = spearmanr(x, y)
        for control_name, control_matrix in controls.items():
            row[f"pred_wer_partial_{control_name}"] = partial_corr(x, targets["pred_wer"], control_matrix)
            row[f"word_errors_partial_{control_name}"] = partial_corr(x, targets["word_errors"], control_matrix)
        report.append(row)

    report.sort(key=lambda r: abs(float(r["pred_wer_partial_ref_words+true_frames"])), reverse=True)
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
                "mean_n_frames": float(np.mean([float(row["n_frames"]) for row in chunk_rows])),
                "mean_true_frames": float(np.mean([float(row["true_frames"]) for row in chunk_rows])),
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


def write_plots(rows: Sequence[Dict[str, object]], features: Sequence[str], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping plots: {exc}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    y = numeric_array(rows, "pred_wer")
    for feature in features:
        x = numeric_array(rows, feature)
        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, s=14, alpha=0.55)
        plt.xlabel(feature)
        plt.ylabel("Whisper WER on pred head")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"scatter_{feature}_vs_pred_wer.png", dpi=160)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Second-stage CP analysis. Reads per_utterance_width_vs_wer.csv, "
            "extracts richer width features from saved arrays, and reports raw/length-controlled correlations."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--activity-percentile", type=float, default=60.0)
    parser.add_argument("--global-high-percentiles", nargs="+", type=float, default=[75.0, 90.0, 95.0])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--path-rewrite",
        nargs="*",
        default=[],
        help="Optional OLD=NEW path rewrites, useful when CSV paths come from another machine.",
    )
    parser.add_argument(
        "--plot-features",
        nargs="+",
        default=["width_active_mean", "width_frac_above_global_q90", "width_high_band_mean"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    apply_path_rewrites(rows, args.path_rewrite)
    high_thresholds = collect_global_width_thresholds(rows, args.global_high_percentiles)
    enriched = add_feature_columns(rows, high_thresholds, args.activity_percentile)
    features = candidate_feature_names(enriched)
    report = correlation_report(enriched, features)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_utterance_cp_width_features.csv", enriched)
    write_csv(args.out_dir / "feature_correlation_report.csv", report)

    top = report[: args.top_k]
    summary = {
        "n": len(enriched),
        "input_csv": str(args.input_csv),
        "activity_percentile": args.activity_percentile,
        "global_width_thresholds": high_thresholds,
        "top_features_by_abs_partial_corr_with_wer_controlling_ref_words_and_true_frames": top,
        "deciles": {row["feature"]: decile_table(enriched, str(row["feature"])) for row in top[:5]},
    }
    (args.out_dir / "feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_plots(enriched, args.plot_features, args.out_dir / "plots")

    print(f"[SAVE] {args.out_dir / 'per_utterance_cp_width_features.csv'}")
    print(f"[SAVE] {args.out_dir / 'feature_correlation_report.csv'}")
    print(f"[SAVE] {args.out_dir / 'feature_summary.json'}")
    print("\nTop features after controlling for ref_words+true_frames:")
    for row in top[: min(10, len(top))]:
        print(
            f"{row['feature']}: "
            f"partial WER={float(row['pred_wer_partial_ref_words+true_frames']):.4f}, "
            f"partial word_errors={float(row['word_errors_partial_ref_words+true_frames']):.4f}, "
            f"raw WER={float(row['pred_wer_pearson']):.4f}"
        )


if __name__ == "__main__":
    main()
