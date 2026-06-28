from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


LEN_RE = re.compile(r"\[len=(\d+)\]")
UTT_ID_RE = re.compile(r"(?:train|val|test|cal)_\d+")

NUMERIC_INPUT_COLUMNS = {
    "pred_wer",
    "word_errors",
    "ref_words",
    "true_frames",
    "width_mean",
    "width_median",
    "width_p90",
    "width_p95",
}


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
    required = {"pred_path", "pred_wer", "word_errors", "ref_words"}
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
        for key in ["pred_path", "lower_path", "upper_path", "text_path", "clean_path"]:
            if key not in row:
                continue
            value = str(row[key])
            for old, new in parsed:
                if value.startswith(old):
                    value = new + value[len(old):]
            row[key] = value


def utterance_id(path_or_name: str) -> str:
    match = UTT_ID_RE.search(Path(path_or_name).name)
    if match:
        return match.group(0)
    stem = Path(path_or_name).stem
    stem = re.sub(r"\[len=\d+\]", "", stem)
    stem = re.sub(r"_ch\d+", "", stem)
    stem = re.sub(r"_(pred|lower_alpha=[^_]+|upper_alpha=[^_]+)$", "", stem)
    return stem.strip("_-. ")


def speaker_id(path_or_name: str) -> Optional[str]:
    path = Path(path_or_name)
    if len(path.parts) >= 2:
        return path.parent.name
    return None


def true_frames_from_row(row: Dict[str, object], fallback: int) -> int:
    if "true_frames" in row and row["true_frames"] not in {"", None}:
        try:
            return min(int(float(row["true_frames"])), fallback)
        except ValueError:
            pass
    for key in ["pred_path", "clean_path", "text_path"]:
        match = LEN_RE.search(str(row.get(key, "")))
        if match:
            return min(int(match.group(1)), fallback)
    return fallback


def load_2d_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Array path does not exist: {path}. "
            "Run this where the arrays are mounted, or use --path-rewrite OLD=NEW."
        )
    arr = np.squeeze(np.load(path).astype(np.float32))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}: {path}")
    return arr


def align_pred_clean(pred: np.ndarray, clean: np.ndarray, true_t: int) -> Tuple[np.ndarray, np.ndarray]:
    if pred.shape != clean.shape and pred.T.shape == clean.shape:
        pred = pred.T
    elif pred.shape != clean.shape and clean.T.shape == pred.shape:
        clean = clean.T

    min_f = min(pred.shape[0], clean.shape[0])
    min_t = min(pred.shape[1], clean.shape[1], true_t)
    return pred[:min_f, :min_t], clean[:min_f, :min_t]


def index_clean_files(clean_root: Path, channel: str) -> Dict[Tuple[Optional[str], str], Path]:
    index: Dict[Tuple[Optional[str], str], Path] = {}
    fallback_index: Dict[Tuple[Optional[str], str], Path] = {}
    for path in sorted(clean_root.rglob("*.npy")):
        name = path.name
        if "_ch" in name and channel not in name:
            continue
        key = (speaker_id(path), utterance_id(name))
        if "_ch" in name:
            index.setdefault(key, path)
        else:
            fallback_index.setdefault(key, path)
    fallback_index.update(index)
    return fallback_index


def find_clean_path(row: Dict[str, object], clean_index: Dict[Tuple[Optional[str], str], Path]) -> Optional[Path]:
    pred_path = str(row["pred_path"])
    key = (speaker_id(pred_path), utterance_id(pred_path))
    if key in clean_index:
        return clean_index[key]
    key_without_speaker = (None, utterance_id(pred_path))
    if key_without_speaker in clean_index:
        return clean_index[key_without_speaker]
    for (speaker, uid), path in clean_index.items():
        if uid == utterance_id(pred_path):
            return path
    return None


def pred_clean_features(pred: np.ndarray, clean: np.ndarray) -> Dict[str, float]:
    diff = pred - clean
    abs_diff = np.abs(diff)
    pred_centered = pred - pred.mean()
    clean_centered = clean - clean.mean()
    denom = float(np.linalg.norm(pred_centered) * np.linalg.norm(clean_centered))
    corr = float(np.sum(pred_centered * clean_centered) / denom) if denom > 0 else float("nan")
    snr_num = float(np.sum(clean * clean))
    snr_den = float(np.sum(diff * diff))
    return {
        "pred_clean_l1_mean": float(abs_diff.mean()),
        "pred_clean_l1_median": float(np.median(abs_diff)),
        "pred_clean_l1_p90": float(np.percentile(abs_diff, 90.0)),
        "pred_clean_l1_p95": float(np.percentile(abs_diff, 95.0)),
        "pred_clean_mse": float(np.mean(diff * diff)),
        "pred_clean_rmse": float(np.sqrt(np.mean(diff * diff))),
        "pred_clean_max_abs": float(abs_diff.max()),
        "pred_clean_bias": float(diff.mean()),
        "pred_clean_corr": corr,
        "pred_clean_snr_db": float(10.0 * math.log10((snr_num + 1e-12) / (snr_den + 1e-12))),
    }


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


def feature_report(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    feature_names = [key for key in rows[0] if key.startswith("pred_clean_")]
    targets = {
        "pred_wer": numeric_array(rows, "pred_wer"),
        "word_errors": numeric_array(rows, "word_errors"),
    }
    controls = {
        "ref_words": numeric_array(rows, "ref_words")[:, None],
        "true_frames": numeric_array(rows, "true_frames")[:, None],
        "ref_words+true_frames": np.column_stack([numeric_array(rows, "ref_words"), numeric_array(rows, "true_frames")]),
    }
    report: List[Dict[str, object]] = []
    for feature in feature_names:
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


def maybe_plot(rows: Sequence[Dict[str, object]], features: Sequence[str], out_dir: Path) -> None:
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
        description="Correlate Whisper WER on pred with pred-vs-clean spectrogram distance."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="per_utterance_width_vs_wer.csv from the CP analysis.")
    parser.add_argument("--clean-root", type=Path, required=True, help="Root containing clean mel/spec .npy files for the same split.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--path-rewrite",
        nargs="*",
        default=[],
        help="Optional OLD=NEW path rewrites, useful when CSV paths come from another machine.",
    )
    parser.add_argument(
        "--plot-features",
        nargs="+",
        default=["pred_clean_l1_mean", "pred_clean_rmse", "pred_clean_corr", "pred_clean_snr_db"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    apply_path_rewrites(rows, args.path_rewrite)
    clean_index = index_clean_files(args.clean_root, args.channel)
    print(f"[INFO] indexed clean files: {len(clean_index)}")

    enriched: List[Dict[str, object]] = []
    missing = 0
    for row in tqdm(rows, desc="Computing pred-clean distances"):
        clean_path = find_clean_path(row, clean_index)
        if clean_path is None:
            missing += 1
            continue
        pred = load_2d_array(row["pred_path"])
        clean = load_2d_array(clean_path)
        true_t = true_frames_from_row(row, min(pred.shape[-1], clean.shape[-1]))
        pred, clean = align_pred_clean(pred, clean, true_t)
        out = dict(row)
        out["clean_path"] = str(clean_path)
        out["true_frames"] = float(true_t)
        out.update(pred_clean_features(pred, clean))
        enriched.append(out)

    if not enriched:
        raise RuntimeError(f"No pred-clean pairs matched. Missing clean rows: {missing}")
    if missing:
        print(f"[WARN] skipped rows with no clean match: {missing}")

    report = feature_report(enriched)
    top = report[: args.top_k]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_utterance_pred_clean_diff.csv", enriched)
    write_csv(args.out_dir / "pred_clean_diff_correlation_report.csv", report)
    summary = {
        "n": len(enriched),
        "input_csv": str(args.input_csv),
        "clean_root": str(args.clean_root),
        "missing_clean_matches": missing,
        "top_features_by_abs_partial_corr_with_wer_controlling_ref_words_and_true_frames": top,
        "deciles": {row["feature"]: decile_table(enriched, str(row["feature"])) for row in top[:5]},
    }
    (args.out_dir / "pred_clean_diff_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    maybe_plot(enriched, args.plot_features, args.out_dir / "plots")

    print(f"[SAVE] {args.out_dir / 'per_utterance_pred_clean_diff.csv'}")
    print(f"[SAVE] {args.out_dir / 'pred_clean_diff_correlation_report.csv'}")
    print(f"[SAVE] {args.out_dir / 'pred_clean_diff_summary.json'}")
    print("\nTop pred-clean distance features:")
    for row in top:
        print(
            f"{row['feature']}: "
            f"partial WER={float(row['pred_wer_partial_ref_words+true_frames']):.4f}, "
            f"partial word_errors={float(row['word_errors_partial_ref_words+true_frames']):.4f}, "
            f"raw WER={float(row['pred_wer_pearson']):.4f}"
        )


if __name__ == "__main__":
    main()
