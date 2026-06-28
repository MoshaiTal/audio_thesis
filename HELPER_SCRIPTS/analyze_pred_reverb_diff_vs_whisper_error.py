from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable

from HELPER_SCRIPTS.analyze_pred_clean_diff_vs_whisper_error import (
    align_pred_clean,
    apply_path_rewrites,
    decile_table,
    find_clean_path,
    index_clean_files,
    load_2d_array,
    load_rows,
    numeric_array,
    partial_corr,
    pearsonr,
    spearmanr,
    true_frames_from_row,
    write_csv,
)


def pred_reverb_features(pred: np.ndarray, reverb: np.ndarray) -> Dict[str, float]:
    diff = pred - reverb
    abs_diff = np.abs(diff)
    pred_centered = pred - pred.mean()
    reverb_centered = reverb - reverb.mean()
    denom = float(np.linalg.norm(pred_centered) * np.linalg.norm(reverb_centered))
    corr = float(np.sum(pred_centered * reverb_centered) / denom) if denom > 0 else float("nan")
    snr_num = float(np.sum(reverb * reverb))
    snr_den = float(np.sum(diff * diff))
    return {
        "pred_reverb_l1_mean": float(abs_diff.mean()),
        "pred_reverb_l1_median": float(np.median(abs_diff)),
        "pred_reverb_l1_p90": float(np.percentile(abs_diff, 90.0)),
        "pred_reverb_l1_p95": float(np.percentile(abs_diff, 95.0)),
        "pred_reverb_mse": float(np.mean(diff * diff)),
        "pred_reverb_rmse": float(np.sqrt(np.mean(diff * diff))),
        "pred_reverb_max_abs": float(abs_diff.max()),
        "pred_reverb_bias": float(diff.mean()),
        "pred_reverb_corr": corr,
        "pred_reverb_change_snr_db": float(10.0 * math.log10((snr_num + 1e-12) / (snr_den + 1e-12))),
    }


def feature_report(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    feature_names = [key for key in rows[0] if key.startswith("pred_reverb_")]
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
        description="Correlate Whisper WER on pred with pred-vs-reverberant spectrogram distance."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="per_utterance_width_vs_wer.csv from the CP analysis.")
    parser.add_argument("--reverb-root", type=Path, required=True, help="Root containing reverberant mel/spec .npy files for the same split.")
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
        default=["pred_reverb_l1_mean", "pred_reverb_rmse", "pred_reverb_corr", "pred_reverb_change_snr_db"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    apply_path_rewrites(rows, args.path_rewrite)
    reverb_index = index_clean_files(args.reverb_root, args.channel)
    print(f"[INFO] indexed reverb files: {len(reverb_index)}")

    enriched: List[Dict[str, object]] = []
    missing = 0
    for row in tqdm(rows, desc="Computing pred-reverb distances"):
        reverb_path = find_clean_path(row, reverb_index)
        if reverb_path is None:
            missing += 1
            continue
        pred = load_2d_array(row["pred_path"])
        reverb = load_2d_array(reverb_path)
        true_t = true_frames_from_row(row, min(pred.shape[-1], reverb.shape[-1]))
        pred, reverb = align_pred_clean(pred, reverb, true_t)
        out = dict(row)
        out["reverb_path"] = str(reverb_path)
        out["true_frames"] = float(true_t)
        out.update(pred_reverb_features(pred, reverb))
        enriched.append(out)

    if not enriched:
        raise RuntimeError(f"No pred-reverb pairs matched. Missing reverb rows: {missing}")
    if missing:
        print(f"[WARN] skipped rows with no reverb match: {missing}")

    report = feature_report(enriched)
    top = report[: args.top_k]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "per_utterance_pred_reverb_diff.csv", enriched)
    write_csv(args.out_dir / "pred_reverb_diff_correlation_report.csv", report)
    summary = {
        "n": len(enriched),
        "input_csv": str(args.input_csv),
        "reverb_root": str(args.reverb_root),
        "missing_reverb_matches": missing,
        "top_features_by_abs_partial_corr_with_wer_controlling_ref_words_and_true_frames": top,
        "deciles": {row["feature"]: decile_table(enriched, str(row["feature"])) for row in top[:5]},
    }
    (args.out_dir / "pred_reverb_diff_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    maybe_plot(enriched, args.plot_features, args.out_dir / "plots")

    print(f"[SAVE] {args.out_dir / 'per_utterance_pred_reverb_diff.csv'}")
    print(f"[SAVE] {args.out_dir / 'pred_reverb_diff_correlation_report.csv'}")
    print(f"[SAVE] {args.out_dir / 'pred_reverb_diff_summary.json'}")
    print("\nTop pred-reverb distance features:")
    for row in top:
        print(
            f"{row['feature']}: "
            f"partial WER={float(row['pred_wer_partial_ref_words+true_frames']):.4f}, "
            f"partial word_errors={float(row['word_errors_partial_ref_words+true_frames']):.4f}, "
            f"raw WER={float(row['pred_wer_pearson']):.4f}"
        )


if __name__ == "__main__":
    main()
