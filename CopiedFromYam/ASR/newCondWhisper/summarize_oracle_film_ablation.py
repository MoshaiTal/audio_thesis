#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional


def pct(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{100.0 * x:.3f}"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def last_history_row(run_dir: Path) -> Dict[str, Any]:
    history = load_json(run_dir / "history.json")
    if isinstance(history, list) and history:
        return history[-1]
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize oracle pred-clean FiLM ablation runs.")
    parser.add_argument("base_dir", type=str, help="Directory containing run subdirectories.")
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    rows = []
    for run_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        final_summary = load_json(run_dir / "final_summary.json") or {}
        initial = final_summary.get("initial", {})
        final_val = final_summary.get("final_val", {})
        hist_last = last_history_row(run_dir)
        pred_base = final_summary.get("test_pred_baseline") or initial.get("pred_baseline_val", {})
        clean_base = final_summary.get("test_clean_baseline") or initial.get("clean_baseline_val", {})

        test_model = None
        for key, value in final_summary.items():
            if (
                key.startswith("test_")
                and key not in {"test_pred_baseline", "test_input_baseline", "test_clean_baseline"}
                and isinstance(value, dict)
            ):
                test_model = value
        test_model = test_model or {}

        rows.append(
            {
                "run": run_dir.name,
                "input": initial.get("input_source") or hist_last.get("input_source", ""),
                "condition": initial.get("condition_source") or hist_last.get("condition_source", ""),
                "adapter": initial.get("adapter_mode") or hist_last.get("adapter_mode", ""),
                "conditioner": initial.get("conditioner_mode") or hist_last.get("conditioner_mode", ""),
                "position": initial.get("injection_position") or hist_last.get("injection_position", ""),
                "layers": ",".join(str(x) for x in (initial.get("selected_layers") or hist_last.get("selected_layers") or [])),
                "pred_test_wer_%": pct(pred_base.get("wer")),
                "input_test_wer_%": pct((final_summary.get("test_input_baseline") or {}).get("wer")),
                "clean_test_wer_%": pct(clean_base.get("wer")),
                "film_test_wer_%": pct(test_model.get("wer")),
                "test_delta_vs_pred_pp": pct(final_summary.get("test_delta_vs_pred")),
                "test_delta_vs_input_pp": pct(final_summary.get("test_delta_vs_input")),
                "best_val_delta_vs_pred_pp": pct(final_summary.get("best_val_delta_vs_pred")),
                "final_val_wer_%": pct(final_val.get("wer")),
                "last_true_minus_zero_val_pp": pct(hist_last.get("true_minus_zero_wer")),
                "last_true_minus_shuffled_val_pp": pct(hist_last.get("true_minus_shuffled_wer")),
                "last_update_abs": hist_last.get("last_debug", {}).get("mean_layer_update_abs", ""),
                "last_true_minus_shuffled_ce": hist_last.get("train_true_minus_shuffled_asr_loss", ""),
                "last_true_minus_zero_ce": hist_last.get("train_true_minus_zero_asr_loss", ""),
            }
        )

    fields = [
        "run",
        "input",
        "condition",
        "adapter",
        "conditioner",
        "position",
        "layers",
        "pred_test_wer_%",
        "input_test_wer_%",
        "clean_test_wer_%",
        "film_test_wer_%",
        "test_delta_vs_pred_pp",
        "test_delta_vs_input_pp",
        "best_val_delta_vs_pred_pp",
        "final_val_wer_%",
        "last_true_minus_zero_val_pp",
        "last_true_minus_shuffled_val_pp",
        "last_update_abs",
        "last_true_minus_shuffled_ce",
        "last_true_minus_zero_ce",
    ]

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
