import os
import pathlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from jiwer import wer
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from CopiedFromYam.ASR.utils import build_paired_file_rows

np.random.seed(55)
random.seed(55)


def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    return processor, model


def transcribe_with_probs(processor, model, data: torch.Tensor, device: str,
                          data_type: str = "SPECTROGRAM",
                          num_beams: int = 5,
                          no_repeat_ngram_size: int = 2) -> str:
    model.config.forced_decoder_ids = None
    model.to(device)

    if data_type == "AUDIO":
        inputs = processor(data, return_tensors="pt", sampling_rate=16000).to(device)
    else:
        inputs = data.to(device)
        inputs = {"input_features": inputs if inputs.ndim == 3 else inputs.unsqueeze(0)}

    outputs = model.generate(
        **inputs,
        return_dict_in_generate=True,
        max_length=3000,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
        num_beams=num_beams,
    )
    if data_type == "AUDIO":
        return processor.decode(outputs.sequences[0], skip_special_tokens=True).lower()
    return processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].lower()


def calculate_one_wer(preds: str, targets: str) -> float:
    return wer(preds, targets)


def load_signal(path: str):
    return librosa.load(path, sr=16000) if path.endswith((".wav", ".flac")) else (np.load(path), 16000)


def utterance_width_score(lower_signal: np.ndarray, upper_signal: np.ndarray, mode: str = "mean") -> float:
    width = np.abs(upper_signal - lower_signal)
    if mode == "mean":
        return float(width.mean())
    if mode == "median":
        return float(np.median(width))
    if mode == "p90":
        return float(np.percentile(width, 90.0))
    raise ValueError(f"Unsupported WIDTH_SCORE_MODE: {mode}")


def collect_rows_for_split(split: str,
                           output_root: str,
                           clean_root: str,
                           reverb_root: str,
                           targets_root: str,
                           wanted_ch: str,
                           wanted_alpha: str) -> List[Dict[str, str]]:
    folders = sorted(os.listdir(os.path.join(output_root, split)))
    all_rows: List[Dict[str, str]] = []

    for folder in folders:
        output_folder_path = os.path.join(output_root, split, folder)
        targets_folder_path = os.path.join(targets_root, split, folder)
        clean_folder_path = os.path.join(clean_root, split, folder)
        reverb_folder_path = os.path.join(reverb_root, split, folder)

        rows = build_paired_file_rows(
            output_folder_path=output_folder_path,
            clean_folder_path=clean_folder_path,
            reverb_folder_path=reverb_folder_path,
            targets_folder_path=targets_folder_path,
            WANTED_CH=wanted_ch,
            WANTED_ALPHA=wanted_alpha,
        )
        print(f"[PAIRING] split={split} folder={folder}: matched {len(rows)} rows for alpha={wanted_alpha}, ch={wanted_ch}")
        all_rows.extend(rows)
    return all_rows


def decode_rows(rows: List[Dict[str, str]], processor, model, device: str,
                data_type: str,
                allow_print: bool,
                width_score_mode: str) -> pd.DataFrame:
    data = {
        "stem": [],
        "target": [],
        "reverb_text": [],
        "pred_text": [],
        "clean_text": [],
        "reverb_wer": [],
        "pred_wer": [],
        "clean_wer": [],
        "width_score": [],
    }

    for row in rows:
        pred_signal, _ = load_signal(row["pred"])
        upper_signal, _ = load_signal(row["upper"])
        lower_signal, _ = load_signal(row["lower"])
        clean_signal, _ = load_signal(row["clean"])
        reverb_signal, _ = load_signal(row["reverb"])

        with open(row["text"], encoding="utf-8") as f:
            target_text = f.read().strip().lower()

        pred_text = transcribe_with_probs(processor, model, torch.from_numpy(pred_signal).float(), device, data_type)
        reverb_text = transcribe_with_probs(processor, model, torch.from_numpy(reverb_signal).float(), device, data_type)
        clean_text = transcribe_with_probs(processor, model, torch.from_numpy(clean_signal).float(), device, data_type)

        score = utterance_width_score(lower_signal, upper_signal, mode=width_score_mode)

        data["stem"].append(row["stem"])
        data["target"].append(target_text)
        data["reverb_text"].append(reverb_text)
        data["pred_text"].append(pred_text)
        data["clean_text"].append(clean_text)
        data["reverb_wer"].append(calculate_one_wer(reverb_text, target_text))
        data["pred_wer"].append(calculate_one_wer(pred_text, target_text))
        data["clean_wer"].append(calculate_one_wer(clean_text, target_text))
        data["width_score"].append(score)

        if allow_print:
            print("\n----------------------------------------------\n")
            print("stem:", row["stem"])
            print("target:", target_text)
            print("reverb:", reverb_text)
            print("pred  :", pred_text)
            print("clean :", clean_text)
            print("width :", score)

    return pd.DataFrame(data)


def fit_tau_on_cal(cal_df: pd.DataFrame,
                   mode: str = "oracle_best_on_cal",
                   percentile: float = 60.0) -> Tuple[float, Dict[str, float]]:
    scores = cal_df["width_score"].to_numpy()

    if mode == "percentile":
        tau = float(np.percentile(scores, percentile))
        gated = np.where(scores <= tau, cal_df["pred_wer"].to_numpy(), cal_df["reverb_wer"].to_numpy())
        return tau, {
            "cal_gated_mean_wer": float(np.mean(gated)),
            "cal_pred_mean_wer": float(cal_df["pred_wer"].mean()),
            "cal_reverb_mean_wer": float(cal_df["reverb_wer"].mean()),
            "fit_mode": mode,
            "percentile": percentile,
        }

    if mode == "oracle_best_on_cal":
        percentiles = list(range(5, 100, 5))
        best_tau = None
        best_wer = None
        best_p = None
        for p in percentiles:
            tau = float(np.percentile(scores, p))
            gated = np.where(scores <= tau, cal_df["pred_wer"].to_numpy(), cal_df["reverb_wer"].to_numpy())
            mean_wer = float(np.mean(gated))
            if best_wer is None or mean_wer < best_wer:
                best_wer = mean_wer
                best_tau = tau
                best_p = p
        return float(best_tau), {
            "cal_gated_mean_wer": float(best_wer),
            "cal_pred_mean_wer": float(cal_df["pred_wer"].mean()),
            "cal_reverb_mean_wer": float(cal_df["reverb_wer"].mean()),
            "fit_mode": mode,
            "best_percentile": float(best_p),
        }

    raise ValueError(f"Unsupported TAU_FIT_MODE: {mode}")


def apply_gating(df: pd.DataFrame, tau: float) -> pd.DataFrame:
    out = df.copy()
    use_pred = out["width_score"] <= tau
    out["gating_choice"] = np.where(use_pred, "pred", "reverb")
    out["gated_text"] = np.where(use_pred, out["pred_text"], out["reverb_text"])
    out["gated_wer"] = np.where(use_pred, out["pred_wer"], out["reverb_wer"])
    out["tau"] = tau
    return out


def save_results(df: pd.DataFrame,
                 split_name: str,
                 results_dir: Path,
                 wanted_alpha: str,
                 wanted_ch: str,
                 width_score_mode: str,
                 tau_fit_mode: str,
                 tau: float,
                 fit_info: Dict[str, float]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{split_name}_alpha={wanted_alpha}_{wanted_ch}_utterance_gating_{width_score_mode}_{tau_fit_mode}"

    per_file_csv = results_dir / f"wer_per_file_{tag}.csv"
    summary_csv = results_dir / f"wer_summary_{tag}.csv"

    df.to_csv(per_file_csv, index=False)

    summary_rows = [
        {"metric": "reverb_wer", "mean_wer": float(df["reverb_wer"].mean()), "tau": tau},
        {"metric": "pred_wer", "mean_wer": float(df["pred_wer"].mean()), "tau": tau},
        {"metric": "gated_wer", "mean_wer": float(df["gated_wer"].mean()), "tau": tau},
        {"metric": "clean_wer", "mean_wer": float(df["clean_wer"].mean()), "tau": tau},
        {"metric": "fraction_pred_chosen", "mean_wer": float((df["gating_choice"] == "pred").mean()), "tau": tau},
    ]

    for k, v in fit_info.items():
        summary_rows.append({"metric": k, "mean_wer": float(v) if isinstance(v, (int, float, np.floating)) else v, "tau": tau})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved per-file results to:", per_file_csv)
    print("Saved summary to:", summary_csv)
    print("\nMean WER summary:")
    print(summary_df[summary_df["metric"].isin(["reverb_wer", "pred_wer", "gated_wer", "clean_wer", "fraction_pred_chosen"])])


def main():
    processor, wspr_model = load_whisper_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wspr_model.to(device)

    cal_rows = collect_rows_for_split(
        split=CAL_SPLIT,
        output_root=OUTPUT_ROOT,
        clean_root=CLEAN_ROOT,
        reverb_root=REVERB_ROOT,
        targets_root=TARGETS_ROOT,
        wanted_ch=WANTED_CH,
        wanted_alpha=WANTED_ALPHA,
    )
    cal_df = decode_rows(cal_rows, processor, wspr_model, device, DATA_TYPE, ALLOW_PRINT_CAL, WIDTH_SCORE_MODE)
    tau, fit_info = fit_tau_on_cal(cal_df, mode=TAU_FIT_MODE, percentile=WIDTH_PERCENTILE)

    print(f"\n[TAU] selected tau={tau:.6f} using mode={TAU_FIT_MODE}")
    print("[TAU] fit info:", fit_info)

    test_rows = collect_rows_for_split(
        split=EVAL_SPLIT,
        output_root=OUTPUT_ROOT,
        clean_root=CLEAN_ROOT,
        reverb_root=REVERB_ROOT,
        targets_root=TARGETS_ROOT,
        wanted_ch=WANTED_CH,
        wanted_alpha=WANTED_ALPHA,
    )
    test_df = decode_rows(test_rows, processor, wspr_model, device, DATA_TYPE, ALLOW_PRINT_EVAL, WIDTH_SCORE_MODE)
    gated_df = apply_gating(test_df, tau)

    save_results(
        gated_df,
        split_name=EVAL_SPLIT,
        results_dir=RESULTS_DIR,
        wanted_alpha=WANTED_ALPHA,
        wanted_ch=WANTED_CH,
        width_score_mode=WIDTH_SCORE_MODE,
        tau_fit_mode=TAU_FIT_MODE,
        tau=tau,
        fit_info=fit_info,
    )


if __name__ == "__main__":
    TARGETS_ROOT = r"/storage/tal/thesis/DataBase_BIUREV/transcription_matched"
    CLEAN_ROOT = r"/storage/tal/thesis/DataBase_BIUREV/clean_melspec"
    REVERB_ROOT = r"/storage/tal/thesis/DataBase_BIUREV/reverb_melspec"
    OUTPUT_ROOT = r"/storage/tal/thesis/DataBase_BIUREV/dereverb_mel/calibrated_rcps"
    RESULTS_DIR = Path("/storage/tal/thesis/whisper_results")

    CAL_SPLIT = "cal"
    EVAL_SPLIT = "test"

    DATA_TYPE = "SPECTROGRAM"   # or "AUDIO" if you switch to wav paths
    WANTED_CH = "ch1"
    WANTED_ALPHA = "0.2"

    WIDTH_SCORE_MODE = "mean"   # "mean", "median", "p90"
    TAU_FIT_MODE = "percentile"   # "oracle_best_on_cal" or "percentile"
    WIDTH_PERCENTILE = 60.0      # used only when TAU_FIT_MODE == "percentile"

    ALLOW_PRINT_CAL = False
    ALLOW_PRINT_EVAL = True

    main()
