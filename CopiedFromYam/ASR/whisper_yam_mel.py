import os
import re
import pathlib
import random
from typing import Optional

import inflect
import librosa
import numpy as np
import pandas as pd
import torch
from jiwer import wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoFeatureExtractor

from CopiedFromYam.ASR.utils import build_paired_file_rows

np.random.seed(55)
random.seed(55)


def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
    return processor, model, feature_extractor


class LogitAveragingModel(WhisperForConditionalGeneration):
    def __init__(self, original_model: WhisperForConditionalGeneration, logits_weights=None):
        super().__init__(original_model.config)
        self.load_state_dict(original_model.state_dict())
        self.logits_weights = logits_weights

    def forward(self, *args, **kwargs):
        if self.logits_weights is None:
            logits_weights = (torch.ones(NUM_OF_SLICES) / NUM_OF_SLICES)
        else:
            logits_weights = self.logits_weights

        outputs = super().forward(*args, **kwargs)

        if hasattr(outputs, 'logits') and outputs.logits is not None:
            num_beams = 5
            batch_size = int(outputs.logits.shape[0] / num_beams)
            reshaped_logits = outputs.logits.view(batch_size, num_beams, outputs.logits.size(1), outputs.logits.size(2))
            logits_weights = logits_weights.to(outputs.logits.device)
            averaged_logits = (reshaped_logits * logits_weights.view(batch_size, 1, 1, 1)).sum(dim=0)
            manipulated_logits = averaged_logits.repeat(batch_size, 1, 1)
            outputs.logits = manipulated_logits
        return outputs


def transcribe_with_probs(processor, model, data: torch.Tensor, device: str,
                          num_beams: int = 5, no_repeat_ngram_size: int = 2):
    model.config.forced_decoder_ids = None
    model.to(device)

    if DATA_TYPE == "AUDIO":
        inputs = processor(data, return_tensors="pt", sampling_rate=16000).to(device)
    else:
        inputs = data.to(device)
        inputs = {"input_features": inputs if inputs.ndim == 3 else inputs.unsqueeze(0)}

    if (data.ndim > 1 and DATA_TYPE == "AUDIO") or (data.ndim > 2 and DATA_TYPE == "SPECTROGRAM"):
        wrapped_model = LogitAveragingModel(model).to(device)
        outputs = wrapped_model.generate(
            input_features=inputs['input_features'],
            max_length=3000,
            return_dict_in_generate=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            num_beams=num_beams,
        )
        return processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].lower()

    outputs = model.generate(
        **inputs,
        return_dict_in_generate=True,
        max_length=3000,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
        num_beams=num_beams,
    )
    return processor.decode(outputs.sequences[0], skip_special_tokens=True).lower()


def transcribe_with_probs2(processor, model, data: torch.Tensor, device: str, slices_weights: torch.Tensor,
                           num_beams: int = 5, no_repeat_ngram_size: int = 2):
    model.to(device)
    if DATA_TYPE == "AUDIO":
        inputs = processor(data, return_tensors="pt", sampling_rate=16000).to(device)
    else:
        inputs = data.to(device)
        inputs = {"input_features": inputs if inputs.ndim == 3 else inputs.unsqueeze(0)}

    if (data.ndim > 1 and DATA_TYPE == "AUDIO") or (data.ndim > 2 and DATA_TYPE == "SPECTROGRAM"):
        wrapped_model = LogitAveragingModel(model, slices_weights).to(device)
        outputs = wrapped_model.generate(
            **inputs,
            max_length=3000,
            return_dict_in_generate=True,
            output_scores=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
            num_beams=num_beams,
        )
        return processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].lower()

    outputs = model.generate(
        **inputs,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=3000,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
        num_beams=num_beams,
    )
    return processor.decode(outputs.sequences[0], skip_special_tokens=True).lower()


def replace_numbers_with_words(text):
    p = inflect.engine()
    return re.sub(r'\d+', lambda x: p.number_to_words(int(x.group())), text)


def calculate_one_wer(preds, targets):
    return wer(preds, targets)


def create_intervals(lower, upper):
    if ERORR_TRIAL[0]:
        error = 1.1 * np.abs(upper - lower)
        noisy_error = error + np.random.normal(0, ERORR_TRIAL[1], size=error.shape)
        start_tensor = lower - noisy_error
        end_tensor = lower + noisy_error
    else:
        start_tensor = lower
        end_tensor = upper
    return start_tensor, end_tensor


def lin_space_hanukia(lower_path, upper_path, NUM_OF_SLICES=10):
    if isinstance(lower_path, np.ndarray):
        lower = lower_path
        upper = upper_path
    else:
        (lower, _), (upper, _) = librosa.load(lower_path, sr=16000), librosa.load(upper_path, sr=16000)
    start_tensor, end_tensor = create_intervals(lower, upper)
    interpolated_tensors = []
    for i in range(NUM_OF_SLICES):
        alpha = i / (NUM_OF_SLICES - 1)
        interpolated_tensor = (1 - alpha) * start_tensor + alpha * end_tensor
        interpolated_tensors.append(interpolated_tensor.astype(np.float32))
    return interpolated_tensors


def create_random_interpolated_tensors(lower_path, upper_path, NUM_OF_SLICES=10):
    if isinstance(lower_path, np.ndarray):
        lower = lower_path
        upper = upper_path
    else:
        (lower, _), (upper, _) = librosa.load(lower_path, sr=16000), librosa.load(upper_path, sr=16000)
    start_tensor, end_tensor = create_intervals(lower, upper)
    random_tensors = []
    for _ in range(NUM_OF_SLICES):
        alpha = np.random.random(size=start_tensor.shape).astype(np.float32)
        random_tensor = (1 - alpha) * start_tensor + alpha * end_tensor
        random_tensors.append(random_tensor.astype(np.float32))
    return random_tensors


def choose_interpolate_type(name):
    if name == 'Linear':
        return lin_space_hanukia
    if name == 'Random':
        return create_random_interpolated_tensors
    raise ValueError("Interpolate name is not valid. Choose one of 'Linear' or 'Random'")


# def calc_slice_weight(cur_index, pred_index):
#     dist = np.abs(cur_index - pred_index) + 1
#     return float((np.random.random(1) / dist)[0])

def calc_slice_weight(cur_index, pred_index):
    dist = np.abs(cur_index - pred_index) + 1
    return 1.0 / dist


def load_signal(path: str):
    return librosa.load(path, sr=16000) if path.endswith(('.wav', '.flac')) else (np.load(path), 16000)


def main():
    processor, wspr_model, feature_extractor = load_whisper_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wspr_model.to(device)

    folders = sorted(os.listdir(output_path))
    d = {}

    for folder in folders:
        output_folder_path = os.path.join(output_path, folder)
        targets_folder_path = os.path.join(targets_path, folder)
        clean_folder_path = os.path.join(clean_path, folder)
        reverb_folder_path = os.path.join(reverb_path, folder)

        rows = build_paired_file_rows(
            output_folder_path=output_folder_path,
            clean_folder_path=clean_folder_path,
            reverb_folder_path=reverb_folder_path,
            targets_folder_path=targets_folder_path,
            WANTED_CH=WANTED_CH,
            WANTED_ALPHA=WANTED_ALPHA,
        )

        print(f"[PAIRING] folder={folder}: matched {len(rows)} rows for alpha={WANTED_ALPHA}, ch={WANTED_CH}")

        for row in rows:
            pred_signal, _ = load_signal(row['pred'])
            upper_signal, _ = load_signal(row['upper'])
            lower_signal, _ = load_signal(row['lower'])
            clean_signal, _ = load_signal(row['clean'])
            reverb_signal, _ = load_signal(row['reverb'])

            # Use the model's calibrated lower/upper outputs, not oracle error-based intervals.
            intervals = np.array(
                choose_interpolate_type(INTERPOLATE_NAME)(lower_signal, upper_signal, NUM_OF_SLICES=NUM_OF_SLICES)
            )

            transcription_pred = transcribe_with_probs(processor, wspr_model, torch.from_numpy(pred_signal).float(), device)
            transcription_clean = transcribe_with_probs(processor, wspr_model, torch.from_numpy(clean_signal).float(), device)
            transcription_reverb = transcribe_with_probs(processor, wspr_model, torch.from_numpy(reverb_signal).float(), device)

            with open(row['text'], encoding='utf-8') as f:
                transcription_targets = f.read().strip().lower()

            pred_idx = int(np.abs(intervals - pred_signal).mean(axis=(1, 2)).argmin())
            slices_weights = [calc_slice_weight(i, pred_idx) for i in range(intervals.shape[0])]
            slices_weights = torch.tensor(slices_weights, dtype=torch.float32)
            slices_weights = slices_weights / slices_weights.sum()

            transcription_spec_slices_combined = transcribe_with_probs2(
                processor,
                wspr_model,
                torch.from_numpy(intervals).float(),
                device,
                slices_weights=slices_weights,
            )

            if ALLOW_PRINT:
                print("\n----------------------------------------------\n")
                print("stem:", row['stem'])
                print("target:", transcription_targets)
                print("reverb:", transcription_reverb)
                print("pred  :", transcription_pred)
                print("clean :", transcription_clean)
                print("comb  :", transcription_spec_slices_combined)

            d.setdefault('stem', []).append(row['stem'])
            d.setdefault('target', []).append(transcription_targets)
            d.setdefault('reverb_text', []).append(transcription_reverb)
            d.setdefault('pred_text', []).append(transcription_pred)
            d.setdefault('clean_text', []).append(transcription_clean)
            d.setdefault('comb_text', []).append(transcription_spec_slices_combined)

            d.setdefault('reverb_wer', []).append(calculate_one_wer(transcription_reverb, transcription_targets))
            d.setdefault('pred_wer', []).append(calculate_one_wer(transcription_pred, transcription_targets))
            d.setdefault('clean_wer', []).append(calculate_one_wer(transcription_clean, transcription_targets))
            d.setdefault('comb_wer', []).append(calculate_one_wer(transcription_spec_slices_combined, transcription_targets))

    # df = pd.DataFrame(d)
    # print(df.mean(axis=0))

    df = pd.DataFrame(d)

    results_dir = pathlib.Path("/storage/tal/thesis/whisper_results")
    results_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{pathlib.Path(output_path).name}_alpha={WANTED_ALPHA}_{WANTED_CH}_{INTERPOLATE_NAME}"
    per_file_csv = results_dir / f"wer_per_file_{tag}.csv"
    summary_csv = results_dir / f"wer_summary_{tag}.csv"

    df.to_csv(per_file_csv, index=False)

    wer_cols = ["reverb_wer", "pred_wer", "clean_wer", "comb_wer"]
    wer_cols = [c for c in wer_cols if c in df.columns]

    summary_df = pd.DataFrame({
        "metric": wer_cols,
        "mean_wer": [df[c].mean() for c in wer_cols],
    })
    summary_df.to_csv(summary_csv, index=False)

    print("\nSaved per-file results to:", per_file_csv)
    print("Saved summary WERs to:", summary_csv)
    print("\nMean WER summary:")
    print(summary_df)


if __name__ == "__main__":
    targets_path = r'/storage/tal/thesis/DataBase_BIUREV/transcription_matched/test'
    clean_path = r'/storage/tal/thesis/DataBase_BIUREV/clean_melspec/test'
    reverb_path = r'/storage/tal/thesis/DataBase_BIUREV/reverb_melspec/test'
    output_path = r'/storage/tal/thesis/DataBase_BIUREV/dereverb_mel/calibrated_rcps/test'

    NUM_OF_SLICES = 10
    WANTED_CH = "ch1"
    WANTED_ALPHA = "0.5"   # choose the calibrated interval to use
    ERORR_TRIAL = [False, 0.01]
    ALLOW_PRINT = True
    DATA_TYPE = "SPECTROGRAM"  # or "AUDIO"

    for INTERPOLATE_NAME in ['Linear']:
        print(f"######################### {INTERPOLATE_NAME} ##################")
        main()
