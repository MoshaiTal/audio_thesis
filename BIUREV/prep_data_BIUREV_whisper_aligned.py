#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import random

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from sklearn.model_selection import train_test_split
from transformers import WhisperFeatureExtractor


# ----------------- Paths -----------------

dev_clean_dir = "/storage/tal/thesis/BIUREV/dev-clean"
test_clean_dir = "/storage/tal/thesis/BIUREV/test-clean"
rir_dir = "/storage/tal/thesis/raw_rirs"

db_root = "/storage/tal/thesis/DataBase_BIUREV_whisper_aligned"
output_clean_wav_dir = f"{db_root}/clean_rec"
output_reverb_wav_dir = f"{db_root}/reverb_rec"
output_clean_spec_dir = f"{db_root}/clean_spec"
output_reverb_spec_dir = f"{db_root}/reverb_spec"
output_clean_mel_dir = f"{db_root}/clean_melspec"
output_reverb_mel_dir = f"{db_root}/reverb_melspec"
output_clean_whisper_mel_dir = f"{db_root}/clean_whisper_melspec"
output_reverb_whisper_mel_dir = f"{db_root}/reverb_whisper_melspec"


# ----------------- Backend-matched feature parameters -----------------

sr = 16000

# Whisper feature extractor settings at 16 kHz:
# 25 ms window = 400 samples, 10 ms hop = 160 samples.
WHISPER_N_FFT = 400
WHISPER_HOP_LENGTH = 160
WHISPER_WIN_LENGTH = 400
WHISPER_WINDOW = "hann"

# The original U-Net expects a 256-bin frequency image. Whisper-aligned STFT
# has 201 one-sided bins for n_fft=400, so we pad frequency to 256.
SPEC_FREQ_BINS_FOR_UNET = 256

mics_num = 1
max_duration_sec = 30
TARGET_T = 3000

# Whisper uses power mel and log10 internally with a floor around 1e-10.
# For the STFT input, use log10 power on the same STFT grid so the frontend
# input is as compatible as possible with the downstream mel target.
SPEC_EPS = 1e-10
SPEC_PAD_VALUE = float(np.log10(SPEC_EPS))

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")


def pad_or_trim_time(x: np.ndarray, target_t: int = TARGET_T, pad_value: float = 0.0):
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    _, t = x.shape
    valid_t = min(t, target_t)
    if t > target_t:
        return x[:, :target_t], valid_t
    if t == target_t:
        return x, valid_t
    return np.pad(x, ((0, 0), (0, target_t - t)), mode="constant", constant_values=pad_value), valid_t


def pad_frequency_for_unet(x: np.ndarray, freq_bins: int = SPEC_FREQ_BINS_FOR_UNET, pad_value: float = SPEC_PAD_VALUE):
    if x.shape[0] > freq_bins:
        return x[:freq_bins]
    if x.shape[0] == freq_bins:
        return x
    return np.pad(x, ((0, freq_bins - x.shape[0]), (0, 0)), mode="constant", constant_values=pad_value)


def compute_features(audio: np.ndarray):
    # STFT/log-power path aligned to Whisper's frame grid.
    stft = librosa.stft(
        audio,
        n_fft=WHISPER_N_FFT,
        hop_length=WHISPER_HOP_LENGTH,
        win_length=WHISPER_WIN_LENGTH,
        window=WHISPER_WINDOW,
        center=True,
    )
    power = np.abs(stft) ** 2
    log_power_spec = np.log10(np.maximum(power, SPEC_EPS)).astype(np.float32)
    log_power_spec, spec_valid_t = pad_or_trim_time(log_power_spec, TARGET_T, SPEC_PAD_VALUE)
    log_power_spec = pad_frequency_for_unet(log_power_spec).astype(np.float32)

    # Exact Whisper input features. These are the tensors that should be fed to
    # WhisperForConditionalGeneration(input_features=...).
    whisper_mel = feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="np",
        padding="max_length",
    ).input_features[0].astype(np.float32)
    mel_valid_t = min(int(np.ceil(len(audio) / WHISPER_HOP_LENGTH)), TARGET_T)

    return log_power_spec, whisper_mel, spec_valid_t, mel_valid_t


def ensure_split_dirs(split_name: str):
    base_paths = {
        "clean_wav": os.path.join(output_clean_wav_dir, split_name),
        "reverb_wav": os.path.join(output_reverb_wav_dir, split_name),
        "clean_spec": os.path.join(output_clean_spec_dir, split_name),
        "reverb_spec": os.path.join(output_reverb_spec_dir, split_name),
        "clean_mel": os.path.join(output_clean_mel_dir, split_name),
        "reverb_mel": os.path.join(output_reverb_mel_dir, split_name),
        "clean_whisper_mel": os.path.join(output_clean_whisper_mel_dir, split_name),
        "reverb_whisper_mel": os.path.join(output_reverb_whisper_mel_dir, split_name),
    }
    for bp in base_paths.values():
        os.makedirs(bp, exist_ok=True)
    return base_paths


def list_audio_files(root):
    files = []
    for dirpath, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith((".wav", ".flac")):
                files.append(os.path.join(dirpath, name))
    return sorted(files)


def speaker_id_from_biurev_path(file_path: str, root_dir: str) -> str:
    rel = os.path.relpath(file_path, root_dir)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) >= 2 else "unknown_speaker"


def load_random_rir(rir_files):
    rir_path = random.choice(rir_files)
    rir, _ = librosa.load(rir_path, sr=sr)
    rir = rir / (np.linalg.norm(rir) + 1e-8)
    return rir


def apply_reverb(clean_audio, rir):
    return fftconvolve(clean_audio, rir, mode="full")[: len(clean_audio)]


def save_feature_set(base_paths, split_name, spk, idx, audio, is_reverb=False, ch=1):
    spec, whisper_mel, spec_len, mel_len = compute_features(audio)
    base_name = f"{split_name}_{idx:05d}"
    if is_reverb:
        base_name += f"_ch{ch}"
    # STFT and Whisper-mel now share the same 10 ms frame grid, so one length is safe.
    base_name += f"_[len={spec_len}]"

    wav_key = "reverb_wav" if is_reverb else "clean_wav"
    spec_key = "reverb_spec" if is_reverb else "clean_spec"
    mel_key = "reverb_mel" if is_reverb else "clean_mel"
    whisper_key = "reverb_whisper_mel" if is_reverb else "clean_whisper_mel"

    wav_dir = os.path.join(base_paths[wav_key], spk)
    spec_dir = os.path.join(base_paths[spec_key], spk)
    mel_dir = os.path.join(base_paths[mel_key], spk)
    whisper_dir = os.path.join(base_paths[whisper_key], spk)
    for path in [wav_dir, spec_dir, mel_dir, whisper_dir]:
        os.makedirs(path, exist_ok=True)

    sf.write(os.path.join(wav_dir, base_name + ".wav"), audio, sr)
    np.save(os.path.join(spec_dir, base_name + ".npy"), spec)
    # Keep clean_melspec/reverb_melspec as aliases for backward compatibility.
    np.save(os.path.join(mel_dir, base_name + ".npy"), whisper_mel)
    np.save(os.path.join(whisper_dir, base_name + ".npy"), whisper_mel)


def process_and_save(clean_files, rir_files, split_name, start_index=0):
    base_paths = ensure_split_dirs(split_name)

    for i, clean_path in enumerate(clean_files, start=start_index):
        if not os.path.isfile(clean_path):
            continue
        try:
            spk = speaker_id_from_biurev_path(
                clean_path,
                dev_clean_dir if split_name == "train" else test_clean_dir,
            )
            clean_audio, _ = librosa.load(clean_path, sr=sr)
            if max_duration_sec is not None:
                clean_audio = clean_audio[: int(sr * max_duration_sec)]

            save_feature_set(base_paths, split_name, spk, i, clean_audio, is_reverb=False)

            for ch in range(1, mics_num + 1):
                rir = load_random_rir(rir_files)
                reverb_audio = apply_reverb(clean_audio, rir)
                save_feature_set(base_paths, split_name, spk, i, reverb_audio, is_reverb=True, ch=ch)

            if (i + 1) % 100 == 0:
                print(f"[{split_name}] processed {i + 1 - start_index}/{len(clean_files)}")
        except Exception as exc:
            print(f"Error processing {clean_path}: {exc}")


def group_by_speaker(files, root_dir):
    spk2files = {}
    for fp in files:
        spk = speaker_id_from_biurev_path(fp, root_dir)
        spk2files.setdefault(spk, []).append(fp)
    for spk in spk2files:
        spk2files[spk] = sorted(spk2files[spk])
    return spk2files


def main():
    dev_files = list_audio_files(dev_clean_dir)
    test_files = list_audio_files(test_clean_dir)
    if not dev_files:
        raise SystemExit(f"No audio found under dev_clean_dir: {dev_clean_dir}")
    if not test_files:
        raise SystemExit(f"No audio found under test_clean_dir: {test_clean_dir}")

    rir_files = glob.glob(os.path.join(rir_dir, "**", "*.wav"), recursive=True)
    if not rir_files:
        raise SystemExit(f"No RIR wav files found under: {rir_dir}")

    dev_spk2files = group_by_speaker(dev_files, dev_clean_dir)
    train_speakers = sorted(dev_spk2files.keys())
    train_files = [fp for spk in train_speakers for fp in dev_spk2files[spk]]

    test_spk2files = group_by_speaker(test_files, test_clean_dir)
    test_speakers = sorted(test_spk2files.keys())
    if len(test_speakers) < 3:
        raise SystemExit("Not enough distinct speakers in test-clean to split into val/test/cal.")

    val_spks, rest_spks = train_test_split(test_speakers, test_size=0.85, random_state=42)
    test_spks, cal_spks = train_test_split(rest_spks, test_size=0.3, random_state=42)

    val_files = [fp for spk in val_spks for fp in test_spk2files[spk]]
    test_files_split = [fp for spk in test_spks for fp in test_spk2files[spk]]
    cal_files = [fp for spk in cal_spks for fp in test_spk2files[spk]]

    print(f"Speakers: train={len(train_speakers)} val={len(val_spks)} test={len(test_spks)} cal={len(cal_spks)}")
    print(f"Files:    train={len(train_files)} val={len(val_files)} test={len(test_files_split)} cal={len(cal_files)}")
    print(f"Found {len(rir_files)} RIR files")
    print(f"Whisper-aligned STFT: n_fft={WHISPER_N_FFT}, hop={WHISPER_HOP_LENGTH}, win={WHISPER_WIN_LENGTH}, window={WHISPER_WINDOW}")

    process_and_save(train_files, rir_files, "train", start_index=0)
    process_and_save(val_files, rir_files, "val", start_index=0)
    process_and_save(test_files_split, rir_files, "test", start_index=0)
    process_and_save(cal_files, rir_files, "cal", start_index=0)
    print("Done!")


if __name__ == "__main__":
    main()
