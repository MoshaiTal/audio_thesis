#!/usr/bin/env python3
import os
import glob
import random
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve
from sklearn.model_selection import train_test_split
from transformers import WhisperFeatureExtractor

# ----------------- Parameters -----------------

dev_clean_dir  = "/storage/tal/thesis/BIUREV/dev-clean"
test_clean_dir = "/storage/tal/thesis/BIUREV/test-clean"
rir_dir = "/storage/tal/thesis/raw_rirs"

output_clean_wav_dir   = "/storage/tal/thesis/DataBase_BIUREV/clean_rec"
output_reverb_wav_dir  = "/storage/tal/thesis/DataBase_BIUREV/reverb_rec"
output_clean_spec_dir  = "/storage/tal/thesis/DataBase_BIUREV/clean_spec"
output_reverb_spec_dir = "/storage/tal/thesis/DataBase_BIUREV/reverb_spec"
output_clean_mel_dir   = "/storage/tal/thesis/DataBase_BIUREV/clean_melspec"
output_reverb_mel_dir  = "/storage/tal/thesis/DataBase_BIUREV/reverb_melspec"

sr         = 16000
n_fft      = 512
hop_length = 128
mics_num   = 1
max_duration_sec = 30
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")

TARGET_T = 3000
SPEC_EPS = 1e-8
SPEC_PAD_VALUE = float(np.log(SPEC_EPS))

# ----------------------------------------------

def pad_or_trim_time(x: np.ndarray, target_T: int = TARGET_T, pad_value: float = 0.0):
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {x.shape}")
    F, T = x.shape
    valid_T = min(T, target_T)
    if T > target_T:
        return x[:, :target_T], valid_T
    if T == target_T:
        return x, valid_T
    pad_width = ((0, 0), (0, target_T - T))
    return np.pad(x, pad_width, mode="constant", constant_values=pad_value), valid_T


def compute_features(audio: np.ndarray):
    # STFT/log-spec path
    D = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hamming",
        center=True,
    )  # (257, T)

    mag = np.abs(D)
    log_spectrogram = np.log(mag + SPEC_EPS).astype(np.float32)
    log_spectrogram, spec_valid_T = pad_or_trim_time(
        log_spectrogram,
        target_T=TARGET_T,
        pad_value=SPEC_PAD_VALUE,
    )
    log_spectrogram = log_spectrogram[:-1, :]  # (256, 3000)

    # Whisper log-mel path (already fixed-length features)
    mel = feature_extractor(audio, sampling_rate=sr, return_tensors="np").input_features[0]
    log_mel = np.squeeze(mel).astype(np.float32)  # (80, 3000)
    mel_valid_T = min(int(np.ceil(len(audio) / sr * 100)), TARGET_T)

    return log_spectrogram, log_mel, spec_valid_T, mel_valid_T


def ensure_split_dirs(split_name: str):
    base_paths = {
        "clean_wav": os.path.join(output_clean_wav_dir, split_name),
        "reverb_wav": os.path.join(output_reverb_wav_dir, split_name),
        "clean_spec": os.path.join(output_clean_spec_dir, split_name),
        "reverb_spec": os.path.join(output_reverb_spec_dir, split_name),
        "clean_mel": os.path.join(output_clean_mel_dir, split_name),
        "reverb_mel": os.path.join(output_reverb_mel_dir, split_name),
    }
    for bp in base_paths.values():
        os.makedirs(os.path.join(bp, "clean"), exist_ok=True)
        os.makedirs(os.path.join(bp, "reverb"), exist_ok=True)
    return base_paths


def list_audio_files(root):
    exts = (".wav", ".flac")
    files = []
    for r, _, fnames in os.walk(root):
        for f in fnames:
            if f.lower().endswith(exts):
                files.append(os.path.join(r, f))
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
    return fftconvolve(clean_audio, rir, mode="full")[:len(clean_audio)]


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

            wav_clean_dir   = os.path.join(base_paths["clean_wav"], spk)
            wav_reverb_dir  = os.path.join(base_paths["reverb_wav"], spk)
            spec_clean_dir  = os.path.join(base_paths["clean_spec"], spk)
            spec_reverb_dir = os.path.join(base_paths["reverb_spec"], spk)
            mel_clean_dir   = os.path.join(base_paths["clean_mel"], spk)
            mel_reverb_dir  = os.path.join(base_paths["reverb_mel"], spk)

            for d in [wav_clean_dir, wav_reverb_dir, spec_clean_dir, spec_reverb_dir, mel_clean_dir, mel_reverb_dir]:
                os.makedirs(d, exist_ok=True)

            clean_audio, _ = librosa.load(clean_path, sr=sr)
            if max_duration_sec is not None:
                max_samples = int(sr * max_duration_sec)
                clean_audio = clean_audio[:max_samples]

            clean_spec, clean_mel, clean_spec_len, clean_mel_len = compute_features(clean_audio)

            # Use STFT valid length in filenames so the spec path stays consistent.
            # Mels are still saved under the same basename for convenience, but the loader
            # should infer valid time from the array itself for each data type.
            clean_base_name = f"{split_name}_{i:05d}_[len={clean_spec_len}]"
            clean_wav_name = clean_base_name + ".wav"
            clean_npy_name = clean_base_name + ".npy"

            sf.write(os.path.join(wav_clean_dir, clean_wav_name), clean_audio, sr)
            np.save(os.path.join(spec_clean_dir, clean_npy_name), clean_spec)
            np.save(os.path.join(mel_clean_dir,  clean_npy_name), clean_mel)

            for ch in range(1, mics_num + 1):
                rir = load_random_rir(rir_files)
                reverb_audio = apply_reverb(clean_audio, rir)
                reverb_spec, reverb_mel, reverb_spec_len, reverb_mel_len = compute_features(reverb_audio)

                reverb_base_name = f"{split_name}_{i:05d}_ch{ch}_[len={reverb_spec_len}]"
                reverb_wav_name = reverb_base_name + ".wav"
                reverb_npy_name = reverb_base_name + ".npy"

                sf.write(os.path.join(wav_reverb_dir, reverb_wav_name), reverb_audio, sr)
                np.save(os.path.join(spec_reverb_dir, reverb_npy_name), reverb_spec)
                np.save(os.path.join(mel_reverb_dir,  reverb_npy_name), reverb_mel)

            if (i + 1) % 100 == 0:
                print(f"[{split_name}] processed {i+1-start_index}/{len(clean_files)}")

        except Exception as e:
            print(f"Error processing {clean_path}: {e}")


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
        raise SystemExit("Not enough distinct speakers in test-recs to split into val/test/cal.")

    val_spks, rest_spks = train_test_split(test_speakers, test_size=0.85, random_state=42)
    test_spks, cal_spks = train_test_split(rest_spks, test_size=0.3, random_state=42)

    val_files = [fp for spk in val_spks for fp in test_spk2files[spk]]
    test_files_split = [fp for spk in test_spks for fp in test_spk2files[spk]]
    cal_files = [fp for spk in cal_spks for fp in test_spk2files[spk]]

    print(f"Speakers: train={len(train_speakers)} val={len(val_spks)} test={len(test_spks)} cal={len(cal_spks)}")
    print(f"Files:    train={len(train_files)} val={len(val_files)} test={len(test_files_split)} cal={len(cal_files)}")
    print(f"Found {len(rir_files)} RIR files")

    process_and_save(train_files, rir_files, "train", start_index=0)
    process_and_save(val_files, rir_files, "val", start_index=0)
    process_and_save(test_files_split, rir_files, "test", start_index=0)
    process_and_save(cal_files, rir_files, "cal", start_index=0)

    print("Done!")


if __name__ == "__main__":
    main()
