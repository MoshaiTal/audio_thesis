#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import pathlib
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

from transformers import WhisperFeatureExtractor

from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.set_env import set_device, set_seed
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.criterions_and_optimizers_tree import set_criterion
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import load_checkpoint, make_ckpt_path
from CopiedFromYam.BACKBONE.MODEL_SECTION.training_page_same_alpha import evaluate_individual_heads
from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.calibrate_from_main import (
    call_for_calibration_train,
    call_for_calibration_inf,
)
from CopiedFromYam.DATA.normalization_utils import (
    denormalize_from_minus1_1,
    load_or_create_train_minmax,
)

SR = 16000
N_FFT = 512
HOP_LENGTH = 128
WIN_LENGTH = 512
WINDOW = "hamming"
CENTER = True
EPS = 1e-8

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
_lambda = {}
gain = {}


def resolve_ckpt(cfg: dict) -> str:
    base = cfg["General"]["weights_path"]
    last_path = make_ckpt_path(base, "last")
    best_path = make_ckpt_path(base, "best")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path
    if os.path.exists(base):
        return base
    raise FileNotFoundError(f"Could not find checkpoint. Tried: {best_path}, {last_path}, {base}")


def calib_params_path(cfg: dict) -> pathlib.Path:
    w = pathlib.Path(cfg["General"]["weights_path"])
    name = cfg["General"].get("CALIB_NAME", "")
    tag = name if name else "none"
    return w.parent / f"calib_params_{tag}.pkl"


def save_calib_params(cfg: dict, lambdas: dict, gains: dict, meta=None) -> None:
    p = calib_params_path(cfg)
    payload = {"lambdas": lambdas, "gains": gains, "meta": meta or {}}
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    print(f"[CALIB] Saved params to: {p}")


def load_calib_params(cfg: dict):
    p = calib_params_path(cfg)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        payload = pickle.load(f)
    print(f"[CALIB] Loaded params from: {p}")
    return payload


def insert_ch_before_bracket(stem: str, ch: int) -> str:
    if "[" not in stem:
        return f"{stem}_ch{ch}"
    return stem[: stem.find("[")] + f"ch{ch}_" + stem[stem.find("["):]


def make_out_roots(cfg: dict):
    db = pathlib.Path(cfg["General"]["data_base_path"])
    calib_name = cfg["General"].get("CALIB_NAME", "")
    tag = calib_name if calib_name else "none"
    sub = f"calibrated_{tag}"
    return (
        db / "dereverb_rec" / sub,
        db / "dereverb_spec" / sub,
        db / "dereverb_mel" / sub,
    )


def audio_to_whisper_mel(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    feats = feature_extractor(audio.astype(np.float32), sampling_rate=sr, return_tensors='np').input_features[0]
    return feats.astype(np.float32)


def ensure_mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x[:, 0]


def chunks_to_full_padded(chunks: np.ndarray) -> np.ndarray:
    """
    chunks: [S, 256, 256]
    returns full padded [S*256, 256], assuming chunk layout [chunk, freq, time].
    """
    if chunks.ndim != 3:
        raise ValueError(f"Expected [S,256,256] chunks, got shape {chunks.shape}")
    # IMPORTANT:
    # training data chunks are [F, T_chunk] per chunk.
    # To reconstruct the full utterance, concatenate along the LAST axis (time), not flatten rows.
    full_FxT = chunks.transpose(1, 0, 2).reshape(chunks.shape[1], -1)  # [F, S*256]
    return full_FxT.astype(np.float32)


def get_reverb_wav_path(cfg: dict, split: str, speaker: str, clean_stem: str, ch: int) -> pathlib.Path:
    db = pathlib.Path(cfg["General"]["data_base_path"])
    return db / "reverb_rec" / split / speaker / (insert_ch_before_bracket(clean_stem, ch) + ".wav")


def get_clean_spec_path(cfg: dict, split: str, speaker: str, clean_stem: str) -> pathlib.Path:
    db = pathlib.Path(cfg["General"]["data_base_path"])
    return db / "clean_spec" / split / speaker / f"{clean_stem}.npy"


def librosa_logmag_and_phase(audio: np.ndarray):
    D = librosa.stft(
        audio,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=CENTER,
    )
    mag = np.abs(D)
    phase = np.angle(D)
    # Return [F,T] log-mag and [T,F] phase (phase shape kept convenient for ISTFT helper below)
    return np.log(mag + EPS).astype(np.float32), phase.T.astype(np.float32)


def spec256_FxT_to_wav_librosa(spec_FxT_raw: np.ndarray, last_bin_1xT_raw: np.ndarray, phase_T257: np.ndarray, out_len: int):
    """
    spec_FxT_raw: [256, T]
    last_bin_1xT_raw: [1, T]
    phase_T257: [T, 257]
    """
    T_use = min(spec_FxT_raw.shape[1], last_bin_1xT_raw.shape[1], phase_T257.shape[0])
    spec_257xT = np.concatenate([spec_FxT_raw[:, :T_use], last_bin_1xT_raw[:, :T_use]], axis=0)  # [257,T]
    mag = np.maximum(np.exp(spec_257xT) - EPS, 0.0).astype(np.float32)
    complex_spec = mag * np.exp(1j * phase_T257[:T_use, :].T)  # [257,T]
    wav = librosa.istft(
        complex_spec,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=WINDOW,
        center=CENTER,
        length=out_len,
    ).astype(np.float32)

    peak = np.max(np.abs(wav)) + 1e-12
    if peak > 0.99:
        wav = 0.99 * wav / peak
    return wav


def save_spec_png(path: pathlib.Path, spec_FxT: np.ndarray, title: str):
    plt.figure(figsize=(12, 4))
    plt.imshow(spec_FxT, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency bin")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def maybe_load_or_fit_calibration(model, device, config):
    if config["General"].get("CALIB_NAME", "") not in ["rcps", "krcps"]:
        print("[CALIB] CALIB_NAME not rcps/krcps -> calibration disabled.")
        return
    payload = load_calib_params(config)
    if payload is not None:
        _lambda.update(payload["lambdas"])
        gain.update(payload["gains"])
        return

    current_config = {"current_step": {"training_phase": ["eval", "whole"], "trained_heads": ["all"]}}
    criterion = set_criterion(current_config, device)
    cal_loader = load_data("cal")
    _, cal_outputs_np, cal_targets_np, _ = evaluate_individual_heads(model, cal_loader, criterion, device, current_config)
    cal_outputs = torch.tensor(cal_outputs_np, dtype=torch.float32).permute(1, 0, 2, 3)
    cal_targets = torch.tensor(cal_targets_np, dtype=torch.float32)
    calib_tuple = (cal_outputs, cal_targets, None)

    for head, dic_params in config["Model params"]["heads_params"].items():
        if head == 0:
            continue
        alpha = str(dic_params["alpha"])
        lam, g = call_for_calibration_train(calib_tuple, head, alpha, config["General"]["CALIB_NAME"])
        _lambda[alpha] = lam
        gain[alpha] = g
        print(f"[CALIB] fitted {config['General']['CALIB_NAME']} alpha={alpha} (head={head})")
    save_calib_params(config, _lambda, gain, meta={"source": "DataLoader cal split"})


def infer_and_save_split(split: str, model, device, config):
    loader = load_data(split)
    ds = loader.dataset
    mics_num = int(config["General"]["mics_num"])
    rec_root, spec_root, mel_root = make_out_roots(config)
    norm = load_or_create_train_minmax(config["General"]["data_base_path"], config["General"]["DATA_TYPE"])
    norm_min, norm_max = norm["min"], norm["max"]
    print(f"[NORM] Using [{norm_min:.5f}, {norm_max:.5f}] from {norm['path']}")

    model.eval()
    for idx in range(len(ds)):
        reverb, clean, last_layer, last_time, mask, paths_list = ds[idx]
        speaker = paths_list[1]
        fname = paths_list[2]
        clean_stem = os.path.splitext(fname)[0]

        reverb_t = torch.from_numpy(reverb).float() if isinstance(reverb, np.ndarray) else reverb.float()
        mask_t = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
        reverb_t = reverb_t.to(device)
        mask_t = mask_t.to(device)

        out_rec = rec_root / split / speaker
        out_spec = spec_root / split / speaker
        out_mel = mel_root / split / speaker
        out_rec.mkdir(parents=True, exist_ok=True)
        out_spec.mkdir(parents=True, exist_ok=True)
        out_mel.mkdir(parents=True, exist_ok=True)

        for ch in range(1, mics_num + 1):
            x = reverb_t if reverb_t.ndim == 4 else reverb_t[:, ch - 1: ch, :, :]
            with torch.no_grad():
                outs = model(x, mask_t)

            pred_chunks_norm = outs[0].detach().cpu().numpy().astype(np.float32)  # [S,F,Tchunk]
            lo_chunks_norm = {}
            hi_chunks_norm = {}
            pred_t_for_cal = outs[0].detach().cpu()
            for head, dic_params in config["Model params"]["heads_params"].items():
                if head == 0:
                    continue
                alpha = str(dic_params["alpha"])
                lo = outs[head].detach().cpu()
                hi = outs[-head].detach().cpu()
                if config["General"].get("CALIB_NAME", "") in ["rcps", "krcps"]:
                    lo, hi = call_for_calibration_inf(
                        lo,
                        hi,
                        pred_t_for_cal,
                        config["General"]["CALIB_NAME"],
                        _lambda[alpha],
                        gain[alpha],
                    )
                lo_chunks_norm[alpha] = lo.numpy().astype(np.float32)
                hi_chunks_norm[alpha] = hi.numpy().astype(np.float32)

            # Rebuild [F,T] padded spectrograms from chunks
            pred_spec_norm_full_FxT = chunks_to_full_padded(pred_chunks_norm)
            lo_spec_norm_full_FxT = {a: chunks_to_full_padded(lo_chunks_norm[a]) for a in lo_chunks_norm}
            hi_spec_norm_full_FxT = {a: chunks_to_full_padded(hi_chunks_norm[a]) for a in hi_chunks_norm}

            reverb_wav = get_reverb_wav_path(config, split, speaker, clean_stem, ch)
            if not reverb_wav.exists():
                continue

            zr, fs = sf.read(str(reverb_wav))
            zr = ensure_mono(zr).astype(np.float32)
            if fs != SR:
                zr = librosa.resample(zr, orig_sr=fs, target_sr=SR).astype(np.float32)

            # Use actual STFT shape from wav as source of truth
            log_mag_257xT, phase_T257 = librosa_logmag_and_phase(zr)
            T_phase = log_mag_257xT.shape[1]
            last_bin_1xT_raw = log_mag_257xT[-1:, :T_phase]

            pred_spec_norm_FxT = pred_spec_norm_full_FxT[:, :T_phase]
            lo_spec_norm_FxT = {a: lo_spec_norm_full_FxT[a][:, :T_phase] for a in lo_spec_norm_full_FxT}
            hi_spec_norm_FxT = {a: hi_spec_norm_full_FxT[a][:, :T_phase] for a in hi_spec_norm_full_FxT}

            pred_spec_raw_FxT = denormalize_from_minus1_1(pred_spec_norm_FxT, norm_min, norm_max)
            lo_spec_raw_FxT = {a: denormalize_from_minus1_1(lo_spec_norm_FxT[a], norm_min, norm_max) for a in lo_spec_norm_FxT}
            hi_spec_raw_FxT = {a: denormalize_from_minus1_1(hi_spec_norm_FxT[a], norm_min, norm_max) for a in hi_spec_norm_FxT}

            # Save in training-consistent [F,T] format
            np.save(out_spec / f"{clean_stem}_ch{ch}_pred.npy", pred_spec_raw_FxT.astype(np.float32))
            for a in lo_spec_raw_FxT:
                np.save(out_spec / f"{clean_stem}_ch{ch}_lower_alpha={a}.npy", lo_spec_raw_FxT[a].astype(np.float32))
                np.save(out_spec / f"{clean_stem}_ch{ch}_upper_alpha={a}.npy", hi_spec_raw_FxT[a].astype(np.float32))

            def save_rec_and_mel_from_FxT(tag: str, spec_FxT_raw: np.ndarray):
                wav = spec256_FxT_to_wav_librosa(spec_FxT_raw, last_bin_1xT_raw, phase_T257, len(zr))
                sf.write(out_rec / f"{clean_stem}_ch{ch}_{tag}.wav", wav, SR)
                mel = audio_to_whisper_mel(wav, SR)
                np.save(out_mel / f"{clean_stem}_ch{ch}_{tag}.npy", mel)

            save_rec_and_mel_from_FxT("pred", pred_spec_raw_FxT)
            for a in lo_spec_raw_FxT:
                save_rec_and_mel_from_FxT(f"lower_alpha={a}", lo_spec_raw_FxT[a])
                save_rec_and_mel_from_FxT(f"upper_alpha={a}", hi_spec_raw_FxT[a])


        if (idx + 1) % 50 == 0:
            print(f"[{split}] processed {idx + 1}/{len(ds)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_splits", default="cal,test", help="Comma-separated: train,val,test,cal")
    args = parser.parse_args()

    config = load_config()
    set_seed()

    device, ngpu, gpu_ids, gpu_names, multi_gpu = set_device()
    print(f"Using device: {device}, ngpu={ngpu}")

    model = create_model().to(device)
    if (device.type == "cuda") and (ngpu > 1):
        model = nn.DataParallel(model, gpu_ids)

    ckpt = resolve_ckpt(config)
    model, _, start_epoch, _ = load_checkpoint(model=model, optimizer=None, load_path=ckpt, map_location="cpu")
    print(f"Loaded checkpoint: {ckpt} (epoch={start_epoch - 1})")

    maybe_load_or_fit_calibration(model, device, config)

    splits = [s.strip() for s in args.run_splits.split(",") if s.strip()]
    for sp in splits:
        if sp not in ["train", "val", "test", "cal"]:
            raise ValueError(f"Unknown split: {sp}")
        print(f"\n========== Running inference on split: {sp} ==========")
        infer_and_save_split(sp, model, device, config)

    rec_root, spec_root, mel_root = make_out_roots(config)
    print("\nDone.")
    print(f"Saved WAVs to:  {rec_root}")
    print(f"Saved specs to: {spec_root}")
    print(f"Saved mels to:  {mel_root}")
