# Written by Tal Moshai / March 2026
# How to run examples:
# python run_inference_and_save.py --split cal  --out /storage/tal/thesis/infer
# python run_inference_and_save.py --split test --out /storage/tal/thesis/infer

import os
import pickle
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from pathlib import Path
import scipy.io
import argparse

import CopiedFromYam.DATA.utils as utils

from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.set_env import set_device, set_seed
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import load_checkpoint, make_ckpt_path
from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.MODEL_SECTION.criterions_and_optimizers_tree import set_criterion
from CopiedFromYam.BACKBONE.MODEL_SECTION.training_page_same_alpha import evaluate_individual_heads
from CopiedFromYam.CALIBRATION.krcps_yam.calibrate_from_main import call_for_calibration_inf

config = load_config()

K = 512
overlap = 0.75
eps = 2.2204 * np.exp(-16)

MAT = scipy.io.loadmat("synt_win.mat")
synt_win = MAT["synt_win"]

def resolve_best_ckpt(cfg: dict) -> str:
    base = cfg["General"]["weights_path"]
    best_path = make_ckpt_path(base, "best")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(base):
        print(f"WARNING: best checkpoint not found, falling back to base path: {base}")
        return base
    raise FileNotFoundError(f"Could not find checkpoint. Tried: {best_path} and {base}")

def load_global_minmax(config):
    # Use same convention as Yam: saved during spectrogram prep
    # If your path differs, change here.
    mics = config["General"]["mics_num"]
    dataset = config["General"]["dataset"]
    min_max_file = Path(f"./data/spectrograms/{dataset}/mics{mics}/train/global_min_max.p")
    with open(min_max_file, "rb") as f:
        log_max_clean, log_min_clean, log_max_reverb, log_min_reverb = pickle.load(f)
    return log_max_clean, log_min_clean, log_max_reverb, log_min_reverb

def calib_params_path(config):
    weights_base = Path(config["General"]["weights_path"])
    return weights_base.parent / f"calib_params_{config['General']['CALIB_NAME']}.pkl"

def load_calib_params(config):
    if config["General"]["CALIB_NAME"] not in ["rcps", "krcps"]:
        return None, None
    p = calib_params_path(config)
    with open(p, "rb") as f:
        payload = pickle.load(f)
    return payload["lambdas"], payload["gains"]

def get_phase_from_reverb_wav(config, split, speaker, clean_npy_name):
    """
    Reconstruct wav path names that match your dataset generator:
      clean wav:  DataBase/clean_rec/<split>/<speaker>/<base>.wav
      reverb wav: DataBase/reverb_rec/<split>/<speaker>/<base>_ch1_.wav  (with the same base format)
    Your Dataset uses 'ch{i}_' before '[' — so we insert that for reverb.
    """
    base = clean_npy_name.replace(".npy", "")  # e.g. train_00001_[len=123]
    db = Path(config["General"]["data_base_path"])

    # Reverb wav name: insert ch1_ before '['
    if "[" in base:
        reverb_base = base[:base.find("[")] + "ch1_" + base[base.find("["):]
    else:
        reverb_base = base + "_ch1"

    reverb_wav = db / "reverb_rec" / split / speaker / (reverb_base + ".wav")
    if not reverb_wav.exists():
        return None, None

    z, fs = sf.read(str(reverb_wav))
    _, phase = utils.stft(z, K, overlap)
    z = z / 1.1 / np.max(np.abs(z))
    return phase, fs

def save_spec_and_wav(out_dir, stem, spec_mag, phase, fs):
    """
    spec_mag is expected to be linear magnitude with shape (F, T)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}.npy", spec_mag.astype(np.float32))
    if phase is not None:
        audio = utils.istft(spec_mag.T, phase, synt_win)
        audio = audio / 1.1 / np.max(np.abs(audio))
        sf.write(out_dir / f"{stem}.wav", audio, fs)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test", "cal"], default="test")
    ap.add_argument("--out", default="/storage/tal/thesis/inference_outputs")
    args = ap.parse_args()

    config = load_config()
    set_seed()
    device, ngpu, gpu_ids, *_ = set_device()

    model = create_model().to(device)
    if device.type == "cuda" and ngpu > 1:
        model = nn.DataParallel(model, gpu_ids)

    ckpt = resolve_best_ckpt(config)
    model, _, start_epoch, _ = load_checkpoint(model=model, optimizer=None, load_path=ckpt, map_location="cpu")
    print(f"Loaded checkpoint: {ckpt}")

    # Ensure we get reconstructed arrays
    current_config = {"current_step": {"training_phase": ["eval", "whole"], "trained_heads": ["all"]}}
    criterion = set_criterion(current_config, device)

    loader = load_data(args.split)

    # outputs_np: [Nfiles, num_heads, F, T], targets_np: [Nfiles, F, T], inputs_np: [Nfiles, ...]
    _, outputs_np, targets_np, inputs_np = evaluate_individual_heads(model, loader, criterion, device, current_config)

    log_max_clean, log_min_clean, log_max_reverb, log_min_reverb = load_global_minmax(config)

    # Calibration params (optional)
    lambdas, gains = load_calib_params(config)

    out_root = Path(args.out) / config["General"]["save_name"] / args.split
    out_root.mkdir(parents=True, exist_ok=True)

    # The dataset returns paths_list = last 3 parts: [split, speaker, filename.npy]
    # evaluate_individual_heads doesn't return paths_list, so we read them from the dataset directly.
    # This assumes loader.dataset.clean_files is aligned with outputs_np indexing.
    ds = loader.dataset
    clean_paths = ds.clean_files

    for i in range(outputs_np.shape[0]):
        # Identify file naming
        clean_path = Path(clean_paths[i])
        speaker = clean_path.parent.name
        split = clean_path.parent.parent.name  # train/val/test/cal
        clean_name = clean_path.name  # e.g. train_00001_[len=...].npy
        stem = clean_name.replace(".npy", "")

        # Phase from reverb wav (ch1)
        phase, fs = get_phase_from_reverb_wav(config, split, speaker, clean_name)

        # Head 0 prediction (normalized domain) -> denormalize to magnitude
        pred0 = outputs_np[i, 0]  # (F, T) normalized
        pred0_mag = utils.denormalize_log_spec(torch.tensor(pred0), log_max_clean, log_min_clean).numpy()

        save_dir = out_root / speaker
        save_spec_and_wav(save_dir, stem + "_pred", pred0_mag, phase, fs or 16000)

        # Save clean reference spectrogram too (denormalize targets)
        tgt = targets_np[i]
        tgt_mag = utils.denormalize_log_spec(torch.tensor(tgt), log_max_clean, log_min_clean).numpy()
        save_spec_and_wav(save_dir, stem + "_clean", tgt_mag, phase, fs or 16000)

        # For each quantile head: save lower/upper, with optional calibration
        # outputs_np includes all heads in the model's internal order.
        pred_raw = torch.tensor(outputs_np[i], dtype=torch.float32)  # [num_heads, F, T]
        pred_main = pred_raw[0]

        for head, dic_params in config["Model params"]["heads_params"].items():
            if head == 0:
                continue
            alpha = str(dic_params["alpha"])

            lo = pred_raw[head]
            hi = pred_raw[-head]

            # Apply RCPS/KRCPS calibration if enabled and params exist
            if config["General"]["CALIB_NAME"] in ["rcps", "krcps"]:
                if lambdas is None or alpha not in lambdas:
                    raise RuntimeError(f"Missing calib params for alpha={alpha}. Run calibrate_params_from_split.py")
                lo, hi = call_for_calibration_inf(
                    lo, hi, pred_main,
                    config["General"]["CALIB_NAME"],
                    lambdas[alpha],
                    gains[alpha],
                )

            lo_mag = utils.denormalize_log_spec(lo, log_max_clean, log_min_clean).numpy()
            hi_mag = utils.denormalize_log_spec(hi, log_max_clean, log_min_clean).numpy()

            save_spec_and_wav(save_dir, f"{stem}_lower_alpha={alpha}", lo_mag, phase, fs or 16000)
            save_spec_and_wav(save_dir, f"{stem}_upper_alpha={alpha}", hi_mag, phase, fs or 16000)

        if (i + 1) % 50 == 0:
            print(f"[{args.split}] saved {i+1}/{outputs_np.shape[0]}")

    print(f"Done. Saved to: {out_root}")