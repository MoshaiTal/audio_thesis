#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn

from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import load_checkpoint, make_ckpt_path
from CopiedFromYam.set_env import set_device, set_seed


def resolve_ckpt(cfg: dict) -> str:
    base = cfg["General"]["weights_path"]
    best_path = make_ckpt_path(base, "best")
    last_path = make_ckpt_path(base, "last")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(last_path):
        return last_path
    if os.path.exists(base):
        return base
    raise FileNotFoundError(
        f"Could not find checkpoint. Tried: {best_path}, {last_path}, {base}"
    )


def denormalize_minus1_1(x: np.ndarray, norm_min: float, norm_max: float) -> np.ndarray:
    return (x + 1.0) * 0.5 * (norm_max - norm_min) + norm_min


def chunks_to_full_mel(chunks: np.ndarray, mel_bins: int = 80) -> np.ndarray:
    if chunks.ndim == 4 and chunks.shape[1] == 1:
        chunks = chunks[:, 0]
    if chunks.ndim != 3:
        raise ValueError(f"Expected chunks [S,F,T], got {chunks.shape}")
    full = chunks.transpose(1, 0, 2).reshape(chunks.shape[1], -1)
    return full[:mel_bins].astype(np.float32)


def make_mel_root(cfg: dict, output_tag: str) -> pathlib.Path:
    db = pathlib.Path(cfg["General"]["data_base_path"])
    return db / "dereverb_mel" / f"calibrated_{output_tag}"


def infer_and_save_split(split: str, model, device, cfg: dict, output_tag: str) -> None:
    loader = load_data(split)
    dataset = loader.dataset
    norm_min = float(dataset.norm_min)
    norm_max = float(dataset.norm_max)
    mics_num = int(cfg["General"]["mics_num"])
    out_root = make_mel_root(cfg, output_tag)

    model.eval()
    print(
        f"[INFO] split={split}, samples={len(dataset)}, "
        f"norm=({norm_min:.6f}, {norm_max:.6f})"
    )
    print(f"[INFO] saving mels under: {out_root / split}")

    for idx in range(len(dataset)):
        reverb, _, _, _, mask, paths_list = dataset[idx]
        speaker = paths_list[1]
        fname = paths_list[2]
        clean_stem = os.path.splitext(fname)[0]
        out_dir = out_root / split / speaker
        out_dir.mkdir(parents=True, exist_ok=True)

        reverb_t = torch.from_numpy(reverb).float().to(device)
        mask_t = torch.from_numpy(mask).float().to(device)

        for ch in range(1, mics_num + 1):
            x = reverb_t if reverb_t.ndim == 4 else reverb_t[:, ch - 1 : ch, :, :]
            with torch.no_grad():
                outs = model(x, mask_t)

            pred_norm = chunks_to_full_mel(
                outs[0].detach().cpu().numpy().astype(np.float32)
            )
            pred = denormalize_minus1_1(pred_norm, norm_min, norm_max)
            np.save(out_dir / f"{clean_stem}_ch{ch}_pred.npy", pred.astype(np.float32))

            for head, dic_params in cfg["Model params"]["heads_params"].items():
                head = int(head)
                if head == 0:
                    continue
                alpha = str(dic_params["alpha"])
                lo_norm = chunks_to_full_mel(
                    outs[head].detach().cpu().numpy().astype(np.float32)
                )
                hi_norm = chunks_to_full_mel(
                    outs[-head].detach().cpu().numpy().astype(np.float32)
                )
                np.save(
                    out_dir / f"{clean_stem}_ch{ch}_lower_alpha={alpha}.npy",
                    denormalize_minus1_1(lo_norm, norm_min, norm_max).astype(np.float32),
                )
                np.save(
                    out_dir / f"{clean_stem}_ch{ch}_upper_alpha={alpha}.npy",
                    denormalize_minus1_1(hi_norm, norm_min, norm_max).astype(np.float32),
                )

        if (idx + 1) % 50 == 0:
            print(f"[{split}] processed {idx + 1}/{len(dataset)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate direct dereverb_mel outputs from the original mel model."
    )
    parser.add_argument("--run_splits", default="val", help="Comma-separated: train,val,test,cal")
    parser.add_argument(
        "--output_tag",
        required=True,
        help="Outputs go to dereverb_mel/calibrated_<output_tag>",
    )
    args = parser.parse_args()

    cfg = load_config()
    if cfg["General"].get("DATA_TYPE") != "melspec":
        raise RuntimeError("This generator expects General.DATA_TYPE == melspec.")

    set_seed()
    device, ngpu, gpu_ids, _, _ = set_device()
    model = create_model().to(device)
    if device.type == "cuda" and ngpu > 1:
        model = nn.DataParallel(model, gpu_ids)

    ckpt = resolve_ckpt(cfg)
    model, _, start_epoch, _ = load_checkpoint(
        model=model,
        optimizer=None,
        load_path=ckpt,
        map_location="cpu",
    )
    print(f"[INFO] Loaded checkpoint: {ckpt} epoch={start_epoch - 1}")

    for split in [s.strip() for s in args.run_splits.split(",") if s.strip()]:
        if split not in {"train", "val", "test", "cal"}:
            raise ValueError(f"Unknown split: {split}")
        infer_and_save_split(split, model, device, cfg, args.output_tag)

    print("[DONE]")
    print(f"Saved mels to: {make_mel_root(cfg, args.output_tag)}")


if __name__ == "__main__":
    main()

