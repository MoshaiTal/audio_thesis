from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


UTT_ID_RE = re.compile(r"(?:train|val|test|cal)_\d+")
LEN_RE = re.compile(r"\[len=(\d+)\]")
LAYER_RE = re.compile(r"\(layer\s+(\d+)\|(\d+)\)")


def utterance_id(path_or_name: str) -> str:
    match = UTT_ID_RE.search(Path(path_or_name).name)
    if match:
        return match.group(0)
    return normalize_stem(path_or_name)


def normalize_stem(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    stem = re.sub(r"\[len=\d+\]", "", stem)
    stem = re.sub(r"\(layer\s+\d+\|\d+\)", "", stem)
    stem = re.sub(r"_ch\d+", "", stem)
    stem = re.sub(r"_(pred|lower_alpha=.*|upper_alpha=.*)$", "", stem)
    return stem.strip("_-. ")


def is_pred_layer(path: Path, layer: int) -> bool:
    match = LAYER_RE.search(path.name)
    return bool(match and int(match.group(1)) == layer)


def index_mels(root: Path, channel: Optional[str] = None, pred_layer: Optional[int] = None) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.npy")):
        if channel and channel not in path.name and "_ch" in path.name:
            continue
        if pred_layer is not None and not is_pred_layer(path, pred_layer):
            continue
        index.setdefault(utterance_id(path.name), path)
        index.setdefault(normalize_stem(path.name), path)
    return index


def load_mel(path: Path) -> np.ndarray:
    mel = np.load(path).astype(np.float32)
    mel = np.squeeze(mel)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel, got {mel.shape}: {path}")
    if mel.shape[0] != 80 and mel.shape[1] == 80:
        mel = mel.T
    return mel


def true_len_from_paths(paths: List[Path], fallback: int) -> int:
    for path in paths:
        match = LEN_RE.search(path.name)
        if match:
            return int(match.group(1))
    return fallback


def trim_same_time(clean: np.ndarray, reverb: np.ndarray, pred: np.ndarray, true_t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = min(true_t, clean.shape[1], reverb.shape[1], pred.shape[1])
    return clean[:, :t], reverb[:, :t], pred[:, :t]


def norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def plot_one(uid: str, clean: np.ndarray, reverb: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    stacked = np.concatenate([clean.reshape(-1), reverb.reshape(-1), pred.reshape(-1)])
    vmin, vmax = np.percentile(stacked, [1, 99])
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    rows = [
        ("clean", clean),
        ("reverb", reverb),
        ("pred layer 0", pred),
    ]
    for ax, (title, mel) in zip(axes, rows):
        im = ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("mel bin")
    axes[-1].set_xlabel("time frame")
    fig.suptitle(uid)
    fig.colorbar(im, ax=axes, shrink=0.8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot clean/reverb/pred mel triplets from saved mel-U-Net outputs.")
    parser.add_argument("--clean-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/clean_melspec/val"))
    parser.add_argument("--reverb-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/reverb_melspec/val"))
    parser.add_argument("--pred-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/dereverb_mel/mel_unet_v1/val"))
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--pred-layer", type=int, default=0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("/storage/tal/thesis/script_res/mel_unet_triplet_plots"))
    args = parser.parse_args()

    clean_idx = index_mels(args.clean_root)
    reverb_idx = index_mels(args.reverb_root, channel=args.channel)
    pred_idx = index_mels(args.pred_root, pred_layer=args.pred_layer)

    rows = []
    common = sorted(set(clean_idx) & set(reverb_idx) & set(pred_idx))
    if not common:
        raise RuntimeError(
            "No clean/reverb/pred mel triplets matched. Check roots and pred-layer. "
            f"clean={len(clean_idx)} reverb={len(reverb_idx)} pred={len(pred_idx)}"
        )

    for uid in common[: args.limit]:
        clean_path, reverb_path, pred_path = clean_idx[uid], reverb_idx[uid], pred_idx[uid]
        clean = load_mel(clean_path)
        reverb = load_mel(reverb_path)
        pred = load_mel(pred_path)
        true_t = true_len_from_paths([clean_path, reverb_path, pred_path], min(clean.shape[1], reverb.shape[1], pred.shape[1]))
        clean, reverb, pred = trim_same_time(clean, reverb, pred, true_t)
        out_png = args.out_dir / f"{uid}_mel_triplet.png"
        plot_one(uid, clean, reverb, pred, out_png)
        rows.append(
            {
                "uid": uid,
                "clean_path": str(clean_path),
                "reverb_path": str(reverb_path),
                "pred_path": str(pred_path),
                "clean_reverb_corr": norm_corr(clean, reverb),
                "clean_pred_corr": norm_corr(clean, pred),
                "clean_reverb_mae": mae(clean, reverb),
                "clean_pred_mae": mae(clean, pred),
                "plot": str(out_png),
            }
        )

    csv_path = args.out_dir / "mel_triplet_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] matched triplets: {len(common)}")
    print(f"[SAVE] plots: {args.out_dir}")
    print(f"[SAVE] metrics: {csv_path}")


if __name__ == "__main__":
    main()
