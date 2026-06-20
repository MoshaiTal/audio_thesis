from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm


LAYER_RE = re.compile(r"\(layer\s+(\d+)\|(\d+)\)")


def is_pred_layer(path: Path, layer: int) -> bool:
    match = LAYER_RE.search(path.name)
    return bool(match and int(match.group(1)) == layer)


def load_mel(path: Path) -> np.ndarray:
    mel = np.load(path).astype(np.float32)
    mel = np.squeeze(mel)
    if mel.ndim != 2:
        raise ValueError(f"Expected 2D mel, got {mel.shape}: {path}")
    if mel.shape[0] != 80 and mel.shape[1] == 80:
        mel = mel.T
    return mel


def mel_to_power(mel: np.ndarray, scale: str) -> np.ndarray:
    if scale == "whisper":
        # Whisper features are approximately: normalized = (log10(power) + 4) / 4.
        log10_power = mel * 4.0 - 4.0
        return np.maximum(10.0 ** log10_power, 1e-10)
    if scale == "log10":
        return np.maximum(10.0 ** mel, 1e-10)
    if scale == "ln":
        return np.maximum(np.exp(mel), 1e-10)
    if scale == "power":
        return np.maximum(mel, 1e-10)
    raise ValueError(f"Unknown mel scale: {scale}")


def safe_out_name(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"\(layer\s+(\d+)\|(\d+)\)", r"_layer\1of\2", stem)
    return stem + ".wav"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate mel-to-audio inversion for listening/debugging. "
            "Do not use these wav files for the direct Whisper mel evaluation."
        )
    )
    parser.add_argument("--mel-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/dereverb_mel/mel_unet_v1/val"))
    parser.add_argument("--out-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/dereverb_rec/mel_unet_v1/val"))
    parser.add_argument("--pred-layer", type=int, default=0)
    parser.add_argument("--scale", choices=["whisper", "log10", "ln", "power"], default="whisper")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=400)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--n-iter", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("This script needs librosa installed in the remote environment.") from exc

    paths = [p for p in sorted(args.mel_root.rglob("*.npy")) if is_pred_layer(p, args.pred_layer)]
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise RuntimeError(f"No layer {args.pred_layer} mel files found under {args.mel_root}")

    for path in tqdm(paths, desc="Inverting mels"):
        rel_parent = path.parent.relative_to(args.mel_root)
        out_dir = args.out_root / rel_parent
        out_dir.mkdir(parents=True, exist_ok=True)

        mel = load_mel(path)
        mel_power = mel_to_power(mel, args.scale)
        audio = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=args.sr,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            win_length=args.win_length,
            n_iter=args.n_iter,
            power=1.0,
            fmin=0.0,
            fmax=8000.0,
        )
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = 0.98 * audio / peak
        sf.write(out_dir / safe_out_name(path), audio.astype(np.float32), args.sr)

    print(f"[INFO] inverted files: {len(paths)}")
    print(f"[SAVE] {args.out_root}")


if __name__ == "__main__":
    main()
