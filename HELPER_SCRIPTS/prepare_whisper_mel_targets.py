from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy import signal
from tqdm import tqdm
from transformers import WhisperFeatureExtractor


LEN_RE = re.compile(r"\[len=(\d+)\]")


def read_audio(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        audio = signal.resample_poly(audio, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio, sr


def strip_reverb_channel(stem: str) -> str:
    return re.sub(r"_ch\d+", "", stem)


def out_name_for(path: Path) -> str:
    return strip_reverb_channel(path.stem) + ".npy"


def iter_wavs(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute exact Whisper input_features from clean/reverb wav files. "
            "These are the backend-matched targets/baselines for Whisper."
        )
    )
    parser.add_argument("--db-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV"))
    parser.add_argument("--splits", type=str, default="train,val,cal,test")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--make-reverb", action="store_true", help="Also precompute reverb_whisper_melspec.")
    args = parser.parse_args()

    extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)

    jobs = [("clean_rec", "clean_whisper_melspec")]
    if args.make_reverb:
        jobs.append(("reverb_rec", "reverb_whisper_melspec"))

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        for src_name, dst_name in jobs:
            src_root = args.db_root / src_name / split
            dst_root = args.db_root / dst_name / split
            paths = list(iter_wavs(src_root))
            if args.limit is not None:
                paths = paths[: args.limit]
            if not paths:
                print(f"[WARN] no wavs found under {src_root}")
                continue

            print(f"[INFO] {src_name}/{split}: {len(paths)} files -> {dst_root}")
            for wav_path in tqdm(paths, desc=f"{dst_name}/{split}"):
                rel_parent = wav_path.parent.relative_to(src_root)
                out_dir = dst_root / rel_parent
                out_dir.mkdir(parents=True, exist_ok=True)

                audio, sr = read_audio(wav_path, args.sample_rate)
                features = extractor(
                    audio,
                    sampling_rate=sr,
                    return_tensors="np",
                    padding="max_length",
                ).input_features[0].astype(np.float32)
                np.save(out_dir / out_name_for(wav_path), features)

    print("[DONE]")


if __name__ == "__main__":
    main()
