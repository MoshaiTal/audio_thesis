from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


UTT_ID_RE = re.compile(r"(?:train|val|test|cal)_\d+")
LEN_RE = re.compile(r"\[len=(\d+)\]")
LAYER_RE = re.compile(r"\(layer\s+(\d+)\|(\d+)\)")
ALPHA_RE = re.compile(r"alpha=([^_\)\]]+)")


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def edit_distance(ref_words: Sequence[str], hyp_words: Sequence[str]) -> int:
    prev = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        cur = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            cost = 0 if ref_word == hyp_word else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> Tuple[float, int, int]:
    ref_words = clean_text(reference).split()
    hyp_words = clean_text(hypothesis).split()
    errors = edit_distance(ref_words, hyp_words)
    ref_len = len(ref_words)
    return errors / max(ref_len, 1), errors, ref_len


def utterance_key(path_or_name: str, channel: str) -> str:
    stem = Path(path_or_name).stem
    stem = re.sub(r"\[len=\d+\]", "", stem)
    stem = re.sub(r"\(layer\s+\d+\|\d+\)", "", stem)
    stem = re.sub(rf"_{re.escape(channel)}_", "_", stem)
    stem = re.sub(rf"_{re.escape(channel)}(?=_|\b)", "", stem)
    stem = re.sub(r"_(pred|lower_alpha=[^_]+|upper_alpha=[^_]+)$", "", stem)
    stem = re.sub(r"_(lower|upper|pred)$", "", stem)
    match = UTT_ID_RE.search(stem)
    return match.group(0) if match else stem.strip("_-. ")


def true_len_from_name(path_or_name: str, fallback: int) -> int:
    match = LEN_RE.search(Path(path_or_name).name)
    return int(match.group(1)) if match else fallback


def alpha_from_name(path_or_name: str) -> Optional[str]:
    match = ALPHA_RE.search(Path(path_or_name).stem)
    return match.group(1) if match else None


def looks_like_role(path: Path, role: str, channel: str, alpha: Optional[str]) -> bool:
    name = path.name
    if role == "pred":
        if f"_{channel}_pred" in name or name.endswith("_pred.npy"):
            return True
        match = LAYER_RE.search(name)
        return bool(match and int(match.group(1)) == 0)
    if role == "lower":
        if f"_{channel}_lower_alpha=" in name or "_lower_alpha=" in name:
            return alpha is None or f"lower_alpha={alpha}" in name
        match = LAYER_RE.search(name)
        return bool(match and int(match.group(1)) == 1)
    if role == "upper":
        if f"_{channel}_upper_alpha=" in name or "_upper_alpha=" in name:
            return alpha is None or f"upper_alpha={alpha}" in name
        match = LAYER_RE.search(name)
        return bool(match and int(match.group(1)) == 2)
    return False


def index_outputs(root: Path, role: str, channel: str, alpha: Optional[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.npy")):
        if looks_like_role(path, role, channel, alpha):
            out.setdefault(utterance_key(path.name, channel), path)
    return out


def sort_alpha_key(alpha: str) -> Tuple[int, object]:
    try:
        return (0, float(alpha))
    except ValueError:
        return (1, alpha)


def available_alphas(root: Path, channel: str) -> List[str]:
    found = set()
    for path in sorted(root.rglob("*.npy")):
        if f"_{channel}_lower_alpha=" not in path.name and "_lower_alpha=" not in path.name:
            continue
        alpha = alpha_from_name(path.name)
        if alpha is not None:
            found.add(alpha)
    return sorted(found, key=sort_alpha_key)


def index_pred_audio(root: Path, channel: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".wav", ".flac"}:
            continue
        if f"_{channel}_pred" in path.name or path.stem.endswith("_pred"):
            out.setdefault(utterance_key(path.name, channel), path)
    return out


def index_texts(root: Path, channel: str) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if "_ch" in path.name and channel not in path.name:
            continue
        out.setdefault(utterance_key(path.name, channel), path)
    return out


def load_2d_array(path: Path) -> np.ndarray:
    arr = np.squeeze(np.load(path).astype(np.float32))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D time-frequency array, got {arr.shape}: {path}")
    return arr


def as_whisper_mel(path: Path, true_t: int, max_t: int) -> torch.Tensor:
    import torch
    import torch.nn.functional as F

    mel = load_2d_array(path)
    if mel.shape[0] != 80 and mel.shape[1] == 80:
        mel = mel.T
    if mel.shape[0] > 80:
        mel = mel[:80, :]
    if mel.shape[0] < 80:
        mel = np.pad(mel, ((0, 80 - mel.shape[0]), (0, 0)), mode="constant")
    mel = mel[:, :true_t]
    x = torch.from_numpy(mel)
    if x.shape[-1] > max_t:
        return x[:, :max_t]
    return F.pad(x, (0, max_t - x.shape[-1]))


def width_summary(lower_path: Path, upper_path: Path, modes: Sequence[str], true_t: Optional[int] = None) -> Dict[str, float]:
    lower = load_2d_array(lower_path)
    upper = load_2d_array(upper_path)
    if lower.shape != upper.shape:
        min_f = min(lower.shape[0], upper.shape[0])
        min_t = min(lower.shape[1], upper.shape[1])
        lower = lower[:min_f, :min_t]
        upper = upper[:min_f, :min_t]
    if true_t is not None:
        lower = lower[:, :true_t]
        upper = upper[:, :true_t]
    width = np.abs(upper - lower)
    values: Dict[str, float] = {}
    for mode in modes:
        if mode == "mean":
            values["width_mean"] = float(width.mean())
        elif mode == "median":
            values["width_median"] = float(np.median(width))
        elif mode == "p90":
            values["width_p90"] = float(np.percentile(width, 90.0))
        elif mode == "p95":
            values["width_p95"] = float(np.percentile(width, 95.0))
        elif mode == "max":
            values["width_max"] = float(width.max())
        else:
            raise ValueError(f"Unsupported width mode: {mode}")
    return values


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    return pearsonr(rankdata(x), rankdata(y))


def permutation_p_value(x: np.ndarray, y: np.ndarray, corr: float, n_perm: int, seed: int) -> float:
    if n_perm <= 0 or not np.isfinite(corr):
        return float("nan")
    rng = random.Random(seed)
    y_perm = list(y)
    more_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(y_perm)
        if abs(pearsonr(x, np.asarray(y_perm))) >= abs(corr):
            more_extreme += 1
    return (more_extreme + 1.0) / (n_perm + 1.0)


def decile_table(rows: List[Dict[str, object]], width_col: str) -> List[Dict[str, float]]:
    valid = [r for r in rows if math.isfinite(float(r[width_col]))]
    valid.sort(key=lambda r: float(r[width_col]))
    if not valid:
        return []
    bins: List[Dict[str, float]] = []
    for bin_idx, chunk in enumerate(np.array_split(valid, min(10, len(valid))), start=1):
        chunk_rows = list(chunk)
        bins.append(
            {
                "bin": float(bin_idx),
                "n": float(len(chunk_rows)),
                "width_min": float(min(float(r[width_col]) for r in chunk_rows)),
                "width_max": float(max(float(r[width_col]) for r in chunk_rows)),
                "mean_wer": float(np.mean([float(r["pred_wer"]) for r in chunk_rows])),
                "mean_word_errors": float(np.mean([float(r["word_errors"]) for r in chunk_rows])),
            }
        )
    return bins


@dataclass
class Pair:
    key: str
    pred: Path
    lower: Path
    upper: Path
    text: Path
    pred_audio: Optional[Path]


def collect_pairs(args: argparse.Namespace) -> List[Pair]:
    if args.alpha is None:
        alphas = available_alphas(args.pred_root, args.channel)
        if len(alphas) > 1:
            args.alpha = alphas[0]
            print(f"[INFO] detected alphas={alphas}; using alpha={args.alpha}. Pass --alpha to choose another.")
        elif len(alphas) == 1:
            args.alpha = alphas[0]
            print(f"[INFO] detected alpha={args.alpha}")

    pred = index_outputs(args.pred_root, "pred", args.channel, args.alpha)
    lower = index_outputs(args.pred_root, "lower", args.channel, args.alpha)
    upper = index_outputs(args.pred_root, "upper", args.channel, args.alpha)
    text = index_texts(args.text_root, args.channel)
    pred_audio = index_pred_audio(args.pred_root, args.channel) if args.whisper_input == "audio" else {}

    common = sorted(set(pred) & set(lower) & set(upper) & set(text))
    if args.whisper_input == "audio":
        common = sorted(set(common) & set(pred_audio))
    if args.limit is not None:
        common = common[: args.limit]
    if not common:
        raise RuntimeError(
            "No matched pred/lower/upper/text rows found. "
            f"pred={len(pred)} lower={len(lower)} upper={len(upper)} text={len(text)} pred_audio={len(pred_audio)}"
        )
    return [Pair(k, pred[k], lower[k], upper[k], text[k], pred_audio.get(k)) for k in common]


def transcribe_pair(
    pair: Pair,
    model,
    processor,
    args: argparse.Namespace,
    device,
) -> str:
    import torch

    if args.whisper_input == "whisper-mel":
        fallback_len = load_2d_array(pair.pred).shape[-1]
        true_t = true_len_from_name(pair.pred.name, fallback_len)
        input_features = as_whisper_mel(pair.pred, true_t=true_t, max_t=args.max_t).unsqueeze(0).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features.float(),
                num_beams=args.num_beams,
                early_stopping=True,
                repetition_penalty=args.repetition_penalty,
            )
        return clean_text(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])

    if pair.pred_audio is None:
        raise RuntimeError(f"Missing pred audio for {pair.key}")
    import librosa

    audio, _ = librosa.load(pair.pred_audio, sr=16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            num_beams=args.num_beams,
            early_stopping=True,
            repetition_penalty=args.repetition_penalty,
        )
    return clean_text(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def maybe_plot(rows: List[Dict[str, object]], width_col: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skipping scatter plot: {exc}")
        return
    x = np.asarray([float(r[width_col]) for r in rows], dtype=np.float64)
    y = np.asarray([float(r["pred_wer"]) for r in rows], dtype=np.float64)
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=16, alpha=0.65)
    plt.xlabel(width_col)
    plt.ylabel("Whisper WER on pred head")
    plt.title("CP interval width vs Whisper error")
    plt.grid(True, alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlate dereverberation CP interval width with per-utterance Whisper WER."
    )
    parser.add_argument("--pred-root", type=Path, required=True, help="Root containing pred/lower/upper U-Net outputs.")
    parser.add_argument("--text-root", type=Path, required=True, help="Root containing reference transcription .txt files.")
    parser.add_argument("--out-dir", type=Path, default=Path("script_res/cp_width_vs_whisper_error"))
    parser.add_argument("--alpha", type=str, default=None, help="Alpha to analyze, e.g. 0.2. Defaults to the first match.")
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--whisper-input", choices=["whisper-mel", "audio"], default="whisper-mel")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching; current decoding is per utterance.")
    parser.add_argument("--max-t", type=int, default=3000)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--width-modes", nargs="+", default=["mean", "median", "p90", "p95"])
    parser.add_argument("--primary-width", type=str, default="width_mean")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

    pairs = collect_pairs(args)
    print(f"[INFO] matched examples: {len(pairs)}")

    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    rows: List[Dict[str, object]] = []
    for pair in tqdm(pairs, desc="Transcribing pred head"):
        reference = clean_text(pair.text.read_text(encoding="utf-8"))
        hypothesis = transcribe_pair(pair, model, processor, args, device)
        pred_wer, word_errors, ref_words = word_error_rate(reference, hypothesis)
        fallback_len = load_2d_array(pair.pred).shape[-1]
        true_t = true_len_from_name(pair.pred.name, fallback_len)
        widths = width_summary(pair.lower, pair.upper, args.width_modes, true_t=true_t)
        row: Dict[str, object] = {
            "key": pair.key,
            "alpha": args.alpha or alpha_from_name(pair.lower.name) or "",
            "reference": reference,
            "hypothesis": hypothesis,
            "pred_wer": pred_wer,
            "word_errors": word_errors,
            "ref_words": ref_words,
            "pred_path": str(pair.pred),
            "lower_path": str(pair.lower),
            "upper_path": str(pair.upper),
            "text_path": str(pair.text),
            "true_frames": true_t,
        }
        row.update(widths)
        rows.append(row)

    if args.primary_width not in rows[0]:
        raise ValueError(f"--primary-width must be one of the written width columns, got {args.primary_width}")

    per_file_csv = args.out_dir / "per_utterance_width_vs_wer.csv"
    write_csv(per_file_csv, rows)

    summary: Dict[str, object] = {
        "n": len(rows),
        "model_name": args.model_name,
        "whisper_input": args.whisper_input,
        "alpha": args.alpha,
        "mean_pred_wer": float(np.mean([float(r["pred_wer"]) for r in rows])),
        "correlations": {},
        "deciles": decile_table(rows, args.primary_width),
    }

    y = np.asarray([float(r["pred_wer"]) for r in rows], dtype=np.float64)
    for width_col in [k for k in rows[0] if k.startswith("width_")]:
        x = np.asarray([float(r[width_col]) for r in rows], dtype=np.float64)
        pearson = pearsonr(x, y)
        summary["correlations"][width_col] = {
            "pearson_r": pearson,
            "pearson_perm_p_two_sided": permutation_p_value(x, y, pearson, args.permutations, args.seed),
            "spearman_r": spearmanr(x, y),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_json = args.out_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    maybe_plot(rows, args.primary_width, args.out_dir / "scatter_width_vs_wer.png")

    print(f"[SAVE] per-utterance rows: {per_file_csv}")
    print(f"[SAVE] summary: {summary_json}")
    print(json.dumps(summary["correlations"], indent=2))


if __name__ == "__main__":
    main()
