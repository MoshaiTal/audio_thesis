from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer


UTT_ID_RE = re.compile(r"(?:train|val|test|cal)_\d+")
LEN_RE = re.compile(r"\[len=(\d+)\]")
LAYER_RE = re.compile(r"\(layer\s+(\d+)\|(\d+)\)")


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def normalize_stem(path_or_name: str) -> str:
    stem = Path(path_or_name).stem
    stem = re.sub(r"\[len=\d+\]", "", stem)
    stem = re.sub(r"\(layer\s+\d+\|\d+\)", "", stem)
    stem = re.sub(r"_ch\d+", "", stem)
    stem = re.sub(r"_(pred|lower_alpha=.*|upper_alpha=.*)$", "", stem)
    return stem.strip("_-. ")


def utterance_id(path_or_name: str) -> str:
    match = UTT_ID_RE.search(Path(path_or_name).name)
    if match:
        return match.group(0)
    return normalize_stem(path_or_name)


def valid_len(path: Path, fallback: int = 3000) -> int:
    match = LEN_RE.search(path.name)
    if not match:
        return fallback
    return min(int(match.group(1)), fallback)


def layer_id(path: Path) -> Optional[int]:
    match = LAYER_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1))


def index_features(root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.npy")):
        index.setdefault(utterance_id(path.name), path)
        index.setdefault(normalize_stem(path.name), path)
    return index


def index_prediction_layers(root: Path) -> Dict[int, Dict[str, Path]]:
    by_layer: Dict[int, Dict[str, Path]] = {}
    no_layer: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.npy")):
        lid = layer_id(path)
        target = no_layer if lid is None else by_layer.setdefault(lid, {})
        target.setdefault(utterance_id(path.name), path)
        target.setdefault(normalize_stem(path.name), path)

    if no_layer and not by_layer:
        by_layer[0] = no_layer
    return by_layer


def index_texts(root: Path, channel: Optional[str]) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if channel and channel not in path.name and "_ch" in path.name:
            continue
        index.setdefault(utterance_id(path.name), path)
        index.setdefault(normalize_stem(path.name), path)
    return index


def load_features(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected [80,T] features, got {x.shape}: {path}")
    if x.shape[0] != 80 and x.shape[1] == 80:
        x = x.T
    if x.shape[0] > 80:
        x = x[:80]
    if x.shape[0] < 80:
        x = np.pad(x, ((0, 80 - x.shape[0]), (0, 0)), mode="constant")
    return x


def crop_pair(a: np.ndarray, b: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    t = min(a.shape[-1], b.shape[-1], length)
    return a[:, :t], b[:, :t]


def feature_stats(x: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p50": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    av = a.reshape(-1).astype(np.float64)
    bv = b.reshape(-1).astype(np.float64)
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(av, bv) / denom)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FeatureDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, Path, Path, Path, Path]],
        tokenizer: WhisperTokenizer,
        variant: str,
    ):
        self.items = items
        self.tokenizer = tokenizer
        self.variant = variant

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        uid, clean_path, reverb_path, pred_path, text_path = self.items[idx]
        path_by_variant = {"clean": clean_path, "reverb": reverb_path, "pred": pred_path}
        text = clean_text(text_path.read_text(encoding="utf-8").strip())
        return {
            "uid": uid,
            "features": torch.from_numpy(load_features(path_by_variant[self.variant])),
            "label_ids": torch.tensor(self.tokenizer(text).input_ids, dtype=torch.long),
        }


class Collator:
    def __init__(self, tokenizer: WhisperTokenizer, max_t: int):
        self.pad_id = tokenizer.pad_token_id
        self.max_t = max_t

    def _pad_or_crop(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.max_t:
            return x[:, : self.max_t]
        return F.pad(x, (0, self.max_t - x.shape[-1]))

    def __call__(self, batch: List[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        features = torch.stack([self._pad_or_crop(item["features"]) for item in batch])
        labels = pad_sequence([item["label_ids"] for item in batch], batch_first=True, padding_value=self.pad_id)
        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == self.pad_id] = -100
        return {"features": features, "labels": labels_for_loss, "labels_text_ids": labels}


@torch.no_grad()
def evaluate_wer(
    model,
    processor: WhisperProcessor,
    tokenizer: WhisperTokenizer,
    items: List[Tuple[str, Path, Path, Path, Path]],
    variant: str,
    batch_size: int,
    max_t: int,
    device: torch.device,
) -> Dict[str, object]:
    dataset = FeatureDataset(items, tokenizer, variant)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Collator(tokenizer, max_t))
    total_loss = 0.0
    refs: List[str] = []
    hyps: List[str] = []
    for batch in tqdm(loader, desc=f"Whisper {variant}"):
        input_features = batch["features"].to(device).float()
        labels = batch["labels"].to(device)
        out = model(input_features=input_features, labels=labels, use_cache=False)
        total_loss += float(out.loss.item())
        generated_ids = model.generate(input_features=input_features, num_beams=5, early_stopping=True, repetition_penalty=1.2)
        hyps.extend(clean_text(t) for t in processor.batch_decode(generated_ids, skip_special_tokens=True))
        refs.extend(clean_text(t) for t in tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True))
    val_wer = wer(refs, hyps)
    return {
        "variant": variant,
        "num_examples": len(items),
        "val_loss": total_loss / max(len(loader), 1),
        "val_wer": val_wer,
        "val_wer_pct": f"{val_wer:.3%}",
    }


def build_items(
    clean_idx: Dict[str, Path],
    reverb_idx: Dict[str, Path],
    pred_idx: Dict[str, Path],
    text_idx: Dict[str, Path],
    limit: Optional[int],
) -> List[Tuple[str, Path, Path, Path, Path]]:
    common = sorted(set(clean_idx) & set(reverb_idx) & set(pred_idx) & set(text_idx))
    if limit is not None:
        common = common[:limit]
    return [(uid, clean_idx[uid], reverb_idx[uid], pred_idx[uid], text_idx[uid]) for uid in common]


def summarize_layer(items: List[Tuple[str, Path, Path, Path, Path]], layer: int) -> Dict[str, object]:
    clean_values: List[np.ndarray] = []
    reverb_values: List[np.ndarray] = []
    pred_values: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []

    for uid, clean_path, reverb_path, pred_path, _ in tqdm(items, desc=f"Stats layer {layer}"):
        length = valid_len(clean_path)
        clean = load_features(clean_path)
        reverb = load_features(reverb_path)
        pred = load_features(pred_path)
        clean_c, reverb_c = crop_pair(clean, reverb, length)
        clean_p, pred_c = crop_pair(clean, pred, length)

        rows.append(
            {
                "uid": uid,
                "layer": layer,
                "valid_len": min(clean_c.shape[-1], clean_p.shape[-1]),
                "clean_reverb_mse": mse(clean_c, reverb_c),
                "clean_pred_mse": mse(clean_p, pred_c),
                "clean_reverb_mae": mae(clean_c, reverb_c),
                "clean_pred_mae": mae(clean_p, pred_c),
                "clean_reverb_corr": corr(clean_c, reverb_c),
                "clean_pred_corr": corr(clean_p, pred_c),
                "pred_minus_reverb_mse": mse(clean_p, pred_c) - mse(clean_c, reverb_c),
                "pred_minus_reverb_corr": corr(clean_p, pred_c) - corr(clean_c, reverb_c),
            }
        )
        clean_values.append(clean_c)
        reverb_values.append(reverb_c)
        pred_values.append(pred_c)

    def concat(xs: List[np.ndarray]) -> np.ndarray:
        if not xs:
            return np.zeros((80, 1), dtype=np.float32)
        return np.concatenate(xs, axis=1)

    clean_all = concat(clean_values)
    reverb_all = concat(reverb_values)
    pred_all = concat(pred_values)

    summary: Dict[str, object] = {"layer": layer, "num_examples": len(items)}
    for prefix, arr in [("clean", clean_all), ("reverb", reverb_all), ("pred", pred_all)]:
        for key, value in feature_stats(arr).items():
            summary[f"{prefix}_{key}"] = value

    for metric in [
        "clean_reverb_mse",
        "clean_pred_mse",
        "clean_reverb_mae",
        "clean_pred_mae",
        "clean_reverb_corr",
        "clean_pred_corr",
        "pred_minus_reverb_mse",
        "pred_minus_reverb_corr",
    ]:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
        summary[f"{metric}_median"] = float(np.median(values)) if values else float("nan")

    return {"summary": summary, "rows": rows}


def plot_examples(
    items: List[Tuple[str, Path, Path, Path, Path]],
    layer: int,
    out_dir: Path,
    num_examples: int,
    max_t: int,
) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for uid, clean_path, reverb_path, pred_path, _ in items[:num_examples]:
        length = min(valid_len(clean_path), max_t)
        clean = load_features(clean_path)[:, :length]
        reverb = load_features(reverb_path)[:, :length]
        pred = load_features(pred_path)[:, :length]
        diff = pred - clean

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        for ax, title, data in [
            (axes[0], "clean Whisper features", clean),
            (axes[1], "reverb Whisper features", reverb),
            (axes[2], f"pred Whisper features layer {layer}", pred),
            (axes[3], "pred - clean", diff),
        ]:
            im = ax.imshow(data, aspect="auto", origin="lower", interpolation="nearest")
            ax.set_title(title)
            ax.set_ylabel("mel bin")
            fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
        axes[-1].set_xlabel("Whisper frame")
        fig.suptitle(uid)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{uid}_layer{layer}.png", dpi=140)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose backend-matched STFT-to-Whisper-mel predictions.")
    parser.add_argument("--clean-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV_whisper_aligned/clean_whisper_melspec/val"))
    parser.add_argument("--reverb-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV_whisper_aligned/reverb_whisper_melspec/val"))
    parser.add_argument("--pred-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV_whisper_aligned/dereverb_whisper_mel/stft_to_whisper_mel_v1/val"))
    parser.add_argument("--text-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/transcription_matched/val"))
    parser.add_argument("--out-dir", type=Path, default=Path("/storage/tal/thesis/script_res/stft_to_whisper_mel_diagnostic"))
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--layers", nargs="*", type=int, default=None, help="Prediction layers to test. Default: all detected layers.")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-t", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--plot-examples", type=int, default=5)
    parser.add_argument("--skip-whisper", action="store_true", help="Only compute feature stats and plots; skip WER evaluation.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    clean_idx = index_features(args.clean_root)
    reverb_idx = index_features(args.reverb_root)
    pred_layers = index_prediction_layers(args.pred_root)
    text_idx = index_texts(args.text_root, channel=args.channel)

    detected_layers = sorted(pred_layers)
    layers = args.layers if args.layers is not None else detected_layers
    if not layers:
        raise RuntimeError(f"No prediction .npy files found under {args.pred_root}")

    print(f"[INDEX] clean={len(clean_idx)} reverb={len(reverb_idx)} text={len(text_idx)}")
    print(f"[INDEX] detected prediction layers: {detected_layers}")

    all_summary_rows: List[Dict[str, object]] = []
    all_per_sample_rows: List[Dict[str, object]] = []
    wer_rows: List[Dict[str, object]] = []

    tokenizer = None
    processor = None
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.skip_whisper:
        tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
        processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids

    baseline_done = False
    for layer in layers:
        if layer not in pred_layers:
            print(f"[WARN] requested layer {layer} was not found; skipping")
            continue
        items = build_items(clean_idx, reverb_idx, pred_layers[layer], text_idx, args.limit)
        print(f"[MATCH] layer={layer} examples={len(items)}")
        if not items:
            continue

        stats = summarize_layer(items, layer)
        all_summary_rows.append(stats["summary"])
        all_per_sample_rows.extend(stats["rows"])
        plot_examples(items, layer, args.out_dir, args.plot_examples, args.max_t)

        if not args.skip_whisper and model is not None and processor is not None and tokenizer is not None:
            if not baseline_done:
                for variant in ["clean", "reverb"]:
                    row = evaluate_wer(model, processor, tokenizer, items, variant, args.batch_size, args.max_t, device)
                    row["layer"] = "baseline"
                    wer_rows.append(row)
                    write_csv(args.out_dir / "wer_by_layer.csv", wer_rows)
                    print(row)
                baseline_done = True

            pred_row = evaluate_wer(model, processor, tokenizer, items, "pred", args.batch_size, args.max_t, device)
            pred_row["layer"] = layer
            wer_rows.append(pred_row)
            write_csv(args.out_dir / "wer_by_layer.csv", wer_rows)
            print(pred_row)

    write_csv(args.out_dir / "feature_summary_by_layer.csv", all_summary_rows)
    write_csv(args.out_dir / "per_sample_metrics.csv", all_per_sample_rows)

    aggregate = {
        "clean_root": str(args.clean_root),
        "reverb_root": str(args.reverb_root),
        "pred_root": str(args.pred_root),
        "text_root": str(args.text_root),
        "layers_tested": [int(row["layer"]) for row in all_summary_rows],
        "num_summary_rows": len(all_summary_rows),
        "num_per_sample_rows": len(all_per_sample_rows),
        "wer_rows": wer_rows,
    }
    (args.out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(f"[SAVE] {args.out_dir}")
    print("[CHECK] This script feeds saved [80,T] Whisper input_features directly into Whisper.")


if __name__ == "__main__":
    main()
