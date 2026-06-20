from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def index_arrays(root: Path, channel: Optional[str] = None, pred_layer: Optional[int] = None) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.npy")):
        if channel and channel not in path.name and "_ch" in path.name:
            continue
        if pred_layer is not None and not is_pred_layer(path, pred_layer):
            continue
        index.setdefault(utterance_id(path.name), path)
        index.setdefault(normalize_stem(path.name), path)
    return index


def index_texts(root: Path, channel: Optional[str] = None) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.txt")):
        if channel and channel not in path.name and "_ch" in path.name:
            continue
        index.setdefault(utterance_id(path.name), path)
        index.setdefault(normalize_stem(path.name), path)
    return index


def load_array(path: Path) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    x = np.squeeze(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got {x.shape}: {path}")
    if x.shape[0] > x.shape[1] and x.shape[1] in {80, 256}:
        x = x.T
    return x


def load_norm_stats(path: Optional[Path]) -> Optional[Tuple[float, float]]:
    if path is None:
        return None
    data = np.load(path)
    return float(data["min_val"]), float(data["max_val"])


def denormalize_minus1_1(x: np.ndarray, stats: Optional[Tuple[float, float]]) -> np.ndarray:
    if stats is None:
        return x
    min_val, max_val = stats
    return (x + 1.0) * 0.5 * (max_val - min_val) + min_val


def spec_to_power(spec: np.ndarray, scale: str) -> np.ndarray:
    if scale == "ln_power":
        return np.maximum(np.exp(spec), 1e-10)
    if scale == "log10_power":
        return np.maximum(10.0 ** spec, 1e-10)
    if scale == "power":
        return np.maximum(spec, 1e-10)
    if scale == "ln_magnitude":
        mag = np.maximum(np.exp(spec), 1e-10)
        return mag * mag
    if scale == "magnitude":
        mag = np.maximum(spec, 1e-10)
        return mag * mag
    raise ValueError(f"Unknown spec scale: {scale}")


def whisper_mel_from_power(power_spec: np.ndarray, mel_filters: torch.Tensor) -> np.ndarray:
    freq_bins = mel_filters.shape[1]
    if power_spec.shape[0] < freq_bins:
        power_spec = np.pad(power_spec, ((0, freq_bins - power_spec.shape[0]), (0, 0)), mode="constant")
    if power_spec.shape[0] > freq_bins:
        power_spec = power_spec[:freq_bins, :]

    spec_t = torch.from_numpy(power_spec).float()
    mel = mel_filters @ spec_t
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.numpy().astype(np.float32)


def true_len_from_paths(paths: List[Path], fallback: int) -> int:
    for path in paths:
        match = LEN_RE.search(path.name)
        if match:
            return int(match.group(1))
    return fallback


class StftDirectMelDataset(Dataset):
    def __init__(
        self,
        clean_root: Path,
        reverb_root: Path,
        pred_root: Path,
        text_root: Path,
        tokenizer: WhisperTokenizer,
        processor: WhisperProcessor,
        channel: str,
        pred_layer: int,
        spec_scale: str,
        pred_norm_stats: Optional[Tuple[float, float]],
        limit: Optional[int],
    ):
        clean_idx = index_arrays(clean_root)
        reverb_idx = index_arrays(reverb_root, channel=channel)
        pred_idx = index_arrays(pred_root, pred_layer=pred_layer)
        text_idx = index_texts(text_root, channel=channel)

        common = sorted(set(clean_idx) & set(reverb_idx) & set(pred_idx) & set(text_idx))
        if limit is not None:
            common = common[:limit]
        if not common:
            raise RuntimeError(
                "No clean/reverb/pred/text matches found. "
                f"clean={len(clean_idx)} reverb={len(reverb_idx)} pred={len(pred_idx)} text={len(text_idx)}"
            )

        self.items = []
        self.tokenizer = tokenizer
        self.spec_scale = spec_scale
        self.pred_norm_stats = pred_norm_stats
        self.mel_filters = processor.feature_extractor.mel_filters
        if not isinstance(self.mel_filters, torch.Tensor):
            self.mel_filters = torch.tensor(self.mel_filters, dtype=torch.float32)
        self.mel_filters = self.mel_filters.float()
        if self.mel_filters.shape[0] != 80 and self.mel_filters.shape[1] == 80:
            self.mel_filters = self.mel_filters.T
        if self.mel_filters.shape[0] != 80:
            raise ValueError(f"Expected Whisper mel filters to have 80 mel rows, got {tuple(self.mel_filters.shape)}")

        for uid in common:
            clean_path = clean_idx[uid]
            reverb_path = reverb_idx[uid]
            pred_path = pred_idx[uid]
            text_path = text_idx[uid]
            text = clean_text(text_path.read_text(encoding="utf-8").strip())
            fallback = min(load_array(clean_path).shape[1], load_array(reverb_path).shape[1], load_array(pred_path).shape[1])
            true_t = true_len_from_paths([clean_path, reverb_path, pred_path, text_path], fallback)
            self.items.append((uid, clean_path, reverb_path, pred_path, true_t, text))

        print(f"[INFO] matched STFT-direct examples: {len(self.items)}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_as_whisper_mel(self, path: Path, true_t: int, is_pred: bool) -> torch.Tensor:
        spec = load_array(path)[:, :true_t]
        if is_pred:
            spec = denormalize_minus1_1(spec, self.pred_norm_stats)
        power = spec_to_power(spec, self.spec_scale)
        mel = whisper_mel_from_power(power, self.mel_filters)
        return torch.from_numpy(mel)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        uid, clean_path, reverb_path, pred_path, true_t, text = self.items[idx]
        return {
            "uid": uid,
            "clean": self._load_as_whisper_mel(clean_path, true_t, is_pred=False),
            "reverb": self._load_as_whisper_mel(reverb_path, true_t, is_pred=False),
            "pred": self._load_as_whisper_mel(pred_path, true_t, is_pred=True),
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
        out = {key: torch.stack([self._pad_or_crop(item[key]) for item in batch]) for key in ["clean", "reverb", "pred"]}
        labels = pad_sequence([item["label_ids"] for item in batch], batch_first=True, padding_value=self.pad_id)
        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == self.pad_id] = -100
        out["labels"] = labels_for_loss
        out["labels_text_ids"] = labels
        return out


@torch.no_grad()
def evaluate_variant(model, loader, processor, tokenizer, device, variant: str) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    refs: List[str] = []
    hyps: List[str] = []
    for batch in tqdm(loader, desc=f"Evaluating STFT->mel {variant}"):
        input_features = batch[variant].to(device).float()
        labels = batch["labels"].to(device)
        out = model(input_features=input_features, labels=labels, use_cache=False)
        total_loss += float(out.loss.item())
        generated_ids = model.generate(input_features=input_features, num_beams=5, early_stopping=True, repetition_penalty=1.2)
        hyps.extend(clean_text(t) for t in processor.batch_decode(generated_ids, skip_special_tokens=True))
        refs.extend(clean_text(t) for t in tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True))
    val_wer = wer(refs, hyps)
    return {"variant": variant, "val_loss": total_loss / max(len(loader), 1), "val_wer": val_wer, "val_wer_pct": f"{val_wer:.3%}"}


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "val_loss", "val_wer", "val_wer_pct"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate STFT-domain clean/reverb/pred by converting directly to Whisper mel features.")
    parser.add_argument("--clean-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/clean_spec/val"))
    parser.add_argument("--reverb-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/reverb_spec/val"))
    parser.add_argument("--pred-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/dereverb_spec/stft_unet_v1/val"))
    parser.add_argument("--text-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/transcription_matched/val"))
    parser.add_argument("--pred-norm-stats", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/norm_stats/spec_train_minmax.npz"))
    parser.add_argument("--no-pred-denorm", action="store_true")
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--pred-layer", type=int, default=0)
    parser.add_argument("--spec-scale", choices=["ln_power", "log10_power", "power", "ln_magnitude", "magnitude"], default="ln_power")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-t", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=["clean", "reverb", "pred"])
    parser.add_argument("--out-csv", type=Path, default=Path("/storage/tal/thesis/script_res/stft_unet_v1_direct_whisper_eval.csv"))
    args = parser.parse_args()

    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    pred_stats = None if args.no_pred_denorm else load_norm_stats(args.pred_norm_stats)

    dataset = StftDirectMelDataset(
        clean_root=args.clean_root,
        reverb_root=args.reverb_root,
        pred_root=args.pred_root,
        text_root=args.text_root,
        tokenizer=tokenizer,
        processor=processor,
        channel=args.channel,
        pred_layer=args.pred_layer,
        spec_scale=args.spec_scale,
        pred_norm_stats=pred_stats,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer, args.max_t))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    rows: List[Dict[str, object]] = []
    for variant in args.variants:
        if variant not in {"clean", "reverb", "pred"}:
            raise ValueError(f"Unknown variant {variant}")
        row = evaluate_variant(model, loader, processor, tokenizer, device, variant)
        rows.append(row)
        write_rows(args.out_csv, rows)
        print(row)

    print("[CHECK] Direct path used: STFT/power -> mel filterbank -> Whisper input_features. No audio reconstruction.")
    print(f"[SAVE] {args.out_csv}")


if __name__ == "__main__":
    main()
