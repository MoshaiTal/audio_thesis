from __future__ import annotations

import argparse
import csv
import json
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


MEL_EXTS = {".npy"}
LEN_RE = re.compile(r"\[len=(\d+)\]")
TBINS_RE = re.compile(r"actual_tbins=(\d+)")
UTT_ID_RE = re.compile(r"(?:train|val|test|cal)_\d+")


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def utterance_id(path_or_stem: str) -> str:
    text = Path(path_or_stem).stem
    m = UTT_ID_RE.search(text)
    if m:
        return m.group(0)
    return normalize_stem(path_or_stem)


def normalize_stem(path_or_stem: str) -> str:
    stem = Path(path_or_stem).stem
    stem = re.sub(r"\[len=\d+\]", "", stem)
    stem = re.sub(r"actual_tbins=\d+", "", stem)
    stem = re.sub(r"(_?mel|_?spec|_?rec)$", "", stem)
    stem = re.sub(r"_ch\d+_(pred|lower_alpha=.*|upper_alpha=.*)$", "", stem)
    stem = re.sub(r"_ch\d+_", "_", stem)
    return stem.strip("_-. ")


def infer_true_t(*path_strs: str, fallback: Optional[int] = None) -> int:
    for path_str in path_strs:
        m = LEN_RE.search(path_str)
        if m:
            return int(m.group(1))
        m = TBINS_RE.search(path_str)
        if m:
            return int(m.group(1)) + 1
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not infer true length from paths: {path_strs}")


def is_center_pred_file(path: Path, channel: Optional[str]) -> bool:
    name = path.name
    if channel:
        return bool(re.search(rf"_{re.escape(channel)}_pred(?:\.|$)", name))
    return bool(re.search(r"_pred(?:\.|$)", name))


def file_matches_channel(path: Path, channel: Optional[str]) -> bool:
    return channel is None or channel in path.name


def index_mel_files(root: Path, channel: Optional[str] = None, require_center_pred: bool = False) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEL_EXTS:
            continue
        if require_center_pred and not is_center_pred_file(path, channel):
            continue
        if not require_center_pred and not file_matches_channel(path, channel):
            continue
        uid = utterance_id(path.name)
        out.setdefault(uid, path)
        out.setdefault(normalize_stem(path.name), path)
    return out


def read_manifest(path: Optional[Path], limit: Optional[int]) -> List[Dict[str, str]]:
    if path is None:
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit is not None else rows


def find_path(row: Dict[str, str], index: Dict[str, Path]) -> Optional[Path]:
    candidates = []
    if "stem" in row:
        candidates.append(row["stem"])
    for key in ["pred", "clean", "reverb", "text"]:
        if key in row:
            candidates.append(row[key])
    for cand in candidates:
        uid = utterance_id(cand)
        if uid in index:
            return index[uid]
        stem = normalize_stem(cand)
        if stem in index:
            return index[stem]
    return None


def load_mel(path: Path, true_t: int) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D mel [80,T] or feature [F,T], got {x.shape}: {path}")
    if x.shape[0] > 80:
        x = x[:80, :]
    if x.shape[0] < 80:
        x = np.pad(x, ((0, 80 - x.shape[0]), (0, 0)), mode="constant")
    return x[:, :true_t]


class GeneratedMelDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        clean_root: Path,
        reverb_root: Path,
        pred_root: Path,
        tokenizer: WhisperTokenizer,
        channel: str,
        limit: Optional[int],
    ):
        rows = read_manifest(manifest, limit)
        clean_idx = index_mel_files(clean_root)
        reverb_idx = index_mel_files(reverb_root, channel=channel)
        pred_idx = index_mel_files(pred_root, channel=channel, require_center_pred=True)

        self.items = []
        self.tok = tokenizer
        for row in rows:
            uid = utterance_id(row.get("stem") or row.get("pred") or row.get("text", ""))
            clean_path = find_path(row, clean_idx)
            reverb_path = find_path(row, reverb_idx)
            pred_path = find_path(row, pred_idx)
            text_path = Path(row["text"]) if "text" in row else None
            if clean_path and reverb_path and pred_path and text_path and text_path.exists():
                text = text_path.read_text(encoding="utf-8").strip()
                true_t = infer_true_t(str(pred_path), str(clean_path), str(reverb_path), row.get("pred", ""))
                self.items.append((uid, clean_path, reverb_path, pred_path, true_t, clean_text(text)))

        if not self.items:
            raise ValueError("No matched clean/reverb/pred mel triplets found.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        uid, clean_path, reverb_path, pred_path, true_t, text = self.items[idx]
        clean = load_mel(clean_path, true_t)
        reverb = load_mel(reverb_path, true_t)
        pred = load_mel(pred_path, true_t)
        return {
            "uid": uid,
            "clean": torch.from_numpy(clean),
            "reverb": torch.from_numpy(reverb),
            "pred": torch.from_numpy(pred),
            "label_ids": torch.tensor(self.tok(text).input_ids, dtype=torch.long),
            "clean_path": str(clean_path),
            "reverb_path": str(reverb_path),
            "pred_path": str(pred_path),
        }


class Collator:
    def __init__(self, tokenizer: WhisperTokenizer, max_t: int = 3000):
        self.pad_id = tokenizer.pad_token_id
        self.max_t = max_t

    def _pad_or_crop(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.max_t:
            return x[:, : self.max_t]
        return F.pad(x, (0, self.max_t - x.shape[-1]))

    def __call__(self, batch: List[Dict]) -> Dict:
        out = {
            key: torch.stack([self._pad_or_crop(item[key]) for item in batch])
            for key in ["clean", "reverb", "pred"]
        }
        labels = pad_sequence([item["label_ids"] for item in batch], batch_first=True, padding_value=self.pad_id)
        labels_for_loss = labels.clone()
        for i, item in enumerate(batch):
            length = len(item["label_ids"])
            if length < labels_for_loss.shape[1]:
                labels_for_loss[i, length:] = -100
        out["labels"] = labels_for_loss
        out["labels_text_ids"] = labels
        return out


@torch.no_grad()
def evaluate_variant(model, loader, processor, tokenizer, device, variant: str) -> Dict[str, str | float]:
    model.eval()
    total_loss = 0.0
    preds: List[str] = []
    refs: List[str] = []

    for batch in tqdm(loader, desc=f"Evaluating {variant}"):
        features = batch[variant].to(device).float()
        labels = batch["labels"].to(device)
        out = model(input_features=features, labels=labels, use_cache=False)
        total_loss += out.loss.item()

        gen_ids = model.generate(
            input_features=features,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_strs = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True)
        preds += [clean_text(t) for t in pred_strs]
        refs += [clean_text(t) for t in ref_strs]

    val_wer = wer(refs, preds)
    return {
        "variant": variant,
        "val_loss": total_loss / max(len(loader), 1),
        "val_wer": val_wer,
        "val_wer_pct": f"{val_wer:.3%}",
    }


def write_results(path: Path, rows: List[Dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "val_loss", "val_wer", "val_wer_pct"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated clean/reverb/pred mel files with vanilla Whisper.")
    parser.add_argument("--manifest", type=Path, default=Path("/storage/tal/thesis/condwhisper_manifests/val_cpwidth.jsonl"))
    parser.add_argument("--clean-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/clean_melspec/val"))
    parser.add_argument("--reverb-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/reverb_melspec/val"))
    parser.add_argument("--pred-root", type=Path, required=True, help="Example: /storage/tal/thesis/DataBase_BIUREV/dereverb_mel/calibrated_asr_aware_v1/val")
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-t", type=int, default=3000)
    parser.add_argument("--variants", nargs="+", default=["clean", "reverb", "pred"])
    parser.add_argument("--out-csv", type=Path, default=Path("/storage/tal/thesis/script_res/generated_mel_whisper_eval.csv"))
    args = parser.parse_args()

    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    ds = GeneratedMelDataset(
        manifest=args.manifest,
        clean_root=args.clean_root,
        reverb_root=args.reverb_root,
        pred_root=args.pred_root,
        tokenizer=tokenizer,
        channel=args.channel,
        limit=args.limit,
    )
    print(f"[INFO] Found {len(ds)} matched mel triplets")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer, max_t=args.max_t))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    rows = []
    for variant in args.variants:
        if variant not in {"clean", "reverb", "pred"}:
            raise ValueError(f"Unknown variant {variant}. Choose clean, reverb, pred.")
        row = evaluate_variant(model, loader, processor, tokenizer, device, variant)
        rows.append(row)
        print(row)
        write_results(args.out_csv, rows)

    print(f"[SAVE] {args.out_csv}")


if __name__ == "__main__":
    main()
