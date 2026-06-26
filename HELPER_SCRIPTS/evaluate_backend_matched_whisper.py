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
    if match:
        return int(match.group(1)) == layer
    if layer == 0 and re.search(r"_pred(?:\.|$)", path.name):
        return True
    return False


def valid_len(path: Path, max_t: int = 3000) -> int:
    match = LEN_RE.search(path.name)
    if not match:
        return max_t
    return max(1, min(int(match.group(1)), max_t))


def index_features(root: Path, pred_layer: Optional[int] = None) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    npy_paths = sorted(root.rglob("*.npy"))
    layer_paths = [path for path in npy_paths if pred_layer is None or is_pred_layer(path, pred_layer)]

    # Some save paths contain only one prediction file per utterance and do not
    # encode "_pred" or "(layer ...)" in the filename. In that case, do not
    # silently produce an empty prediction index.
    if pred_layer is not None and npy_paths and not layer_paths:
        print(
            f"[WARN] No files in {root} matched pred_layer={pred_layer}; "
            "using all .npy files in that root."
        )
        layer_paths = npy_paths

    for path in layer_paths:
        if pred_layer is not None and len(layer_paths) != len(npy_paths) and not is_pred_layer(path, pred_layer):
            continue
        index.setdefault(utterance_id(path.name), path)
        index.setdefault(normalize_stem(path.name), path)
    return index


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


class BackendMatchedDataset(Dataset):
    def __init__(
        self,
        clean_root: Path,
        reverb_root: Path,
        pred_root: Path,
        text_root: Path,
        tokenizer: WhisperTokenizer,
        channel: str,
        pred_layer: int,
        limit: Optional[int],
        pred_tail: str,
        max_t: int,
    ):
        clean_idx = index_features(clean_root)
        reverb_idx = index_features(reverb_root)
        pred_idx = index_features(pred_root, pred_layer=pred_layer)
        text_idx = index_texts(text_root, channel=channel)

        common = sorted(set(clean_idx) & set(reverb_idx) & set(pred_idx) & set(text_idx))
        if limit is not None:
            common = common[:limit]
        if not common:
            raise RuntimeError(
                "No clean/reverb/pred/text matches found. "
                f"clean={len(clean_idx)} reverb={len(reverb_idx)} pred={len(pred_idx)} text={len(text_idx)}. "
                f"pred_root={pred_root}"
            )

        self.items: List[Tuple[str, Path, Path, Path, str]] = []
        self.tokenizer = tokenizer
        self.pred_tail = pred_tail
        self.max_t = max_t
        for uid in common:
            text = clean_text(text_idx[uid].read_text(encoding="utf-8").strip())
            self.items.append((uid, clean_idx[uid], reverb_idx[uid], pred_idx[uid], text))
        print(f"[INFO] matched backend-matched examples: {len(self.items)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        uid, clean_path, reverb_path, pred_path, text = self.items[idx]
        clean = load_features(clean_path)
        reverb = load_features(reverb_path)
        pred = load_features(pred_path)

        # Saved model predictions contain zeros after the real utterance because
        # the model output is multiplied by the training mask. Whisper expects a
        # valid padded/silent log-mel tail, not arbitrary zeros. For inference,
        # using the reverb tail is allowed because it comes from the observed
        # input audio, not from clean speech.
        if self.pred_tail != "zero":
            length = valid_len(pred_path, self.max_t)
            source = {"reverb": reverb, "clean": clean}.get(self.pred_tail)
            if source is None:
                raise ValueError(f"Unknown pred_tail mode: {self.pred_tail}")
            pred[:, length:] = source[:, length:]

        return {
            "uid": uid,
            "clean": torch.from_numpy(clean),
            "reverb": torch.from_numpy(reverb),
            "pred": torch.from_numpy(pred),
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
    total_loss = 0.0
    refs: List[str] = []
    hyps: List[str] = []
    for batch in tqdm(loader, desc=f"Evaluating {variant}"):
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
    parser = argparse.ArgumentParser(description="Evaluate exact Whisper input_features without any waveform or mel conversion.")
    parser.add_argument("--clean-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV_whisper_aligned/clean_whisper_melspec/val"))
    parser.add_argument("--reverb-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV_whisper_aligned/reverb_whisper_melspec/val"))
    parser.add_argument("--pred-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/dereverb_whisper_mel/stft_to_whisper_mel_v1/val"))
    parser.add_argument("--text-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/transcription_matched/val"))
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--pred-layer", type=int, default=0)
    parser.add_argument("--pred-tail", choices=["zero", "reverb", "clean"], default="zero")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-t", type=int, default=3000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", nargs="+", default=["clean", "reverb", "pred"])
    parser.add_argument("--out-csv", type=Path, default=Path("/storage/tal/thesis/script_res/stft_to_whisper_mel_v1_eval.csv"))
    args = parser.parse_args()

    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    dataset = BackendMatchedDataset(
        args.clean_root,
        args.reverb_root,
        args.pred_root,
        args.text_root,
        tokenizer,
        args.channel,
        args.pred_layer,
        args.limit,
        args.pred_tail,
        args.max_t,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer, args.max_t))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    rows: List[Dict[str, object]] = []
    for variant in args.variants:
        row = evaluate_variant(model, loader, processor, tokenizer, device, variant)
        rows.append(row)
        write_rows(args.out_csv, rows)
        print(row)

    print("[CHECK] No domain crossing: saved [80,T] input_features -> Whisper model(input_features=...).")
    print(f"[SAVE] {args.out_csv}")


if __name__ == "__main__":
    main()
