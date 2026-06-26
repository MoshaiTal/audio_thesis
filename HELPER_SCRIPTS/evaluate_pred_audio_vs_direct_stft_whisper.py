from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer


LEN_RE = re.compile(r"\[len=(\d+)\]")
TBINS_RE = re.compile(r"actual_tbins=(\d+)")
UTT_ID_RE = re.compile(r"(?:train|val|test|cal)_\d+")


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def normalize_stem(path_or_stem: str) -> str:
    stem = Path(path_or_stem).stem
    stem = re.sub(r"\[len=\d+\]", "", stem)
    stem = re.sub(r"actual_tbins=\d+", "", stem)
    stem = re.sub(r"(_?mel|_?spec|_?rec)$", "", stem)
    stem = re.sub(r"_ch\d+_(pred|lower_alpha=.*|upper_alpha=.*)$", "", stem)
    stem = re.sub(r"_ch\d+_", "_", stem)
    return stem.strip("_-. ")


def utterance_id(path_or_stem: str) -> str:
    text = Path(path_or_stem).stem
    match = UTT_ID_RE.search(text)
    if match:
        return match.group(0)
    return normalize_stem(path_or_stem)


def infer_true_t(*path_strs: str, fallback: Optional[int] = None) -> int:
    for path_str in path_strs:
        match = LEN_RE.search(path_str)
        if match:
            return int(match.group(1))
        match = TBINS_RE.search(path_str)
        if match:
            return int(match.group(1)) + 1
    if fallback is not None:
        return fallback
    raise ValueError(f"Could not infer true length from paths: {path_strs}")


def file_matches_channel(path: Path, channel: Optional[str]) -> bool:
    return channel is None or channel in path.name


def is_center_pred_file(path: Path, channel: Optional[str]) -> bool:
    name = path.name
    if channel:
        return bool(re.search(rf"_{re.escape(channel)}_pred(?:\.|$)", name))
    return bool(re.search(r"_pred(?:\.|$)", name))


def index_npy_files(root: Path, channel: Optional[str] = None, require_center_pred: bool = False) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in root.rglob("*.npy"):
        if require_center_pred and not is_center_pred_file(path, channel):
            continue
        if not require_center_pred and not file_matches_channel(path, channel):
            continue
        out.setdefault(utterance_id(path.name), path)
        out.setdefault(normalize_stem(path.name), path)
    return out


def read_manifest(path: Path, limit: Optional[int]) -> List[Dict[str, str]]:
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


def load_whisper_features(path: Path, true_t: int) -> np.ndarray:
    x = np.load(path).astype(np.float32)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D features, got {x.shape}: {path}")
    if x.shape[0] > 80:
        x = x[:80]
    if x.shape[0] < 80:
        x = np.pad(x, ((0, 80 - x.shape[0]), (0, 0)), mode="constant")
    return x[:, :true_t]


def whisper_logmel_from_logmag512(
    logmag_256: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    target_hop_length: int = 160,
    max_t: int = 3000,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    last_bin_mode: str = "floor",
) -> np.ndarray:
    """
    Diagnostic direct path:
        predicted log(|STFT_512|)
        -> power
        -> Slaney mel filterbank
        -> Whisper-style log compression
        -> time interpolation from 8 ms frames to 10 ms Whisper frames

    It avoids iSTFT and reverb phase. Because the model STFT grid is 512/128
    while Whisper normally uses 400/160, this is a diagnostic approximation,
    not a bit-exact Whisper frontend.
    """
    spec = np.load(logmag_256).astype(np.float32) if isinstance(logmag_256, Path) else logmag_256.astype(np.float32)
    spec = np.squeeze(spec)
    if spec.ndim != 2:
        raise ValueError(f"Expected [256,T] logmag, got {spec.shape}")
    if spec.shape[0] > 256:
        spec = spec[:256]
    if spec.shape[0] < 256:
        floor = float(np.min(spec)) if spec.size else -20.0
        spec = np.pad(spec, ((0, 256 - spec.shape[0]), (0, 0)), mode="constant", constant_values=floor)

    if last_bin_mode == "repeat":
        last = spec[-1:, :]
    else:
        last = np.full((1, spec.shape[1]), float(np.min(spec)), dtype=np.float32)
    spec_257 = np.concatenate([spec, last], axis=0)

    mag = np.maximum(np.exp(spec_257), 0.0)
    power = mag ** 2

    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=80,
        fmin=fmin,
        fmax=fmax or sr / 2,
        norm="slaney",
        htk=False,
    ).astype(np.float32)
    mel = np.maximum(mel_basis @ power, 1e-10)
    logmel = np.log10(mel)
    logmel = np.maximum(logmel, np.max(logmel) - 8.0)
    logmel = (logmel + 4.0) / 4.0

    # Convert old 8 ms STFT frame count to Whisper's 10 ms frame grid.
    target_t = int(round(logmel.shape[1] * hop_length / target_hop_length))
    target_t = max(1, min(target_t, max_t))
    x = torch.from_numpy(logmel[None, :, :])
    x = F.interpolate(x, size=target_t, mode="linear", align_corners=False)[0].numpy().astype(np.float32)
    return x


class AudioVsDirectDataset(Dataset):
    def __init__(
        self,
        manifest: Path,
        clean_root: Path,
        reverb_root: Path,
        pred_mel_root: Path,
        pred_spec_root: Path,
        tokenizer: WhisperTokenizer,
        channel: str,
        limit: Optional[int],
        direct_last_bin_mode: str,
    ):
        rows = read_manifest(manifest, limit)
        clean_idx = index_npy_files(clean_root)
        reverb_idx = index_npy_files(reverb_root, channel=channel)
        pred_mel_idx = index_npy_files(pred_mel_root, channel=channel, require_center_pred=True)
        pred_spec_idx = index_npy_files(pred_spec_root, channel=channel, require_center_pred=True)

        self.items = []
        self.tok = tokenizer
        self.direct_last_bin_mode = direct_last_bin_mode

        for row in rows:
            uid = utterance_id(row.get("stem") or row.get("pred") or row.get("text", ""))
            clean_path = find_path(row, clean_idx)
            reverb_path = find_path(row, reverb_idx)
            pred_mel_path = find_path(row, pred_mel_idx)
            pred_spec_path = find_path(row, pred_spec_idx)
            text_path = Path(row["text"]) if "text" in row else None
            if clean_path and reverb_path and pred_mel_path and pred_spec_path and text_path and text_path.exists():
                text = clean_text(text_path.read_text(encoding="utf-8").strip())
                true_t = infer_true_t(str(pred_mel_path), str(pred_spec_path), str(clean_path), str(reverb_path), row.get("pred", ""))
                self.items.append((uid, clean_path, reverb_path, pred_mel_path, pred_spec_path, true_t, text))

        if not self.items:
            raise ValueError(
                "No matched clean/reverb/pred-mel/pred-spec/text examples found. "
                "Check roots, manifest, and channel."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        uid, clean_path, reverb_path, pred_mel_path, pred_spec_path, true_t, text = self.items[idx]
        clean = load_whisper_features(clean_path, true_t)
        reverb = load_whisper_features(reverb_path, true_t)
        pred_audio = load_whisper_features(pred_mel_path, true_t)
        pred_direct = whisper_logmel_from_logmag512(pred_spec_path, last_bin_mode=self.direct_last_bin_mode)
        return {
            "uid": uid,
            "clean": torch.from_numpy(clean),
            "reverb": torch.from_numpy(reverb),
            "pred_audio": torch.from_numpy(pred_audio),
            "pred_direct_stft": torch.from_numpy(pred_direct),
            "label_ids": torch.tensor(self.tok(text).input_ids, dtype=torch.long),
        }


class Collator:
    def __init__(self, tokenizer: WhisperTokenizer, max_t: int = 3000):
        self.pad_id = tokenizer.pad_token_id
        self.max_t = max_t

    def _pad_or_crop(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.max_t:
            return x[:, : self.max_t]
        return F.pad(x, (0, self.max_t - x.shape[-1]))

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        out = {
            key: torch.stack([self._pad_or_crop(item[key]) for item in batch])
            for key in ["clean", "reverb", "pred_audio", "pred_direct_stft"]
        }
        labels = pad_sequence([item["label_ids"] for item in batch], batch_first=True, padding_value=self.pad_id)
        labels_for_loss = labels.clone()
        for i, item in enumerate(batch):
            labels_for_loss[i, len(item["label_ids"]) :] = -100
        out["labels"] = labels_for_loss
        out["labels_text_ids"] = labels
        return out


@torch.no_grad()
def evaluate_variant(model, loader, processor, tokenizer, device, variant: str) -> Dict[str, object]:
    total_loss = 0.0
    refs: List[str] = []
    hyps: List[str] = []
    for batch in tqdm(loader, desc=f"Evaluating {variant}"):
        features = batch[variant].to(device).float()
        labels = batch["labels"].to(device)
        out = model(input_features=features, labels=labels, use_cache=False)
        total_loss += float(out.loss.item())
        gen_ids = model.generate(input_features=features, num_beams=5, early_stopping=True, repetition_penalty=1.2)
        hyps.extend(clean_text(t) for t in processor.batch_decode(gen_ids, skip_special_tokens=True))
        refs.extend(clean_text(t) for t in tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True))
    val_wer = wer(refs, hyps)
    return {
        "variant": variant,
        "val_loss": total_loss / max(len(loader), 1),
        "val_wer": val_wer,
        "val_wer_pct": f"{val_wer:.3%}",
    }


def write_results(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "val_loss", "val_wer", "val_wer_pct"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pred audio-path mel vs direct STFT-to-mel diagnostic.")
    parser.add_argument("--manifest", type=Path, default=Path("/storage/tal/thesis/condwhisper_manifests/val_cpwidth.jsonl"))
    parser.add_argument("--clean-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/clean_melspec/val"))
    parser.add_argument("--reverb-root", type=Path, default=Path("/storage/tal/thesis/DataBase_BIUREV/reverb_melspec/val"))
    parser.add_argument("--pred-mel-root", type=Path, required=True)
    parser.add_argument("--pred-spec-root", type=Path, required=True)
    parser.add_argument("--channel", type=str, default="ch1")
    parser.add_argument("--direct-last-bin-mode", choices=["floor", "repeat"], default="floor")
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-t", type=int, default=3000)
    parser.add_argument("--variants", nargs="+", default=["clean", "reverb", "pred_audio", "pred_direct_stft"])
    parser.add_argument("--out-csv", type=Path, default=Path("/storage/tal/thesis/script_res/pred_audio_vs_direct_stft_whisper.csv"))
    args = parser.parse_args()

    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language=args.language, task=args.task)
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task=args.task)
    ds = AudioVsDirectDataset(
        manifest=args.manifest,
        clean_root=args.clean_root,
        reverb_root=args.reverb_root,
        pred_mel_root=args.pred_mel_root,
        pred_spec_root=args.pred_spec_root,
        tokenizer=tokenizer,
        channel=args.channel,
        limit=args.limit,
        direct_last_bin_mode=args.direct_last_bin_mode,
    )
    print(f"[INFO] matched examples: {len(ds)}")
    print("[NOTE] pred_audio = saved mel from iSTFT waveform path.")
    print("[NOTE] pred_direct_stft = diagnostic log-STFT magnitude -> mel path, no phase/iSTFT.")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=Collator(tokenizer, max_t=args.max_t))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    rows: List[Dict[str, object]] = []
    valid = {"clean", "reverb", "pred_audio", "pred_direct_stft"}
    for variant in args.variants:
        if variant not in valid:
            raise ValueError(f"Unknown variant {variant}; choose from {sorted(valid)}")
        row = evaluate_variant(model, loader, processor, tokenizer, device, variant)
        rows.append(row)
        print(row)
        write_results(args.out_csv, rows)
    print(f"[SAVE] {args.out_csv}")


if __name__ == "__main__":
    main()
