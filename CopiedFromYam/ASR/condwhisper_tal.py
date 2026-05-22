from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    GenerationMixin,
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.whisper.modeling_whisper import BaseModelOutput

from CopiedFromYam.ASR.utils import build_paired_file_rows
from CopiedFromYam.ASR.newCondWhisper.cpcondwhisper_blocks import CPCondWhisperModel

# ============================================================
# Helpers
# ============================================================
LEN_RE = re.compile(r"\[len=(\d+)\]")
TBINS_RE = re.compile(r"actual_tbins=(\d+)")

def infer_true_T(path_str: str) -> int:
    m = LEN_RE.search(path_str)
    if m:
        return int(m.group(1))
    m = TBINS_RE.search(path_str)
    if m:
        return int(m.group(1)) + 1
    raise ValueError(f"Could not infer true length from path: {path_str}")


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower())


def create_manifest_rows(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


# ============================================================
# Dataset / Collator
# ============================================================
class CPMelDataset(Dataset):
    """Loads a mel input (pred/clean/etc.) plus CP width and transcript targets."""

    def __init__(
        self,
        manifest: str | Path,
        tokenizer: WhisperTokenizer,
        input_key: str = "pred",
        clean_targets: bool = True,
        normalize_cp: bool = True,
        log_cp: bool = True,
    ):
        self.rows = [json.loads(l) for l in Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tok = tokenizer
        self.input_key = input_key
        self.clean_targets = clean_targets
        self.normalize_cp = normalize_cp
        self.log_cp = log_cp

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        true_T = infer_true_T(r[self.input_key])

        pred = np.load(r[self.input_key]).astype(np.float32)[:, :true_T]
        lower = np.load(r["lower"]).astype(np.float32)[:, :true_T]
        upper = np.load(r["upper"]).astype(np.float32)[:, :true_T]
        width = np.abs(upper - lower).astype(np.float32)

        if self.log_cp:
            width = np.log1p(width)
        if self.normalize_cp:
            mu = float(width.mean())
            sigma = float(width.std())
            width = (width - mu) / max(sigma, 1e-5)

        with open(r["text"], "r", encoding="utf-8") as f:
            text = f.read().strip()
        target_text = clean_text(text) if self.clean_targets else text

        label_ids = self.tok(target_text).input_ids
        return {
            "pred": torch.from_numpy(pred),
            "cp_width": torch.from_numpy(width),
            "lower": torch.from_numpy(lower),
            "upper": torch.from_numpy(upper),
            "length": torch.tensor(true_T, dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "text": text,
            "target_text": target_text,
            "stem": r.get("stem", Path(r["pred"]).stem),
        }


class CPMelCollator:
    def __init__(self, tokenizer: WhisperTokenizer, max_T: int = 3000):
        self.pad_id = tokenizer.pad_token_id
        self.max_T = max_T

    def __call__(self, batch: List[Dict]) -> Dict:
        pred = torch.stack([F.pad(item["pred"], (0, self.max_T - item["pred"].shape[-1])) for item in batch])
        width = torch.stack([F.pad(item["cp_width"], (0, self.max_T - item["cp_width"].shape[-1])) for item in batch])
        lower = torch.stack([F.pad(item["lower"], (0, self.max_T - item["lower"].shape[-1])) for item in batch])
        upper = torch.stack([F.pad(item["upper"], (0, self.max_T - item["upper"].shape[-1])) for item in batch])
        lengths = torch.tensor([int(item["length"].item()) for item in batch], dtype=torch.long)

        label_ids = [item["label_ids"] for item in batch]
        lengths = [len(x) for x in label_ids]
        labels = pad_sequence(label_ids, batch_first=True, padding_value=self.pad_id)
        decoder_attention_mask = torch.zeros_like(labels)
        for i, L in enumerate(lengths):
            decoder_attention_mask[i, :L] = 1

        labels_for_loss = labels.clone()
        for i, L in enumerate(lengths):
            if L < labels_for_loss.shape[1]:
                labels_for_loss[i, L:] = -100

        return {
            "input_features": pred,
            "pred": pred,
            "cp_width": width,
            "lower": lower,
            "upper": upper,
            "lengths": torch.tensor([int(item["length"].item()) for item in batch], dtype=torch.long),
            "labels": labels_for_loss,
            "labels_text_ids": labels,
            "decoder_attention_mask": decoder_attention_mask,
            "texts": [item["text"] for item in batch],
            "target_texts": [item["target_text"] for item in batch],
            "stems": [item["stem"] for item in batch],
        }


# ============================================================
# Model: top-half multi-layer FiLM on the Whisper encoder
# ============================================================
class MultiLayerFiLMEncoder(nn.Module):
    """
    Minimal-change version:
    - keeps Whisper's encoder blocks intact
    - injects CP information in several encoder layers instead of only once at the end
    - uses a residual FiLM gate so the model starts exactly at the pretrained behavior
    """

    def __init__(
        self,
        config: WhisperConfig,
        conditioned_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        self.config = config
        self.inner = WhisperEncoder(config)
        self.conv1 = self.inner.conv1
        self.conv2 = self.inner.conv2
        self.layers = self.inner.layers
        self.d_model = config.d_model

        n_layers = config.num_hidden_layers
        self.conditioned_layers = conditioned_layers or list(range(n_layers // 2, n_layers))
        self.conditioned_layers_set = set(self.conditioned_layers)

        # CP encoder: width map -> sequence embedding aligned with encoder time steps.
        self.cp_conv1 = nn.Conv1d(self.inner.num_mel_bins, self.d_model, kernel_size=3, padding=1)
        self.cp_conv2 = nn.Conv1d(self.d_model, self.d_model, kernel_size=3, stride=2, padding=1)
        # Keep the CP pathway close to neutral at startup so the model begins near vanilla Whisper.
        nn.init.normal_(self.cp_conv1.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.cp_conv1.bias)
        nn.init.normal_(self.cp_conv2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.cp_conv2.bias)

        self.film_gamma = nn.ModuleDict()
        self.film_beta = nn.ModuleDict()
        self.layer_gates = nn.ParameterDict()
        for layer_idx in self.conditioned_layers:
            self.film_gamma[str(layer_idx)] = nn.Linear(self.d_model, self.d_model)
            self.film_beta[str(layer_idx)] = nn.Linear(self.d_model, self.d_model)
            # Keep the FiLM path near-identity, but not exactly zero.
            # Exact zero init dead-ends training because both the gate and the
            # FiLM heads receive no gradient when modulated == hidden_states.
            nn.init.normal_(self.film_gamma[str(layer_idx)].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.film_gamma[str(layer_idx)].bias)
            nn.init.normal_(self.film_beta[str(layer_idx)].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.film_beta[str(layer_idx)].bias)
            self.layer_gates[str(layer_idx)] = nn.Parameter(torch.tensor(0.01))

    def encode_cp(self, cp_width: torch.Tensor) -> torch.Tensor:
        cp = F.gelu(self.cp_conv1(cp_width))
        cp = F.gelu(self.cp_conv2(cp)).transpose(1, 2)
        return cp

    def apply_film(self, hidden_states: torch.Tensor, cp_embed: torch.Tensor, layer_idx: int) -> torch.Tensor:
        key = str(layer_idx)
        gamma = self.film_gamma[key](cp_embed)
        beta = self.film_beta[key](cp_embed)
        modulated = hidden_states * (1.0 + gamma) + beta
        gate = torch.tanh(self.layer_gates[key])
        return hidden_states + gate * (modulated - hidden_states)

    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ) -> BaseModelOutput:
        pred = input_features[:, 0]
        cp_width = input_features[:, 1]

        expected_seq_length = self.config.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        if pred.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects mel length {expected_seq_length}, found {pred.shape[-1]}."
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = F.gelu(self.conv1(pred))
        hidden_states = F.gelu(self.conv2(hidden_states)).permute(0, 2, 1)
        cp_embed = self.encode_cp(cp_width)

        pos = self.inner.embed_positions.weight
        hidden_states = hidden_states + pos
        hidden_states = F.dropout(hidden_states, p=self.inner.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(hidden_states, attention_mask=None)
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if idx in self.conditioned_layers_set:
                hidden_states = self.apply_film(hidden_states, cp_embed, idx)

            if output_attentions:
                all_attentions = all_attentions + (None,)

        hidden_states = self.inner.layer_norm(hidden_states)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)

        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions)


class MultiLayerFiLMWhisperForConditionalGeneration(WhisperForConditionalGeneration, GenerationMixin):
    def __init__(self, config: WhisperConfig, conditioned_layers: Optional[List[int]] = None):
        super().__init__(config)
        self.model.encoder = MultiLayerFiLMEncoder(config, conditioned_layers=conditioned_layers)

    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(input_features=input_features, attention_mask=attention_mask)

        return super().forward(
            input_features=None,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        conditioned_layers: Optional[List[int]] = None,
        **kwargs,
    ):
        base = WhisperForConditionalGeneration.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(base.config, conditioned_layers=conditioned_layers)
        model.generation_config = copy.deepcopy(base.generation_config)
        model.load_state_dict(base.state_dict(), strict=False)
        model.model.encoder.inner.load_state_dict(base.model.encoder.state_dict())
        missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
        print("Missing after transplant:", [k for k in missing if "encoder" not in k])
        if unexpected:
            print("Unexpected during transplant:", unexpected)
        return model


# ============================================================
# Training / eval helpers
# ============================================================
def build_cond_input(pred: torch.Tensor, cp_width: torch.Tensor) -> torch.Tensor:
    return torch.cat([pred.unsqueeze(1), cp_width.unsqueeze(1)], dim=1)


def masked_temperature_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL(student || teacher) over valid label positions."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    token_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)
    mask = mask.to(dtype=token_kl.dtype)
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0) * (temperature ** 2)


def film_gate_penalty(model) -> torch.Tensor:
    enc = model.model.encoder
    gates = [torch.tanh(p).abs() for p in enc.layer_gates.values()]
    if not gates:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack([g.mean() for g in gates]).mean()


def latent_delta_penalty(base_hidden: torch.Tensor, refined_hidden: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
    mask = enc_mask.unsqueeze(-1).float()
    denom = mask.sum().clamp_min(1.0)
    return (((refined_hidden - base_hidden) ** 2) * mask).sum() / denom


@torch.no_grad()
def evaluate_model(model, dataloader, processor, tokenizer, device, cp_mode: str = "true") -> Tuple[float, float, List[str], List[str]]:
    model.eval()
    total_loss = 0.0
    preds: List[str] = []
    refs: List[str] = []

    for batch in tqdm(dataloader, desc=f"Evaluating ({cp_mode})"):
        pred = batch["pred"].to(device).float()
        lower = batch["lower"].to(device).float()
        upper = batch["upper"].to(device).float()
        input_lengths = batch["lengths"].to(device).long()
        if cp_mode == "zero":
            lower = pred.clone()
            upper = pred.clone()
        elif cp_mode == "shuffled":
            perm = torch.randperm(pred.shape[0], device=pred.device)
            lower = lower[perm]
            upper = upper[perm]

        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(
            pred=pred,
            lower=lower,
            upper=upper,
            input_lengths=input_lengths,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        total_loss += out["loss"].item()

        gen_ids = model.generate(
            pred=pred,
            lower=lower,
            upper=upper,
            input_lengths=input_lengths,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_strs = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True)

        preds += [clean_text(t) for t in pred_strs]
        refs += [clean_text(t) for t in ref_strs]

    return total_loss / max(len(dataloader), 1), wer(refs, preds), refs, preds


@torch.no_grad()
def evaluate_vanilla_on_pred(dataloader, processor, tokenizer, device, model_name: str, language: str, task: str) -> Tuple[float, float]:
    vanilla = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    vanilla.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    vanilla.config.forced_decoder_ids = forced_decoder_ids
    vanilla.generation_config.forced_decoder_ids = forced_decoder_ids

    total_loss = 0.0
    preds: List[str] = []
    refs: List[str] = []

    for batch in tqdm(dataloader, desc="Evaluating vanilla"):
        pred = batch["input_features"].to(device).float()
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = vanilla(
            input_features=pred,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )
        total_loss += out.loss.item()

        gen_ids = vanilla.generate(
            input_features=pred,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_strs = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True)

        preds += [clean_text(t) for t in pred_strs]
        refs += [clean_text(t) for t in ref_strs]

    return total_loss / max(len(dataloader), 1), wer(refs, preds)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    grad_clip: float = 1.0,
    lambda_delta: float = 0.05,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        pred = batch["pred"].to(device).float()
        lower = batch["lower"].to(device).float()
        upper = batch["upper"].to(device).float()
        input_lengths = batch["lengths"].to(device).long()
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(
            pred=pred,
            lower=lower,
            upper=upper,
            input_lengths=input_lengths,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        loss = out["loss"]
        delta_loss = latent_delta_penalty(out["base_hidden"], out["refined_hidden"], out["enc_mask"])
        loss = loss + lambda_delta * delta_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


# ============================================================
# Manifest creation
# ============================================================
def create_manifests_if_missing(
    splits_to_use,
    manifest_dir: Path,
    manifest_name: str,
    rebuild_manifests: bool,
    output_root: Path,
    clean_root: Path,
    reverb_root: Path,
    targets_root: Path,
    wanted_ch: str,
    wanted_alpha: str,
) -> None:
    for split in splits_to_use:
        out_manifest = manifest_dir / f"{split}_{manifest_name}"
        if out_manifest.exists() and not rebuild_manifests:
            print(f"[MANIFEST] Keeping existing {out_manifest}")
            continue

        rows_all = []
        split_output_root = output_root / split
        split_clean_root = clean_root / split
        split_reverb_root = reverb_root / split
        split_targets_root = targets_root / split

        if not split_output_root.exists():
            print(f"[WARN] Missing split output root: {split_output_root}")
            continue

        folders = sorted([p.name for p in split_output_root.iterdir() if p.is_dir()])
        for folder in folders:
            rows = build_paired_file_rows(
                output_folder_path=str(split_output_root / folder),
                clean_folder_path=str(split_clean_root / folder),
                reverb_folder_path=str(split_reverb_root / folder),
                targets_folder_path=str(split_targets_root / folder),
                WANTED_CH=wanted_ch,
                WANTED_ALPHA=wanted_alpha,
            )
            rows_all.extend(rows)

        print(f"[MANIFEST] split={split}: paired {len(rows_all)} rows")
        create_manifest_rows(rows_all, out_manifest)


# ============================================================
# Freezing policy
# ============================================================
def freeze_for_multilayer_film(
    model: MultiLayerFiLMWhisperForConditionalGeneration,
    unfreeze_bottom_k_conditioned_layers: int = 0,
) -> None:
    # Freeze all first.
    for p in model.parameters():
        p.requires_grad = False

    enc = model.model.encoder

    # CP pathway + FiLM heads are always trainable.
    for p in enc.cp_conv1.parameters():
        p.requires_grad = True
    for p in enc.cp_conv2.parameters():
        p.requires_grad = True
    for p in enc.film_gamma.parameters():
        p.requires_grad = True
    for p in enc.film_beta.parameters():
        p.requires_grad = True
    for p in enc.layer_gates.parameters():
        p.requires_grad = True

    # Optionally unfreeze the lowest k conditioned encoder blocks to adapt the representation.
    # This allows the CP/FiLM path to reshape encoder states more effectively.
    if unfreeze_bottom_k_conditioned_layers > 0:
        conditioned_sorted = sorted(enc.conditioned_layers)
        for idx in conditioned_sorted[:unfreeze_bottom_k_conditioned_layers]:
            for p in enc.layers[idx].parameters():
                p.requires_grad = True


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)

    # -------- Data --------
    targets_root = Path("/storage/tal/thesis/DataBase_BIUREV/transcription_matched")
    clean_root = Path("/storage/tal/thesis/DataBase_BIUREV/clean_melspec")
    reverb_root = Path("/storage/tal/thesis/DataBase_BIUREV/reverb_melspec")
    output_root = Path("/storage/tal/thesis/DataBase_BIUREV/dereverb_mel/calibrated_rcps")

    manifest_dir = Path("/storage/tal/thesis/condwhisper_manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_name = "cpwidth.jsonl"
    splits_to_use = ["train", "val", "test"]
    rebuild_manifests = False

    wanted_ch = "ch1"
    wanted_alpha = "0.2"

    # -------- Model / train setup --------
    model_name = "openai/whisper-small"
    language = "en"
    task = "transcribe"
    batch_size = 8
    lr = 3e-5
    n_epochs = 20
    run_cp_ablations_every_epoch = False

    create_manifests_if_missing(
        splits_to_use=splits_to_use,
        manifest_dir=manifest_dir,
        manifest_name=manifest_name,
        rebuild_manifests=rebuild_manifests,
        output_root=output_root,
        clean_root=clean_root,
        reverb_root=reverb_root,
        targets_root=targets_root,
        wanted_ch=wanted_ch,
        wanted_alpha=wanted_alpha,
    )

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CPMelDataset(manifest_dir / f"train_{manifest_name}", tokenizer)
    val_ds = CPMelDataset(manifest_dir / f"val_{manifest_name}", tokenizer)
    val_clean_ds = CPMelDataset(manifest_dir / f"val_{manifest_name}", tokenizer, input_key="clean")
    test_ds = CPMelDataset(manifest_dir / f"test_{manifest_name}", tokenizer)
    collator = CPMelCollator(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model = CPCondWhisperModel(
        model_name=model_name,
        mel_bins=80,
        d_cond=128,
        num_latent_blocks=4,
        num_heads=8,
        freeze_whisper=True,
    ).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.base.config.forced_decoder_ids = forced_decoder_ids
    model.base.generation_config.forced_decoder_ids = forced_decoder_ids

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Using latent Condformer-style CP conditioning")
    print(f"Trainable params: {trainable}/{total}")

    vanilla_val_loss, vanilla_val_wer = evaluate_vanilla_on_pred(
        val_loader, processor, tokenizer, device, model_name, language, task
    )
    print({
        "baseline": "vanilla_pred_only_val",
        "val_loss": vanilla_val_loss,
        "val_wer": f"{vanilla_val_wer:.3%}",
    })

    vanilla_clean_val_loss, vanilla_clean_val_wer = evaluate_vanilla_on_pred(
        val_clean_loader, processor, tokenizer, device, model_name, language, task
    )
    print({
        "baseline": "vanilla_clean_val",
        "val_loss": vanilla_clean_val_loss,
        "val_wer": f"{vanilla_clean_val_wer:.3%}",
    })

    # Epoch-0 evaluation: measure the freshly initialized conditional model before any updates.
    init_val_loss, init_val_wer, _, _ = evaluate_model(model, val_loader, processor, tokenizer, device, cp_mode="true")
    init_zero_val_loss, init_zero_val_wer, _, _ = evaluate_model(model, val_loader, processor, tokenizer, device, cp_mode="zero")
    init_row = {
        "train_loss": None,
        "val_loss": init_val_loss,
        "val_wer": f"{init_val_wer:.3%}",
        "val_zero_cp_loss": init_zero_val_loss,
        "val_zero_cp_wer": f"{init_zero_val_wer:.3%}",
        "epoch": 0,
    }

    if run_cp_ablations_every_epoch:
        _, init_shuf_wer, _, _ = evaluate_model(model, val_loader, processor, tokenizer, device, cp_mode="shuffled")
        init_row["val_shuffled_cp_wer"] = f"{init_shuf_wer:.3%}"

    print(init_row)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    total_steps = n_epochs * max(len(train_loader), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(20, int(0.02 * total_steps)),
        num_training_steps=total_steps,
    )

    best_val_wer = init_val_wer
    epochs_without_improve = 0
    best_path = Path("/storage/tal/thesis/weights/condwhisper_multifilm_toplayers_best.pt")

    torch.save({"model": model.state_dict(), "val_wer": best_val_wer, "epoch": 0}, best_path)
    print(f"[SAVE] Epoch-0 baseline saved to {best_path}")

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_wer, _, _ = evaluate_model(model, val_loader, processor, tokenizer, device, cp_mode="true")

        row = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": f"{val_wer:.3%}",
            "epoch": epoch + 1,
        }

        if run_cp_ablations_every_epoch:
            _, val_zero_wer, _, _ = evaluate_model(model, val_loader, processor, tokenizer, device, cp_mode="zero")
            _, val_shuf_wer, _, _ = evaluate_model(model, val_loader, processor, tokenizer, device, cp_mode="shuffled")
            row["val_zero_cp_wer"] = f"{val_zero_wer:.3%}"
            row["val_shuffled_cp_wer"] = f"{val_shuf_wer:.3%}"

        print(row)

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            epochs_without_improve = 0
            torch.save({"model": model.state_dict(), "val_wer": best_val_wer}, best_path)
            print(f"[SAVE] Best model saved to {best_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= 2:
                print(f"[EARLY STOP] Validation WER has not improved for {epochs_without_improve} epochs; stopping early.")
                break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[LOAD] Loaded best checkpoint from {best_path} (val_wer={ckpt['val_wer']:.3%})")

    test_loss, test_wer, _, _ = evaluate_model(model, test_loader, processor, tokenizer, device, cp_mode="true")
    zero_test_loss, zero_test_wer, _, _ = evaluate_model(model, test_loader, processor, tokenizer, device, cp_mode="zero")
    shuf_test_loss, shuf_test_wer, _, _ = evaluate_model(model, test_loader, processor, tokenizer, device, cp_mode="shuffled")
    print({
        "test_loss": test_loss,
        "test_wer": f"{test_wer:.3%}",
        "test_zero_cp_wer": f"{zero_test_wer:.3%}",
        "test_shuffled_cp_wer": f"{shuf_test_wer:.3%}",
    })

    vanilla_test_loss, vanilla_test_wer = evaluate_vanilla_on_pred(
        test_loader, processor, tokenizer, device, model_name, language, task
    )
    print({
        "vanilla_pred_only_test_loss": vanilla_test_loss,
        "vanilla_pred_only_test_wer": f"{vanilla_test_wer:.3%}",
    })
