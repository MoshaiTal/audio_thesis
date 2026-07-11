#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import cer, wer
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

from CopiedFromYam.ASR.newCondWhisper.condwhisper_film import (
    CondWhisperVNextCollator,
    CondWhisperVNextDataset,
    clean_text,
    count_trainable_parameters,
    freeze_all,
    inv_softplus,
    is_better,
    make_time_mask,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


@torch.no_grad()
def evaluate_baseline(whisper, processor, tokenizer, dataloader, device, source_key: str, show_progress: bool, num_beams: int):
    whisper.eval()
    refs, hyps, losses = [], [], []
    for batch in tqdm(dataloader, desc=f"Eval baseline {source_key}", disable=not show_progress):
        x = batch[source_key].to(device).float()
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = whisper(input_features=x, labels=labels, decoder_attention_mask=decoder_attention_mask, use_cache=False, return_dict=True)
        gen_ids = whisper.generate(
            input_features=x,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels_text_ids, skip_special_tokens=True)
        refs.extend([clean_text(t) for t in ref_texts])
        hyps.extend([clean_text(t) for t in pred_texts])
        losses.append(float(out.loss.item()))
    return {"loss": float(np.mean(losses)), "wer": float(wer(refs, hyps)), "cer": float(cer(refs, hyps))}


def sequence_logit_kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    target_len = min(student_logits.shape[1], teacher_logits.shape[1], labels.shape[1])
    student_logits = student_logits[:, :target_len, :]
    teacher_logits = teacher_logits[:, :target_len, :]
    labels = labels[:, :target_len]
    valid = labels.ne(-100)
    if not bool(valid.any()):
        return student_logits.new_tensor(0.0)

    scaled_student = student_logits[valid] / temperature
    scaled_teacher = teacher_logits[valid] / temperature
    return F.kl_div(
        F.log_softmax(scaled_student, dim=-1),
        F.softmax(scaled_teacher, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def encoder_hidden_kd_loss(
    student_hidden_states,
    teacher_hidden_states,
    lengths: torch.Tensor,
    layer_indices: List[int],
    normalize: bool = False,
) -> torch.Tensor:
    if not layer_indices:
        return student_hidden_states[-1].new_tensor(0.0)

    losses = []
    for layer_idx in layer_indices:
        # HF encoder hidden states include the convolutional embedding output at
        # index 0, then one entry after each encoder layer.
        state_idx = min(layer_idx + 1, len(student_hidden_states) - 1, len(teacher_hidden_states) - 1)
        student = student_hidden_states[state_idx]
        teacher = teacher_hidden_states[state_idx].detach()
        target_len = min(student.shape[1], teacher.shape[1])
        student = student[:, :target_len, :]
        teacher = teacher[:, :target_len, :]
        if normalize:
            student = F.layer_norm(student, student.shape[-1:])
            teacher = F.layer_norm(teacher, teacher.shape[-1:])
        enc_lengths = ((lengths + 1) // 2).clamp(max=target_len)
        mask = make_time_mask(enc_lengths, target_len).to(student.device).float()
        sqerr = (student - teacher).pow(2).mean(dim=-1)
        losses.append((sqerr * mask).sum() / mask.sum().clamp(min=1.0))
    return torch.stack(losses).mean()


class EnhancedFeatureConditioner(nn.Module):
    """Builds a time-local conditioning sequence from the enhanced/side mel.

    This follows the Interspeech 2022 setup more closely than the previous
    oracle script: the condition generator is driven primarily by enhanced
    features, while the original input remains Whisper's acoustic input.
    """

    def __init__(self, n_mels: int, d_model: int, hidden: int = 192, out_scale: float = 0.10, mode: str = "enhanced"):
        super().__init__()
        if mode not in {"enhanced", "enhanced_diff", "oracle_error"}:
            raise ValueError(f"Unsupported conditioner mode: {mode}")
        self.mode = mode
        self.out_scale = out_scale
        if mode == "enhanced":
            in_ch = n_mels
        elif mode == "enhanced_diff":
            in_ch = n_mels * 3
        else:
            in_ch = n_mels * 2
        self.local_net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=1, bias=False),
        )
        self.global_net = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        nn.init.normal_(self.local_net[-1].weight, mean=0.0, std=7e-4)
        nn.init.normal_(self.global_net[-1].weight, mean=0.0, std=7e-4)

    def forward(self, input_mel: torch.Tensor, cond_mel: torch.Tensor, lengths: torch.Tensor, target_len: int):
        if self.mode == "enhanced":
            x = cond_mel
        elif self.mode == "enhanced_diff":
            diff = cond_mel - input_mel
            x = torch.cat([cond_mel, diff, diff.abs()], dim=1)
        else:
            diff = cond_mel - input_mel
            x = torch.cat([diff, diff.abs()], dim=1)

        local = self.local_net(x)
        if local.shape[-1] != target_len:
            local = F.interpolate(local, size=target_len, mode="linear", align_corners=False)

        enc_lengths = ((lengths + 1) // 2).clamp(max=target_len)
        mask = make_time_mask(enc_lengths, target_len).float()
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (local * mask[:, None, :]).sum(dim=-1) / denom
        glob = self.global_net(pooled)
        return self.out_scale * torch.tanh(local.transpose(1, 2)), self.out_scale * torch.tanh(glob)


class PreLayerFiLM(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 128, film_scale: float = 0.10, init_residual_scale: float = 0.002):
        super().__init__()
        self.film_scale = film_scale
        self.norm = nn.LayerNorm(d_model)
        self.gamma = nn.Sequential(nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
        self.beta = nn.Sequential(nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
        nn.init.normal_(self.gamma[-1].weight, mean=0.0, std=7e-4)
        nn.init.zeros_(self.gamma[-1].bias)
        nn.init.normal_(self.beta[-1].weight, mean=0.0, std=7e-4)
        nn.init.zeros_(self.beta[-1].bias)
        self.log_residual_scale = nn.Parameter(torch.tensor(inv_softplus(init_residual_scale), dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, cond_local: torch.Tensor, cond_global: torch.Tensor):
        cond = cond_local + cond_global[:, None, :]
        z = self.norm(hidden_states)
        gamma = self.film_scale * torch.tanh(self.gamma(cond))
        beta = self.film_scale * torch.tanh(self.beta(cond))
        delta = z * gamma + beta
        scale = F.softplus(self.log_residual_scale)
        out = hidden_states + scale * delta
        return out, {
            "residual_scale": scale,
            "gamma_abs": gamma.abs().mean(),
            "beta_abs": beta.abs().mean(),
            "layer_update_abs": (out - hidden_states).abs().mean(),
        }


class PreLayerSideCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        bottleneck: int = 128,
        num_heads: int = 4,
        attn_dropout: float = 0.0,
        init_residual_scale: float = 0.002,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.cond_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        nn.init.normal_(self.out[-1].weight, mean=0.0, std=7e-4)
        nn.init.zeros_(self.out[-1].bias)
        self.log_residual_scale = nn.Parameter(torch.tensor(inv_softplus(init_residual_scale), dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor, cond_local: torch.Tensor, cond_global: torch.Tensor):
        cond = cond_local + cond_global[:, None, :]
        query = self.query_norm(hidden_states)
        key_value = self.cond_norm(cond)
        attn_out, attn_weights = self.attn(query, key_value, key_value, need_weights=False)
        delta = self.out(attn_out)
        scale = F.softplus(self.log_residual_scale)
        out = hidden_states + scale * delta
        return out, {
            "residual_scale": scale,
            "gamma_abs": attn_out.abs().mean(),
            "beta_abs": delta.abs().mean(),
            "layer_update_abs": (out - hidden_states).abs().mean(),
        }


class HookedEnhancedFiLMWhisper(nn.Module):
    def __init__(
        self,
        whisper: WhisperForConditionalGeneration,
        selected_layers: List[int],
        conditioner_hidden: int = 192,
        conditioner_scale: float = 0.10,
        conditioner_mode: str = "enhanced",
        adapter_bottleneck: int = 128,
        adapter_film_scale: float = 0.10,
        init_residual_scale: float = 0.002,
        fusion_mode: str = "film",
        side_attn_heads: int = 4,
        side_attn_dropout: float = 0.0,
    ):
        super().__init__()
        if fusion_mode not in {"film", "cross_attention"}:
            raise ValueError(f"Unsupported fusion mode: {fusion_mode}")
        self.base = whisper
        self.selected_layers = sorted(selected_layers)
        self.fusion_mode = fusion_mode
        d_model = whisper.model.config.d_model
        n_mels = whisper.model.config.num_mel_bins
        self.conditioner = EnhancedFeatureConditioner(n_mels, d_model, conditioner_hidden, conditioner_scale, conditioner_mode)
        if fusion_mode == "film":
            self.adapters = nn.ModuleDict({
                str(layer_idx): PreLayerFiLM(d_model, adapter_bottleneck, adapter_film_scale, init_residual_scale)
                for layer_idx in self.selected_layers
            })
        else:
            self.adapters = nn.ModuleDict({
                str(layer_idx): PreLayerSideCrossAttention(
                    d_model,
                    adapter_bottleneck,
                    side_attn_heads,
                    side_attn_dropout,
                    init_residual_scale,
                )
                for layer_idx in self.selected_layers
            })
        self._cond_local: Optional[torch.Tensor] = None
        self._cond_global: Optional[torch.Tensor] = None
        self._last_debug: List[Dict] = []
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.selected_layers:
            layer = self.base.model.encoder.layers[idx]

            def hook(module, args, layer_idx=idx):
                if self._cond_local is None or self._cond_global is None:
                    return args
                hidden = args[0]
                new_hidden, dbg = self.adapters[str(layer_idx)](hidden, self._cond_local, self._cond_global)
                self._last_debug.append({"layer": layer_idx, **dbg})
                return (new_hidden, *args[1:])

            self._handles.append(layer.register_forward_pre_hook(hook))

    def _prepare_condition(self, input_features: torch.Tensor, condition_features: torch.Tensor, lengths: torch.Tensor):
        # Whisper's second encoder convolution downsamples by 2.
        target_len = (input_features.shape[-1] + 1) // 2
        self._cond_local, self._cond_global = self.conditioner(input_features, condition_features, lengths, target_len)
        self._last_debug = []

    def _clear_condition(self):
        self._cond_local = None
        self._cond_global = None

    def forward(self, input_features, condition_features, lengths, labels, decoder_attention_mask):
        self._prepare_condition(input_features, condition_features, lengths)
        try:
            out = self.base(
                input_features=input_features,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            debug = self.debug_summary()
        finally:
            self._clear_condition()
        return {"loss": out.loss, "logits": out.logits, "encoder_hidden_states": out.encoder_hidden_states, "debug": debug}

    @torch.no_grad()
    def generate(self, input_features, condition_features, lengths, **kwargs):
        self._prepare_condition(input_features, condition_features, lengths)
        try:
            gen_ids = self.base.generate(input_features=input_features, **kwargs)
            debug = self.debug_summary()
        finally:
            self._clear_condition()
        return gen_ids, debug

    def debug_summary(self):
        if not self._last_debug:
            zero = torch.tensor(0.0, device=next(self.parameters()).device)
            return {"mean_layer_update_abs": zero, "adapter_debug": []}
        return {
            "mean_layer_update_abs": torch.stack([d["layer_update_abs"] for d in self._last_debug]).mean(),
            "adapter_debug": self._last_debug,
        }


@torch.no_grad()
def evaluate_model(model, processor, tokenizer, dataloader, device, input_source: str, condition_source: str, show_progress: bool, num_beams: int):
    model.eval()
    refs, hyps, losses, updates = [], [], [], []
    for batch in tqdm(dataloader, desc="Eval hooked FiLM", disable=not show_progress):
        input_features = batch[input_source].to(device).float()
        condition_features = batch[condition_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = model(input_features, condition_features, lengths, labels, decoder_attention_mask)
        gen_ids, dbg = model.generate(
            input_features,
            condition_features,
            lengths,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels_text_ids, skip_special_tokens=True)
        refs.extend([clean_text(t) for t in ref_texts])
        hyps.extend([clean_text(t) for t in pred_texts])
        losses.append(float(out["loss"].item()))
        updates.append(float(dbg["mean_layer_update_abs"].detach().cpu().item()))
    return {
        "loss": float(np.mean(losses)),
        "wer": float(wer(refs, hyps)),
        "cer": float(cer(refs, hyps)),
        "mean_layer_update_abs": float(np.mean(updates)),
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    input_source: str,
    condition_source: str,
    lambda_update: float,
    show_progress: bool,
    teacher_model=None,
    teacher_source: str = "reverb",
    lambda_logit_kd: float = 0.0,
    lambda_hidden_kd: float = 0.0,
    hidden_kd_layers: Optional[List[int]] = None,
    hidden_kd_normalize: bool = False,
    lambda_update_ceiling: float = 0.0,
    update_ceiling: float = 0.0,
    kd_temperature: float = 2.0,
):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    total = asr = logit_kd_total = hidden_kd_total = update_reg_total = update_ceiling_total = 0.0
    n = 0
    last_update = 0.0
    for batch in tqdm(dataloader, desc="Training hooked FiLM", disable=not show_progress):
        input_features = batch[input_source].to(device).float()
        condition_features = batch[condition_source].to(device).float()
        teacher_features = batch[teacher_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = model(input_features, condition_features, lengths, labels, decoder_attention_mask)
        update_reg = out["debug"]["mean_layer_update_abs"]
        logit_kd_loss = out["loss"].new_tensor(0.0)
        hidden_kd_loss = out["loss"].new_tensor(0.0)
        need_teacher = teacher_model is not None and (lambda_logit_kd > 0.0 or lambda_hidden_kd > 0.0)
        if need_teacher:
            with torch.no_grad():
                teacher_out = teacher_model(
                    input_features=teacher_features,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            if lambda_logit_kd > 0.0:
                logit_kd_loss = sequence_logit_kd_loss(out["logits"], teacher_out.logits, labels, kd_temperature)
            if lambda_hidden_kd > 0.0:
                hidden_kd_loss = encoder_hidden_kd_loss(
                    out["encoder_hidden_states"],
                    teacher_out.encoder_hidden_states,
                    lengths,
                    hidden_kd_layers or [],
                    normalize=hidden_kd_normalize,
                )
        update_ceiling_loss = out["loss"].new_tensor(0.0)
        if lambda_update_ceiling > 0.0 and update_ceiling > 0.0:
            update_ceiling_loss = F.relu(update_reg - update_ceiling).pow(2)
        loss = (
            out["loss"]
            + lambda_logit_kd * logit_kd_loss
            + lambda_hidden_kd * hidden_kd_loss
            + lambda_update * update_reg
            + lambda_update_ceiling * update_ceiling_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        asr += float(out["loss"].item())
        logit_kd_total += float(logit_kd_loss.detach().cpu().item())
        hidden_kd_total += float(hidden_kd_loss.detach().cpu().item())
        update_reg_total += float(update_reg.detach().cpu().item())
        update_ceiling_total += float(update_ceiling_loss.detach().cpu().item())
        last_update = float(update_reg.detach().cpu().item())
        n += 1
    return {
        "train_total_loss": total / max(n, 1),
        "train_asr_loss": asr / max(n, 1),
        "train_logit_kd_loss": logit_kd_total / max(n, 1),
        "train_hidden_kd_loss": hidden_kd_total / max(n, 1),
        "train_update_reg": update_reg_total / max(n, 1),
        "train_update_ceiling_loss": update_ceiling_total / max(n, 1),
        "last_update_abs": last_update,
    }


def unfreeze_last_encoder_layers(whisper: WhisperForConditionalGeneration, last_n: int, include_layer_norm: bool = True) -> None:
    if last_n > 0:
        layers = whisper.model.encoder.layers
        start = max(0, len(layers) - last_n)
        for idx in range(start, len(layers)):
            for p in layers[idx].parameters():
                p.requires_grad = True
    if include_layer_norm:
        for p in whisper.model.encoder.layer_norm.parameters():
            p.requires_grad = True


def unfreeze_selected_encoder_layer_norms(whisper: WhisperForConditionalGeneration, selected_layers: List[int]) -> int:
    unfrozen = 0
    layers = whisper.model.encoder.layers
    for idx in selected_layers:
        if idx < 0 or idx >= len(layers):
            continue
        for name, param in layers[idx].named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True
                unfrozen += param.numel()
    return unfrozen


def freeze_adapter_residual_scales(model: nn.Module) -> int:
    frozen = 0
    for module in model.modules():
        if hasattr(module, "log_residual_scale"):
            module.log_residual_scale.requires_grad = False
            frozen += 1
    return frozen


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def main():
    parser = argparse.ArgumentParser(description="Paper-style enhanced-feature FiLM for Whisper using encoder hooks.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="/storage/tal/thesis/hooked_enhanced_film")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lr-decay-after-epoch", type=int, default=0)
    parser.add_argument("--lr-after-decay", type=float, default=0.0)
    parser.add_argument("--freeze-residual-scale-after-epoch", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selected-layers", default="3,6,9,11")
    parser.add_argument("--input-source", choices=["pred", "reverb", "clean"], default="reverb")
    parser.add_argument("--condition-source", choices=["pred", "reverb", "clean"], default="pred")
    parser.add_argument("--conditioner-mode", choices=["enhanced", "enhanced_diff", "oracle_error"], default="enhanced")
    parser.add_argument("--conditioner-hidden", type=int, default=192)
    parser.add_argument("--conditioner-scale", type=float, default=0.10)
    parser.add_argument("--adapter-bottleneck", type=int, default=128)
    parser.add_argument("--adapter-film-scale", type=float, default=0.10)
    parser.add_argument("--init-residual-scale", type=float, default=0.002)
    parser.add_argument("--fusion-mode", choices=["film", "cross_attention"], default="film")
    parser.add_argument("--side-attn-heads", type=int, default=4)
    parser.add_argument("--side-attn-dropout", type=float, default=0.0)
    parser.add_argument("--lambda-update", type=float, default=0.02)
    parser.add_argument("--lambda-update-ceiling", type=float, default=0.0)
    parser.add_argument("--update-ceiling", type=float, default=0.0)
    parser.add_argument("--teacher-source", choices=["pred", "reverb", "clean"], default="reverb")
    parser.add_argument("--lambda-logit-kd", type=float, default=0.0)
    parser.add_argument("--lambda-hidden-kd", type=float, default=0.0)
    parser.add_argument("--hidden-kd-layers", default="", help="Comma-separated encoder layers for hidden KD. Defaults to selected layers.")
    parser.add_argument("--hidden-kd-normalize", action="store_true")
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--unfreeze-last-encoder-layers", type=int, default=0)
    parser.add_argument("--unfreeze-encoder-layer-norm", action="store_true")
    parser.add_argument("--unfreeze-selected-layer-norms", action="store_true")
    parser.add_argument("--eval-num-beams", type=int, default=1)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", default="", help="Optional model checkpoint to load before eval-only or final test evaluation.")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_layers = [int(x) for x in args.selected_layers.split(",") if x.strip()]
    hidden_kd_layers = [int(x) for x in args.hidden_kd_layers.split(",") if x.strip()] if args.hidden_kd_layers else selected_layers

    tokenizer = WhisperTokenizer.from_pretrained(args.model_name)
    processor = WhisperProcessor.from_pretrained(args.model_name)
    collator = CondWhisperVNextCollator(pad_token_id=tokenizer.pad_token_id, max_T=3000)
    train_ds = CondWhisperVNextDataset(args.train_manifest, tokenizer)
    val_ds = CondWhisperVNextDataset(args.val_manifest, tokenizer)
    test_ds = CondWhisperVNextDataset(args.test_manifest, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collator)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    for m in [baseline_whisper, whisper]:
        m.config.forced_decoder_ids = forced_decoder_ids
        m.generation_config.forced_decoder_ids = forced_decoder_ids
    freeze_all(baseline_whisper)
    freeze_all(whisper)
    unfreeze_last_encoder_layers(
        whisper,
        last_n=args.unfreeze_last_encoder_layers,
        include_layer_norm=args.unfreeze_encoder_layer_norm or args.unfreeze_last_encoder_layers > 0,
    )
    selected_layer_norm_params = 0
    if args.unfreeze_selected_layer_norms:
        selected_layer_norm_params = unfreeze_selected_encoder_layer_norms(whisper, selected_layers)

    model = HookedEnhancedFiLMWhisper(
        whisper=whisper,
        selected_layers=selected_layers,
        conditioner_hidden=args.conditioner_hidden,
        conditioner_scale=args.conditioner_scale,
        conditioner_mode=args.conditioner_mode,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_film_scale=args.adapter_film_scale,
        init_residual_scale=args.init_residual_scale,
        fusion_mode=args.fusion_mode,
        side_attn_heads=args.side_attn_heads,
        side_attn_dropout=args.side_attn_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[LOAD] Checkpoint loaded from {checkpoint_path}", flush=True)

    print(json.dumps({
        "trainable_params": count_trainable_parameters(model),
        "input_source": args.input_source,
        "condition_source": args.condition_source,
        "selected_layers": selected_layers,
        "conditioner_mode": args.conditioner_mode,
        "fusion_mode": args.fusion_mode,
        "side_attn_heads": args.side_attn_heads,
        "lr_decay_after_epoch": args.lr_decay_after_epoch,
        "lr_after_decay": args.lr_after_decay,
        "freeze_residual_scale_after_epoch": args.freeze_residual_scale_after_epoch,
        "lambda_update_ceiling": args.lambda_update_ceiling,
        "update_ceiling": args.update_ceiling,
        "teacher_source": args.teacher_source,
        "lambda_logit_kd": args.lambda_logit_kd,
        "lambda_hidden_kd": args.lambda_hidden_kd,
        "hidden_kd_layers": hidden_kd_layers,
        "hidden_kd_normalize": args.hidden_kd_normalize,
        "unfreeze_last_encoder_layers": args.unfreeze_last_encoder_layers,
        "unfreeze_encoder_layer_norm": args.unfreeze_encoder_layer_norm,
        "unfreeze_selected_layer_norms": args.unfreeze_selected_layer_norms,
        "selected_layer_norm_trainable_params": selected_layer_norm_params,
    }, separators=(",", ":")), flush=True)

    baseline_input = evaluate_baseline(baseline_whisper, processor, tokenizer, val_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
    init_val = evaluate_model(model, processor, tokenizer, val_dl, device, args.input_source, args.condition_source, args.show_progress, args.eval_num_beams)
    initial = {"baseline_input_val": baseline_input, "init_val": init_val, "args": vars(args)}
    (out_dir / "initial_summary.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(json.dumps({"initial_delta_vs_input": init_val["wer"] - baseline_input["wer"], "init_val": init_val}, separators=(",", ":")), flush=True)
    if args.eval_only:
        test_input = evaluate_baseline(baseline_whisper, processor, tokenizer, test_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
        test_model = evaluate_model(model, processor, tokenizer, test_dl, device, args.input_source, args.condition_source, args.show_progress, args.eval_num_beams)
        eval_summary = {
            "initial": initial,
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
            "test_input_baseline": test_input,
            "test_hooked_film": test_model,
            "test_delta_vs_input": float(test_model["wer"] - test_input["wer"]),
        }
        (out_dir / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
        print(json.dumps(eval_summary, separators=(",", ":")), flush=True)
        return

    history = []
    best = None
    best_path = out_dir / "best_model.pt"
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        if args.lr_decay_after_epoch > 0 and epoch == args.lr_decay_after_epoch + 1 and args.lr_after_decay > 0.0:
            set_optimizer_lr(optimizer, args.lr_after_decay)
            print(json.dumps({"epoch": epoch, "lr_changed_to": args.lr_after_decay}, separators=(",", ":")), flush=True)
        if args.freeze_residual_scale_after_epoch > 0 and epoch == args.freeze_residual_scale_after_epoch + 1:
            frozen = freeze_adapter_residual_scales(model)
            print(json.dumps({"epoch": epoch, "frozen_residual_scale_params": frozen}, separators=(",", ":")), flush=True)

        train_metrics = train_one_epoch(
            model,
            train_dl,
            optimizer,
            device,
            args.input_source,
            args.condition_source,
            args.lambda_update,
            args.show_progress,
            teacher_model=baseline_whisper,
            teacher_source=args.teacher_source,
            lambda_logit_kd=args.lambda_logit_kd,
            lambda_hidden_kd=args.lambda_hidden_kd,
            hidden_kd_layers=hidden_kd_layers,
            hidden_kd_normalize=args.hidden_kd_normalize,
            lambda_update_ceiling=args.lambda_update_ceiling,
            update_ceiling=args.update_ceiling,
            kd_temperature=args.kd_temperature,
        )
        val_metrics = evaluate_model(model, processor, tokenizer, val_dl, device, args.input_source, args.condition_source, args.show_progress, args.eval_num_beams)
        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "delta_vs_input_baseline_wer": float(val_metrics["wer"] - baseline_input["wer"]),
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps({
            "epoch": epoch,
            "val_wer": f"{100.0 * val_metrics['wer']:.3f}%",
            "delta_vs_input": f"{100.0 * row['delta_vs_input_baseline_wer']:+.3f}pp",
            "val_loss": round(val_metrics["loss"], 4),
            "train_asr_loss": round(train_metrics["train_asr_loss"], 4),
            "train_logit_kd_loss": round(train_metrics["train_logit_kd_loss"], 4),
            "train_hidden_kd_loss": round(train_metrics["train_hidden_kd_loss"], 4),
            "train_update_ceiling_loss": round(train_metrics["train_update_ceiling_loss"], 6),
            "update_abs": round(val_metrics["mean_layer_update_abs"], 5),
        }, separators=(",", ":")), flush=True)
        candidate = {"wer": val_metrics["wer"], "cer": val_metrics["cer"], "loss": val_metrics["loss"]}
        if is_better(candidate, best):
            best = candidate
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "best_metrics": best, "args": vars(args)}, best_path)
            print(f"[SAVE] Best model saved to {best_path}", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_input = evaluate_baseline(baseline_whisper, processor, tokenizer, test_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
    test_model = evaluate_model(model, processor, tokenizer, test_dl, device, args.input_source, args.condition_source, args.show_progress, args.eval_num_beams)
    final = {
        "initial": initial,
        "best_val_metrics": best,
        "test_input_baseline": test_input,
        "test_hooked_film": test_model,
        "test_delta_vs_input": float(test_model["wer"] - test_input["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
