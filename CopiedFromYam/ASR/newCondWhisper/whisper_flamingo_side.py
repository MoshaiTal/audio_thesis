#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    is_better,
    make_time_mask,
)
from CopiedFromYam.ASR.newCondWhisper.paper_enhanced_film_whisper import evaluate_baseline

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


class SideSequenceEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        d_model: int,
        hidden: int = 256,
        dropout: float = 0.05,
        mode: str = "enhanced_diff",
        bottleneck_tokens: int = 0,
        bottleneck_heads: int = 4,
        time_gate: bool = False,
        time_gate_hidden: int = 128,
        time_gate_init: float = 0.5,
        time_gate_placement: str = "feature",
    ):
        super().__init__()
        if time_gate_placement not in {"feature", "memory"}:
            raise ValueError(f"Unsupported time gate placement: {time_gate_placement}")
        if mode not in {
            "enhanced",
            "enhanced_diff",
            "oracle_error",
            "residual_error",
            "mel_error",
            "mel_abs_error",
            "mel_error_both",
        }:
            raise ValueError(f"Unsupported side mode: {mode}")
        self.mode = mode
        self.bottleneck_tokens = bottleneck_tokens
        self.use_time_gate = time_gate
        self.time_gate_placement = time_gate_placement
        self._last_time_gate_mean: Optional[torch.Tensor] = None
        in_ch = {
            "enhanced": n_mels,
            "enhanced_diff": n_mels * 3,
            "oracle_error": n_mels * 2,
            "residual_error": n_mels * 2,
            "mel_error": n_mels,
            "mel_abs_error": n_mels,
            "mel_error_both": n_mels * 2,
        }[mode]
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, d_model, kernel_size=1, bias=False),
        )
        if time_gate:
            init = min(max(time_gate_init, 1e-4), 1.0 - 1e-4)
            self.time_gate = nn.Sequential(
                nn.Conv1d(n_mels * 4, time_gate_hidden, kernel_size=9, padding=4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(time_gate_hidden, 1, kernel_size=1),
            )
            nn.init.normal_(self.time_gate[-1].weight, mean=0.0, std=1e-3)
            nn.init.constant_(self.time_gate[-1].bias, float(np.log(init / (1.0 - init))))
        if bottleneck_tokens > 0:
            self.query = nn.Parameter(torch.randn(bottleneck_tokens, d_model) * 0.02)
            self.query_attn = nn.MultiheadAttention(d_model, bottleneck_heads, dropout=dropout, batch_first=True)
            self.query_norm = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )
            nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=7e-4)
            nn.init.zeros_(self.ffn[-1].bias)

    def _features(self, input_features: torch.Tensor, side_features: torch.Tensor) -> torch.Tensor:
        if self.mode == "enhanced":
            return side_features
        diff = side_features - input_features
        if self.mode == "enhanced_diff":
            return torch.cat([side_features, diff, diff.abs()], dim=1)
        if self.mode in {"oracle_error", "mel_error_both"}:
            return torch.cat([diff, diff.abs()], dim=1)
        if self.mode == "mel_error":
            return diff
        if self.mode == "mel_abs_error":
            return diff.abs()
        residual = input_features - side_features
        return torch.cat([residual, residual.abs()], dim=1)

    def forward(self, input_features: torch.Tensor, side_features: torch.Tensor, lengths: torch.Tensor):
        x = self._features(input_features, side_features)
        gate = None
        if self.use_time_gate:
            diff = side_features - input_features
            gate_in = torch.cat([input_features, side_features, diff, diff.abs()], dim=1)
            gate = torch.sigmoid(self.time_gate(gate_in))
            if self.time_gate_placement == "feature":
                x = x * gate
            self._last_time_gate_mean = gate.mean()
        else:
            self._last_time_gate_mean = None
        memory = self.net(x)
        target_len = (input_features.shape[-1] + 1) // 2
        if memory.shape[-1] != target_len:
            memory = F.interpolate(memory, size=target_len, mode="linear", align_corners=False)
        if gate is not None and self.time_gate_placement == "memory":
            if gate.shape[-1] != target_len:
                gate = F.interpolate(gate, size=target_len, mode="linear", align_corners=False)
            memory = memory * gate
        memory = memory.transpose(1, 2)
        enc_lengths = ((lengths + 1) // 2).clamp(max=target_len)
        keep = make_time_mask(enc_lengths, target_len)
        memory = memory * keep[:, :, None].float()
        if self.bottleneck_tokens <= 0:
            return memory, ~keep

        query = self.query[None, :, :].expand(input_features.shape[0], -1, -1)
        tokens, _ = self.query_attn(query, memory, memory, key_padding_mask=~keep, need_weights=False)
        tokens = self.query_norm(query + tokens)
        tokens = tokens + self.ffn(tokens)
        token_mask = torch.zeros(
            tokens.shape[:2],
            device=tokens.device,
            dtype=torch.bool,
        )
        return tokens, token_mask


class FlamingoGatedCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.05, mlp_ratio: float = 2.0, use_ffw: bool = True):
        super().__init__()
        self.use_ffw = use_ffw
        self.attn_norm = nn.LayerNorm(d_model)
        self.side_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        hidden = int(d_model * mlp_ratio)
        self.ffw = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.alpha_attn = nn.Parameter(torch.tensor(0.0))
        self.alpha_ffw = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, side_memory: torch.Tensor, side_key_padding_mask: torch.Tensor):
        q = self.attn_norm(x)
        kv = self.side_norm(side_memory)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=side_key_padding_mask, need_weights=False)
        x1 = x + torch.tanh(self.alpha_attn) * attn_out
        if self.use_ffw:
            ffw_out = self.ffw(x1)
            out = x1 + torch.tanh(self.alpha_ffw) * ffw_out
        else:
            ffw_out = torch.zeros_like(x1)
            out = x1
        return out, {
            "attn_gate": torch.tanh(self.alpha_attn),
            "ffw_gate": torch.tanh(self.alpha_ffw),
            "attn_abs": attn_out.abs().mean(),
            "ffw_abs": ffw_out.abs().mean(),
            "layer_update_abs": (out - x).abs().mean(),
        }


def unfreeze_layer_norms(module: nn.Module) -> int:
    total = 0
    for submodule in module.modules():
        if isinstance(submodule, nn.LayerNorm):
            for param in submodule.parameters():
                param.requires_grad = True
                total += param.numel()
    return total


def unfreeze_module_parameters(module: nn.Module) -> int:
    total = 0
    for param in module.parameters():
        param.requires_grad = True
        total += param.numel()
    return total


class WhisperFlamingoSide(nn.Module):
    def __init__(
        self,
        whisper: WhisperForConditionalGeneration,
        selected_decoder_layers: List[int],
        side_mode: str = "enhanced_diff",
        side_hidden: int = 256,
        side_heads: int = 4,
        side_dropout: float = 0.05,
        side_bottleneck_tokens: int = 0,
        side_bottleneck_heads: int = 4,
        side_time_gate: bool = False,
        side_time_gate_hidden: int = 128,
        side_time_gate_init: float = 0.5,
        side_time_gate_placement: str = "feature",
        flamingo_heads: int = 4,
        flamingo_dropout: float = 0.05,
        flamingo_mlp_ratio: float = 2.0,
        use_flamingo_ffw: bool = True,
        max_update_abs: float = 0.0,
        unfreeze_selected_decoder_layer_norms: bool = False,
        unfreeze_decoder_final_layer_norm: bool = False,
        unfreeze_selected_decoder_encoder_attn: bool = False,
    ):
        super().__init__()
        self.whisper = whisper
        self.selected_decoder_layers = sorted(selected_decoder_layers)
        self.max_update_abs = max_update_abs
        freeze_all(self.whisper)
        self.unfrozen_whisper_params = 0
        if unfreeze_selected_decoder_layer_norms:
            for idx in self.selected_decoder_layers:
                self.unfrozen_whisper_params += unfreeze_layer_norms(self.whisper.model.decoder.layers[idx])
        if unfreeze_decoder_final_layer_norm and hasattr(self.whisper.model.decoder, "layer_norm"):
            self.unfrozen_whisper_params += unfreeze_layer_norms(self.whisper.model.decoder.layer_norm)
        if unfreeze_selected_decoder_encoder_attn:
            for idx in self.selected_decoder_layers:
                layer = self.whisper.model.decoder.layers[idx]
                self.unfrozen_whisper_params += unfreeze_module_parameters(layer.encoder_attn)

        d_model = whisper.model.config.d_model
        n_mels = whisper.model.config.num_mel_bins
        self.side_encoder = SideSequenceEncoder(
            n_mels,
            d_model,
            side_hidden,
            side_dropout,
            side_mode,
            bottleneck_tokens=side_bottleneck_tokens,
            bottleneck_heads=side_bottleneck_heads,
            time_gate=side_time_gate,
            time_gate_hidden=side_time_gate_hidden,
            time_gate_init=side_time_gate_init,
            time_gate_placement=side_time_gate_placement,
        )
        self.blocks = nn.ModuleDict({
            str(idx): FlamingoGatedCrossAttentionBlock(d_model, flamingo_heads, flamingo_dropout, flamingo_mlp_ratio, use_flamingo_ffw)
            for idx in self.selected_decoder_layers
        })
        self._side_memory: Optional[torch.Tensor] = None
        self._side_key_padding_mask: Optional[torch.Tensor] = None
        self._last_debug: List[Dict] = []
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.selected_decoder_layers:
            layer = self.whisper.model.decoder.layers[idx]

            def hook(module, args, layer_idx=idx):
                if self._side_memory is None or self._side_key_padding_mask is None:
                    return args
                hidden = args[0]
                side_memory = self._side_memory
                side_mask = self._side_key_padding_mask
                if side_memory.shape[0] != hidden.shape[0]:
                    if hidden.shape[0] < side_memory.shape[0]:
                        side_memory = side_memory[: hidden.shape[0]]
                        side_mask = side_mask[: hidden.shape[0]]
                    elif hidden.shape[0] % side_memory.shape[0] == 0:
                        repeat = hidden.shape[0] // side_memory.shape[0]
                        side_memory = side_memory.repeat_interleave(repeat, dim=0)
                        side_mask = side_mask.repeat_interleave(repeat, dim=0)
                    else:
                        repeat = (hidden.shape[0] + side_memory.shape[0] - 1) // side_memory.shape[0]
                        side_memory = side_memory.repeat_interleave(repeat, dim=0)[: hidden.shape[0]]
                        side_mask = side_mask.repeat_interleave(repeat, dim=0)[: hidden.shape[0]]
                new_hidden, dbg = self.blocks[str(layer_idx)](hidden, side_memory, side_mask)
                if self.max_update_abs > 0.0:
                    delta = new_hidden - hidden
                    update_abs = delta.abs().mean().detach()
                    if float(update_abs.item()) > self.max_update_abs:
                        delta = delta * (self.max_update_abs / update_abs.clamp(min=1e-8))
                        new_hidden = hidden + delta
                        dbg["layer_update_abs"] = (new_hidden - hidden).abs().mean()
                self._last_debug.append({"layer": layer_idx, **dbg})
                return (new_hidden, *args[1:])

            self._handles.append(layer.register_forward_pre_hook(hook))

    def _prepare_side(self, input_features: torch.Tensor, side_features: torch.Tensor, lengths: torch.Tensor):
        self._side_memory, self._side_key_padding_mask = self.side_encoder(input_features, side_features, lengths)
        self._last_debug = []

    def _clear_side(self):
        self._side_memory = None
        self._side_key_padding_mask = None

    def forward(self, input_features, side_features, lengths, labels, decoder_attention_mask):
        self._prepare_side(input_features, side_features, lengths)
        try:
            out = self.whisper(
                input_features=input_features,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            debug = self.debug_summary()
        finally:
            self._clear_side()
        return {"loss": out.loss, "logits": out.logits, "debug": debug}

    @torch.no_grad()
    def generate(self, input_features, side_features, lengths, **kwargs):
        self._prepare_side(input_features, side_features, lengths)
        try:
            gen_ids = self.whisper.generate(input_features=input_features, **kwargs)
            debug = self.debug_summary()
        finally:
            self._clear_side()
        return gen_ids, debug

    def debug_summary(self):
        side_gate = self.side_encoder._last_time_gate_mean
        if side_gate is None:
            side_gate = torch.tensor(1.0, device=next(self.parameters()).device)
        if not self._last_debug:
            zero = torch.tensor(0.0, device=next(self.parameters()).device)
            return {"mean_layer_update_abs": zero, "mean_attn_gate": zero, "mean_ffw_gate": zero, "mean_side_time_gate": side_gate}
        return {
            "mean_layer_update_abs": torch.stack([d["layer_update_abs"] for d in self._last_debug]).mean(),
            "mean_attn_gate": torch.stack([d["attn_gate"] for d in self._last_debug]).mean(),
            "mean_ffw_gate": torch.stack([d["ffw_gate"] for d in self._last_debug]).mean(),
            "mean_side_time_gate": side_gate,
        }


@torch.no_grad()
def evaluate_model(model, processor, tokenizer, dataloader, device, input_source: str, side_source: str, show_progress: bool, num_beams: int):
    model.eval()
    refs, hyps, losses, updates, attn_gates, ffw_gates, side_time_gates = [], [], [], [], [], [], []
    for batch in tqdm(dataloader, desc="Eval Whisper-Flamingo side", disable=not show_progress):
        input_features = batch[input_source].to(device).float()
        side_features = batch[side_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = model(input_features, side_features, lengths, labels, decoder_attention_mask)
        gen_ids, dbg = model.generate(
            input_features,
            side_features,
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
        attn_gates.append(float(dbg["mean_attn_gate"].detach().cpu().item()))
        ffw_gates.append(float(dbg["mean_ffw_gate"].detach().cpu().item()))
        side_time_gates.append(float(dbg["mean_side_time_gate"].detach().cpu().item()))
    return {
        "loss": float(np.mean(losses)),
        "wer": float(wer(refs, hyps)),
        "cer": float(cer(refs, hyps)),
        "mean_layer_update_abs": float(np.mean(updates)),
        "mean_attn_gate": float(np.mean(attn_gates)),
        "mean_ffw_gate": float(np.mean(ffw_gates)),
        "mean_side_time_gate": float(np.mean(side_time_gates)),
    }


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    input_source: str,
    side_source: str,
    lambda_update: float,
    show_progress: bool,
    scheduler=None,
    max_train_steps: int = 0,
    global_step: int = 0,
):
    model.train()
    total = asr = update = 0.0
    n = 0
    for batch in tqdm(dataloader, desc="Training Whisper-Flamingo side", disable=not show_progress):
        if max_train_steps > 0 and global_step >= max_train_steps:
            break
        input_features = batch[input_source].to(device).float()
        side_features = batch[side_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = model(input_features, side_features, lengths, labels, decoder_attention_mask)
        update_reg = out["debug"]["mean_layer_update_abs"]
        loss = out["loss"] + lambda_update * update_reg
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        global_step += 1
        total += float(loss.item())
        asr += float(out["loss"].item())
        update += float(update_reg.detach().cpu().item())
        n += 1
    return {
        "train_total_loss": total / max(n, 1),
        "train_asr_loss": asr / max(n, 1),
        "train_update_reg": update / max(n, 1),
        "global_step": global_step,
    }


def main():
    parser = argparse.ArgumentParser(description="Whisper-Flamingo-style gated decoder cross-attention for side information.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="/storage/tal/thesis/script_res/whisper_flamingo_side")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selected-decoder-layers", default="0,2,4,6")
    parser.add_argument("--input-source", choices=["pred", "reverb", "clean"], default="reverb")
    parser.add_argument("--side-source", choices=["pred", "reverb", "clean"], default="clean")
    parser.add_argument(
        "--side-mode",
        choices=[
            "enhanced",
            "enhanced_diff",
            "oracle_error",
            "residual_error",
            "mel_error",
            "mel_abs_error",
            "mel_error_both",
        ],
        default="enhanced_diff",
    )
    parser.add_argument("--side-hidden", type=int, default=256)
    parser.add_argument("--side-heads", type=int, default=4)
    parser.add_argument("--side-dropout", type=float, default=0.05)
    parser.add_argument("--side-bottleneck-tokens", type=int, default=0)
    parser.add_argument("--side-bottleneck-heads", type=int, default=4)
    parser.add_argument("--side-time-gate", action="store_true")
    parser.add_argument("--side-time-gate-hidden", type=int, default=128)
    parser.add_argument("--side-time-gate-init", type=float, default=0.5)
    parser.add_argument("--side-time-gate-placement", choices=["feature", "memory"], default="feature")
    parser.add_argument("--flamingo-heads", type=int, default=4)
    parser.add_argument("--flamingo-dropout", type=float, default=0.05)
    parser.add_argument("--flamingo-mlp-ratio", type=float, default=2.0)
    parser.add_argument("--disable-flamingo-ffw", action="store_true")
    parser.add_argument("--max-update-abs", type=float, default=0.0)
    parser.add_argument("--unfreeze-selected-decoder-layer-norms", action="store_true")
    parser.add_argument("--unfreeze-decoder-final-layer-norm", action="store_true")
    parser.add_argument("--unfreeze-selected-decoder-encoder-attn", action="store_true")
    parser.add_argument("--lambda-update", type=float, default=0.01)
    parser.add_argument("--eval-num-beams", type=int, default=5)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_layers = [int(x) for x in args.selected_decoder_layers.split(",") if x.strip()]
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
    model = WhisperFlamingoSide(
        whisper=whisper,
        selected_decoder_layers=selected_layers,
        side_mode=args.side_mode,
        side_hidden=args.side_hidden,
        side_heads=args.side_heads,
        side_dropout=args.side_dropout,
        side_bottleneck_tokens=args.side_bottleneck_tokens,
        side_bottleneck_heads=args.side_bottleneck_heads,
        side_time_gate=args.side_time_gate,
        side_time_gate_hidden=args.side_time_gate_hidden,
        side_time_gate_init=args.side_time_gate_init,
        side_time_gate_placement=args.side_time_gate_placement,
        flamingo_heads=args.flamingo_heads,
        flamingo_dropout=args.flamingo_dropout,
        flamingo_mlp_ratio=args.flamingo_mlp_ratio,
        use_flamingo_ffw=not args.disable_flamingo_ffw,
        max_update_abs=args.max_update_abs,
        unfreeze_selected_decoder_layer_norms=args.unfreeze_selected_decoder_layer_norms,
        unfreeze_decoder_final_layer_norm=args.unfreeze_decoder_final_layer_norm,
        unfreeze_selected_decoder_encoder_attn=args.unfreeze_selected_decoder_encoder_attn,
    ).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.warmup_steps > 0:
        def lr_lambda(step: int) -> float:
            return min(1.0, float(step + 1) / float(max(args.warmup_steps, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(json.dumps({
        "experiment": "whisper_flamingo_side",
        "trainable_params": count_trainable_parameters(model),
        "unfrozen_whisper_params": model.unfrozen_whisper_params,
        "selected_decoder_layers": selected_layers,
        "args": vars(args),
    }, separators=(",", ":")), flush=True)

    baseline_input = evaluate_baseline(baseline_whisper, processor, tokenizer, val_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
    init_val = evaluate_model(model, processor, tokenizer, val_dl, device, args.input_source, args.side_source, args.show_progress, args.eval_num_beams)
    initial = {"baseline_input_val": baseline_input, "init_val": init_val, "args": vars(args)}
    (out_dir / "initial_summary.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(json.dumps({"initial_delta_vs_input": init_val["wer"] - baseline_input["wer"], "init_val": init_val}, separators=(",", ":")), flush=True)

    history: List[Dict] = []
    best = None
    best_path = out_dir / "best_model.pt"
    no_improve = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_dl,
            optimizer,
            device,
            args.input_source,
            args.side_source,
            args.lambda_update,
            args.show_progress,
            scheduler=scheduler,
            max_train_steps=args.max_train_steps,
            global_step=global_step,
        )
        global_step = int(train_metrics.pop("global_step"))
        if args.max_train_steps > 0 and global_step == 0:
            break
        val_metrics = evaluate_model(model, processor, tokenizer, val_dl, device, args.input_source, args.side_source, args.show_progress, args.eval_num_beams)
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
            "update_abs": round(val_metrics["mean_layer_update_abs"], 5),
            "attn_gate": round(val_metrics["mean_attn_gate"], 5),
            "ffw_gate": round(val_metrics["mean_ffw_gate"], 5),
            "side_time_gate": round(val_metrics["mean_side_time_gate"], 5),
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
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    test_input = evaluate_baseline(baseline_whisper, processor, tokenizer, test_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
    test_model = evaluate_model(model, processor, tokenizer, test_dl, device, args.input_source, args.side_source, args.show_progress, args.eval_num_beams)
    final = {
        "initial": initial,
        "best_val_metrics": best,
        "test_input_baseline": test_input,
        "test_whisper_flamingo_side": test_model,
        "test_delta_vs_input": float(test_model["wer"] - test_input["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
