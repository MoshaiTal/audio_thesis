#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import cer, wer
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from transformers.modeling_outputs import BaseModelOutput

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


class SideMemoryTokenEncoder(nn.Module):
    """Convert side information into fixed memory tokens for Whisper decoder cross-attention."""

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        num_tokens: int = 16,
        hidden: int = 192,
        num_heads: int = 4,
        dropout: float = 0.05,
        token_scale: float = 0.05,
        mode: str = "enhanced_diff",
    ):
        super().__init__()
        if mode not in {"enhanced", "enhanced_diff", "oracle_error", "residual_error"}:
            raise ValueError(f"Unsupported side mode: {mode}")
        self.num_tokens = num_tokens
        self.token_scale = token_scale
        self.mode = mode
        in_ch = {
            "enhanced": n_mels,
            "enhanced_diff": n_mels * 3,
            "oracle_error": n_mels * 2,
            "residual_error": n_mels * 2,
        }[mode]
        self.local_net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=1, bias=False),
        )
        self.query = nn.Parameter(torch.randn(num_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        nn.init.normal_(self.local_net[-1].weight, mean=0.0, std=7e-4)
        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=7e-4)
        nn.init.zeros_(self.ffn[-1].bias)

    def _features(self, input_features: torch.Tensor, side_features: torch.Tensor) -> torch.Tensor:
        if self.mode == "enhanced":
            return side_features
        diff = side_features - input_features
        if self.mode == "enhanced_diff":
            return torch.cat([side_features, diff, diff.abs()], dim=1)
        if self.mode == "oracle_error":
            return torch.cat([diff, diff.abs()], dim=1)
        residual = input_features - side_features
        return torch.cat([residual, residual.abs()], dim=1)

    def forward(self, input_features: torch.Tensor, side_features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self._features(input_features, side_features)
        local = self.local_net(x).transpose(1, 2)
        keep = make_time_mask(lengths, local.shape[1])
        query = self.query[None, :, :].expand(input_features.shape[0], -1, -1)
        tokens, _ = self.cross_attn(query, local, local, key_padding_mask=~keep)
        tokens = self.norm1(query + tokens)
        tokens = self.norm2(tokens + self.ffn(tokens))
        return self.token_scale * torch.tanh(tokens)


class EncoderMemorySideTokenWhisper(nn.Module):
    def __init__(
        self,
        whisper: WhisperForConditionalGeneration,
        num_side_tokens: int = 16,
        side_hidden: int = 192,
        side_heads: int = 4,
        side_dropout: float = 0.05,
        side_token_scale: float = 0.005,
        side_mode: str = "enhanced_diff",
        placement: str = "tail_delta",
    ):
        super().__init__()
        if placement not in {"tail_delta", "prefix_shift"}:
            raise ValueError(f"Unsupported placement: {placement}")
        self.whisper = whisper
        self.placement = placement
        freeze_all(self.whisper)
        d_model = whisper.model.config.d_model
        n_mels = whisper.model.config.num_mel_bins
        self.side_encoder = SideMemoryTokenEncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_tokens=num_side_tokens,
            hidden=side_hidden,
            num_heads=side_heads,
            dropout=side_dropout,
            token_scale=side_token_scale,
            mode=side_mode,
        )

    def _pack_side_tokens(self, base_hidden: torch.Tensor, side_tokens: torch.Tensor, lengths: torch.Tensor):
        bsz, seq_len, d_model = base_hidden.shape
        enc_lengths = ((lengths + 1) // 2).clamp(max=seq_len)
        rows = []
        for b in range(bsz):
            k = min(side_tokens.shape[1], seq_len)
            if self.placement == "prefix_shift":
                audio_len = min(int(enc_lengths[b].item()), seq_len - k)
                pad_len = seq_len - k - audio_len
                row = torch.cat(
                    [
                        side_tokens[b, :k, :],
                        base_hidden[b, :audio_len, :],
                        base_hidden.new_zeros((pad_len, d_model)),
                    ],
                    dim=0,
                )
            else:
                start = min(int(enc_lengths[b].item()), max(seq_len - k, 0))
                row = base_hidden[b].clone()
                row[start : start + k, :] = row[start : start + k, :] + side_tokens[b, :k, :]
            rows.append(row)
        return torch.stack(rows, dim=0)

    def encode_with_side_memory(self, input_features: torch.Tensor, side_features: torch.Tensor, lengths: torch.Tensor):
        with torch.no_grad():
            base = self.whisper.model.encoder(input_features=input_features, return_dict=True).last_hidden_state
        side_tokens = self.side_encoder(input_features, side_features, lengths)
        packed_hidden = self._pack_side_tokens(base.detach(), side_tokens, lengths)
        return BaseModelOutput(last_hidden_state=packed_hidden), side_tokens

    def forward(self, input_features, side_features, lengths, labels, decoder_attention_mask):
        encoder_outputs, side_tokens = self.encode_with_side_memory(input_features, side_features, lengths)
        out = self.whisper(
            encoder_outputs=encoder_outputs,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return {"loss": out.loss, "logits": out.logits, "token_abs": side_tokens.abs().mean()}

    @torch.no_grad()
    def generate(self, input_features, side_features, lengths, **kwargs):
        encoder_outputs, side_tokens = self.encode_with_side_memory(input_features, side_features, lengths)
        gen_ids = self.whisper.generate(
            encoder_outputs=encoder_outputs,
            **kwargs,
        )
        return gen_ids, {"token_abs": side_tokens.abs().mean()}


@torch.no_grad()
def evaluate_model(model, processor, tokenizer, dataloader, device, input_source: str, side_source: str, show_progress: bool, num_beams: int):
    model.eval()
    refs, hyps, losses, token_vals = [], [], [], []
    for batch in tqdm(dataloader, desc="Eval encoder-memory side tokens", disable=not show_progress):
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
        token_vals.append(float(dbg["token_abs"].detach().cpu().item()))
    return {
        "loss": float(np.mean(losses)),
        "wer": float(wer(refs, hyps)),
        "cer": float(cer(refs, hyps)),
        "mean_token_abs": float(np.mean(token_vals)),
    }


def train_one_epoch(model, dataloader, optimizer, device, input_source: str, side_source: str, lambda_token: float, show_progress: bool):
    model.train()
    total = asr = token_reg = 0.0
    n = 0
    for batch in tqdm(dataloader, desc="Training encoder-memory side tokens", disable=not show_progress):
        input_features = batch[input_source].to(device).float()
        side_features = batch[side_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = model(input_features, side_features, lengths, labels, decoder_attention_mask)
        loss = out["loss"] + lambda_token * out["token_abs"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.side_encoder.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item())
        asr += float(out["loss"].item())
        token_reg += float(out["token_abs"].detach().cpu().item())
        n += 1
    return {
        "train_total_loss": total / max(n, 1),
        "train_asr_loss": asr / max(n, 1),
        "train_token_abs": token_reg / max(n, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Frozen Whisper with side memory tokens packed into encoder padding slots.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="/storage/tal/thesis/script_res/encoder_memory_side_tokens")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-source", choices=["pred", "reverb", "clean"], default="reverb")
    parser.add_argument("--side-source", choices=["pred", "reverb", "clean"], default="clean")
    parser.add_argument("--side-mode", choices=["enhanced", "enhanced_diff", "oracle_error", "residual_error"], default="enhanced_diff")
    parser.add_argument("--num-side-tokens", type=int, default=16)
    parser.add_argument("--side-hidden", type=int, default=192)
    parser.add_argument("--side-heads", type=int, default=4)
    parser.add_argument("--side-dropout", type=float, default=0.05)
    parser.add_argument("--side-token-scale", type=float, default=0.005)
    parser.add_argument("--placement", choices=["tail_delta", "prefix_shift"], default="tail_delta")
    parser.add_argument("--lambda-token", type=float, default=0.001)
    parser.add_argument("--eval-num-beams", type=int, default=5)
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
    token_whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    for m in [baseline_whisper, token_whisper]:
        m.config.forced_decoder_ids = forced_decoder_ids
        m.generation_config.forced_decoder_ids = forced_decoder_ids
    freeze_all(baseline_whisper)
    model = EncoderMemorySideTokenWhisper(
        whisper=token_whisper,
        num_side_tokens=args.num_side_tokens,
        side_hidden=args.side_hidden,
        side_heads=args.side_heads,
        side_dropout=args.side_dropout,
        side_token_scale=args.side_token_scale,
        side_mode=args.side_mode,
        placement=args.placement,
    ).to(device)
    optimizer = torch.optim.AdamW(model.side_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(json.dumps({"experiment": "encoder_memory_side_tokens", "trainable_params": count_trainable_parameters(model), "args": vars(args)}, separators=(",", ":")), flush=True)

    baseline_input = evaluate_baseline(baseline_whisper, processor, tokenizer, val_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
    init_val = evaluate_model(model, processor, tokenizer, val_dl, device, args.input_source, args.side_source, args.show_progress, args.eval_num_beams)
    initial = {"baseline_input_val": baseline_input, "init_val": init_val, "args": vars(args)}
    (out_dir / "initial_summary.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(json.dumps({"initial_delta_vs_input": init_val["wer"] - baseline_input["wer"], "init_val": init_val}, separators=(",", ":")), flush=True)

    history: List[Dict] = []
    best = None
    best_path = out_dir / "best_model.pt"
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_dl, optimizer, device, args.input_source, args.side_source, args.lambda_token, args.show_progress)
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
            "token_abs": round(val_metrics["mean_token_abs"], 5),
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
    test_model = evaluate_model(model, processor, tokenizer, test_dl, device, args.input_source, args.side_source, args.show_progress, args.eval_num_beams)
    final = {
        "initial": initial,
        "best_val_metrics": best,
        "test_input_baseline": test_input,
        "test_encoder_memory_side_tokens": test_model,
        "test_delta_vs_input": float(test_model["wer"] - test_input["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
