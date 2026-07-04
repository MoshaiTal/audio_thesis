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
    evaluate_baseline_pred,
    freeze_all,
    is_better,
    make_time_mask,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


def print_compact_epoch(row: Dict) -> None:
    summary = {
        "epoch": row["epoch"],
        "val_wer": f"{100.0 * row['val_wer']:.3f}%",
        "delta_vs_pred": f"{100.0 * row['delta_vs_pred_baseline_wer']:+.3f}pp",
        "val_loss": round(row["val_loss"], 4),
        "train_loss": round(row["train_loss"], 4),
        "true-zero": f"{100.0 * row['true_minus_zero_wer']:+.3f}pp",
        "true-shuffled": f"{100.0 * row['true_minus_shuffled_wer']:+.3f}pp",
        "prompt_abs": round(row["val_prompt_abs"], 5),
    }
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


class PredReverbPromptEncoder(nn.Module):
    """Generate fixed-length side-information prompt tokens from pred-reverb cues."""

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        num_prompt_tokens: int = 8,
        hidden: int = 128,
        num_heads: int = 4,
        dropout: float = 0.05,
        prompt_scale: float = 0.05,
    ):
        super().__init__()
        self.num_prompt_tokens = num_prompt_tokens
        self.prompt_scale = prompt_scale
        in_ch = n_mels * 4  # pred, reverb, pred-reverb, abs(pred-reverb)
        self.local_net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=1),
        )
        self.query = nn.Parameter(torch.randn(num_prompt_tokens, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.global_mlp = nn.Sequential(
            nn.Linear(8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.normal_(self.local_net[-1].weight, mean=0.0, std=7e-4)
        nn.init.constant_(self.local_net[-1].bias, 0.0)
        nn.init.normal_(self.global_mlp[-1].weight, mean=0.0, std=7e-4)
        nn.init.constant_(self.global_mlp[-1].bias, 0.0)

    def _global_stats(self, pred: torch.Tensor, reverb: torch.Tensor, diff: torch.Tensor, abs_diff: torch.Tensor, lengths: torch.Tensor):
        B, _, T = pred.shape
        mask = make_time_mask(lengths, T).float()
        denom = mask.sum(dim=1).clamp(min=1.0)

        pred_e = pred.mean(dim=1)
        reverb_e = reverb.mean(dim=1)
        diff_e = diff.mean(dim=1)
        abs_e = abs_diff.mean(dim=1)

        def masked_mean(x):
            return (x * mask).sum(dim=1) / denom

        def masked_std(x, mean):
            return torch.sqrt((((x - mean[:, None]) ** 2) * mask).sum(dim=1) / denom + 1e-8)

        mean_abs = masked_mean(abs_e)
        mean_diff = masked_mean(diff_e)
        mean_pred = masked_mean(pred_e)
        mean_reverb = masked_mean(reverb_e)
        std_abs = masked_std(abs_e, mean_abs)
        std_diff = masked_std(diff_e, mean_diff)

        rmse, p90 = [], []
        for b in range(B):
            L = int(lengths[b].item())
            er = diff[b, :, :L].reshape(-1)
            ae = abs_diff[b, :, :L].reshape(-1)
            rmse.append(torch.sqrt(torch.mean(er * er) + 1e-8))
            p90.append(torch.quantile(ae, 0.90))

        return torch.stack(
            [mean_abs, std_abs, mean_diff, std_diff, mean_pred, mean_reverb, torch.stack(rmse), torch.stack(p90)],
            dim=1,
        )

    def forward(self, pred: torch.Tensor, reverb: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        diff = pred - reverb
        abs_diff = torch.abs(diff)
        x = torch.cat([pred, reverb, diff, abs_diff], dim=1)
        local = self.local_net(x).transpose(1, 2)  # [B, T, D]
        keep = make_time_mask(lengths, local.shape[1])

        q = self.query[None, :, :].expand(pred.shape[0], -1, -1)
        prompt, _ = self.cross_attn(q, local, local, key_padding_mask=~keep)
        prompt = self.norm1(q + prompt)
        prompt = self.norm2(prompt + self.ffn(prompt))

        global_bias = self.global_mlp(self._global_stats(pred, reverb, diff, abs_diff, lengths))
        prompt = prompt + global_bias[:, None, :]
        return self.prompt_scale * torch.tanh(prompt)


class PredReverbPromptWhisper(nn.Module):
    """Frozen Whisper with trainable pred-reverb prompt tokens appended to encoder output."""

    def __init__(
        self,
        whisper: WhisperForConditionalGeneration,
        num_prompt_tokens: int = 8,
        prompt_hidden: int = 128,
        prompt_heads: int = 4,
        prompt_scale: float = 0.20,
    ):
        super().__init__()
        self.whisper = whisper
        freeze_all(self.whisper)
        d_model = whisper.model.config.d_model
        n_mels = whisper.model.config.num_mel_bins
        self.prompt_encoder = PredReverbPromptEncoder(
            n_mels=n_mels,
            d_model=d_model,
            num_prompt_tokens=num_prompt_tokens,
            hidden=prompt_hidden,
            num_heads=prompt_heads,
            prompt_scale=prompt_scale,
        )

    def encode_with_prompts(self, pred: torch.Tensor, reverb: torch.Tensor, lengths: torch.Tensor):
        with torch.no_grad():
            enc = self.whisper.model.encoder(input_features=pred, return_dict=True)
            base_hidden = enc.last_hidden_state

        prompts = self.prompt_encoder(pred, reverb, lengths)
        # Whisper generation treats encoder sequences longer than the normal
        # 30-second length as long-form audio and requires timestamp mode. Keep
        # the encoder length fixed and avoid deleting audio tokens by adding the
        # side prompts into the first K encoder positions.
        k = prompts.shape[1]
        hidden = base_hidden.detach().clone()
        hidden[:, :k, :] = hidden[:, :k, :] + prompts
        attention_mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)
        return hidden, attention_mask, prompts

    def forward(self, pred, reverb, lengths, labels, decoder_attention_mask):
        hidden, attention_mask, prompts = self.encode_with_prompts(pred, reverb, lengths)
        out = self.whisper(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return {"loss": out.loss, "logits": out.logits, "prompt_abs": prompts.abs().mean()}

    @torch.no_grad()
    def generate(self, pred, reverb, lengths, **kwargs):
        hidden, attention_mask, prompts = self.encode_with_prompts(pred, reverb, lengths)
        gen_ids = self.whisper.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
            attention_mask=attention_mask,
            **kwargs,
        )
        return gen_ids, {"prompt_abs": prompts.abs().mean()}


def condition_from_batch(batch: Dict, mode: str, device: torch.device) -> torch.Tensor:
    pred = batch["pred"].to(device).float()
    reverb = batch["reverb"].to(device).float()
    if mode == "true":
        return reverb
    if mode == "zero":
        return pred
    if mode == "shuffled":
        return reverb[torch.randperm(reverb.shape[0], device=device)]
    raise ValueError(f"Unsupported condition mode: {mode}")


@torch.no_grad()
def evaluate_prompt_model(model, processor, tokenizer, dataloader, device, condition_mode: str, show_progress: bool):
    model.eval()
    refs, preds, losses, prompt_vals = [], [], [], []
    for batch in tqdm(dataloader, desc=f"Eval prompt mode={condition_mode}", disable=not show_progress):
        pred = batch["pred"].to(device).float()
        reverb = condition_from_batch(batch, condition_mode, device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(pred=pred, reverb=reverb, lengths=lengths, labels=labels, decoder_attention_mask=decoder_attention_mask)
        gen_ids, dbg = model.generate(pred=pred, reverb=reverb, lengths=lengths, num_beams=5, early_stopping=True, repetition_penalty=1.2)

        pred_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels_text_ids, skip_special_tokens=True)
        refs.extend([clean_text(x) for x in ref_texts])
        preds.extend([clean_text(x) for x in pred_texts])
        losses.append(float(out["loss"].item()))
        prompt_vals.append(float(dbg["prompt_abs"].detach().cpu().item()))

    return {
        "loss": float(np.mean(losses)),
        "wer": float(wer(refs, preds)),
        "cer": float(cer(refs, preds)),
        "prompt_abs": float(np.mean(prompt_vals)),
    }


def train_one_epoch(model, dataloader, optimizer, device, show_progress: bool):
    model.train()
    total_loss = 0.0
    prompt_vals = []
    n = 0
    for batch in tqdm(dataloader, desc="Training prompt tokens", disable=not show_progress):
        pred = batch["pred"].to(device).float()
        reverb = batch["reverb"].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(pred=pred, reverb=reverb, lengths=lengths, labels=labels, decoder_attention_mask=decoder_attention_mask)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.prompt_encoder.parameters(), 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        prompt_vals.append(float(out["prompt_abs"].detach().cpu().item()))
        n += 1
    return {"train_loss": total_loss / max(n, 1), "train_prompt_abs": float(np.mean(prompt_vals))}


def main():
    parser = argparse.ArgumentParser(description="Frozen Whisper with pred-reverb side-information prompt tokens.")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--output-dir", type=str, default="/storage/tal/thesis/pred_reverb_prompt_tokens")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-prompt-tokens", type=int, default=8)
    parser.add_argument("--prompt-hidden", type=int, default=128)
    parser.add_argument("--prompt-heads", type=int, default=4)
    parser.add_argument("--prompt-scale", type=float, default=0.05)
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
    prompt_whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    for m in [baseline_whisper, prompt_whisper]:
        m.config.forced_decoder_ids = forced_decoder_ids
        m.generation_config.forced_decoder_ids = forced_decoder_ids
    freeze_all(baseline_whisper)

    model = PredReverbPromptWhisper(
        whisper=prompt_whisper,
        num_prompt_tokens=args.num_prompt_tokens,
        prompt_hidden=args.prompt_hidden,
        prompt_heads=args.prompt_heads,
        prompt_scale=args.prompt_scale,
    ).to(device)

    trainable = count_trainable_parameters(model)
    print(json.dumps({"experiment": "pred_reverb_prompt_tokens", "trainable_params": trainable}, separators=(",", ":")), flush=True)
    optimizer = torch.optim.AdamW(model.prompt_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    baseline_pred_val = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, val_dl, device, args.show_progress)
    init_val = evaluate_prompt_model(model, processor, tokenizer, val_dl, device, condition_mode="true", show_progress=args.show_progress)
    initial_summary = {
        "note": "Frozen Whisper; train only pred-reverb prompt-token generator.",
        "pred_baseline_val": baseline_pred_val,
        "prompt_init_val": init_val,
        "trainable_params": trainable,
        "num_prompt_tokens": args.num_prompt_tokens,
        "prompt_scale": args.prompt_scale,
    }
    (out_dir / "initial_summary.json").write_text(json.dumps(initial_summary, indent=2), encoding="utf-8")

    history: List[Dict] = []
    best_metrics = None
    best_path = out_dir / "best_model.pt"
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_dl, optimizer, device, args.show_progress)
        val_true = evaluate_prompt_model(model, processor, tokenizer, val_dl, device, "true", args.show_progress)
        val_zero = evaluate_prompt_model(model, processor, tokenizer, val_dl, device, "zero", args.show_progress)
        val_shuf = evaluate_prompt_model(model, processor, tokenizer, val_dl, device, "shuffled", args.show_progress)

        row = {
            "epoch": epoch,
            **train_metrics,
            "val_loss": val_true["loss"],
            "val_wer": val_true["wer"],
            "val_cer": val_true["cer"],
            "val_prompt_abs": val_true["prompt_abs"],
            "val_zero_condition_wer": val_zero["wer"],
            "val_shuffled_condition_wer": val_shuf["wer"],
            "true_minus_zero_wer": float(val_true["wer"] - val_zero["wer"]),
            "true_minus_shuffled_wer": float(val_true["wer"] - val_shuf["wer"]),
            "delta_vs_pred_baseline_wer": float(val_true["wer"] - baseline_pred_val["wer"]),
        }
        history.append(row)
        print_compact_epoch(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        candidate = {"wer": val_true["wer"], "cer": val_true["cer"], "loss": val_true["loss"]}
        if is_better(candidate, best_metrics):
            best_metrics = candidate
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "best_metrics": best_metrics, "args": vars(args)}, best_path)
            print(f"[SAVE] Best model saved to {best_path}", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[EARLY-STOP] No validation improvement by WER/CER/loss for {args.patience} consecutive epochs.", flush=True)
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    final_val = evaluate_prompt_model(model, processor, tokenizer, val_dl, device, "true", args.show_progress)
    test_pred = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, test_dl, device, args.show_progress)
    test_prompt = evaluate_prompt_model(model, processor, tokenizer, test_dl, device, "true", args.show_progress)
    final_summary = {
        "note": "Frozen Whisper; train only pred-reverb prompt-token generator.",
        "initial": initial_summary,
        "final_val": final_val,
        "test_pred_baseline": test_pred,
        "test_prompt": test_prompt,
        "best_val_metrics": best_metrics,
        "best_val_delta_vs_pred": float(best_metrics["wer"] - baseline_pred_val["wer"]) if best_metrics is not None else None,
        "test_delta_vs_pred": float(test_prompt["wer"] - test_pred["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (out_dir / "final_report.txt").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[SAVE] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
