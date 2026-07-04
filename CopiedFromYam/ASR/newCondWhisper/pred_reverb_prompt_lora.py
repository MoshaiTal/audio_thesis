#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

from CopiedFromYam.ASR.newCondWhisper.condwhisper_film import (
    CondWhisperVNextCollator,
    CondWhisperVNextDataset,
    count_trainable_parameters,
    evaluate_baseline_pred,
    freeze_all,
    is_better,
)
from CopiedFromYam.ASR.newCondWhisper.pred_reverb_prompt_tokens import (
    PredReverbPromptWhisper,
    evaluate_prompt_model,
    print_compact_epoch,
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


class LoRALinear(nn.Module):
    """Frozen Linear layer plus a small trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * nn.functional.linear(
            nn.functional.linear(self.dropout(x), self.lora_a),
            self.lora_b,
        )


def install_decoder_cross_attn_lora(
    whisper: WhisperForConditionalGeneration,
    rank: int,
    alpha: float,
    dropout: float,
    targets: List[str],
    layers: List[int] | None = None,
) -> int:
    """Install LoRA only on decoder encoder-attention projections."""
    installed = 0
    decoder_layers = whisper.model.decoder.layers
    layer_ids = layers if layers is not None else list(range(len(decoder_layers)))
    for layer_idx in layer_ids:
        attn = decoder_layers[layer_idx].encoder_attn
        for name in targets:
            proj = getattr(attn, name)
            if not isinstance(proj, nn.Linear):
                raise TypeError(f"Expected decoder layer {layer_idx}.encoder_attn.{name} to be nn.Linear")
            setattr(attn, name, LoRALinear(proj, rank=rank, alpha=alpha, dropout=dropout))
            installed += 1
    return installed


def train_one_epoch(model, dataloader, optimizer, device, show_progress: bool):
    model.train()
    total_loss = 0.0
    prompt_vals = []
    n = 0
    for batch in tqdm(dataloader, desc="Training prompt+LoRA", disable=not show_progress):
        pred = batch["pred"].to(device).float()
        reverb = batch["reverb"].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(pred=pred, reverb=reverb, lengths=lengths, labels=labels, decoder_attention_mask=decoder_attention_mask)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        prompt_vals.append(float(out["prompt_abs"].detach().cpu().item()))
        n += 1
    return {"train_loss": total_loss / max(n, 1), "train_prompt_abs": float(np.mean(prompt_vals))}


def parse_int_list(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Pred-reverb prompt tokens + LoRA on frozen Whisper decoder cross-attention.")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--output-dir", type=str, default="/storage/tal/thesis/pred_reverb_prompt_lora")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-prompt-tokens", type=int, default=4)
    parser.add_argument("--prompt-hidden", type=int, default=128)
    parser.add_argument("--prompt-heads", type=int, default=4)
    parser.add_argument("--prompt-scale", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", type=str, default="q_proj,k_proj,v_proj,out_proj")
    parser.add_argument("--lora-decoder-layers", type=str, default="", help="Comma-separated decoder layer ids. Empty means all decoder layers.")
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

    targets = [x.strip() for x in args.lora_targets.split(",") if x.strip()]
    lora_layers = parse_int_list(args.lora_decoder_layers) if args.lora_decoder_layers.strip() else None
    installed = install_decoder_cross_attn_lora(
        model.whisper,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=targets,
        layers=lora_layers,
    )
    model.to(device)

    trainable = count_trainable_parameters(model)
    print(
        json.dumps(
            {
                "experiment": "pred_reverb_prompt_lora",
                "trainable_params": trainable,
                "lora_modules": installed,
                "lora_targets": targets,
                "lora_decoder_layers": lora_layers if lora_layers is not None else "all",
            },
            separators=(",", ":"),
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    baseline_pred_val = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, val_dl, device, args.show_progress)
    init_val = evaluate_prompt_model(model, processor, tokenizer, val_dl, device, condition_mode="true", show_progress=args.show_progress)
    initial_summary = {
        "note": "Frozen Whisper base; train pred-reverb prompt generator and LoRA on decoder cross-attention.",
        "pred_baseline_val": baseline_pred_val,
        "prompt_lora_init_val": init_val,
        "trainable_params": trainable,
        "num_prompt_tokens": args.num_prompt_tokens,
        "prompt_scale": args.prompt_scale,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_targets": targets,
        "lora_decoder_layers": lora_layers if lora_layers is not None else "all",
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
        "note": "Frozen Whisper base; train pred-reverb prompt generator and LoRA on decoder cross-attention.",
        "initial": initial_summary,
        "final_val": final_val,
        "test_pred_baseline": test_pred,
        "test_prompt_lora": test_prompt,
        "best_val_metrics": best_metrics,
        "best_val_delta_vs_pred": float(best_metrics["wer"] - baseline_pred_val["wer"]) if best_metrics is not None else None,
        "test_delta_vs_pred": float(test_prompt["wer"] - test_pred["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (out_dir / "final_report.txt").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[SAVE] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
