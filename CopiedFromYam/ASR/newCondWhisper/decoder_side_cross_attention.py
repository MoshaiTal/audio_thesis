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

from CopiedFromYam.ASR.newCondWhisper.clean_hidden_side_cross_attention import (
    HiddenSideCrossAttention,
    freeze_adapter_residual_scales,
    set_optimizer_lr,
)
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


class DecoderSideCrossAttentionWhisper(nn.Module):
    """Keep Whisper's normal encoder path intact and add optional side attention in decoder layers."""

    def __init__(
        self,
        whisper: WhisperForConditionalGeneration,
        side_whisper: WhisperForConditionalGeneration,
        selected_decoder_layers: List[int],
        adapter_bottleneck: int = 256,
        side_attn_heads: int = 4,
        side_attn_dropout: float = 0.05,
        init_residual_scale: float = 0.001,
        max_update_abs: float = 0.0,
    ):
        super().__init__()
        self.whisper = whisper
        self.side_whisper = side_whisper
        self.selected_decoder_layers = sorted(selected_decoder_layers)
        self.max_update_abs = max_update_abs

        freeze_all(self.whisper)
        freeze_all(self.side_whisper)

        d_model = whisper.model.config.d_model
        self.adapters = nn.ModuleDict({
            str(layer_idx): HiddenSideCrossAttention(
                d_model=d_model,
                bottleneck=adapter_bottleneck,
                num_heads=side_attn_heads,
                attn_dropout=side_attn_dropout,
                init_residual_scale=init_residual_scale,
            )
            for layer_idx in self.selected_decoder_layers
        })
        self._side_memory: Optional[torch.Tensor] = None
        self._side_key_padding_mask: Optional[torch.Tensor] = None
        self._last_debug: List[Dict] = []
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        for idx in self.selected_decoder_layers:
            layer = self.whisper.model.decoder.layers[idx]

            def hook(module, args, output, layer_idx=idx):
                if self._side_memory is None or self._side_key_padding_mask is None:
                    return output
                hidden = output[0] if isinstance(output, tuple) else output
                before = hidden
                side_memory = self._side_memory
                side_key_padding_mask = self._side_key_padding_mask
                if side_memory.shape[0] != hidden.shape[0]:
                    if hidden.shape[0] % side_memory.shape[0] != 0:
                        raise RuntimeError(
                            f"Cannot align side batch {side_memory.shape[0]} with decoder batch {hidden.shape[0]}"
                        )
                    repeat_factor = hidden.shape[0] // side_memory.shape[0]
                    side_memory = side_memory.repeat_interleave(repeat_factor, dim=0)
                    side_key_padding_mask = side_key_padding_mask.repeat_interleave(repeat_factor, dim=0)
                new_hidden, dbg = self.adapters[str(layer_idx)](
                    hidden,
                    side_memory,
                    side_key_padding_mask,
                )
                if self.max_update_abs > 0.0:
                    delta = new_hidden - before
                    update_abs = delta.abs().mean().detach()
                    if float(update_abs.item()) > self.max_update_abs:
                        delta = delta * (self.max_update_abs / update_abs.clamp(min=1e-8))
                        new_hidden = before + delta
                        dbg["layer_update_abs"] = (new_hidden - before).abs().mean()
                        dbg["hard_limited"] = torch.tensor(1.0, device=hidden.device)
                    else:
                        dbg["hard_limited"] = torch.tensor(0.0, device=hidden.device)
                self._last_debug.append({"layer": layer_idx, **dbg})
                if isinstance(output, tuple):
                    return (new_hidden, *output[1:])
                return new_hidden

            self._handles.append(layer.register_forward_hook(hook))

    def _encoder_mask(self, lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        enc_lengths = ((lengths + 1) // 2).clamp(max=seq_len)
        return make_time_mask(enc_lengths, seq_len)

    @torch.no_grad()
    def _prepare_side(self, side_features: torch.Tensor, lengths: torch.Tensor):
        side_out = self.side_whisper.model.encoder(input_features=side_features, return_dict=True)
        self._side_memory = side_out.last_hidden_state
        keep = self._encoder_mask(lengths, self._side_memory.shape[1])
        self._side_memory = self._side_memory * keep[:, :, None].float()
        self._side_key_padding_mask = ~keep
        self._last_debug = []

    def _clear_side(self):
        self._side_memory = None
        self._side_key_padding_mask = None

    def forward(self, input_features, side_features, lengths, labels, decoder_attention_mask):
        self._prepare_side(side_features, lengths)
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
        self._prepare_side(side_features, lengths)
        try:
            gen_ids = self.whisper.generate(input_features=input_features, **kwargs)
            debug = self.debug_summary()
        finally:
            self._clear_side()
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
def evaluate_model(model, processor, tokenizer, dataloader, device, input_source: str, side_source: str, show_progress: bool, num_beams: int):
    model.eval()
    refs, hyps, losses, updates = [], [], [], []
    for batch in tqdm(dataloader, desc="Eval decoder-side side cross-attn", disable=not show_progress):
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
    side_source: str,
    lambda_update: float,
    lambda_update_ceiling: float,
    update_ceiling: float,
    show_progress: bool,
):
    model.train()
    total = asr = update_reg_total = update_ceiling_total = 0.0
    n = 0
    last_update = 0.0
    for batch in tqdm(dataloader, desc="Training decoder-side side cross-attn", disable=not show_progress):
        input_features = batch[input_source].to(device).float()
        side_features = batch[side_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(input_features, side_features, lengths, labels, decoder_attention_mask)
        update_reg = out["debug"]["mean_layer_update_abs"]
        update_ceiling_loss = out["loss"].new_tensor(0.0)
        if lambda_update_ceiling > 0.0 and update_ceiling > 0.0:
            update_ceiling_loss = F.relu(update_reg - update_ceiling).pow(2)
        loss = out["loss"] + lambda_update * update_reg + lambda_update_ceiling * update_ceiling_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total += float(loss.item())
        asr += float(out["loss"].item())
        update_reg_total += float(update_reg.detach().cpu().item())
        update_ceiling_total += float(update_ceiling_loss.detach().cpu().item())
        last_update = float(update_reg.detach().cpu().item())
        n += 1
    return {
        "train_total_loss": total / max(n, 1),
        "train_asr_loss": asr / max(n, 1),
        "train_update_reg": update_reg_total / max(n, 1),
        "train_update_ceiling_loss": update_ceiling_total / max(n, 1),
        "last_update_abs": last_update,
    }


def main():
    parser = argparse.ArgumentParser(description="Decoder-side side cross-attention for Whisper.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="/storage/tal/thesis/script_res/decoder_side_cross_attention")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-decay-after-epoch", type=int, default=2)
    parser.add_argument("--lr-after-decay", type=float, default=2e-6)
    parser.add_argument("--freeze-residual-scale-after-epoch", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selected-decoder-layers", default="3,6")
    parser.add_argument("--input-source", choices=["pred", "reverb", "clean"], default="reverb")
    parser.add_argument("--side-source", choices=["pred", "reverb", "clean"], default="clean")
    parser.add_argument("--adapter-bottleneck", type=int, default=256)
    parser.add_argument("--side-attn-heads", type=int, default=4)
    parser.add_argument("--side-attn-dropout", type=float, default=0.05)
    parser.add_argument("--init-residual-scale", type=float, default=0.001)
    parser.add_argument("--max-update-abs", type=float, default=0.02)
    parser.add_argument("--lambda-update", type=float, default=0.08)
    parser.add_argument("--lambda-update-ceiling", type=float, default=300.0)
    parser.add_argument("--update-ceiling", type=float, default=0.02)
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
    side_whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    for m in [baseline_whisper, whisper, side_whisper]:
        m.config.forced_decoder_ids = forced_decoder_ids
        m.generation_config.forced_decoder_ids = forced_decoder_ids
    freeze_all(baseline_whisper)

    model = DecoderSideCrossAttentionWhisper(
        whisper=whisper,
        side_whisper=side_whisper,
        selected_decoder_layers=selected_layers,
        adapter_bottleneck=args.adapter_bottleneck,
        side_attn_heads=args.side_attn_heads,
        side_attn_dropout=args.side_attn_dropout,
        init_residual_scale=args.init_residual_scale,
        max_update_abs=args.max_update_abs,
    ).to(device)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    print(json.dumps({
        "experiment": "decoder_side_cross_attention",
        "trainable_params": count_trainable_parameters(model),
        "selected_decoder_layers": selected_layers,
        "args": vars(args),
    }, separators=(",", ":")), flush=True)

    baseline_input = evaluate_baseline(
        baseline_whisper,
        processor,
        tokenizer,
        val_dl,
        device,
        args.input_source,
        args.show_progress,
        args.eval_num_beams,
    )
    init_val = evaluate_model(model, processor, tokenizer, val_dl, device, args.input_source, args.side_source, args.show_progress, args.eval_num_beams)
    initial = {"baseline_input_val": baseline_input, "init_val": init_val, "args": vars(args)}
    (out_dir / "initial_summary.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(json.dumps({"initial_delta_vs_input": init_val["wer"] - baseline_input["wer"], "init_val": init_val}, separators=(",", ":")), flush=True)

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
            args.side_source,
            args.lambda_update,
            args.lambda_update_ceiling,
            args.update_ceiling,
            args.show_progress,
        )
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
    test_input = evaluate_baseline(
        baseline_whisper,
        processor,
        tokenizer,
        test_dl,
        device,
        args.input_source,
        args.show_progress,
        args.eval_num_beams,
    )
    test_model = evaluate_model(model, processor, tokenizer, test_dl, device, args.input_source, args.side_source, args.show_progress, args.eval_num_beams)
    final = {
        "initial": initial,
        "best_val_metrics": best,
        "test_input_baseline": test_input,
        "test_decoder_side_cross_attention": test_model,
        "test_delta_vs_input": float(test_model["wer"] - test_input["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
