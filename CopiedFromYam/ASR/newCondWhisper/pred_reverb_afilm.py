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
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from transformers.modeling_outputs import BaseModelOutput

from CopiedFromYam.ASR.newCondWhisper.condwhisper_film import (
    CPFiLMAdapter,
    CondWhisperVNextCollator,
    CondWhisperVNextDataset,
    count_trainable_parameters,
    evaluate_baseline_pred,
    freeze_all,
    is_better,
    make_time_mask,
    unfreeze_last_encoder_layers,
)
from CopiedFromYam.ASR.newCondWhisper.oracle_pred_clean_film import (
    evaluate_oracle_model,
    print_compact_epoch,
    train_one_epoch_oracle,
)


class PredReverbAFiLMConditioner(nn.Module):
    """Attention-based FiLM conditioner for deployable pred-reverb cues.

    This follows the AFiLM idea at the conditioning-generator level: compress
    time-frequency conditioning features into a temporal sequence, run
    self-attention over that sequence, and use the contextualized sequence to
    generate layer-wise FiLM parameters in the adapter.
    """

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        hidden: int = 128,
        out_scale: float = 0.20,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.out_scale = out_scale
        in_ch = n_mels * 4  # pred, reverb, pred-reverb, abs(pred-reverb)
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=1),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=max(d_model * 2, 512),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.global_mlp = nn.Sequential(
            nn.Linear(d_model + 8, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.normal_(self.input_proj[-1].weight, mean=0.0, std=7e-4)
        nn.init.constant_(self.input_proj[-1].bias, 0.0)
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

    def forward(self, pred: torch.Tensor, reverb: torch.Tensor, target_len: int, lengths: torch.Tensor):
        diff = pred - reverb
        abs_diff = torch.abs(diff)
        x = torch.cat([pred, reverb, diff, abs_diff], dim=1)
        local = self.input_proj(x)
        if local.shape[-1] != target_len:
            local = F.interpolate(local, size=target_len, mode="linear", align_corners=False)

        seq = local.transpose(1, 2)
        enc_lengths = ((lengths + 1) // 2).clamp(max=target_len)
        keep = make_time_mask(enc_lengths, target_len)
        seq = self.temporal_encoder(seq, src_key_padding_mask=~keep)
        seq = seq * keep[:, :, None].float()

        denom = keep.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = seq.sum(dim=1) / denom
        stats = self._global_stats(pred, reverb, diff, abs_diff, lengths)
        glob = self.global_mlp(torch.cat([pooled, stats], dim=1))
        return self.out_scale * torch.tanh(seq.transpose(1, 2)), self.out_scale * torch.tanh(glob)


class PredReverbAFiLMWhisper(nn.Module):
    def __init__(
        self,
        whisper_student: WhisperForConditionalGeneration,
        selected_layers: List[int],
        conditioner_hidden: int = 128,
        conditioner_scale: float = 0.20,
        conditioner_layers: int = 2,
        conditioner_heads: int = 4,
        adapter_bottleneck: int = 128,
        adapter_film_scale: float = 0.25,
        init_residual_scale: float = 0.03,
        adapter_mode: str = "residual_film",
    ):
        super().__init__()
        self.student = whisper_student
        self.selected_layers = sorted(selected_layers)

        d_model = whisper_student.model.config.d_model
        n_mels = whisper_student.model.config.num_mel_bins
        self.conditioner = PredReverbAFiLMConditioner(
            n_mels=n_mels,
            d_model=d_model,
            hidden=conditioner_hidden,
            out_scale=conditioner_scale,
            num_layers=conditioner_layers,
            num_heads=conditioner_heads,
        )
        self.adapters = nn.ModuleDict({
            str(i): CPFiLMAdapter(
                d_model=d_model,
                bottleneck=adapter_bottleneck,
                film_scale=adapter_film_scale,
                init_residual_scale=init_residual_scale,
                adapter_mode=adapter_mode,
            )
            for i in self.selected_layers
        })

    def encode_student(self, pred, clean, lengths, output_hidden_states: bool = True):
        # The argument is named clean for compatibility with the shared
        # train/eval helpers; in this script it is the reverb conditioning mel.
        reverb = clean
        enc = self.student.model.encoder
        x = F.gelu(enc.conv1(pred))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)

        Tenc = x.shape[1]
        pos = enc.embed_positions.weight[:Tenc, :].to(x.dtype)
        x = x + pos[None, :, :]

        cond_local, cond_global = self.conditioner(pred, reverb, Tenc, lengths)
        hidden_states = [x] if output_hidden_states else None
        adapter_debug = []

        for idx, layer in enumerate(enc.layers):
            x = layer(x, attention_mask=None, output_attentions=False)[0]
            if idx in self.selected_layers:
                x, dbg = self.adapters[str(idx)](x, cond_local, cond_global)
                adapter_debug.append(
                    {
                        "layer": idx,
                        "residual_scale": dbg["residual_scale"],
                        "gamma_abs": dbg["gamma_abs"],
                        "beta_abs": dbg["beta_abs"],
                        "delta_abs": dbg["delta_abs"],
                        "layer_update_abs": dbg["layer_update_abs"],
                    }
                )
            if output_hidden_states:
                hidden_states.append(x)

        x = enc.layer_norm(x)
        return {
            "encoder_outputs": BaseModelOutput(last_hidden_state=x),
            "hidden_states": tuple(hidden_states) if output_hidden_states else None,
            "adapter_debug": adapter_debug,
            "cond_local_abs": cond_local.abs().mean(),
            "cond_global_abs": cond_global.abs().mean(),
        }

    def forward(self, pred, clean, lengths, labels, decoder_attention_mask, output_hidden_states: bool = True):
        from transformers.models.whisper.modeling_whisper import shift_tokens_right

        enc = self.encode_student(pred, clean, lengths, output_hidden_states=output_hidden_states)
        out = self.student(
            encoder_outputs=enc["encoder_outputs"],
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        decoder_input_ids = shift_tokens_right(
            labels,
            self.student.config.pad_token_id,
            self.student.config.decoder_start_token_id,
        )
        out_for_kd = self.student(
            encoder_outputs=enc["encoder_outputs"],
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return {
            "loss": out.loss,
            "logits": out_for_kd.logits,
            "hidden_states": enc["hidden_states"],
            "adapter_debug": enc["adapter_debug"],
            "cond_local_abs": enc["cond_local_abs"],
            "cond_global_abs": enc["cond_global_abs"],
        }

    @torch.no_grad()
    def generate(self, pred, clean, lengths, **kwargs):
        enc = self.encode_student(pred, clean, lengths, output_hidden_states=False)
        gen_ids = self.student.generate(encoder_outputs=enc["encoder_outputs"], **kwargs)
        return gen_ids, enc


def main():
    parser = argparse.ArgumentParser(description="Pred-reverb AFiLM-conditioned Whisper experiment.")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--output-dir", type=str, default="/storage/tal/thesis/pred_reverb_afilm")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selected-layers", type=str, default="3")
    parser.add_argument("--last-n-unfrozen-encoder-layers", type=int, default=0)
    parser.add_argument("--conditioner-hidden", type=int, default=128)
    parser.add_argument("--conditioner-scale", type=float, default=0.20)
    parser.add_argument("--conditioner-layers", type=int, default=2)
    parser.add_argument("--conditioner-heads", type=int, default=4)
    parser.add_argument("--adapter-bottleneck", type=int, default=128)
    parser.add_argument("--adapter-film-scale", type=float, default=0.25)
    parser.add_argument("--init-residual-scale", type=float, default=0.03)
    parser.add_argument("--adapter-mode", choices=["residual_film", "direct_film"], default="residual_film")
    parser.add_argument("--lambda-hidden-kd", type=float, default=0.10)
    parser.add_argument("--lambda-logit-kd", type=float, default=0.20)
    parser.add_argument("--lambda-adapter-update", type=float, default=0.001)
    parser.add_argument("--eval-condition-ablations", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    args = parser.parse_args()

    selected_layers = [int(x) for x in args.selected_layers.split(",") if x.strip()]
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
    teacher_whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    student_whisper = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    for m in [baseline_whisper, teacher_whisper, student_whisper]:
        m.config.forced_decoder_ids = forced_decoder_ids
        m.generation_config.forced_decoder_ids = forced_decoder_ids

    freeze_all(baseline_whisper)
    freeze_all(teacher_whisper)
    freeze_all(student_whisper)
    unfreeze_last_encoder_layers(student_whisper, args.last_n_unfrozen_encoder_layers)

    model = PredReverbAFiLMWhisper(
        whisper_student=student_whisper,
        selected_layers=selected_layers,
        conditioner_hidden=args.conditioner_hidden,
        conditioner_scale=args.conditioner_scale,
        conditioner_layers=args.conditioner_layers,
        conditioner_heads=args.conditioner_heads,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_film_scale=args.adapter_film_scale,
        init_residual_scale=args.init_residual_scale,
        adapter_mode=args.adapter_mode,
    ).to(device)

    trainable = count_trainable_parameters(model)
    print(
        json.dumps(
            {
                "trainable_params": trainable,
                "experiment": "pred_reverb_afilm",
                "selected_layers": selected_layers,
                "adapter_mode": args.adapter_mode,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    baseline_pred_val = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, val_dl, device, args.show_progress)
    init_val = evaluate_oracle_model(
        model,
        teacher_whisper,
        processor,
        tokenizer,
        val_dl,
        device,
        selected_layers,
        args.show_progress,
        condition_source="reverb",
    )

    initial_summary = {
        "note": "Pred-reverb AFiLM conditioner.",
        "pred_baseline_val": baseline_pred_val,
        "pred_reverb_afilm_init_val": init_val,
        "trainable_params": trainable,
        "selected_layers": selected_layers,
        "adapter_mode": args.adapter_mode,
        "conditioner_layers": args.conditioner_layers,
        "conditioner_heads": args.conditioner_heads,
    }
    (out_dir / "initial_summary.json").write_text(json.dumps(initial_summary, indent=2), encoding="utf-8")

    history: List[Dict] = []
    best_metrics = None
    best_path = out_dir / "best_model.pt"
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch_oracle(
            model=model,
            teacher=teacher_whisper,
            dataloader=train_dl,
            optimizer=optimizer,
            device=device,
            selected_layers=selected_layers,
            lambda_hidden_kd=args.lambda_hidden_kd,
            lambda_logit_kd=args.lambda_logit_kd,
            lambda_adapter_update=args.lambda_adapter_update,
            show_progress=args.show_progress,
            condition_source="reverb",
        )
        val_metrics = evaluate_oracle_model(
            model,
            teacher_whisper,
            processor,
            tokenizer,
            val_dl,
            device,
            selected_layers,
            args.show_progress,
            condition_source="reverb",
        )

        val_zero_metrics = None
        val_shuffled_metrics = None
        if args.eval_condition_ablations:
            val_zero_metrics = evaluate_oracle_model(
                model,
                teacher_whisper,
                processor,
                tokenizer,
                val_dl,
                device,
                selected_layers,
                args.show_progress,
                condition_mode="zero",
                condition_source="reverb",
            )
            val_shuffled_metrics = evaluate_oracle_model(
                model,
                teacher_whisper,
                processor,
                tokenizer,
                val_dl,
                device,
                selected_layers,
                args.show_progress,
                condition_mode="shuffled",
                condition_source="reverb",
            )

        row = {
            "epoch": epoch,
            "lr": args.lr,
            "selected_layers": selected_layers,
            "condition_source": "reverb",
            "adapter_mode": args.adapter_mode,
            "lambda_hidden_kd": args.lambda_hidden_kd,
            "lambda_logit_kd": args.lambda_logit_kd,
            "lambda_adapter_update": args.lambda_adapter_update,
            **{k: v for k, v in train_metrics.items() if k not in {"last_grad_norms", "last_debug"}},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "delta_vs_pred_baseline_wer": float(val_metrics["wer"] - baseline_pred_val["wer"]),
            "last_grad_norms": train_metrics["last_grad_norms"],
            "last_debug": train_metrics["last_debug"],
        }
        if args.eval_condition_ablations:
            row["val_zero_condition_wer"] = val_zero_metrics["wer"]
            row["val_shuffled_condition_wer"] = val_shuffled_metrics["wer"]
            row["true_minus_zero_wer"] = float(val_metrics["wer"] - val_zero_metrics["wer"])
            row["true_minus_shuffled_wer"] = float(val_metrics["wer"] - val_shuffled_metrics["wer"])
        history.append(row)
        print_compact_epoch(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

        candidate = {"wer": val_metrics["wer"], "cer": val_metrics["cer"], "loss": val_metrics["loss"]}
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
    final_val = evaluate_oracle_model(model, teacher_whisper, processor, tokenizer, val_dl, device, selected_layers, args.show_progress, condition_source="reverb")
    test_pred = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, test_dl, device, args.show_progress)
    test_afilm = evaluate_oracle_model(model, teacher_whisper, processor, tokenizer, test_dl, device, selected_layers, args.show_progress, condition_source="reverb")

    final_summary = {
        "note": "Pred-reverb AFiLM conditioner.",
        "initial": initial_summary,
        "final_val": final_val,
        "test_pred_baseline": test_pred,
        "test_pred_reverb_afilm": test_afilm,
        "best_val_metrics": best_metrics,
        "best_val_delta_vs_pred": float(best_metrics["wer"] - baseline_pred_val["wer"]) if best_metrics is not None else None,
        "test_delta_vs_pred": float(test_afilm["wer"] - test_pred["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (out_dir / "final_report.txt").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[SAVE] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
