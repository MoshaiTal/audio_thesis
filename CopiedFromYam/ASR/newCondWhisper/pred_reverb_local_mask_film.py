#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer

from CopiedFromYam.ASR.newCondWhisper.condwhisper_film import (
    CondWhisperVNextCollator,
    CondWhisperVNextDataset,
    count_trainable_parameters,
    evaluate_baseline_pred,
    freeze_all,
    is_better,
    unfreeze_last_encoder_layers,
)
from CopiedFromYam.ASR.newCondWhisper.oracle_pred_clean_film import (
    evaluate_oracle_model,
    print_compact_epoch,
    train_one_epoch_oracle,
)
from CopiedFromYam.ASR.newCondWhisper.pred_reverb_afilm import PredReverbAFiLMWhisper


class PredReverbLocalMaskConditioner(nn.Module):
    """Local time-frequency mask conditioner for pred-reverb differences.

    This keeps the conditioning signal explicitly tied to problematic
    time-frequency bins. Unlike the AFiLM conditioner, it does not use temporal
    self-attention or global utterance pooling. Zero condition therefore means
    zero local difference signal.
    """

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        hidden: int = 128,
        out_scale: float = 0.20,
        mask_hidden: int = 64,
    ):
        super().__init__()
        self.out_scale = out_scale
        self.d_model = d_model
        diff_ch = n_mels * 2  # diff, abs(diff)

        self.mask_net = nn.Sequential(
            nn.Conv1d(diff_ch, mask_hidden, kernel_size=7, padding=3, bias=False),
            nn.GELU(),
            nn.Conv1d(mask_hidden, n_mels, kernel_size=1, bias=True),
        )
        self.local_proj = nn.Sequential(
            nn.Conv1d(n_mels * 4, hidden, kernel_size=9, padding=4, bias=False),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=1, bias=False),
        )

        nn.init.constant_(self.mask_net[-1].bias, 0.0)
        nn.init.normal_(self.local_proj[-1].weight, mean=0.0, std=7e-4)

    def forward(self, pred: torch.Tensor, reverb: torch.Tensor, target_len: int, lengths: torch.Tensor):
        diff = pred - reverb
        abs_diff = torch.abs(diff)

        mask = torch.sigmoid(self.mask_net(torch.cat([diff, abs_diff], dim=1)))
        masked_diff = diff * mask
        masked_abs = abs_diff * mask

        # Only local difference-derived channels are projected. This avoids
        # giving the conditioner a generic copy of pred/reverb that can become
        # an unconditional adapter signal.
        local_input = torch.cat([diff, abs_diff, masked_diff, masked_abs], dim=1)
        local = self.local_proj(local_input)
        if local.shape[-1] != target_len:
            local = F.interpolate(local, size=target_len, mode="linear", align_corners=False)

        global_vec = torch.zeros(pred.shape[0], self.d_model, device=pred.device, dtype=pred.dtype)
        return self.out_scale * torch.tanh(local), global_vec


class PredReverbLocalMaskFiLMWhisper(PredReverbAFiLMWhisper):
    def __init__(
        self,
        whisper_student: WhisperForConditionalGeneration,
        selected_layers: List[int],
        conditioner_hidden: int = 128,
        conditioner_scale: float = 0.20,
        mask_hidden: int = 64,
        adapter_bottleneck: int = 128,
        adapter_film_scale: float = 0.25,
        init_residual_scale: float = 0.03,
        adapter_mode: str = "residual_film",
    ):
        super().__init__(
            whisper_student=whisper_student,
            selected_layers=selected_layers,
            conditioner_hidden=conditioner_hidden,
            conditioner_scale=conditioner_scale,
            conditioner_layers=1,
            conditioner_heads=4,
            adapter_bottleneck=adapter_bottleneck,
            adapter_film_scale=adapter_film_scale,
            init_residual_scale=init_residual_scale,
            adapter_mode=adapter_mode,
        )
        d_model = whisper_student.model.config.d_model
        n_mels = whisper_student.model.config.num_mel_bins
        self.conditioner = PredReverbLocalMaskConditioner(
            n_mels=n_mels,
            d_model=d_model,
            hidden=conditioner_hidden,
            out_scale=conditioner_scale,
            mask_hidden=mask_hidden,
        )


def main():
    parser = argparse.ArgumentParser(description="Pred-reverb local-mask FiLM Whisper experiment.")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--output-dir", type=str, default="/storage/tal/thesis/pred_reverb_local_mask_film")
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
    parser.add_argument("--mask-hidden", type=int, default=64)
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

    model = PredReverbLocalMaskFiLMWhisper(
        whisper_student=student_whisper,
        selected_layers=selected_layers,
        conditioner_hidden=args.conditioner_hidden,
        conditioner_scale=args.conditioner_scale,
        mask_hidden=args.mask_hidden,
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
                "experiment": "pred_reverb_local_mask_film",
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
    init_val = evaluate_oracle_model(model, teacher_whisper, processor, tokenizer, val_dl, device, selected_layers, args.show_progress, condition_source="reverb")

    initial_summary = {
        "note": "Pred-reverb local mask FiLM conditioner.",
        "pred_baseline_val": baseline_pred_val,
        "pred_reverb_local_mask_film_init_val": init_val,
        "trainable_params": trainable,
        "selected_layers": selected_layers,
        "adapter_mode": args.adapter_mode,
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
        val_metrics = evaluate_oracle_model(model, teacher_whisper, processor, tokenizer, val_dl, device, selected_layers, args.show_progress, condition_source="reverb")

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
    test_model = evaluate_oracle_model(model, teacher_whisper, processor, tokenizer, test_dl, device, selected_layers, args.show_progress, condition_source="reverb")

    final_summary = {
        "note": "Pred-reverb local mask FiLM conditioner.",
        "initial": initial_summary,
        "final_val": final_val,
        "test_pred_baseline": test_pred,
        "test_pred_reverb_local_mask_film": test_model,
        "best_val_metrics": best_metrics,
        "best_val_delta_vs_pred": float(best_metrics["wer"] - baseline_pred_val["wer"]) if best_metrics is not None else None,
        "test_delta_vs_pred": float(test_model["wer"] - test_pred["wer"]),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (out_dir / "final_report.txt").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[SAVE] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
