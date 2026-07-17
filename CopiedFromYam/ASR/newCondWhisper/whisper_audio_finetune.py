#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
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
)

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **_: object):
        return iterable


@torch.no_grad()
def evaluate_whisper(model, processor, tokenizer, dataloader, device, source_key: str, show_progress: bool, num_beams: int):
    model.eval()
    refs, hyps, losses = [], [], []
    for batch in tqdm(dataloader, desc=f"Eval Whisper {source_key}", disable=not show_progress):
        input_features = batch[source_key].to(device).float()
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = model(
            input_features=input_features,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        gen_ids = model.generate(
            input_features=input_features,
            num_beams=num_beams,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels_text_ids, skip_special_tokens=True)
        refs.extend([clean_text(t) for t in ref_texts])
        hyps.extend([clean_text(t) for t in pred_texts])
        losses.append(float(out.loss.detach().cpu().item()))
    return {
        "loss": float(np.mean(losses)),
        "wer": float(wer(refs, hyps)),
        "cer": float(cer(refs, hyps)),
    }


def train_one_epoch(model, dataloader, optimizer, device, source_key: str, show_progress: bool, scheduler=None, max_train_steps: int = 0, global_step: int = 0):
    model.train()
    total = 0.0
    n = 0
    for batch in tqdm(dataloader, desc=f"Fine-tune Whisper {source_key}", disable=not show_progress):
        if max_train_steps > 0 and global_step >= max_train_steps:
            break
        input_features = batch[source_key].to(device).float()
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        out = model(
            input_features=input_features,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        optimizer.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(out.loss.detach().cpu().item())
        global_step += 1
        n += 1
    return {"train_loss": total / max(n, 1), "global_step": global_step}


def unfreeze_layer_norms(module: torch.nn.Module) -> int:
    total = 0
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.LayerNorm):
            for param in submodule.parameters():
                param.requires_grad = True
                total += param.numel()
    return total


def unfreeze_last_decoder_layers(model: WhisperForConditionalGeneration, last_n: int) -> int:
    if last_n <= 0:
        return 0
    total = 0
    layers = model.model.decoder.layers
    start = max(0, len(layers) - last_n)
    for layer in layers[start:]:
        for param in layer.parameters():
            param.requires_grad = True
            total += param.numel()
    return total


def main():
    parser = argparse.ArgumentParser(description="Audio-only Whisper fine-tuning stage before side-information insertion.")
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--model-name", default="openai/whisper-small")
    parser.add_argument("--output-dir", default="/storage/tal/thesis/script_res/whisper_audio_finetune_reverb")
    parser.add_argument("--input-source", choices=["pred", "reverb", "clean"], default="reverb")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--finetune-mode",
        choices=["full", "decoder_norms", "last_decoder_layers"],
        default="full",
    )
    parser.add_argument("--unfreeze-last-decoder-layers", type=int, default=1)
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
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(device)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.config.forced_decoder_ids = forced_decoder_ids
    model.generation_config.forced_decoder_ids = forced_decoder_ids

    unfrozen_params = None
    if args.finetune_mode != "full":
        freeze_all(model)
        unfrozen = 0
        if args.finetune_mode == "decoder_norms":
            unfrozen += unfreeze_layer_norms(model.model.decoder)
        elif args.finetune_mode == "last_decoder_layers":
            unfrozen += unfreeze_last_decoder_layers(model, args.unfreeze_last_decoder_layers)
        if hasattr(model.model.decoder, "layer_norm"):
            unfrozen += unfreeze_layer_norms(model.model.decoder.layer_norm)
        unfrozen_params = unfrozen

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable Whisper parameters selected for audio fine-tuning.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.warmup_steps > 0:
        def lr_lambda(step: int) -> float:
            return min(1.0, float(step + 1) / float(max(args.warmup_steps, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    initial_val = evaluate_whisper(model, processor, tokenizer, val_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
    initial = {
        "initial_val": initial_val,
        "args": vars(args),
        "trainable_params": count_trainable_parameters(model),
        "unfrozen_params": unfrozen_params,
    }
    (out_dir / "initial_summary.json").write_text(json.dumps(initial, indent=2), encoding="utf-8")
    print(json.dumps({
        "initial_val": initial_val,
        "trainable_params": count_trainable_parameters(model),
        "unfrozen_params": unfrozen_params,
        "args": vars(args),
    }, separators=(",", ":")), flush=True)

    best = None
    best_dir = out_dir / "best_model"
    history: List[Dict] = []
    no_improve = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_dl,
            optimizer,
            device,
            args.input_source,
            args.show_progress,
            scheduler=scheduler,
            max_train_steps=args.max_train_steps,
            global_step=global_step,
        )
        global_step = int(train_metrics.pop("global_step"))
        val_metrics = evaluate_whisper(model, processor, tokenizer, val_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
        row = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "delta_vs_initial_wer": float(val_metrics["wer"] - initial_val["wer"]),
            "global_step": global_step,
        }
        history.append(row)
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps({
            "epoch": epoch,
            "val_wer": f"{100.0 * val_metrics['wer']:.3f}%",
            "delta_vs_initial": f"{100.0 * row['delta_vs_initial_wer']:+.3f}pp",
            "val_loss": round(val_metrics["loss"], 4),
            "train_loss": round(train_metrics["train_loss"], 4),
            "global_step": global_step,
        }, separators=(",", ":")), flush=True)
        candidate = {"wer": val_metrics["wer"], "cer": val_metrics["cer"], "loss": val_metrics["loss"]}
        if is_better(candidate, best):
            best = candidate
            no_improve = 0
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"[SAVE] Best Whisper checkpoint saved to {best_dir}", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                break
        if args.max_train_steps > 0 and global_step >= args.max_train_steps:
            break

    if best_dir.exists():
        model = WhisperForConditionalGeneration.from_pretrained(best_dir).to(device)
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
    test_metrics = evaluate_whisper(model, processor, tokenizer, test_dl, device, args.input_source, args.show_progress, args.eval_num_beams)
    final = {
        "initial": initial,
        "best_val_metrics": best,
        "test_audio_finetuned": test_metrics,
        "best_model_dir": str(best_dir),
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
