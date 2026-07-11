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
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right

from CopiedFromYam.ASR.newCondWhisper.condwhisper_film import (
    CPFiLMAdapter,
    CondWhisperVNextCollator,
    CondWhisperVNextDataset,
    clean_text,
    count_trainable_parameters,
    evaluate_baseline_pred,
    expected_encoder_len,
    freeze_all,
    grad_norms,
    inv_softplus,
    is_better,
    make_time_mask,
    masked_hidden_mse,
    masked_kl,
    unfreeze_last_encoder_layers,
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
        "train_asr_loss": round(row["train_asr_loss"], 4),
        "train_total_loss": round(row["train_total_loss"], 4),
    }
    if "train_condition_contrast_loss" in row:
        summary["contrast_loss"] = round(row["train_condition_contrast_loss"], 4)
    if "train_true_minus_shuffled_asr_loss" in row:
        summary["true-shuf-ce"] = round(row["train_true_minus_shuffled_asr_loss"], 4)
    if "train_true_minus_zero_asr_loss" in row:
        summary["true-zero-ce"] = round(row["train_true_minus_zero_asr_loss"], 4)
    if "true_minus_zero_wer" in row:
        summary["true-zero"] = f"{100.0 * row['true_minus_zero_wer']:+.3f}pp"
        summary["true-shuffled"] = f"{100.0 * row['true_minus_shuffled_wer']:+.3f}pp"
    if row.get("last_debug") and row["last_debug"].get("mean_layer_update_abs") is not None:
        summary["update_abs"] = round(float(row["last_debug"]["mean_layer_update_abs"]), 5)
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


@torch.no_grad()
def evaluate_baseline_source(whisper, processor, tokenizer, dataloader, device, source_key: str, show_progress: bool):
    whisper.eval()
    refs, preds, losses = [], [], []
    for batch in tqdm(dataloader, desc=f"Eval baseline {source_key}", disable=not show_progress):
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        input_features = batch[source_key].to(device).float()
        out = whisper(
            input_features=input_features,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        gen_ids = whisper.generate(
            input_features=input_features,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels_text_ids, skip_special_tokens=True)
        refs.extend([clean_text(x) for x in ref_texts])
        preds.extend([clean_text(x) for x in pred_texts])
        losses.append(float(out.loss.item()))
    return {"loss": float(np.mean(losses)), "wer": float(wer(refs, preds)), "cer": float(cer(refs, preds))}


class PredCleanConditioner(nn.Module):
    """Oracle conditioner that receives the clean target mel during train/eval.

    This is not deployable as-is because clean is unavailable at inference. It is
    an upper-bound experiment: if FiLM can exploit pred-clean information, then a
    future model can try to predict this error map from CP/reverb cues.
    """

    def __init__(
        self,
        n_mels: int,
        d_model: int,
        hidden: int = 128,
        out_scale: float = 0.20,
        conditioner_mode: str = "full",
    ):
        super().__init__()
        if conditioner_mode not in {"full", "local_error", "global_error"}:
            raise ValueError(f"Unsupported conditioner_mode: {conditioner_mode}")
        self.out_scale = out_scale
        self.conditioner_mode = conditioner_mode
        in_ch = n_mels * 2 if conditioner_mode == "local_error" else n_mels * 4
        self.local_net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, d_model, kernel_size=1),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(12, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.normal_(self.local_net[-1].weight, mean=0.0, std=7e-4)
        nn.init.constant_(self.local_net[-1].bias, 0.0)
        nn.init.normal_(self.global_mlp[-1].weight, mean=0.0, std=7e-4)
        nn.init.constant_(self.global_mlp[-1].bias, 0.0)

    def _global_feats(self, pred: torch.Tensor, clean: torch.Tensor, error: torch.Tensor, abs_error: torch.Tensor, lengths: torch.Tensor):
        B, _, T = pred.shape
        mask = make_time_mask(lengths, T).float()
        denom = mask.sum(dim=1).clamp(min=1.0)

        pred_e = pred.mean(dim=1)
        clean_e = clean.mean(dim=1)
        err_e = error.mean(dim=1)
        abs_e = abs_error.mean(dim=1)

        def masked_mean(x):
            return (x * mask).sum(dim=1) / denom

        def masked_std(x, mean):
            return torch.sqrt((((x - mean[:, None]) ** 2) * mask).sum(dim=1) / denom + 1e-8)

        mean_abs = masked_mean(abs_e)
        std_abs = masked_std(abs_e, mean_abs)
        mean_err = masked_mean(err_e)
        std_err = masked_std(err_e, mean_err)
        mean_pred = masked_mean(pred_e)
        mean_clean = masked_mean(clean_e)

        rmse, p90, p95, max_abs, corr_pc, snr_db = [], [], [], [], [], []
        for b in range(B):
            L = int(lengths[b].item())
            ae = abs_error[b, :, :L].reshape(-1)
            er = error[b, :, :L].reshape(-1)
            pr = pred[b, :, :L].reshape(-1)
            cl = clean[b, :, :L].reshape(-1)
            rmse.append(torch.sqrt(torch.mean(er * er) + 1e-8))
            p90.append(torch.quantile(ae, 0.90))
            p95.append(torch.quantile(ae, 0.95))
            max_abs.append(torch.max(ae))
            if torch.std(pr) > 1e-8 and torch.std(cl) > 1e-8:
                corr_pc.append(torch.corrcoef(torch.stack([pr, cl]))[0, 1])
            else:
                corr_pc.append(torch.tensor(0.0, device=pred.device))
            snr = 10.0 * torch.log10((torch.sum(cl * cl) + 1e-8) / (torch.sum(er * er) + 1e-8))
            snr_db.append(snr)

        return torch.stack(
            [
                mean_abs,
                std_abs,
                mean_err,
                std_err,
                mean_pred,
                mean_clean,
                torch.stack(rmse),
                torch.stack(p90),
                torch.stack(p95),
                torch.stack(max_abs),
                torch.stack(corr_pc),
                torch.stack(snr_db),
            ],
            dim=1,
        )

    def forward(self, pred, clean, target_len, lengths):
        error = pred - clean
        abs_error = torch.abs(error)
        if self.conditioner_mode == "local_error":
            x = torch.cat([error, abs_error], dim=1)
        else:
            x = torch.cat([pred, clean, error, abs_error], dim=1)
        local = self.local_net(x)
        if local.shape[-1] != target_len:
            local = F.interpolate(local, size=target_len, mode="linear", align_corners=False)
        glob = self.global_mlp(self._global_feats(pred, clean, error, abs_error, lengths))
        if self.conditioner_mode == "local_error":
            glob = torch.zeros_like(glob)
        elif self.conditioner_mode == "global_error":
            local = torch.zeros_like(local)
        return self.out_scale * torch.tanh(local), self.out_scale * torch.tanh(glob)


class OraclePredCleanFiLMWhisper(nn.Module):
    def __init__(
        self,
        whisper_student: WhisperForConditionalGeneration,
        selected_layers: List[int],
        conditioner_hidden: int = 128,
        conditioner_scale: float = 0.20,
        adapter_bottleneck: int = 128,
        adapter_film_scale: float = 0.25,
        init_residual_scale: float = 0.002,
        adapter_mode: str = "residual_film",
        conditioner_mode: str = "full",
        injection_position: str = "after_layer",
    ):
        super().__init__()
        if injection_position not in {"after_layer", "before_layer"}:
            raise ValueError(f"Unsupported injection_position: {injection_position}")
        self.student = whisper_student
        self.selected_layers = sorted(selected_layers)
        self.injection_position = injection_position

        d_model = whisper_student.model.config.d_model
        n_mels = whisper_student.model.config.num_mel_bins
        self.conditioner = PredCleanConditioner(
            n_mels=n_mels,
            d_model=d_model,
            hidden=conditioner_hidden,
            out_scale=conditioner_scale,
            conditioner_mode=conditioner_mode,
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
        enc = self.student.model.encoder
        x = F.gelu(enc.conv1(pred))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)

        Tenc = x.shape[1]
        pos = enc.embed_positions.weight[:Tenc, :].to(x.dtype)
        x = x + pos[None, :, :]

        cond_local, cond_global = self.conditioner(pred, clean, Tenc, lengths)
        hidden_states = [x] if output_hidden_states else None
        adapter_debug = []

        for idx, layer in enumerate(enc.layers):
            if idx in self.selected_layers and self.injection_position == "before_layer":
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
            x = layer(x, attention_mask=None, output_attentions=False)[0]
            if idx in self.selected_layers and self.injection_position == "after_layer":
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

    def forward(
        self,
        pred,
        clean,
        lengths,
        labels,
        decoder_attention_mask,
        output_hidden_states: bool = True,
        output_logits: bool = True,
    ):
        enc = self.encode_student(pred, clean, lengths, output_hidden_states=output_hidden_states)
        out = self.student(
            encoder_outputs=enc["encoder_outputs"],
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        logits = None
        if output_logits:
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
            logits = out_for_kd.logits
        return {
            "loss": out.loss,
            "logits": logits,
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


@torch.no_grad()
def evaluate_oracle_model(
    model,
    teacher,
    processor,
    tokenizer,
    dataloader,
    device,
    selected_layers,
    show_progress: bool,
    condition_mode: str = "true",
    condition_source: str = "clean",
    input_source: str = "pred",
    eval_num_beams: int = 5,
    max_eval_batches: int = 0,
):
    model.eval()
    teacher.eval()
    refs, preds, losses = [], [], []
    hidden_kd_vals, logit_kd_vals = [], []
    update_vals, cond_local_vals, cond_global_vals = [], [], []

    desc = f"Eval FiLM source={condition_source} mode={condition_mode}"
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, disable=not show_progress)):
        if max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break
        input_mel = batch[input_source].to(device).float()
        clean = batch["clean"].to(device).float()
        condition = batch[condition_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        if condition_mode == "true":
            clean_cond = condition
        elif condition_mode == "zero":
            clean_cond = input_mel
        elif condition_mode == "shuffled":
            clean_cond = condition[torch.randperm(condition.shape[0], device=condition.device)]
        else:
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")

        student_out = model(pred=input_mel, clean=clean_cond, lengths=lengths, labels=labels, decoder_attention_mask=decoder_attention_mask)
        decoder_input_ids = shift_tokens_right(labels, model.student.config.pad_token_id, model.student.config.decoder_start_token_id)
        teacher_out = teacher(
            input_features=clean,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        lengths_enc = torch.tensor([expected_encoder_len(int(L.item())) for L in lengths], device=device, dtype=torch.long)
        hidden_kd = 0.0
        for layer_idx in selected_layers:
            hidden_kd = hidden_kd + masked_hidden_mse(
                student_out["hidden_states"][layer_idx + 1],
                teacher_out.encoder_hidden_states[layer_idx + 1],
                lengths_enc,
            )
        hidden_kd = hidden_kd / max(len(selected_layers), 1)
        logit_kd = masked_kl(student_out["logits"], teacher_out.logits, decoder_attention_mask, temperature=1.0)

        gen_ids, enc_dbg = model.generate(
            pred=input_mel,
            clean=clean_cond,
            lengths=lengths,
            num_beams=eval_num_beams,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(labels_text_ids, skip_special_tokens=True)

        refs.extend([clean_text(x) for x in ref_texts])
        preds.extend([clean_text(x) for x in pred_texts])
        losses.append(float(student_out["loss"].item()))
        hidden_kd_vals.append(float(hidden_kd.detach().cpu().item()))
        logit_kd_vals.append(float(logit_kd.detach().cpu().item()))
        cond_local_vals.append(float(enc_dbg["cond_local_abs"].detach().cpu().item()))
        cond_global_vals.append(float(enc_dbg["cond_global_abs"].detach().cpu().item()))
        if enc_dbg["adapter_debug"]:
            update_vals.append(float(np.mean([d["layer_update_abs"].detach().cpu().item() for d in enc_dbg["adapter_debug"]])))

    return {
        "loss": float(np.mean(losses)),
        "wer": float(wer(refs, preds)),
        "cer": float(cer(refs, preds)),
        "mean_hidden_kd": float(np.mean(hidden_kd_vals)),
        "mean_logit_kd": float(np.mean(logit_kd_vals)),
        "mean_layer_update_abs": float(np.mean(update_vals)) if update_vals else 0.0,
        "mean_cond_local_abs": float(np.mean(cond_local_vals)),
        "mean_cond_global_abs": float(np.mean(cond_global_vals)),
    }


def train_one_epoch_oracle(
    model,
    teacher,
    dataloader,
    optimizer,
    device,
    selected_layers,
    lambda_hidden_kd,
    lambda_logit_kd,
    lambda_input_logit_kd,
    lambda_adapter_update,
    show_progress: bool,
    condition_source: str = "clean",
    input_source: str = "pred",
    lambda_condition_contrast: float = 0.0,
    condition_contrast_margin: float = 0.02,
    lambda_zero_condition_contrast: float = 0.0,
    zero_condition_contrast_margin: float = 0.02,
    contrast_every_n_batches: int = 1,
    max_train_batches: int = 0,
):
    model.train()
    teacher.eval()
    total_loss = asr_loss = hidden_kd_loss = logit_kd_loss = input_logit_kd_loss = update_reg_loss = contrast_loss_total = 0.0
    true_asr_contrast_loss = 0.0
    shuffled_asr_loss = 0.0
    zero_asr_loss = 0.0
    zero_contrast_loss_total = 0.0
    contrast_n = 0
    zero_contrast_n = 0
    n = 0
    last_debug = None
    last_grad_norms = None

    desc = f"Training FiLM source={condition_source}"
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, disable=not show_progress)):
        if max_train_batches > 0 and batch_idx >= max_train_batches:
            break
        input_mel = batch[input_source].to(device).float()
        clean = batch["clean"].to(device).float()
        condition = batch[condition_source].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        need_hidden_kd = lambda_hidden_kd > 0.0
        need_logit_kd = lambda_logit_kd > 0.0
        need_input_logit_kd = lambda_input_logit_kd > 0.0
        need_teacher = need_hidden_kd or need_logit_kd
        student_out = model(
            pred=input_mel,
            clean=condition,
            lengths=lengths,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=need_hidden_kd,
            output_logits=need_logit_kd or need_input_logit_kd,
        )
        hidden_kd = torch.tensor(0.0, device=device)
        logit_kd = torch.tensor(0.0, device=device)
        input_logit_kd = torch.tensor(0.0, device=device)
        decoder_input_ids = None
        if need_teacher:
            decoder_input_ids = shift_tokens_right(labels, model.student.config.pad_token_id, model.student.config.decoder_start_token_id)
            with torch.no_grad():
                teacher_out = teacher(
                    input_features=clean,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=need_hidden_kd,
                    use_cache=False,
                    return_dict=True,
                )

            if need_hidden_kd:
                lengths_enc = torch.tensor([expected_encoder_len(int(L.item())) for L in lengths], device=device, dtype=torch.long)
                for layer_idx in selected_layers:
                    hidden_kd = hidden_kd + masked_hidden_mse(
                        student_out["hidden_states"][layer_idx + 1],
                        teacher_out.encoder_hidden_states[layer_idx + 1],
                        lengths_enc,
                    )
                hidden_kd = hidden_kd / max(len(selected_layers), 1)
            if need_logit_kd:
                logit_kd = masked_kl(student_out["logits"], teacher_out.logits, decoder_attention_mask, temperature=1.0)
        if need_input_logit_kd:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, model.student.config.pad_token_id, model.student.config.decoder_start_token_id)
            with torch.no_grad():
                input_teacher_out = teacher(
                    input_features=input_mel,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            input_logit_kd = masked_kl(student_out["logits"], input_teacher_out.logits, decoder_attention_mask, temperature=1.0)

        if student_out["adapter_debug"]:
            update_reg = torch.stack([d["layer_update_abs"] for d in student_out["adapter_debug"]]).mean()
        else:
            update_reg = torch.tensor(0.0, device=device)

        contrast_loss = torch.tensor(0.0, device=device)
        shuffled_loss = torch.tensor(0.0, device=device)
        zero_contrast_loss = torch.tensor(0.0, device=device)
        zero_loss = torch.tensor(0.0, device=device)
        do_extra_contrast = contrast_every_n_batches <= 1 or (batch_idx % contrast_every_n_batches == 0)
        if do_extra_contrast and lambda_condition_contrast > 0.0 and condition.shape[0] > 1:
            perm = torch.randperm(condition.shape[0], device=condition.device)
            if torch.equal(perm, torch.arange(condition.shape[0], device=condition.device)):
                perm = torch.roll(perm, shifts=1)
            shuffled_condition = condition[perm]
            shuffled_out = model(
                pred=input_mel,
                clean=shuffled_condition,
                lengths=lengths,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=False,
            )
            shuffled_loss = shuffled_out["loss"]
            contrast_loss = F.relu(student_out["loss"] - shuffled_loss + condition_contrast_margin)
            true_asr_contrast_loss += float(student_out["loss"].detach().cpu().item())
            contrast_n += 1
        if do_extra_contrast and lambda_zero_condition_contrast > 0.0:
            zero_out = model(
                pred=input_mel,
                clean=input_mel,
                lengths=lengths,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=False,
            )
            zero_loss = zero_out["loss"]
            zero_contrast_loss = F.relu(student_out["loss"] - zero_loss + zero_condition_contrast_margin)
            zero_asr_loss += float(zero_loss.detach().cpu().item())
            zero_contrast_n += 1

        loss = (
            student_out["loss"]
            + lambda_hidden_kd * hidden_kd
            + lambda_logit_kd * logit_kd
            + lambda_input_logit_kd * input_logit_kd
            + lambda_adapter_update * update_reg
            + lambda_condition_contrast * contrast_loss
            + lambda_zero_condition_contrast * zero_contrast_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norms = grad_norms(model)
        optimizer.step()

        total_loss += float(loss.item())
        asr_loss += float(student_out["loss"].item())
        hidden_kd_loss += float(hidden_kd.detach().cpu().item())
        logit_kd_loss += float(logit_kd.detach().cpu().item())
        input_logit_kd_loss += float(input_logit_kd.detach().cpu().item())
        update_reg_loss += float(update_reg.detach().cpu().item())
        contrast_loss_total += float(contrast_loss.detach().cpu().item())
        zero_contrast_loss_total += float(zero_contrast_loss.detach().cpu().item())
        shuffled_asr_loss += float(shuffled_loss.detach().cpu().item()) if lambda_condition_contrast > 0.0 and condition.shape[0] > 1 else 0.0
        n += 1

        last_debug = {
            "cond_local_abs": float(student_out["cond_local_abs"].detach().cpu().item()),
            "cond_global_abs": float(student_out["cond_global_abs"].detach().cpu().item()),
            "mean_layer_update_abs": float(update_reg.detach().cpu().item()),
            "adapter_layers": [
                {
                    "layer": int(d["layer"]),
                    "residual_scale": float(d["residual_scale"].detach().cpu().item()),
                    "gamma_abs": float(d["gamma_abs"].detach().cpu().item()),
                    "beta_abs": float(d["beta_abs"].detach().cpu().item()),
                    "delta_abs": float(d["delta_abs"].detach().cpu().item()),
                    "layer_update_abs": float(d["layer_update_abs"].detach().cpu().item()),
                }
                for d in student_out["adapter_debug"]
            ],
        }

    return {
        "train_total_loss": total_loss / max(n, 1),
        "train_asr_loss": asr_loss / max(n, 1),
        "train_hidden_kd_loss": hidden_kd_loss / max(n, 1),
        "train_logit_kd_loss": logit_kd_loss / max(n, 1),
        "train_input_logit_kd_loss": input_logit_kd_loss / max(n, 1),
        "train_update_reg": update_reg_loss / max(n, 1),
        "train_condition_contrast_loss": contrast_loss_total / max(n, 1),
        "train_zero_condition_contrast_loss": zero_contrast_loss_total / max(n, 1),
        "train_shuffled_asr_loss": shuffled_asr_loss / max(contrast_n, 1),
        "train_zero_asr_loss": zero_asr_loss / max(zero_contrast_n, 1),
        "train_true_minus_shuffled_asr_loss": (
            (true_asr_contrast_loss - shuffled_asr_loss) / max(contrast_n, 1) if contrast_n > 0 else 0.0
        ),
        "train_true_minus_zero_asr_loss": (
            (asr_loss - zero_asr_loss) / max(zero_contrast_n, 1) if zero_contrast_n > 0 else 0.0
        ),
        "last_grad_norms": last_grad_norms,
        "last_debug": last_debug,
    }


def main():
    parser = argparse.ArgumentParser(description="FiLM-conditioned Whisper experiment with clean or reverb conditioning source.")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--output-dir", type=str, default="/storage/tal/thesis/oracle_pred_clean_film")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--selected-layers", type=str, default="3,6,9,11")
    parser.add_argument("--last-n-unfrozen-encoder-layers", type=int, default=2)
    parser.add_argument("--conditioner-hidden", type=int, default=128)
    parser.add_argument("--conditioner-scale", type=float, default=0.20)
    parser.add_argument("--adapter-bottleneck", type=int, default=128)
    parser.add_argument("--adapter-film-scale", type=float, default=0.25)
    parser.add_argument("--init-residual-scale", type=float, default=0.002)
    parser.add_argument("--adapter-mode", choices=["residual_film", "direct_film", "gated_residual_film"], default="residual_film")
    parser.add_argument("--conditioner-mode", choices=["full", "local_error", "global_error"], default="full")
    parser.add_argument("--lambda-hidden-kd", type=float, default=0.50)
    parser.add_argument("--lambda-logit-kd", type=float, default=0.20)
    parser.add_argument("--lambda-input-logit-kd", type=float, default=0.0)
    parser.add_argument("--lambda-adapter-update", type=float, default=0.02)
    parser.add_argument("--lambda-condition-contrast", type=float, default=0.0)
    parser.add_argument("--condition-contrast-margin", type=float, default=0.02)
    parser.add_argument("--condition-contrast-warmup-epochs", type=int, default=0)
    parser.add_argument("--lambda-zero-condition-contrast", type=float, default=0.0)
    parser.add_argument("--zero-condition-contrast-margin", type=float, default=0.02)
    parser.add_argument("--contrast-every-n-batches", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--input-source", choices=["pred", "reverb", "clean"], default="pred")
    parser.add_argument("--condition-source", choices=["clean", "pred", "reverb"], default="clean")
    parser.add_argument("--eval-condition-ablations", action="store_true")
    parser.add_argument("--injection-position", choices=["after_layer", "before_layer"], default="after_layer")
    parser.add_argument("--eval-num-beams", type=int, default=5)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument("--eval-ablation-frequency", type=int, default=1)
    parser.add_argument("--skip-init-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
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

    model = OraclePredCleanFiLMWhisper(
        whisper_student=student_whisper,
        selected_layers=selected_layers,
        conditioner_hidden=args.conditioner_hidden,
        conditioner_scale=args.conditioner_scale,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_film_scale=args.adapter_film_scale,
        init_residual_scale=args.init_residual_scale,
        adapter_mode=args.adapter_mode,
        conditioner_mode=args.conditioner_mode,
        injection_position=args.injection_position,
    ).to(device)

    trainable = count_trainable_parameters(model)
    experiment_name = (
        "oracle_pred_clean_film"
        if args.input_source == "pred" and args.condition_source == "clean"
        else f"{args.input_source}_input_{args.condition_source}_cond_film"
    )
    print(
        json.dumps(
            {
                "trainable_params": trainable,
                "experiment": experiment_name,
                "input_source": args.input_source,
                "condition_source": args.condition_source,
                "adapter_mode": args.adapter_mode,
                "conditioner_mode": args.conditioner_mode,
                "injection_position": args.injection_position,
                "lambda_condition_contrast": args.lambda_condition_contrast,
                "lambda_input_logit_kd": args.lambda_input_logit_kd,
                "condition_contrast_margin": args.condition_contrast_margin,
                "condition_contrast_warmup_epochs": args.condition_contrast_warmup_epochs,
                "lambda_zero_condition_contrast": args.lambda_zero_condition_contrast,
                "zero_condition_contrast_margin": args.zero_condition_contrast_margin,
                "contrast_every_n_batches": args.contrast_every_n_batches,
                "max_train_batches": args.max_train_batches,
                "eval_num_beams": args.eval_num_beams,
                "max_eval_batches": args.max_eval_batches,
                "eval_ablation_frequency": args.eval_ablation_frequency,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    print("[PHASE] Evaluating validation baselines.", flush=True)
    baseline_cache: Dict[str, Dict[str, float]] = {}
    baseline_pred_val = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, val_dl, device, args.show_progress)
    baseline_cache["pred"] = baseline_pred_val
    if args.input_source in baseline_cache:
        baseline_input_val = baseline_cache[args.input_source]
    else:
        baseline_input_val = evaluate_baseline_source(
            baseline_whisper, processor, tokenizer, val_dl, device, args.input_source, args.show_progress
        )
        baseline_cache[args.input_source] = baseline_input_val
    if "clean" in baseline_cache:
        baseline_clean_val = baseline_cache["clean"]
    else:
        baseline_clean_val = evaluate_baseline_source(baseline_whisper, processor, tokenizer, val_dl, device, "clean", args.show_progress)
        baseline_cache["clean"] = baseline_clean_val
    init_val = None
    if not args.skip_init_eval:
        print("[PHASE] Evaluating initial FiLM model.", flush=True)
        init_val = evaluate_oracle_model(
            model,
            teacher_whisper,
            processor,
            tokenizer,
            val_dl,
            device,
            selected_layers,
            args.show_progress,
            condition_source=args.condition_source,
            input_source=args.input_source,
            eval_num_beams=args.eval_num_beams,
            max_eval_batches=args.max_eval_batches,
        )

    initial_summary = {
        "note": f"FiLM conditioning source: {args.condition_source}.",
        "pred_baseline_val": baseline_pred_val,
        "input_baseline_val": baseline_input_val,
        "clean_baseline_val": baseline_clean_val,
        f"{experiment_name}_init_val": init_val,
        "trainable_params": trainable,
        "selected_layers": selected_layers,
        "input_source": args.input_source,
        "condition_source": args.condition_source,
        "adapter_mode": args.adapter_mode,
        "conditioner_mode": args.conditioner_mode,
        "injection_position": args.injection_position,
        "lambda_input_logit_kd": args.lambda_input_logit_kd,
        "lambda_condition_contrast": args.lambda_condition_contrast,
        "condition_contrast_margin": args.condition_contrast_margin,
        "condition_contrast_warmup_epochs": args.condition_contrast_warmup_epochs,
        "lambda_zero_condition_contrast": args.lambda_zero_condition_contrast,
        "zero_condition_contrast_margin": args.zero_condition_contrast_margin,
        "contrast_every_n_batches": args.contrast_every_n_batches,
        "max_train_batches": args.max_train_batches,
        "eval_num_beams": args.eval_num_beams,
        "max_eval_batches": args.max_eval_batches,
        "eval_ablation_frequency": args.eval_ablation_frequency,
    }
    (out_dir / "initial_summary.json").write_text(json.dumps(initial_summary, indent=2), encoding="utf-8")

    if args.eval_only:
        eval_only_metrics = init_val
        if eval_only_metrics is None:
            print("[PHASE] Evaluating FiLM/wrapper model for eval-only run.", flush=True)
            eval_only_metrics = evaluate_oracle_model(
                model,
                teacher_whisper,
                processor,
                tokenizer,
                val_dl,
                device,
                selected_layers,
                args.show_progress,
                condition_source=args.condition_source,
                input_source=args.input_source,
                eval_num_beams=args.eval_num_beams,
                max_eval_batches=args.max_eval_batches,
            )
        eval_only_summary = {
            "initial": initial_summary,
            "eval_only_val": eval_only_metrics,
            "delta_vs_pred_baseline": float(eval_only_metrics["wer"] - baseline_pred_val["wer"]),
            "delta_vs_input_baseline": float(eval_only_metrics["wer"] - baseline_input_val["wer"]),
        }
        (out_dir / "eval_only_summary.json").write_text(json.dumps(eval_only_summary, indent=2), encoding="utf-8")
        print(json.dumps(eval_only_summary, ensure_ascii=False, separators=(",", ":")), flush=True)
        return

    history: List[Dict] = []
    best_metrics = None
    best_path = out_dir / "best_model.pt"
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"[PHASE] Starting epoch {epoch}/{args.epochs}.", flush=True)
        effective_lambda_condition_contrast = (
            args.lambda_condition_contrast
            if epoch > args.condition_contrast_warmup_epochs
            else 0.0
        )
        train_metrics = train_one_epoch_oracle(
            model=model,
            teacher=teacher_whisper,
            dataloader=train_dl,
            optimizer=optimizer,
            device=device,
            selected_layers=selected_layers,
            lambda_hidden_kd=args.lambda_hidden_kd,
            lambda_logit_kd=args.lambda_logit_kd,
            lambda_input_logit_kd=args.lambda_input_logit_kd,
            lambda_adapter_update=args.lambda_adapter_update,
            show_progress=args.show_progress,
            condition_source=args.condition_source,
            input_source=args.input_source,
            lambda_condition_contrast=effective_lambda_condition_contrast,
            condition_contrast_margin=args.condition_contrast_margin,
            lambda_zero_condition_contrast=args.lambda_zero_condition_contrast,
            zero_condition_contrast_margin=args.zero_condition_contrast_margin,
            contrast_every_n_batches=args.contrast_every_n_batches,
            max_train_batches=args.max_train_batches,
        )
        print(f"[PHASE] Evaluating epoch {epoch} true condition.", flush=True)
        val_metrics = evaluate_oracle_model(
            model,
            teacher_whisper,
            processor,
            tokenizer,
            val_dl,
            device,
            selected_layers,
            args.show_progress,
            condition_source=args.condition_source,
            input_source=args.input_source,
            eval_num_beams=args.eval_num_beams,
            max_eval_batches=args.max_eval_batches,
        )
        val_zero_metrics = None
        val_shuffled_metrics = None
        do_eval_ablations = args.eval_condition_ablations and (
            args.eval_ablation_frequency <= 1 or epoch % args.eval_ablation_frequency == 0
        )
        if do_eval_ablations:
            print(f"[PHASE] Evaluating epoch {epoch} zero/shuffled condition ablations.", flush=True)
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
                condition_source=args.condition_source,
                input_source=args.input_source,
                eval_num_beams=args.eval_num_beams,
                max_eval_batches=args.max_eval_batches,
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
                condition_source=args.condition_source,
                input_source=args.input_source,
                eval_num_beams=args.eval_num_beams,
                max_eval_batches=args.max_eval_batches,
            )

        row = {
            "epoch": epoch,
            "lr": args.lr,
            "selected_layers": selected_layers,
            "input_source": args.input_source,
            "condition_source": args.condition_source,
            "adapter_mode": args.adapter_mode,
            "conditioner_mode": args.conditioner_mode,
            "injection_position": args.injection_position,
            "lambda_hidden_kd": args.lambda_hidden_kd,
            "lambda_logit_kd": args.lambda_logit_kd,
            "lambda_input_logit_kd": args.lambda_input_logit_kd,
            "lambda_adapter_update": args.lambda_adapter_update,
            "lambda_condition_contrast": effective_lambda_condition_contrast,
            "condition_contrast_margin": args.condition_contrast_margin,
            "condition_contrast_warmup_epochs": args.condition_contrast_warmup_epochs,
            "lambda_zero_condition_contrast": args.lambda_zero_condition_contrast,
            "zero_condition_contrast_margin": args.zero_condition_contrast_margin,
            "contrast_every_n_batches": args.contrast_every_n_batches,
            "max_train_batches": args.max_train_batches,
            "eval_num_beams": args.eval_num_beams,
            "max_eval_batches": args.max_eval_batches,
            "eval_ablation_frequency": args.eval_ablation_frequency,
            **{k: v for k, v in train_metrics.items() if k not in {"last_grad_norms", "last_debug"}},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "delta_vs_pred_baseline_wer": float(val_metrics["wer"] - baseline_pred_val["wer"]),
            "delta_vs_input_baseline_wer": float(val_metrics["wer"] - baseline_input_val["wer"]),
            "last_grad_norms": train_metrics["last_grad_norms"],
            "last_debug": train_metrics["last_debug"],
        }
        if do_eval_ablations:
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

    print("[PHASE] Evaluating final validation and test summaries.", flush=True)
    final_val = evaluate_oracle_model(
        model,
        teacher_whisper,
        processor,
        tokenizer,
        val_dl,
        device,
        selected_layers,
        args.show_progress,
        condition_source=args.condition_source,
            input_source=args.input_source,
            eval_num_beams=args.eval_num_beams,
            max_eval_batches=args.max_eval_batches,
        )
    test_pred = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, test_dl, device, args.show_progress)
    test_input = evaluate_baseline_source(baseline_whisper, processor, tokenizer, test_dl, device, args.input_source, args.show_progress)
    test_clean = evaluate_baseline_source(baseline_whisper, processor, tokenizer, test_dl, device, "clean", args.show_progress)
    test_oracle = evaluate_oracle_model(
        model,
        teacher_whisper,
        processor,
        tokenizer,
        test_dl,
        device,
        selected_layers,
        args.show_progress,
        condition_source=args.condition_source,
        input_source=args.input_source,
        eval_num_beams=args.eval_num_beams,
        max_eval_batches=args.max_eval_batches,
    )

    final_summary = {
        "note": f"FiLM conditioning source: {args.condition_source}.",
        "initial": initial_summary,
        "final_val": final_val,
        "test_pred_baseline": test_pred,
        "test_input_baseline": test_input,
        "test_clean_baseline": test_clean,
        f"test_{experiment_name}": test_oracle,
        "best_val_metrics": best_metrics,
        "best_val_delta_vs_pred": float(best_metrics["wer"] - baseline_pred_val["wer"]) if best_metrics is not None else None,
        "test_delta_vs_pred": float(test_oracle["wer"] - test_pred["wer"]),
        "test_delta_vs_input": float(test_oracle["wer"] - test_input["wer"]),
        "input_source": args.input_source,
        "condition_source": args.condition_source,
        "conditioner_mode": args.conditioner_mode,
        "injection_position": args.injection_position,
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    (out_dir / "final_report.txt").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(f"[SAVE] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
