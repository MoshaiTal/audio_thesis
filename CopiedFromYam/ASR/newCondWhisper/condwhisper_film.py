#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import cer, wer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.modeling_whisper import shift_tokens_right


LEN_RE = re.compile(r"\[len=(\d+)\]")
TBINS_RE = re.compile(r"actual_tbins=(\d+)")


def infer_true_T(path_str: str) -> Optional[int]:
    m = LEN_RE.search(path_str)
    if m:
        return int(m.group(1))
    m = TBINS_RE.search(path_str)
    if m:
        return int(m.group(1)) + 1
    return None


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower())


def load_manifest(manifest_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class CondWhisperVNextDataset(Dataset):
    def __init__(self, manifest_path: str | Path, tokenizer: WhisperTokenizer):
        self.rows = load_manifest(Path(manifest_path))
        self.tok = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, path_str: str) -> Tuple[torch.Tensor, int]:
        arr = np.load(path_str).astype(np.float32)
        true_T = infer_true_T(path_str)
        if true_T is not None:
            arr = arr[:, :true_T]
        return torch.from_numpy(arr), arr.shape[1]

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        if "clean" not in row:
            raise KeyError("This script requires a 'clean' mel path in each manifest row.")

        pred, pred_T = self._load(row["pred"])
        lower, lower_T = self._load(row["lower"])
        upper, upper_T = self._load(row["upper"])
        clean, clean_T = self._load(row["clean"])

        T = min(pred_T, lower_T, upper_T, clean_T)
        pred = pred[:, :T]
        lower = lower[:, :T]
        upper = upper[:, :T]
        clean = clean[:, :T]
        width = torch.abs(upper - lower)

        text_path = Path(row["text"])
        text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else ""
        label_ids = self.tok(text).input_ids

        return {
            "stem": row.get("stem", Path(row["pred"]).stem),
            "text": text,
            "pred": pred,
            "lower": lower,
            "upper": upper,
            "width": width,
            "clean": clean,
            "length": T,
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
        }


class CondWhisperVNextCollator:
    def __init__(self, pad_token_id: int, max_T: int = 3000):
        self.pad_token_id = pad_token_id
        self.max_T = max_T

    def _pad_mel(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.max_T:
            x = x[:, : self.max_T]
        return F.pad(x, (0, self.max_T - x.shape[-1]))

    def __call__(self, batch: List[Dict]) -> Dict:
        pred = torch.stack([self._pad_mel(item["pred"]) for item in batch])
        lower = torch.stack([self._pad_mel(item["lower"]) for item in batch])
        upper = torch.stack([self._pad_mel(item["upper"]) for item in batch])
        width = torch.stack([self._pad_mel(item["width"]) for item in batch])
        clean = torch.stack([self._pad_mel(item["clean"]) for item in batch])

        label_ids = [item["label_ids"] for item in batch]
        labels_text_ids = pad_sequence(label_ids, batch_first=True, padding_value=self.pad_token_id)
        decoder_attention_mask = (labels_text_ids != self.pad_token_id).long()
        labels = labels_text_ids.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "stems": [item["stem"] for item in batch],
            "texts": [item["text"] for item in batch],
            "pred": pred,
            "lower": lower,
            "upper": upper,
            "width": width,
            "clean": clean,
            "lengths": torch.tensor([min(item["length"], self.max_T) for item in batch], dtype=torch.long),
            "labels": labels,
            "labels_text_ids": labels_text_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }


def make_time_mask(lengths: torch.Tensor, max_T: int) -> torch.Tensor:
    t = torch.arange(max_T, device=lengths.device)[None, :]
    return t < lengths[:, None]


def expected_encoder_len(T: int) -> int:
    return (T + 1) // 2


def inv_softplus(x: float) -> float:
    return float(math.log(math.expm1(x)))


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()
    return {"total": total}


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_last_encoder_layers(whisper: WhisperForConditionalGeneration, last_n: int) -> None:
    if last_n <= 0:
        return
    enc_layers = whisper.model.encoder.layers
    start = max(0, len(enc_layers) - last_n)
    for idx in range(start, len(enc_layers)):
        for p in enc_layers[idx].parameters():
            p.requires_grad = True
    for p in whisper.model.encoder.layer_norm.parameters():
        p.requires_grad = True


class CPConditioner(nn.Module):
    def __init__(self, n_mels: int, d_model: int, hidden: int = 128, out_scale: float = 0.20):
        super().__init__()
        self.out_scale = out_scale
        in_ch = n_mels * 4
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

    def _global_feats(self, pred, lower, upper, width, lengths):
        B, _, T = pred.shape
        mask = make_time_mask(lengths, T).float()
        denom = mask.sum(dim=1).clamp(min=1.0)

        pred_e = pred.mean(dim=1)
        width_e = width.mean(dim=1)
        width_flux = torch.abs(width_e[:, 1:] - width_e[:, :-1])
        width_flux = F.pad(width_flux, (1, 0))

        def masked_mean(x):
            return (x * mask).sum(dim=1) / denom

        def masked_std(x, mean):
            return torch.sqrt((((x - mean[:, None]) ** 2) * mask).sum(dim=1) / denom + 1e-8)

        mean_pred = masked_mean(pred_e)
        std_pred = masked_std(pred_e, mean_pred)
        mean_width = masked_mean(width_e)
        std_width = masked_std(width_e, mean_width)
        mean_flux = masked_mean(width_flux)
        std_flux = masked_std(width_flux, mean_flux)

        hi = int(round(0.70 * pred.shape[1]))
        width_hi = width[:, hi:, :] if hi < pred.shape[1] else width[:, -1:, :]

        hi_mean, hi_std, width_max, low_q, high_q, corr_pw = [], [], [], [], [], []
        for b in range(B):
            L = int(lengths[b].item())
            hv = width_hi[b, :, :L].reshape(-1)
            wv = width_e[b, :L]
            pv = pred_e[b, :L]
            hi_mean.append(hv.mean())
            hi_std.append(hv.std(unbiased=False))
            width_max.append(wv.max())
            low_q.append(torch.quantile(wv, 0.1))
            high_q.append(torch.quantile(wv, 0.9))
            if L > 1 and torch.std(wv) > 1e-8 and torch.std(pv) > 1e-8:
                corr_pw.append(torch.corrcoef(torch.stack([wv, pv]))[0, 1])
            else:
                corr_pw.append(torch.tensor(0.0, device=pred.device))

        return torch.stack(
            [
                mean_pred,
                std_pred,
                mean_width,
                std_width,
                mean_flux,
                std_flux,
                torch.stack(hi_mean),
                torch.stack(hi_std),
                torch.stack(width_max),
                torch.stack(low_q),
                torch.stack(high_q),
                torch.stack(corr_pw),
            ],
            dim=1,
        )

    def forward(self, pred, lower, upper, width, target_len, lengths):
        x = torch.cat([pred, lower, upper, width], dim=1)
        local = self.local_net(x)
        if local.shape[-1] != target_len:
            local = F.interpolate(local, size=target_len, mode="linear", align_corners=False)
        glob = self.global_mlp(self._global_feats(pred, lower, upper, width, lengths))
        return self.out_scale * torch.tanh(local), self.out_scale * torch.tanh(glob)


class CPFiLMAdapter(nn.Module):
    def __init__(
        self,
        d_model: int,
        bottleneck: int = 128,
        film_scale: float = 0.25,
        init_residual_scale: float = 0.002,
    ):
        super().__init__()
        self.film_scale = film_scale
        self.norm = nn.LayerNorm(d_model)
        self.gamma_net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        self.beta_net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        self.delta_net = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d_model),
        )
        for mod in [self.gamma_net[-1], self.beta_net[-1], self.delta_net[-1]]:
            nn.init.normal_(mod.weight, mean=0.0, std=7e-4)
            nn.init.constant_(mod.bias, 0.0)
        self.log_residual_scale = nn.Parameter(torch.tensor(inv_softplus(init_residual_scale), dtype=torch.float32))

    def forward(self, x, cp_local, cp_global):
        cond = cp_local.transpose(1, 2) + cp_global[:, None, :]
        z = self.norm(x)
        gamma = self.film_scale * torch.tanh(self.gamma_net(cond))
        beta = self.film_scale * torch.tanh(self.beta_net(cond))
        z_film = z * (1.0 + gamma) + beta
        delta = torch.tanh(self.delta_net(z_film))
        residual_scale = F.softplus(self.log_residual_scale)
        x_new = x + residual_scale * delta
        debug = {
            "residual_scale": residual_scale,
            "gamma_abs": gamma.abs().mean(),
            "beta_abs": beta.abs().mean(),
            "delta_abs": delta.abs().mean(),
            "layer_update_abs": (x_new - x).abs().mean(),
        }
        return x_new, debug


class CondWhisperVNext(nn.Module):
    def __init__(
        self,
        whisper_student: WhisperForConditionalGeneration,
        selected_layers: List[int],
        conditioner_hidden: int = 128,
        conditioner_scale: float = 0.20,
        adapter_bottleneck: int = 128,
        adapter_film_scale: float = 0.25,
        init_residual_scale: float = 0.002,
    ):
        super().__init__()
        self.student = whisper_student
        self.selected_layers = sorted(selected_layers)

        d_model = whisper_student.model.config.d_model
        n_mels = whisper_student.model.config.num_mel_bins

        self.conditioner = CPConditioner(
            n_mels=n_mels,
            d_model=d_model,
            hidden=conditioner_hidden,
            out_scale=conditioner_scale,
        )
        self.adapters = nn.ModuleDict({
            str(i): CPFiLMAdapter(
                d_model=d_model,
                bottleneck=adapter_bottleneck,
                film_scale=adapter_film_scale,
                init_residual_scale=init_residual_scale,
            )
            for i in self.selected_layers
        })

    def encode_student(self, pred, lower, upper, width, lengths, output_hidden_states: bool = True):
        enc = self.student.model.encoder
        x = F.gelu(enc.conv1(pred))
        x = F.gelu(enc.conv2(x))
        x = x.permute(0, 2, 1)

        Tenc = x.shape[1]
        pos = enc.embed_positions.weight[:Tenc, :].to(x.dtype)
        x = x + pos[None, :, :]

        lengths_enc = torch.tensor(
            [expected_encoder_len(int(L.item())) for L in lengths],
            device=lengths.device,
            dtype=torch.long,
        )
        cp_local, cp_global = self.conditioner(pred, lower, upper, width, Tenc, lengths)

        hidden_states = [x] if output_hidden_states else None
        adapter_debug = []

        for idx, layer in enumerate(enc.layers):
            x = layer(x, attention_mask=None, output_attentions=False)[0]
            if idx in self.selected_layers:
                x, dbg = self.adapters[str(idx)](x, cp_local, cp_global)
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
            "cp_local_abs": cp_local.abs().mean(),
            "cp_global_abs": cp_global.abs().mean(),
        }

    def forward(self, pred, lower, upper, width, lengths, labels, decoder_attention_mask):
        enc = self.encode_student(pred, lower, upper, width, lengths, output_hidden_states=True)
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
            "cp_local_abs": enc["cp_local_abs"],
            "cp_global_abs": enc["cp_global_abs"],
        }

    @torch.no_grad()
    def generate(self, pred, lower, upper, width, lengths, **kwargs):
        enc = self.encode_student(pred, lower, upper, width, lengths, output_hidden_states=False)
        gen_ids = self.student.generate(encoder_outputs=enc["encoder_outputs"], **kwargs)
        return gen_ids, enc


def masked_kl(student_logits, teacher_logits, mask, temperature: float = 1.0):
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(dim=-1)
    kl = kl * mask.float()
    denom = mask.float().sum().clamp(min=1.0)
    return (temperature ** 2) * kl.sum() / denom


def masked_hidden_mse(student_h, teacher_h, lengths_enc):
    mask = make_time_mask(lengths_enc, student_h.shape[1]).float()[:, :, None]
    diff = ((student_h - teacher_h) ** 2) * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def grad_norms(model: nn.Module) -> Dict[str, float]:
    vals = [p.grad.detach().norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
    return {"grad_norm_total_trainable": float(np.mean(vals)) if vals else 0.0}


@torch.no_grad()
def evaluate_baseline_pred(whisper, processor, tokenizer, dataloader, device, show_progress: bool):
    whisper.eval()
    refs, preds, losses = [], [], []
    for batch in tqdm(dataloader, desc="Eval baseline pred", disable=not show_progress):
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        pred = batch["pred"].to(device).float()
        out = whisper(
            input_features=pred,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        gen_ids = whisper.generate(
            input_features=pred,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_texts = tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True)
        refs.extend([clean_text(x) for x in ref_texts])
        preds.extend([clean_text(x) for x in pred_texts])
        losses.append(float(out.loss.item()))
    return {"loss": float(np.mean(losses)), "wer": float(wer(refs, preds)), "cer": float(cer(refs, preds))}


@torch.no_grad()
def evaluate_model(model, teacher, processor, tokenizer, dataloader, device, selected_layers, show_progress: bool):
    model.eval()
    teacher.eval()
    refs, preds, losses = [], [], []
    hidden_kd_vals, logit_kd_vals = [], []
    update_vals, cp_local_vals, cp_global_vals = [], [], []

    for batch in tqdm(dataloader, desc="Eval CondWhisper vNext", disable=not show_progress):
        pred = batch["pred"].to(device).float()
        lower = batch["lower"].to(device).float()
        upper = batch["upper"].to(device).float()
        width = batch["width"].to(device).float()
        clean = batch["clean"].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        labels_text_ids = batch["labels_text_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        student_out = model(
            pred=pred,
            lower=lower,
            upper=upper,
            width=width,
            lengths=lengths,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        decoder_input_ids = shift_tokens_right(
            labels,
            model.student.config.pad_token_id,
            model.student.config.decoder_start_token_id,
        )
        teacher_out = teacher(
            input_features=clean,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        lengths_enc = torch.tensor(
            [expected_encoder_len(int(L.item())) for L in lengths],
            device=device,
            dtype=torch.long,
        )

        hidden_kd = 0.0
        for layer_idx in selected_layers:
            hidden_kd = hidden_kd + masked_hidden_mse(
                student_out["hidden_states"][layer_idx + 1],
                teacher_out.encoder_hidden_states[layer_idx + 1],
                lengths_enc,
            )
        hidden_kd = hidden_kd / max(len(selected_layers), 1)

        logit_kd = masked_kl(
            student_out["logits"],
            teacher_out.logits,
            decoder_attention_mask,
            temperature=1.0,
        )

        gen_ids, enc_dbg = model.generate(
            pred=pred,
            lower=lower,
            upper=upper,
            width=width,
            lengths=lengths,
            num_beams=5,
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
        cp_local_vals.append(float(enc_dbg["cp_local_abs"].detach().cpu().item()))
        cp_global_vals.append(float(enc_dbg["cp_global_abs"].detach().cpu().item()))
        if enc_dbg["adapter_debug"]:
            update_vals.append(float(np.mean([d["layer_update_abs"].detach().cpu().item() for d in enc_dbg["adapter_debug"]])))

    return {
        "loss": float(np.mean(losses)),
        "wer": float(wer(refs, preds)),
        "cer": float(cer(refs, preds)),
        "mean_hidden_kd": float(np.mean(hidden_kd_vals)),
        "mean_logit_kd": float(np.mean(logit_kd_vals)),
        "mean_layer_update_abs": float(np.mean(update_vals)) if update_vals else 0.0,
        "mean_cp_local_abs": float(np.mean(cp_local_vals)),
        "mean_cp_global_abs": float(np.mean(cp_global_vals)),
    }


def is_better(candidate: Dict[str, float], best: Optional[Dict[str, float]], wer_eps: float = 1e-6, cer_eps: float = 1e-6) -> bool:
    if best is None:
        return True
    if candidate["wer"] < best["wer"] - wer_eps:
        return True
    if abs(candidate["wer"] - best["wer"]) <= wer_eps:
        if candidate["cer"] < best["cer"] - cer_eps:
            return True
        if abs(candidate["cer"] - best["cer"]) <= cer_eps and candidate["loss"] < best["loss"]:
            return True
    return False


def train_one_epoch(
    model,
    teacher,
    dataloader,
    optimizer,
    device,
    selected_layers,
    lambda_hidden_kd,
    lambda_logit_kd,
    lambda_adapter_update,
    show_progress: bool,
):
    model.train()
    teacher.eval()
    total_loss = asr_loss = hidden_kd_loss = logit_kd_loss = update_reg_loss = 0.0
    n = 0
    last_debug = None
    last_grad_norms = None

    for batch in tqdm(dataloader, desc="Training", disable=not show_progress):
        pred = batch["pred"].to(device).float()
        lower = batch["lower"].to(device).float()
        upper = batch["upper"].to(device).float()
        width = batch["width"].to(device).float()
        clean = batch["clean"].to(device).float()
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        student_out = model(
            pred=pred,
            lower=lower,
            upper=upper,
            width=width,
            lengths=lengths,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

        decoder_input_ids = shift_tokens_right(
            labels,
            model.student.config.pad_token_id,
            model.student.config.decoder_start_token_id,
        )
        with torch.no_grad():
            teacher_out = teacher(
                input_features=clean,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        lengths_enc = torch.tensor(
            [expected_encoder_len(int(L.item())) for L in lengths],
            device=device,
            dtype=torch.long,
        )

        hidden_kd = 0.0
        for layer_idx in selected_layers:
            hidden_kd = hidden_kd + masked_hidden_mse(
                student_out["hidden_states"][layer_idx + 1],
                teacher_out.encoder_hidden_states[layer_idx + 1],
                lengths_enc,
            )
        hidden_kd = hidden_kd / max(len(selected_layers), 1)

        logit_kd = masked_kl(student_out["logits"], teacher_out.logits, decoder_attention_mask, temperature=1.0)

        if student_out["adapter_debug"]:
            update_reg = torch.stack([d["layer_update_abs"] for d in student_out["adapter_debug"]]).mean()
        else:
            update_reg = torch.tensor(0.0, device=device)

        loss = (
            student_out["loss"]
            + lambda_hidden_kd * hidden_kd
            + lambda_logit_kd * logit_kd
            + lambda_adapter_update * update_reg
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norms = grad_norms(model)
        optimizer.step()

        total_loss += float(loss.item())
        asr_loss += float(student_out["loss"].item())
        hidden_kd_loss += float(hidden_kd.detach().cpu().item())
        logit_kd_loss += float(logit_kd.detach().cpu().item())
        update_reg_loss += float(update_reg.detach().cpu().item())
        n += 1

        last_debug = {
            "cp_local_abs": float(student_out["cp_local_abs"].detach().cpu().item()),
            "cp_global_abs": float(student_out["cp_global_abs"].detach().cpu().item()),
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
        "train_update_reg": update_reg_loss / max(n, 1),
        "last_grad_norms": last_grad_norms,
        "last_debug": last_debug,
    }


def main():
    parser = argparse.ArgumentParser(description="CondWhisper vNext: multi-layer CP-conditioned FiLM adapters + clean-teacher distillation.")
    parser.add_argument("--train-manifest", type=str, required=True)
    parser.add_argument("--val-manifest", type=str, required=True)
    parser.add_argument("--test-manifest", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="openai/whisper-small")
    parser.add_argument("--output-dir", type=str, default="/storage/tal/thesis/condwhisper_vnext")
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
    parser.add_argument("--lambda-hidden-kd", type=float, default=0.50)
    parser.add_argument("--lambda-logit-kd", type=float, default=0.20)
    parser.add_argument("--lambda-adapter-update", type=float, default=0.02)
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

    model = CondWhisperVNext(
        whisper_student=student_whisper,
        selected_layers=selected_layers,
        conditioner_hidden=args.conditioner_hidden,
        conditioner_scale=args.conditioner_scale,
        adapter_bottleneck=args.adapter_bottleneck,
        adapter_film_scale=args.adapter_film_scale,
        init_residual_scale=args.init_residual_scale,
    ).to(device)

    trainable = count_trainable_parameters(model)
    print(json.dumps({"trainable_params": trainable}, ensure_ascii=False, separators=(",", ":")), flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    baseline_pred_val = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, val_dl, device, args.show_progress)
    init_val = evaluate_model(model, teacher_whisper, processor, tokenizer, val_dl, device, selected_layers, args.show_progress)

    initial_summary = {
        "pred_baseline_val": baseline_pred_val,
        "condwhisper_vnext_init_val": init_val,
        "trainable_params": trainable,
        "selected_layers": selected_layers,
    }
    (out_dir / "initial_summary.json").write_text(json.dumps(initial_summary, indent=2), encoding="utf-8")

    history: List[Dict] = []
    best_metrics = None
    best_path = out_dir / "best_model.pt"
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
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
        )
        val_metrics = evaluate_model(model, teacher_whisper, processor, tokenizer, val_dl, device, selected_layers, args.show_progress)

        row = {
            "epoch": epoch,
            "lr": args.lr,
            "selected_layers": selected_layers,
            "lambda_hidden_kd": args.lambda_hidden_kd,
            "lambda_logit_kd": args.lambda_logit_kd,
            "lambda_adapter_update": args.lambda_adapter_update,
            **{k: v for k, v in train_metrics.items() if k not in {"last_grad_norms", "last_debug"}},
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "delta_vs_pred_baseline_wer": float(val_metrics["wer"] - baseline_pred_val["wer"]),
            "last_grad_norms": train_metrics["last_grad_norms"],
            "last_debug": train_metrics["last_debug"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, separators=(",", ":")), flush=True)
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

    final_val = evaluate_model(model, teacher_whisper, processor, tokenizer, val_dl, device, selected_layers, args.show_progress)
    test_pred = evaluate_baseline_pred(baseline_whisper, processor, tokenizer, test_dl, device, args.show_progress)
    test_cond = evaluate_model(model, teacher_whisper, processor, tokenizer, test_dl, device, selected_layers, args.show_progress)

    final_summary = {
        "initial": initial_summary,
        "final_val": final_val,
        "test_pred_baseline": test_pred,
        "test_condwhisper_vnext": test_cond,
        "best_val_metrics": best_metrics,
        "best_val_delta_vs_pred": float(best_metrics["wer"] - baseline_pred_val["wer"]) if best_metrics is not None else None,
    }
    (out_dir / "final_summary.json").write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    report_lines = [
        "Initial validation summary:",
        json.dumps(initial_summary, indent=2),
        "",
        "Final validation summary:",
        json.dumps(final_val, indent=2),
        "",
        "Test summaries:",
        json.dumps(
            {
                "test_pred_baseline": test_pred,
                "test_condwhisper_vnext": test_cond,
                "best_val_metrics": best_metrics,
                "best_val_delta_vs_pred": float(best_metrics["wer"] - baseline_pred_val["wer"]) if best_metrics is not None else None,
            },
            indent=2,
        ),
    ]
    (out_dir / "final_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[SAVE] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
