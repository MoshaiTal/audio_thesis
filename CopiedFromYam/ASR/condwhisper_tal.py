from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    get_linear_schedule_with_warmup,
)

from CopiedFromYam.ASR.utils import build_paired_file_rows
from CopiedFromYam.ASR.newCondWhisper.cpcondwhisper_blocks import CPCondWhisperModel


LEN_RE = re.compile(r"\[len=(\d+)\]")
TBINS_RE = re.compile(r"actual_tbins=(\d+)")


def infer_true_T(path_str: str) -> int:
    m = LEN_RE.search(path_str)
    if m:
        return int(m.group(1))
    m = TBINS_RE.search(path_str)
    if m:
        return int(m.group(1)) + 1
    raise ValueError(f"Could not infer true length from path: {path_str}")


def clean_text(text: str) -> str:
    return re.sub(r"[^a-z ]", "", text.lower()).strip()


def create_manifest_rows(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


class CPMelDataset(Dataset):
    """Loads a mel input plus conditioning features.

    conditioning_mode="cp" uses inference-available CP interval features.
    conditioning_mode="pred_clean" is an oracle experiment that uses clean,
    pred-clean, and abs(pred-clean) as the conditioner input.
    """

    def __init__(
        self,
        manifest: str | Path,
        tokenizer: WhisperTokenizer,
        input_key: str = "pred",
        clean_targets: bool = True,
        log_cp: bool = True,
        conditioning_mode: str = "cp",
    ):
        self.rows = [json.loads(l) for l in Path(manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tok = tokenizer
        self.input_key = input_key
        self.clean_targets = clean_targets
        self.log_cp = log_cp
        self.conditioning_mode = conditioning_mode
        if conditioning_mode not in {"cp", "pred_clean"}:
            raise ValueError(f"Unsupported conditioning_mode: {conditioning_mode}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        r = self.rows[idx]
        true_T = infer_true_T(r[self.input_key])

        pred = np.load(r[self.input_key]).astype(np.float32)[:, :true_T]
        lower = np.load(r["lower"]).astype(np.float32)[:, :true_T]
        upper = np.load(r["upper"]).astype(np.float32)[:, :true_T]

        if self.conditioning_mode == "cp":
            width = np.abs(upper - lower).astype(np.float32)
            log_width = np.log1p(width).astype(np.float32) if self.log_cp else width
            lower_delta = (pred - lower).astype(np.float32)
            upper_delta = (upper - pred).astype(np.float32)
            cond_features = np.concatenate(
                [lower, upper, width, log_width, lower_delta, upper_delta],
                axis=0,
            ).astype(np.float32)
        else:
            clean = np.load(r["clean"]).astype(np.float32)[:, :true_T]
            T = min(pred.shape[1], clean.shape[1])
            pred = pred[:, :T]
            clean = clean[:, :T]
            signed_error = (pred - clean).astype(np.float32)
            abs_error = np.abs(signed_error).astype(np.float32)
            cond_features = np.concatenate(
                [clean, signed_error, abs_error],
                axis=0,
            ).astype(np.float32)

        with open(r["text"], "r", encoding="utf-8") as f:
            text = f.read().strip()
        target_text = clean_text(text) if self.clean_targets else text

        label_ids = self.tok(target_text).input_ids
        return {
            "pred": torch.from_numpy(pred),
            "cp_features": torch.from_numpy(cond_features),
            "length": torch.tensor(pred.shape[1], dtype=torch.long),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            "text": text,
            "target_text": target_text,
            "stem": r.get("stem", Path(r["pred"]).stem),
        }


class CPMelCollator:
    def __init__(self, tokenizer: WhisperTokenizer, max_T: int = 3000):
        self.pad_id = tokenizer.pad_token_id
        self.max_T = max_T

    def _pad_or_crop(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.max_T:
            return x[:, : self.max_T]
        return F.pad(x, (0, self.max_T - x.shape[-1]))

    def __call__(self, batch: List[Dict]) -> Dict:
        pred = torch.stack([self._pad_or_crop(item["pred"]) for item in batch])
        cp_features = torch.stack([self._pad_or_crop(item["cp_features"]) for item in batch])
        input_lengths = torch.tensor(
            [min(int(item["length"].item()), self.max_T) for item in batch],
            dtype=torch.long,
        )

        label_ids = [item["label_ids"] for item in batch]
        label_lengths = [len(x) for x in label_ids]
        labels = pad_sequence(label_ids, batch_first=True, padding_value=self.pad_id)
        decoder_attention_mask = torch.zeros_like(labels)
        for i, L in enumerate(label_lengths):
            decoder_attention_mask[i, :L] = 1

        labels_for_loss = labels.clone()
        for i, L in enumerate(label_lengths):
            if L < labels_for_loss.shape[1]:
                labels_for_loss[i, L:] = -100

        return {
            "input_features": pred,
            "pred": pred,
            "cp_features": cp_features,
            "lengths": input_lengths,
            "labels": labels_for_loss,
            "labels_text_ids": labels,
            "decoder_attention_mask": decoder_attention_mask,
            "texts": [item["text"] for item in batch],
            "target_texts": [item["target_text"] for item in batch],
            "stems": [item["stem"] for item in batch],
        }


def latent_delta_penalty(base_hidden: torch.Tensor, refined_hidden: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
    mask = enc_mask.unsqueeze(-1).float()
    denom = mask.sum().clamp_min(1.0)
    return (((refined_hidden - base_hidden) ** 2) * mask).sum() / denom


def zero_uncertainty_features(pred: torch.Tensor) -> torch.Tensor:
    """Build CP features representing a zero-width interval centered on pred."""
    zeros = torch.zeros_like(pred)
    return torch.cat([pred, pred, zeros, zeros, zeros, zeros], dim=1)


def zero_conditioning_features(pred: torch.Tensor, cond_feature_groups: int) -> torch.Tensor:
    return torch.zeros(
        pred.shape[0],
        pred.shape[1] * cond_feature_groups,
        pred.shape[2],
        dtype=pred.dtype,
        device=pred.device,
    )


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    processor,
    tokenizer,
    device,
    cp_mode: str = "true",
    cond_feature_groups: int = 6,
) -> Tuple[float, float, List[str], List[str]]:
    model.eval()
    total_loss = 0.0
    preds: List[str] = []
    refs: List[str] = []

    for batch in tqdm(dataloader, desc=f"Evaluating ({cp_mode})"):
        pred = batch["pred"].to(device).float()
        cp_features = batch["cp_features"].to(device).float()
        input_lengths = batch["lengths"].to(device).long()

        if cp_mode == "zero":
            if cond_feature_groups == 6:
                cp_features = zero_uncertainty_features(pred)
            else:
                cp_features = zero_conditioning_features(pred, cond_feature_groups)
        elif cp_mode == "shuffled":
            perm = torch.randperm(cp_features.shape[0], device=cp_features.device)
            cp_features = cp_features[perm]

        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(
            pred=pred,
            cp_features=cp_features,
            input_lengths=input_lengths,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        total_loss += out["loss"].item()

        gen_ids = model.generate(
            pred=pred,
            cp_features=cp_features,
            input_lengths=input_lengths,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_strs = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True)

        preds += [clean_text(t) for t in pred_strs]
        refs += [clean_text(t) for t in ref_strs]

    return total_loss / max(len(dataloader), 1), wer(refs, preds), refs, preds


@torch.no_grad()
def evaluate_vanilla_on_pred(dataloader, processor, tokenizer, device, model_name: str, language: str, task: str) -> Tuple[float, float]:
    vanilla = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    vanilla.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    vanilla.config.forced_decoder_ids = forced_decoder_ids
    vanilla.generation_config.forced_decoder_ids = forced_decoder_ids

    total_loss = 0.0
    preds: List[str] = []
    refs: List[str] = []

    for batch in tqdm(dataloader, desc="Evaluating vanilla"):
        pred = batch["input_features"].to(device).float()
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = vanilla(
            input_features=pred,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
        )
        total_loss += out.loss.item()

        gen_ids = vanilla.generate(
            input_features=pred,
            num_beams=5,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        pred_strs = processor.batch_decode(gen_ids, skip_special_tokens=True)
        ref_strs = tokenizer.batch_decode(batch["labels_text_ids"], skip_special_tokens=True)

        preds += [clean_text(t) for t in pred_strs]
        refs += [clean_text(t) for t in ref_strs]

    return total_loss / max(len(dataloader), 1), wer(refs, preds)


def print_grad_debug(model) -> None:
    print("[GRAD DEBUG] mean abs gradients for trainable parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            grad = None if p.grad is None else float(p.grad.detach().abs().mean().item())
            print(f"  {name}: {grad}")


def condition_only_parameters(model):
    """Train only paths that require the external condition.

    The frozen Whisper encoder already supplies the pred representation. For
    the oracle experiment we want to test whether clean-pred information helps,
    not whether a large unconditional latent adapter can fit the training set.
    """
    for p in model.latent_blocks.parameters():
        p.requires_grad = False

    params = list(model.conditioner.parameters())
    for block in model.latent_blocks:
        for module in (block.global_to_cond, block.film_gamma, block.film_beta, block.cond_delta):
            for p in module.parameters():
                p.requires_grad = True
                params.append(p)
        block.cond_gate.requires_grad = True
        params.append(block.cond_gate)
    return params


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    grad_clip: float = 1.0,
    lambda_delta: float = 0.0,
    debug_first_batch: bool = False,
) -> float:
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        pred = batch["pred"].to(device).float()
        cp_features = batch["cp_features"].to(device).float()
        input_lengths = batch["lengths"].to(device).long()
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        out = model(
            pred=pred,
            cp_features=cp_features,
            input_lengths=input_lengths,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        loss = out["loss"]
        if lambda_delta > 0:
            delta_loss = latent_delta_penalty(out["base_hidden"], out["refined_hidden"], out["enc_mask"])
            loss = loss + lambda_delta * delta_loss

        loss.backward()
        if debug_first_batch and step == 0:
            print_grad_debug(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


def create_manifests_if_missing(
    splits_to_use,
    manifest_dir: Path,
    manifest_name: str,
    rebuild_manifests: bool,
    output_root: Path,
    clean_root: Path,
    reverb_root: Path,
    targets_root: Path,
    wanted_ch: str,
    wanted_alpha: str,
) -> None:
    for split in splits_to_use:
        out_manifest = manifest_dir / f"{split}_{manifest_name}"
        if out_manifest.exists() and not rebuild_manifests:
            print(f"[MANIFEST] Keeping existing {out_manifest}")
            continue

        rows_all = []
        split_output_root = output_root / split
        split_clean_root = clean_root / split
        split_reverb_root = reverb_root / split
        split_targets_root = targets_root / split

        if not split_output_root.exists():
            print(f"[WARN] Missing split output root: {split_output_root}")
            continue

        folders = sorted([p.name for p in split_output_root.iterdir() if p.is_dir()])
        for folder in folders:
            rows = build_paired_file_rows(
                output_folder_path=str(split_output_root / folder),
                clean_folder_path=str(split_clean_root / folder),
                reverb_folder_path=str(split_reverb_root / folder),
                targets_folder_path=str(split_targets_root / folder),
                WANTED_CH=wanted_ch,
                WANTED_ALPHA=wanted_alpha,
            )
            rows_all.extend(rows)

        print(f"[MANIFEST] split={split}: paired {len(rows_all)} rows")
        create_manifest_rows(rows_all, out_manifest)


if __name__ == "__main__":
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)

    targets_root = Path("/storage/tal/thesis/DataBase_BIUREV/transcription_matched")
    clean_root = Path("/storage/tal/thesis/DataBase_BIUREV/clean_melspec")
    reverb_root = Path("/storage/tal/thesis/DataBase_BIUREV/reverb_melspec")
    output_root = Path("/storage/tal/thesis/DataBase_BIUREV/dereverb_mel/calibrated_rcps")


    # manifest_dir = Path("/storage/tal/thesis/condwhisper_manifests")
    # manifest_name = "cpwidth.jsonl"
    manifest_dir = Path("/storage/tal/thesis/condwhisper_manifests_calibrated")
    manifest_name = "cpwidth_ols.jsonl"

    manifest_dir.mkdir(parents=True, exist_ok=True)
    splits_to_use = ["train", "val", "test"]
    rebuild_manifests = False

    wanted_ch = "ch1"
    wanted_alpha = "0.2"

    model_name = "openai/whisper-small"
    language = "en"
    task = "transcribe"
    batch_size = 8
    n_epochs = 20
    patience = 6
    run_cp_ablations_every_epoch = True
    conditioning_mode = "pred_clean"
    cond_feature_groups = 3 if conditioning_mode == "pred_clean" else 6
    # The Whisper encoder/decoder are frozen, so the conditioning stack must
    # stay close to the vanilla encoder manifold. Without this penalty the
    # latent adapters can destroy generation after a single epoch, even when
    # the conditioning features are zeroed/shuffled.
    lambda_delta = 0.01

    create_manifests_if_missing(
        splits_to_use=splits_to_use,
        manifest_dir=manifest_dir,
        manifest_name=manifest_name,
        rebuild_manifests=rebuild_manifests,
        output_root=output_root,
        clean_root=clean_root,
        reverb_root=reverb_root,
        targets_root=targets_root,
        wanted_ch=wanted_ch,
        wanted_alpha=wanted_alpha,
    )

    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CPMelDataset(manifest_dir / f"train_{manifest_name}", tokenizer, conditioning_mode=conditioning_mode)
    val_ds = CPMelDataset(manifest_dir / f"val_{manifest_name}", tokenizer, conditioning_mode=conditioning_mode)
    val_clean_ds = CPMelDataset(manifest_dir / f"val_{manifest_name}", tokenizer, input_key="clean", conditioning_mode=conditioning_mode)
    val_reverb_ds = CPMelDataset(manifest_dir / f"val_{manifest_name}", tokenizer, input_key="reverb", conditioning_mode=conditioning_mode)
    test_ds = CPMelDataset(manifest_dir / f"test_{manifest_name}", tokenizer, conditioning_mode=conditioning_mode)
    collator = CPMelCollator(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    val_clean_loader = DataLoader(val_clean_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    val_reverb_loader = DataLoader(val_reverb_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model = CPCondWhisperModel(
        model_name=model_name,
        mel_bins=80,
        d_cond=128,
        cond_feature_groups=cond_feature_groups,
        include_pred_in_conditioner=(conditioning_mode == "cp"),
        num_latent_blocks=4,
        num_heads=8,
        freeze_whisper=True,
    ).to(device)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
    model.base.config.forced_decoder_ids = forced_decoder_ids
    model.base.generation_config.forced_decoder_ids = forced_decoder_ids

    train_params = condition_only_parameters(model) if conditioning_mode == "pred_clean" else list(model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Using fixed latent Condformer-style conditioning_mode={conditioning_mode}")
    print(f"Trainable params: {trainable}/{total}")

    init_val_loss, init_val_wer, _, _ = evaluate_model(
        model, val_loader, processor, tokenizer, device, cp_mode="true", cond_feature_groups=cond_feature_groups
    )
    init_zero_val_loss, init_zero_val_wer, _, _ = evaluate_model(
        model, val_loader, processor, tokenizer, device, cp_mode="zero", cond_feature_groups=cond_feature_groups
    )
    _, init_shuf_wer, _, _ = evaluate_model(
        model, val_loader, processor, tokenizer, device, cp_mode="shuffled", cond_feature_groups=cond_feature_groups
    )
    print({
        "train_loss": None,
        "val_loss": init_val_loss,
        "val_wer": f"{init_val_wer:.3%}",
        "val_zero_cond_loss": init_zero_val_loss,
        "val_zero_cond_wer": f"{init_zero_val_wer:.3%}",
        "val_shuffled_cond_wer": f"{init_shuf_wer:.3%}",
        "epoch": 0,
    })

    optimizer = torch.optim.AdamW(
        [{"params": train_params, "lr": 1e-4 if conditioning_mode == "pred_clean" else 3e-5}],
        weight_decay=0.0,
    )
    total_steps = n_epochs * max(len(train_loader), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(20, int(0.02 * total_steps)),
        num_training_steps=total_steps,
    )

    best_val_wer = init_val_wer
    epochs_without_improve = 0
    best_path = Path(f"/storage/tal/thesis/weights/condwhisper_{conditioning_mode}_fixed_best.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"model": model.state_dict(), "val_wer": best_val_wer, "epoch": 0}, best_path)
    print(f"[SAVE] Epoch-0 baseline saved to {best_path}")

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            lambda_delta=lambda_delta,
            debug_first_batch=(epoch == 0),
        )
        val_loss, val_wer, _, _ = evaluate_model(
            model, val_loader, processor, tokenizer, device, cp_mode="true", cond_feature_groups=cond_feature_groups
        )

        row = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_wer": f"{val_wer:.3%}",
            "epoch": epoch + 1,
        }

        if run_cp_ablations_every_epoch:
            _, val_zero_wer, _, _ = evaluate_model(
                model, val_loader, processor, tokenizer, device, cp_mode="zero", cond_feature_groups=cond_feature_groups
            )
            _, val_shuf_wer, _, _ = evaluate_model(
                model, val_loader, processor, tokenizer, device, cp_mode="shuffled", cond_feature_groups=cond_feature_groups
            )
            row["val_zero_cond_wer"] = f"{val_zero_wer:.3%}"
            row["val_shuffled_cond_wer"] = f"{val_shuf_wer:.3%}"

        print(row)

        if val_wer < best_val_wer:
            best_val_wer = val_wer
            epochs_without_improve = 0
            torch.save({"model": model.state_dict(), "val_wer": best_val_wer, "epoch": epoch + 1}, best_path)
            print(f"[SAVE] Best model saved to {best_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"[EARLY STOP] Validation WER has not improved for {epochs_without_improve} epochs.")
                break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[LOAD] Loaded best checkpoint from {best_path} (val_wer={ckpt['val_wer']:.3%})")

    test_loss, test_wer, _, _ = evaluate_model(
        model, test_loader, processor, tokenizer, device, cp_mode="true", cond_feature_groups=cond_feature_groups
    )
    zero_test_loss, zero_test_wer, _, _ = evaluate_model(
        model, test_loader, processor, tokenizer, device, cp_mode="zero", cond_feature_groups=cond_feature_groups
    )
    shuf_test_loss, shuf_test_wer, _, _ = evaluate_model(
        model, test_loader, processor, tokenizer, device, cp_mode="shuffled", cond_feature_groups=cond_feature_groups
    )
    print({
        "test_loss": test_loss,
        "test_wer": f"{test_wer:.3%}",
        "test_zero_cond_loss": zero_test_loss,
        "test_zero_cond_wer": f"{zero_test_wer:.3%}",
        "test_shuffled_cond_loss": shuf_test_loss,
        "test_shuffled_cond_wer": f"{shuf_test_wer:.3%}",
    })

    vanilla_test_loss, vanilla_test_wer = evaluate_vanilla_on_pred(
        test_loader, processor, tokenizer, device, model_name, language, task
    )
    print({
        "vanilla_pred_only_test_loss": vanilla_test_loss,
        "vanilla_pred_only_test_wer": f"{vanilla_test_wer:.3%}",
    })
