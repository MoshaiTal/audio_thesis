import os
import numpy as np
import torch
# import torch.nn as nn
from tqdm import tqdm
import wandb

from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.BACKBONE.MODEL_SECTION.utils import preprocess_batches, reconstruct_from_batch
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import save_checkpoint, save_best_weights
from CopiedFromYam.BACKBONE.MODEL_SECTION.losses import spearman, cosine_similarity_loss
from CopiedFromYam.BACKBONE.MODEL_SECTION.dereverb_asr_aware_losses import (
    ASRAwareDereverbLoss,
    DereverbLossWeights,
)

config = load_config()


def _sanitize_actual_pixels_mask(actual_pixels_mask, where="train", check=False):
    if actual_pixels_mask is None:
        raise RuntimeError(f"[MASK-SANITIZE][{where}] mask is None")

    if not torch.is_tensor(actual_pixels_mask):
        raise RuntimeError(
            f"[MASK-SANITIZE][{where}] mask is not a tensor, got {type(actual_pixels_mask)}"
        )

    actual_pixels_mask = actual_pixels_mask.float()

    if check:
        if not torch.isfinite(actual_pixels_mask).all():
            raise RuntimeError(f"[MASK-SANITIZE][{where}] mask contains NaN/Inf")
        min_v, max_v = torch.aminmax(actual_pixels_mask)
        if min_v.item() < 0.0 or max_v.item() > 1.0:
            actual_pixels_mask = (actual_pixels_mask > 0.5).float()

    return actual_pixels_mask


def _compute_loss_weights(mask: torch.Tensor, where: str = "") -> torch.Tensor:
    pre = mask.mean(dim=(1, 2)).float().clamp_min(0.0)
    return pre / pre.sum().clamp_min(1e-8)


def _build_asr_aware_head0_loss(device) -> ASRAwareDereverbLoss:
    """ASR-friendly loss for head 0, the dereverb point estimate."""
    weights = DereverbLossWeights(
        mel_l1=1.0,
        mel_time_delta=0.40,
        mel_freq_delta=0.10,
        mel_modulation=0.05,
        residual_to_reverb=0.02,
        mrstft=0.0,
        whisper_logmel=0.0,
    )
    return ASRAwareDereverbLoss(weights=weights, sample_rate=16000).to(device)


def _asr_aware_head0_forward(
    asr_aware_head0_loss: ASRAwareDereverbLoss,
    output: torch.Tensor,
    target: torch.Tensor,
    data: torch.Tensor,
    actual_pixels_mask: torch.Tensor,
):
    return asr_aware_head0_loss(
        pred_mel=output,
        clean_mel=target,
        reverb_mel=data,
        mask=actual_pixels_mask,
    )

def training_loop(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    best_val_loss,
    current_config,
    start_epoch=0
):
    # Keep DP intact outside; inside we operate on underlying module for head-freezing logic
    is_dp = isinstance(model, torch.nn.DataParallel)
    core_model = model.module if is_dp else model

    # -------------------------
    # Optional freezing logic (your original intent)
    # -------------------------
    if config['Model params'].get('split_model', False):
        # Enable gradients only for specified heads
        for i in range((len(core_model.heads) + 1) // 2):
            if current_config['current_step']['trained_heads'] != ['all'] and (i not in current_config['current_step']['trained_heads']):
                for param_l, param_u in zip(core_model.heads[i].parameters(), core_model.heads[-i].parameters()):
                    param_l.requires_grad = False
                    param_u.requires_grad = False
                core_model.heads[i].eval()
                core_model.heads[-i].eval()

        # Freeze encoder/shared decoder in "part"
        if current_config['current_step']['training_phase'][1] == "part":
            for param in core_model.encoder.parameters():
                param.requires_grad = False
            for param in core_model.shared_decoder.parameters():
                param.requires_grad = False
            core_model.encoder.eval()
            core_model.shared_decoder.eval()

        trainable_params = sum(p.numel() for p in core_model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

    # -------------------------
    # Epoch loop (RESUMABLE)
    # -------------------------
    for epoch in range(start_epoch, current_config['current_step']['num_epochs']):
        print(f"\nEpoch {epoch + 1} out of {current_config['current_step']['num_epochs']}")

        train_loss, model, train_total_loss = train(model, train_loader, optimizer, criterion, device, current_config)
        val_loss, _, _, _ = evaluate_individual_heads(model, test_loader, criterion, device, current_config)

        # ---- determine heads trained this phase ----
        if current_config['current_step']['trained_heads'] == ['all']:
            heads_to_train = np.arange(0, (len(core_model.heads) + 1) // 2, dtype=int)
        else:
            heads_to_train = np.array(current_config['current_step']['trained_heads'], dtype=int)

        heads_idx = np.unique(np.concatenate([heads_to_train, -heads_to_train]))
        print(f"Heads trained this phase: {heads_idx}")

        # ---- ALWAYS save LAST checkpoint (resume point) ----
        if config['General'].get('save_weights', True):
            save_checkpoint(model, optimizer, epoch, np.array(val_loss, copy=True), tag="last")

        # ---- Save BEST only if improved ----
        if val_loss[heads_idx].mean() < best_val_loss[heads_idx].mean():
            best_val_loss[heads_idx] = val_loss[heads_idx]
            if config['General'].get('save_weights', True):
                save_best_weights(model, optimizer, epoch, best_val_loss)

        # ---- W&B logging ----
        if config['logging'].get('activate_wandb', False):
            for i, loss in enumerate(val_loss):
                wandb.log({f"Head_{i}/val_loss": loss})
            wandb.log({"train/total_loss_epoch": train_total_loss, "epoch": epoch}, commit=True)

        print(
            f"Epoch {epoch + 1}/{current_config['current_step']['num_epochs']}, "
            f"Train Total Loss: {train_total_loss}, "
            f"Train Loss: {train_loss}, "
            f"Val Loss: {val_loss}\n"
        )

        # ---- Scheduler step ----
        if current_config['current_step']['scheduler_name'] != 'CosineAnnealingLR':
            scheduler.step(val_loss[heads_to_train].mean())
        else:
            scheduler.step()

        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")

    # -------------------------
    # Unfreeze/reset train states
    # -------------------------
    if config['Model params'].get('split_model', False):
        for i in range((len(core_model.heads) + 1) // 2):
            if current_config['current_step']['trained_heads'] != ['all'] and (i not in current_config['current_step']['trained_heads']):
                for param_l, param_u in zip(core_model.heads[i].parameters(), core_model.heads[-i].parameters()):
                    param_l.requires_grad = True
                    param_u.requires_grad = True
                core_model.heads[i].train()
                core_model.heads[-i].train()

        if current_config['current_step']['training_phase'][1] == "part":
            for param in core_model.encoder.parameters():
                param.requires_grad = True
            for param in core_model.shared_decoder.parameters():
                param.requires_grad = True
            core_model.encoder.train()
            core_model.shared_decoder.train()

    return model, best_val_loss


def train(model, train_loader, optimizer, criterion, device, current_config):
    is_dp = isinstance(model, torch.nn.DataParallel)
    core_model = model.module if is_dp else model
    core_model.train()
    asr_aware_head0_loss = _build_asr_aware_head0_loss(device)
    train._debug_printed = False
    train._crit_debug_printed = False

    # print("[DEBUG-CRIT-ALL]")
    # for i, c in enumerate(criterion):
    #     print(i, type(c), getattr(c, "__dict__", {}))
    #
    # print("[DEBUG-HEAD-CONFIGS]")
    # for i, hp in enumerate(config['Model params']['heads_params']):
    #     print(i, hp)

    running_losses = [0.0] * len(core_model.heads)
    running_total_loss = 0.0
    num_batches = 0

    for data, target, last_layer_per_rec_index, last_real_time_bin_per_rec_idx, actual_pixels_mask, _ in tqdm(
        train_loader, desc="Batch Progress", ascii=True
    ):
        data = preprocess_batches(data, last_real_time_bin_per_rec_idx).to(device)
        target = preprocess_batches(target, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = preprocess_batches(actual_pixels_mask, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = _sanitize_actual_pixels_mask(
            actual_pixels_mask,
            where="train",
            check=(num_batches % 200 == 0),
        )
        # print("[MASK-TYPE-AFTER-SANITIZE]", type(actual_pixels_mask))

        all_outputs = core_model(data, actual_pixels_mask)

        # if not train._crit_debug_printed:
        #     with torch.no_grad():
        #         raw0 = criterion[0](all_outputs[0] * actual_pixels_mask, target * actual_pixels_mask)
        #         print("[DEBUG-CRIT] criterion[0] type:", type(criterion[0]))
        #         print("[DEBUG-CRIT] raw0 min/max/mean:",
        #               raw0.min().item(), raw0.max().item(), raw0.mean().item())
        #
        #         tmp_loss0 = (raw0.mean(dim=(1, 2)) * (
        #                 actual_pixels_mask.mean(dim=(1, 2)) / actual_pixels_mask.mean(dim=(1, 2)).sum()
        #         )).mean()
        #         print("[DEBUG-CRIT] computed head0 loss:", tmp_loss0.item())
        #         print("[DEBUG-CRIT] global config head0:", config['Model params']['heads_params'][0])
        #     train._crit_debug_printed = True

        # if not train._debug_printed:
        #     with torch.no_grad():
        #         output0 = all_outputs[0]
        #
        #         def _stats(name, x):
        #             x = x.detach()
        #             print(
        #                 f"[DEBUG] {name}: "
        #                 f"shape={tuple(x.shape)}, "
        #                 f"min={x.min().item():.6f}, "
        #                 f"max={x.max().item():.6f}, "
        #                 f"mean={x.mean().item():.6f}, "
        #                 f"std={x.std().item():.6f}"
        #             )
        #
        #         _stats("data", data)
        #         _stats("target", target)
        #         _stats("mask", actual_pixels_mask)
        #         _stats("output0", output0)
        #         _stats("target*mask", target * actual_pixels_mask)
        #         _stats("output0*mask", output0 * actual_pixels_mask)
        #
        #         valid_ratio = actual_pixels_mask.mean().item()
        #         print(f"[DEBUG] valid mask ratio = {valid_ratio:.6f}")
        #
        #         masked_mae = torch.abs((output0 - target) * actual_pixels_mask).sum() / (
        #                 actual_pixels_mask.sum() + 1e-8
        #         )
        #         print(f"[DEBUG] masked MAE(output0,target) = {masked_mae.item():.6f}")
        #     train._debug_printed = True

        optimizer.zero_grad()
        loss_w_per_sample = _compute_loss_weights(actual_pixels_mask, where="train")

        if current_config['current_step']['trained_heads'] == ['all']:
            heads_to_train = range((len(core_model.heads) + 1) // 2)
        else:
            heads_to_train = current_config['current_step']['trained_heads']

        per_head_loss_tensors = {}
        total_loss = 0.0

        for head_idx in heads_to_train:
            if head_idx == 0:
                output = all_outputs[head_idx]
                loss, head0_loss_terms = _asr_aware_head0_forward(
                    asr_aware_head0_loss,
                    output=output,
                    target=target,
                    data=data,
                    actual_pixels_mask=actual_pixels_mask,
                )

            else:
                output = all_outputs[0]
                output_l, output_u = all_outputs[head_idx], all_outputs[-head_idx]

                if config['Model params']['heads_params'][head_idx]['loss'] == 'Residuals':
                    loss_quantile_l = (
                        criterion[head_idx](
                            output_l * actual_pixels_mask,
                            torch.minimum(
                                output * actual_pixels_mask + (target * actual_pixels_mask - output * actual_pixels_mask),
                                output * actual_pixels_mask
                            )
                        ).mean(dim=(1, 2)) * loss_w_per_sample
                    ).mean()

                    loss_quantile_u = (
                        criterion[-head_idx](
                            output_u * actual_pixels_mask,
                            torch.maximum(
                                output * actual_pixels_mask + (target * actual_pixels_mask - output * actual_pixels_mask),
                                output * actual_pixels_mask
                            )
                        ).mean(dim=(1, 2)) * loss_w_per_sample
                    ).mean()
                else:
                    loss_quantile_l = criterion[head_idx](
                        output_l * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample
                    )
                    loss_quantile_u = criterion[-head_idx](
                        output_u * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample
                    )

                loss = 0.5 * loss_quantile_u + 0.5 * loss_quantile_l

                if config['Model params']['heads_params'][head_idx].get('loss_corr', False):
                    interval = (output_u - output_l).reshape(output_u.shape[0], -1)
                    error = torch.abs(output - target).reshape(output_u.shape[0], -1)
                    loss_corr = spearman(interval * actual_pixels_mask.reshape(output_u.shape[0], -1),
                                        error * actual_pixels_mask.reshape(output_u.shape[0], -1),
                                        "l2", 0.01)
                    loss = loss + 0.5 * loss_corr

                if config['Model params']['heads_params'][head_idx].get('loss_div', False) and len(heads_to_train) > 1:
                    loss_div = cosine_similarity_loss(head_idx, all_outputs)
                    loss = loss + 0.1 * loss_div

            per_head_loss_tensors[head_idx] = loss
            total_loss = total_loss + loss
            running_losses[head_idx] += float(loss.detach().item())
            if head_idx != 0:
                running_losses[-head_idx] += float(loss.detach().item())

        if num_batches % 200 == 0 and not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total_loss: {float(total_loss.detach().item())}")

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_total_loss += float(total_loss.detach().item())
        num_batches += 1

        if config['logging'].get('activate_wandb', False):
            log_dict = {"train/total_loss_batch": float(total_loss.detach().item())}
            for k, v in per_head_loss_tensors.items():
                log_dict[f"train/head_{k}_loss_batch"] = float(v.detach().item())
            if 0 in per_head_loss_tensors and 'head0_loss_terms' in locals():
                for k, v in head0_loss_terms.items():
                    log_dict[f"train/head0_asr_aware/{k}"] = float(v.detach().item())
            wandb.log(log_dict, commit=False)

    avg_head_losses = np.array(running_losses) / max(num_batches, 1)
    avg_total_loss = running_total_loss / max(num_batches, 1)

    # print("[DEBUG-END-TRAIN] avg_total_loss:", avg_total_loss)
    # print("[DEBUG-END-TRAIN] avg_head_losses:", avg_head_losses)

    return avg_head_losses, model, avg_total_loss


def evaluate_individual_heads(model, data_loader, criterion, device, current_config):
    is_dp = isinstance(model, torch.nn.DataParallel)
    core_model = model.module if is_dp else model
    core_model.eval()
    asr_aware_head0_loss = _build_asr_aware_head0_loss(device)

    running_losses = [0.0] * len(core_model.heads)

    # Only reconstruct/save during explicit eval phase
    do_recon = (current_config['current_step']['training_phase'][0] == 'eval')

    all_model_outputs = [] if do_recon else None
    all_targets = [] if do_recon else None
    all_input = [] if do_recon else None

    for data, target, last_layer_per_rec_index, last_real_time_bin_per_rec_idx, actual_pixels_mask, paths_list in tqdm(
        data_loader, desc="Batch Progress", ascii=True
    ):
        data = preprocess_batches(data, last_real_time_bin_per_rec_idx).to(device)
        target = preprocess_batches(target, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = preprocess_batches(actual_pixels_mask, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = _sanitize_actual_pixels_mask(actual_pixels_mask, where="val")

        loss_w_per_sample = _compute_loss_weights(actual_pixels_mask, where="val")

        with torch.no_grad():

            # Tal's Addition:
            # --- FIX for inference tensor dataset batching ---
            # If someone accidentally swapped batch and channel dims, fix it:
            # expected: [B, 1, 256, 256]
            if data.ndim == 4 and data.shape[0] == 1 and data.shape[1] > 1:
                # looks like [1, B, 256, 256] -> convert to [B, 1, 256, 256]
                data = data.permute(1, 0, 2, 3).contiguous()

            # Also: if data came as [B, 256, 256] (missing channel), add channel
            if data.ndim == 3:
                data = data.unsqueeze(1)
            # -------------------------------------------------------------

            all_outputs = core_model(data, actual_pixels_mask)

        # Save outputs only in eval
        if do_recon and config['General']['save_output'][0]:
            for h_i, head in enumerate(all_outputs):
                # reconstructed = reconstruct_from_batch(head.cpu().detach().squeeze().numpy(),
                #     np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0))
                reconstructed = reconstruct_from_batch(
                    head.cpu().detach().squeeze().numpy(),
                    np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0),
                    mel_spec=config['General']['MEL_SPEC']
                )
                for layer_i in range(reconstructed.shape[0]):
                    if paths_list[0][layer_i] in ['cal', 'test']:
                        folder_path_to_save = os.path.join(
                            config['General']['save_output'][1],
                            config['General']['save_name'],
                            paths_list[0][layer_i],
                            paths_list[1][layer_i]
                        )
                        os.makedirs(folder_path_to_save, exist_ok=True)
                        file_path = os.path.join(folder_path_to_save, paths_list[2][layer_i].replace('.npy', f'(layer {h_i}|{len(all_outputs)}).npy'))
                        np.save(file_path, reconstructed[layer_i])

        if do_recon:
            # all_model_outputs.append(
            #     np.array([reconstruct_from_batch(head.cpu().detach().squeeze().numpy(),np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0))
            #         for head in all_outputs]).transpose((1, 0, 2, 3)))
            all_model_outputs.append(
                np.array([reconstruct_from_batch(head.cpu().detach().squeeze().numpy(), np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0), mel_spec=config['General']['MEL_SPEC'])
                    for head in all_outputs]).transpose((1, 0, 2, 3)))
            all_targets.append(
                reconstruct_from_batch(target.cpu().detach().squeeze().numpy(),
                    np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0), mel_spec=config['General']['MEL_SPEC']))
            all_input.append(
                reconstruct_from_batch(data.cpu().detach().squeeze().numpy(),
                    np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0), mel_spec=config['General']['MEL_SPEC']))

        # Losses
        for output_idx in range(len(config['Model params']['heads_params'])):
            if output_idx == 0:
                output = all_outputs[output_idx]
                loss, head0_loss_terms = _asr_aware_head0_forward(
                    asr_aware_head0_loss,
                    output=output,
                    target=target,
                    data=data,
                    actual_pixels_mask=actual_pixels_mask,
                )
            else:
                output = all_outputs[0]
                output_l, output_u = all_outputs[output_idx], all_outputs[-output_idx]

                if config['Model params']['heads_params'][output_idx]['loss'] == 'Residuals':
                    loss_quantile_l = (
                        criterion[output_idx](output_l, torch.minimum(output + (target - output), output)).mean(dim=(1, 2)) * loss_w_per_sample).mean()

                    loss_quantile_u = (
                        criterion[-output_idx](
                            output_u,torch.maximum(output + (target - output), output)).mean(dim=(1, 2)) * loss_w_per_sample).mean()
                else:
                    loss_quantile_l = criterion[output_idx](
                        output_l * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample
                    )
                    loss_quantile_u = criterion[-output_idx](
                        output_u * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample
                    )

                loss = 0.5 * loss_quantile_u + 0.5 * loss_quantile_l

                if config['Model params']['heads_params'][output_idx].get('loss_corr', False):
                    interval = (output_u - output_l).reshape(output_u.shape[0], -1)
                    error = torch.abs(output - target).reshape(output_u.shape[0], -1)
                    loss_corr = spearman(interval, error, "l2", 0.01)
                    loss += 0.5 * loss_corr

                if config['Model params']['heads_params'][output_idx].get('loss_div', False) and len(config['Model params']['heads_params']) > 3:
                    loss_div = cosine_similarity_loss(output_idx, all_outputs)
                    loss += 0.1 * loss_div

            running_losses[output_idx] += float(loss.item())
            if output_idx != 0:
                running_losses[-output_idx] += float(loss.item())

    # running_losses = 10 * np.array(running_losses) / len(data_loader)
    running_losses = np.array(running_losses) / len(data_loader)

    if do_recon:
        all_model_outputs = np.concatenate(all_model_outputs)
        all_targets = np.concatenate(all_targets)
        all_input = np.concatenate(all_input)
    else:
        all_model_outputs, all_targets, all_input = None, None, None

    # print("[DEBUG-END-VAL] running_losses avg:", running_losses)

    return running_losses, all_model_outputs, all_targets, all_input
