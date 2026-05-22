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

config = load_config()


def _sanitize_actual_pixels_mask(actual_pixels_mask, where="train"):
    if actual_pixels_mask is None:
        raise RuntimeError(f"[MASK-SANITIZE][{where}] mask is None")

    if not torch.is_tensor(actual_pixels_mask):
        raise RuntimeError(
            f"[MASK-SANITIZE][{where}] mask is not a tensor, got {type(actual_pixels_mask)}"
        )

    actual_pixels_mask = actual_pixels_mask.float()

    if not torch.isfinite(actual_pixels_mask).all():
        raise RuntimeError(f"[MASK-SANITIZE][{where}] mask contains NaN/Inf")

    min_v = actual_pixels_mask.min().item()
    max_v = actual_pixels_mask.max().item()
    mean_v = actual_pixels_mask.mean().item()

    if min_v < 0.0 or max_v > 1.0:
        print(f"[MASK-SANITIZE][{where}] original min/max/mean:", min_v, max_v, mean_v)
        actual_pixels_mask = (actual_pixels_mask > 0.5).float()
        print(
            f"[MASK-SANITIZE][{where}] binarized min/max/mean:",
            actual_pixels_mask.min().item(),
            actual_pixels_mask.max().item(),
            actual_pixels_mask.mean().item(),
        )

    return actual_pixels_mask


def _compute_loss_weights(mask: torch.Tensor, where: str = "") -> torch.Tensor:
    pre = mask.mean(dim=(1, 2)).float().clamp_min(0.0)
    if not torch.isfinite(pre).all():
        raise RuntimeError(f"[{where}] loss weights contain NaN/Inf before normalization")
    den = pre.sum()
    if (not torch.isfinite(den)) or den.item() <= 0:
        print(f"[BAD-WEIGHTS]{'[' + where + ']' if where else ''} pre-norm:", pre.detach().cpu())
        raise RuntimeError(f"[{where}] Bad denominator for loss weights: {den.item()}")
    return pre / den

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


def _module_grad_norm(module) -> float:
    sq = 0.0
    has_grad = False
    for p in module.parameters():
        if p.grad is not None:
            g = p.grad.detach().data.norm(2).item()
            sq += g * g
            has_grad = True
    return (sq ** 0.5) if has_grad else 0.0


def train(model, train_loader, optimizer, criterion, device, current_config):
    is_dp = isinstance(model, torch.nn.DataParallel)
    core_model = model.module if is_dp else model
    core_model.train()
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

    if not hasattr(train, "_grad_debug_printed"):
        train._grad_debug_printed = False
    train._grad_debug_printed = False


    for data, target, last_layer_per_rec_index, last_real_time_bin_per_rec_idx, actual_pixels_mask, _ in tqdm(
        train_loader, desc="Batch Progress", ascii=True
    ):
        data = preprocess_batches(data, last_real_time_bin_per_rec_idx).to(device)
        target = preprocess_batches(target, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = preprocess_batches(actual_pixels_mask, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = _sanitize_actual_pixels_mask(actual_pixels_mask, where="train")
        # print("[MASK-TYPE-AFTER-SANITIZE]", type(actual_pixels_mask))

        if (not torch.isfinite(data).all()) or (not torch.isfinite(target).all()) or (not torch.isfinite(actual_pixels_mask).all()):
            print("[BAD-BATCH-TRAIN] finite checks:",
                  "data", torch.isfinite(data).all().item(),
                  "target", torch.isfinite(target).all().item(),
                  "mask", torch.isfinite(actual_pixels_mask).all().item())
            print("[BAD-BATCH-TRAIN] data min/max/mean:", data.min().item(), data.max().item(), data.mean().item())
            print("[BAD-BATCH-TRAIN] target min/max/mean:", target.min().item(), target.max().item(), target.mean().item())
            print("[BAD-BATCH-TRAIN] mask min/max/mean:", actual_pixels_mask.min().item(), actual_pixels_mask.max().item(), actual_pixels_mask.mean().item())
            raise RuntimeError("Non-finite tensor detected in train batch before forward")

        all_outputs = core_model(data, actual_pixels_mask)

        for out_i, out in enumerate(all_outputs):
            if not torch.isfinite(out).all():
                print(f"[BAD-OUTPUT-TRAIN] head {out_i} has non-finite values")
                print(f"[BAD-OUTPUT-TRAIN] head {out_i} min/max/mean:", out.min().item(), out.max().item(), out.mean().item())
                raise RuntimeError(f"Non-finite output detected in train forward at head {out_i}")

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
                loss = (
                    criterion[head_idx](output * actual_pixels_mask, target * actual_pixels_mask)
                    .mean(dim=(1, 2)) * loss_w_per_sample
                ).mean()

                if loss.item() < 0 or (not torch.isfinite(loss)):
                    print("[NEG-TRAIN-HEAD0] loss:", loss.item())
                    print("[NEG-TRAIN-HEAD0] criterion[0] type:", type(criterion[0]))
                    raw0 = criterion[0](output * actual_pixels_mask, target * actual_pixels_mask)
                    pre_norm_w = actual_pixels_mask.mean(dim=(1, 2)).float()
                    den_dbg = pre_norm_w.sum()
                    w_dbg = pre_norm_w / den_dbg if torch.isfinite(den_dbg) and den_dbg.item() != 0 else pre_norm_w
                    per_sample_raw = raw0.mean(dim=(1, 2))

                    print("[NEG-TRAIN-HEAD0] raw0 min/max/mean:",
                          raw0.min().item(), raw0.max().item(), raw0.mean().item())
                    print("[NEG-TRAIN-HEAD0] output min/max/mean:",
                          output.min().item(), output.max().item(), output.mean().item())
                    print("[NEG-TRAIN-HEAD0] target min/max/mean:",
                          target.min().item(), target.max().item(), target.mean().item())
                    print("[NEG-TRAIN-HEAD0] mask min/max/mean:",
                          actual_pixels_mask.min().item(),
                          actual_pixels_mask.max().item(),
                          actual_pixels_mask.mean().item())
                    print("[NEG-TRAIN-HEAD0] finite checks:",
                          "output", torch.isfinite(output).all().item(),
                          "target", torch.isfinite(target).all().item(),
                          "mask", torch.isfinite(actual_pixels_mask).all().item(),
                          "raw0", torch.isfinite(raw0).all().item(),
                          "weights", torch.isfinite(loss_w_per_sample).all().item(),
                          "loss", torch.isfinite(loss).item())
                    print("[NEG-TRAIN-HEAD0] pre_norm_w:", pre_norm_w.detach().cpu())
                    print("[NEG-TRAIN-HEAD0] den_dbg:", den_dbg.item())
                    print("[NEG-TRAIN-HEAD0] norm_w:", w_dbg.detach().cpu())
                    print("[NEG-TRAIN-HEAD0] per_sample_raw:", per_sample_raw.detach().cpu())
                    print("[NEG-TRAIN-HEAD0] weighted_terms:", (per_sample_raw * w_dbg).detach().cpu())
                    print("[NEG-TRAIN-HEAD0] recomputed_loss:", (per_sample_raw * w_dbg).mean().item())
                    raise RuntimeError("Head 0 train loss became negative or non-finite")
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

                if not torch.isfinite(loss):
                    print(f"[BAD-LOSS] head_idx={head_idx}, loss={loss.item()}")
                    raise RuntimeError("Non-finite loss")

                if loss.item() < 0:
                    print(f"[NEG-LOSS] head_idx={head_idx}, loss={loss.item()}")
                    print("[NEG-LOSS] loss_w_per_sample min/max/mean:",
                          loss_w_per_sample.min().item(),
                          loss_w_per_sample.max().item(),
                          loss_w_per_sample.mean().item())
                    raise RuntimeError(f"Negative loss at head {head_idx}")

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

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite total_loss: {float(total_loss.detach().item())}")

        tracked_param = None
        try:
            if hasattr(core_model, "shared_decoder"):
                tracked_param = next(core_model.shared_decoder.parameters(), None)
        except Exception:
            tracked_param = None

        before_update = tracked_param.detach().clone() if tracked_param is not None else None

        total_loss.backward()

        grad_norms = {}
        total_sq = 0.0
        for p in core_model.parameters():
            if p.grad is not None:
                g = p.grad.detach().data.norm(2).item()
                total_sq += g * g
        grad_norms["global"] = total_sq ** 0.5
        grad_norms["encoder"] = _module_grad_norm(core_model.encoder)
        grad_norms["shared_decoder"] = _module_grad_norm(core_model.shared_decoder)
        for head_idx in heads_to_train:
            grad_norms[f"head_{head_idx}"] = _module_grad_norm(core_model.heads[head_idx])

        # should_print_grad_debug = not train._grad_debug_printed
        # if should_print_grad_debug:
        #     print("[GRAD-DEBUG] total_loss:", float(total_loss.detach().item()))
        #     for k, v in grad_norms.items():
        #         print(f"[GRAD-DEBUG] {k}: {v:.6e}")
        #     print("[GRAD-DEBUG] per-head losses:", {k: float(v.detach().item()) for k, v in per_head_loss_tensors.items()})

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # if should_print_grad_debug and before_update is not None:
        #     after_update = tracked_param.detach()
        #     print("[PARAM-DEBUG] shared_decoder first param update mean abs:", float((after_update - before_update).abs().mean().item()))
        #     print("[PARAM-DEBUG] shared_decoder first param update max abs:", float((after_update - before_update).abs().max().item()))
        #     train._grad_debug_printed = True

        running_total_loss += float(total_loss.detach().item())
        num_batches += 1

        if config['logging'].get('activate_wandb', False):
            log_dict = {"train/total_loss_batch": float(total_loss.detach().item())}
            for k, v in per_head_loss_tensors.items():
                log_dict[f"train/head_{k}_loss_batch"] = float(v.detach().item())
            for k, v in grad_norms.items():
                log_dict[f"train/grad_norm/{k}"] = float(v)
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

        if (not torch.isfinite(data).all()) or (not torch.isfinite(target).all()) or (not torch.isfinite(actual_pixels_mask).all()):
            print("[BAD-BATCH-VAL] finite checks:",
                  "data", torch.isfinite(data).all().item(),
                  "target", torch.isfinite(target).all().item(),
                  "mask", torch.isfinite(actual_pixels_mask).all().item())
            print("[BAD-BATCH-VAL] data min/max/mean:", data.min().item(), data.max().item(), data.mean().item())
            print("[BAD-BATCH-VAL] target min/max/mean:", target.min().item(), target.max().item(), target.mean().item())
            print("[BAD-BATCH-VAL] mask min/max/mean:", actual_pixels_mask.min().item(), actual_pixels_mask.max().item(), actual_pixels_mask.mean().item())
            raise RuntimeError("Non-finite tensor detected in val batch before forward")

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
                loss = (criterion[output_idx](output * actual_pixels_mask, target * actual_pixels_mask).mean(dim=(1, 2)) * loss_w_per_sample).mean()
                if loss.item() < 0 or (not torch.isfinite(loss)):
                    print("[NEG-VAL-HEAD0] loss:", loss.item())
                    print("[NEG-VAL-HEAD0] criterion[0] type:", type(criterion[0]))
                    raw0 = criterion[0](output * actual_pixels_mask, target * actual_pixels_mask)
                    pre_norm_w = actual_pixels_mask.mean(dim=(1, 2)).float()
                    den_dbg = pre_norm_w.sum()
                    w_dbg = pre_norm_w / den_dbg if torch.isfinite(den_dbg) and den_dbg.item() != 0 else pre_norm_w
                    per_sample_raw = raw0.mean(dim=(1, 2))

                    print("[NEG-VAL-HEAD0] raw0 min/max/mean:",
                          raw0.min().item(), raw0.max().item(), raw0.mean().item())
                    print("[NEG-VAL-HEAD0] output min/max/mean:",
                          output.min().item(), output.max().item(), output.mean().item())
                    print("[NEG-VAL-HEAD0] target min/max/mean:",
                          target.min().item(), target.max().item(), target.mean().item())
                    print("[NEG-VAL-HEAD0] mask min/max/mean:",
                          actual_pixels_mask.min().item(),
                          actual_pixels_mask.max().item(),
                          actual_pixels_mask.mean().item())
                    print("[NEG-VAL-HEAD0] finite checks:",
                          "output", torch.isfinite(output).all().item(),
                          "target", torch.isfinite(target).all().item(),
                          "mask", torch.isfinite(actual_pixels_mask).all().item(),
                          "raw0", torch.isfinite(raw0).all().item(),
                          "weights", torch.isfinite(loss_w_per_sample).all().item(),
                          "loss", torch.isfinite(loss).item())
                    print("[NEG-VAL-HEAD0] pre_norm_w:", pre_norm_w.detach().cpu())
                    print("[NEG-VAL-HEAD0] den_dbg:", den_dbg.item())
                    print("[NEG-VAL-HEAD0] norm_w:", w_dbg.detach().cpu())
                    print("[NEG-VAL-HEAD0] per_sample_raw:", per_sample_raw.detach().cpu())
                    print("[NEG-VAL-HEAD0] weighted_terms:", (per_sample_raw * w_dbg).detach().cpu())
                    print("[NEG-VAL-HEAD0] recomputed_loss:", (per_sample_raw * w_dbg).mean().item())
                    raise RuntimeError("Head 0 val loss became negative or non-finite")
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
