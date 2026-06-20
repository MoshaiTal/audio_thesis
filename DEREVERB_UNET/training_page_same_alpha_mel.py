import os

import numpy as np
import torch
from tqdm import tqdm
import wandb

from DEREVERB_UNET.load_config import load_config
from DEREVERB_UNET.losses import (
    cosine_similarity_loss,
    spearman,
)
from DEREVERB_UNET.utils import (
    preprocess_batches,
    reconstruct_from_batch,
)
from DEREVERB_UNET.weights import save_best_weights
from DEREVERB_UNET.weights import save_last_checkpoint


config = load_config()


def expand_heads_to_train(model, trained_heads):
    if trained_heads == ["all"]:
        return list(range((len(model.heads) + 1) // 2))
    return [int(head) for head in trained_heads]


def loss_weights_from_mask(mask):
    weights = mask.mean(dim=(1, 2)).float()
    return weights / weights.sum().clamp_min(1e-8)


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
    start_epoch=0,
):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    heads_to_train = expand_heads_to_train(
        model,
        current_config["current_step"]["trained_heads"],
    )

    if config["Model params"]["split_model"]:
        for i in range((len(model.heads) + 1) // 2):
            if i not in heads_to_train:
                for param_l, param_u in zip(
                    model.heads[i].parameters(),
                    model.heads[-i].parameters(),
                ):
                    param_l.requires_grad = False
                    param_u.requires_grad = False
                model.heads[i].eval()
                model.heads[-i].eval()

        if current_config["current_step"]["training_phase"][1] == "part":
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.shared_decoder.parameters():
                param.requires_grad = False
            model.encoder.eval()
            model.shared_decoder.eval()

        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters: {trainable_params}")

    for epoch in range(start_epoch, current_config["current_step"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1} out of {current_config['current_step']['num_epochs']}")

        train_loss, model = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            current_config,
        )
        val_loss, _, _, _ = evaluate_individual_heads(
            model,
            test_loader,
            criterion,
            device,
            current_config,
        )

        heads_idx = np.unique(np.concatenate([heads_to_train, [-h for h in heads_to_train]]))
        if val_loss[heads_idx].mean() < best_val_loss[heads_idx].mean():
            best_val_loss[heads_idx] = val_loss[heads_idx]
            if config["General"]["save_weights"]:
                save_best_weights(model, optimizer, epoch, best_val_loss)

        if config["logging"]["activate_wandb"]:
            for i, loss in enumerate(val_loss):
                wandb.log({f"Head_{i}/val_loss": loss})
            wandb.log({"epoch": epoch}, commit=True)

        print(
            f"Epoch {epoch + 1}/{current_config['current_step']['num_epochs']}, "
            f"Train Loss: {train_loss}, Val Loss: {val_loss}\n"
        )

        if current_config["current_step"]["scheduler_name"] != "CosineAnnealingLR":
            scheduler.step(val_loss[heads_to_train].mean())
        else:
            scheduler.step()

        if config["General"]["save_weights"]:
            save_last_checkpoint(model, optimizer, scheduler, epoch, best_val_loss)

        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")

    if config["Model params"]["split_model"]:
        for i in range((len(model.heads) + 1) // 2):
            if i not in heads_to_train:
                for param_l, param_u in zip(
                    model.heads[i].parameters(),
                    model.heads[-i].parameters(),
                ):
                    param_l.requires_grad = True
                    param_u.requires_grad = True
                model.heads[i].train()
                model.heads[-i].train()

        if current_config["current_step"]["training_phase"][1] == "part":
            for param in model.encoder.parameters():
                param.requires_grad = True
            for param in model.shared_decoder.parameters():
                param.requires_grad = True
            model.encoder.train()
            model.shared_decoder.train()

    return model, best_val_loss


def train(model, train_loader, optimizer, criterion, device, current_config):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.train()
    running_losses = [0.0] * len(model.heads)
    heads_to_train = expand_heads_to_train(
        model,
        current_config["current_step"]["trained_heads"],
    )

    for data, target, last_real_time_bin_per_rec_idx, last_layer_per_rec_index, actual_pixels_mask, _ in tqdm(
        train_loader,
        desc="Batch Progress",
        ascii=True,
    ):
        data = preprocess_batches(data, last_real_time_bin_per_rec_idx).to(device)
        target = preprocess_batches(target, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = preprocess_batches(
            actual_pixels_mask,
            last_real_time_bin_per_rec_idx,
        ).to(device)

        all_outputs = model(data, actual_pixels_mask)
        optimizer.zero_grad()
        loss_w_per_sample = loss_weights_from_mask(actual_pixels_mask)

        total_loss = 0.0
        for head_idx in heads_to_train:
            if head_idx == 0:
                output = all_outputs[0]
                loss = (
                    criterion[0](
                        output * actual_pixels_mask,
                        target * actual_pixels_mask,
                    ).mean(dim=(1, 2))
                    * loss_w_per_sample
                ).mean()
            else:
                output = all_outputs[0]
                output_l = all_outputs[head_idx]
                output_u = all_outputs[-head_idx]

                if config["Model params"]["heads_params"][head_idx]["loss"] == "Residuals":
                    loss_quantile_l = (
                        criterion[head_idx](
                            output_l * actual_pixels_mask,
                            torch.minimum(target * actual_pixels_mask, output * actual_pixels_mask),
                        ).mean(dim=(1, 2))
                        * loss_w_per_sample
                    ).mean()
                    loss_quantile_u = (
                        criterion[-head_idx](
                            output_u * actual_pixels_mask,
                            torch.maximum(target * actual_pixels_mask, output * actual_pixels_mask),
                        ).mean(dim=(1, 2))
                        * loss_w_per_sample
                    ).mean()
                else:
                    loss_quantile_l = criterion[head_idx](
                        output_l * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample,
                    )
                    loss_quantile_u = criterion[-head_idx](
                        output_u * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample,
                    )

                loss = 0.5 * loss_quantile_u + 0.5 * loss_quantile_l

                if config["Model params"]["heads_params"][head_idx]["loss_corr"]:
                    interval = (output_u - output_l).reshape(output_u.shape[0], -1)
                    error = torch.abs(output - target).reshape(output_u.shape[0], -1)
                    mask_flat = actual_pixels_mask.reshape(output_u.shape[0], -1)
                    loss = loss + 0.5 * spearman(
                        interval * mask_flat,
                        error * mask_flat,
                        "l2",
                        0.01,
                    )

                if (
                    config["Model params"]["heads_params"][head_idx]["loss_div"]
                    and len(heads_to_train) > 1
                ):
                    loss = loss + 0.1 * cosine_similarity_loss(head_idx, all_outputs)

            total_loss = total_loss + loss
            running_losses[head_idx] += float(loss.detach().item())
            if head_idx != 0:
                running_losses[-head_idx] += float(loss.detach().item())

        total_loss.backward()
        optimizer.step()

    return np.array(running_losses) / len(train_loader), model


def evaluate_individual_heads(model, data_loader, criterion, device, current_config):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    running_losses = [0.0] * len(model.heads)
    all_model_outputs = []
    all_targets = []
    all_input = []

    for data, target, last_real_time_bin_per_rec_idx, last_layer_per_rec_index, actual_pixels_mask, paths_list in tqdm(
        data_loader,
        desc="Batch Progress",
        ascii=True,
    ):
        data = preprocess_batches(data, last_real_time_bin_per_rec_idx).to(device)
        target = preprocess_batches(target, last_real_time_bin_per_rec_idx).to(device)
        actual_pixels_mask = preprocess_batches(
            actual_pixels_mask,
            last_real_time_bin_per_rec_idx,
        ).to(device)
        loss_w_per_sample = loss_weights_from_mask(actual_pixels_mask)

        with torch.no_grad():
            all_outputs = model(data, actual_pixels_mask)

        if current_config["current_step"]["training_phase"][0] == "eval" and config["General"]["save_output"][0]:
            for h_i, head in enumerate(all_outputs):
                reconstructed = reconstruct_from_batch(
                    head.cpu().detach().squeeze().numpy(),
                    np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0),
                )
                for layer_i in range(reconstructed.shape[0]):
                    if paths_list[0][layer_i] in ["train", "val", "cal", "test"]:
                        folder_path_to_save = os.path.join(
                            config["General"]["save_output"][1],
                            config["General"]["save_name"],
                            paths_list[0][layer_i],
                            paths_list[1][layer_i],
                        )
                        os.makedirs(folder_path_to_save, exist_ok=True)
                        file_path = os.path.join(
                            folder_path_to_save,
                            paths_list[2][layer_i].replace(
                                ".npy",
                                f"(layer {h_i}|{len(all_outputs)}).npy",
                            ),
                        )
                        np.save(file_path, reconstructed[layer_i])

        all_model_outputs.append(
            np.array(
                [
                    reconstruct_from_batch(
                        head.cpu().detach().squeeze().numpy(),
                        np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0),
                    )
                    for head in all_outputs
                ]
            ).transpose((1, 0, 2, 3))
        )
        all_targets.append(
            reconstruct_from_batch(
                target.cpu().detach().squeeze().numpy(),
                np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0),
            )
        )
        all_input.append(
            reconstruct_from_batch(
                data.cpu().detach().squeeze().numpy(),
                np.cumsum(last_real_time_bin_per_rec_idx.numpy() + 1, axis=0),
            )
        )

        for output_idx in range(len(config["Model params"]["heads_params"])):
            if output_idx == 0:
                output = all_outputs[0]
                loss = (
                    criterion[0](
                        output * actual_pixels_mask,
                        target * actual_pixels_mask,
                    ).mean(dim=(1, 2))
                    * loss_w_per_sample
                ).mean()
            else:
                output = all_outputs[0]
                output_l = all_outputs[output_idx]
                output_u = all_outputs[-output_idx]

                if config["Model params"]["heads_params"][output_idx]["loss"] == "Residuals":
                    loss_quantile_l = (
                        criterion[output_idx](
                            output_l,
                            torch.minimum(target, output),
                        ).mean(dim=(1, 2))
                        * loss_w_per_sample
                    ).mean()
                    loss_quantile_u = (
                        criterion[-output_idx](
                            output_u,
                            torch.maximum(target, output),
                        ).mean(dim=(1, 2))
                        * loss_w_per_sample
                    ).mean()
                else:
                    loss_quantile_l = criterion[output_idx](
                        output_l * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample,
                    )
                    loss_quantile_u = criterion[-output_idx](
                        output_u * actual_pixels_mask,
                        target * actual_pixels_mask,
                        loss_w_per_sample,
                    )
                loss = 0.5 * loss_quantile_u + 0.5 * loss_quantile_l

                if config["Model params"]["heads_params"][output_idx]["loss_corr"]:
                    interval = (output_u - output_l).reshape(output_u.shape[0], -1)
                    error = torch.abs(output - target).reshape(output_u.shape[0], -1)
                    loss = loss + 0.5 * spearman(interval, error, "l2", 0.01)

                if (
                    config["Model params"]["heads_params"][output_idx]["loss_div"]
                    and len(config["Model params"]["heads_params"]) > 3
                ):
                    loss = loss + 0.1 * cosine_similarity_loss(output_idx, all_outputs)

            running_losses[output_idx] += float(loss.item())
            if output_idx != 0:
                running_losses[-output_idx] += float(loss.item())

    running_losses = np.array(running_losses) / len(data_loader)
    all_model_outputs = np.concatenate(all_model_outputs)
    all_targets = np.concatenate(all_targets)
    all_input = np.concatenate(all_input)
    return running_losses, all_model_outputs, all_targets, all_input
