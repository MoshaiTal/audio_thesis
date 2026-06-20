from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from CopiedFromYam.BACKBONE.MODEL_SECTION.criterions_and_optimizers_tree import (
    set_criterion,
    set_optimizer,
    set_scheduler,
)
from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.training_page_same_alpha_original import (
    evaluate_individual_heads,
    training_loop,
)
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import (
    load_checkpoint,
    make_ckpt_path,
)
from CopiedFromYam.set_env import (
    create_check_points_file,
    create_output_folder,
    create_weights_folder,
    set_device,
    set_seed,
)


def main(current_config, step_index):
    config = load_config()
    if config["General"]["DATA_TYPE"] != "melspec":
        raise RuntimeError("This experiment requires General.DATA_TYPE: melspec")
    if config["Model params"]["unet_arch"] not in {
        "vanilla_mel_padded",
        "vanilla_mel_native",
    }:
        raise RuntimeError(
            "This experiment requires Model params.unet_arch: "
            "vanilla_mel_native or vanilla_mel_padded"
        )

    best_val_loss = np.array(
        [np.inf] * (2 * len(config["Model params"]["heads_params"]) - 1)
    )
    device, ngpu, gpu_ids, _, _ = set_device()
    set_seed()

    create_check_points_file()
    create_weights_folder()
    create_output_folder()

    train_loader = load_data("train")
    val_loader = load_data("val")

    model = create_model().to(device)
    if device.type == "cuda" and ngpu > 1:
        model = nn.DataParallel(model, gpu_ids)

    optimizer = set_optimizer(model, current_config)
    scheduler = set_scheduler(optimizer, current_config)
    criterion = set_criterion(current_config, device)

    start_epoch = 0
    base_path = config["General"]["weights_path"]
    last_path = make_ckpt_path(base_path, "last")
    best_path = make_ckpt_path(base_path, "best")

    if os.path.exists(last_path):
        model, optimizer, start_epoch, checkpoint_val_loss = load_checkpoint(
            model,
            optimizer,
            last_path,
            map_location="cpu",
        )
        if checkpoint_val_loss is not None:
            best_val_loss = np.array(checkpoint_val_loss)
        print(f"Resuming mel-original experiment from LAST at epoch {start_epoch}")
        scheduler = set_scheduler(optimizer, current_config)
    elif os.path.exists(best_path):
        model, optimizer, start_epoch, checkpoint_val_loss = load_checkpoint(
            model,
            optimizer,
            best_path,
            map_location="cpu",
        )
        if checkpoint_val_loss is not None:
            best_val_loss = np.array(checkpoint_val_loss)
        print(f"Starting mel-original experiment from BEST; next epoch {start_epoch}")
        scheduler = set_scheduler(optimizer, current_config)
    else:
        print("No mel-original checkpoint found. Starting from scratch.")

    if current_config["current_step"]["training_phase"][0] != "eval":
        training_loop(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            best_val_loss=best_val_loss,
            current_config=current_config,
            start_epoch=start_epoch,
        )
    else:
        evaluate_individual_heads(model, val_loader, criterion, device, current_config)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    config = load_config()

    if config["logging"].get("activate_wandb", False):
        wandb.login(key=config["logging"]["wandb_key"])
        wandb.init(project=config["logging"]["wandb_project"], config=config)

    current_config = {
        "current_step": {
            "batch_size": config["Train params"]["batch_size"],
            "dropout_prob": config["Model params"]["dropout_prob"],
            "optimizer_name": config["Train params"]["optimizer_name"],
            "weight_decay": config["Train params"]["weight_decay"],
        }
    }

    for step_index, step in enumerate(config["Training plan"].values()):
        current_config["current_step"]["trained_heads"] = step["trained_heads"]
        current_config["current_step"]["training_phase"] = step["training_phase"]
        current_config["current_step"]["num_epochs"] = step["num_epochs"]
        current_config["current_step"]["lr"] = step["lr"]
        current_config["current_step"]["scheduler_name"] = step["scheduler"]["name"]
        current_config["current_step"]["scheduler_params"] = step["scheduler"][
            "params"
        ]
        print(current_config["current_step"])
        main(current_config, step_index)
