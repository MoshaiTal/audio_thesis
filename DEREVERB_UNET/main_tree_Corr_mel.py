import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from DEREVERB_UNET.criterions_and_optimizers_tree import (
    set_criterion,
    set_optimizer,
    set_scheduler,
)
from DEREVERB_UNET.data_loader import load_data
from DEREVERB_UNET.load_config import load_config
from DEREVERB_UNET.model_creator import create_model
from DEREVERB_UNET.training_page_same_alpha_mel import (
    evaluate_individual_heads,
    training_loop,
)
from DEREVERB_UNET.weights import load_best_weights
from DEREVERB_UNET.weights import load_training_checkpoint
from DEREVERB_UNET.set_env import (
    create_check_points_file,
    create_output_folder,
    create_weights_folder,
    set_device,
    set_seed,
)


def expand_heads_for_best_loss(model, trained_heads):
    if trained_heads == ["all"]:
        return list(range((len(model.heads) + 1) // 2))
    return [int(head) for head in trained_heads]


def main(current_config):
    config = load_config()
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

    core_model = model.module if isinstance(model, nn.DataParallel) else model
    best_val_loss = np.array([np.inf] * len(core_model.heads))
    start_epoch = 0

    if current_config["current_step"]["training_phase"][0] in {"retrain", "eval"}:
        model, best_val_loss = load_best_weights(
            model,
            config["General"]["weights_path"],
        )

    optimizer = set_optimizer(model, current_config)
    scheduler = set_scheduler(optimizer, current_config)
    criterion = set_criterion(current_config, device)

    if current_config["current_step"]["training_phase"][0] == "resume":
        model, optimizer, scheduler, loaded_best_val_loss, start_epoch = load_training_checkpoint(
            model,
            optimizer,
            scheduler,
        )
        if loaded_best_val_loss is not None:
            best_val_loss = loaded_best_val_loss

    if current_config["current_step"]["training_phase"][0] != "eval":
        training_loop(
            model,
            train_loader,
            val_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            best_val_loss,
            current_config,
            start_epoch=start_epoch,
        )
    else:
        _, val_output, val_target, val_inputs = evaluate_individual_heads(
            model,
            val_loader,
            criterion,
            device,
            current_config,
        )
        print(
            f"Val samples: {val_target.shape[0]} "
            f"outputs shape: {val_output.shape} "
            f"inputs shape: {val_inputs.shape}"
        )


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
        if not step.get("active", True):
            continue
        current = current_config["current_step"]
        current["trained_heads"] = step["trained_heads"]
        current["training_phase"] = step["training_phase"]
        current["num_epochs"] = step["num_epochs"]
        current["lr"] = step["lr"]
        current["scheduler_name"] = step["scheduler"]["name"]
        current["scheduler_params"] = step["scheduler"]["params"]
        print(current)
        main(current_config)
