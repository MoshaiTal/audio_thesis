import os

import numpy as np
import torch
import torch.nn as nn
import wandb

from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.calibrate_from_main import run_calibration
from CopiedFromYam.BACKBONE.MODEL_SECTION.criterions_and_optimizers_tree import (
    set_criterion,
    set_optimizer,
    set_scheduler,
)
from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.training_page_same_alpha import (
    evaluate_individual_heads,
    training_loop,
)
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import load_checkpoint, make_ckpt_path
from CopiedFromYam.set_env import (
    create_check_points_file,
    create_output_folder,
    create_weights_folder,
    set_device,
    set_seed,
)


def main(current_config, step_index):
    config = load_config()
    best_val_loss = np.array([np.inf] * (2 * len(config["Model params"]["heads_params"]) - 1))

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
    phase = current_config["current_step"]["training_phase"][0]
    if phase in {"retrain", "eval"}:
        base_path = config["General"]["weights_path"]
        last_path = make_ckpt_path(base_path, "last")
        best_path = make_ckpt_path(base_path, "best")
        load_path = last_path if os.path.exists(last_path) else best_path
        if os.path.exists(load_path):
            model, optimizer, start_epoch, ckpt_best_val = load_checkpoint(
                model,
                optimizer,
                load_path,
                map_location="cpu",
            )
            if ckpt_best_val is not None:
                best_val_loss = np.array(ckpt_best_val)
            print(f"Loaded checkpoint: {load_path}; next epoch={start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")
        scheduler = set_scheduler(optimizer, current_config)

    if phase != "eval":
        model, best_val_loss = training_loop(
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
        return

    _, val_output, val_target, val_inputs = evaluate_individual_heads(
        model,
        val_loader,
        criterion,
        device,
        current_config,
    )
    test_loader = load_data("test")
    cal_loader = load_data("cal")
    test_loss, test_output, test_targets, test_inputs = evaluate_individual_heads(
        model,
        test_loader,
        criterion,
        device,
        current_config,
    )
    cal_loss, cal_output, cal_targets, cal_inputs = evaluate_individual_heads(
        model,
        cal_loader,
        criterion,
        device,
        current_config,
    )
    print(
        f"Val samples: {val_target.shape[0]}\n"
        f"Test samples: {test_targets.shape[0]}\n"
        f"Cal samples: {cal_targets.shape[0]}"
    )
    run_calibration(config)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    config = load_config()
    if config["logging"]["activate_wandb"]:
        wandb.login(key=config["logging"]["wandb_key"])
        wandb.init(project=config["logging"]["wandb_project"], config=config)

    current_config = {"current_step": {}}
    current_config["current_step"]["batch_size"] = config["Train params"]["batch_size"]
    current_config["current_step"]["dropout_prob"] = config["Model params"]["dropout_prob"]
    current_config["current_step"]["optimizer_name"] = config["Train params"]["optimizer_name"]
    current_config["current_step"]["weight_decay"] = config["Train params"]["weight_decay"]

    for step_index, step in enumerate(config["Training plan"].values()):
        if not step.get("active", True):
            continue
        current_config["current_step"]["trained_heads"] = step["trained_heads"]
        current_config["current_step"]["training_phase"] = step["training_phase"]
        current_config["current_step"]["num_epochs"] = step["num_epochs"]
        current_config["current_step"]["lr"] = step["lr"]
        current_config["current_step"]["scheduler_name"] = step["scheduler"]["name"]
        current_config["current_step"]["scheduler_params"] = step["scheduler"]["params"]
        print(current_config["current_step"])
        main(current_config, step_index)
