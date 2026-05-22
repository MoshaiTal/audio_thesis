import os
import numpy as np
import torch
import torch.nn as nn
import wandb

from CopiedFromYam.BACKBONE.CALIBRATION.krcps_yam.calibrate_from_main import run_calibration
from CopiedFromYam.BACKBONE.MODEL_SECTION.data_loader import load_data
from CopiedFromYam.BACKBONE.MODEL_SECTION.model_creator import create_model
from CopiedFromYam.BACKBONE.MODEL_SECTION.criterions_and_optimizers_tree import (set_optimizer, set_criterion, set_scheduler)
from CopiedFromYam.BACKBONE.MODEL_SECTION.training_page_same_alpha import training_loop, evaluate_individual_heads
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config
from CopiedFromYam.set_env import (set_device, set_seed, create_check_points_file, create_weights_folder, create_output_folder)

# IMPORTANT: we rely on these from weights.py:
from CopiedFromYam.BACKBONE.MODEL_SECTION.weights import load_checkpoint, make_ckpt_path

def main(current_config, step_index):
    config = load_config()

    # best_val_loss is tracked per head (matches your usage)
    best_val_loss = np.array([np.inf] * (2 * len(config['Model params']['heads_params']) - 1))

    device, ngpu, gpu_ids, gpu_names, multi_gpu = set_device()
    set_seed()

    create_check_points_file()
    create_weights_folder()
    create_output_folder()

    train_loader = load_data('train')
    val_loader   = load_data('val')

    model = create_model().to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        model = nn.DataParallel(model, gpu_ids)

    # Create optimizer/scheduler/criterion FIRST (so we can restore optimizer state on resume)
    optimizer = set_optimizer(model, current_config)
    scheduler = set_scheduler(optimizer, current_config)
    criterion = set_criterion(current_config, device)

    # -------------------------
    # RESUME LOGIC (from *_last.pth if exists, else *_best.pth if exists)
    # -------------------------
    start_epoch = 0
    should_resume = (current_config['current_step']['training_phase'][0] in ["retrain", "eval"] or step_index == 0)

    if should_resume:
        base_path = config['General']['weights_path']
        last_path = make_ckpt_path(base_path, "last")
        best_path = make_ckpt_path(base_path, "best")

        if os.path.exists(last_path):
            model, optimizer, start_epoch, ckpt_best_val = load_checkpoint(model, optimizer, last_path, map_location="cpu")
            if ckpt_best_val is not None:
                best_val_loss = np.array(ckpt_best_val)
            print(f"Resuming from LAST checkpoint at epoch {start_epoch}")
        elif os.path.exists(best_path):
            model, optimizer, start_epoch, ckpt_best_val = load_checkpoint(model, optimizer, best_path, map_location="cpu")
            if ckpt_best_val is not None:
                best_val_loss = np.array(ckpt_best_val)
            print(f"Starting from BEST checkpoint (no last checkpoint). Next epoch: {start_epoch}")
        else:
            print("No checkpoint found. Starting from scratch.")

        # Recreate scheduler AFTER optimizer is loaded (safe)
        scheduler = set_scheduler(optimizer, current_config)

    # -------------------------
    # TRAIN / EVAL
    # -------------------------
    if current_config['current_step']['training_phase'][0] != "eval":
        model, best_val_loss = training_loop(
            model=model,
            train_loader   = train_loader,
            test_loader    = val_loader,
            optimizer      = optimizer,
            scheduler      = scheduler,
            criterion      = criterion,
            device         = device,
            best_val_loss  = best_val_loss,
            current_config = current_config,
            start_epoch    = start_epoch
        )

    else:
        # Eval branch (unchanged idea)
        _, val_output, val_target, val_inputs = evaluate_individual_heads(model, val_loader, criterion, device, current_config)

        test_loader = load_data('test')
        cal_loader  = load_data('cal')

        test_loss, test_output, test_targets, test_inputs = evaluate_individual_heads(model, test_loader, criterion, device, current_config)
        cal_loss, cal_output, cal_targets, cal_inputs = evaluate_individual_heads(model, cal_loader, criterion, device, current_config)

        print(
            f"Train samples: {len(train_loader.dataset)}\n"
            f"Val samples: {len(val_loader.dataset)}\n"
            f"Test samples: {test_targets.shape[0] if test_targets is not None else 'N/A'}\n"
            f"Cal samples: {cal_targets.shape[0] if cal_targets is not None else 'N/A'}"
        )
        # run_calibration(config)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    config = load_config()

    if config['logging']['activate_wandb']:
        wandb.login(key=config['logging']['wandb_key'])
        wandb.init(project=config['logging']['wandb_project'], config=config)

    current_config = {}
    current_config['current_step'] = {}
    current_config['current_step']['batch_size']     = config['Train params']['batch_size']
    current_config['current_step']['dropout_prob']   = config['Model params']['dropout_prob']
    current_config['current_step']['optimizer_name'] = config['Train params']['optimizer_name']
    current_config['current_step']['weight_decay']   = config['Train params']['weight_decay']

    for step_index, s in enumerate(config['Training plan'].values()):
        current_config['current_step']['trained_heads']    = s['trained_heads']
        current_config['current_step']['training_phase']   = s['training_phase']
        current_config['current_step']['num_epochs']       = s['num_epochs']
        current_config['current_step']['lr']               = s['lr']
        current_config['current_step']['scheduler_name']   = s['scheduler']['name']
        current_config['current_step']['scheduler_params'] = s['scheduler']['params']

        print(current_config['current_step'])
        main(current_config, step_index)
