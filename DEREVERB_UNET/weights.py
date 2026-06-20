import os

import torch

try:
    from DEREVERB_UNET.load_config import load_config
except ModuleNotFoundError:
    from load_config import load_config


config = load_config()


def last_checkpoint_path(base_path=None):
    save_path = base_path or config["General"]["weights_path"]
    file_root, suffix = os.path.splitext(save_path)
    if file_root.endswith("_best"):
        file_root = file_root[: -len("_best")]
    return file_root + "_last" + suffix


def save_best_weights(model, optimizer, epoch, val_loss, model_name=""):
    save_path = config["General"]["weights_path"]
    file_root, suffix = os.path.splitext(save_path)
    save_path = file_root + model_name + suffix
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }
    torch.save(state, save_path)
    print(f"Best weights saved to {save_path}")


def save_last_checkpoint(model, optimizer, scheduler, epoch, best_val_loss):
    save_path = last_checkpoint_path()
    tmp_path = save_path + ".tmp"
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": best_val_loss,
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, save_path)
    print(f"Last checkpoint saved to {save_path}")


def load_best_weights(model, load_path, val_loss=None, member_idx=None):
    keep_dp = False
    if isinstance(model, torch.nn.DataParallel):
        gpu_ids = model.device_ids
        model = model.module
        keep_dp = True

    # PyTorch 2.6 changed torch.load's default to weights_only=True.
    # These checkpoints are created locally and include numpy validation-loss
    # metadata, so they need the old full-checkpoint loading behavior.
    checkpoint = torch.load(
        load_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    target = model.models[member_idx] if member_idx is not None else model
    target.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    if val_loss is not None:
        val_loss[member_idx] = checkpoint.get("val_loss", None)
    else:
        val_loss = checkpoint.get("val_loss", None)
    print(f"Loaded weights from {load_path}, Epoch: {epoch}, Validation Loss: {val_loss}")

    if keep_dp:
        model = torch.nn.DataParallel(model, gpu_ids)

    return model, val_loss


def load_training_checkpoint(model, optimizer, scheduler, load_path=None):
    load_path = load_path or last_checkpoint_path()
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Resume checkpoint was not found: {load_path}")

    keep_dp = False
    if isinstance(model, torch.nn.DataParallel):
        gpu_ids = model.device_ids
        model = model.module
        keep_dp = True

    checkpoint = torch.load(
        load_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_val_loss = checkpoint.get("val_loss", None)
    print(
        f"Resumed checkpoint from {load_path}, "
        f"last finished epoch={checkpoint.get('epoch', -1)}, next epoch={start_epoch}"
    )

    if keep_dp:
        model = torch.nn.DataParallel(model, gpu_ids)

    return model, optimizer, scheduler, best_val_loss, start_epoch
