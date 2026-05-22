from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import *
import torch
config = load_config()

# MODEL_SECTION/weights.py
import os
import torch

def make_ckpt_path(base_path: str, tag: str) -> str:
    file_root, suffix = os.path.splitext(base_path)
    return f"{file_root}_{tag}{suffix}"   # e.g. ensambels3_last.pth / ensambels3_best.pth

def save_checkpoint(model, optimizer, epoch, val_loss, tag="last"):
    """
    tag:
      - "last": always overwrite each epoch (resume point)
      - "best": overwrite only when improved
    """
    base_path = config['General']['weights_path']
    save_path = make_ckpt_path(base_path, tag)

    # handle DataParallel
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

    state = {
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "val_loss": val_loss,
    }
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")

def save_best_weights(model, optimizer, epoch, val_loss):
    # keep your existing call-sites intact
    save_checkpoint(model, optimizer, epoch, val_loss, tag="best")

def load_checkpoint(model, optimizer, load_path, map_location="cpu"):
    """
    Loads model + optimizer state if present.
    Returns: model, optimizer, start_epoch, val_loss
    start_epoch is the *next* epoch to run (checkpoint_epoch + 1)
    """
    keep_DP = False
    if isinstance(model, torch.nn.DataParallel):
        gpu_ids = model.device_ids
        model = model.module
        keep_DP = True

    checkpoint = torch.load(load_path, map_location=map_location, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    val_loss = checkpoint.get("val_loss", None)

    if keep_DP:
        model = torch.nn.DataParallel(model, gpu_ids)

    print(f"Loaded checkpoint from {load_path} (epoch={epoch})")
    start_epoch = epoch + 1
    return model, optimizer, start_epoch, val_loss

def load_best_weights(model, load_path, val_loss=None, member_idx=None):
    """
    KEEP THIS ONLY if other code expects it.
    BUT: it loads *best* only, not resume-accurate.
    Prefer load_checkpoint(..., *_last.pth) for resuming training.
    """
    keep_DP = False
    if isinstance(model, torch.nn.DataParallel):
        gpu_ids = model.device_ids
        model = model.module
        keep_DP = True

    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
    target = model.models[member_idx] if member_idx is not None else model
    target.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint.get("epoch", -1)
    ckpt_val_loss = checkpoint.get("val_loss", None)
    if val_loss is not None and member_idx is not None:
        val_loss[member_idx] = ckpt_val_loss
    else:
        val_loss = ckpt_val_loss

    print(f"Loaded weights from {load_path}, Epoch: {epoch}, Validation Loss: {val_loss}")

    if keep_DP:
        model = torch.nn.DataParallel(model, gpu_ids)

    return model, val_loss

# def save_best_weights(model, optimizer, epoch, val_loss,model_name=""):
#     save_path=config['General']['weights_path']
#     file_root,suffix=os.path.splitext(save_path)
#     save_path=file_root+model_name+suffix
#     state = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'val_loss': val_loss,
#     }
#     torch.save(state, save_path)
#     print(f"Best weights saved to {save_path}")
#
#
# def load_best_weights(model, load_path,val_loss=None, member_idx=None):
#     """
#     Loads weights into either:
#     - a full model (single, non-ensemble), or
#     - one member of an EnsembleUNet (if `member_idx` is provided).
#
#     Args:
#         model: full model (nn.Module or EnsembleUNet)
#         load_path: checkpoint file to load from
#         member_idx: if not None, loads into model.models[member_idx]
#     """
#     keep_DP = False
#     if isinstance(model, torch.nn.DataParallel):
#         gpu_ids = model.device_ids
#         model = model.module
#         keep_DP = True
#
#     # checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
#     checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
#     target = model.models[member_idx] if member_idx is not None else model
#
#     target.load_state_dict(checkpoint['model_state_dict'])
#
#     epoch = checkpoint.get('epoch', -1)
#     if val_loss is not None:
#         val_loss[member_idx] = checkpoint.get('val_loss', None)
#     else:
#         val_loss = checkpoint.get('val_loss', None)
#     print(f"Loaded weights from {load_path}, Epoch: {epoch}, Validation Loss: {val_loss}")
#
#     if keep_DP:
#         model = torch.nn.DataParallel(model, gpu_ids)
#
#     return model, val_loss
