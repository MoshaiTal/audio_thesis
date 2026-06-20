import torch
import torch.nn as nn
import torch.optim as optim

try:
    from DEREVERB_UNET.load_config import load_config
    from DEREVERB_UNET.losses import (
        GradMSELoss,
        PinballLoss,
        Residuals,
    )
except ModuleNotFoundError:
    from load_config import load_config
    from losses import GradMSELoss, PinballLoss, Residuals


config = load_config()


def _make_criterion(criterion_name, head, device):
    reduction = config["Model params"].get("reduction", "none")

    if criterion_name == "Pinball":
        alpha = config["Model params"]["heads_params"][head]["alpha"]
        return (
            PinballLoss(quantile=alpha / 2, reduction=reduction),
            PinballLoss(quantile=1 - alpha / 2, reduction=reduction),
        )

    if criterion_name == "GradMSELoss":
        return (
            GradMSELoss(device, reduction=reduction),
            GradMSELoss(device, reduction=reduction),
        )

    if criterion_name == "Residuals":
        return Residuals(reduction=reduction), Residuals(reduction=reduction)

    if criterion_name == "MSE":
        return nn.MSELoss(reduction=reduction), nn.MSELoss(reduction=reduction)

    if criterion_name == "MAE":
        return nn.L1Loss(reduction=reduction), nn.L1Loss(reduction=reduction)

    raise ValueError(f"Unsupported criterion name: {criterion_name}")


def set_criterion(current_config, device):
    """
    Return criteria aligned with model outputs:

        output[0]  -> central prediction
        output[h]  -> lower CP head h
        output[-h] -> upper CP head h

    The original Yam indexing used -(head + 1), which is fine only when every
    head is a lower/upper pair. Once head 0 is a central MSE head, that puts
    head 0's MSE into criterion[-1], where head 1 expects upper Pinball.
    """
    num_heads = len(config["Model params"]["heads_params"])
    criteria = [None] * (2 * num_heads - 1)

    for head in range(num_heads):
        criterion_name = config["Model params"]["heads_params"][head]["loss"]
        lower_or_center, upper = _make_criterion(criterion_name, head, device)

        if head == 0:
            criteria[0] = lower_or_center
        else:
            criteria[head] = lower_or_center
            criteria[-head] = upper

    return criteria


def set_optimizer(model, current_config):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    weight_decay = current_config["current_step"].get("weight_decay", 0)
    momentum = current_config["current_step"].get("momentum", 0)
    lr = current_config["current_step"]["lr"]
    optimizer_name = current_config["current_step"]["optimizer_name"]

    if optimizer_name == "Adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_name == "SGD":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    if optimizer_name == "RMSprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer name: {optimizer_name}")


def set_scheduler(optimizer, current_config):
    scheduler_name = current_config["current_step"]["scheduler_name"]
    scheduler_params = current_config["current_step"].get("scheduler_params", {})

    if scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get("mode", "min"),
            factor=scheduler_params.get("factor", 0.1),
            patience=scheduler_params.get("patience", 2),
            min_lr=scheduler_params.get("min_lr", 1e-9),
            eps=scheduler_params.get("eps", 1e-9),
        )
    if scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get("T_max", 5),
            eta_min=scheduler_params.get("eta_min", 1e-6),
        )
    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get("step_size", 30),
            gamma=scheduler_params.get("gamma", 0.1),
        )
    if scheduler_name == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_params.get("gamma", 0.95),
        )
    if scheduler_name == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_params.get("max_lr", 0.1),
            total_steps=scheduler_params.get("total_steps", 1000),
            pct_start=scheduler_params.get("pct_start", 0.3),
            anneal_strategy=scheduler_params.get("anneal_strategy", "cos"),
            div_factor=scheduler_params.get("div_factor", 25.0),
            final_div_factor=scheduler_params.get("final_div_factor", 1e4),
        )

    raise ValueError(f"Unsupported scheduler name: {scheduler_name}")
