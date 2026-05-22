# import numpy as np
import torch.optim as optim
import torch.nn as nn
from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import *
from CopiedFromYam.BACKBONE.MODEL_SECTION.losses import *
config = load_config()

# Tal's changes
def set_criterion(current_config, device):
    heads_params = config['Model params']['heads_params']
    H = len(heads_params)                  # e.g. 5 (0..4)
    num_outputs = 2 * H - 1                # e.g. 9
    criterions = [None] * num_outputs

    # Always fill head 0 (no mirror)
    head0_loss = heads_params[0]['loss']
    reduction0 = config['Model params'].get('reduction', 'none')  # keep your default behavior
    if head0_loss == 'MSE':
        criterions[0] = nn.MSELoss(reduction=reduction0)
    elif head0_loss == 'MAE':
        criterions[0] = nn.L1Loss(reduction=reduction0)
    elif head0_loss == 'GradMSELoss':
        criterions[0] = GradMSELoss(device, reduction=config['Model params'].get('reduction', 'mean'))
    elif head0_loss == 'Residuals':
        criterions[0] = Residuals(reduction=config['Model params'].get('reduction', 'mean'))
    else:
        raise ValueError(f"Unsupported loss for head 0: {head0_loss}")

    # Fill mirrored heads 1..H-1
    for k in range(1, H):
        criterion_name = heads_params[k]['loss']
        alpha = float(heads_params[k]['alpha'])

        if criterion_name == 'Pinball':
            reduction = config['Model params'].get('reduction', 'mean')
            q_l = alpha / 2.0
            q_u = 1.0 - alpha / 2.0
            criterions[k]  = PinballLoss(quantile=q_l, reduction=reduction)   # lower
            criterions[-k] = PinballLoss(quantile=q_u, reduction=reduction)   # upper (mirror)

        elif criterion_name == 'GradMSELoss':
            reduction = config['Model params'].get('reduction', 'mean')
            loss_obj = GradMSELoss(device, reduction=reduction)
            criterions[k]  = loss_obj
            criterions[-k] = loss_obj

        elif criterion_name == 'Residuals':
            reduction = config['Model params'].get('reduction', 'mean')
            loss_obj = Residuals(reduction=reduction)
            criterions[k]  = loss_obj
            criterions[-k] = loss_obj

        elif criterion_name == 'MSE':
            reduction = config['Model params'].get('reduction', 'none')
            loss_obj = nn.MSELoss(reduction=reduction)
            criterions[k]  = loss_obj
            criterions[-k] = loss_obj

        elif criterion_name == 'MAE':
            reduction = config['Model params'].get('reduction', 'none')
            loss_obj = nn.L1Loss(reduction=reduction)
            criterions[k]  = loss_obj
            criterions[-k] = loss_obj

        else:
            raise ValueError(f"Unsupported criterion name: {criterion_name}")

    # Safety check
    if any(c is None for c in criterions):
        missing = [i for i,c in enumerate(criterions) if c is None]
        raise RuntimeError(f"Some criteria not set: indices {missing}")

    return criterions
# -------------------------------

# def set_criterion(current_config, device):
#     criterions_l = [None] * (2 * len(config['Model params']['heads_params']))
#     for head in range(len(config['Model params']['heads_params'])):
#         criterion_name = config['Model params']['heads_params'][head]['loss']
#         if criterion_name == 'Pinball':
#             reduction  = config['Model params'].get('reduction', 'mean')
#             quantile_l = config['Model params']['heads_params'][head]['alpha'] / 2
#             quantile_u = 1-config['Model params']['heads_params'][head]['alpha'] / 2
#             criterions_l[head]      = PinballLoss(quantile=quantile_l,reduction=reduction)
#             criterions_l[-(head+1)] = PinballLoss(quantile=quantile_u,reduction=reduction)
#
#         elif criterion_name == 'GradMSELoss':
#             reduction = config['Model params'].get('reduction', 'mean')
#             criterions_l[head]      = GradMSELoss(device, reduction=reduction)
#             criterions_l[-(head+1)] = GradMSELoss(device, reduction=reduction)
#
#         elif criterion_name == 'Residuals':
#             reduction = config['Model params'].get('reduction', 'mean')  # Default to 'mean' reduction
#             criterions_l[head] = Residuals(reduction=reduction)
#             criterions_l[-(head+1)] = Residuals(reduction=reduction)
#
#         elif criterion_name == 'MSE':
#             reduction = config['Model params'].get('reduction', 'none')  # Default to 'mean' reduction
#             criterions_l[head]      = nn.MSELoss(reduction=reduction)
#             criterions_l[-(head+1)] = nn.MSELoss(reduction=reduction)
#
#         elif criterion_name == 'MAE':
#             reduction = config['Model params'].get('reduction', 'none')  # Default to 'none' reduction
#             criterions_l[head] = nn.L1Loss(reduction=reduction)
#             criterions_l[-(head+1)] = nn.L1Loss(reduction=reduction)
#
#         else:
#             raise ValueError(f"Unsupported criterion name: {criterion_name}")
#     return criterions_l

def set_optimizer(model, current_config):
    if isinstance(model, torch.nn.DataParallel) :
        model=model.module
    weight_decay = current_config['current_step'].get('weight_decay', 0)
    momentum = current_config['current_step'].get('momentum', 0)
    lr = current_config['current_step']['lr']  # Unified learning rate
    optimizer_name = current_config['current_step']['optimizer_name']
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(),lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(),lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")

def set_scheduler(optimizer, current_config):
    scheduler_name = current_config['current_step']['scheduler_name']
    scheduler_params = current_config['current_step'].get('scheduler_params', {})

    if scheduler_name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_params.get('mode', 'min'),
            factor=scheduler_params.get('factor', 0.1),
            patience=scheduler_params.get('patience', 2),
            min_lr=scheduler_params.get('min_lr', 1e-9),
            eps=scheduler_params.get('eps', 1e-9)
        )
    elif scheduler_name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params.get('T_max', 5),
            eta_min=scheduler_params.get('eta_min', 1e-6)
        )
    elif scheduler_name == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params.get('step_size', 30),
            gamma=scheduler_params.get('gamma', 0.1)
        )
    elif scheduler_name == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_params.get('gamma', 0.95)
        )
    elif scheduler_name == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_params.get('max_lr', 0.1),
            total_steps=scheduler_params.get('total_steps', 1000),
            pct_start=scheduler_params.get('pct_start', 0.3),
            anneal_strategy=scheduler_params.get('anneal_strategy', 'cos'),
            div_factor=scheduler_params.get('div_factor', 25.0),
            final_div_factor=scheduler_params.get('final_div_factor', 1e4)
        )
    else:
        raise ValueError(f"Unsupported scheduler name: {scheduler_name}")
