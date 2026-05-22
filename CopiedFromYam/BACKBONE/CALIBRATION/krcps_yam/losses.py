import torch

def loss_zero_one(target, l, u, reduction="mean", dim=0):
    loss = torch.where(torch.logical_and(target >= l, target <= u), 0.0, 1.0)
    if reduction == "mean":
        return torch.mean(loss, dim=dim)
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction {reduction}")