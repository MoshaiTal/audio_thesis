from torch import nn

from CopiedFromYam.BACKBONE.CALIBRATION.fast_soft_sort.pytorch_ops import soft_rank
import torch
import torch.nn.functional as F

from torch.nn.modules.loss import _Loss
import numpy as np


class GradL1Loss(_Loss):
    '''
    Computes the images gradients loss suggested in "Burst Image Deblurring"
    '''

    def __init__(self, device, size_average=None, reduce=None, reduction='mean'):
        super(GradL1Loss, self).__init__(size_average, reduce, reduction)

        self.reduction = reduction

        prewitt_filter = 1 / 6 * np.array([[1, 0, -1],
                                           [1, 0, -1],
                                           [1, 0, -1]])

        self.prewitt_filter_horizontal = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                         kernel_size=prewitt_filter.shape,
                                                         padding=prewitt_filter.shape[0] // 2).to(device)

        self.prewitt_filter_horizontal.weight.data.copy_(torch.from_numpy(prewitt_filter).to(device))
        self.prewitt_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device))

        self.prewitt_filter_vertical = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                                       kernel_size=prewitt_filter.shape,
                                                       padding=prewitt_filter.shape[0] // 2).to(device)

        self.prewitt_filter_vertical.weight.data.copy_(torch.from_numpy(prewitt_filter.T).to(device))
        self.prewitt_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])).to(device))

    def get_gradients(self, img):
        if len(img.shape)==3:
            img_r=img.unsqueeze(1).to(dtype=torch.float32)
        else:
            img_r = img[:, 0:1, :, :].to(dtype=torch.float32)
        grad_x = self.prewitt_filter_horizontal(img_r)
        grad_y = self.prewitt_filter_vertical(img_r)
        grad = torch.cat([grad_x, grad_y], dim=1)

        return grad

    def forward(self, input, target):
        input_grad = self.get_gradients(input)
        target_grad = self.get_gradients(target)

        return 0.1 * F.l1_loss(input, target, reduction=self.reduction) + F.l1_loss(input_grad, target_grad,
                                                                                    reduction=self.reduction)
class GradMSELoss(GradL1Loss):
    def forward(self, input, target):
        input_grad = self.get_gradients(input)
        target_grad = self.get_gradients(target)

        return 0.1 * F.mse_loss(input, target, reduction=self.reduction) + F.mse_loss(input_grad, target_grad,
                                                                                      reduction=self.reduction).mean(axis=1)
class PinballLoss():
    def __init__(self, quantile, reduction="none"):
        assert (quantile > 0 and quantile < 1)
        assert reduction in {"mean", "sum","none"}
        self.quantile = quantile
        self.reduction = reduction
    
    def __call__(self, output, target, weights):
        assert output.shape == target.shape
        loss = torch.zeros_like(target)
        error = output - target
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
        loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])
        if self.reduction == 'sum':
            loss = loss.sum()
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'none':
            loss = (loss.mean(dim=(1,2))*weights).mean()
        return loss



def cosine_similarity_loss(i,heads):
    loss_upp,loss_low = 0,0
    batch_num=heads[0].shape[0]
    for head_num in range((1+len(heads)-1)//2):
        if head_num==i or 0 :
            continue
        cos_sim = F.cosine_similarity( heads[i].view(batch_num, -1), heads[head_num].view(batch_num,-1), dim=1)+1
        loss_low += cos_sim.mean()  # Mean over batch
        cos_sim =F.cosine_similarity(heads[-i].view(batch_num,-1), heads[-head_num].view(batch_num,-1), dim=1)+1
        loss_upp += cos_sim.mean()  # Mean over batch
    return 0.1*(loss_low+loss_upp)/((head_num-1)*2)

def Residuals(reduction='mean'):
    return nn.MSELoss(reduction=reduction)

def corrcoef_batch(target, pred):
    pred_n = pred - pred.mean(dim=1, keepdim=True)
    target_n = target - target.mean(dim=1, keepdim=True)
    # Avoid division by zero by adding epsilon only to non-zero norms
    pred_norm = torch.norm(pred_n, dim=1, keepdim=True)
    target_norm = torch.norm(target_n, dim=1, keepdim=True)

    pred_n = pred_n / (pred_norm + (pred_norm == 0).float() * 1e-8)
    target_n = target_n / (target_norm + (target_norm == 0).float() * 1e-8)

    return (pred_n * target_n).sum(dim=1)



def spearman(target, pred, regularization="l2", regularization_strength=0.1):
    pred_ranks = soft_rank(
        pred,
        regularization=regularization,
        regularization_strength=regularization_strength,
    )
    pred_ranks = pred_ranks.to(pred.device)
    coef=corrcoef_batch(target, pred_ranks)
    coef_pos=(1 - coef.mean())/2


    return  coef_pos

