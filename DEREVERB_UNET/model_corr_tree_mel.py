import copy

import torch
import torch.nn as nn

from DEREVERB_UNET.load_config import load_config


config = load_config()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            m.bias.data.fill_(0)


def unet_conv(nc, output_nc, kernel_size, outermost=False, innermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    downrelu = nn.LeakyReLU(0.2, True)
    downconv = nn.Conv2d(
        nc,
        output_nc,
        kernel_size,
        stride=2,
        padding=padding[kernel_size],
    )
    if outermost:
        return nn.Sequential(downconv, downrelu)
    if innermost:
        downnorm = nn.BatchNorm2d(output_nc)
        return nn.Sequential(downconv, downnorm, nn.ReLU(True))
    downnorm = nn.BatchNorm2d(output_nc)
    return nn.Sequential(downconv, downnorm, downrelu)


def unet_upconv(nc, output_nc, kernel_size, use_drop=False, outermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    uprelu = nn.ReLU(True)
    upconv = nn.ConvTranspose2d(
        nc,
        output_nc,
        kernel_size,
        stride=2,
        padding=padding[kernel_size],
    )
    if outermost:
        return nn.Sequential(upconv, nn.Tanh())

    upnorm = nn.BatchNorm2d(output_nc)
    if use_drop:
        return nn.Sequential(upconv, upnorm, nn.Dropout(0.5), uprelu)
    return nn.Sequential(upconv, upnorm, uprelu)


class MelSplitUNet(nn.Module):
    """
    U-Net for true 80-bin Whisper mel spectrograms.

    This keeps Yam/the paper's block style:
    Conv-BN-LeakyReLU encoder, deconv-BN-ReLU decoder, skip connections,
    tanh output, and split output heads.

    Minimal geometry change:
        80x256 -> 40x128 -> 20x64 -> 10x32 -> 5x16
    instead of the original 256x256 -> ... -> 1x1 STFT image path.
    """

    def __init__(self, ngf=64, nc=1, kernel_size=4, split_location=3):
        super().__init__()
        self.nc = nc

        self.encoder = nn.ModuleList(
            [
                unet_conv(nc, ngf, kernel_size, outermost=True),
                unet_conv(ngf, ngf * 2, kernel_size),
                unet_conv(ngf * 2, ngf * 4, kernel_size),
                unet_conv(ngf * 4, ngf * 8, kernel_size, innermost=True),
            ]
        )

        decoder = nn.ModuleList(
            [
                unet_upconv(ngf * 8, ngf * 4, kernel_size, use_drop=True),
                unet_upconv(ngf * 8, ngf * 2, kernel_size),
                unet_upconv(ngf * 4, ngf, kernel_size),
                unet_upconv(ngf * 2, 1, kernel_size, outermost=True),
            ]
        )

        # Keep the original split-head idea. For this 4-layer decoder,
        # split_location=3 means only the final output block is head-specific.
        self.split_location = int(max(0, min(split_location, len(decoder) - 1)))
        self.shared_decoder = nn.ModuleList(decoder[: self.split_location])
        base_head = nn.ModuleList(decoder[self.split_location :])
        self.heads = nn.ModuleList(
            [
                copy.deepcopy(base_head)
                for _ in range(2 * len(config["Model params"]["heads_params"]) - 1)
            ]
        )

        for i, head in enumerate(self.heads):
            torch.manual_seed(i)
            head.apply(weights_init)

        self.dim = 1

    def forward(self, x, mask):
        features = []
        for conv in self.encoder:
            x = conv(x)
            features.append(x)

        bottleneck = features[-1]
        for i, upconv in enumerate(self.shared_decoder):
            if i == 0:
                x = upconv(bottleneck)
            else:
                x = upconv(torch.cat((x, features[-(i + 1)]), self.dim))

        outputs = []
        for head in self.heads:
            head_x = x
            for i, upconv in enumerate(head):
                skip = features[-(self.split_location + i + 1)]
                head_x = upconv(torch.cat((head_x, skip), self.dim))
            outputs.append(head_x.squeeze(1) * mask)
        return outputs

