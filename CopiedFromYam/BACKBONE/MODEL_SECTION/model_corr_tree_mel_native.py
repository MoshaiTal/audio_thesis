import copy

import torch
import torch.nn as nn

from CopiedFromYam.BACKBONE.MODEL_SECTION.load_config import load_config


config = load_config()


def weights_init(module):
    classname = module.__class__.__name__
    if "Conv" in classname:
        module.weight.data.normal_(0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            module.bias.data.zero_()
    elif "BatchNorm2d" in classname:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()
    elif "Linear" in classname:
        module.weight.data.normal_(0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            module.bias.data.zero_()


def unet_conv(nc, output_nc, kernel_size, outermost=False, innermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    downconv = nn.Conv2d(
        nc,
        output_nc,
        kernel_size,
        stride=2,
        padding=padding[kernel_size],
    )
    if outermost:
        return nn.Sequential(downconv, nn.LeakyReLU(0.2, True))
    if innermost:
        return nn.Sequential(
            downconv,
            nn.BatchNorm2d(output_nc),
            nn.ReLU(True),
        )
    return nn.Sequential(
        downconv,
        nn.BatchNorm2d(output_nc),
        nn.LeakyReLU(0.2, True),
    )


def unet_upconv(nc, output_nc, kernel_size, use_drop=False, outermost=False):
    padding = {4: 1, (8, 4): (3, 1), (4, 8): (1, 3)}
    upconv = nn.ConvTranspose2d(
        nc,
        output_nc,
        kernel_size,
        stride=2,
        padding=padding[kernel_size],
    )
    if outermost:
        return nn.Sequential(upconv, nn.Tanh())

    layers = [upconv, nn.BatchNorm2d(output_nc)]
    if use_drop:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class MelNativeSplitUNet(nn.Module):
    """
    Paper-style U-Net adapted to true Whisper mel geometry.

    The original paper uses 256x256 log-STFT images and downsamples eight
    times. Whisper mels are 80xT, so eight frequency downsamplings are not
    possible without padding fake bins. This model keeps the same block style,
    skip connections, tanh output, and split-head CP structure, but uses four
    down/up stages:

        80 x 256 -> 40 x 128 -> 20 x 64 -> 10 x 32 -> 5 x 16

    This is the minimal geometry change needed for true 80-bin mels.
    """

    def __init__(self, ngf=64, nc=1, kernel_size=4, split_location=None):
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

        if split_location is None:
            split_location = len(decoder) - 1
        self.split_location = int(max(0, min(split_location, len(decoder) - 1)))
        self.shared_decoder = nn.ModuleList(decoder[: self.split_location])

        base_head = nn.ModuleList(decoder[self.split_location :])
        self.heads = nn.ModuleList(
            [
                copy.deepcopy(base_head)
                for _ in range(2 * len(config["Model params"]["heads_params"]) - 1)
            ]
        )
        for head_index, head in enumerate(self.heads):
            torch.manual_seed(head_index)
            head.apply(weights_init)

        self.dim = 1

    def forward(self, x, mask):
        features = []
        for conv in self.encoder:
            x = conv(x)
            features.append(x)

        bottleneck = features[-1]
        for index, upconv in enumerate(self.shared_decoder):
            if index == 0:
                x = upconv(bottleneck)
            else:
                x = upconv(torch.cat((x, features[-(index + 1)]), dim=self.dim))

        outputs = []
        for head in self.heads:
            head_x = x
            for index, upconv in enumerate(head):
                skip = features[-(self.split_location + index + 1)]
                head_x = upconv(torch.cat((head_x, skip), dim=1))
            head_x = head_x.squeeze(1)
            outputs.append(head_x * mask)
        return outputs

