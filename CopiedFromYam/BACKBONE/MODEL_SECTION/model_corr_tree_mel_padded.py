import torch
import torch.nn as nn
import torch.nn.functional as F

from CopiedFromYam.BACKBONE.MODEL_SECTION.model_corr_tree import SplitUNet


class MelPaddedSplitUNet(nn.Module):
    """
    Wrapper for running the original 256-bin U-Net on 80-bin Whisper mels.

    The paper-style U-Net downsamples many times and expects a 256x256 image.
    Whisper mels have only 80 frequency bins, so this wrapper pads frequency
    to 256 before the U-Net and crops the outputs back to the original 80 bins.

    The training objective is unchanged; this only fixes geometry.
    """

    def __init__(
        self,
        ngf=64,
        nc=1,
        kernel_size=(2, 4),
        split_location=7,
        padded_freq_bins=256,
        pad_value=0.0,
    ):
        super().__init__()
        self.base = SplitUNet(
            ngf=ngf,
            nc=nc,
            kernel_size=kernel_size,
            split_location=split_location,
        )
        self.padded_freq_bins = int(padded_freq_bins)
        self.pad_value = float(pad_value)

    @property
    def encoder(self):
        return self.base.encoder

    @property
    def shared_decoder(self):
        return self.base.shared_decoder

    @property
    def heads(self):
        return self.base.heads

    def _pad_input(self, x):
        if x.shape[-2] >= self.padded_freq_bins:
            return x, x.shape[-2]
        pad_f = self.padded_freq_bins - x.shape[-2]
        return F.pad(x, (0, 0, 0, pad_f), value=self.pad_value), x.shape[-2]

    def _pad_mask(self, mask):
        if mask.shape[-2] >= self.padded_freq_bins:
            return mask
        pad_f = self.padded_freq_bins - mask.shape[-2]
        return F.pad(mask, (0, 0, 0, pad_f), value=0.0)

    def forward(self, x, mask):
        x_pad, original_freq_bins = self._pad_input(x)
        mask_pad = self._pad_mask(mask)
        outputs = self.base(x_pad, mask_pad)
        return [out[:, :original_freq_bins, :] for out in outputs]

