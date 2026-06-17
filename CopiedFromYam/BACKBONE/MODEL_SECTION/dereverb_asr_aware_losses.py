from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bct(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"Expected [B,T] or [B,C,T], got {tuple(x.shape)}")
    return x


def _match_time(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n = min(a.shape[-1], b.shape[-1])
    return a[..., :n], b[..., :n]


def _safe_log_mag(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=eps))


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=x.dtype, device=x.device)
    mask, x = _match_time(mask, x)
    return (x * mask).sum() / (mask.sum() + 1e-8)


def time_delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred, target = _match_time(pred, target)
    loss = torch.abs((pred[..., 1:] - pred[..., :-1]) - (target[..., 1:] - target[..., :-1]))
    delta_mask = mask[..., 1:] * mask[..., :-1] if mask is not None else None
    return _masked_mean(loss, delta_mask)


def freq_delta_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    pred, target = _match_time(pred, target)
    if pred.shape[-2] < 2:
        return pred.new_tensor(0.0)
    loss = torch.abs((pred[:, 1:, :] - pred[:, :-1, :]) - (target[:, 1:, :] - target[:, :-1, :]))
    delta_mask = mask[:, 1:, :] * mask[:, :-1, :] if mask is not None else None
    return _masked_mean(loss, delta_mask)


def soft_speech_activity_mask(
    clean_spec: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    threshold_quantile: float = 0.30,
    temperature: float = 0.08,
) -> torch.Tensor:
    """
    Return a soft [B,1,T] speech-activity mask derived from the clean target.

    The model is trained on normalized log spectra, so the mean value across
    frequency is a stable relative frame-energy proxy. A per-utterance
    quantile avoids requiring the original waveform scale.
    """
    if clean_spec.dim() != 3:
        raise ValueError(f"Expected clean spectrum [B,F,T], got {tuple(clean_spec.shape)}")

    frame_score = clean_spec.mean(dim=1, keepdim=True)
    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_time = valid_mask.amax(dim=1, keepdim=True) > 0.5
        elif valid_mask.dim() == 2:
            valid_time = valid_mask.unsqueeze(1) > 0.5
        else:
            raise ValueError(f"Expected valid mask [B,F,T] or [B,T], got {tuple(valid_mask.shape)}")
    else:
        valid_time = torch.ones_like(frame_score, dtype=torch.bool)

    thresholds = []
    for batch_idx in range(frame_score.shape[0]):
        values = frame_score[batch_idx][valid_time[batch_idx]]
        if values.numel() == 0:
            thresholds.append(frame_score.new_tensor(0.0))
        else:
            thresholds.append(torch.quantile(values.detach(), threshold_quantile))
    threshold = torch.stack(thresholds).view(-1, 1, 1)

    speech = torch.sigmoid((frame_score - threshold) / max(float(temperature), 1e-4))
    return speech * valid_time.to(dtype=clean_spec.dtype)


def mid_speech_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    start_bin: int = 32,
    end_bin: int = 129,
    speech_quantile: float = 0.30,
    speech_temperature: float = 0.08,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    L1 and temporal-delta losses in the ASR-critical 1-4 kHz region.

    With a 512-point STFT at 16 kHz, bins 32:129 correspond approximately
    to 1-4 kHz. The soft clean-speech mask prevents silence from dominating.
    """
    pred, target = _match_time(pred, target)
    end_bin = min(end_bin, pred.shape[-2], target.shape[-2])
    start_bin = min(start_bin, end_bin)
    if end_bin <= start_bin:
        zero = pred.new_tensor(0.0)
        return zero, zero

    speech = soft_speech_activity_mask(
        target,
        valid_mask=mask,
        threshold_quantile=speech_quantile,
        temperature=speech_temperature,
    )
    pred_mid = pred[:, start_bin:end_bin, :]
    target_mid = target[:, start_bin:end_bin, :]
    speech_mid = speech.expand(-1, end_bin - start_bin, -1)

    if mask is not None:
        valid_mid = mask[:, start_bin:end_bin, :]
        speech_mid = speech_mid * valid_mid

    l1 = _masked_mean(torch.abs(pred_mid - target_mid), speech_mid)
    delta_error = torch.abs(
        (pred_mid[..., 1:] - pred_mid[..., :-1])
        - (target_mid[..., 1:] - target_mid[..., :-1])
    )
    delta_mask = speech_mid[..., 1:] * speech_mid[..., :-1]
    time_delta = _masked_mean(delta_error, delta_mask)
    return l1, time_delta


def mel_modulation_loss(
    pred_mel: torch.Tensor,
    target_mel: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    fft_sizes: Sequence[int] = (64, 128, 256),
) -> torch.Tensor:
    """
    Compares temporal modulation of mel energy.

    This is meant to fight robotic/flat speech. It does not compare individual
    mel bins directly; it compares how the speech envelope moves over time.
    """
    pred_mel, target_mel = _match_time(pred_mel, target_mel)
    if mask is not None:
        mask, pred_mel = _match_time(mask, pred_mel)
        target_mel = target_mel[..., : pred_mel.shape[-1]]
        pred_mel = pred_mel * mask
        target_mel = target_mel * mask
    pred_env = pred_mel.mean(dim=1)
    target_env = target_mel.mean(dim=1)
    pred_env = pred_env - pred_env.mean(dim=-1, keepdim=True)
    target_env = target_env - target_env.mean(dim=-1, keepdim=True)

    losses = []
    for n_fft in fft_sizes:
        if pred_env.shape[-1] < n_fft:
            continue
        pred_spec = torch.fft.rfft(pred_env, n=n_fft, dim=-1).abs()
        target_spec = torch.fft.rfft(target_env, n=n_fft, dim=-1).abs()
        losses.append(F.l1_loss(_safe_log_mag(pred_spec), _safe_log_mag(target_spec)))
    if not losses:
        return F.l1_loss(pred_env, target_env)
    return torch.stack(losses).mean()


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes: Sequence[int] = (256, 512, 1024, 2048),
        hop_sizes: Sequence[int] = (64, 128, 256, 512),
        win_lengths: Optional[Sequence[int]] = None,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(hop_sizes)
        self.win_lengths = tuple(win_lengths or fft_sizes)
        self.eps = eps
        if not (len(self.fft_sizes) == len(self.hop_sizes) == len(self.win_lengths)):
            raise ValueError("fft_sizes, hop_sizes and win_lengths must have the same length")

    def _stft_mag(self, x: torch.Tensor, n_fft: int, hop: int, win_length: int) -> torch.Tensor:
        x = _to_bct(x).reshape(-1, x.shape[-1])
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        z = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        return z.abs().clamp_min(self.eps)

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        pred_wav, target_wav = _match_time(pred_wav, target_wav)
        losses = []
        for n_fft, hop, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            pred_mag = self._stft_mag(pred_wav, n_fft, hop, win_length)
            target_mag = self._stft_mag(target_wav, n_fft, hop, win_length)
            sc = torch.linalg.norm(target_mag - pred_mag, ord="fro") / (torch.linalg.norm(target_mag, ord="fro") + self.eps)
            log_mag = F.l1_loss(torch.log(pred_mag), torch.log(target_mag))
            losses.append(sc + log_mag)
        return torch.stack(losses).mean()


class TorchWhisperLogMelLoss(nn.Module):
    """
    Differentiable approximation of Whisper's log-mel frontend.

    Use this only when the dereverb model outputs waveform. If your model
    already outputs Whisper-like 80-bin log-mels, use direct L1 between pred
    and clean mels instead.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: float = 8000.0,
        eps: float = 1e-10,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        mel_fb = self._build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb, persistent=False)

    @staticmethod
    def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + freq / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> torch.Tensor:
        freqs = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)
        mel_min = self._hz_to_mel(torch.tensor(self.f_min))
        mel_max = self._hz_to_mel(torch.tensor(self.f_max))
        mels = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        hz = self._mel_to_hz(mels)

        fb = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        for i in range(self.n_mels):
            left, center, right = hz[i], hz[i + 1], hz[i + 2]
            up = (freqs - left) / (center - left + 1e-12)
            down = (right - freqs) / (right - center + 1e-12)
            fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
        return fb

    def log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        wav = _to_bct(wav).reshape(-1, wav.shape[-1])
        window = torch.hann_window(self.n_fft, device=wav.device, dtype=wav.dtype)
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        )
        power = spec.abs().pow(2)
        mel = torch.matmul(self.mel_fb.to(power.device, power.dtype), power)
        log_spec = torch.clamp(mel, min=self.eps).log10()
        max_per_sample = log_spec.flatten(1).amax(dim=1).view(-1, 1, 1)
        log_spec = torch.maximum(log_spec, max_per_sample - 8.0)
        return (log_spec + 4.0) / 4.0

    def forward(self, pred_wav: torch.Tensor, target_wav: torch.Tensor) -> torch.Tensor:
        pred_mel = self.log_mel(pred_wav)
        target_mel = self.log_mel(target_wav)
        pred_mel, target_mel = _match_time(pred_mel, target_mel)
        return F.l1_loss(pred_mel, target_mel)


@dataclass
class DereverbLossWeights:
    mel_l1: float = 1.0
    mel_time_delta: float = 0.25
    mel_freq_delta: float = 0.10
    mel_modulation: float = 0.15
    mid_speech_l1: float = 0.0
    mid_speech_time_delta: float = 0.0
    residual_to_reverb: float = 0.02
    mrstft: float = 0.0
    whisper_logmel: float = 0.0


class ASRAwareDereverbLoss(nn.Module):
    """
    Drop-in combined loss for dereverb training.

    Required for mel-output models:
        pred_mel, clean_mel

    Optional:
        reverb_mel for residual regularization
        pred_wav, clean_wav for MR-STFT / Whisper-logmel waveform losses

    The goal is to avoid over-smoothed/robotic outputs that are numerically
    close to clean but bad for ASR.
    """

    def __init__(
        self,
        weights: Optional[DereverbLossWeights] = None,
        sample_rate: int = 16000,
        mrstft_fft_sizes: Sequence[int] = (256, 512, 1024, 2048),
    ):
        super().__init__()
        self.weights = weights or DereverbLossWeights()
        self.mrstft_loss = (
            MultiResolutionSTFTLoss(fft_sizes=mrstft_fft_sizes)
            if self.weights.mrstft > 0
            else None
        )
        self.whisper_logmel_loss = (
            TorchWhisperLogMelLoss(sample_rate=sample_rate)
            if self.weights.whisper_logmel > 0
            else None
        )

    def forward(
        self,
        pred_mel: torch.Tensor,
        clean_mel: torch.Tensor,
        reverb_mel: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pred_wav: Optional[torch.Tensor] = None,
        clean_wav: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_mel, clean_mel = _match_time(pred_mel, clean_mel)
        terms: Dict[str, torch.Tensor] = {}
        if mask is not None:
            mask, pred_mel = _match_time(mask, pred_mel)
            clean_mel = clean_mel[..., : pred_mel.shape[-1]]

        terms["mel_l1"] = _masked_mean(torch.abs(pred_mel - clean_mel), mask)
        terms["mel_time_delta"] = (
            time_delta_loss(pred_mel, clean_mel, mask)
            if self.weights.mel_time_delta > 0
            else pred_mel.new_tensor(0.0)
        )
        terms["mel_freq_delta"] = (
            freq_delta_loss(pred_mel, clean_mel, mask)
            if self.weights.mel_freq_delta > 0
            else pred_mel.new_tensor(0.0)
        )
        terms["mel_modulation"] = (
            mel_modulation_loss(pred_mel, clean_mel, mask)
            if self.weights.mel_modulation > 0
            else pred_mel.new_tensor(0.0)
        )
        if self.weights.mid_speech_l1 > 0 or self.weights.mid_speech_time_delta > 0:
            mid_l1, mid_time_delta = mid_speech_losses(
                pred_mel,
                clean_mel,
                mask=mask,
            )
            terms["mid_speech_l1"] = mid_l1
            terms["mid_speech_time_delta"] = mid_time_delta
        else:
            terms["mid_speech_l1"] = pred_mel.new_tensor(0.0)
            terms["mid_speech_time_delta"] = pred_mel.new_tensor(0.0)

        if reverb_mel is not None and self.weights.residual_to_reverb > 0:
            reverb_mel, pred_for_res = _match_time(reverb_mel, pred_mel)
            res_mask = mask[..., : pred_for_res.shape[-1]] if mask is not None else None
            terms["residual_to_reverb"] = _masked_mean(torch.abs(pred_for_res - reverb_mel), res_mask)
        else:
            terms["residual_to_reverb"] = pred_mel.new_tensor(0.0)

        if (
            pred_wav is not None
            and clean_wav is not None
            and (self.mrstft_loss is not None or self.whisper_logmel_loss is not None)
        ):
            pred_wav, clean_wav = _match_time(pred_wav, clean_wav)
            terms["mrstft"] = (
                self.mrstft_loss(pred_wav, clean_wav)
                if self.mrstft_loss is not None
                else pred_mel.new_tensor(0.0)
            )
            terms["whisper_logmel"] = (
                self.whisper_logmel_loss(pred_wav, clean_wav)
                if self.whisper_logmel_loss is not None
                else pred_mel.new_tensor(0.0)
            )
        else:
            terms["mrstft"] = pred_mel.new_tensor(0.0)
            terms["whisper_logmel"] = pred_mel.new_tensor(0.0)

        total = pred_mel.new_tensor(0.0)
        for name, value in terms.items():
            weight = getattr(self.weights, name)
            total = total + float(weight) * value
        terms["total"] = total
        return total, terms


def make_default_asr_aware_loss(sample_rate: int = 16000) -> ASRAwareDereverbLoss:
    """
    Conservative first setting.

    Start here. If the output is still robotic, increase mel_time_delta and
    mel_modulation. If output is too close to reverb / not dereverbed enough,
    reduce residual_to_reverb.
    """
    weights = DereverbLossWeights(
        mel_l1=1.0,
        mel_time_delta=0.25,
        mel_freq_delta=0.10,
        mel_modulation=0.15,
        mid_speech_l1=0.0,
        mid_speech_time_delta=0.0,
        residual_to_reverb=0.02,
        mrstft=0.10,
        whisper_logmel=0.20,
    )
    return ASRAwareDereverbLoss(weights=weights, sample_rate=sample_rate)
