from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


LEN_FACTOR = 2  # Whisper encoder downsamples time by 2 with conv2 stride.


def expected_encoder_len(input_lengths: torch.Tensor) -> torch.Tensor:
    return (input_lengths + 1) // LEN_FACTOR


def make_time_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    t = torch.arange(max_len, device=lengths.device)[None, :]
    return t < lengths[:, None]


class CPConditioner(nn.Module):
    """
    Builds local + global CP-conditioned representations.

    Important change from the previous version:
    This receives richer uncertainty features:
      pred, lower, upper, raw width, log width, pred-lower, upper-pred.

    The previous version used only per-sample normalized width, which can erase
    global uncertainty scale. This version keeps raw and log-scaled information.
    """

    def __init__(self, mel_bins: int = 80, d_cond: int = 128, cond_feature_groups: int = 6):
        super().__init__()
        in_ch = mel_bins * (1 + cond_feature_groups)  # pred + uncertainty feature groups
        self.local_net = nn.Sequential(
            nn.Conv1d(in_ch, 192, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(192, d_cond, kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.global_net = nn.Sequential(
            nn.Linear(d_cond, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, d_cond),
        )

    def forward(
        self,
        pred: torch.Tensor,
        cp_features: torch.Tensor,
        input_lengths: torch.Tensor,
        target_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([pred, cp_features], dim=1)
        local = self.local_net(x)  # [B, d_cond, T]
        local_ds = F.adaptive_avg_pool1d(local, target_seq_len).transpose(1, 2)  # [B, S, d_cond]

        mask = make_time_mask(input_lengths, pred.shape[-1]).float()  # [B, T]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (local * mask[:, None, :]).sum(dim=-1) / denom  # [B, d_cond]
        global_vec = self.global_net(pooled)  # [B, d_cond]
        return local_ds, global_vec


class LatentCondSABlock(nn.Module):
    """
    Condformer-style latent conditional self-attention block.

    Crucial fix:
    the old version zero-initialized both the residual gates and the output
    projections. That makes the block an exact no-op with no useful gradient.
    This version starts very close to no-op, but not exactly no-op.
    """

    def __init__(self, d_model: int, d_cond: int, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")

        self.d_model = d_model
        self.d_cond = d_cond
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.global_to_cond = nn.Linear(d_cond, d_cond)
        self.q_lfm = nn.Linear(d_model + d_cond, d_model)
        self.k_lfm = nn.Linear(d_model + d_cond, d_model)

        hidden_dim = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        # Near-identity initialization with a live gradient path.
        nn.init.normal_(self.q_lfm.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.q_lfm.bias)
        nn.init.normal_(self.k_lfm.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.k_lfm.bias)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.ffn[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.ffn[-1].bias)

        self.attn_gate = nn.Parameter(torch.tensor(0.01))
        self.ffn_gate = nn.Parameter(torch.tensor(0.01))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)  # [B, H, S, Dh]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_local: torch.Tensor,
        cond_global: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.norm1(hidden_states)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        cond = cond_local + self.global_to_cond(cond_global)[:, None, :]
        q = q + self.q_lfm(torch.cat([q, cond], dim=-1))
        k = k + self.k_lfm(torch.cat([k, cond], dim=-1))

        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            keep = key_padding_mask[:, None, None, :]  # [B,1,1,S]
            scores = scores.masked_fill(~keep, -1e4)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(hidden_states.size(0), hidden_states.size(1), self.d_model)
        out = self.out_proj(out)

        hidden_states = hidden_states + torch.tanh(self.attn_gate) * out
        ff = self.ffn(self.norm2(hidden_states))
        hidden_states = hidden_states + torch.tanh(self.ffn_gate) * ff

        if key_padding_mask is not None:
            hidden_states = hidden_states * key_padding_mask.unsqueeze(-1).float()
        return hidden_states


class CPCondWhisperModel(nn.Module):
    """
    Pred-only CP-conditioned Whisper.

    Pipeline:
      pred -> frozen Whisper encoder -> latent CP-conditioned blocks -> Whisper decoder

    The conditioner receives pred + normalized CP width.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        mel_bins: int = 80,
        d_cond: int = 128,
        cond_feature_groups: int = 6,
        num_latent_blocks: int = 4,
        num_heads: int = 8,
        freeze_whisper: bool = True,
    ):
        super().__init__()
        self.base = WhisperForConditionalGeneration.from_pretrained(model_name)
        if freeze_whisper:
            for p in self.base.parameters():
                p.requires_grad = False

        d_model = self.base.model.config.d_model
        self.conditioner = CPConditioner(
            mel_bins=mel_bins,
            d_cond=d_cond,
            cond_feature_groups=cond_feature_groups,
        )
        self.latent_blocks = nn.ModuleList(
            [LatentCondSABlock(d_model=d_model, d_cond=d_cond, num_heads=num_heads) for _ in range(num_latent_blocks)]
        )

    def encode_and_condition(
        self,
        pred: torch.Tensor,
        cp_features: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.base.model.encoder(input_features=pred, return_dict=True)
        base_hidden = enc.last_hidden_state  # [B, S, D]
        seq_len = base_hidden.size(1)
        cond_local, cond_global = self.conditioner(pred, cp_features, input_lengths, seq_len)

        enc_lengths = expected_encoder_len(input_lengths).clamp(max=seq_len)
        enc_mask = make_time_mask(enc_lengths, seq_len)

        hidden = base_hidden
        for block in self.latent_blocks:
            hidden = block(hidden, cond_local, cond_global, key_padding_mask=enc_mask)
        hidden = hidden * enc_mask.unsqueeze(-1).float()
        return base_hidden, hidden, enc_mask

    def forward(
        self,
        pred: torch.Tensor,
        cp_features: torch.Tensor,
        input_lengths: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base_hidden, refined_hidden, enc_mask = self.encode_and_condition(pred, cp_features, input_lengths)
        outputs = self.base(
            encoder_outputs=BaseModelOutput(last_hidden_state=refined_hidden),
            attention_mask=enc_mask.long(),
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "base_hidden": base_hidden,
            "refined_hidden": refined_hidden,
            "enc_mask": enc_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        pred: torch.Tensor,
        cp_features: torch.Tensor,
        input_lengths: torch.Tensor,
        **generate_kwargs,
    ) -> torch.Tensor:
        _, refined_hidden, enc_mask = self.encode_and_condition(pred, cp_features, input_lengths)
        return self.base.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=refined_hidden),
            attention_mask=enc_mask.long(),
            **generate_kwargs,
        )
