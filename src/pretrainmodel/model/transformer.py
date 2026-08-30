"""Factorised spatiotemporal Transformer.

Full attention over ``T x S`` jointly costs ``O((T*S)^2)``, which for 12 steps and
1024 sensors is a 12288-token sequence per sample -- untenable at this budget.
Each block therefore attends over time (per sensor) and then over sensors (per
timestep), which is ``O(T^2 * S + S^2 * T)``.  This is the standard factorisation
and it keeps the 20M-50M parameter target reachable.

The forecasting objective is *direct multi-horizon*: the encoder sees the whole
context bidirectionally and a head emits all H future steps at once.  Causal
masking is therefore off by default and is not a silent omission -- it is
available via ``causal=True`` for an autoregressive variant, and is tested, but a
causal mask over the context would only handicap a direct forecaster.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pretrainmodel.config import Config, DataConfig, ModelConfig
from pretrainmodel.model.embedding import SpatiotemporalEmbedding

__all__ = ["SpatiotemporalTransformer", "build_model", "causal_mask", "count_parameters"]


def causal_mask(length: int, device: torch.device) -> Tensor:
    """Additive float mask, ``(length, length)``, blocking attention to the future."""
    return torch.triu(torch.full((length, length), float("-inf"), device=device), diagonal=1)


class _Attention(nn.Module):
    """Multi-head self-attention over one axis."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        result: Tensor = x + out
        return result


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mult, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = x + self.net(self.norm(x))
        return result


class _Block(nn.Module):
    """Temporal attention, then spatial attention, then a feed-forward layer."""

    def __init__(self, model: ModelConfig) -> None:
        super().__init__()
        self.temporal = _Attention(model.d_model, model.num_heads, model.dropout)
        self.spatial = _Attention(model.d_model, model.num_heads, model.dropout)
        self.ffn = _FeedForward(model.d_model, model.ffn_mult, model.dropout)

    def forward(self, x: Tensor, temporal_mask: Tensor | None) -> Tensor:
        b, t, s, d = x.shape
        # (B, T, S, D) -> (B*S, T, D): every sensor's own time series.
        h = x.permute(0, 2, 1, 3).reshape(b * s, t, d)
        h = self.temporal(h, attn_mask=temporal_mask)
        x = h.reshape(b, s, t, d).permute(0, 2, 1, 3)

        # (B, T, S, D) -> (B*T, S, D): every timestep's sensor cross-section.
        h = x.reshape(b * t, s, d)
        h = self.spatial(h)
        x = h.reshape(b, t, s, d)

        out: Tensor = self.ffn(x)
        return out


class SpatiotemporalTransformer(nn.Module):
    """Compact encoder with a direct multi-horizon forecast head."""

    def __init__(self, model: ModelConfig, data: DataConfig, *, causal: bool = False) -> None:
        super().__init__()
        self.context_steps = data.context_steps
        self.horizon_steps = data.horizon_steps
        self.num_features = data.num_features
        self.causal = causal

        self.embedding = SpatiotemporalEmbedding(model, data)
        self.blocks = nn.ModuleList(_Block(model) for _ in range(model.num_layers))
        self.norm = nn.LayerNorm(model.d_model)
        self.head = nn.Linear(model.d_model, data.horizon_steps * data.num_features)

    def forward(self, values: Tensor, observed: Tensor, sensor_ids: Tensor) -> Tensor:
        """Return the forecast, shape ``(B, horizon_steps, S, F)``."""
        if values.shape[1] != self.context_steps:
            raise ValueError(f"expected {self.context_steps} context steps, got {values.shape[1]}")
        x = self.embedding(values, observed, sensor_ids)
        mask = causal_mask(self.context_steps, values.device) if self.causal else None
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)

        # Forecast from the final context step's representation.
        last = x[:, -1]  # (B, S, D)
        out = self.head(last)  # (B, S, H*F)
        b, s, _ = out.shape
        # .contiguous() is load-bearing under FSDP2, not cosmetic: a wrapped module
        # that returns a *view* can have its pre-backward hook dropped by a later
        # in-place op, which skips the parameter all-gather and silently produces
        # wrong gradients rather than raising.
        forecast: Tensor = (
            out.reshape(b, s, self.horizon_steps, self.num_features)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        return forecast


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Parameter count, emitted into every run manifest (spec 5.1)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


def build_model(cfg: Config, *, causal: bool = False) -> SpatiotemporalTransformer:
    return SpatiotemporalTransformer(cfg.model, cfg.data, causal=causal)
