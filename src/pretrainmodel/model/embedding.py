"""Input embedding for spatiotemporal sensor windows.

Three signals are combined: the observed value, *which sensor* it came from, and
*when* it was measured.

Missingness is represented explicitly rather than imputed.  Traffic sensors drop
out constantly, and a missing reading is not the same as zero flow: substituting
zero teaches the model that outages are congestion.  Missing positions are zeroed
for numerical safety and then marked with a learned missing-embedding, so the
model can condition on "no reading here" as a first-class input.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from pretrainmodel.config import DataConfig, ModelConfig

__all__ = ["SpatiotemporalEmbedding", "sinusoidal_positions"]


def sinusoidal_positions(length: int, dim: int, device: torch.device) -> Tensor:
    """Standard sinusoidal position table, shape ``(length, dim)``."""
    if dim % 2 != 0:
        raise ValueError(f"position dim must be even, got {dim}")
    position = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    scale = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-math.log(10_000.0) / dim)
    )
    table = torch.zeros(length, dim, dtype=torch.float32, device=device)
    table[:, 0::2] = torch.sin(position * scale)
    table[:, 1::2] = torch.cos(position * scale)
    return table


class SpatiotemporalEmbedding(nn.Module):
    """Project ``(B, T, S, F)`` sensor readings to ``(B, T, S, d_model)``."""

    positions: Tensor

    def __init__(self, model: ModelConfig, data: DataConfig) -> None:
        super().__init__()
        self.d_model = model.d_model
        self.num_features = data.num_features
        self.value_proj = nn.Linear(data.num_features, model.d_model)
        self.sensor_embedding = nn.Embedding(model.max_sensors, model.d_model)
        self.missing_embedding = nn.Parameter(torch.zeros(model.d_model))
        self.time_features = model.time_features
        self.dropout = nn.Dropout(model.dropout)
        self.register_buffer(
            "positions",
            sinusoidal_positions(data.context_steps, model.d_model, torch.device("cpu")),
            persistent=False,
        )

    def forward(self, values: Tensor, observed: Tensor, sensor_ids: Tensor) -> Tensor:
        """
        Args:
            values: ``(B, T, S, F)``; missing entries may be NaN.
            observed: ``(B, T, S)`` boolean; True where a real reading exists.
            sensor_ids: ``(S,)`` long; global sensor index for the embedding table.
        """
        if values.ndim != 4:
            raise ValueError(f"values must be (B, T, S, F), got {tuple(values.shape)}")
        if observed.shape != values.shape[:3]:
            raise ValueError(
                f"observed must be {tuple(values.shape[:3])}, got {tuple(observed.shape)}"
            )
        if sensor_ids.shape[0] != values.shape[2]:
            raise ValueError(
                f"sensor_ids must have {values.shape[2]} entries, got {sensor_ids.shape[0]}"
            )

        # NaN must not reach the projection: a single NaN poisons every downstream
        # activation through the attention softmax.
        safe = torch.nan_to_num(values, nan=0.0)
        safe = safe * observed.unsqueeze(-1).to(safe.dtype)

        h = self.value_proj(safe)
        h = h + self.sensor_embedding(sensor_ids).unsqueeze(0).unsqueeze(0)

        if self.time_features:
            positions = self.positions[: values.shape[1]].to(h.dtype)
            h = h + positions.unsqueeze(0).unsqueeze(2)

        missing = (~observed).unsqueeze(-1).to(h.dtype)
        h = h + missing * self.missing_embedding.to(h.dtype)
        out: Tensor = self.dropout(h)
        return out
