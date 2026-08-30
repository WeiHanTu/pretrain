"""Masked multi-horizon forecasting loss.

Missing targets are excluded from the loss and from every reported metric using
*the same mask*.  Two failure modes motivate this.  Scoring a fabricated value
teaches the model to predict the imputation rather than the road.  Scoring one way
and reporting another produces a validation number that no loss ever optimised --
the metric drifts from the objective and nobody notices until the model is
deployed.

An all-missing batch returns zero loss with zero gradient rather than NaN, so a
sensor outage cannot silently destroy a run's weights.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["ForecastMetrics", "forecast_metrics", "masked_mae", "masked_mse", "masked_rmse"]


def _prepare(pred: Tensor, target: Tensor, observed: Tensor) -> tuple[Tensor, Tensor]:
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} != target {tuple(target.shape)}")
    if observed.shape != target.shape[:3]:
        raise ValueError(f"observed must be {tuple(target.shape[:3])}, got {tuple(observed.shape)}")
    mask = observed.unsqueeze(-1).to(pred.dtype)
    # NaN in a masked-out target still propagates through multiplication, so the
    # target is sanitised before the mask is applied.
    safe_target = torch.nan_to_num(target, nan=0.0)
    return mask, safe_target


def masked_mae(pred: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    mask, safe_target = _prepare(pred, target, observed)
    denom = mask.sum()
    if denom == 0:
        return pred.sum() * 0.0
    return ((pred - safe_target).abs() * mask).sum() / denom


def masked_mse(pred: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    mask, safe_target = _prepare(pred, target, observed)
    denom = mask.sum()
    if denom == 0:
        return pred.sum() * 0.0
    return ((pred - safe_target).pow(2) * mask).sum() / denom


def masked_rmse(pred: Tensor, target: Tensor, observed: Tensor) -> Tensor:
    return torch.sqrt(masked_mse(pred, target, observed) + 1e-12)


class ForecastMetrics(dict[str, float]):
    """Plain dict of scalar metrics, JSON-serialisable for the run manifest."""


def forecast_metrics(
    pred: Tensor, target: Tensor, observed: Tensor, *, horizons: tuple[int, ...] = ()
) -> ForecastMetrics:
    """Overall MAE/RMSE plus per-horizon MAE at the declared horizons (spec 5.2)."""
    metrics = ForecastMetrics(
        mae=float(masked_mae(pred, target, observed)),
        rmse=float(masked_rmse(pred, target, observed)),
        observed_fraction=float(observed.to(pred.dtype).mean()),
    )
    for h in horizons:
        if 1 <= h <= pred.shape[1]:
            idx = h - 1
            metrics[f"mae_h{h}"] = float(
                masked_mae(
                    pred[:, idx : idx + 1],
                    target[:, idx : idx + 1],
                    observed[:, idx : idx + 1],
                )
            )
    return metrics
