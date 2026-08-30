"""Model shape/dtype contracts, masking behaviour and the tiny-batch overfit gate."""

from __future__ import annotations

import pytest
import torch

from pretrainmodel.config import Config, from_mapping
from pretrainmodel.model.objective import (
    forecast_metrics,
    masked_mae,
    masked_mse,
    masked_rmse,
)
from pretrainmodel.model.transformer import build_model, causal_mask, count_parameters

B, S, F = 2, 6, 1
CTX, HOR = 8, 4


def _cfg(**overrides: object) -> Config:
    raw: dict[str, object] = {
        "run": {"name": "t"},
        "data": {
            "root": "data/x",
            "context_steps": CTX,
            "horizon_steps": HOR,
            "num_sensors": S,
            "num_features": F,
        },
        "model": {"d_model": 32, "num_layers": 2, "num_heads": 4, "max_sensors": S},
        "optim": {"lr": 1e-3, "warmup_steps": 1, "total_steps": 50},
        "train": {"micro_batch_size": 2, "max_steps": 50},
        "checkpoint": {"dir": "checkpoints/t"},
    }
    raw.update(overrides)
    return from_mapping(raw)


def _inputs(observed_all: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.randn(B, CTX, S, F)
    observed = torch.ones(B, CTX, S, dtype=torch.bool)
    if not observed_all:
        observed[:, 0, 0] = False
        values[:, 0, 0] = float("nan")
    return values, observed, torch.arange(S)


# --------------------------------------------------------------------------- #
# Contracts
# --------------------------------------------------------------------------- #


def test_forward_shape_and_dtype() -> None:
    model = build_model(_cfg())
    out = model(*_inputs())
    assert out.shape == (B, HOR, S, F)
    assert out.dtype == torch.float32


def test_forward_rejects_wrong_context_length() -> None:
    model = build_model(_cfg())
    values, observed, ids = _inputs()
    with pytest.raises(ValueError, match="context steps"):
        model(values[:, :-1], observed[:, :-1], ids)


def test_forward_rejects_mismatched_mask() -> None:
    model = build_model(_cfg())
    values, _, ids = _inputs()
    with pytest.raises(ValueError, match="observed must be"):
        model(values, torch.ones(B, CTX, S + 1, dtype=torch.bool), ids)


def test_forward_rejects_wrong_sensor_id_count() -> None:
    model = build_model(_cfg())
    values, observed, _ = _inputs()
    with pytest.raises(ValueError, match="sensor_ids"):
        model(values, observed, torch.arange(S + 1))


def test_nan_inputs_do_not_produce_nan_outputs() -> None:
    """A single missing reading must not poison the whole batch through softmax."""
    model = build_model(_cfg())
    out = model(*_inputs(observed_all=False))
    assert torch.isfinite(out).all()


def test_parameter_count_is_reported() -> None:
    model = build_model(_cfg())
    assert count_parameters(model) > 0
    assert count_parameters(model, trainable_only=True) == count_parameters(model)


# --------------------------------------------------------------------------- #
# Causal masking
# --------------------------------------------------------------------------- #


def test_causal_mask_is_upper_triangular_negative_infinity() -> None:
    mask = causal_mask(4, torch.device("cpu"))
    assert mask.shape == (4, 4)
    assert torch.isneginf(mask[0, 1])
    assert mask[1, 0] == 0.0
    assert torch.diagonal(mask).eq(0).all()


def test_causal_model_ignores_future_context() -> None:
    """With causal=True, perturbing a later step must not change earlier attention.

    Verified through the encoder rather than the head, since the head reads the
    final position and legitimately sees everything.
    """
    torch.manual_seed(0)
    model = build_model(_cfg(), causal=True).eval()
    values, observed, ids = _inputs()

    with torch.no_grad():
        x = model.embedding(values, observed, ids)
        mask = causal_mask(CTX, values.device)
        a = model.blocks[0](x, mask)

        perturbed = x.clone()
        perturbed[:, -1] += 10.0
        b = model.blocks[0](perturbed, mask)

    torch.testing.assert_close(a[:, :-1], b[:, :-1])


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #


def test_masked_loss_ignores_unobserved_positions() -> None:
    pred = torch.zeros(1, HOR, S, F)
    target = torch.zeros(1, HOR, S, F)
    target[0, 0, 0, 0] = 1000.0
    observed = torch.ones(1, HOR, S, dtype=torch.bool)
    observed[0, 0, 0] = False
    assert float(masked_mae(pred, target, observed)) == pytest.approx(0.0)


def test_masked_loss_ignores_nan_in_unobserved_targets() -> None:
    """NaN in a masked-out target must not propagate through the multiplication."""
    pred = torch.zeros(1, HOR, S, F)
    target = torch.zeros(1, HOR, S, F)
    target[0, 0, 0, 0] = float("nan")
    observed = torch.ones(1, HOR, S, dtype=torch.bool)
    observed[0, 0, 0] = False
    assert torch.isfinite(masked_mae(pred, target, observed))


def test_all_missing_batch_returns_zero_not_nan() -> None:
    """A total sensor outage must not destroy the weights with a NaN gradient."""
    pred = torch.zeros(1, HOR, S, F, requires_grad=True)
    target = torch.zeros(1, HOR, S, F)
    observed = torch.zeros(1, HOR, S, dtype=torch.bool)
    loss = masked_mae(pred, target, observed)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


def test_rmse_is_sqrt_of_mse() -> None:
    pred, target = torch.randn(1, HOR, S, F), torch.randn(1, HOR, S, F)
    observed = torch.ones(1, HOR, S, dtype=torch.bool)
    assert float(masked_rmse(pred, target, observed)) == pytest.approx(
        float(masked_mse(pred, target, observed)) ** 0.5, rel=1e-5
    )


def test_forecast_metrics_reports_per_horizon() -> None:
    pred, target = torch.randn(1, HOR, S, F), torch.randn(1, HOR, S, F)
    observed = torch.ones(1, HOR, S, dtype=torch.bool)
    m = forecast_metrics(pred, target, observed, horizons=(1, 3, 99))
    assert "mae" in m and "rmse" in m
    assert "mae_h1" in m and "mae_h3" in m
    assert "mae_h99" not in m, "horizon beyond the forecast length must be skipped"
    assert m["observed_fraction"] == pytest.approx(1.0)


def test_loss_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="!="):
        masked_mae(
            torch.zeros(1, HOR, S, F),
            torch.zeros(1, HOR + 1, S, F),
            torch.ones(1, HOR, S, dtype=torch.bool),
        )
