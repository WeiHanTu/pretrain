"""Tiny-batch overfit gate (plan.md A3).

The cheapest possible sanity check on a training stack: a small model, given one
fixed batch repeatedly, must be able to memorise it.  Failure localises the bug to
the model, the masking or the optimizer -- never to the data pipeline or the
schedule -- which is exactly the ambiguity that makes a slow-converging run
expensive to debug on rented hardware.

The threshold is a *predeclared* fraction of the initial loss, recorded in the
artifact alongside the result, so the gate cannot be quietly loosened after seeing
a disappointing number.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from pretrainmodel.config import Config
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.manifest import ShardManifest
from pretrainmodel.data.shard import build_index
from pretrainmodel.model.objective import masked_mae
from pretrainmodel.model.transformer import build_model, count_parameters
from pretrainmodel.training.loop import seed_everything

__all__ = ["run_overfit_gate"]


def run_overfit_gate(
    cfg: Config,
    *,
    repo_root: Path,
    steps: int = 300,
    threshold: float = 0.05,
    lr: float = 3e-3,
) -> dict[str, Any]:
    """Drive a fixed batch's loss below ``threshold * initial_loss``."""
    seed_everything(cfg.run.seed, deterministic=False)
    root = Path(cfg.data.root)
    if not root.is_absolute():
        root = repo_root / root
    manifest = ShardManifest.read(root / "shards.json")
    manifest.verify(root)
    index = build_index(manifest)
    dataset = WindowDataset(root, manifest, index)

    batch = dataset.batch(index.sample_ids[: cfg.train.micro_batch_size])
    model = build_model(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sensor_ids = torch.arange(cfg.data.num_sensors, dtype=torch.long)

    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = model(batch.context_values, batch.context_observed, sensor_ids)
        loss = masked_mae(pred, batch.target_values, batch.target_observed)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))

    initial = losses[0]
    final = min(losses[-10:])
    target = threshold * initial
    return {
        "gate": "tiny-batch-overfit",
        "config": cfg.run.name,
        "parameter_count": count_parameters(model),
        "batch_size": len(batch),
        "steps": steps,
        "learning_rate": lr,
        "threshold": threshold,
        "threshold_note": (
            "final loss must fall below threshold * initial_loss; "
            "declared before the run and recorded here so it cannot be loosened after the fact"
        ),
        "initial_loss": initial,
        "final_loss": final,
        "target_loss": target,
        "loss_curve_sampled": losses[:: max(1, steps // 20)],
        "passed": bool(math.isfinite(final) and final < target),
    }
