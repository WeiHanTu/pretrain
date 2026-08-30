"""Single-process training loop.

The same loop runs at world size one and under a process group; rank-0-only
behaviour is a branch, not a separate code path, so the single-GPU reference and
the distributed runs cannot silently diverge.

Two caps are enforced on every run, not just cloud ones: ``max_steps`` and
``max_wall_seconds``.  A runaway loop on a laptop wastes an afternoon; the same
loop on rented GPUs bills for it.  Making the cap unconditional means the cloud
path is exercised locally rather than bolted on when it matters.

Sample IDs consumed at each step are recorded as the loop runs, so the coverage
invariant is checked against what was *actually* pulled rather than what the
sampler intended.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from pretrainmodel.config import Config
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.loader import LoaderState, ShardedSampler
from pretrainmodel.model.objective import forecast_metrics, masked_mae
from pretrainmodel.observability.events import EventLog
from pretrainmodel.observability.timing import (
    PhaseTimer,
    TimingAccumulator,
    memory_snapshot,
    reset_memory_stats,
)

__all__ = [
    "StepRecord",
    "TrainResult",
    "build_optimizer",
    "learning_rate_at",
    "seed_everything",
    "train",
]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed every RNG the loop touches.

    ``deterministic`` additionally forbids nondeterministic kernels.  It is
    required for the bit-exactness oracle and costs throughput, so it is opt-in
    and driven by config rather than always on.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def learning_rate_at(step: int, cfg: Config) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio * lr``.

    A pure function of the step so a resumed run reconstructs the schedule exactly
    rather than restarting it -- a restarted warmup after a resume is a classic
    silent discontinuity in the loss curve.
    """
    warmup = cfg.optim.warmup_steps
    total = cfg.optim.total_steps
    peak = cfg.optim.lr
    floor = peak * cfg.optim.min_lr_ratio

    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    loss: float
    grad_norm: float
    lr: float
    seconds: float
    sample_ids: list[str]
    phases: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainResult:
    run_id: str
    steps: list[StepRecord] = field(default_factory=list)
    consumed_ids: list[str] = field(default_factory=list)
    final_loss: float = float("nan")
    stopped_because: str = ""
    step_times: list[float] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def losses(self) -> list[float]:
        return [s.loss for s in self.steps]


def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    """Construct the optimizer.

    Exposed separately because under FSDP2 the optimizer must be created *after*
    the model is sharded -- it has to see DTensor parameters, not the pre-shard
    ones -- and because a resume needs to load state into the same instance the
    loop will step.
    """
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr,
        betas=(cfg.optim.betas[0], cfg.optim.betas[1]),
        weight_decay=cfg.optim.weight_decay,
    )


def train(
    cfg: Config,
    model: nn.Module,
    dataset: WindowDataset,
    sampler: ShardedSampler,
    *,
    run_id: str = "local",
    events: EventLog | None = None,
    device: torch.device | None = None,
    epoch: int = 0,
    start_step: int = 0,
    resume_from: LoaderState | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    on_step: Callable[[int, LoaderState], None] | None = None,
    enable_timing: bool = True,
) -> TrainResult:
    """Run optimizer steps from ``start_step`` up to ``cfg.train.max_steps``.

    ``start_step`` is the *global* step, so a resumed run reconstructs the learning
    rate from the schedule rather than restarting warmup -- the classic silent
    discontinuity across a resume boundary.

    ``on_step`` is called after each completed step with the step index and the
    loader state at that boundary, which is where checkpointing hooks in. Keeping
    checkpoint policy out of the loop means the loop stays identical between a
    control run and a resumed one.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.train()

    if optimizer is None:
        optimizer = build_optimizer(cfg, model)
    sensor_ids = torch.arange(cfg.data.num_sensors, dtype=torch.long, device=device)
    autocast_dtype = torch.bfloat16 if cfg.train.dtype == "bf16" else None

    rank_ids = sampler.rank_epoch_ids(epoch)
    micro = cfg.train.micro_batch_size
    accum = cfg.train.grad_accum_steps
    per_step = micro * accum

    result = TrainResult(run_id=run_id)
    timings = TimingAccumulator(method=PhaseTimer(device, enabled=enable_timing).method)
    reset_memory_stats(device)
    start = time.monotonic()
    cursor = sampler.resume_offset(resume_from) if resume_from is not None else 0

    for step in range(start_step, cfg.train.max_steps):
        if time.monotonic() - start > cfg.train.max_wall_seconds:
            result.stopped_because = "max_wall_seconds"
            break
        if cursor + per_step > len(rank_ids):
            result.stopped_because = "epoch_exhausted"
            break

        step_start = time.monotonic()
        lr = learning_rate_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_ids: list[str] = []
        loss_sum: Tensor | None = None
        timer = PhaseTimer(device, enabled=enable_timing)

        for _ in range(accum):
            with timer.phase("data_wait"):
                ids = rank_ids[cursor : cursor + micro]
                cursor += micro
                step_ids.extend(ids)
                batch = dataset.batch(ids).to(device)

            with timer.phase("forward"):
                if autocast_dtype is not None:
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        pred = model(batch.context_values, batch.context_observed, sensor_ids)
                        loss = masked_mae(pred.float(), batch.target_values, batch.target_observed)
                else:
                    pred = model(batch.context_values, batch.context_observed, sensor_ids)
                    loss = masked_mae(pred, batch.target_values, batch.target_observed)

            with timer.phase("backward"):
                # Scale so the reported loss is comparable regardless of accumulation.
                (loss / accum).backward()  # type: ignore[no-untyped-call]

            # Accumulate on device. Converting the loss to a Python float here would
            # force a device synchronisation on EVERY micro-batch, serialising the
            # pipeline on GPU and making the phase timings describe a run nobody
            # would actually want to do. It is read once, after finalize().
            detached = loss.detach()
            loss_sum = detached if loss_sum is None else loss_sum + detached

        with timer.phase("optimizer"):
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
            optimizer.step()

        # The single synchronisation per step, at a boundary the optimizer already
        # orders. Reads after it are free.
        phase_durations = timer.finalize()
        grad_norm = float(grad_norm_t)
        total_loss = float(loss_sum) / accum if loss_sum is not None else float("nan")

        record = StepRecord(
            step=step,
            loss=total_loss,
            grad_norm=grad_norm,
            lr=lr,
            seconds=time.monotonic() - step_start,
            sample_ids=step_ids,
            phases=phase_durations,
        )
        result.steps.append(record)
        timings.add(record.seconds, phase_durations)
        result.consumed_ids.extend(step_ids)

        if events and (step % max(1, cfg.train.log_every) == 0 or step == cfg.train.max_steps - 1):
            events.emit(
                "train_step",
                step=step,
                loss=record.loss,
                grad_norm=record.grad_norm,
                lr=record.lr,
                seconds=record.seconds,
                samples=len(step_ids),
            )

        if on_step is not None:
            on_step(
                step + 1,
                sampler.state_dict(epoch=epoch, rank_samples_consumed=cursor),
            )

        if not math.isfinite(total_loss):
            result.stopped_because = "non_finite_loss"
            if events:
                events.emit("non_finite_loss", step=step, loss=total_loss)
            break
    else:
        result.stopped_because = "max_steps"

    if result.steps:
        result.final_loss = result.steps[-1].loss
    result.metrics = {
        "steps_completed": len(result.steps),
        "samples_consumed": len(result.consumed_ids),
        "final_loss": result.final_loss,
        "stopped_because": result.stopped_because,
    }
    if result.steps:
        elapsed = sum(s.seconds for s in result.steps)
        summary = timings.summary()
        step_total = summary["step_total"]
        result.timing = summary
        result.metrics["step_time_p50"] = step_total["p50"]
        result.metrics["step_time_p95"] = step_total["p95"]
        result.metrics["samples_per_second"] = len(result.consumed_ids) / max(1e-9, elapsed)
        result.metrics["timing_method"] = summary["timing_method"]
        result.metrics["step_time_breakdown"] = {
            name: stats["p50"] for name, stats in summary["phases"].items()
        }
        result.metrics["unaccounted_fraction"] = summary["unaccounted_fraction"]
        result.metrics.update(memory_snapshot(device))
        result.step_times = [s.seconds for s in result.steps]
    return result


@torch.no_grad()
def evaluate(
    cfg: Config,
    model: nn.Module,
    dataset: WindowDataset,
    sample_ids: list[str],
    *,
    device: torch.device | None = None,
    batch_size: int = 8,
) -> dict[str, float]:
    """Masked forecast metrics over the given samples."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    sensor_ids = torch.arange(cfg.data.num_sensors, dtype=torch.long, device=device)

    preds: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    for i in range(0, len(sample_ids), batch_size):
        batch = dataset.batch(sample_ids[i : i + batch_size]).to(device)
        preds.append(model(batch.context_values, batch.context_observed, sensor_ids))
        targets.append(batch.target_values)
        masks.append(batch.target_observed)

    model.train()
    if not preds:
        return {}
    return dict(
        forecast_metrics(
            torch.cat(preds), torch.cat(targets), torch.cat(masks), horizons=(1, 3, 6, 12)
        )
    )
