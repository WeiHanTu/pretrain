"""Periodic checkpointing for the real training path.

This module exists because of a gap that survived all of Phase B: ``save_checkpoint``
was called only from the resume *experiment*, so the B3/B4 oracles proved the
checkpoint contract worked while no actual training run ever wrote one.
``checkpoint.every_steps`` was parsed, validated, reported on by the preflight --
and honoured by nothing.

That is the difference between "the mechanism is correct" and "the mechanism is
installed", and only the second one survives a spot preemption.

Checkpoint policy lives here rather than inside ``train()`` so the loop stays
identical between a control run and a resumed one; the loop calls an ``on_step``
hook and does not know or care what it does.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from pretrainmodel.config import Config, config_hash
from pretrainmodel.data.loader import LoaderState
from pretrainmodel.distributed.checkpoint import (
    LoadedCheckpoint,
    latest_committed,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from pretrainmodel.observability.events import EventLog

__all__ = ["PeriodicCheckpointer", "resume_if_available"]


class PeriodicCheckpointer:
    """Save every ``cfg.checkpoint.every_steps`` steps; install as ``on_step``.

    ``every_steps = 0`` disables saving. That is a legitimate choice for a throwaway
    run, but it is never the right one on preemptible hardware, which is why the
    preflight checks the cadence *and* proves a checkpoint actually lands.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        root: Path | str | None = None,
        data_manifest_hash: str = "",
        run_id: str = "",
        epoch: int = 0,
        events: EventLog | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.root = Path(root) if root is not None else Path(cfg.checkpoint.dir)
        self.data_manifest_hash = data_manifest_hash
        self.run_id = run_id
        self.epoch = epoch
        self.events = events
        self.saved_steps: list[int] = []
        self._config_hash = config_hash(cfg)

    @property
    def enabled(self) -> bool:
        return self.cfg.checkpoint.every_steps > 0

    def __call__(self, step: int, loader_state: LoaderState) -> None:
        if self.enabled and step % self.cfg.checkpoint.every_steps == 0:
            self.save_now(step, loader_state, reason="periodic")

    def save_now(
        self, step: int, loader_state: LoaderState, *, reason: str = "manual"
    ) -> Path | None:
        """Write and commit a checkpoint, then prune older ones."""
        path = save_checkpoint(
            self.root,
            model=self.model,
            optimizer=self.optimizer,
            step=step,
            epoch=self.epoch,
            loader_state=loader_state,
            config_hash=self._config_hash,
            data_manifest_hash=self.data_manifest_hash,
            run_id=self.run_id,
        )
        self.saved_steps.append(step)
        removed = prune_checkpoints(self.root, self.cfg.checkpoint.keep_last)
        if self.events:
            self.events.emit(
                "checkpoint_committed",
                step=step,
                reason=reason,
                path=str(path),
                pruned=[p.name for p in removed],
            )
        return path


def resume_if_available(
    cfg: Config,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    root: Path | str | None = None,
    data_manifest_hash: str = "",
    events: EventLog | None = None,
) -> LoadedCheckpoint | None:
    """Resume from the newest committed checkpoint, if one exists.

    Returns ``None`` when there is nothing to resume from, which is the normal case
    for a first run. Only *committed* checkpoints are considered, so a save that was
    interrupted mid-write is skipped rather than resumed from.
    """
    path = latest_committed(Path(root) if root is not None else Path(cfg.checkpoint.dir))
    if path is None:
        return None
    loaded = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        seed=cfg.run.seed,
        config_hash=config_hash(cfg),
        data_manifest_hash=data_manifest_hash or None,
    )
    if events:
        events.emit(
            "resumed_from_checkpoint",
            step=loaded.step,
            path=str(path),
            rng_continuity=loaded.rng_continuity,
            resharded=loaded.resharded,
            note=loaded.note,
        )
    return loaded
