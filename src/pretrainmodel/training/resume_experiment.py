"""Run the resume oracles end to end (plan.md B3, B4).

Three phases, each a *separate* set of OS processes:

1. **control**   -- N steps, uninterrupted.
2. **interrupt** -- N/2 steps, then a committed checkpoint. The processes exit.
3. **resume**    -- fresh processes load the checkpoint and run N/2 -> N.

The processes must genuinely be new.  Resuming inside the same process would keep
warm CUDA/CPU allocator state, live RNG objects and an already-constructed
optimizer, none of which exist after a real crash -- so an in-process "resume" test
can pass while the actual recovery path is broken.

Phase 3 may use a different world size, which is what makes this both the B3
(same size, bit-exact) and B4 (resharded, statistical) experiment.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pretrainmodel.config import Config, config_hash, load_config
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.loader import LoaderState, ShardedSampler
from pretrainmodel.data.manifest import ShardManifest, manifest_hash
from pretrainmodel.data.shard import build_index
from pretrainmodel.distributed.checkpoint import (
    latest_committed,
    load_checkpoint,
    save_checkpoint,
)
from pretrainmodel.distributed.parallel import build_device_mesh, shard_model, sharding_report
from pretrainmodel.model.transformer import build_model
from pretrainmodel.training.compare import full_snapshot
from pretrainmodel.training.loop import TrainResult, build_optimizer, seed_everything, train

__all__ = ["PhaseResult", "run_phase", "run_resume_experiment"]


class _SimulatedInterrupt(RuntimeError):
    """Raised to stop the interrupt phase at its checkpoint step.

    The interrupt phase runs the *same* config as the control -- an interruption
    stops a run, it does not rewrite the step budget. Shortening max_steps instead
    would change the config hash, and the checkpoint provenance guard would
    (correctly) refuse the resume.
    """


def _prepare(
    cfg: Config, root: Path, rank: int, world_size: int
) -> tuple[WindowDataset, ShardedSampler, ShardManifest]:
    manifest = ShardManifest.read(root / "shards.json")
    manifest.verify(root)
    index = build_index(manifest)
    dataset = WindowDataset(root, manifest, index)
    sampler = ShardedSampler(
        index.sample_ids,
        seed=cfg.run.seed,
        world_size=world_size,
        rank=rank,
        shuffle=cfg.data.shuffle,
        drop_last=cfg.data.drop_last,
        dataset_version=manifest.dataset_version,
    )
    return dataset, sampler, manifest


def _worker(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    phase: str,
    config_path: str,
    data_root: str,
    ckpt_root: str,
    max_steps: int,
    checkpoint_at: int,
    resume_defect: str,
) -> None:
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        base = load_config(config_path)
        cfg = replace(base, train=replace(base.train, max_steps=max_steps))
        seed_everything(cfg.run.seed, deterministic=cfg.determinism.deterministic_algorithms)
        interrupted_at: int | None = None

        dataset, sampler, manifest = _prepare(cfg, Path(data_root), rank, world_size)
        mesh = build_device_mesh()
        model = shard_model(build_model(cfg), mesh)
        optimizer = build_optimizer(cfg, model)

        start_step = 0
        resume_from: LoaderState | None = None
        rng_continuity = True
        resharded = False
        load_note = ""

        if phase == "resume":
            ckpt = latest_committed(ckpt_root)
            if ckpt is None:
                raise RuntimeError(f"no committed checkpoint under {ckpt_root}")
            loaded = load_checkpoint(
                ckpt,
                model=model,
                optimizer=optimizer,
                seed=cfg.run.seed,
                config_hash=config_hash(cfg),
                data_manifest_hash=manifest_hash(manifest),
            )
            start_step = loaded.step
            resume_from = loaded.loader_state
            rng_continuity = loaded.rng_continuity
            resharded = loaded.resharded
            load_note = loaded.note

            # Injected resume defects (negative controls). Each is a real bug people
            # ship, and each leaves a plausible loss curve that a tolerance-based
            # comparison would accept.
            if resume_defect == "reset_optimizer":
                # Optimizer state discarded: AdamW's moments and per-parameter step
                # counter restart, so bias correction is wrong for thousands of steps.
                optimizer = build_optimizer(cfg, model)
            elif resume_defect == "ignore_loader_state":
                # Dataloader cursor not restored: the epoch replays from the top and
                # the samples after the boundary are consumed twice.
                resume_from = None

        def on_step(step: int, loader_state: LoaderState) -> None:
            nonlocal interrupted_at
            if phase == "interrupt" and step == checkpoint_at:
                save_checkpoint(
                    ckpt_root,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    epoch=0,
                    loader_state=loader_state,
                    config_hash=config_hash(cfg),
                    data_manifest_hash=manifest_hash(manifest),
                    run_id=f"{phase}-ws{world_size}",
                )
                interrupted_at = step
                raise _SimulatedInterrupt(f"interrupted after step {step}")

        try:
            result = train(
                cfg,
                model,
                dataset,
                sampler,
                run_id=f"{phase}-ws{world_size}-r{rank}",
                optimizer=optimizer,
                start_step=start_step,
                resume_from=resume_from,
                on_step=on_step,
            )
        except _SimulatedInterrupt:
            # These processes stop here. The resume runs in genuinely new ones, which
            # is what makes this a recovery test rather than a loop restart.
            result = TrainResult(run_id=f"{phase}-ws{world_size}-r{rank}")
            result.stopped_because = "simulated_interrupt"

        gathered_losses: list[list[float] | None] = [None] * world_size
        gathered_ids: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(gathered_losses, result.losses)
        dist.all_gather_object(gathered_ids, result.consumed_ids)

        snapshot = full_snapshot(model, optimizer)
        report = sharding_report(model)

        if rank == 0:
            torch.save(
                {
                    "phase": phase,
                    "world_size": world_size,
                    "start_step": start_step,
                    "max_steps": max_steps,
                    "losses_by_rank": [x or [] for x in gathered_losses],
                    "ids_by_rank": [x or [] for x in gathered_ids],
                    "expected_epoch_ids": sampler.expected_epoch_ids(0),
                    "snapshot": snapshot,
                    "sharding": report,
                    "rng_continuity": rng_continuity,
                    "resharded": resharded,
                    "load_note": load_note,
                    "interrupted_at": interrupted_at,
                    "config_hash": config_hash(cfg),
                    "resume_defect": resume_defect,
                },
                Path(out_dir) / f"{phase}-ws{world_size}.pt",
            )
    finally:
        dist.destroy_process_group()


class PhaseResult(dict[str, Any]):
    """Deserialised output of one phase."""


def run_phase(
    phase: str,
    *,
    world_size: int,
    out_dir: Path,
    config_path: Path,
    data_root: Path,
    ckpt_root: Path,
    max_steps: int,
    checkpoint_at: int,
    resume_defect: str = "none",
) -> PhaseResult:
    """Spawn one phase and return what rank 0 recorded."""
    with tempfile.TemporaryDirectory() as rdv:
        mp.spawn(  # type: ignore[attr-defined,no-untyped-call]
            _worker,
            args=(
                world_size,
                os.path.join(rdv, "rendezvous"),
                str(out_dir),
                phase,
                str(config_path),
                str(data_root),
                str(ckpt_root),
                max_steps,
                checkpoint_at,
                resume_defect,
            ),
            nprocs=world_size,
            join=True,
        )
    payload = torch.load(out_dir / f"{phase}-ws{world_size}.pt", weights_only=False)
    return PhaseResult(payload)


def run_resume_experiment(
    *,
    config_path: Path,
    data_root: Path,
    workdir: Path,
    control_world_size: int,
    resume_world_size: int,
    max_steps: int,
    checkpoint_at: int,
    resume_defect: str = "none",
) -> dict[str, PhaseResult]:
    """Run control, interrupt and resume; return all three phase outputs."""
    workdir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "phases"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = workdir / "checkpoints"

    control = run_phase(
        "control",
        world_size=control_world_size,
        out_dir=out_dir,
        config_path=config_path,
        data_root=data_root,
        ckpt_root=ckpt_root,
        max_steps=max_steps,
        checkpoint_at=checkpoint_at,
    )
    run_phase(
        "interrupt",
        world_size=control_world_size,
        out_dir=out_dir,
        config_path=config_path,
        data_root=data_root,
        ckpt_root=ckpt_root,
        max_steps=max_steps,
        checkpoint_at=checkpoint_at,
    )
    resumed = run_phase(
        "resume",
        world_size=resume_world_size,
        out_dir=out_dir,
        config_path=config_path,
        data_root=data_root,
        ckpt_root=ckpt_root,
        max_steps=max_steps,
        checkpoint_at=checkpoint_at,
        resume_defect=resume_defect,
    )
    return {"control": control, "resume": resumed}


def json_safe(obj: Any) -> Any:
    """Convert tensors to plain values for artifact serialisation."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")
