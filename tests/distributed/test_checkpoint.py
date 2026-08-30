"""Checkpoint contract: atomic commit, discovery, provenance and pruning.

Runs under a real Gloo process group because the commit protocol is a collective:
the marker must be written only after *every* rank's output has landed, and that
ordering cannot be tested in a single process.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytestmark = pytest.mark.distributed

CONFIG_HASH = "a" * 64
OTHER_HASH = "b" * 64


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))


def _worker(rank: int, world_size: int, init_file: str, shared: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        from pretrainmodel.config import from_mapping
        from pretrainmodel.data.loader import LoaderState
        from pretrainmodel.distributed.checkpoint import (
            COMMIT_MARKER,
            CheckpointError,
            latest_committed,
            list_committed,
            load_checkpoint,
            prune_checkpoints,
            read_marker,
            save_checkpoint,
        )
        from pretrainmodel.distributed.parallel import build_device_mesh, shard_model
        from pretrainmodel.training.loop import build_optimizer

        cfg = from_mapping(
            {
                "run": {"name": "ck"},
                "data": {"root": "x", "context_steps": 4, "horizon_steps": 4, "num_sensors": 4},
                "model": {"d_model": 8, "num_layers": 1, "num_heads": 2, "max_sensors": 4},
                "optim": {"lr": 1e-3, "warmup_steps": 1, "total_steps": 10},
                "train": {"micro_batch_size": 1, "max_steps": 5},
                "checkpoint": {"dir": "x"},
            }
        )
        root = Path(shared) / "ckpts"
        mesh = build_device_mesh()
        model = shard_model(_model(), mesh)
        opt = build_optimizer(cfg, model)
        model(torch.randn(2, 8)).sum().backward()
        opt.step()
        ls = LoaderState(
            epoch=0, global_samples_consumed=4 * world_size, seed=1, shuffle=True, drop_last=True
        )

        results: dict[str, Any] = {}

        for step in (10, 20, 30):
            save_checkpoint(
                root,
                model=model,
                optimizer=opt,
                step=step,
                epoch=0,
                loader_state=ls,
                config_hash=CONFIG_HASH,
                run_id="ck",
            )

        results["committed_steps"] = [s for s, _ in list_committed(root)]
        results["latest"] = latest_committed(root).name
        marker = read_marker(root / "step-00000030")
        results["marker_world_size"] = marker.world_size
        results["marker_sharded"] = marker.sharded
        results["marker_step"] = marker.step

        # An interrupted save: the DCP payload exists but was never committed.
        if rank == 0:
            (root / "step-00000040").mkdir(parents=True, exist_ok=True)
            (root / "step-00000040" / "__0_0.distcp").write_bytes(b"partial")
        dist.barrier()
        results["uncommitted_ignored"] = latest_committed(root).name == "step-00000030"

        try:
            load_checkpoint(root / "step-00000040", model=model, optimizer=opt, seed=1)
            results["uncommitted_load_raised"] = False
        except CheckpointError as exc:
            results["uncommitted_load_raised"] = COMMIT_MARKER in str(exc)

        # Provenance guard.
        try:
            load_checkpoint(
                root / "step-00000030", model=model, optimizer=opt, seed=1, config_hash=OTHER_HASH
            )
            results["config_guard_raised"] = False
        except CheckpointError:
            results["config_guard_raised"] = True

        loaded = load_checkpoint(
            root / "step-00000030",
            model=model,
            optimizer=opt,
            seed=1,
            config_hash=OTHER_HASH,
            allow_config_change=True,
        )
        results["override_works"] = loaded.step == 30
        results["rng_continuity_same_ws"] = loaded.rng_continuity
        results["resharded_same_ws"] = loaded.resharded
        results["loader_state_restored"] = (
            loaded.loader_state.global_samples_consumed == 4 * world_size
        )

        # Corrupt marker -> treated as uncommitted, not as a usable checkpoint.
        if rank == 0:
            (root / "step-00000030" / COMMIT_MARKER).write_text("{not json")
        dist.barrier()
        results["corrupt_marker_ignored"] = latest_committed(root).name == "step-00000020"

        pruned = prune_checkpoints(root, keep_last=1)
        dist.barrier()
        results["pruned"] = sorted(p.name for p in pruned)
        results["after_prune"] = [s for s, _ in list_committed(root)]
        results["uncommitted_survived_prune"] = (root / "step-00000040").is_dir()

        if rank == 0:
            Path(shared, "results.json").write_text(json.dumps(results, sort_keys=True))
    finally:
        dist.destroy_process_group()


def _run(world_size: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        mp.spawn(  # type: ignore[attr-defined,no-untyped-call]
            _worker,
            args=(world_size, os.path.join(td, "rdv"), td),
            nprocs=world_size,
            join=True,
        )
        data: dict[str, Any] = json.loads(Path(td, "results.json").read_text())
        return data


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    return _run(2)


def test_saved_checkpoints_are_discoverable(results: dict[str, Any]) -> None:
    assert results["committed_steps"] == [10, 20, 30]
    assert results["latest"] == "step-00000030"


def test_marker_records_topology_and_sharding(results: dict[str, Any]) -> None:
    assert results["marker_world_size"] == 2
    assert results["marker_sharded"] is True
    assert results["marker_step"] == 30


def test_uncommitted_checkpoint_is_never_selected(results: dict[str, Any]) -> None:
    """A half-written directory must not become 'latest'.

    Otherwise a spot-VM reclaim mid-save silently poisons the next resume, and the
    failure presents as model divergence rather than a truncated write.
    """
    assert results["uncommitted_ignored"] is True


def test_loading_an_uncommitted_checkpoint_raises(results: dict[str, Any]) -> None:
    assert results["uncommitted_load_raised"] is True


def test_corrupt_marker_is_treated_as_uncommitted(results: dict[str, Any]) -> None:
    assert results["corrupt_marker_ignored"] is True


def test_config_change_is_refused_by_default(results: dict[str, Any]) -> None:
    """Resuming across a changed run definition is silent trajectory corruption."""
    assert results["config_guard_raised"] is True


def test_config_change_can_be_overridden_explicitly(results: dict[str, Any]) -> None:
    assert results["override_works"] is True


def test_loader_state_round_trips(results: dict[str, Any]) -> None:
    assert results["loader_state_restored"] is True


def test_same_world_size_preserves_rng_continuity(results: dict[str, Any]) -> None:
    assert results["rng_continuity_same_ws"] is True
    assert results["resharded_same_ws"] is False


def test_prune_keeps_the_newest_and_spares_uncommitted(results: dict[str, Any]) -> None:
    """Pruning must not destroy the forensic record of a failed save."""
    assert results["after_prune"] == [20]
    assert results["pruned"] == ["step-00000010"]
    assert results["uncommitted_survived_prune"] is True
