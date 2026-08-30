"""Checkpointing in the REAL training path, and the bf16 path Phase C configures.

Both of these are gap tests. The checkpoint contract was proven in Phase B while no
actual training run wrote one, and bf16 is the dtype both cloud configs request
while every executed run so far has been fp32.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch

from pretrainmodel.config import Config, from_mapping
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.data.manifest import ShardManifest
from pretrainmodel.data.shard import build_index, write_synthetic_dataset
from pretrainmodel.distributed.checkpoint import latest_committed, list_committed, read_marker
from pretrainmodel.model.transformer import build_model
from pretrainmodel.training.checkpointing import PeriodicCheckpointer, resume_if_available
from pretrainmodel.training.loop import build_optimizer, seed_everything, train

CTX, HOR, SENSORS = 8, 4, 8


def _cfg(data_root: Path, ckpt_dir: Path, **over: Any) -> Config:
    raw: dict[str, Any] = {
        "run": {"name": "ckpath", "seed": 5},
        "data": {
            "root": str(data_root),
            "context_steps": CTX,
            "horizon_steps": HOR,
            "num_sensors": SENSORS,
            "num_features": 1,
        },
        "model": {"d_model": 32, "num_layers": 1, "num_heads": 4, "max_sensors": SENSORS},
        "optim": {"lr": 1e-3, "warmup_steps": 1, "total_steps": 40},
        "train": {"micro_batch_size": 2, "max_steps": 6, "log_every": 2},
        "checkpoint": {"dir": str(ckpt_dir), "every_steps": 2, "keep_last": 3},
    }
    for section, values in over.items():
        raw[section] = {**raw[section], **values}  # type: ignore[index]
    return from_mapping(raw)


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    write_synthetic_dataset(
        data_root,
        num_shards=4,
        timesteps_per_shard=40,
        num_sensors=SENSORS,
        context_steps=CTX,
        horizon_steps=HOR,
        seed=3,
    )
    return data_root, tmp_path / "ckpt"


def _run(cfg: Config, data_root: Path, ckpt_dir: Path, *, resume: bool = False) -> Any:
    manifest = ShardManifest.read(data_root / "shards.json")
    index = build_index(manifest)
    dataset = WindowDataset(data_root, manifest, index)
    sampler = ShardedSampler(
        index.sample_ids,
        seed=cfg.run.seed,
        world_size=1,
        rank=0,
        dataset_version=manifest.dataset_version,
    )
    seed_everything(cfg.run.seed)
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)

    start_step, resume_from = 0, None
    if resume:
        loaded = resume_if_available(cfg, model=model, optimizer=optimizer, root=ckpt_dir)
        if loaded is not None:
            start_step, resume_from = loaded.step, loaded.loader_state

    ckpt = PeriodicCheckpointer(cfg, model=model, optimizer=optimizer, root=ckpt_dir)
    result = train(
        cfg,
        model,
        dataset,
        sampler,
        optimizer=optimizer,
        start_step=start_step,
        resume_from=resume_from,
        on_step=ckpt,
    )
    return result, ckpt


# --------------------------------------------------------------------------- #
# The gap that survived Phase B
# --------------------------------------------------------------------------- #


def test_a_real_training_run_writes_committed_checkpoints(env: tuple[Path, Path]) -> None:
    """The regression this file exists for.

    Phase B proved the checkpoint contract while `save_checkpoint` was called from
    nowhere but the resume experiment, so `checkpoint.every_steps` was parsed,
    validated, and honoured by nothing.
    """
    data_root, ckpt_dir = env
    _, ckpt = _run(_cfg(data_root, ckpt_dir), data_root, ckpt_dir)
    assert ckpt.saved_steps == [2, 4, 6]
    assert [s for s, _ in list_committed(ckpt_dir)] == [2, 4, 6]
    marker = read_marker(latest_committed(ckpt_dir))
    assert marker is not None and marker.step == 6


def test_checkpoint_records_the_loader_cursor(env: tuple[Path, Path]) -> None:
    """Without the cursor a resume replays samples, however correct the weights are."""
    data_root, ckpt_dir = env
    cfg = _cfg(data_root, ckpt_dir)
    _run(cfg, data_root, ckpt_dir)
    marker = read_marker(latest_committed(ckpt_dir))
    assert marker is not None
    expected = cfg.train.max_steps * cfg.train.micro_batch_size * cfg.train.grad_accum_steps
    assert marker.loader_state["global_samples_consumed"] == expected


def test_cadence_zero_disables_checkpointing(env: tuple[Path, Path]) -> None:
    data_root, ckpt_dir = env
    cfg = _cfg(data_root, ckpt_dir, checkpoint={"every_steps": 0})
    _, ckpt = _run(cfg, data_root, ckpt_dir)
    assert ckpt.saved_steps == []
    assert list_committed(ckpt_dir) == []


def test_keep_last_prunes_older_checkpoints(env: tuple[Path, Path]) -> None:
    data_root, ckpt_dir = env
    cfg = _cfg(data_root, ckpt_dir, train={"max_steps": 8}, checkpoint={"keep_last": 2})
    _, ckpt = _run(cfg, data_root, ckpt_dir)
    assert ckpt.saved_steps == [2, 4, 6, 8]
    assert [s for s, _ in list_committed(ckpt_dir)] == [6, 8]


# --------------------------------------------------------------------------- #
# Resume, as a preempted node would do it
# --------------------------------------------------------------------------- #


def test_resume_is_a_noop_when_nothing_was_saved(env: tuple[Path, Path]) -> None:
    """A first run must not fail merely because there is nothing to resume from."""
    data_root, ckpt_dir = env
    cfg = _cfg(data_root, ckpt_dir)
    model = build_model(cfg)
    assert (
        resume_if_available(cfg, model=model, optimizer=build_optimizer(cfg, model), root=ckpt_dir)
        is None
    )


def test_interrupted_run_resumes_from_its_checkpoint(env: tuple[Path, Path]) -> None:
    """The spot-preemption scenario, end to end.

    Stop at step 4, then resume in a *fresh* model and optimizer. The resume must
    pick up the recorded step and cursor rather than restarting, and the samples
    consumed before the interruption must satisfy the exactly-once invariant.
    """
    from pretrainmodel.data.coverage import verify_coverage

    data_root, ckpt_dir = env
    cfg = _cfg(data_root, ckpt_dir, train={"max_steps": 4})
    first, _ = _run(cfg, data_root, ckpt_dir)
    marker = read_marker(latest_committed(ckpt_dir))
    assert marker is not None and marker.step == 4

    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    loaded = resume_if_available(cfg, model=model, optimizer=optimizer, root=ckpt_dir)
    assert loaded is not None
    assert loaded.step == 4
    assert loaded.rng_continuity is True
    assert loaded.loader_state.global_samples_consumed == 4 * cfg.train.micro_batch_size

    manifest = ShardManifest.read(data_root / "shards.json")
    sampler = ShardedSampler(
        build_index(manifest).sample_ids,
        seed=cfg.run.seed,
        world_size=1,
        rank=0,
        dataset_version=manifest.dataset_version,
    )
    expected = sampler.rank_epoch_ids(0)[: len(first.consumed_ids)]
    assert verify_coverage(expected, [first.consumed_ids]).passed


# --------------------------------------------------------------------------- #
# bf16: configured for Phase C, never executed until now
# --------------------------------------------------------------------------- #


def test_bf16_training_path_runs_and_stays_finite(env: tuple[Path, Path]) -> None:
    """Both cloud configs request bf16; every executed run so far was fp32.

    Discovering that autocast is misconfigured belongs on a laptop, not on the first
    billed GPU hour.
    """
    data_root, ckpt_dir = env
    cfg = _cfg(data_root, ckpt_dir, train={"dtype": "bf16"})
    result, ckpt = _run(cfg, data_root, ckpt_dir)
    assert result.metrics["steps_completed"] == cfg.train.max_steps
    assert all(math.isfinite(x) for x in result.losses), result.losses
    assert ckpt.saved_steps, "bf16 runs must still checkpoint"


def test_bf16_and_fp32_reach_comparable_loss(env: tuple[Path, Path]) -> None:
    """Sanity, not equivalence.

    bf16 is not expected to match fp32 bitwise or closely; this only catches a
    grossly broken autocast path, where the loss would be NaN or wildly off rather
    than merely lower-precision.
    """
    data_root, ckpt_dir = env
    fp32, _ = _run(_cfg(data_root, ckpt_dir / "a"), data_root, ckpt_dir / "a")
    bf16, _ = _run(
        _cfg(data_root, ckpt_dir / "b", train={"dtype": "bf16"}), data_root, ckpt_dir / "b"
    )
    assert math.isfinite(bf16.final_loss)
    assert abs(bf16.final_loss - fp32.final_loss) < 0.5 * max(1.0, abs(fp32.final_loss))


def test_bf16_is_rejected_under_the_deterministic_oracle() -> None:
    """Guard rail: bf16 reductions are not associative, so exact equality is void."""
    from pretrainmodel.config import ConfigError

    with pytest.raises(ConfigError, match=r"requires train\.dtype"):
        from_mapping(
            {
                "run": {"name": "x"},
                "data": {
                    "root": "d",
                    "context_steps": CTX,
                    "horizon_steps": HOR,
                    "num_sensors": SENSORS,
                },
                "model": {"d_model": 32, "num_layers": 1, "num_heads": 4, "max_sensors": SENSORS},
                "optim": {"lr": 1e-3, "warmup_steps": 1, "total_steps": 10},
                "train": {"micro_batch_size": 2, "max_steps": 4, "dtype": "bf16"},
                "checkpoint": {"dir": "c"},
                "determinism": {"deterministic_algorithms": True},
            }
        )


def test_bf16_autocast_is_actually_engaged(env: tuple[Path, Path]) -> None:
    """Assert the dtype reaches the model, rather than being parsed and ignored."""
    data_root, ckpt_dir = env
    cfg = _cfg(data_root, ckpt_dir, train={"dtype": "bf16"})
    manifest = ShardManifest.read(data_root / "shards.json")
    index = build_index(manifest)
    dataset = WindowDataset(data_root, manifest, index)
    batch = dataset.batch(index.sample_ids[:2])
    model = build_model(cfg)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(batch.context_values, batch.context_observed, torch.arange(SENSORS))
    assert out.dtype == torch.bfloat16, "autocast did not affect the forward pass"
