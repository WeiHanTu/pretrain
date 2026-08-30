"""End-to-end training on a synthetic fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from pretrainmodel.config import Config, from_mapping
from pretrainmodel.data.coverage import verify_coverage
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.data.manifest import manifest_hash
from pretrainmodel.data.shard import build_index, write_synthetic_dataset
from pretrainmodel.model.transformer import build_model
from pretrainmodel.observability.manifest import FormalRunError, RunManifest
from pretrainmodel.training.loop import learning_rate_at, seed_everything, train
from pretrainmodel.training.overfit import run_overfit_gate

REPO_ROOT = Path(__file__).resolve().parents[2]
CTX, HOR, SENSORS = 8, 4, 8


def _cfg(root: Path, **train_overrides: Any) -> Config:
    train_cfg: dict[str, Any] = {"micro_batch_size": 4, "max_steps": 12, "log_every": 4}
    total_steps = train_overrides.pop("total_steps", 40)
    train_cfg.update(train_overrides)
    return from_mapping(
        {
            "run": {"name": "itest", "seed": 3},
            "data": {
                "root": str(root),
                "context_steps": CTX,
                "horizon_steps": HOR,
                "num_sensors": SENSORS,
                "num_features": 1,
            },
            "model": {"d_model": 32, "num_layers": 2, "num_heads": 4, "max_sensors": SENSORS},
            "optim": {"lr": 2e-3, "warmup_steps": 2, "total_steps": total_steps},
            "train": train_cfg,
            "checkpoint": {"dir": "checkpoints/itest"},
        }
    )


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, Config]:
    write_synthetic_dataset(
        tmp_path,
        num_shards=4,
        timesteps_per_shard=40,
        num_sensors=SENSORS,
        context_steps=CTX,
        horizon_steps=HOR,
        seed=5,
    )
    return tmp_path, _cfg(tmp_path)


def _build(root: Path, cfg: Config) -> tuple[WindowDataset, ShardedSampler]:
    from pretrainmodel.data.manifest import ShardManifest

    manifest = ShardManifest.read(root / "shards.json")
    index = build_index(manifest)
    return (
        WindowDataset(root, manifest, index),
        ShardedSampler(
            index.sample_ids,
            seed=cfg.run.seed,
            world_size=1,
            rank=0,
            dataset_version=manifest.dataset_version,
        ),
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def test_training_runs_and_loss_is_finite(fixture: tuple[Path, Config]) -> None:
    root, cfg = fixture
    dataset, sampler = _build(root, cfg)
    seed_everything(cfg.run.seed)
    result = train(cfg, build_model(cfg), dataset, sampler)
    assert len(result.steps) == cfg.train.max_steps
    assert all(math.isfinite(loss) for loss in result.losses)
    assert result.stopped_because == "max_steps"


def test_training_consumes_each_sample_at_most_once(fixture: tuple[Path, Config]) -> None:
    """The coverage invariant must hold over what training actually pulled."""
    root, cfg = fixture
    dataset, sampler = _build(root, cfg)
    seed_everything(cfg.run.seed)
    result = train(cfg, build_model(cfg), dataset, sampler)
    expected = sampler.rank_epoch_ids(0)[: len(result.consumed_ids)]
    assert verify_coverage(expected, [result.consumed_ids]).passed


def test_same_seed_gives_an_identical_loss_sequence(fixture: tuple[Path, Config]) -> None:
    """Baseline reproducibility.

    Not the bit-exactness oracle from spec 8.1 -- that gate covers a save/resume
    boundary and belongs to Phase B. This only establishes that two fresh runs of
    the same config agree, without which no resume comparison would mean anything.
    """
    root, cfg = fixture
    dataset, sampler = _build(root, cfg)

    seed_everything(cfg.run.seed, deterministic=True)
    first = train(cfg, build_model(cfg), dataset, sampler).losses
    seed_everything(cfg.run.seed, deterministic=True)
    second = train(cfg, build_model(cfg), dataset, sampler).losses
    torch.use_deterministic_algorithms(False)

    assert first == second


def test_different_seeds_diverge(fixture: tuple[Path, Config]) -> None:
    """If seeds did not matter, the seed-variance band would be meaningless."""
    root, cfg = fixture
    dataset, sampler = _build(root, cfg)
    seed_everything(1)
    a = train(cfg, build_model(cfg), dataset, sampler).losses
    seed_everything(2)
    b = train(cfg, build_model(cfg), dataset, sampler).losses
    assert a != b


def test_wall_clock_cap_stops_the_run(fixture: tuple[Path, Config]) -> None:
    """Every run carries a hard time cap, exercised locally so it works in cloud."""
    root, _ = fixture
    cfg = _cfg(root, max_steps=100000, max_wall_seconds=1, total_steps=100000)
    dataset, sampler = _build(root, cfg)
    seed_everything(cfg.run.seed)
    result = train(cfg, build_model(cfg), dataset, sampler)
    assert result.stopped_because in {"max_wall_seconds", "epoch_exhausted"}
    assert len(result.steps) < cfg.train.max_steps


def test_gradient_accumulation_changes_the_effective_batch(
    fixture: tuple[Path, Config],
) -> None:
    root, _ = fixture
    cfg = _cfg(root, micro_batch_size=2, grad_accum_steps=3, max_steps=4)
    dataset, sampler = _build(root, cfg)
    seed_everything(cfg.run.seed)
    result = train(cfg, build_model(cfg), dataset, sampler)
    assert cfg.global_batch_size(1) == 6
    assert len(result.steps[0].sample_ids) == 6


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #


def test_learning_rate_warms_up_then_decays(fixture: tuple[Path, Config]) -> None:
    _, cfg = fixture
    warm = [learning_rate_at(s, cfg) for s in range(cfg.optim.warmup_steps)]
    assert warm == sorted(warm)
    assert learning_rate_at(cfg.optim.warmup_steps - 1, cfg) == pytest.approx(cfg.optim.lr)
    mid = learning_rate_at(cfg.optim.total_steps // 2, cfg)
    end = learning_rate_at(cfg.optim.total_steps, cfg)
    assert cfg.optim.lr > mid > end
    assert end == pytest.approx(cfg.optim.lr * cfg.optim.min_lr_ratio)


def test_learning_rate_is_a_pure_function_of_step(fixture: tuple[Path, Config]) -> None:
    """A resumed run must reconstruct the schedule, never restart warmup."""
    _, cfg = fixture
    assert learning_rate_at(7, cfg) == learning_rate_at(7, cfg)


# --------------------------------------------------------------------------- #
# Run manifest
# --------------------------------------------------------------------------- #


def test_formal_run_refuses_a_dirty_tree(tmp_path: Path) -> None:
    """ "Commit abc123" must describe the code that actually ran."""
    cfg = from_mapping(
        {
            "run": {"name": "formal", "seed": 1, "formal": True},
            "data": {
                "root": str(tmp_path),
                "context_steps": CTX,
                "horizon_steps": HOR,
                "num_sensors": SENSORS,
            },
            "model": {"d_model": 32, "num_layers": 1, "num_heads": 4, "max_sensors": SENSORS},
            "optim": {"lr": 1e-3, "warmup_steps": 1, "total_steps": 10},
            "train": {"micro_batch_size": 2, "max_steps": 5},
            "checkpoint": {"dir": "checkpoints/formal"},
        }
    )
    (tmp_path / "untracked.txt").write_text("dirty")
    with pytest.raises(FormalRunError, match="dirty"):
        RunManifest(cfg, repo_root=tmp_path)


def test_manifest_validates_against_schema(fixture: tuple[Path, Config], tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    root, cfg = fixture
    from pretrainmodel.data.manifest import ShardManifest

    run = RunManifest(
        cfg,
        repo_root=REPO_ROOT,
        data_manifest_hash=manifest_hash(ShardManifest.read(root / "shards.json")),
    )
    run.metrics = {"final_loss": 0.5}
    out = tmp_path / "run_manifest.json"
    run.write(out)
    schema = json.loads((REPO_ROOT / "schemas" / "run_manifest.schema.json").read_text())
    jsonschema.validate(json.loads(out.read_text()), schema)


def test_manifest_records_artifact_digests(fixture: tuple[Path, Config], tmp_path: Path) -> None:
    _, cfg = fixture
    run = RunManifest(cfg, repo_root=tmp_path)
    artifact = tmp_path / "evidence.json"
    artifact.write_text("{}")
    run.record_artifact(artifact)
    assert len(run.artifacts[0]["sha256"]) == 64


# --------------------------------------------------------------------------- #
# Overfit gate
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_tiny_batch_overfit_gate_passes(fixture: tuple[Path, Config]) -> None:
    root, cfg = fixture
    result = run_overfit_gate(cfg, repo_root=root, steps=250, threshold=0.1)
    assert result["passed"], result
    assert result["final_loss"] < result["target_loss"]
