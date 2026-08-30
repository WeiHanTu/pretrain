"""Command-line entry point.

Subcommands:
  make-fixture   generate a synthetic dataset for local development
  verify-data    check shard digests against the manifest (incident I-002 gate)
  train          run the training loop and emit a run manifest
  evaluate       masked forecast metrics for a checkpoint-free model
  overfit        the tiny-batch overfit gate (plan.md A3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from pretrainmodel.config import Config, ConfigError, load_config
from pretrainmodel.data.coverage import verify_coverage
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.data.manifest import ManifestError, ShardManifest, manifest_hash
from pretrainmodel.data.shard import build_index, write_synthetic_dataset
from pretrainmodel.distributed.topology import capture_topology
from pretrainmodel.model.transformer import build_model, count_parameters
from pretrainmodel.observability.cost import estimate_cost
from pretrainmodel.observability.events import EventLog
from pretrainmodel.observability.manifest import RunManifest, new_run_id
from pretrainmodel.training.checkpointing import PeriodicCheckpointer, resume_if_available
from pretrainmodel.training.loop import build_optimizer, evaluate, seed_everything, train

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prepare(cfg: Config) -> tuple[WindowDataset, ShardedSampler, ShardManifest]:
    root = Path(cfg.data.root)
    if not root.is_absolute():
        root = REPO_ROOT / root
    manifest = ShardManifest.read(root / "shards.json")
    manifest.verify(root)  # incident I-002 gate: before any optimizer step
    index = build_index(manifest)
    dataset = WindowDataset(root, manifest, index)
    sampler = ShardedSampler(
        index.sample_ids,
        seed=cfg.run.seed,
        world_size=1,
        rank=0,
        shuffle=cfg.data.shuffle,
        drop_last=cfg.data.drop_last,
        dataset_version=manifest.dataset_version,
    )
    return dataset, sampler, manifest


def cmd_make_fixture(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = REPO_ROOT / cfg.data.root
    manifest = write_synthetic_dataset(
        root,
        dataset_version=args.version,
        num_shards=args.shards,
        timesteps_per_shard=args.timesteps,
        num_sensors=cfg.data.num_sensors,
        num_features=cfg.data.num_features,
        context_steps=cfg.data.context_steps,
        horizon_steps=cfg.data.horizon_steps,
        seed=cfg.run.seed,
    )
    index = build_index(manifest)
    print(f"wrote {len(manifest.shards)} shards to {root}")
    print(f"sample space: {len(index)} windows; manifest sha {manifest_hash(manifest)[:12]}")
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    root = REPO_ROOT / cfg.data.root
    manifest = ShardManifest.read(root / "shards.json")
    try:
        manifest.verify(root)
    except ManifestError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    index = build_index(manifest)
    print(f"OK: {len(manifest.shards)} shards verified, {len(index)} samples")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed_everything(cfg.run.seed, deterministic=cfg.determinism.deterministic_algorithms)
    dataset, sampler, manifest = _prepare(cfg)

    model = build_model(cfg)
    run_id = new_run_id(cfg.run.name)
    run = RunManifest(
        cfg, repo_root=REPO_ROOT, run_id=run_id, data_manifest_hash=manifest_hash(manifest)
    )
    run.model = {
        "parameter_count": count_parameters(model),
        "trainable_parameter_count": count_parameters(model, trainable_only=True),
        "architecture": "factorised-spatiotemporal-transformer",
    }

    out_dir = REPO_ROOT / "runs" / run_id
    optimizer = build_optimizer(cfg, model)
    ckpt_root = REPO_ROOT / cfg.checkpoint.dir
    data_hash = manifest_hash(manifest)

    with EventLog(out_dir / "events-rank0.jsonl", run_id=run_id) as events:
        events.emit("run_started", config=cfg.run.name, params=run.model["parameter_count"])

        start_step, resume_from = 0, None
        if args.resume:
            loaded = resume_if_available(
                cfg,
                model=model,
                optimizer=optimizer,
                root=ckpt_root,
                data_manifest_hash=data_hash,
                events=events,
            )
            if loaded is not None:
                start_step, resume_from = loaded.step, loaded.loader_state
                print(f"resumed from step {loaded.step} ({loaded.note})")

        checkpointer = PeriodicCheckpointer(
            cfg,
            model=model,
            optimizer=optimizer,
            root=ckpt_root,
            data_manifest_hash=data_hash,
            run_id=run_id,
            events=events,
        )
        result = train(
            cfg,
            model,
            dataset,
            sampler,
            run_id=run_id,
            events=events,
            optimizer=optimizer,
            start_step=start_step,
            resume_from=resume_from,
            on_step=checkpointer,
        )
        events.emit("run_finished", **result.metrics, checkpoints=checkpointer.saved_steps)

    coverage = verify_coverage(
        sampler.rank_epoch_ids(0)[: len(result.consumed_ids)], [result.consumed_ids]
    )
    run.topology = capture_topology(global_batch_size=cfg.global_batch_size(1))
    run.metrics = {**result.metrics, "coverage_passed": coverage.passed}
    run.cost = estimate_cost(
        sku=cfg.cost.sku,
        rate_usd_per_gpu_hour=cfg.cost.rate_usd_per_gpu_hour,
        duration_seconds=sum(s.seconds for s in result.steps),
        world_size=1,
        gpus_per_rank=1 if torch.cuda.is_available() else 0,
    )
    payload = run.write(out_dir / "run_manifest.json")
    print(
        f"run {run_id}: {result.metrics['steps_completed']} steps, "
        f"final_loss={result.final_loss:.6f}, stopped={result.stopped_because}"
    )
    print(f"manifest: {out_dir / 'run_manifest.json'}")
    print(f"checkpoints: {checkpointer.saved_steps or 'none (checkpoint.every_steps = 0)'}")
    return 0 if payload["status"] == "completed" else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    seed_everything(cfg.run.seed)
    dataset, sampler, _ = _prepare(cfg)
    model = build_model(cfg)
    metrics = evaluate(cfg, model, dataset, sampler.rank_epoch_ids(0)[: args.limit])
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


def cmd_overfit(args: argparse.Namespace) -> int:
    """Tiny-batch overfit gate: a fixed batch must be memorisable.

    If a model cannot drive the loss down on one repeated batch, nothing about a
    longer run is interpretable -- the bug is in the model, the masking or the
    optimizer, not in the data or the schedule.
    """
    from pretrainmodel.training.overfit import run_overfit_gate

    cfg = load_config(args.config)
    result = run_overfit_gate(cfg, repo_root=REPO_ROOT, steps=args.steps, threshold=args.threshold)
    out = REPO_ROOT / "artifacts" / "gates" / "tiny-overfit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    status = "PASS" if result["passed"] else "FAIL"
    print(
        f"[{status}] initial={result['initial_loss']:.6f} "
        f"final={result['final_loss']:.6f} threshold={result['threshold']}"
    )
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0 if result["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pretrainmodel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("make-fixture", help="generate a synthetic dataset")
    p.add_argument("--config", default="configs/local_smoke.toml")
    p.add_argument("--version", default="synthetic-v1")
    p.add_argument("--shards", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=64)
    p.set_defaults(func=cmd_make_fixture)

    p = sub.add_parser("verify-data", help="verify shard digests")
    p.add_argument("--config", default="configs/local_smoke.toml")
    p.set_defaults(func=cmd_verify_data)

    p = sub.add_parser("train", help="run training")
    p.add_argument("--config", default="configs/local_smoke.toml")
    p.add_argument(
        "--resume",
        action="store_true",
        help="resume from the newest committed checkpoint under checkpoint.dir",
    )
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="report masked forecast metrics")
    p.add_argument("--config", default="configs/local_smoke.toml")
    p.add_argument("--limit", type=int, default=64)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("overfit", help="tiny-batch overfit gate")
    p.add_argument("--config", default="configs/local_smoke.toml")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--threshold", type=float, default=0.05)
    p.set_defaults(func=cmd_overfit)

    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
        return result
    except (ConfigError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
