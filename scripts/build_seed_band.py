#!/usr/bin/env python
"""Build the frozen seed-variance band (plan.md A4).

Runs N control seeds of one config, differing ONLY in run.seed, and writes
artifacts/oracles/seed-band.json.

This must be run BEFORE any BF16, asynchronous-checkpoint or changed-world-size
resume experiment. The band is the pre-registered definition of "equivalent"; a
band built afterwards would be fitted to the result it is supposed to judge.

    uv run python scripts/build_seed_band.py --config configs/seed_band.toml
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from pretrainmodel.config import config_hash, load_config
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.data.manifest import ShardManifest, manifest_hash
from pretrainmodel.data.shard import build_index
from pretrainmodel.model.transformer import build_model
from pretrainmodel.observability.manifest import RunManifest
from pretrainmodel.training.loop import seed_everything, train
from pretrainmodel.training.oracle import build_seed_band

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "artifacts" / "oracles" / "seed-band.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seed_band.toml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()

    if len(args.seeds) < 3:
        print("error: at least 3 control seeds are required", flush=True)
        return 2

    base = load_config(REPO_ROOT / args.config)
    root = REPO_ROOT / base.data.root
    manifest = ShardManifest.read(root / "shards.json")
    manifest.verify(root)
    index = build_index(manifest)

    losses: dict[int, list[float]] = {}
    run_ids: list[str] = []

    for seed in args.seeds:
        cfg = replace(base, run=replace(base.run, seed=seed))
        # Each control run gets its own manifest: the band must be traceable to the
        # exact runs that produced it, not merely asserted.
        run = RunManifest(cfg, repo_root=REPO_ROOT, data_manifest_hash=manifest_hash(manifest))
        seed_everything(seed, deterministic=cfg.determinism.deterministic_algorithms)
        dataset = WindowDataset(root, manifest, index)
        sampler = ShardedSampler(
            index.sample_ids,
            seed=seed,
            world_size=1,
            rank=0,
            shuffle=cfg.data.shuffle,
            drop_last=cfg.data.drop_last,
            dataset_version=manifest.dataset_version,
        )
        result = train(cfg, build_model(cfg), dataset, sampler, run_id=run.run_id)
        losses[seed] = result.losses
        run_ids.append(run.run_id)
        run.metrics = result.metrics
        run.write(REPO_ROOT / "runs" / run.run_id / "run_manifest.json")
        print(f"seed {seed}: {len(result.losses)} steps, final={result.final_loss:.6f}")

    # The band belongs to the *shared* configuration, so it is hashed with the seed
    # field neutralised -- otherwise each control run would have a different hash and
    # no candidate could ever match.
    shared = replace(base, run=replace(base.run, seed=0))
    band = build_seed_band(
        losses,
        config_hash=config_hash(shared),
        notes=(
            f"Control runs: {', '.join(run_ids)}. "
            "CPU, world size 1, fp32. Frozen before any resume experiment."
        ),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(band.to_dict(), indent=2, sort_keys=True) + "\n")

    lo, hi = band.trailing_bounds()
    print(f"\nband over {band.num_steps} steps from seeds {band.seeds}")
    print(f"  margin_factor      {band.margin_factor}")
    print(
        f"  pass rule          >={band.min_inside_fraction:.0%} steps inside + trailing mean inside"
    )
    print(f"  trailing band      [{lo:.6f}, {hi:.6f}] from step {band.trailing_start}")
    print(f"  wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
