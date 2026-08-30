"""torchrun entrypoint for a distributed training run.

    torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$RANK \
             --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29500 \
             -m pretrainmodel.distributed.entrypoint --config configs/two_node_l4.toml

This is the path that runs on real hardware, so it is the path that gets rehearsed
locally.  The only difference between the rehearsal and the paid run is where the
hosts are; the launcher, rendezvous, sharding, checkpointing and evidence capture
are identical code.

Order of operations matters here.  Topology is asserted **before** the first
optimizer step, so a run that claims two hosts and got one dies in seconds rather
than after burning a reservation and producing an artifact that looks like
multi-node evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from pretrainmodel.config import load_config
from pretrainmodel.data.coverage import verify_coverage
from pretrainmodel.data.dataset import WindowDataset
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.data.manifest import ManifestError, ShardManifest, manifest_hash
from pretrainmodel.data.shard import build_index
from pretrainmodel.distributed.launch import init_distributed, shutdown_distributed
from pretrainmodel.distributed.parallel import build_device_mesh, shard_model, sharding_report
from pretrainmodel.distributed.topology import (
    TopologyError,
    assert_topology_matches,
    capture_topology,
)
from pretrainmodel.model.transformer import build_model, count_parameters
from pretrainmodel.observability.cost import estimate_cost
from pretrainmodel.observability.events import EventLog
from pretrainmodel.observability.manifest import RunManifest, new_run_id
from pretrainmodel.training.checkpointing import PeriodicCheckpointer, resume_if_available
from pretrainmodel.training.loop import build_optimizer, seed_everything, train

REPO_ROOT = Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume from the newest committed checkpoint. On preemptible instances "
            "this is how a reclaimed node rejoins without losing the run."
        ),
    )
    parser.add_argument(
        "--rehearsal",
        action="store_true",
        help=(
            "Mark this run as a launch-path rehearsal. Its artifacts are written "
            "under a rehearsal/ prefix and are never multi-node evidence."
        ),
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    info = init_distributed(
        backend=cfg.distributed.backend,
        expect_distinct_hosts=cfg.distributed.expect_distinct_hosts and not args.rehearsal,
    )
    started = time.monotonic()

    try:
        topology = capture_topology(
            global_batch_size=cfg.global_batch_size(info.world_size),
            grad_accum_steps=cfg.train.grad_accum_steps,
        )
        # Fail before spending anything if the claim and the reality disagree.
        if not args.rehearsal:
            assert_topology_matches(
                topology,
                expect_distinct_hosts=cfg.distributed.expect_distinct_hosts,
                expect_world_size=cfg.distributed.expect_world_size,
            )

        root = Path(cfg.data.root)
        if not root.is_absolute():
            root = REPO_ROOT / root
        manifest = ShardManifest.read(root / "shards.json")
        manifest.verify(root)
        index = build_index(manifest)

        dataset = WindowDataset(root, manifest, index)
        sampler = ShardedSampler(
            index.sample_ids,
            seed=cfg.run.seed,
            world_size=info.world_size,
            rank=info.rank,
            shuffle=cfg.data.shuffle,
            drop_last=cfg.data.drop_last,
            dataset_version=manifest.dataset_version,
        )

        seed_everything(cfg.run.seed, deterministic=cfg.determinism.deterministic_algorithms)
        device = torch.device("cuda", info.local_rank) if torch.cuda.is_available() else None
        model = shard_model(build_model(cfg), build_device_mesh())
        optimizer = build_optimizer(cfg, model)

        run_id = new_run_id(cfg.run.name)
        ckpt_root = REPO_ROOT / cfg.checkpoint.dir
        data_hash = manifest_hash(manifest)
        prefix = "rehearsal" if args.rehearsal else "runs"
        out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / prefix / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        with EventLog(
            out_dir / f"events-rank{info.rank}.jsonl", run_id=run_id, rank=info.rank
        ) as events:
            events.emit(
                "rendezvous",
                **info.to_dict(),
                distinct_hosts=topology.distinct_hosts,
                is_multi_node=topology.is_multi_node,
            )
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
                device=device,
                optimizer=optimizer,
                start_step=start_step,
                resume_from=resume_from,
                on_step=checkpointer,
            )
            events.emit("run_finished", **result.metrics, checkpoints=checkpointer.saved_steps)

        coverage = verify_coverage(
            sampler.rank_epoch_ids(0)[: len(result.consumed_ids)], [result.consumed_ids]
        )

        if info.rank == 0:
            run = RunManifest(
                cfg,
                repo_root=REPO_ROOT,
                run_id=run_id,
                data_manifest_hash=manifest_hash(manifest),
            )
            run.topology = topology
            run.model = {
                "parameter_count": count_parameters(model),
                "architecture": "factorised-spatiotemporal-transformer",
            }
            run.metrics = {
                **result.metrics,
                "rank_coverage_passed": coverage.passed,
                "sharding": sharding_report(model),
            }
            run.hardware = {
                "rendezvous": info.to_dict(),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
            run.cost = estimate_cost(
                sku=cfg.cost.sku,
                rate_usd_per_gpu_hour=cfg.cost.rate_usd_per_gpu_hour,
                duration_seconds=time.monotonic() - started,
                world_size=info.world_size,
                gpus_per_rank=1 if torch.cuda.is_available() else 0,
            )
            run.write(out_dir / "run_manifest.json")
            (out_dir / "topology.json").write_text(
                json.dumps(
                    {
                        **topology.to_dict(),
                        "rendezvous": info.to_dict(),
                        "is_multi_node": topology.is_multi_node,
                        "rehearsal": args.rehearsal,
                        "note": (
                            "Launch-path rehearsal on a single host. NOT multi-node evidence."
                            if args.rehearsal or not topology.is_multi_node
                            else "Ranks observed on multiple distinct hosts."
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            print(
                f"rank0: {result.metrics['steps_completed']} steps, "
                f"final_loss={result.final_loss:.6f}, "
                f"world_size={info.world_size}, distinct_hosts={topology.distinct_hosts}, "
                f"multi_node={topology.is_multi_node}"
            )
            print(f"artifacts: {out_dir}")
        return 0
    except (TopologyError, ManifestError, RuntimeError) as exc:
        print(f"rank {info.rank}: {exc}", file=sys.stderr)
        return 1
    finally:
        shutdown_distributed()


if __name__ == "__main__":
    raise SystemExit(main())
