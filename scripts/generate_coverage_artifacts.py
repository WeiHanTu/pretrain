#!/usr/bin/env python
"""Generate the committed rank-coverage evidence (plan.md A2).

Runs the exactly-once invariant under a real Gloo process group at each requested
world size and writes artifacts/coverage/world-size-<n>.json.

Also writes a negative control, so the evidence set contains proof that the gate
*fails* when the sampler is broken.  A detector only ever shown passing data is
not a detector.

    uv run python scripts/generate_coverage_artifacts.py

SCOPE: multiple processes on one host. Not multi-node evidence (CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch.distributed as dist
import torch.multiprocessing as mp

from pretrainmodel.data.coverage import verify_coverage
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.distributed.topology import capture_topology, software_environment
from pretrainmodel.incidents.inject import inject_sampler_defect

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "coverage"
IDS = [f"s{i:04d}" for i in range(96)]
SEED = 2027
EPOCH = 0


def _worker(rank: int, world_size: int, defect: str, out_path: str, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        sampler = ShardedSampler(
            IDS, seed=SEED, world_size=world_size, rank=rank, dataset_version="gloo-evidence"
        )
        expected = sampler.expected_epoch_ids(EPOCH)
        consumed = inject_sampler_defect(
            sampler.rank_epoch_ids(EPOCH), defect, rank=rank, global_order=expected
        )
        gathered: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(gathered, consumed)
        topology = capture_topology()
        if rank == 0:
            result = verify_coverage(
                expected,
                [g if g is not None else [] for g in gathered],
                epoch=EPOCH,
                sampler={"seed": SEED, "shuffle": True, "drop_last": True},
            )
            payload: dict[str, Any] = result.to_artifact()
            payload["notes"] = (
                f"backend=gloo; {world_size} process(es) on {topology.distinct_hosts} "
                f"distinct host(s); defect={defect}; "
                f"torch={software_environment()['torch']}. "
                "Local multi-process test environment, NOT multi-node evidence."
            )
            Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


def run(world_size: int, defect: str = "none") -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "coverage.json"
        mp.spawn(  # type: ignore[no-untyped-call]
            _worker,
            args=(world_size, defect, str(out), str(Path(tmp) / "rendezvous")),
            nprocs=world_size,
            join=True,
        )
        payload: dict[str, Any] = json.loads(out.read_text())
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for ws in args.world_sizes:
        payload = run(ws)
        path = OUT_DIR / f"world-size-{ws}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        status = "PASS" if payload["passed"] else "FAIL"
        print(f"[{status}] world_size={ws} -> {path.relative_to(REPO_ROOT)}")
        if not payload["passed"]:
            failures.append(f"world_size={ws} clean run did not satisfy the invariant")

    # Negative control: the same machinery, deliberately broken.
    payload = run(4, defect="full_dataset_per_rank")
    path = OUT_DIR / "negative-control-full-dataset-per-rank.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"[{'PASS' if not payload['passed'] else 'FAIL'}] negative control -> "
        f"{path.relative_to(REPO_ROOT)}"
    )
    if payload["passed"]:
        failures.append("negative control passed the invariant; the detector is not working")

    if failures:
        for f in failures:
            print(f"ERROR: {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
