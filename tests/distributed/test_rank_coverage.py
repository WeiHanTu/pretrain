"""Rank coverage under a real multi-process collective (Gloo).

The unit tests prove the sampler partitions a list correctly.  This proves it
still holds when ranks are genuinely separate OS processes that discover each
other through a process group and exchange their consumed IDs over a collective --
the place where rank-dependent seeding, environment leakage and ordering bugs
actually show up.

SCOPE, stated plainly: these are multiple processes on ONE host.  Per CLAUDE.md
that is a test environment, not multi-node evidence.  The artifacts written here
record ``distinct_hosts`` so the distinction is visible in the evidence itself,
and the equivalent run across physical hosts belongs to Phase C.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from pretrainmodel.data.coverage import verify_coverage
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.distributed.topology import capture_topology
from pretrainmodel.incidents.inject import inject_sampler_defect

REPO_ROOT = Path(__file__).resolve().parents[2]
IDS = [f"s{i:04d}" for i in range(96)]
SEED = 2027
EPOCH = 0

pytestmark = pytest.mark.distributed


def _worker(rank: int, world_size: int, defect: str, out_path: str, init_file: str) -> None:
    """One rank: join the group, take its slice, report what it consumed."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        sampler = ShardedSampler(
            IDS, seed=SEED, world_size=world_size, rank=rank, dataset_version="gloo-test"
        )
        expected = sampler.expected_epoch_ids(EPOCH)
        consumed = inject_sampler_defect(
            sampler.rank_epoch_ids(EPOCH), defect, rank=rank, global_order=expected
        )

        gathered: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(gathered, consumed)
        topology = capture_topology()

        if rank == 0:
            per_rank = [g if g is not None else [] for g in gathered]
            result = verify_coverage(
                expected,
                per_rank,
                epoch=EPOCH,
                sampler={"seed": SEED, "shuffle": True, "drop_last": True},
            )
            payload: dict[str, Any] = result.to_artifact()
            payload["notes"] = (
                f"{world_size} Gloo process(es) on {topology.distinct_hosts} distinct host(s). "
                "Local multi-process test environment, not multi-node evidence."
            )
            Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()


def _run_group(world_size: int, defect: str = "none") -> dict[str, Any]:
    """Spawn a Gloo group and return the coverage artifact rank 0 produced."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "coverage.json"
        init_file = Path(tmp) / "rendezvous"
        mp.spawn(  # type: ignore[no-untyped-call]
            _worker,
            args=(world_size, defect, str(out), str(init_file)),
            nprocs=world_size,
            join=True,
        )
        result: dict[str, Any] = json.loads(out.read_text())
        return result


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_coverage_holds_across_real_ranks(world_size: int) -> None:
    """The invariant holds at world size 1, 2 and 4 (plan.md A2)."""
    artifact = _run_group(world_size)
    assert artifact["passed"] is True, artifact
    assert artifact["world_size"] == world_size
    assert artifact["consumed_count"] == artifact["expected_count"] == len(IDS)
    assert artifact["per_rank_counts"] == [len(IDS) // world_size] * world_size
    assert artifact["duplicated_ids"] == []
    assert artifact["missing_ids"] == []
    assert artifact["rank_overlaps"] == []


def test_artifacts_validate_against_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((REPO_ROOT / "schemas" / "coverage.schema.json").read_text())
    jsonschema.validate(_run_group(2), schema)


@pytest.mark.parametrize(
    "defect", ["duplicate_within_rank", "duplicate_across_ranks", "drop_sample"]
)
def test_defects_are_detected_across_real_ranks(defect: str) -> None:
    """Negative controls: the gate must fail when the sampler is broken."""
    artifact = _run_group(4, defect=defect)
    assert artifact["passed"] is False, f"{defect} went undetected across real ranks"


def test_every_rank_derives_the_same_epoch_order() -> None:
    """Rank-dependent seeding is the classic way this silently breaks."""
    artifact = _run_group(4)
    assert artifact["passed"] is True
    assert artifact["unexpected_ids"] == []
