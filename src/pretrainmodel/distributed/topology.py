"""Observed process-group topology, and the checks that keep claims honest.

The central idea: a multi-node claim is a *measurement*, not a configuration
setting.  ``expect_distinct_hosts`` states what a run believes it is doing;
``capture_topology`` records what actually happened; ``assert_topology_matches``
aborts when they disagree.

Without that check, four ranks on one laptop and four ranks on two machines emit
identical-looking logs, and the difference -- the only thing that makes the run
evidence of anything -- is invisible in the artifact.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "Topology",
    "TopologyError",
    "assert_topology_matches",
    "capture_topology",
    "software_environment",
]


class TopologyError(RuntimeError):
    """The observed topology contradicts what the run configuration claimed."""


def _host_fingerprint(hostname: str) -> str:
    """Short stable digest of a hostname.

    Lets an artifact prove that N ranks sat on M distinct machines without
    publishing internal hostnames.
    """
    return hashlib.sha256(hostname.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class Topology:
    """What the process group actually looks like, as observed at runtime."""

    backend: str
    rank: int
    local_rank: int
    world_size: int
    hosts: list[str]
    distinct_hosts: int
    node_count: int
    gpus_per_node: int
    rendezvous: str | None = None
    global_batch_size: int = 0
    grad_accum_steps: int = 1
    device: str = "cpu"

    @property
    def is_multi_node(self) -> bool:
        """True only when ranks were observed on more than one physical host.

        Multiple processes, containers or GPUs on one machine are a test
        environment, never multi-node evidence (CLAUDE.md).
        """
        return self.distinct_hosts >= 2

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("rank", None)
        d.pop("local_rank", None)
        return d


def software_environment() -> dict[str, str | None]:
    """Version fingerprint recorded in every run manifest."""
    cuda_version: str | None = torch.version.cuda
    nccl_version: str | None = None
    if torch.cuda.is_available() and dist.is_nccl_available():
        try:
            nccl_version = ".".join(str(v) for v in torch.cuda.nccl.version())  # type: ignore[no-untyped-call]
        except Exception:  # pragma: no cover - platform dependent
            nccl_version = None
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": cuda_version,
        "nccl": nccl_version,
        "platform": f"{platform.system()}-{platform.machine()}",
    }


def capture_topology(
    *,
    global_batch_size: int = 0,
    grad_accum_steps: int = 1,
    hash_hostnames: bool = True,
) -> Topology:
    """Observe the current process group.

    Hostnames are gathered from every rank via the collective itself, so the
    distinct-host count reflects where the ranks really ran rather than what any
    single process was told.
    """
    hostname = socket.gethostname()
    fingerprint = _host_fingerprint(hostname) if hash_hostnames else hostname

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backend = str(dist.get_backend())
        gathered: list[str | None] = [None] * world_size
        dist.all_gather_object(gathered, fingerprint)
        hosts = [h if h is not None else "unknown" for h in gathered]
    else:
        rank, world_size, backend = 0, 1, "none"
        hosts = [fingerprint]

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distinct = len(set(hosts))
    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 0

    return Topology(
        backend=backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        hosts=hosts,
        distinct_hosts=distinct,
        node_count=distinct,
        gpus_per_node=gpus_per_node,
        rendezvous=os.environ.get("MASTER_ADDR"),
        global_batch_size=global_batch_size,
        grad_accum_steps=grad_accum_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def assert_topology_matches(
    topology: Topology,
    *,
    expect_distinct_hosts: bool,
    expect_world_size: int = 0,
) -> None:
    """Abort when the observed topology contradicts the configured claim.

    Deliberately fails *loudly and early*.  A run that was supposed to cross a
    network boundary but silently collapsed onto one host would otherwise produce
    an artifact indistinguishable from real multi-node evidence.
    """
    problems: list[str] = []
    if expect_world_size and topology.world_size != expect_world_size:
        problems.append(f"expected world_size {expect_world_size}, observed {topology.world_size}")
    if expect_distinct_hosts and not topology.is_multi_node:
        problems.append(
            f"configuration claims distinct hosts, but all {topology.world_size} rank(s) "
            f"report a single host ({topology.distinct_hosts} distinct). "
            "This run is NOT multi-node evidence."
        )
    if problems:
        raise TopologyError("topology assertion failed:\n  - " + "\n  - ".join(problems))
