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
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "Topology",
    "TopologyError",
    "assert_topology_matches",
    "capture_topology",
    "instance_zone",
    "software_environment",
]


class TopologyError(RuntimeError):
    """The observed topology contradicts what the run configuration claimed."""


def instance_zone() -> str | None:
    """The GCP zone of this instance, or None when not on GCP.

    Read from the metadata server with a short timeout so a local or non-GCP run
    falls through immediately rather than blocking a training launch on a DNS
    lookup that will never resolve.
    """
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/zone",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(request, timeout=0.5) as response:
            # Returned as "projects/<number>/zones/<zone>".
            return response.read().decode("utf-8").rsplit("/", 1)[-1] or None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _region_of(zone: str | None) -> str | None:
    """us-west4-a -> us-west4. None stays None."""
    if not zone:
        return None
    parts = zone.rsplit("-", 1)
    return parts[0] if len(parts) == 2 and len(parts[1]) == 1 else zone


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
    zones: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    distinct_regions: int = 0

    @property
    def spans_regions(self) -> bool:
        """Whether ranks were observed in more than one cloud region."""
        return self.distinct_regions >= 2

    @property
    def throughput_numbers_comparable(self) -> bool:
        """Whether throughput and scaling figures from this run mean anything.

        A run whose ranks span regions is communicating over a wide-area link with
        roughly two orders of magnitude more latency than an intra-zone one. Its
        scaling efficiency measures that link, not the code, and must never be
        reported as a property of the implementation. Correctness evidence from such
        a run is entirely valid -- arguably a *stronger* network boundary than
        same-zone -- which is exactly why the distinction has to be recorded rather
        than left to whoever reads the number later.
        """
        return not self.spans_regions

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
        d["spans_regions"] = self.spans_regions
        d["throughput_numbers_comparable"] = self.throughput_numbers_comparable
        if self.spans_regions:
            d["throughput_warning"] = (
                f"Ranks span {self.distinct_regions} regions ({sorted(set(self.regions))}). "
                "Throughput, step-time and scaling-efficiency figures from this run "
                "describe the wide-area link between those regions, NOT this "
                "implementation, and must not be reported as scaling results. "
                "Correctness and recovery evidence from this run is unaffected."
            )
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
    zone = instance_zone()

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backend = str(dist.get_backend())
        gathered: list[tuple[str, str | None] | None] = [None] * world_size
        dist.all_gather_object(gathered, (fingerprint, zone))
        pairs = [g if g is not None else ("unknown", None) for g in gathered]
        hosts = [h for h, _ in pairs]
        zones = [z or "unknown" for _, z in pairs]
    else:
        rank, world_size, backend = 0, 1, "none"
        hosts = [fingerprint]
        zones = [zone or "unknown"]

    regions = [_region_of(z) or "unknown" for z in zones]
    distinct_regions = len({r for r in regions if r != "unknown"})

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
        zones=zones,
        regions=regions,
        distinct_regions=distinct_regions,
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
