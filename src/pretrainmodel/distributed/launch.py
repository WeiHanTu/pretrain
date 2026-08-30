"""Process-group bring-up from the environment, the way torchrun provides it.

Everything before Phase C used ``file://`` rendezvous, which works only when every
rank shares a filesystem.  That is fine for local tests and useless on real
hardware: across hosts, ranks find each other over TCP via ``MASTER_ADDR`` and
``MASTER_PORT``.  Those are different code paths, and the TCP one is where the
interesting failures live -- a closed firewall port, a rendezvous timeout, a rank
binding the wrong interface.

So this module exists to make the *paid* launch path the one that is exercised for
free.  The first hour on rented hardware should be spent measuring, not discovering
that rank 1 cannot reach the master.

On the honesty of the checks below: a non-loopback ``MASTER_ADDR`` does **not**
prove two hosts -- it can be the machine's own LAN address, and two ranks on one
box can rendezvous over it happily.  It is a cheap early warning, nothing more.
The actual proof stays where it was: ``capture_topology`` all-gathers each rank's
hostname and counts distinct ones, and ``assert_topology_matches`` aborts when a
run that claims distinct hosts observes one.
"""

from __future__ import annotations

import os
import socket
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "NetworkPreflight",
    "RendezvousInfo",
    "RendezvousWarning",
    "default_backend",
    "init_distributed",
    "network_preflight",
    "rendezvous_from_env",
    "rendezvous_warnings",
    "shutdown_distributed",
]

_LOOPBACK = {"127.0.0.1", "localhost", "::1", ""}


class RendezvousWarning(UserWarning):
    """A rendezvous setting looks inconsistent with the run's stated intent."""


@dataclass(frozen=True, slots=True)
class RendezvousInfo:
    """How this rank found the others. Recorded verbatim in the run manifest."""

    method: str
    backend: str
    rank: int
    local_rank: int
    world_size: int
    master_addr: str
    master_port: str
    nnodes: int
    node_rank: int
    nproc_per_node: int
    hostname: str
    socket_ifname: str | None = None
    launched_by_torchrun: bool = False

    @property
    def master_is_loopback(self) -> bool:
        return self.master_addr in _LOOPBACK

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NetworkPreflight:
    """Whether this node can participate in a TCP rendezvous at all."""

    hostname: str
    hostname_resolves: bool
    resolved_addresses: list[str]
    problems: list[str]
    fatal: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def network_preflight(*, require_hostname: bool = True) -> NetworkPreflight:
    """Check the node can resolve itself before any rendezvous is attempted.

    Found the hard way while rehearsing the launch path: a node that cannot resolve
    its own hostname makes ``--rdzv-backend=c10d`` retry DNS with backoff **for
    ever**. It never errors, so on rented hardware it bills for the entire hang, and
    the symptom ("the job just sits there") points at the network, the firewall or
    the peer node -- anywhere but ``/etc/hosts``.

    torchrun performs that lookup while building its store, which is *before* any
    timeout this process controls, so the only defence is to check first and refuse
    to launch with an actionable message.
    """
    hostname = socket.gethostname()
    addresses: list[str] = []
    problems: list[str] = []
    resolves = True
    try:
        _, _, addrs = socket.gethostbyname_ex(hostname)
        addresses = list(addrs)
    except OSError as exc:
        resolves = False
        problems.append(
            f"this node cannot resolve its own hostname {hostname!r} ({exc}). "
            "c10d rendezvous will hang indefinitely rather than fail. "
            f"Fix: add '127.0.0.1 {hostname}' to /etc/hosts, or launch with "
            "--rdzv-backend=static and an explicit --master-addr IP."
        )
    return NetworkPreflight(
        hostname=hostname,
        hostname_resolves=resolves,
        resolved_addresses=addresses,
        problems=problems,
        fatal=require_hostname and not resolves,
    )


def default_backend() -> str:
    """NCCL on GPU hosts, Gloo otherwise."""
    return "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"


def rendezvous_from_env() -> RendezvousInfo:
    """Read the rendezvous configuration torchrun exports.

    ``TORCHELASTIC_RUN_ID`` is set only by torchrun, which is how a hand-rolled
    launch is distinguished from an elastic one in the manifest.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    nproc = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    return RendezvousInfo(
        method="env://",
        backend=default_backend(),
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=world_size,
        master_addr=os.environ.get("MASTER_ADDR", ""),
        master_port=os.environ.get("MASTER_PORT", ""),
        nnodes=int(os.environ.get("GROUP_WORLD_SIZE", "1")),
        node_rank=int(os.environ.get("GROUP_RANK", "0")),
        nproc_per_node=nproc,
        hostname=socket.gethostname(),
        socket_ifname=os.environ.get("NCCL_SOCKET_IFNAME") or os.environ.get("GLOO_SOCKET_IFNAME"),
        launched_by_torchrun="TORCHELASTIC_RUN_ID" in os.environ,
    )


def rendezvous_warnings(info: RendezvousInfo, *, expect_distinct_hosts: bool) -> list[str]:
    """Cheap pre-flight warnings, checked before the collective is built.

    These catch a misconfigured launch in seconds rather than after a rendezvous
    timeout, which on a rented cluster is the difference between a typo and a
    wasted reservation. None of them is proof of anything: see the module note.
    """
    problems: list[str] = []
    if expect_distinct_hosts:
        if info.master_is_loopback:
            problems.append(
                f"MASTER_ADDR={info.master_addr!r} is loopback, but this run claims "
                "distinct hosts. Ranks on another machine cannot reach it."
            )
        if info.nnodes < 2:
            problems.append(
                f"GROUP_WORLD_SIZE={info.nnodes} reports a single node, but this run "
                "claims distinct hosts. Check --nnodes."
            )
        if info.world_size == info.nproc_per_node:
            problems.append(
                f"world_size ({info.world_size}) equals nproc_per_node "
                f"({info.nproc_per_node}): every rank is on this host."
            )
    if not info.master_port:
        problems.append("MASTER_PORT is unset; TCP rendezvous cannot be established.")
    return problems


def init_distributed(
    *,
    backend: str | None = None,
    timeout_seconds: int = 600,
    expect_distinct_hosts: bool = False,
) -> RendezvousInfo:
    """Initialise the process group from the environment.

    The rendezvous timeout is bounded and explicit. The default is long enough to
    survive a slow second node joining, and short enough that a firewalled port
    fails visibly instead of hanging until someone notices the bill.
    """
    info = rendezvous_from_env()
    chosen = backend or info.backend

    problems: list[str] = []
    # Only fatal when a real rendezvous across hosts is expected; a loopback
    # single-host run works fine without hostname resolution.
    net = network_preflight(require_hostname=expect_distinct_hosts)
    if net.fatal:
        problems.extend(net.problems)
    problems += rendezvous_warnings(info, expect_distinct_hosts=expect_distinct_hosts)
    if problems:
        raise RuntimeError(
            "rendezvous configuration is inconsistent with the run's stated intent:\n  - "
            + "\n  - ".join(problems)
        )

    if not dist.is_initialized():
        dist.init_process_group(
            backend=chosen,
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
    if chosen == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(info.local_rank)
    return info


def shutdown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
