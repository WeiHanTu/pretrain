"""Rendezvous configuration parsing and the pre-launch consistency checks."""

from __future__ import annotations

import pytest

from pretrainmodel.distributed.launch import (
    network_preflight,
    rendezvous_from_env,
    rendezvous_warnings,
)

TORCHRUN_ENV = {
    "RANK": "3",
    "LOCAL_RANK": "1",
    "WORLD_SIZE": "4",
    "LOCAL_WORLD_SIZE": "2",
    "GROUP_WORLD_SIZE": "2",
    "GROUP_RANK": "1",
    "MASTER_ADDR": "10.128.0.5",
    "MASTER_PORT": "29500",
    "TORCHELASTIC_RUN_ID": "abc",
}


def _env(monkeypatch: pytest.MonkeyPatch, **over: str) -> None:
    for key in [*TORCHRUN_ENV, "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"]:
        monkeypatch.delenv(key, raising=False)
    for k, v in {**TORCHRUN_ENV, **over}.items():
        if v is not None:
            monkeypatch.setenv(k, v)


def test_parses_a_torchrun_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    info = rendezvous_from_env()
    assert (info.rank, info.local_rank, info.world_size) == (3, 1, 4)
    assert (info.nnodes, info.node_rank, info.nproc_per_node) == (2, 1, 2)
    assert info.master_addr == "10.128.0.5"
    assert info.launched_by_torchrun is True
    assert info.method == "env://"


def test_hand_rolled_launch_is_distinguishable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manifest should record whether torchrun actually launched the job."""
    _env(monkeypatch, TORCHELASTIC_RUN_ID=None)  # type: ignore[arg-type]
    assert rendezvous_from_env().launched_by_torchrun is False


def test_defaults_to_single_rank_without_an_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in TORCHRUN_ENV:
        monkeypatch.delenv(key, raising=False)
    info = rendezvous_from_env()
    assert (info.rank, info.world_size, info.nnodes) == (0, 1, 1)


def test_records_the_socket_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "ens4")
    assert rendezvous_from_env().socket_ifname == "ens4"


# --------------------------------------------------------------------------- #
# Pre-launch consistency
# --------------------------------------------------------------------------- #


def test_a_correct_multinode_launch_has_no_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    assert rendezvous_warnings(rendezvous_from_env(), expect_distinct_hosts=True) == []


def test_loopback_master_is_rejected_for_a_multinode_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Peers on another machine cannot reach 127.0.0.1."""
    _env(monkeypatch, MASTER_ADDR="127.0.0.1")
    problems = rendezvous_warnings(rendezvous_from_env(), expect_distinct_hosts=True)
    assert any("loopback" in p for p in problems)


def test_single_node_group_is_rejected_for_a_multinode_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _env(monkeypatch, GROUP_WORLD_SIZE="1")
    problems = rendezvous_warnings(rendezvous_from_env(), expect_distinct_hosts=True)
    assert any("single node" in p for p in problems)


def test_all_ranks_on_one_host_is_rejected_for_a_multinode_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """world_size == nproc_per_node means every rank is local, whatever nnodes says."""
    _env(monkeypatch, WORLD_SIZE="2", LOCAL_WORLD_SIZE="2")
    problems = rendezvous_warnings(rendezvous_from_env(), expect_distinct_hosts=True)
    assert any("every rank is on this host" in p for p in problems)


def test_loopback_is_fine_when_no_multinode_claim_is_made(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local rehearsals must not be blocked by checks meant for real launches."""
    _env(monkeypatch, MASTER_ADDR="127.0.0.1")
    assert rendezvous_warnings(rendezvous_from_env(), expect_distinct_hosts=False) == []


def test_missing_master_port_is_always_a_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, MASTER_PORT="")
    problems = rendezvous_warnings(rendezvous_from_env(), expect_distinct_hosts=False)
    assert any("MASTER_PORT" in p for p in problems)


# --------------------------------------------------------------------------- #
# Network preflight
# --------------------------------------------------------------------------- #


def test_network_preflight_reports_hostname_resolution() -> None:
    """Whatever this host does, the result must be self-consistent and actionable."""
    result = network_preflight(require_hostname=True)
    assert result.hostname
    if result.hostname_resolves:
        assert result.resolved_addresses
        assert result.problems == []
        assert result.fatal is False
    else:
        # The message must name the remedy, not just the symptom: an unresolvable
        # hostname makes c10d retry DNS forever instead of failing.
        assert result.fatal is True
        assert any("/etc/hosts" in p for p in result.problems)


def test_network_preflight_is_not_fatal_when_not_required() -> None:
    assert network_preflight(require_hostname=False).fatal is False


# --------------------------------------------------------------------------- #
# Region awareness
# --------------------------------------------------------------------------- #


def test_cross_region_run_marks_throughput_as_incomparable() -> None:
    """A WAN-spanning run's scaling numbers describe the link, not the code.

    Correctness evidence from such a run is fully valid -- two regions is a
    *stronger* network boundary than one zone -- which is exactly why the artifact
    must distinguish the two rather than leave it to whoever reads the number later.
    """
    from pretrainmodel.distributed.topology import Topology

    topo = Topology(
        backend="nccl",
        rank=0,
        local_rank=0,
        world_size=2,
        hosts=["a1b2c3", "d4e5f6"],
        distinct_hosts=2,
        node_count=2,
        gpus_per_node=1,
        zones=["us-west3-a", "us-west4-a"],
        regions=["us-west3", "us-west4"],
        distinct_regions=2,
    )
    assert topo.is_multi_node is True, "two regions is still genuinely multi-node"
    assert topo.spans_regions is True
    assert topo.throughput_numbers_comparable is False
    payload = topo.to_dict()
    assert "throughput_warning" in payload
    assert "NOT this" in payload["throughput_warning"]
    assert (
        "Correctness and recovery evidence from this run is unaffected"
        in (payload["throughput_warning"])
    )


def test_same_region_run_carries_no_warning() -> None:
    from pretrainmodel.distributed.topology import Topology

    topo = Topology(
        backend="nccl",
        rank=0,
        local_rank=0,
        world_size=2,
        hosts=["a1b2c3", "d4e5f6"],
        distinct_hosts=2,
        node_count=2,
        gpus_per_node=1,
        zones=["us-central1-a", "us-central1-a"],
        regions=["us-central1", "us-central1"],
        distinct_regions=1,
    )
    assert topo.spans_regions is False
    assert topo.throughput_numbers_comparable is True
    assert "throughput_warning" not in topo.to_dict()


def test_region_is_derived_from_zone() -> None:
    from pretrainmodel.distributed.topology import _region_of

    assert _region_of("us-west4-a") == "us-west4"
    assert _region_of("europe-west1-b") == "europe-west1"
    assert _region_of(None) is None
    assert _region_of("unknown") == "unknown"


def test_instance_zone_returns_none_off_gcp() -> None:
    """Must fall through fast rather than blocking a launch on an unreachable host."""
    from pretrainmodel.distributed.topology import instance_zone

    assert instance_zone() is None
