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
