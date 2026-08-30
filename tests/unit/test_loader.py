"""Sampler determinism, exactly-once partitioning and resume portability."""

from __future__ import annotations

import pytest

from pretrainmodel.data.coverage import verify_coverage
from pretrainmodel.data.loader import (
    IncompatibleLoaderStateError,
    LoaderState,
    ShardedSampler,
)

IDS = [f"s{i:04d}" for i in range(24)]


def sampler(
    rank: int, world_size: int, *, seed: int = 5, shuffle: bool = True, drop_last: bool = True
) -> ShardedSampler:
    return ShardedSampler(
        IDS,
        seed=seed,
        world_size=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
        dataset_version="v1",
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_duplicate_ids_in_universe_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        ShardedSampler(["a", "a"], seed=1, world_size=1, rank=0)


def test_rank_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ShardedSampler(IDS, seed=1, world_size=2, rank=2)


# --------------------------------------------------------------------------- #
# Order determinism
# --------------------------------------------------------------------------- #


def test_epoch_order_is_identical_on_every_rank() -> None:
    orders = [sampler(r, 4).epoch_order(0) for r in range(4)]
    assert all(o == orders[0] for o in orders)


def test_epoch_order_is_independent_of_world_size() -> None:
    """The load-bearing property for changed-world-size resume.

    If the permutation depended on world size, a checkpoint saved at 2 ranks and
    resumed at 4 would silently reshuffle the epoch, and any loss-continuity claim
    would be comparing two different data streams.
    """
    assert (
        sampler(0, 1).epoch_order(3) == sampler(0, 2).epoch_order(3) == sampler(0, 8).epoch_order(3)
    )


def test_epoch_order_changes_between_epochs() -> None:
    s = sampler(0, 2)
    assert s.epoch_order(0) != s.epoch_order(1)


def test_epoch_order_changes_with_seed() -> None:
    assert sampler(0, 2, seed=1).epoch_order(0) != sampler(0, 2, seed=2).epoch_order(0)


def test_shuffle_disabled_preserves_index_order() -> None:
    assert sampler(0, 1, shuffle=False).epoch_order(7) == IDS


def test_epoch_order_is_a_permutation() -> None:
    assert sorted(sampler(0, 3).epoch_order(2)) == sorted(IDS)


# --------------------------------------------------------------------------- #
# Exactly-once partitioning
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 6, 8])
def test_ranks_partition_the_epoch_exactly(world_size: int) -> None:
    expected = sampler(0, world_size).expected_epoch_ids(0)
    consumed = [sampler(r, world_size).rank_epoch_ids(0) for r in range(world_size)]
    result = verify_coverage(expected, consumed)
    assert result.passed, result.failure_summary()


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_every_rank_gets_the_same_number_of_steps(world_size: int) -> None:
    """Unequal step counts across ranks deadlock a collective, so they must match."""
    counts = {len(sampler(r, world_size).rank_epoch_ids(0)) for r in range(world_size)}
    assert len(counts) == 1
    assert counts.pop() == sampler(0, world_size).samples_per_rank()


def test_drop_last_truncates_to_a_divisible_count() -> None:
    s = sampler(0, 5, drop_last=True)
    assert s.usable_count() == 20
    assert len(s.expected_epoch_ids(0)) == 20


def test_drop_last_disabled_keeps_every_sample() -> None:
    s = sampler(0, 5, drop_last=False)
    assert s.usable_count() == len(IDS)
    consumed = [sampler(r, 5, drop_last=False).rank_epoch_ids(0) for r in range(5)]
    assert verify_coverage(s.expected_epoch_ids(0), consumed).passed


def test_ranks_are_disjoint() -> None:
    a = set(sampler(0, 4).rank_epoch_ids(0))
    b = set(sampler(1, 4).rank_epoch_ids(0))
    assert not (a & b)


# --------------------------------------------------------------------------- #
# Resume at the same world size
# --------------------------------------------------------------------------- #


def test_resume_offset_round_trips() -> None:
    s = sampler(0, 2)
    state = s.state_dict(epoch=0, rank_samples_consumed=3)
    assert state.global_samples_consumed == 6
    assert s.resume_offset(state) == 3


def test_resume_yields_exactly_the_unconsumed_tail() -> None:
    world_size = 2
    consumed_per_rank = 3
    states = [
        sampler(r, world_size).state_dict(epoch=0, rank_samples_consumed=consumed_per_rank)
        for r in range(world_size)
    ]
    before = [
        sampler(r, world_size).rank_epoch_ids(0)[:consumed_per_rank] for r in range(world_size)
    ]
    after = [sampler(r, world_size).rank_epoch_ids_from(states[r]) for r in range(world_size)]

    expected = sampler(0, world_size).expected_epoch_ids(0)
    combined = [b + a for b, a in zip(before, after, strict=True)]
    assert verify_coverage(expected, combined).passed


# --------------------------------------------------------------------------- #
# Resume across a world-size change (spec 8.3)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("old_ws", "new_ws"), [(2, 4), (4, 2), (1, 2), (2, 1), (4, 8), (8, 4)])
def test_resume_at_a_different_world_size_consumes_each_sample_once(
    old_ws: int, new_ws: int
) -> None:
    """Save under one world size, resume under another, still exactly once.

    This is the operational scenario that matters: a node dies, the job comes back
    at a smaller world size, and training must continue over the *remaining* data
    without replaying or skipping any of it.
    """
    consumed_per_rank = 2
    consumed_before = [
        sampler(r, old_ws).rank_epoch_ids(0)[:consumed_per_rank] for r in range(old_ws)
    ]
    state = sampler(0, old_ws).state_dict(epoch=0, rank_samples_consumed=consumed_per_rank)

    resumed = [sampler(r, new_ws).rank_epoch_ids_from(state) for r in range(new_ws)]

    expected = sampler(0, old_ws).expected_epoch_ids(0)
    result = verify_coverage(expected, [*consumed_before, *resumed])
    assert result.passed, result.failure_summary()


def test_resume_rejects_a_position_that_would_split_a_sample() -> None:
    """A global position that does not divide by the new world size is refused.

    Rounding it would make some rank replay or skip work, which is precisely the
    silent error the invariant exists to prevent.
    """
    state = LoaderState(
        epoch=0,
        global_samples_consumed=5,
        seed=5,
        shuffle=True,
        drop_last=True,
        dataset_version="v1",
        num_samples=len(IDS),
    )
    with pytest.raises(IncompatibleLoaderStateError, match="not divisible"):
        sampler(0, 2).resume_offset(state)


# --------------------------------------------------------------------------- #
# State compatibility
# --------------------------------------------------------------------------- #


def test_state_rejects_seed_change() -> None:
    state = sampler(0, 2, seed=1).state_dict(epoch=0, rank_samples_consumed=2)
    with pytest.raises(IncompatibleLoaderStateError, match="seed"):
        sampler(0, 2, seed=2).resume_offset(state)


def test_state_rejects_shuffle_change() -> None:
    state = sampler(0, 2, shuffle=True).state_dict(epoch=0, rank_samples_consumed=2)
    with pytest.raises(IncompatibleLoaderStateError, match="shuffle"):
        sampler(0, 2, shuffle=False).resume_offset(state)


def test_state_rejects_drop_last_change() -> None:
    state = sampler(0, 2, drop_last=True).state_dict(epoch=0, rank_samples_consumed=2)
    with pytest.raises(IncompatibleLoaderStateError, match="drop_last"):
        sampler(0, 2, drop_last=False).resume_offset(state)


def test_state_rejects_sampler_version_change() -> None:
    state = sampler(0, 2).state_dict(epoch=0, rank_samples_consumed=2)
    stale = LoaderState(**{**state.to_dict(), "sampler_version": 0})
    with pytest.raises(IncompatibleLoaderStateError, match="sampler_version"):
        sampler(0, 2).resume_offset(stale)


def test_state_rejects_dataset_size_change() -> None:
    state = sampler(0, 2).state_dict(epoch=0, rank_samples_consumed=2)
    smaller = ShardedSampler(IDS[:12], seed=5, world_size=2, rank=0, dataset_version="v1")
    with pytest.raises(IncompatibleLoaderStateError, match="dataset size changed"):
        smaller.resume_offset(state)


def test_state_rejects_dataset_version_change() -> None:
    state = sampler(0, 2).state_dict(epoch=0, rank_samples_consumed=2)
    other = ShardedSampler(IDS, seed=5, world_size=2, rank=0, dataset_version="v2")
    with pytest.raises(IncompatibleLoaderStateError, match="dataset_version"):
        other.resume_offset(state)


def test_state_reports_every_incompatibility() -> None:
    state = sampler(0, 2, seed=1, shuffle=True).state_dict(epoch=0, rank_samples_consumed=2)
    with pytest.raises(IncompatibleLoaderStateError) as exc:
        sampler(0, 2, seed=9, shuffle=False).resume_offset(state)
    assert "seed" in str(exc.value)
    assert "shuffle" in str(exc.value)


def test_state_serialises_round_trip() -> None:
    state = sampler(0, 2).state_dict(epoch=3, rank_samples_consumed=4)
    assert LoaderState.from_dict(state.to_dict()) == state
