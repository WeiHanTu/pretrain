"""Rank-correct, resumable, world-size-portable sample ordering.

This module is the one that makes the two headline claims verifiable, so its two
design decisions are stated explicitly.

**The epoch order does not depend on world size.**  The permutation for an epoch is
drawn from ``(seed, epoch)`` alone.  Every rank, at every world size, derives the
same global order and then takes a slice of it.  If the order depended on the
number of ranks, a checkpoint saved at world size 2 and resumed at world size 4
could not be said to continue the same epoch at all -- the samples would be
reshuffled underneath the resume, and any "loss continuity" claim would be
comparing two different data streams.

**Loader position is stored as a global sample count, never a per-rank cursor.**
A per-rank cursor is meaningless after a reshard: rank 1 of 2 and rank 1 of 4 are
not at the same place in the epoch.  Storing the global count and dividing at load
time is what lets a run resume at a different world size and still consume each
remaining sample exactly once.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "SAMPLER_VERSION",
    "IncompatibleLoaderStateError",
    "LoaderState",
    "ShardedSampler",
]

# Bump when the ordering algorithm changes.  A checkpoint written by an older
# sampler must not be silently resumed under new ordering semantics.
SAMPLER_VERSION = 1


def _epoch_seed(seed: int, epoch: int) -> int:
    """Derive a per-epoch seed that is identical on every rank.

    Hashed rather than ``seed + epoch`` so that neighbouring runs (seed 1 epoch 1
    and seed 2 epoch 0) do not silently share a permutation.
    """
    payload = f"{SAMPLER_VERSION}|{seed}|{epoch}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class LoaderState:
    """Serialisable dataloader position, checkpointed with the model.

    ``global_samples_consumed`` counts across all ranks, which is what makes the
    state portable across a world-size change.
    """

    epoch: int
    global_samples_consumed: int
    seed: int
    shuffle: bool
    drop_last: bool
    sampler_version: int = SAMPLER_VERSION
    dataset_version: str = ""
    num_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LoaderState:
        return cls(**raw)


class IncompatibleLoaderStateError(RuntimeError):
    """A stored loader state cannot be resumed under the current configuration."""


class ShardedSampler:
    """Deterministic, exactly-once sample assignment across ranks.

    Assignment is strided (``order[rank::world_size]``) rather than contiguous.
    Both satisfy the coverage invariant; strided is used because it keeps each
    rank's samples spread across the whole epoch, so a shard-correlated defect
    shows up on every rank at once instead of hiding on one.
    """

    def __init__(
        self,
        sample_ids: Sequence[str],
        *,
        seed: int,
        world_size: int,
        rank: int,
        shuffle: bool = True,
        drop_last: bool = True,
        dataset_version: str = "",
    ) -> None:
        if world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {world_size}")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("sample_ids contains duplicates; the index is not a valid universe")

        self._ids = list(sample_ids)
        self.seed = seed
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset_version = dataset_version

    # ----------------------------------------------------------------- #
    # Global order
    # ----------------------------------------------------------------- #

    def epoch_order(self, epoch: int) -> list[str]:
        """The full global order for an epoch, before any rank slicing.

        Identical on every rank and independent of world size.
        """
        if not self.shuffle:
            return list(self._ids)
        rng = np.random.default_rng(_epoch_seed(self.seed, epoch))
        perm = rng.permutation(len(self._ids))
        return [self._ids[i] for i in perm]

    def usable_count(self) -> int:
        """How many samples an epoch actually yields under the current settings."""
        n = len(self._ids)
        if self.drop_last:
            return (n // self.world_size) * self.world_size
        return n

    def expected_epoch_ids(self, epoch: int) -> list[str]:
        """The exact multiset the union of all ranks must consume this epoch.

        This is the right-hand side of the coverage invariant.  With
        ``drop_last`` the tail that cannot be divided evenly is excluded here, so
        the invariant stays exact rather than being weakened to "approximately all".
        """
        return self.epoch_order(epoch)[: self.usable_count()]

    # ----------------------------------------------------------------- #
    # Rank view
    # ----------------------------------------------------------------- #

    def rank_epoch_ids(self, epoch: int) -> list[str]:
        """This rank's slice of the epoch."""
        return self.expected_epoch_ids(epoch)[self.rank :: self.world_size]

    def samples_per_rank(self) -> int:
        """Steps per epoch are derived from this; ranks must agree or collectives hang."""
        if self.drop_last:
            return self.usable_count() // self.world_size
        return len(range(self.rank, self.usable_count(), self.world_size))

    # ----------------------------------------------------------------- #
    # Resume
    # ----------------------------------------------------------------- #

    def state_dict(self, epoch: int, rank_samples_consumed: int) -> LoaderState:
        """Capture position as a global count.

        With strided assignment and lockstep ranks, the global count is the
        per-rank count times the world size.  Recording the global figure is what
        survives a reshard.
        """
        return LoaderState(
            epoch=epoch,
            global_samples_consumed=rank_samples_consumed * self.world_size,
            seed=self.seed,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            sampler_version=SAMPLER_VERSION,
            dataset_version=self.dataset_version,
            num_samples=len(self._ids),
        )

    def validate_state(self, state: LoaderState) -> None:
        """Refuse a state that cannot mean what it says under this configuration."""
        problems: list[str] = []
        if state.sampler_version != SAMPLER_VERSION:
            problems.append(
                f"sampler_version {state.sampler_version} != current {SAMPLER_VERSION}; "
                "ordering semantics changed and the epoch cannot be continued"
            )
        if state.seed != self.seed:
            problems.append(f"seed {state.seed} != current {self.seed}")
        if state.shuffle != self.shuffle:
            problems.append(f"shuffle {state.shuffle} != current {self.shuffle}")
        if state.drop_last != self.drop_last:
            problems.append(f"drop_last {state.drop_last} != current {self.drop_last}")
        if state.num_samples and state.num_samples != len(self._ids):
            problems.append(
                f"dataset size changed: state has {state.num_samples}, index has {len(self._ids)}"
            )
        if (
            state.dataset_version
            and self.dataset_version
            and state.dataset_version != self.dataset_version
        ):
            problems.append(
                f"dataset_version {state.dataset_version!r} != current {self.dataset_version!r}"
            )
        if problems:
            raise IncompatibleLoaderStateError(
                "cannot resume dataloader state:\n  - " + "\n  - ".join(problems)
            )

    def resume_offset(self, state: LoaderState) -> int:
        """Per-rank index to resume from, derived from the stored global position.

        A global position that does not divide evenly by the new world size would
        make some rank replay or skip a sample, so it is rejected rather than
        rounded.
        """
        self.validate_state(state)
        if state.global_samples_consumed % self.world_size != 0:
            raise IncompatibleLoaderStateError(
                f"global_samples_consumed={state.global_samples_consumed} is not divisible "
                f"by world_size={self.world_size}; resuming would replay or skip samples. "
                "Checkpoint at a step boundary, or resume at a compatible world size."
            )
        return state.global_samples_consumed // self.world_size

    def rank_epoch_ids_from(self, state: LoaderState) -> list[str]:
        """The remainder of this rank's epoch after a resume."""
        return self.rank_epoch_ids(state.epoch)[self.resume_offset(state) :]

    # ----------------------------------------------------------------- #
    # Iteration
    # ----------------------------------------------------------------- #

    def iter_epoch(self, epoch: int, start_offset: int = 0) -> Iterator[str]:
        yield from self.rank_epoch_ids(epoch)[start_offset:]
