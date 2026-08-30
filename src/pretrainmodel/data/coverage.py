"""The exactly-once sample-consumption invariant.

Silent data duplication is the failure this project is built to catch.  It does
not crash, it does not spike the loss, and it shows up in no throughput metric --
it quietly trains the model on a skewed distribution while every dashboard looks
healthy.  The only way to know it is not happening is to assert it.

The invariant (spec 4.4), for an epoch consumed without replacement::

    multiset_union(consumed_ids_by_rank) == expected_epoch_ids
    and  consumed_ids[i] disjoint from consumed_ids[j]   for all i != j

Both halves matter.  The first catches a sampler that drops or repeats work
globally.  The second catches the classic distributed bug in which every rank
iterates the *entire* dataset: the loss curve looks fine, throughput looks fine,
and an "epoch" silently means world_size passes over the data.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CoverageResult",
    "RankOverlap",
    "verify_coverage",
    "write_coverage_artifact",
]

# Offending-ID lists are capped so a badly broken run cannot emit a 200 MB
# artifact.  Totals stay exact; only the printed examples are truncated.
MAX_REPORTED_IDS = 64


@dataclass(frozen=True, slots=True)
class RankOverlap:
    """Two ranks that consumed at least one identical sample."""

    rank_a: int
    rank_b: int
    shared_ids: list[str]


@dataclass(frozen=True, slots=True)
class CoverageResult:
    """Outcome of the invariant check for one epoch."""

    world_size: int
    epoch: int
    expected_count: int
    consumed_count: int
    per_rank_counts: list[int]
    duplicated_ids: list[str]
    missing_ids: list[str]
    unexpected_ids: list[str]
    rank_overlaps: list[RankOverlap]
    duplicated_total: int = 0
    missing_total: int = 0
    unexpected_total: int = 0
    truncated_id_lists: bool = False
    sampler: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.duplicated_total == 0
            and self.missing_total == 0
            and self.unexpected_total == 0
            and not self.rank_overlaps
            and self.expected_count == self.consumed_count
        )

    def failure_summary(self) -> str:
        """Human-readable reason, naming offending IDs so the bug is actionable."""
        if self.passed:
            return "coverage invariant holds"
        parts: list[str] = [
            f"coverage invariant VIOLATED at world_size={self.world_size}, epoch={self.epoch}",
            f"  expected {self.expected_count} samples, consumed {self.consumed_count}",
        ]
        if self.duplicated_total:
            parts.append(
                f"  {self.duplicated_total} duplicated id(s), e.g. {self.duplicated_ids[:8]}"
            )
        if self.missing_total:
            parts.append(f"  {self.missing_total} missing id(s), e.g. {self.missing_ids[:8]}")
        if self.unexpected_total:
            parts.append(
                f"  {self.unexpected_total} unexpected id(s), e.g. {self.unexpected_ids[:8]}"
            )
        parts.extend(
            f"  ranks {ov.rank_a} and {ov.rank_b} share "
            f"{len(ov.shared_ids)} id(s), e.g. {ov.shared_ids[:4]}"
            for ov in self.rank_overlaps[:8]
        )
        parts.append(f"  per-rank counts: {self.per_rank_counts}")
        return "\n".join(parts)

    def to_artifact(self, sampler: dict[str, Any] | None = None) -> dict[str, Any]:
        """Serialise to the shape declared by schemas/coverage.schema.json."""
        return {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "world_size": self.world_size,
            "epoch": self.epoch,
            "sampler": sampler if sampler is not None else self.sampler,
            "expected_count": self.expected_count,
            "consumed_count": self.consumed_count,
            "per_rank_counts": self.per_rank_counts,
            "passed": self.passed,
            "duplicated_ids": self.duplicated_ids,
            "missing_ids": self.missing_ids,
            "unexpected_ids": self.unexpected_ids,
            "rank_overlaps": [
                {"rank_a": o.rank_a, "rank_b": o.rank_b, "shared_ids": o.shared_ids}
                for o in self.rank_overlaps
            ],
            "truncated_id_lists": self.truncated_id_lists,
        }


def verify_coverage(
    expected_ids: Sequence[str],
    consumed_by_rank: Sequence[Sequence[str]],
    *,
    epoch: int = 0,
    sampler: dict[str, Any] | None = None,
) -> CoverageResult:
    """Check the exactly-once invariant for one epoch.

    ``expected_ids`` is the global epoch order after any ``drop_last`` truncation.
    ``consumed_by_rank[r]`` is what rank *r* actually pulled, in order.
    """
    expected_counter = Counter(expected_ids)
    consumed_counter: Counter[str] = Counter()
    for ids in consumed_by_rank:
        consumed_counter.update(ids)

    duplicated = sorted(i for i, c in consumed_counter.items() if c > expected_counter.get(i, 0))
    missing = sorted(i for i, c in expected_counter.items() if consumed_counter.get(i, 0) < c)
    unexpected = sorted(i for i in consumed_counter if i not in expected_counter)

    overlaps: list[RankOverlap] = []
    rank_sets = [set(ids) for ids in consumed_by_rank]
    for a in range(len(rank_sets)):
        for b in range(a + 1, len(rank_sets)):
            shared = rank_sets[a] & rank_sets[b]
            if shared:
                overlaps.append(
                    RankOverlap(rank_a=a, rank_b=b, shared_ids=sorted(shared)[:MAX_REPORTED_IDS])
                )

    truncated = (
        len(duplicated) > MAX_REPORTED_IDS
        or len(missing) > MAX_REPORTED_IDS
        or len(unexpected) > MAX_REPORTED_IDS
    )

    return CoverageResult(
        world_size=len(consumed_by_rank),
        epoch=epoch,
        expected_count=len(expected_ids),
        consumed_count=sum(len(ids) for ids in consumed_by_rank),
        per_rank_counts=[len(ids) for ids in consumed_by_rank],
        duplicated_ids=duplicated[:MAX_REPORTED_IDS],
        missing_ids=missing[:MAX_REPORTED_IDS],
        unexpected_ids=unexpected[:MAX_REPORTED_IDS],
        rank_overlaps=overlaps,
        duplicated_total=len(duplicated),
        missing_total=len(missing),
        unexpected_total=len(unexpected),
        truncated_id_lists=truncated,
        sampler=sampler or {},
    )


def write_coverage_artifact(
    path: Path,
    result: CoverageResult,
    sampler: dict[str, Any] | None = None,
) -> None:
    """Write the coverage artifact named by plan.md A2."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_artifact(sampler), indent=2, sort_keys=True) + "\n")
