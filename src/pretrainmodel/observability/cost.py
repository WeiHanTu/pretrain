"""Cost ledger.

Every run records what it cost, so scaling numbers can be read next to the money
they consumed.  ``rate_usd_per_gpu_hour`` is an explicit assumption recorded in the
manifest rather than a lookup, because published prices change and an artifact must
stay interpretable later.

CPU runs record a zero rate and a zero cost.  That is honest: the laptop time was
free.  It is not a claim that the work was cheap at scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CostRecord", "estimate_cost"]


@dataclass(frozen=True, slots=True)
class CostRecord:
    sku: str
    rate_usd_per_gpu_hour: float
    gpu_hours: float
    estimated_usd: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "rate_usd_per_gpu_hour": self.rate_usd_per_gpu_hour,
            "gpu_hours": round(self.gpu_hours, 6),
            "estimated_usd": round(self.estimated_usd, 6),
            "note": self.note,
        }


def estimate_cost(
    *,
    sku: str,
    rate_usd_per_gpu_hour: float,
    duration_seconds: float,
    world_size: int,
    gpus_per_rank: int = 1,
) -> CostRecord:
    """Wall-clock GPU-hours across the whole job, not per rank.

    Two nodes for one hour is two GPU-hours, and billing does not care that the
    ranks ran concurrently.
    """
    gpu_hours = (duration_seconds / 3600.0) * max(world_size, 1) * max(gpus_per_rank, 0)
    return CostRecord(
        sku=sku,
        rate_usd_per_gpu_hour=rate_usd_per_gpu_hour,
        gpu_hours=gpu_hours,
        estimated_usd=gpu_hours * rate_usd_per_gpu_hour,
        note="Estimate from a recorded rate assumption, not a billing export.",
    )
