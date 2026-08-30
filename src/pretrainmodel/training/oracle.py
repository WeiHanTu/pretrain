"""The statistical oracle: a pre-registered seed-variance band (spec 8.2).

Why this exists
---------------
A BF16 run, an asynchronous checkpoint, or a resume at a different world size all
change the order of floating-point reductions.  None of them can be bit-exact, so
"the resumed loss matched" needs a definition of *matched* that was fixed before
the experiment.  Without one, the tolerance is chosen after seeing the result and
the claim is unfalsifiable.

The construction
----------------
Control runs differ only in ``run.seed``.  At each step the band is the min/max
envelope across the control seeds, widened by ``margin_factor x range`` on each
side.

The honest limitation, stated up front
--------------------------------------
The probability that a fresh draw falls inside the range of ``n`` previous draws
is ``(n - 1) / (n + 1)``.  With three seeds that is **50%** -- a raw envelope would
reject half of all legitimate runs.  The widening is therefore not cosmetic and it
is not a post-hoc adjustment: it is a declared compensation for a small control
set, fixed before any resume experiment is run.  More seeds would be better; three
is what the compute budget allows, and saying so is preferable to implying a
precision the sample size does not support.

A band is only valid for the config hash it was built from.  Comparing against a
band built from a different configuration is meaningless, so the hash is recorded
and checked.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BandEvaluation", "SeedBand", "build_seed_band", "evaluate_against_band"]

# Declared before any resume experiment. Changing these invalidates the band.
DEFAULT_MARGIN_FACTOR = 1.0
DEFAULT_MIN_INSIDE_FRACTION = 0.90
DEFAULT_TRAILING_FRACTION = 0.25


@dataclass(frozen=True, slots=True)
class SeedBand:
    """A frozen acceptance band derived from control runs."""

    config_hash: str
    seeds: list[int]
    num_steps: int
    lower: list[float]
    upper: list[float]
    control_losses: dict[int, list[float]]
    margin_factor: float = DEFAULT_MARGIN_FACTOR
    min_inside_fraction: float = DEFAULT_MIN_INSIDE_FRACTION
    trailing_fraction: float = DEFAULT_TRAILING_FRACTION
    statistic: str = "per-step training loss; plus trailing-window mean"
    notes: str = ""

    @property
    def trailing_start(self) -> int:
        return max(0, self.num_steps - max(1, int(self.num_steps * self.trailing_fraction)))

    def trailing_bounds(self) -> tuple[float, float]:
        """Band on the trailing-window mean, from the control runs' own means."""
        means = [
            statistics.fmean(losses[self.trailing_start :])
            for losses in self.control_losses.values()
        ]
        lo, hi = min(means), max(means)
        margin = self.margin_factor * (hi - lo)
        return lo - margin, hi + margin

    def to_dict(self) -> dict[str, Any]:
        lo, hi = self.trailing_bounds()
        return {
            "schema_version": 1,
            "oracle": "seed_variance_band",
            "config_hash": self.config_hash,
            "seeds": self.seeds,
            "num_steps": self.num_steps,
            "statistic": self.statistic,
            "interval_rule": (
                "per-step min/max envelope across control seeds, widened by "
                f"margin_factor={self.margin_factor} x (max - min) on each side"
            ),
            "pass_rule": (
                f"at least {self.min_inside_fraction:.0%} of compared steps inside the "
                "per-step band, AND the trailing-window mean inside the trailing band"
            ),
            "margin_factor": self.margin_factor,
            "min_inside_fraction": self.min_inside_fraction,
            "trailing_fraction": self.trailing_fraction,
            "trailing_start_step": self.trailing_start,
            "trailing_band": {"lower": lo, "upper": hi},
            "lower": self.lower,
            "upper": self.upper,
            "control_losses": {str(k): v for k, v in self.control_losses.items()},
            "limitations": (
                "A fresh draw falls inside the raw range of n control draws with "
                "probability (n-1)/(n+1); with 3 seeds that is 50%. The margin factor "
                "is a pre-registered compensation for the small control set, not a "
                "post-hoc adjustment. This band supports a statistical-equivalence "
                "claim only, never bit-exactness, and is valid only for config_hash."
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SeedBand:
        return cls(
            config_hash=raw["config_hash"],
            seeds=list(raw["seeds"]),
            num_steps=raw["num_steps"],
            lower=list(raw["lower"]),
            upper=list(raw["upper"]),
            control_losses={int(k): list(v) for k, v in raw["control_losses"].items()},
            margin_factor=raw["margin_factor"],
            min_inside_fraction=raw["min_inside_fraction"],
            trailing_fraction=raw["trailing_fraction"],
            statistic=raw.get("statistic", ""),
            notes=raw.get("notes", ""),
        )


def build_seed_band(
    control_losses: dict[int, Sequence[float]],
    *,
    config_hash: str,
    margin_factor: float = DEFAULT_MARGIN_FACTOR,
    min_inside_fraction: float = DEFAULT_MIN_INSIDE_FRACTION,
    trailing_fraction: float = DEFAULT_TRAILING_FRACTION,
    notes: str = "",
) -> SeedBand:
    """Construct the band from control runs.

    Requires at least three seeds: with two, the range carries almost no
    information about spread and the resulting band would be arbitrary.
    """
    if len(control_losses) < 3:
        raise ValueError(
            f"a seed-variance band needs at least 3 control seeds, got {len(control_losses)}"
        )
    lengths = {len(v) for v in control_losses.values()}
    if len(lengths) != 1:
        raise ValueError(f"control runs have differing step counts: {sorted(lengths)}")

    num_steps = lengths.pop()
    lower: list[float] = []
    upper: list[float] = []
    for t in range(num_steps):
        values = [losses[t] for losses in control_losses.values()]
        lo, hi = min(values), max(values)
        margin = margin_factor * (hi - lo)
        lower.append(lo - margin)
        upper.append(hi + margin)

    return SeedBand(
        config_hash=config_hash,
        seeds=sorted(control_losses),
        num_steps=num_steps,
        lower=lower,
        upper=upper,
        control_losses={k: list(v) for k, v in control_losses.items()},
        margin_factor=margin_factor,
        min_inside_fraction=min_inside_fraction,
        trailing_fraction=trailing_fraction,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class BandEvaluation:
    """Outcome of judging a candidate trajectory against a frozen band."""

    passed: bool
    inside_fraction: float
    compared_steps: int
    outside_steps: list[int] = field(default_factory=list)
    trailing_mean: float = float("nan")
    trailing_bounds: tuple[float, float] = (float("nan"), float("nan"))
    trailing_inside: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "inside_fraction": self.inside_fraction,
            "compared_steps": self.compared_steps,
            "outside_steps": self.outside_steps[:64],
            "trailing_mean": self.trailing_mean,
            "trailing_lower": self.trailing_bounds[0],
            "trailing_upper": self.trailing_bounds[1],
            "trailing_inside": self.trailing_inside,
            "reason": self.reason,
        }


def evaluate_against_band(
    band: SeedBand,
    losses: Sequence[float],
    *,
    config_hash: str | None = None,
    start_step: int = 0,
) -> BandEvaluation:
    """Judge a candidate trajectory (for example a resumed run) against the band.

    ``start_step`` lets a post-resume tail be compared against the band positions
    it actually corresponds to, rather than against step 0.
    """
    if config_hash is not None and config_hash != band.config_hash:
        return BandEvaluation(
            passed=False,
            inside_fraction=0.0,
            compared_steps=0,
            reason=(
                f"config hash mismatch: band was built for {band.config_hash[:12]}..., "
                f"candidate is {config_hash[:12]}.... The band does not apply."
            ),
        )

    outside: list[int] = []
    compared = 0
    for offset, loss in enumerate(losses):
        step = start_step + offset
        if step >= band.num_steps:
            break
        compared += 1
        if not band.lower[step] <= loss <= band.upper[step]:
            outside.append(step)

    if compared == 0:
        return BandEvaluation(
            passed=False,
            inside_fraction=0.0,
            compared_steps=0,
            reason="no overlapping steps to compare",
        )

    inside_fraction = (compared - len(outside)) / compared
    tail = [
        loss for offset, loss in enumerate(losses) if start_step + offset >= band.trailing_start
    ]
    lo, hi = band.trailing_bounds()
    trailing_mean = statistics.fmean(tail) if tail else float("nan")
    trailing_inside = bool(tail) and lo <= trailing_mean <= hi

    passed = inside_fraction >= band.min_inside_fraction and trailing_inside
    reasons: list[str] = []
    if inside_fraction < band.min_inside_fraction:
        reasons.append(
            f"only {inside_fraction:.1%} of steps inside the band "
            f"(requires {band.min_inside_fraction:.0%})"
        )
    if not trailing_inside:
        reasons.append(f"trailing mean {trailing_mean:.6g} outside [{lo:.6g}, {hi:.6g}]")

    return BandEvaluation(
        passed=passed,
        inside_fraction=inside_fraction,
        compared_steps=compared,
        outside_steps=outside,
        trailing_mean=trailing_mean,
        trailing_bounds=(lo, hi),
        trailing_inside=trailing_inside,
        reason="; ".join(reasons) if reasons else "within the declared band",
    )
