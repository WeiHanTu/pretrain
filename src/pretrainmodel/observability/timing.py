"""Step-phase timing, memory and straggler detection (spec 10.2).

The measurement problem this module exists to get right
-------------------------------------------------------
CUDA kernels are asynchronous. A wall clock wrapped around a forward pass measures
how long it took to *enqueue* the work, not how long the GPU spent doing it. The
resulting numbers are not noisy, they are wrong in a specific and flattering
direction: forward looks nearly free and whatever runs before the next
synchronisation point absorbs the cost. They also look entirely plausible.

Two honest options exist:

1. ``torch.cuda.synchronize()`` around each phase. Accurate, and it destroys the
   overlap it is measuring, so the totals no longer describe the run you care about.
2. CUDA events recorded in-stream, resolved later. The recording is cheap and does
   not serialise; only the read requires a synchronisation, and that is deferred to
   the step boundary where the optimizer step already forces one.

This module uses (2) on CUDA and ``perf_counter`` on CPU, and records
``timing_method`` in every artifact. A step-time breakdown that does not say how it
was measured cannot be compared against another one.

Overhead is not assumed to be negligible: ``PhaseTimer`` can be disabled, and the
per-step total is recorded alongside the sum of phases so the gap between them is
visible rather than hidden.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = [
    "PhaseTimer",
    "StragglerReport",
    "TimingAccumulator",
    "memory_snapshot",
    "reset_memory_stats",
    "straggler_report",
    "summarize_durations",
]


class PhaseTimer:
    """Time named phases within one training step.

    Usage::

        timer = PhaseTimer(device)
        with timer.phase("forward"):
            ...
        durations = timer.finalize()   # seconds, per phase
    """

    def __init__(self, device: torch.device | None = None, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._use_events = bool(
            enabled and device is not None and device.type == "cuda" and torch.cuda.is_available()
        )
        self._cpu: dict[str, float] = {}
        self._events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @property
    def method(self) -> str:
        if not self.enabled:
            return "disabled"
        return "cuda_events" if self._use_events else "perf_counter"

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        if self._use_events:
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            try:
                yield
            finally:
                end.record()
                self._events.append((name, start, end))
        else:
            begin = time.perf_counter()
            try:
                yield
            finally:
                self._cpu[name] = self._cpu.get(name, 0.0) + (time.perf_counter() - begin)

    def finalize(self) -> dict[str, float]:
        """Resolve and return per-phase durations in seconds, then reset.

        On CUDA this is the one synchronisation per step. It sits at the step
        boundary, where the optimizer step has already forced ordering, so it costs
        far less than synchronising per phase.
        """
        if not self.enabled:
            return {}
        if self._use_events:
            torch.cuda.synchronize()
            out: dict[str, float] = {}
            for name, start, end in self._events:
                out[name] = out.get(name, 0.0) + start.elapsed_time(end) / 1000.0
            self._events.clear()
            return out
        out = dict(self._cpu)
        self._cpu.clear()
        return out


def reset_memory_stats(device: torch.device | None = None) -> None:
    """Reset peak-memory counters so a run's peak is its own, not a leftover."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def memory_snapshot(device: torch.device | None = None) -> dict[str, int | None]:
    """Peak allocated and reserved bytes.

    Returns ``None`` on CPU rather than 0. Zero would read as "this run used no
    memory", which is a claim; ``None`` is the truth, which is "not measured here".
    """
    if not torch.cuda.is_available():
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def summarize_durations(values: Sequence[float]) -> dict[str, float]:
    """p50/p95/mean/min/max for one series of durations."""
    if not values:
        return {}
    return {
        "count": float(len(values)),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
        "total": sum(values),
    }


@dataclass(frozen=True, slots=True)
class StragglerReport:
    """Per-rank step-time comparison — the detector for incident I-004.

    A straggler does not fail. Every rank waits for the slowest at each collective,
    so one rank 30% slow makes the *whole job* 30% slow while every per-rank metric
    except its own looks healthy. Global averages hide it completely, which is why
    this compares rank medians rather than pooling them.
    """

    per_rank_p50: list[float]
    per_rank_p95: list[float]
    slowest_rank: int
    fastest_rank: int
    straggler_ratio: float
    threshold: float
    detected: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_rank_p50": self.per_rank_p50,
            "per_rank_p95": self.per_rank_p95,
            "slowest_rank": self.slowest_rank,
            "fastest_rank": self.fastest_rank,
            "straggler_ratio": self.straggler_ratio,
            "threshold": self.threshold,
            "detected": self.detected,
            "note": self.note,
        }


def straggler_report(
    per_rank_step_times: Sequence[Sequence[float]], *, threshold: float = 1.15
) -> StragglerReport:
    """Compare rank medians and flag one rank dragging the job.

    ``threshold`` is the ratio of slowest to fastest rank median above which a
    straggler is declared, and it is declared *before* the experiment rather than
    chosen after seeing the spread. 1.15 is deliberately loose: ordinary scheduling
    jitter across ranks runs a few percent, and a detector that fires on that is
    noise.
    """
    medians = [statistics.median(times) if times else float("nan") for times in per_rank_step_times]
    p95s = [_percentile(times, 0.95) for times in per_rank_step_times]
    usable = [m for m in medians if m == m and m > 0]
    if len(usable) < 2:
        return StragglerReport(
            per_rank_p50=medians,
            per_rank_p95=p95s,
            slowest_rank=-1,
            fastest_rank=-1,
            straggler_ratio=float("nan"),
            threshold=threshold,
            detected=False,
            note="fewer than two ranks with usable timings; straggler detection needs a comparison",
        )

    slowest = max(range(len(medians)), key=lambda i: medians[i])
    fastest = min(range(len(medians)), key=lambda i: medians[i])
    ratio = medians[slowest] / medians[fastest]
    detected = ratio > threshold
    return StragglerReport(
        per_rank_p50=medians,
        per_rank_p95=p95s,
        slowest_rank=slowest,
        fastest_rank=fastest,
        straggler_ratio=ratio,
        threshold=threshold,
        detected=detected,
        note=(
            f"rank {slowest} is {(ratio - 1) * 100:.1f}% slower than rank {fastest} by "
            "median step time; every rank waits for it at each collective, so the whole "
            "job pays that penalty"
            if detected
            else f"rank spread {(ratio - 1) * 100:.1f}% is within the declared "
            f"{(threshold - 1) * 100:.0f}% threshold"
        ),
    )


@dataclass
class TimingAccumulator:
    """Collect per-step phase durations across a run."""

    phases: dict[str, list[float]] = field(default_factory=dict)
    step_totals: list[float] = field(default_factory=list)
    method: str = "perf_counter"

    def add(self, total_seconds: float, durations: dict[str, float]) -> None:
        self.step_totals.append(total_seconds)
        for name, value in durations.items():
            self.phases.setdefault(name, []).append(value)

    def summary(self) -> dict[str, Any]:
        """Per-phase and total summaries, plus the unattributed remainder.

        ``unaccounted`` is reported rather than absorbed: if the phases sum to much
        less than the measured step, something meaningful is happening outside the
        instrumented regions and the breakdown should not be read as complete.
        """
        phase_summary = {name: summarize_durations(v) for name, v in self.phases.items()}
        total = summarize_durations(self.step_totals)
        attributed = sum(s.get("total", 0.0) for s in phase_summary.values())
        measured = total.get("total", 0.0)
        return {
            "timing_method": self.method,
            "step_total": total,
            "phases": phase_summary,
            "attributed_seconds": attributed,
            "measured_seconds": measured,
            "unaccounted_seconds": measured - attributed,
            "unaccounted_fraction": (measured - attributed) / measured if measured else 0.0,
        }
