"""Phase timing, memory reporting and straggler detection."""

from __future__ import annotations

import time

import pytest
import torch

from pretrainmodel.observability.timing import (
    PhaseTimer,
    TimingAccumulator,
    memory_snapshot,
    straggler_report,
    summarize_durations,
)

# --------------------------------------------------------------------------- #
# PhaseTimer
# --------------------------------------------------------------------------- #


def test_phases_are_measured_and_named() -> None:
    timer = PhaseTimer(torch.device("cpu"))
    with timer.phase("a"):
        time.sleep(0.01)
    with timer.phase("b"):
        time.sleep(0.02)
    out = timer.finalize()
    assert set(out) == {"a", "b"}
    assert out["b"] > out["a"] > 0


def test_repeated_phases_accumulate() -> None:
    """Gradient accumulation calls each phase once per micro-batch."""
    timer = PhaseTimer(torch.device("cpu"))
    for _ in range(3):
        with timer.phase("forward"):
            time.sleep(0.005)
    out = timer.finalize()
    assert out["forward"] > 0.01


def test_finalize_resets_so_steps_do_not_leak_into_each_other() -> None:
    timer = PhaseTimer(torch.device("cpu"))
    with timer.phase("x"):
        time.sleep(0.005)
    first = timer.finalize()
    assert timer.finalize() == {}
    assert first["x"] > 0


def test_timer_can_be_disabled_and_costs_nothing() -> None:
    """Instrumentation that cannot be switched off distorts what it measures."""
    timer = PhaseTimer(torch.device("cpu"), enabled=False)
    with timer.phase("x"):
        pass
    assert timer.finalize() == {}
    assert timer.method == "disabled"


def test_method_is_recorded_and_honest_about_the_device() -> None:
    """A breakdown that does not say how it was measured cannot be compared.

    Wall-clock timing around async CUDA work measures kernel launch, not execution,
    so the method is part of the result rather than an implementation detail.
    """
    assert PhaseTimer(torch.device("cpu")).method == "perf_counter"
    expected = "cuda_events" if torch.cuda.is_available() else "perf_counter"
    assert (
        PhaseTimer(torch.device("cuda" if torch.cuda.is_available() else "cpu")).method == expected
    )


def test_exceptions_do_not_lose_the_measurement() -> None:
    timer = PhaseTimer(torch.device("cpu"))
    with pytest.raises(ValueError), timer.phase("boom"):
        raise ValueError("x")
    assert "boom" in timer.finalize()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_summarize_reports_percentiles() -> None:
    s = summarize_durations([1.0, 2.0, 3.0, 4.0, 100.0])
    assert s["min"] == 1.0 and s["max"] == 100.0
    assert s["p50"] == 3.0
    assert s["total"] == 110.0


def test_summarize_handles_empty_input() -> None:
    assert summarize_durations([]) == {}


def test_unaccounted_time_is_reported_not_absorbed() -> None:
    """If phases do not sum to the step, the breakdown is incomplete and must say so."""
    acc = TimingAccumulator()
    acc.add(1.0, {"forward": 0.3, "backward": 0.4})
    summary = acc.summary()
    assert summary["attributed_seconds"] == pytest.approx(0.7)
    assert summary["unaccounted_seconds"] == pytest.approx(0.3)
    assert summary["unaccounted_fraction"] == pytest.approx(0.3)


def test_fully_attributed_step_reports_no_gap() -> None:
    acc = TimingAccumulator()
    acc.add(1.0, {"forward": 0.5, "backward": 0.5})
    assert acc.summary()["unaccounted_fraction"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #


def test_memory_is_none_on_cpu_rather_than_zero() -> None:
    """Zero would read as 'this run used no memory', which is a claim; None is the truth."""
    snap = memory_snapshot()
    if torch.cuda.is_available():
        assert snap["peak_allocated_bytes"] is not None
    else:
        assert snap["peak_allocated_bytes"] is None
        assert snap["peak_reserved_bytes"] is None


# --------------------------------------------------------------------------- #
# Straggler detection (incident I-004)
# --------------------------------------------------------------------------- #


def test_balanced_ranks_are_not_flagged() -> None:
    report = straggler_report([[0.10, 0.11, 0.10], [0.10, 0.10, 0.11]])
    assert report.detected is False
    assert report.straggler_ratio < 1.15


def test_a_slow_rank_is_detected_and_named() -> None:
    """One rank 40% slow taxes the whole job; every other per-rank metric looks fine."""
    report = straggler_report([[0.10] * 5, [0.14] * 5, [0.10] * 5])
    assert report.detected is True
    assert report.slowest_rank == 1
    assert report.fastest_rank in (0, 2)
    assert report.straggler_ratio == pytest.approx(1.4, rel=1e-3)
    assert "40.0% slower" in report.note


def test_threshold_is_declared_and_recorded() -> None:
    """The threshold is fixed before the experiment, not chosen after seeing the spread."""
    report = straggler_report([[0.10] * 3, [0.12] * 3], threshold=1.5)
    assert report.threshold == 1.5
    assert report.detected is False, "20% spread must not trip a 50% threshold"
    assert straggler_report([[0.10] * 3, [0.12] * 3], threshold=1.1).detected is True


def test_ordinary_jitter_does_not_trip_the_default() -> None:
    """A detector that fires on scheduling noise is noise."""
    assert straggler_report([[0.100] * 5, [0.103] * 5, [0.101] * 5]).detected is False


def test_single_rank_cannot_support_a_straggler_claim() -> None:
    report = straggler_report([[0.1, 0.1]])
    assert report.detected is False
    assert "needs a comparison" in report.note


def test_report_serialises_for_the_artifact() -> None:
    payload = straggler_report([[0.10] * 3, [0.20] * 3]).to_dict()
    assert payload["detected"] is True
    assert payload["threshold"] == 1.15
    assert payload["slowest_rank"] == 1
