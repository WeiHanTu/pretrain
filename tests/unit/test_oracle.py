"""The seed-variance oracle: construction, pass rule and its declared limits."""

from __future__ import annotations

import pytest

from pretrainmodel.training.oracle import (
    SeedBand,
    build_seed_band,
    evaluate_against_band,
)

CONTROLS = {
    1: [1.00, 0.80, 0.60, 0.50, 0.45, 0.42, 0.40, 0.39],
    2: [1.02, 0.83, 0.62, 0.52, 0.46, 0.43, 0.41, 0.40],
    3: [0.98, 0.79, 0.58, 0.49, 0.44, 0.41, 0.39, 0.38],
}
HASH = "a" * 64


def _band(**kw: float) -> SeedBand:
    return build_seed_band(CONTROLS, config_hash=HASH, **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #


def test_band_requires_at_least_three_seeds() -> None:
    """Two seeds carry almost no information about spread."""
    with pytest.raises(ValueError, match="at least 3"):
        build_seed_band({1: [1.0], 2: [1.0]}, config_hash=HASH)


def test_band_rejects_control_runs_of_different_lengths() -> None:
    bad = {1: [1.0, 0.5], 2: [1.0, 0.5], 3: [1.0]}
    with pytest.raises(ValueError, match="differing step counts"):
        build_seed_band(bad, config_hash=HASH)


def test_band_contains_every_control_run() -> None:
    """A band that excludes its own controls would be self-contradictory."""
    band = _band()
    for seed, losses in CONTROLS.items():
        result = evaluate_against_band(band, losses)
        assert result.passed, f"seed {seed} fell outside its own band: {result.reason}"


def test_widening_is_applied() -> None:
    band = _band(margin_factor=1.0)
    raw_lo = min(CONTROLS[s][0] for s in CONTROLS)
    raw_hi = max(CONTROLS[s][0] for s in CONTROLS)
    assert band.lower[0] < raw_lo
    assert band.upper[0] > raw_hi
    assert band.upper[0] - band.lower[0] == pytest.approx(3 * (raw_hi - raw_lo))


def test_zero_margin_reduces_to_the_raw_envelope() -> None:
    band = _band(margin_factor=0.0)
    assert band.lower[0] == pytest.approx(min(CONTROLS[s][0] for s in CONTROLS))
    assert band.upper[0] == pytest.approx(max(CONTROLS[s][0] for s in CONTROLS))


# --------------------------------------------------------------------------- #
# Pass rule
# --------------------------------------------------------------------------- #


def test_a_clearly_diverged_run_fails() -> None:
    band = _band()
    diverged = [x * 3 + 1.0 for x in CONTROLS[1]]
    result = evaluate_against_band(band, diverged)
    assert not result.passed
    assert result.outside_steps
    assert "inside the band" in result.reason


def test_a_run_that_flatlines_high_fails_on_the_trailing_check() -> None:
    """Tracking the band early then stalling must not pass."""
    band = _band()
    stalled = [*CONTROLS[1][:4], 0.9, 0.9, 0.9, 0.9]
    result = evaluate_against_band(band, stalled)
    assert not result.passed
    assert not result.trailing_inside


def test_a_band_from_another_config_is_refused() -> None:
    """A band is only valid for the configuration it was built from."""
    band = _band()
    result = evaluate_against_band(band, CONTROLS[1], config_hash="b" * 64)
    assert not result.passed
    assert "config hash mismatch" in result.reason


def test_no_overlapping_steps_is_a_failure_not_a_pass() -> None:
    """An empty comparison must never be reported as agreement."""
    band = _band()
    result = evaluate_against_band(band, [0.4], start_step=999)
    assert not result.passed
    assert result.compared_steps == 0


def test_post_resume_tail_is_compared_at_the_right_steps() -> None:
    """A resumed tail is judged against the band positions it corresponds to."""
    band = _band()
    tail = CONTROLS[2][4:]
    assert evaluate_against_band(band, tail, start_step=4).passed
    # The same values compared from step 0 sit against the early, much wider
    # part of the curve and must not be silently accepted as a match.
    shifted = evaluate_against_band(band, tail, start_step=0)
    assert shifted.inside_fraction < 1.0


# --------------------------------------------------------------------------- #
# Serialisation and disclosure
# --------------------------------------------------------------------------- #


def test_band_round_trips() -> None:
    band = _band()
    assert SeedBand.from_dict(band.to_dict()).to_dict() == band.to_dict()


def test_artifact_declares_its_rule_and_limitations() -> None:
    """The artifact must state the pass rule and the small-sample caveat."""
    payload = _band().to_dict()
    assert payload["interval_rule"]
    assert payload["pass_rule"]
    assert "(n-1)/(n+1)" in payload["limitations"]
    assert "never bit-exactness" in payload["limitations"]
    assert payload["config_hash"] == HASH
