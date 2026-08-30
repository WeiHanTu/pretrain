"""The coverage verifier, exercised against each injected sampler defect.

A detector that has only ever been shown passing data is not a detector.  Every
defect in ``SAMPLER_DEFECTS`` gets a test proving the invariant fails, and fails
with the offending sample IDs named.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pretrainmodel.data.coverage import (
    MAX_REPORTED_IDS,
    verify_coverage,
    write_coverage_artifact,
)
from pretrainmodel.data.loader import ShardedSampler
from pretrainmodel.incidents.inject import SAMPLER_DEFECTS, inject_sampler_defect

IDS = [f"s{i:04d}" for i in range(32)]
WORLD_SIZE = 4


def _sampler(rank: int, world_size: int = WORLD_SIZE) -> ShardedSampler:
    return ShardedSampler(IDS, seed=17, world_size=world_size, rank=rank)


def _run(defect: str, world_size: int = WORLD_SIZE) -> tuple[list[str], list[list[str]]]:
    """Simulate one epoch across ranks with a defect applied."""
    expected = _sampler(0, world_size).expected_epoch_ids(0)
    consumed = [
        inject_sampler_defect(
            _sampler(r, world_size).rank_epoch_ids(0),
            defect,
            rank=r,
            global_order=expected,
        )
        for r in range(world_size)
    ]
    return expected, consumed


# --------------------------------------------------------------------------- #
# Clean case
# --------------------------------------------------------------------------- #


def test_clean_run_passes() -> None:
    expected, consumed = _run("none")
    result = verify_coverage(expected, consumed)
    assert result.passed
    assert result.failure_summary() == "coverage invariant holds"
    assert result.consumed_count == result.expected_count == len(IDS)
    assert result.per_rank_counts == [len(IDS) // WORLD_SIZE] * WORLD_SIZE


# --------------------------------------------------------------------------- #
# Every defect must be caught
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("defect", [d for d in SAMPLER_DEFECTS if d != "none"])
def test_every_injected_defect_is_detected(defect: str) -> None:
    expected, consumed = _run(defect)
    result = verify_coverage(expected, consumed)
    assert not result.passed, f"{defect} went undetected"
    assert "VIOLATED" in result.failure_summary()


def test_duplicate_within_rank_names_the_id() -> None:
    expected, consumed = _run("duplicate_within_rank")
    result = verify_coverage(expected, consumed)
    assert result.duplicated_total == WORLD_SIZE
    for rank_ids in consumed:
        assert rank_ids[0] in result.duplicated_ids


def test_duplicate_across_ranks_is_caught_by_disjointness() -> None:
    """Counts alone look right here; only pairwise disjointness catches it."""
    expected, consumed = _run("duplicate_across_ranks")
    result = verify_coverage(expected, consumed)
    assert result.rank_overlaps, "overlap between ranks not reported"
    overlap = result.rank_overlaps[0]
    assert {overlap.rank_a, overlap.rank_b} == {0, 1}
    assert expected[0] in overlap.shared_ids


def test_dropped_sample_is_reported_as_missing() -> None:
    expected, consumed = _run("drop_sample")
    result = verify_coverage(expected, consumed)
    assert result.missing_total == 1
    assert result.consumed_count == result.expected_count - 1
    dropped = _sampler(0).rank_epoch_ids(0)[-1]
    assert result.missing_ids == [dropped]


def test_full_dataset_per_rank_is_caught() -> None:
    """The dangerous one: every rank reads everything and nothing crashes."""
    expected, consumed = _run("full_dataset_per_rank")
    result = verify_coverage(expected, consumed)
    assert not result.passed
    assert result.consumed_count == len(IDS) * WORLD_SIZE
    assert result.duplicated_total == len(IDS)
    assert len(result.rank_overlaps) == WORLD_SIZE * (WORLD_SIZE - 1) // 2


def test_unexpected_id_is_reported() -> None:
    expected, consumed = _run("none")
    consumed[2] = [*consumed[2], "not-a-real-sample"]
    result = verify_coverage(expected, consumed)
    assert result.unexpected_ids == ["not-a-real-sample"]
    assert not result.passed


def test_unknown_defect_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown defect"):
        inject_sampler_defect(["a"], "explode", rank=0)


# --------------------------------------------------------------------------- #
# Reporting limits
# --------------------------------------------------------------------------- #


def test_offending_id_lists_are_capped_but_totals_stay_exact() -> None:
    """A badly broken run must not be able to write an enormous artifact."""
    big = [f"x{i:05d}" for i in range(MAX_REPORTED_IDS * 3)]
    result = verify_coverage(big, [big, big])
    assert result.duplicated_total == len(big)
    assert len(result.duplicated_ids) == MAX_REPORTED_IDS
    assert result.truncated_id_lists is True


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #


def test_artifact_matches_the_declared_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "coverage.schema.json").read_text()
    )
    expected, consumed = _run("none")
    result = verify_coverage(expected, consumed)
    out = tmp_path / "coverage.json"
    write_coverage_artifact(out, result, sampler={"seed": 17, "shuffle": True, "drop_last": True})
    payload = json.loads(out.read_text())
    jsonschema.validate(payload, schema)
    assert payload["passed"] is True
    assert payload["world_size"] == WORLD_SIZE


def test_failing_artifact_also_validates(tmp_path: Path) -> None:
    """A failure artifact must still be machine-checkable, not free-form text."""
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "coverage.schema.json").read_text()
    )
    expected, consumed = _run("full_dataset_per_rank")
    result = verify_coverage(expected, consumed)
    out = tmp_path / "coverage-bad.json"
    write_coverage_artifact(out, result, sampler={"seed": 17, "shuffle": True, "drop_last": True})
    payload = json.loads(out.read_text())
    jsonschema.validate(payload, schema)
    assert payload["passed"] is False
    assert payload["rank_overlaps"]
