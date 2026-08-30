"""Every committed evidence artifact must validate against its declared schema.

A schema nobody validates against is documentation, not a gate.  This test makes
the schemas load-bearing: an incident report that omits its counterfactual or its
limitations fails CI rather than shipping incomplete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"
COVERAGE_DIR = REPO_ROOT / "artifacts" / "coverage"
INCIDENT_DIR = REPO_ROOT / "reports" / "incidents"


def _schema(name: str) -> dict[str, object]:
    data: dict[str, object] = json.loads((SCHEMA_DIR / name).read_text())
    return data


@pytest.mark.parametrize("path", sorted(SCHEMA_DIR.glob("*.json")), ids=lambda p: p.name)
def test_schemas_are_themselves_valid(path: Path) -> None:
    jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))


@pytest.mark.parametrize("path", sorted(COVERAGE_DIR.glob("*.json")), ids=lambda p: p.name)
def test_committed_coverage_artifacts_validate(path: Path) -> None:
    jsonschema.validate(json.loads(path.read_text()), _schema("coverage.schema.json"))


@pytest.mark.parametrize("path", sorted(INCIDENT_DIR.glob("*.json")), ids=lambda p: p.name)
def test_committed_incident_reports_validate(path: Path) -> None:
    jsonschema.validate(json.loads(path.read_text()), _schema("incident.schema.json"))


@pytest.mark.parametrize("path", sorted(INCIDENT_DIR.glob("*.json")), ids=lambda p: p.name)
def test_every_incident_json_has_a_markdown_report(path: Path) -> None:
    assert path.with_suffix(".md").is_file(), f"{path.name} has no narrative report"


@pytest.mark.parametrize("path", sorted(INCIDENT_DIR.glob("*.json")), ids=lambda p: p.name)
def test_injected_incidents_are_labelled_as_experiments(path: Path) -> None:
    """An induced failure must never be presented as an organic incident."""
    payload = json.loads(path.read_text())
    if payload["injected"]:
        text = path.with_suffix(".md").read_text().lower()
        assert "injected experiment" in text, (
            f"{path.name} is injected but its report does not say so up front"
        )


def test_clean_coverage_artifacts_report_pass() -> None:
    for ws in (1, 2, 4):
        payload = json.loads((COVERAGE_DIR / f"world-size-{ws}.json").read_text())
        assert payload["passed"] is True
        assert payload["world_size"] == ws


def test_negative_control_artifact_reports_failure() -> None:
    """The evidence set must contain proof that the detector can fail."""
    path = COVERAGE_DIR / "negative-control-full-dataset-per-rank.json"
    payload = json.loads(path.read_text())
    assert payload["passed"] is False
    assert payload["consumed_count"] > payload["expected_count"]
    assert payload["rank_overlaps"]


def test_no_artifact_claims_multi_node_evidence() -> None:
    """Nothing captured locally may imply a physical network boundary was crossed."""
    for path in COVERAGE_DIR.glob("*.json"):
        payload = json.loads(path.read_text())
        notes = str(payload.get("notes", ""))
        assert "distinct host(s)" in notes, f"{path.name} does not record host count"
        assert "NOT multi-node evidence" in notes, f"{path.name} lacks its scope disclaimer"


# --------------------------------------------------------------------------- #
# Seed-variance band
# --------------------------------------------------------------------------- #

BAND_PATH = REPO_ROOT / "artifacts" / "oracles" / "seed-band.json"


@pytest.mark.skipif(not BAND_PATH.exists(), reason="seed band not yet generated")
def test_committed_seed_band_is_usable() -> None:
    """The frozen band must load and must still contain its own control runs."""
    from pretrainmodel.training.oracle import SeedBand, evaluate_against_band

    band = SeedBand.from_dict(json.loads(BAND_PATH.read_text()))
    assert len(band.seeds) >= 3, "a band needs at least 3 control seeds"
    assert band.num_steps > 0
    for seed, losses in band.control_losses.items():
        result = evaluate_against_band(band, losses, config_hash=band.config_hash)
        assert result.passed, f"control seed {seed} falls outside its own band"


@pytest.mark.skipif(not BAND_PATH.exists(), reason="seed band not yet generated")
def test_committed_seed_band_discloses_its_rule_and_limits() -> None:
    """The pass rule and the small-sample caveat must be in the artifact itself."""
    payload = json.loads(BAND_PATH.read_text())
    assert payload["oracle"] == "seed_variance_band"
    assert payload["pass_rule"]
    assert payload["interval_rule"]
    assert "(n-1)/(n+1)" in payload["limitations"]
    assert "never bit-exactness" in payload["limitations"]
    assert len(payload["config_hash"]) == 64


@pytest.mark.skipif(not BAND_PATH.exists(), reason="seed band not yet generated")
def test_seed_band_control_runs_are_traceable() -> None:
    """plan.md A4 requires *traceable* control runs.

    Run manifests live under runs/, which is gitignored, so a copy of each control
    run's manifest is committed beside the band. Without this a reviewer cloning the
    repo cannot resolve the run IDs the band cites.
    """
    import re

    payload = json.loads(BAND_PATH.read_text())
    cited = re.findall(r"[A-Za-z0-9_]+-\d{8}T\d{6}Z-[0-9a-f]{8}", payload.get("notes", ""))
    assert len(cited) == len(payload["seeds"]), (
        f"band cites {len(cited)} run ids for {len(payload['seeds'])} seeds"
    )
    control_dir = BAND_PATH.parent / "control-runs"
    for run_id in cited:
        manifest = control_dir / f"{run_id}.json"
        assert manifest.is_file(), f"control run {run_id} is cited but not committed"
        data = json.loads(manifest.read_text())
        assert data["config_hash"]
        assert data["formal"] is True, "a control run backing an oracle must be formal"
