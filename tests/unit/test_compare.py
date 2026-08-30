"""The exact-equality comparator and its refusal condition."""

from __future__ import annotations

import pytest
import torch

from pretrainmodel.training.compare import (
    OracleNotApplicableError,
    compare_exact,
    compare_tensor_maps,
    np_unravel,
)


def _state(scale: float = 1.0) -> dict[str, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    w = torch.arange(12, dtype=torch.float32).reshape(3, 4) * scale
    return {"model": {"w": w}, "optim": {"state.0.exp_avg": w * 0.1}}


# --------------------------------------------------------------------------- #
# Tensor comparison
# --------------------------------------------------------------------------- #


def test_identical_maps_have_no_differences() -> None:
    assert compare_tensor_maps(_state()["model"], _state()["model"], label="model") == []


def test_single_bit_difference_is_caught() -> None:
    """The gate must be sensitive to the smallest representable change."""
    a = _state()["model"]
    b = {"w": a["w"].clone()}
    b["w"][1, 2] = torch.nextafter(b["w"][1, 2], torch.tensor(1e30))
    diffs = compare_tensor_maps(a, b, label="model")
    assert len(diffs) == 1
    assert diffs[0].kind == "value"
    assert diffs[0].first_index == [1, 2]
    assert diffs[0].max_abs_diff is not None and diffs[0].max_abs_diff > 0


def test_shape_and_dtype_mismatches_are_distinguished() -> None:
    a = {"w": torch.zeros(2, 2)}
    assert compare_tensor_maps(a, {"w": torch.zeros(3, 2)}, label="m")[0].kind == "shape"
    assert compare_tensor_maps(a, {"w": torch.zeros(2, 2).double()}, label="m")[0].kind == "dtype"


def test_missing_and_extra_keys_are_reported() -> None:
    a = {"w": torch.zeros(2), "b": torch.zeros(2)}
    diffs = compare_tensor_maps(a, {"w": torch.zeros(2), "c": torch.zeros(2)}, label="m")
    kinds = {d.kind for d in diffs}
    assert kinds == {"missing", "unexpected"}


def test_np_unravel_matches_row_major_order() -> None:
    assert np_unravel(0, (3, 4)) == [0, 0]
    assert np_unravel(6, (3, 4)) == [1, 2]
    assert np_unravel(11, (3, 4)) == [2, 3]


# --------------------------------------------------------------------------- #
# The oracle
# --------------------------------------------------------------------------- #


def _compare(**over: object) -> object:
    kwargs: dict[str, object] = {
        "control_state": _state(),
        "resumed_state": _state(),
        "control_losses": [[1.0, 0.9, 0.8, 0.7], [1.1, 0.95, 0.85, 0.75]],
        "resumed_losses": [[0.8, 0.7], [0.85, 0.75]],
        "control_sample_ids": ["c", "d"],
        "resumed_sample_ids": ["c", "d"],
        "boundary_step": 2,
    }
    kwargs.update(over)
    return compare_exact(**kwargs)  # type: ignore[arg-type]


def test_matching_runs_pass() -> None:
    result = _compare()
    assert result.passed  # type: ignore[attr-defined]


def test_diverging_loss_fails() -> None:
    result = _compare(resumed_losses=[[0.8, 0.71], [0.85, 0.75]])
    assert not result.passed  # type: ignore[attr-defined]
    assert not result.loss_match  # type: ignore[attr-defined]


def test_diverging_sample_ids_fails() -> None:
    """A wrong dataloader cursor is the defect this specifically isolates."""
    result = _compare(resumed_sample_ids=["a", "b"])
    assert not result.passed  # type: ignore[attr-defined]
    assert not result.sample_id_match  # type: ignore[attr-defined]


def test_diverging_weights_fails_and_names_the_tensor() -> None:
    result = _compare(resumed_state=_state(scale=1.001))
    assert not result.passed  # type: ignore[attr-defined]
    assert "model.w" in result.summary()  # type: ignore[attr-defined]


def test_oracle_refuses_a_resharded_run() -> None:
    """A world-size change cannot be bit-exact, so the gate must refuse, not fail.

    Reporting 'fail' would invite someone to loosen a tolerance until it passes.
    Refusing says the wrong oracle was chosen.
    """
    with pytest.raises(OracleNotApplicableError, match="does not apply"):
        _compare(resharded=True)


def test_oracle_refuses_when_rng_continuity_was_lost() -> None:
    with pytest.raises(OracleNotApplicableError, match="seed-variance band"):
        _compare(rng_continuity=False)


def test_a_fault_on_a_non_zero_rank_is_caught() -> None:
    """Rank 0 alone is not representative: each rank has its own micro-batch."""
    result = _compare(resumed_losses=[[0.8, 0.7], [0.85, 0.99]])
    assert not result.passed  # type: ignore[attr-defined]
    assert not result.loss_match  # type: ignore[attr-defined]


def test_vacuous_comparison_is_refused() -> None:
    """An empty comparison trivially passes; that is the easiest way to fake a gate."""
    with pytest.raises(OracleNotApplicableError, match="vacuous"):
        _compare(control_sample_ids=[], resumed_sample_ids=[])
    with pytest.raises(OracleNotApplicableError, match="vacuous"):
        _compare(control_losses=[], resumed_losses=[])
    with pytest.raises(OracleNotApplicableError, match="vacuous"):
        _compare(control_state={"model": {}, "optim": {}}, resumed_state={"model": {}, "optim": {}})


def test_summary_reports_what_was_actually_compared() -> None:
    """The pass message must state its own sample sizes, so 'passed' is auditable."""
    summary = _compare().summary()  # type: ignore[attr-defined]
    assert "tensors" in summary and "loss values" in summary and "sample IDs" in summary
