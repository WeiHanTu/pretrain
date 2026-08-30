#!/usr/bin/env python
"""Run the resume oracles and write their evidence (plan.md B3, B4).

    uv run python scripts/run_resume_oracle.py

Produces:
    artifacts/oracles/exact-resume.json      B3: bit-exact same-world-size resume
    artifacts/checkpoints/reshard-matrix.json B4: DCP load across world sizes

B3 is run with two negative controls, because a gate only ever shown passing is
not a gate.

B4 deliberately stops short of a loss comparison; see NOTE_GLOBAL_BATCH below.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from pretrainmodel.data.coverage import verify_coverage
from pretrainmodel.training.compare import OracleNotApplicableError, compare_exact
from pretrainmodel.training.resume_experiment import run_resume_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/deterministic_resume.toml"
DATA = REPO_ROOT / "data/processed/synthetic/v1"
MAX_STEPS = 20
CHECKPOINT_AT = 10

NOTE_GLOBAL_BATCH = (
    "Loss trajectories are NOT compared across a world-size change here. Resharding "
    "N -> M with fixed grad_accum_steps changes the global batch size (world_size x "
    "micro_batch x grad_accum), which changes the optimisation trajectory for real "
    "reasons. Judging such a run against a seed-variance band built at a different "
    "global batch would conflate the reshard with the batch-size change and could "
    "'pass' or 'fail' for reasons unrelated to checkpoint correctness. A like-for-like "
    "loss comparison requires holding the global batch constant across the reshard by "
    "compensating grad_accum_steps, which is itself a config change and therefore a "
    "separate experiment. What is verified here is the mechanical claim: state loads "
    "at a different world size with genuinely different shard shapes, and the epoch "
    "continues to be consumed exactly once."
)


def _flat(ids_by_rank: list[list[str]]) -> list[str]:
    return [i for rank_ids in ids_by_rank for i in rank_ids]


def run_b3() -> dict[str, Any]:
    """Bit-exact same-world-size resume, plus negative controls."""
    cases: list[dict[str, Any]] = []

    for defect, expectation in [
        ("none", "pass"),
        ("reset_optimizer", "fail"),
        ("ignore_loader_state", "fail"),
    ]:
        with tempfile.TemporaryDirectory() as td:
            out = run_resume_experiment(
                config_path=CONFIG,
                data_root=DATA,
                workdir=Path(td),
                control_world_size=2,
                resume_world_size=2,
                max_steps=MAX_STEPS,
                checkpoint_at=CHECKPOINT_AT,
                resume_defect=defect,
            )
        control, resumed = out["control"], out["resume"]
        half = CHECKPOINT_AT
        comparison = compare_exact(
            control_state=control["snapshot"],
            resumed_state=resumed["snapshot"],
            control_losses=control["losses_by_rank"][0],
            resumed_losses=resumed["losses_by_rank"][0],
            control_sample_ids=_flat([r[half * 4 :] for r in control["ids_by_rank"]]),
            resumed_sample_ids=_flat(resumed["ids_by_rank"]),
            boundary_step=CHECKPOINT_AT,
            rng_continuity=resumed["rng_continuity"],
            resharded=resumed["resharded"],
        )
        actual = "pass" if comparison.passed else "fail"
        cases.append(
            {
                "resume_defect": defect,
                "expected": expectation,
                "actual": actual,
                "as_expected": actual == expectation,
                "summary": comparison.summary(),
                **comparison.to_dict(),
            }
        )
        print(
            f"  [{'OK' if actual == expectation else 'UNEXPECTED'}] "
            f"defect={defect:22s} expected={expectation:4s} actual={actual}"
        )

    return {
        "schema_version": 1,
        "oracle": "exact_equality",
        "spec": "8.1",
        "world_size": 2,
        "backend": "gloo",
        "device": "cpu",
        "fsdp2": True,
        "max_steps": MAX_STEPS,
        "checkpoint_at": CHECKPOINT_AT,
        "constraints": {
            "dtype": "fp32",
            "deterministic_algorithms": True,
            "async_checkpoint": False,
            "shuffle": False,
            "dropout": 0.0,
        },
        "passed": all(c["as_expected"] for c in cases),
        "cases": cases,
        "limitations": (
            "One host, CPU, Gloo, 2 ranks: NOT multi-node evidence. Bit-exactness is "
            "claimed only under the constraints listed above; it is not claimed for "
            "bf16, asynchronous checkpointing, or any change of world size."
        ),
    }


def run_b4() -> dict[str, Any]:
    """DCP load across a world-size change."""
    cases: list[dict[str, Any]] = []
    for old_ws, new_ws in [(1, 2), (2, 1), (2, 4), (4, 2)]:
        with tempfile.TemporaryDirectory() as td:
            out = run_resume_experiment(
                config_path=CONFIG,
                data_root=DATA,
                workdir=Path(td),
                control_world_size=old_ws,
                resume_world_size=new_ws,
                max_steps=MAX_STEPS,
                checkpoint_at=CHECKPOINT_AT,
            )
        control, resumed = out["control"], out["resume"]

        # The exact oracle must refuse: a reshard cannot be bit-exact in principle.
        try:
            compare_exact(
                control_state=control["snapshot"],
                resumed_state=resumed["snapshot"],
                control_losses=control["losses_by_rank"][0],
                resumed_losses=resumed["losses_by_rank"][0],
                control_sample_ids=[],
                resumed_sample_ids=[],
                boundary_step=CHECKPOINT_AT,
                rng_continuity=resumed["rng_continuity"],
                resharded=resumed["resharded"],
            )
            refused = False
            refusal = "ORACLE DID NOT REFUSE -- bit-exactness would have been claimed"
        except OracleNotApplicableError as exc:
            refused = True
            refusal = str(exc)

        # After the reshard the epoch must still be consumed exactly once.
        #
        # The comparison is against the PREFIX of the epoch order that was actually
        # consumed, not the whole epoch: these runs stop at max_steps well before the
        # epoch is exhausted, so checking against the full epoch would always report
        # the untouched tail as "missing". The prefix is well defined across the
        # reshard precisely because the epoch permutation does not depend on world
        # size -- that property is what makes this check meaningful at all.
        consumed_before = [r[: CHECKPOINT_AT * 4] for r in control["ids_by_rank"]]
        consumed_all = [*consumed_before, *resumed["ids_by_rank"]]
        total_consumed = sum(len(r) for r in consumed_all)
        coverage = verify_coverage(control["expected_epoch_ids"][:total_consumed], consumed_all)

        case = {
            "from_world_size": old_ws,
            "to_world_size": new_ws,
            "dcp_load_succeeded": True,
            "rng_continuity": resumed["rng_continuity"],
            "resharded": resumed["resharded"],
            "exact_oracle_refused": refused,
            "refusal_reason": refusal,
            "load_note": resumed["load_note"],
            "shard_local_fraction_before": control["sharding"]["local_fraction"],
            "shard_local_fraction_after": resumed["sharding"]["local_fraction"],
            "shard_example_before": control["sharding"]["examples"][0],
            "shard_example_after": resumed["sharding"]["examples"][0],
            "coverage_after_resume_passed": coverage.passed,
            "coverage_summary": coverage.failure_summary(),
            "samples_consumed_total": total_consumed,
            "expected_epoch_size_before": len(control["expected_epoch_ids"]),
            "expected_epoch_size_after": len(resumed["expected_epoch_ids"]),
            "global_batch_before": old_ws * 4,
            "global_batch_after": new_ws * 4,
        }
        cases.append(case)
        print(
            f"  [{'OK' if refused else 'BROKEN'}] {old_ws} -> {new_ws}: "
            f"refused={refused} coverage={coverage.passed} "
            f"local_frac {control['sharding']['local_fraction']:.3f}"
            f" -> {resumed['sharding']['local_fraction']:.3f}"
        )
    return {
        "schema_version": 1,
        "experiment": "dcp_world_size_change",
        "spec": "8.3",
        "backend": "gloo",
        "device": "cpu",
        "fsdp2": True,
        "cases": cases,
        "all_refused_exact_oracle": all(c["exact_oracle_refused"] for c in cases),
        "note_loss_comparison": NOTE_GLOBAL_BATCH,
        "limitations": (
            "One host, CPU, Gloo: NOT multi-node evidence. Proves DCP state loads and "
            "reshards across world sizes and that the epoch is still consumed exactly "
            "once; does not compare loss trajectories (see note_loss_comparison)."
        ),
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["b3", "b4", "all"], default="all")
    args = parser.parse_args()

    ok = True
    if args.only in ("b3", "all"):
        print("B3: bit-exact same-world-size resume")
        b3 = run_b3()
        (REPO_ROOT / "artifacts/oracles/exact-resume.json").write_text(
            json.dumps(b3, indent=2, sort_keys=True) + "\n"
        )
        ok = ok and b3["passed"]
        print(f"B3 result: {'PASS' if b3['passed'] else 'FAIL'}")

    if args.only in ("b4", "all"):
        print("\nB4: DCP reshard across world sizes")
        b4 = run_b4()
        path = REPO_ROOT / "artifacts/checkpoints/reshard-matrix.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(b4, indent=2, sort_keys=True) + "\n")
        b4_ok = b4["all_refused_exact_oracle"] and all(
            c["coverage_after_resume_passed"] for c in b4["cases"]
        )
        ok = ok and b4_ok
        print(f"B4 result: {'PASS' if b4_ok else 'FAIL'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
