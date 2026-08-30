"""Exact-equality oracle for same-world-size resume (spec 8.1).

What "bit-exact" means here: after resuming from a checkpoint taken at step *k*,
the parameters, optimizer state, loss sequence and consumed sample IDs from step
*k* onward are **identical** to an uninterrupted control run -- not close, equal.

This is a deliberately brittle gate and that is its value.  Every commonly missed
piece of resume state breaks it loudly:

- forgetting AdamW's per-parameter ``step`` shifts bias correction
- restarting the LR warmup instead of reconstructing it from the global step
- failing to restore RNG state (with dropout enabled)
- resuming the dataloader at the wrong cursor

Each of those produces a *plausible* loss curve, so a tolerance-based comparison
would pass all of them.  Exact equality does not.

The gate refuses to run when RNG continuity was broken -- a resharded checkpoint
cannot be bit-exact even in principle, so certifying it would be a category error
rather than a tight call. That path must use the seed-variance band instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

__all__ = [
    "Difference",
    "ExactComparison",
    "OracleNotApplicableError",
    "compare_exact",
    "compare_tensor_maps",
    "full_snapshot",
]


class OracleNotApplicableError(RuntimeError):
    """The exact oracle cannot be applied to this pair of runs."""


@dataclass(frozen=True, slots=True)
class Difference:
    """One mismatch, described well enough to act on."""

    key: str
    kind: str
    detail: str
    max_abs_diff: float | None = None
    first_index: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "detail": self.detail,
            "max_abs_diff": self.max_abs_diff,
            "first_index": self.first_index,
        }


def full_snapshot(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, Tensor]]:
    """Unsharded CPU snapshot of model and optimizer state, materialised on rank 0.

    Sharded (DTensor) state cannot be compared rank-by-rank across runs at
    different world sizes, because the shards do not correspond. Gathering the full
    tensor first makes the comparison well defined regardless of topology.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

    model_sd, optim_sd = get_state_dict(
        model,
        optimizer,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    flat_optim: dict[str, Tensor] = {}
    _flatten(optim_sd, "", flat_optim)
    flat_model: dict[str, Tensor] = {}
    _flatten(model_sd, "", flat_model)
    return {"model": flat_model, "optim": flat_optim}


def _flatten(obj: Any, prefix: str, out: dict[str, Tensor]) -> None:
    """Flatten nested optimizer state into ``name -> tensor``."""
    if isinstance(obj, Tensor):
        out[prefix] = obj.detach().to("cpu")
        return
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(v, f"{prefix}[{i}]", out)
        return
    if isinstance(obj, (int, float, bool)):
        out[prefix] = torch.tensor(obj)


def compare_tensor_maps(
    control: Mapping[str, Tensor], candidate: Mapping[str, Tensor], *, label: str
) -> list[Difference]:
    """Exact elementwise comparison of two flat tensor maps."""
    differences: list[Difference] = []

    missing = sorted(set(control) - set(candidate))
    extra = sorted(set(candidate) - set(control))
    for key in missing:
        differences.append(Difference(f"{label}.{key}", "missing", "absent from resumed run"))
    for key in extra:
        differences.append(Difference(f"{label}.{key}", "unexpected", "absent from control run"))

    for key in sorted(set(control) & set(candidate)):
        a, b = control[key], candidate[key]
        if a.shape != b.shape:
            differences.append(
                Difference(
                    f"{label}.{key}", "shape", f"control {tuple(a.shape)} vs {tuple(b.shape)}"
                )
            )
            continue
        if a.dtype != b.dtype:
            differences.append(
                Difference(f"{label}.{key}", "dtype", f"control {a.dtype} vs {b.dtype}")
            )
            continue
        if torch.equal(a, b):
            continue

        af, bf = a.float(), b.float()
        diff = (af - bf).abs()
        idx = int(torch.argmax(diff.flatten()).item())
        first = np_unravel(idx, tuple(a.shape))
        differences.append(
            Difference(
                key=f"{label}.{key}",
                kind="value",
                detail=(
                    f"{int((af != bf).sum())} of {a.numel()} elements differ; "
                    f"control={af.flatten()[idx].item():.17g} "
                    f"resumed={bf.flatten()[idx].item():.17g}"
                ),
                max_abs_diff=float(diff.max()),
                first_index=first,
            )
        )
    return differences


def np_unravel(index: int, shape: tuple[int, ...]) -> list[int]:
    """Convert a flat index to a multi-dimensional one."""
    out: list[int] = []
    for dim in reversed(shape):
        out.append(index % dim)
        index //= dim
    return list(reversed(out))


@dataclass
class ExactComparison:
    """Result of the bit-exactness gate."""

    passed: bool
    boundary_step: int
    compared_tensors: int
    differences: list[Difference] = field(default_factory=list)
    loss_match: bool = True
    sample_id_match: bool = True
    control_losses: list[list[float]] = field(default_factory=list)
    resumed_losses: list[list[float]] = field(default_factory=list)
    compared_loss_values: int = 0
    compared_sample_ids: int = 0
    note: str = ""

    def summary(self) -> str:
        if self.passed:
            return (
                f"bit-exact: {self.compared_tensors} tensors, "
                f"{self.compared_loss_values} loss values and "
                f"{self.compared_sample_ids} sample IDs identical "
                f"from step {self.boundary_step}"
            )
        lines = [f"NOT bit-exact at the step-{self.boundary_step} resume boundary:"]
        if not self.sample_id_match:
            lines.append("  consumed sample IDs diverge -> dataloader cursor not restored")
        if not self.loss_match:
            lines.append("  loss sequence diverges after the boundary")
        for d in self.differences[:10]:
            lines.append(f"  {d.key} [{d.kind}] {d.detail}")
        if len(self.differences) > 10:
            lines.append(f"  ... and {len(self.differences) - 10} more")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle": "exact_equality",
            "passed": self.passed,
            "boundary_step": self.boundary_step,
            "compared_tensors": self.compared_tensors,
            "compared_loss_values": self.compared_loss_values,
            "compared_sample_ids": self.compared_sample_ids,
            "loss_match": self.loss_match,
            "sample_id_match": self.sample_id_match,
            "num_differences": len(self.differences),
            "differences": [d.to_dict() for d in self.differences[:32]],
            "control_losses": self.control_losses,
            "resumed_losses": self.resumed_losses,
            "note": self.note,
        }


def compare_exact(
    *,
    control_state: Mapping[str, Mapping[str, Tensor]],
    resumed_state: Mapping[str, Mapping[str, Tensor]],
    control_losses: Sequence[Sequence[float]],
    resumed_losses: Sequence[Sequence[float]],
    control_sample_ids: Sequence[str],
    resumed_sample_ids: Sequence[str],
    boundary_step: int,
    rng_continuity: bool = True,
    resharded: bool = False,
) -> ExactComparison:
    """Apply the bit-exactness gate.

    Raises ``OracleNotApplicableError`` when the runs cannot be bit-exact in
    principle. That is not a failure of the model -- it is a statement that the
    wrong oracle was chosen, and the code refuses rather than reporting a
    misleading pass or fail.
    """
    if resharded or not rng_continuity:
        raise OracleNotApplicableError(
            "the exact-equality oracle does not apply to a run that changed world size "
            "or lost RNG continuity: reduction order and per-rank RNG both change, so "
            "bit-exactness is impossible by construction. Use the seed-variance band "
            "(spec 8.2) for this comparison."
        )

    differences = compare_tensor_maps(control_state["model"], resumed_state["model"], label="model")
    differences += compare_tensor_maps(
        control_state["optim"], resumed_state["optim"], label="optim"
    )

    # Compare every rank, not just rank 0. Each rank computes its loss on its own
    # micro-batch, so a resume fault that lands on one rank only would be invisible
    # if a single rank were treated as representative.
    tail_control = [list(r[boundary_step:]) for r in control_losses]
    tail_resumed = [list(r) for r in resumed_losses]
    loss_match = tail_control == tail_resumed
    compared_loss_values = sum(len(r) for r in tail_control)

    ids_control = list(control_sample_ids)
    ids_resumed = list(resumed_sample_ids)
    sample_id_match = ids_control == ids_resumed

    compared = len(control_state["model"]) + len(control_state["optim"])
    if compared == 0 or compared_loss_values == 0 or not ids_control:
        raise OracleNotApplicableError(
            "refusing to certify a vacuous comparison: "
            f"{compared} tensors, {compared_loss_values} loss values, "
            f"{len(ids_control)} sample ids. An empty comparison trivially 'passes' "
            "and would be the easiest way to fake this gate."
        )
    return ExactComparison(
        passed=not differences and loss_match and sample_id_match,
        boundary_step=boundary_step,
        compared_tensors=compared,
        differences=differences,
        loss_match=loss_match,
        sample_id_match=sample_id_match,
        control_losses=tail_control,
        resumed_losses=tail_resumed,
        compared_loss_values=compared_loss_values,
        compared_sample_ids=len(ids_control),
        note=(
            "Exact equality under the declared constraints: fp32, deterministic "
            "algorithms, synchronous checkpointing, fixed data order, same world size."
        ),
    )
