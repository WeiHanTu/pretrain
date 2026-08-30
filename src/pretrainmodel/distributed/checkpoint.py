"""Distributed checkpoint contract (spec 7).

Layout of one checkpoint::

    <dir>/step-00000010/
        .metadata, __0_0.distcp, ...   DCP payload: sharded model + optimizer
        rank-state/rank-0.pt           per-rank RNG state
        COMMITTED                      written LAST, by rank 0, after a barrier

Three decisions carry the weight here.

**The commit marker is the checkpoint.**  A DCP directory can exist and be
unreadable: a rank died mid-write, a disk filled, a spot VM was reclaimed.  So a
directory is a checkpoint only once ``COMMITTED`` exists, and discovery ignores
everything else.  Without this, "resume from latest" eventually selects a
half-written directory, and the resulting failure looks like model divergence
rather than a truncated write.

**RNG continuity cannot survive a reshard.**  Each rank's RNG state is saved
per rank.  Resuming at the same world size restores each rank's own state, and
bit-exactness is possible.  Resuming at a *different* world size has no state for
the new ranks -- rank 3 of 4 never existed when 2 ranks saved -- so the loader
deterministically reseeds from ``(seed, step, rank)`` and reports
``rng_continuity=False``.

That flag is not advisory.  The exact-equality oracle refuses to run against a
checkpoint that lost RNG continuity, which makes spec 8.1's bit-exact claim and
spec 8.2's statistical claim *mechanically* distinct rather than a matter of
discipline: the code cannot be talked into the stronger claim.

**Provenance is checked, not recorded.**  Resuming across a config or data change
is silent corruption of a training trajectory, so the hashes are compared on load
and a mismatch is an error rather than a log line.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn

from pretrainmodel.data.loader import LoaderState

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "COMMIT_MARKER",
    "CheckpointError",
    "CheckpointMetadata",
    "LoadedCheckpoint",
    "latest_committed",
    "list_committed",
    "load_checkpoint",
    "prune_checkpoints",
    "save_checkpoint",
]

CHECKPOINT_FORMAT_VERSION = 1
COMMIT_MARKER = "COMMITTED"
_RANK_STATE_DIR = "rank-state"


class CheckpointError(RuntimeError):
    """A checkpoint is missing, incomplete, or incompatible with this run."""


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _rng_seed(seed: int, step: int, rank: int) -> int:
    payload = f"reshard-reseed|{seed}|{step}|{rank}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Contents of the COMMITTED marker."""

    format_version: int
    step: int
    epoch: int
    world_size: int
    saved_at: str
    config_hash: str
    data_manifest_hash: str
    run_id: str
    loader_state: dict[str, Any]
    sharded: bool
    dcp_metadata_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "step": self.step,
            "epoch": self.epoch,
            "world_size": self.world_size,
            "saved_at": self.saved_at,
            "config_hash": self.config_hash,
            "data_manifest_hash": self.data_manifest_hash,
            "run_id": self.run_id,
            "loader_state": self.loader_state,
            "sharded": self.sharded,
            "dcp_metadata_sha256": self.dcp_metadata_sha256,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> CheckpointMetadata:
        return cls(
            format_version=raw["format_version"],
            step=raw["step"],
            epoch=raw["epoch"],
            world_size=raw["world_size"],
            saved_at=raw["saved_at"],
            config_hash=raw["config_hash"],
            data_manifest_hash=raw.get("data_manifest_hash", ""),
            run_id=raw.get("run_id", ""),
            loader_state=raw.get("loader_state", {}),
            sharded=raw.get("sharded", False),
            dcp_metadata_sha256=raw.get("dcp_metadata_sha256", ""),
        )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """What a resume recovered, and what it could not."""

    metadata: CheckpointMetadata
    loader_state: LoaderState
    step: int
    epoch: int
    rng_continuity: bool
    resharded: bool
    note: str = ""


def _save_rng(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def _load_rng(path: Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    root: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    loader_state: LoaderState,
    config_hash: str,
    data_manifest_hash: str = "",
    run_id: str = "",
) -> Path:
    """Write one checkpoint and commit it atomically.

    Returns the checkpoint directory. The directory is only a valid checkpoint
    after this function returns; a caller that crashes partway leaves an
    uncommitted directory that discovery will ignore.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    from pretrainmodel.distributed.parallel import is_sharded

    rank, world_size = _rank_world()
    step_dir = Path(root) / f"step-{step:08d}"

    if rank == 0:
        step_dir.mkdir(parents=True, exist_ok=True)
        (step_dir / _RANK_STATE_DIR).mkdir(parents=True, exist_ok=True)
    _barrier()

    model_sd, optim_sd = get_state_dict(model, optimizer)
    dcp.save(  # type: ignore[attr-defined]
        {"model": model_sd, "optim": optim_sd}, checkpoint_id=str(step_dir)
    )

    _save_rng(step_dir / _RANK_STATE_DIR / f"rank-{rank}.pt")

    # Every rank must have finished before the checkpoint may be called complete.
    _barrier()

    if rank == 0:
        dcp_meta = step_dir / ".metadata"
        metadata = CheckpointMetadata(
            format_version=CHECKPOINT_FORMAT_VERSION,
            step=step,
            epoch=epoch,
            world_size=world_size,
            saved_at=datetime.now(UTC).isoformat(timespec="seconds"),
            config_hash=config_hash,
            data_manifest_hash=data_manifest_hash,
            run_id=run_id,
            loader_state=loader_state.to_dict(),
            sharded=is_sharded(model),
            dcp_metadata_sha256=_sha256(dcp_meta) if dcp_meta.is_file() else "",
        )
        (step_dir / COMMIT_MARKER).write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
        )
    _barrier()
    return step_dir


def read_marker(step_dir: str | Path) -> CheckpointMetadata | None:
    """Return the metadata for a committed checkpoint, or None if not committed."""
    marker = Path(step_dir) / COMMIT_MARKER
    if not marker.is_file():
        return None
    try:
        return CheckpointMetadata.from_dict(json.loads(marker.read_text()))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def list_committed(root: str | Path) -> list[tuple[int, Path]]:
    """All committed checkpoints under ``root``, ascending by step.

    Uncommitted or unparseable directories are skipped rather than repaired: a
    half-written checkpoint is evidence of a failure worth investigating, not
    something to silently promote.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("step-"):
            continue
        meta = read_marker(child)
        if meta is not None:
            found.append((meta.step, child))
    return sorted(found)


def latest_committed(root: str | Path) -> Path | None:
    """The highest-step committed checkpoint, or None."""
    found = list_committed(root)
    return found[-1][1] if found else None


def load_checkpoint(
    step_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    seed: int,
    config_hash: str | None = None,
    data_manifest_hash: str | None = None,
    allow_config_change: bool = False,
) -> LoadedCheckpoint:
    """Restore a checkpoint into ``model`` and ``optimizer``."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    step_dir = Path(step_dir)
    rank, world_size = _rank_world()

    metadata = read_marker(step_dir)
    if metadata is None:
        raise CheckpointError(
            f"{step_dir} has no valid {COMMIT_MARKER} marker; it is not a complete "
            "checkpoint and must not be resumed from."
        )
    if metadata.format_version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint format version {metadata.format_version} != {CHECKPOINT_FORMAT_VERSION}"
        )

    problems: list[str] = []
    if config_hash is not None and metadata.config_hash != config_hash:
        problems.append(
            f"config hash {metadata.config_hash[:12]}... != current {config_hash[:12]}..."
        )
    if (
        data_manifest_hash is not None
        and metadata.data_manifest_hash
        and metadata.data_manifest_hash != data_manifest_hash
    ):
        problems.append(
            f"data manifest {metadata.data_manifest_hash[:12]}... != "
            f"current {data_manifest_hash[:12]}..."
        )
    if problems and not allow_config_change:
        raise CheckpointError(
            "refusing to resume across a changed run definition:\n  - "
            + "\n  - ".join(problems)
            + "\nPass allow_config_change=True only if the change is understood."
        )

    model_sd, optim_sd = get_state_dict(model, optimizer)
    state: dict[str, Any] = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=str(step_dir))  # type: ignore[attr-defined]
    set_state_dict(
        model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"]
    )

    resharded = metadata.world_size != world_size
    rng_path = step_dir / _RANK_STATE_DIR / f"rank-{rank}.pt"
    if not resharded and rng_path.is_file():
        _load_rng(rng_path)
        rng_continuity = True
        note = "RNG state restored per rank; bit-exact continuation is possible."
    else:
        reseed = _rng_seed(seed, metadata.step, rank)
        torch.manual_seed(reseed)
        np.random.seed(reseed % (2**32))
        random.seed(reseed)
        rng_continuity = False
        note = (
            f"world size changed {metadata.world_size} -> {world_size}; "
            f"no saved RNG state exists for rank {rank}. Reseeded deterministically "
            "from (seed, step, rank). Bit-exactness is NOT claimable; use the "
            "seed-variance band."
            if resharded
            else f"no saved RNG state for rank {rank}; reseeded deterministically."
        )

    return LoadedCheckpoint(
        metadata=metadata,
        loader_state=LoaderState.from_dict(metadata.loader_state),
        step=metadata.step,
        epoch=metadata.epoch,
        rng_continuity=rng_continuity,
        resharded=resharded,
        note=note,
    )


def prune_checkpoints(root: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the newest ``keep_last`` committed checkpoints.

    Only committed checkpoints are counted *and* only committed ones are deleted.
    An uncommitted directory is left alone: it is the forensic record of a failed
    save, and removing it would destroy the evidence for the incident it caused.
    """
    if keep_last < 1:
        return []
    rank, _ = _rank_world()
    removed: list[Path] = []
    if rank == 0:
        committed = list_committed(root)
        for _, path in committed[:-keep_last] if len(committed) > keep_last else []:
            shutil.rmtree(path)
            removed.append(path)
    _barrier()
    return removed
