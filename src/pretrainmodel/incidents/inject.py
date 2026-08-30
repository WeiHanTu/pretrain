"""Deliberate defect injection.

Every function here breaks something on purpose so that a detector can be shown
to catch it.  Two rules govern their use.

First, an injected failure is an *experiment*, never an organic incident.  Reports
generated from these must set ``injected: true`` (schemas/incident.schema.json) and
must not be narrated as a production outage that happened to the author.

Second, a detector is only credible if it was declared before the injection.  The
value of "I killed rank 2 and my checkpoint logic recovered" is entirely in having
written down what the signal would be first; otherwise the analysis is fitted to
the failure after seeing it.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "SAMPLER_DEFECTS",
    "corrupt_file_bytes",
    "inject_sampler_defect",
    "truncate_file",
]

SAMPLER_DEFECTS = (
    "none",
    "duplicate_within_rank",
    "duplicate_across_ranks",
    "drop_sample",
    "full_dataset_per_rank",
)


def inject_sampler_defect(
    rank_ids: Sequence[str],
    defect: str,
    *,
    rank: int,
    global_order: Sequence[str] | None = None,
) -> list[str]:
    """Return a rank's sample list with a deliberate defect applied.

    ``duplicate_within_rank``
        One rank yields its first sample twice.  Total consumed count still looks
        plausible, which is exactly why a count-only check is insufficient.

    ``duplicate_across_ranks``
        Rank 1 also takes rank 0's first sample.  Detected only by the pairwise
        disjointness half of the invariant.

    ``drop_sample``
        One rank silently skips its last sample -- the shape of an off-by-one in a
        sharding calculation.

    ``full_dataset_per_rank``
        Every rank iterates the entire epoch.  This is the highest-value case: the
        loss curve looks healthy, throughput looks healthy, and an "epoch" quietly
        means ``world_size`` passes over the data.
    """
    if defect not in SAMPLER_DEFECTS:
        raise ValueError(f"unknown defect {defect!r}; expected one of {SAMPLER_DEFECTS}")

    ids = list(rank_ids)
    if defect == "none":
        return ids
    if defect == "duplicate_within_rank":
        return [ids[0], *ids] if ids else ids
    if defect == "duplicate_across_ranks":
        if rank == 1 and global_order:
            return [global_order[0], *ids]
        return ids
    if defect == "drop_sample":
        return ids[:-1] if rank == 0 and ids else ids
    if defect == "full_dataset_per_rank":
        return list(global_order) if global_order is not None else ids
    raise AssertionError(f"unhandled defect {defect!r}")  # pragma: no cover


def corrupt_file_bytes(path: Path, *, offset: int = 0, count: int = 8) -> bytes:
    """Flip bytes in place, leaving the file the same length.

    Length-preserving on purpose: a size check alone would catch truncation, so the
    corruption gate has to rest on the content digest.  Returns the original bytes
    so a test can restore the file.

    The offset is clamped into the file, and a zero-length file is rejected rather
    than silently doing nothing -- a no-op "corruption" would make the detector
    look like it passed when nothing was ever broken.
    """
    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"cannot corrupt an empty file: {path}")
    offset = max(0, min(offset, size - 1))
    count = max(1, min(count, size - offset))

    with path.open("r+b") as fh:
        fh.seek(offset)
        original = fh.read(count)
        fh.seek(offset)
        fh.write(bytes((b ^ 0xFF) for b in original))
        fh.flush()
        os.fsync(fh.fileno())
    return original


def restore_file_bytes(path: Path, original: bytes, *, offset: int = 0) -> None:
    """Undo :func:`corrupt_file_bytes`."""
    with path.open("r+b") as fh:
        fh.seek(offset)
        fh.write(original)
        fh.flush()
        os.fsync(fh.fileno())


def truncate_file(path: Path, *, keep_bytes: int) -> None:
    """Truncate a file, simulating a partially written shard or checkpoint."""
    with path.open("r+b") as fh:
        fh.truncate(max(0, keep_bytes))
        fh.flush()
        os.fsync(fh.fileno())
