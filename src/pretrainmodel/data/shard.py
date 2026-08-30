"""Immutable sample shards and the stable sample-identity scheme.

A shard is a contiguous block of one sensor group's time series, stored as a
compressed ``.npz`` alongside a SHA-256 recorded in the shard manifest.  Training
windows are *derived* from a shard rather than materialised into it, so a window
never appears twice on disk and the sample space is a pure function of the
manifest.

Sample identity (spec 4.2) is the load-bearing idea for the whole coverage
invariant.  An ID is derived only from dataset version, group, window start
timestamp and window shape.  It never depends on rank, worker id, shard filename
or filesystem enumeration order -- if it did, "every sample consumed exactly once"
would be unverifiable across a world-size change, because the two runs would not
agree on what the samples *are*.

Windows never cross a shard boundary.  A window that would span two shards is
dropped, which keeps every sample resident in exactly one file and removes an
entire class of ordering bug at the cost of a few windows per shard seam.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pretrainmodel.data.manifest import ShardEntry, ShardManifest, sha256_file

__all__ = [
    "SHARD_FILE_SUFFIX",
    "SampleRef",
    "ShardData",
    "ShardedIndex",
    "build_index",
    "load_shard",
    "sample_id",
    "write_shard",
    "write_synthetic_dataset",
]

SHARD_FILE_SUFFIX = ".npz"
SAMPLE_ID_BYTES = 8


def sample_id(
    dataset_version: str,
    group: str,
    start_timestamp: int,
    context_steps: int,
    horizon_steps: int,
) -> str:
    """Stable identity for one training window.

    Deliberately excludes rank, worker, shard path and enumeration order.  The
    digest is truncated to 16 hex characters: with fewer than 10^6 samples the
    collision probability is below 10^-7, and the index asserts uniqueness anyway.
    """
    payload = f"{dataset_version}|{group}|{start_timestamp}|{context_steps}|{horizon_steps}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[: SAMPLE_ID_BYTES * 2]


@dataclass(frozen=True, slots=True)
class SampleRef:
    """Where a sample lives, resolved from the manifest without touching the data."""

    sample_id: str
    shard_path: str
    offset: int
    start_timestamp: int
    group: str


@dataclass(frozen=True, slots=True)
class ShardData:
    """In-memory contents of one shard."""

    series: npt.NDArray[np.float32]  # (T, S, F)
    observed: npt.NDArray[np.bool_]  # (T, S)
    timestamps: npt.NDArray[np.int64]  # (T,)
    group: str

    def window(
        self, offset: int, seq_len: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        return (
            self.series[offset : offset + seq_len],
            self.observed[offset : offset + seq_len],
        )


def write_shard(
    path: Path,
    series: npt.NDArray[np.float32],
    observed: npt.NDArray[np.bool_],
    timestamps: npt.NDArray[np.int64],
    group: str,
) -> None:
    """Write one shard.  Shards are write-once; callers must not mutate them."""
    if series.ndim != 3:
        raise ValueError(f"series must be (T, S, F), got shape {series.shape}")
    if observed.shape != series.shape[:2]:
        raise ValueError(f"observed must be (T, S) = {series.shape[:2]}, got {observed.shape}")
    if timestamps.shape != (series.shape[0],):
        raise ValueError(f"timestamps must be (T,) = ({series.shape[0]},), got {timestamps.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        series=series.astype(np.float32, copy=False),
        observed=observed.astype(np.bool_, copy=False),
        timestamps=timestamps.astype(np.int64, copy=False),
        group=np.array(group),
    )


def load_shard(path: Path) -> ShardData:
    """Read one shard from disk."""
    with np.load(path, allow_pickle=False) as z:
        return ShardData(
            series=z["series"].astype(np.float32, copy=False),
            observed=z["observed"].astype(np.bool_, copy=False),
            timestamps=z["timestamps"].astype(np.int64, copy=False),
            group=str(z["group"]),
        )


@dataclass(frozen=True, slots=True)
class ShardedIndex:
    """The complete, canonically ordered sample space for a dataset version.

    Canonical order is (group, shard_index, offset) -- not filesystem order -- so
    two processes on different machines enumerate identical sample lists.  This is
    the universe against which the coverage invariant is checked.
    """

    dataset_version: str
    context_steps: int
    horizon_steps: int
    samples: list[SampleRef]

    @property
    def seq_len(self) -> int:
        return self.context_steps + self.horizon_steps

    @property
    def sample_ids(self) -> list[str]:
        return [s.sample_id for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def by_id(self) -> dict[str, SampleRef]:
        return {s.sample_id: s for s in self.samples}


def build_index(manifest: ShardManifest) -> ShardedIndex:
    """Enumerate every derivable training window in canonical order.

    Pure function of the manifest: it reads no shard bytes, so building the index
    is cheap and cannot be perturbed by a corrupted file.
    """
    seq_len = manifest.seq_len
    stride = manifest.stride
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    samples: list[SampleRef] = []
    for entry in sorted(manifest.shards, key=lambda e: (e.group, e.shard_index)):
        last_start = entry.num_timesteps - seq_len
        if last_start < 0:
            continue
        # Windows must lie entirely inside one shard; a partial tail is dropped.
        for offset in range(0, last_start + 1, stride):
            start_ts = entry.first_timestamp + offset * manifest.sampling_interval_seconds
            samples.append(
                SampleRef(
                    sample_id=sample_id(
                        manifest.dataset_version,
                        entry.group,
                        start_ts,
                        manifest.context_steps,
                        manifest.horizon_steps,
                    ),
                    shard_path=entry.path,
                    offset=offset,
                    start_timestamp=start_ts,
                    group=entry.group,
                )
            )

    seen = len({s.sample_id for s in samples})
    if seen != len(samples):
        raise ValueError(
            f"sample identity collision: {len(samples)} windows produced {seen} unique ids. "
            "Two windows share (version, group, start timestamp, shape)."
        )
    return ShardedIndex(
        dataset_version=manifest.dataset_version,
        context_steps=manifest.context_steps,
        horizon_steps=manifest.horizon_steps,
        samples=samples,
    )


def write_synthetic_dataset(
    root: Path,
    *,
    dataset_version: str = "synthetic-v1",
    num_shards: int = 4,
    timesteps_per_shard: int = 64,
    num_sensors: int = 32,
    num_features: int = 1,
    context_steps: int = 12,
    horizon_steps: int = 12,
    stride: int = 1,
    sampling_interval_seconds: int = 300,
    missing_rate: float = 0.05,
    seed: int = 0,
    groups: tuple[str, ...] = ("g0",),
) -> ShardManifest:
    """Generate a deterministic synthetic dataset for tests.

    Synthetic data exists to exercise the machinery -- coverage, checkpointing,
    corruption detection -- and must never appear in a model-quality claim
    (CLAUDE.md). The signal is a smooth diurnal pattern plus per-sensor phase and
    noise, which is enough for a tiny-batch overfit test to be meaningful.
    """
    rng = np.random.default_rng(seed)
    entries: list[ShardEntry] = []
    base_ts = 1_500_000_000

    for group in groups:
        for shard_index in range(num_shards):
            t0 = base_ts + shard_index * timesteps_per_shard * sampling_interval_seconds
            timestamps = (
                t0 + np.arange(timesteps_per_shard, dtype=np.int64) * sampling_interval_seconds
            )
            tod = (timestamps % 86_400) / 86_400.0
            phase = rng.uniform(0, 2 * np.pi, size=(num_sensors,))
            amplitude = rng.uniform(0.5, 1.5, size=(num_sensors,))
            signal = amplitude[None, :] * np.sin(2 * np.pi * tod[:, None] + phase[None, :])
            noise = rng.normal(0.0, 0.05, size=(timesteps_per_shard, num_sensors))
            values = (signal + noise).astype(np.float32)
            series = np.repeat(values[:, :, None], num_features, axis=2).astype(np.float32)
            observed = rng.random((timesteps_per_shard, num_sensors)) >= missing_rate
            # Missing observations are stored as NaN so an unmasked loss fails loudly
            # rather than silently training on a plausible-looking zero.
            series[~observed] = np.nan

            rel = f"{group}/shard-{shard_index:05d}{SHARD_FILE_SUFFIX}"
            path = root / rel
            write_shard(path, series, observed, timestamps, group)
            entries.append(
                ShardEntry(
                    path=rel,
                    sha256=sha256_file(path),
                    bytes=path.stat().st_size,
                    group=group,
                    shard_index=shard_index,
                    num_timesteps=timesteps_per_shard,
                    num_sensors=num_sensors,
                    num_features=num_features,
                    first_timestamp=int(timestamps[0]),
                    last_timestamp=int(timestamps[-1]),
                )
            )

    manifest = ShardManifest(
        dataset_name="synthetic",
        dataset_version=dataset_version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        sampling_interval_seconds=sampling_interval_seconds,
        context_steps=context_steps,
        horizon_steps=horizon_steps,
        stride=stride,
        shards=entries,
        notes="Synthetic fixture. Tests only; never valid for model-quality claims.",
    )
    manifest.write(root / "shards.json")
    return manifest


def shard_array_keys(path: Path) -> list[str]:
    """Introspection helper used by tests to assert shard layout."""
    with np.load(path, allow_pickle=False) as z:
        keys: list[Any] = list(z.files)
    return [str(k) for k in keys]
