"""Sample identity, index construction and shard integrity."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pretrainmodel.data.manifest import ManifestError, ShardManifest, sha256_file
from pretrainmodel.data.shard import build_index, load_shard, sample_id, write_synthetic_dataset
from pretrainmodel.incidents.inject import corrupt_file_bytes, restore_file_bytes, truncate_file


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, ShardManifest]:
    manifest = write_synthetic_dataset(
        tmp_path,
        num_shards=3,
        timesteps_per_shard=40,
        num_sensors=8,
        context_steps=6,
        horizon_steps=6,
        seed=11,
    )
    return tmp_path, manifest


# --------------------------------------------------------------------------- #
# Sample identity (spec 4.2)
# --------------------------------------------------------------------------- #


def test_sample_id_is_stable() -> None:
    a = sample_id("v1", "g0", 1_500_000_000, 12, 12)
    b = sample_id("v1", "g0", 1_500_000_000, 12, 12)
    assert a == b


def test_sample_id_varies_with_every_component() -> None:
    base = sample_id("v1", "g0", 1_500_000_000, 12, 12)
    assert sample_id("v2", "g0", 1_500_000_000, 12, 12) != base
    assert sample_id("v1", "g1", 1_500_000_000, 12, 12) != base
    assert sample_id("v1", "g0", 1_500_000_300, 12, 12) != base
    assert sample_id("v1", "g0", 1_500_000_000, 24, 12) != base
    assert sample_id("v1", "g0", 1_500_000_000, 12, 24) != base


def test_sample_id_is_hex_and_fixed_width() -> None:
    sid = sample_id("v1", "g0", 0, 12, 12)
    assert len(sid) == 16
    assert set(sid) <= set("0123456789abcdef")


# --------------------------------------------------------------------------- #
# Index construction
# --------------------------------------------------------------------------- #


def test_index_is_independent_of_shard_listing_order(dataset: tuple[Path, ShardManifest]) -> None:
    """Canonical order must not depend on filesystem or manifest ordering.

    If it did, two ranks enumerating shards in different orders would disagree on
    what the epoch contains, and the coverage invariant would be unverifiable.
    """
    _, manifest = dataset
    shuffled = list(manifest.shards)
    random.Random(3).shuffle(shuffled)
    reordered = replace(manifest, shards=shuffled)
    assert build_index(manifest).sample_ids == build_index(reordered).sample_ids


def test_index_has_no_duplicate_ids(dataset: tuple[Path, ShardManifest]) -> None:
    index = build_index(dataset[1])
    assert len(set(index.sample_ids)) == len(index)


def test_windows_never_cross_a_shard_boundary(dataset: tuple[Path, ShardManifest]) -> None:
    """Every window lies inside exactly one shard, so no sample spans two files."""
    _, manifest = dataset
    index = build_index(manifest)
    per_shard = manifest.shards[0].num_timesteps - manifest.seq_len + 1
    assert len(index) == len(manifest.shards) * per_shard
    for ref in index.samples:
        entry = next(e for e in manifest.shards if e.path == ref.shard_path)
        assert 0 <= ref.offset <= entry.num_timesteps - manifest.seq_len


def test_index_respects_stride(dataset: tuple[Path, ShardManifest]) -> None:
    _, manifest = dataset
    strided = replace(manifest, stride=3)
    assert len(build_index(strided)) < len(build_index(manifest))


def test_index_is_empty_when_shards_are_shorter_than_the_window(tmp_path: Path) -> None:
    manifest = write_synthetic_dataset(
        tmp_path, num_shards=2, timesteps_per_shard=5, context_steps=6, horizon_steps=6
    )
    assert len(build_index(manifest)) == 0


def test_index_needs_no_shard_bytes(dataset: tuple[Path, ShardManifest]) -> None:
    """build_index is a pure function of the manifest, so corruption cannot skew it."""
    root, manifest = dataset
    before = build_index(manifest).sample_ids
    target = root / manifest.shards[0].path
    corrupt_file_bytes(target, offset=64, count=32)
    assert build_index(manifest).sample_ids == before


# --------------------------------------------------------------------------- #
# Shard round-trip
# --------------------------------------------------------------------------- #


def test_shard_round_trip_preserves_contents(dataset: tuple[Path, ShardManifest]) -> None:
    root, manifest = dataset
    entry = manifest.shards[0]
    data = load_shard(root / entry.path)
    assert data.series.shape == (entry.num_timesteps, entry.num_sensors, entry.num_features)
    assert data.observed.shape == (entry.num_timesteps, entry.num_sensors)
    assert data.timestamps.shape == (entry.num_timesteps,)
    assert data.group == entry.group
    assert int(data.timestamps[0]) == entry.first_timestamp


def test_missing_values_are_nan_not_zero(dataset: tuple[Path, ShardManifest]) -> None:
    """A missing observation stored as 0.0 is indistinguishable from real zero flow.

    Storing NaN makes an unmasked loss produce NaN immediately instead of training
    on fabricated zeros.
    """
    root, manifest = dataset
    data = load_shard(root / manifest.shards[0].path)
    assert np.isnan(data.series[~data.observed]).all()
    assert not np.isnan(data.series[data.observed]).any()


def test_window_slices_the_expected_span(dataset: tuple[Path, ShardManifest]) -> None:
    root, manifest = dataset
    data = load_shard(root / manifest.shards[0].path)
    values, observed = data.window(4, manifest.seq_len)
    assert values.shape[0] == manifest.seq_len
    assert observed.shape[0] == manifest.seq_len
    np.testing.assert_array_equal(values, data.series[4 : 4 + manifest.seq_len])


# --------------------------------------------------------------------------- #
# Integrity gate (incident I-002)
# --------------------------------------------------------------------------- #


def test_clean_dataset_verifies(dataset: tuple[Path, ShardManifest]) -> None:
    root, manifest = dataset
    manifest.verify(root)


def test_corrupted_shard_is_rejected_and_named(dataset: tuple[Path, ShardManifest]) -> None:
    """Length-preserving corruption must be caught by the digest, not by size."""
    root, manifest = dataset
    target_rel = manifest.shards[1].path
    target = root / target_rel
    size_before = target.stat().st_size

    original = corrupt_file_bytes(target, offset=100, count=16)
    assert target.stat().st_size == size_before, "corruption must preserve length"

    with pytest.raises(ManifestError) as exc:
        manifest.verify(root)
    assert target_rel in str(exc.value)
    assert "sha256 mismatch" in str(exc.value)

    restore_file_bytes(target, original, offset=100)
    manifest.verify(root)


def test_truncated_shard_is_rejected(dataset: tuple[Path, ShardManifest]) -> None:
    root, manifest = dataset
    target = root / manifest.shards[0].path
    truncate_file(target, keep_bytes=32)
    with pytest.raises(ManifestError, match="size mismatch"):
        manifest.verify(root)


def test_missing_shard_is_rejected(dataset: tuple[Path, ShardManifest]) -> None:
    root, manifest = dataset
    (root / manifest.shards[2].path).unlink()
    with pytest.raises(ManifestError, match="missing shard"):
        manifest.verify(root)


def test_verify_reports_every_bad_shard_not_just_the_first(
    dataset: tuple[Path, ShardManifest],
) -> None:
    root, manifest = dataset
    corrupt_file_bytes(root / manifest.shards[0].path, offset=80, count=8)
    (root / manifest.shards[1].path).unlink()
    with pytest.raises(ManifestError) as exc:
        manifest.verify(root)
    assert manifest.shards[0].path in str(exc.value)
    assert manifest.shards[1].path in str(exc.value)


def test_corrupting_an_empty_file_is_an_error(tmp_path: Path) -> None:
    """A no-op 'corruption' would make the detector look like it passed."""
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty file"):
        corrupt_file_bytes(empty)


# --------------------------------------------------------------------------- #
# Manifest persistence
# --------------------------------------------------------------------------- #


def test_manifest_round_trips_through_json(dataset: tuple[Path, ShardManifest]) -> None:
    root, manifest = dataset
    reloaded = ShardManifest.read(root / "shards.json")
    assert reloaded.to_dict() == manifest.to_dict()


def test_manifest_read_rejects_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "shards.json"
    bad.write_text("{not json")
    with pytest.raises(ManifestError, match="malformed JSON"):
        ShardManifest.read(bad)


def test_sha256_file_matches_recorded_digest(dataset: tuple[Path, ShardManifest]) -> None:
    root, manifest = dataset
    entry = manifest.shards[0]
    assert sha256_file(root / entry.path) == entry.sha256
