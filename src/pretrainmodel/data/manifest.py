"""Source and shard manifests.

Two manifests exist and they answer different questions:

``SourceManifest`` records where data came from -- URL, version, licence, file
hashes, units, timezone, graph provenance.  It is what makes a model-quality claim
traceable to a specific download rather than to "the traffic dataset".

``ShardManifest`` records what the immutable preprocessed shards contain and what
they hash to.  It is what makes a *training* run reproducible, and it is the input
to the corruption gate: a shard whose bytes no longer match its recorded digest is
rejected before the first optimizer step rather than silently trained on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ManifestError",
    "ShardEntry",
    "ShardManifest",
    "SourceFile",
    "SourceManifest",
    "content_hash",
    "manifest_hash",
    "sha256_file",
]

_CHUNK = 1 << 20


class ManifestError(RuntimeError):
    """A manifest is missing, malformed, or does not describe the data on disk."""


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256.  Shards can be large; never read them whole."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Source manifest (spec 4.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One file as retrieved from the upstream source."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceManifest:
    """Provenance for one dataset version.

    ``directed_graph`` is deliberately explicit and defaults to False.  A road
    *distance* adjacency matrix is undirected and carries no junction semantics, so
    a conservation-residual metric computed over it would be meaningless.  The flag
    is what a downstream metric checks before it is allowed to run.
    """

    dataset_name: str
    dataset_version: str
    source_url: str
    retrieved_at: str
    license_note: str
    citation: str
    files: list[SourceFile]
    semantic_fields: dict[str, str]
    units: str
    time_range: list[str]
    sampling_interval_seconds: int
    timezone: str
    sensor_count: int
    sensor_id_namespace: str
    missing_value_representation: str
    graph_provenance: str = "none"
    directed_graph: bool = False
    boundary_flows_available: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> SourceManifest:
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise ManifestError(f"source manifest not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}: malformed JSON: {exc}") from exc
        raw["files"] = [SourceFile(**f) for f in raw.get("files", [])]
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ManifestError(f"{path}: {exc}") from exc

    def verify(self, root: Path) -> None:
        """Confirm every recorded source file is present and unmodified."""
        problems: list[str] = []
        for entry in self.files:
            p = root / entry.path
            if not p.is_file():
                problems.append(f"missing source file: {entry.path}")
                continue
            actual = sha256_file(p)
            if actual != entry.sha256:
                problems.append(
                    f"{entry.path}: sha256 mismatch "
                    f"(recorded {entry.sha256[:12]}..., found {actual[:12]}...)"
                )
        if problems:
            raise ManifestError(
                f"source manifest verification failed for "
                f"{self.dataset_name}@{self.dataset_version}:\n  - " + "\n  - ".join(problems)
            )

    def supports_conservation_metric(self) -> bool:
        """Whether a vehicle-conservation residual is defensible for this source.

        Guards spec.md 5.3: speed-only data, or a graph without directed junction
        topology and boundary flows, cannot support a conservation claim.
        """
        return self.directed_graph and self.boundary_flows_available and self.units != "speed"


# --------------------------------------------------------------------------- #
# Shard manifest (spec 4.3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ShardEntry:
    """One immutable preprocessed shard."""

    path: str
    sha256: str
    bytes: int
    group: str
    shard_index: int
    num_timesteps: int
    num_sensors: int
    num_features: int
    first_timestamp: int
    last_timestamp: int


@dataclass(frozen=True, slots=True)
class ShardManifest:
    """The immutable description of a processed dataset version."""

    dataset_name: str
    dataset_version: str
    created_at: str
    sampling_interval_seconds: int
    context_steps: int
    horizon_steps: int
    stride: int
    shards: list[ShardEntry]
    source_manifest_sha256: str = ""
    missing_value_representation: str = "nan"
    notes: str = ""
    schema_version: int = 1

    @property
    def seq_len(self) -> int:
        return self.context_steps + self.horizon_steps

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> ShardManifest:
        try:
            raw = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise ManifestError(f"shard manifest not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}: malformed JSON: {exc}") from exc
        raw["shards"] = [ShardEntry(**s) for s in raw.get("shards", [])]
        try:
            return cls(**raw)
        except TypeError as exc:
            raise ManifestError(f"{path}: {exc}") from exc

    def verify(self, root: Path) -> None:
        """Reject the dataset if any shard is missing, truncated or modified.

        Called before the first optimizer step of every run (incident I-002).  The
        error names the offending file: "training diverged" is a useless symptom if
        the real cause was a shard that changed under you.
        """
        problems: list[str] = []
        for entry in self.shards:
            p = root / entry.path
            if not p.is_file():
                problems.append(f"missing shard: {entry.path}")
                continue
            size = p.stat().st_size
            if size != entry.bytes:
                problems.append(
                    f"{entry.path}: size mismatch (recorded {entry.bytes:,}, found {size:,})"
                )
                continue
            actual = sha256_file(p)
            if actual != entry.sha256:
                problems.append(
                    f"{entry.path}: sha256 mismatch "
                    f"(recorded {entry.sha256[:12]}..., found {actual[:12]}...)"
                )
        if problems:
            raise ManifestError(
                "shard verification failed for "
                f"{self.dataset_name}@{self.dataset_version}:\n  - " + "\n  - ".join(problems)
            )


def content_hash(manifest: ShardManifest) -> str:
    """Digest of what the shards CONTAIN, ignoring when they were created.

    Distinct from :func:`manifest_hash`, and the difference matters across ranks.
    ``manifest_hash`` covers provenance including ``created_at``, so two nodes that
    generate an identical fixture from the same seed still disagree on it. Content
    is what must match between ranks; provenance is per-copy.
    """
    payload = json.dumps(
        {
            "dataset_name": manifest.dataset_name,
            "dataset_version": manifest.dataset_version,
            "context_steps": manifest.context_steps,
            "horizon_steps": manifest.horizon_steps,
            "stride": manifest.stride,
            "sampling_interval_seconds": manifest.sampling_interval_seconds,
            "shards": sorted(
                (s.path, s.sha256, s.bytes, s.num_timesteps, s.num_sensors) for s in manifest.shards
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def manifest_hash(manifest: ShardManifest) -> str:
    """Stable digest of a shard manifest, recorded in every run manifest.

    Two runs sharing this digest consumed byte-identical data.
    """
    payload = json.dumps(manifest.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
