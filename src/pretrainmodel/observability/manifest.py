"""Immutable run manifest (spec 10.3).

A reported number that cannot be traced to a manifest is not evidence.  The
manifest binds a result to the exact commit, config, data and topology that
produced it.

The ``formal`` flag is the gate that keeps that binding meaningful.  A formal run
refuses to start from a dirty working tree, because "commit abc123" would then
describe code that was never what actually ran.  Exploratory runs are still
allowed -- they are simply recorded as non-citable.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pretrainmodel.config import Config, config_hash, to_dict
from pretrainmodel.data.manifest import sha256_file
from pretrainmodel.distributed.topology import (
    Topology,
    capture_topology,
    software_environment,
)
from pretrainmodel.observability.cost import CostRecord

__all__ = ["FormalRunError", "RunManifest", "git_state", "new_run_id"]


class FormalRunError(RuntimeError):
    """A run marked formal does not meet the conditions for citable evidence."""


def new_run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}-{stamp}-{uuid.uuid4().hex[:8]}"


def git_state(root: Path) -> dict[str, Any]:
    """Commit, branch and dirty flag.  Missing git is recorded, never guessed."""

    def _run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return out.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": commit or "unknown",
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else True,
    }


class RunManifest:
    """Builder for the manifest written at the end of every run."""

    def __init__(
        self,
        cfg: Config,
        *,
        repo_root: Path,
        run_id: str | None = None,
        data_manifest_hash: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.repo_root = repo_root
        self.run_id = run_id or new_run_id(cfg.run.name)
        self.git = git_state(repo_root)
        self.started_at = datetime.now(UTC)
        self.data_manifest_hash = data_manifest_hash
        self.model: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.artifacts: list[dict[str, Any]] = []
        self.hardware: dict[str, Any] = {}
        self.topology: Topology | None = None
        self.cost: CostRecord | None = None
        self.status = "started"

        if cfg.run.formal:
            self._assert_citable()

    def _assert_citable(self) -> None:
        problems: list[str] = []
        if self.git["dirty"]:
            problems.append(
                "working tree is dirty; a formal run must be reproducible from a commit. "
                "Commit or stash first, or set run.formal = false."
            )
        if self.git["commit"] == "unknown":
            problems.append("git commit could not be determined")
        if problems:
            raise FormalRunError("refusing to start a formal run:\n  - " + "\n  - ".join(problems))

    def record_artifact(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        self.artifacts.append(
            {
                "path": str(path.relative_to(self.repo_root))
                if path.is_relative_to(self.repo_root)
                else str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    def to_dict(self, *, status: str | None = None) -> dict[str, Any]:
        ended = datetime.now(UTC)
        # Topology is schema-required: every run ran somewhere, and a manifest that
        # omits where is not traceable. Capture it lazily if the caller did not.
        if self.topology is None:
            self.topology = capture_topology(
                global_batch_size=self.cfg.global_batch_size(1),
                grad_accum_steps=self.cfg.train.grad_accum_steps,
            )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "run_name": self.cfg.run.name,
            "formal": self.cfg.run.formal,
            "status": status or self.status,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "ended_at": ended.isoformat(timespec="seconds"),
            "duration_seconds": (ended - self.started_at).total_seconds(),
            "git": self.git,
            "config_hash": config_hash(self.cfg),
            "config": to_dict(self.cfg),
            "data_manifest_hash": self.data_manifest_hash,
            "software": software_environment(),
            "hardware": self.hardware,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        if self.model:
            payload["model"] = self.model
        if self.topology is not None:
            payload["topology"] = self.topology.to_dict()
        if self.cost is not None:
            payload["cost"] = self.cost.to_dict()
        return payload

    def write(self, path: Path, *, status: str = "completed") -> dict[str, Any]:
        payload = self.to_dict(status=status)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload
