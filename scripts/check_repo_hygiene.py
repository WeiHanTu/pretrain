#!/usr/bin/env python
"""Fail the build if data, checkpoints, credentials or bulk artifacts are tracked.

A .gitignore only helps until someone runs ``git add -f``.  This gate inspects what
git is *actually* tracking, so the rule from CLAUDE.md ("keep raw/processed datasets,
checkpoints, credentials and cloud state out of Git") is enforced rather than merely
documented.

Exit status 0 when clean, 1 otherwise.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Paths that must never be tracked, matched against the repo-relative POSIX path.
FORBIDDEN_PATH_PATTERNS = [
    re.compile(r"^data/raw/"),
    re.compile(r"^data/processed/"),
    re.compile(r"^checkpoints/"),
    re.compile(r"^runs/"),
    re.compile(r"^logs/"),
    re.compile(r"^artifacts/generated/"),
    re.compile(r"\.(pt|pth|ckpt|safetensors|h5|hdf5|npz|npy)$"),
    re.compile(r"(^|/)\.env(\.|$)"),
    re.compile(r"\.(pem|key|p12|pfx)$"),
    re.compile(r"(^|/)service-account.*\.json$"),
    re.compile(r"(^|/)terraform\.tfstate"),
    re.compile(r"(^|/)kubeconfig$"),
]

# Content signatures for credentials that may be pasted into an otherwise
# innocuous file (a README, a notebook, an incident report).
SECRET_PATTERNS = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key block"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS access key id"),
    (re.compile(r'"type"\s*:\s*"service_account"'), "GCP service-account key"),
    (re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), "GitHub personal access token"),
    (re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"), "generic API secret key"),
]

# Evidence is meant to be small and reviewable.  A large tracked file is almost
# always a dataset or checkpoint that escaped the ignore rules.
MAX_TRACKED_BYTES = 1_048_576
SIZE_EXEMPT = {"uv.lock"}

TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".txt",
    ".cfg",
    ".ini",
    ".sh",
    ".tf",
    ".gitignore",
}


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [p for p in out.stdout.split("\0") if p]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        files = tracked_files(root)
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        print(f"hygiene: could not list tracked files: {exc}", file=sys.stderr)
        return 1

    problems: list[str] = []

    for rel in files:
        for pattern in FORBIDDEN_PATH_PATTERNS:
            if pattern.search(rel):
                problems.append(f"{rel}: tracked but matches forbidden pattern {pattern.pattern!r}")
                break

        path = root / rel
        if not path.is_file():
            continue

        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES and rel not in SIZE_EXEMPT:
            problems.append(
                f"{rel}: tracked file is {size:,} bytes "
                f"(limit {MAX_TRACKED_BYTES:,}); evidence must stay small and reviewable"
            )

        if path.suffix in TEXT_SUFFIXES or path.name == ".gitignore":
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except (UnicodeDecodeError, OSError):
                continue
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(text):
                    problems.append(f"{rel}: contains what looks like a {label}")

    if problems:
        print("Repository hygiene check FAILED:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    print(f"Repository hygiene check passed ({len(files)} tracked files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
