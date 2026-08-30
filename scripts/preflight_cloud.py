#!/usr/bin/env python
"""Phase C cost-and-safety preflight (plan.md C0).

    uv run python scripts/preflight_cloud.py

Writes artifacts/cloud/preflight.json.

Two kinds of check, kept visibly separate:

**Automated** -- things this machine can actually verify, and does.

**Manual** -- things that require a GCP account, and are therefore recorded as
UNANSWERED until a human answers them. They are not defaulted to "ok". A preflight
that quietly passes the checks it cannot perform is worse than no preflight, since
it converts an unknown into a false assurance right before the money starts.

The gate fails while any manual item is unanswered. That is deliberate: it should
be impossible to reach a green preflight without having looked at the billing page.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pretrainmodel.config import load_config
from pretrainmodel.distributed.launch import network_preflight

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "artifacts" / "cloud" / "preflight.json"
CLOUD_CONFIGS = ["configs/single_l4.toml", "configs/two_node_l4.toml"]

# Manual items. Each stays "unanswered" until a human records the answer in
# reports/access/compute_options.md and flips it here.
MANUAL_CHECKS = [
    {
        "id": "billing_upgraded",
        "question": "Is the billing account upgraded from Free Trial to paid?",
        "why": "The Free Trial forbids attaching GPUs at all. Upgrading also enables "
        "charges beyond the remaining credit, which is why this is a human decision.",
    },
    {
        "id": "gpu_quota_granted",
        "question": (
            "Is NVIDIA_L4_GPUS (or PREEMPTIBLE_NVIDIA_L4_GPUS) quota >= 2 in the target region?"
        ),
        "why": "New paid accounts start at 0. Quota requests take time and are sometimes denied.",
    },
    {
        "id": "capacity_available",
        "question": "Did a dry-run create succeed in the target zone?",
        "why": "Quota is permission, not inventory. Spot L4 capacity varies by zone and hour.",
    },
    {
        "id": "budget_alert_configured",
        "question": "Is a budget alert set on the billing account?",
        "why": "Alerts NOTIFY, they do not CAP. Nothing in GCP stops spend automatically; "
        "the real cap is --max-run-duration on each instance plus teardown.",
    },
    {
        "id": "credit_expiry_noted",
        "question": "Is the credit expiry date recorded? (90 days from SIGNUP, not first use)",
        "why": "The clock starts at signup, so unused credit silently expires.",
    },
]


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "ok": ok, "detail": detail}


def automated_checks() -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    net = network_preflight(require_hostname=True)
    results.append(
        _check(
            "hostname_resolves",
            net.hostname_resolves,
            f"hostname={net.hostname!r} resolves={net.hostname_resolves}. "
            + (net.problems[0] if net.problems else "OK")
            + " Must be re-run ON EACH CLOUD NODE; a failure here hangs c10d rendezvous "
            "indefinitely rather than erroring (see reports/incidents/I-007).",
        )
    )

    for rel in CLOUD_CONFIGS:
        cfg = load_config(REPO_ROOT / rel)
        has_caps = cfg.train.max_steps >= 1 and cfg.train.max_wall_seconds >= 1
        results.append(
            _check(
                f"caps_present::{rel}",
                has_caps,
                f"max_steps={cfg.train.max_steps}, "
                f"max_wall_seconds={cfg.train.max_wall_seconds}, "
                f"checkpoint_every={cfg.checkpoint.every_steps}",
            )
        )
        results.append(
            _check(
                f"checkpoint_cadence::{rel}",
                cfg.checkpoint.every_steps > 0,
                "Spot instances are preempted without warning. A run with no checkpoint "
                "cadence loses everything on preemption; with one, the preemption becomes "
                "a free organic recovery test.",
            )
        )

    for script in ("provision.sh", "teardown.sh", "firewall.sh", "startup.sh"):
        path = REPO_ROOT / "infra" / "gcp" / script
        exists = path.is_file()
        valid = False
        if exists and shutil.which("bash"):
            valid = (
                subprocess.run(
                    ["bash", "-n", str(path)], capture_output=True, check=False
                ).returncode
                == 0
            )
        results.append(
            _check(
                f"script_valid::{script}",
                exists and valid,
                f"exists={exists} syntax_ok={valid} (NOT executed against GCP)",
            )
        )

    provision = (REPO_ROOT / "infra/gcp/provision.sh").read_text()
    results.append(
        _check(
            "instance_self_destruct",
            "--max-run-duration" in provision
            and "--instance-termination-action=DELETE" in provision,
            "Provisioning must set a GCP-enforced deadline. A guest-side shutdown timer "
            "dies with the guest and so misses the one case it exists for.",
        )
    )

    leaked = credential_scan()
    results.append(
        _check(
            "no_credentials_in_artifacts",
            not leaked,
            "clean" if not leaked else f"possible secrets in: {leaked}",
        )
    )
    return results


def credential_scan() -> list[str]:
    """Scan captured evidence for anything that looks like a secret."""
    patterns = [
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r'"type"\s*:\s*"service_account"'),
        re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,}"),
        re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ]
    hits: list[str] = []
    for base in ("artifacts", "reports"):
        for path in (REPO_ROOT / base).rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if any(p.search(text) for p in patterns):
                hits.append(str(path.relative_to(REPO_ROOT)))
    return hits


def main() -> int:
    automated = automated_checks()
    manual = [
        {**m, "status": "unanswered", "answered_by": None, "answer": None} for m in MANUAL_CHECKS
    ]

    automated_ok = all(c["ok"] for c in automated)
    manual_ok = all(m["status"] == "answered" for m in manual)

    payload = {
        "schema_version": 1,
        "gate": "phase-c-preflight",
        "automated_checks": automated,
        "automated_passed": automated_ok,
        "manual_checks": manual,
        "manual_all_answered": manual_ok,
        "ready_to_provision": automated_ok and manual_ok,
        "note": (
            "Manual items are recorded as UNANSWERED rather than defaulted to ok. A "
            "preflight that silently passes checks it cannot perform converts an unknown "
            "into a false assurance at exactly the moment money starts being spent."
        ),
        "cost_note": (
            "Budget alerts notify; they do not cap. The enforced caps are "
            "--max-run-duration with --instance-termination-action=DELETE per instance, "
            "train.max_steps and train.max_wall_seconds in the config, and running "
            "infra/gcp/teardown.sh unconditionally after every session."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    for c in automated:
        print(f"  [{'OK  ' if c['ok'] else 'FAIL'}] {c['check']}")
    print(f"\n  automated: {'PASS' if automated_ok else 'FAIL'}")
    print(f"  manual:    {sum(m['status'] == 'answered' for m in manual)}/{len(manual)} answered")
    print(f"  ready to provision: {payload['ready_to_provision']}")
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    if not payload["ready_to_provision"]:
        print("\nUnanswered before spending anything:", file=sys.stderr)
        for m in manual:
            if m["status"] != "answered":
                print(f"  - {m['question']}\n      why: {m['why']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
