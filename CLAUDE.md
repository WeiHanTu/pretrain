# pretrainmodel

This file is the project-level `/init` context for Claude Code and other coding agents.

## Purpose

Build a small, reproducible multi-node pretraining system over traffic-flow data. The portfolio evidence is distributed-systems judgment: rank-correct data consumption, FSDP2, distributed checkpoints, recovery across failures and world-size changes, and measured cost/performance. The model is deliberately small.

Read these documents before changing code:

1. `intend.md` — product intent, evidence boundaries, success and non-goals.
2. `spec.md` — required architecture, schemas, invariants and acceptance criteria.
3. `plan.md` — ordered implementation and verification work.

If they conflict, use this precedence: `intend.md` → `spec.md` → `plan.md` → implementation convenience.

## Non-negotiable claim boundaries

- Never describe this as frontier-scale or production-scale foundation-model training.
- Never claim multi-node execution until an artifact from distinct physical hosts exists.
- Local processes, containers and multiple GPUs in one host are test environments, not multi-node evidence.
- Never claim bit-exact resume unless the deterministic same-world-size equality test passes.
- Never call a BF16 or changed-world-size trajectory equivalent without a predeclared seed-variance oracle.
- Never claim traffic conservation from speed-only data or from a graph without directed junction/boundary-flow semantics.
- Do not report MFU for the portfolio-scale L4 runs. Report throughput, step-time distributions, peak memory and scaling efficiency.
- Planned work must be labeled planned. Keep document implementation status synchronized with the code and captured artifacts.

## Engineering rules

- Python dependencies use `uv`, `pyproject.toml` and `uv.lock`; do not add `requirements.txt` or unmanaged `pip install` instructions.
- Pin Python and PyTorch versions in the lockfile and run manifests.
- Keep raw/processed datasets, checkpoints, credentials and cloud state out of Git.
- Every experiment writes an immutable run manifest containing the Git commit, config hash, data-manifest hash, environment, topology and cost fields.
- Every reported number must be traceable to a checked artifact under `artifacts/` or `reports/`; generated bulk data stays ignored.
- Synthetic data is for tests only and must never appear in model-quality claims.
- Prefer native PyTorch FSDP2 and `torch.distributed.checkpoint`; borrowing TorchTitan design patterns is fine, hiding all ownership behind a sample launcher is not.
- Keep CPU CI meaningful: data coverage, checkpoint metadata, configs, incident schemas and distributed Gloo tests must run without a GPU.
- Before cloud execution, require a hard `max_steps`, maximum wall-clock duration, checkpoint destination and cleanup command.

## Verification gates

Before marking a phase complete, run the commands defined for that phase in `plan.md` and save the evidence named there. A passing command without captured output is not portfolio evidence.

## Cost policy

- Compute priority: GCP credit → paid compute only with explicit user approval.
- Do not start the GCP 90-day credit window until the local Phase A gate passes.
- Internal GCP target: at most $120 of the $300 credit; reserve the rest for failure and quota/availability surprises.
- Never leave a GPU VM running unattended. Infrastructure must support automatic expiry/deletion.
