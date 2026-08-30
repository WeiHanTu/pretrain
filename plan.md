# Implementation and Verification Plan

**Overall status:** Not started  
**Planning principle:** Finish the three-week defensible minimum before adding model novelty.

## Status vocabulary

- `[ ]` not started
- `[~]` in progress
- `[x]` implemented and verified with named evidence
- `[!]` blocked with a documented reason

Do not mark an item complete based only on code existence. Its verification command and evidence file must exist.

## Track 0 — Access requests in parallel

- [ ] Ask the active UCSD collaborator whether NRP/Nautilus namespace access is available.
- [ ] Confirm whether the relevant namespace permits multi-node GPU Jobs or a PyTorch operator.
- [ ] Assess current eligibility for an ACCESS Explore request and obtain the required advisor letter if applicable.
- [ ] Do **not** start the GCP trial clock until Phase A passes.

Evidence:

- `reports/access/compute_options.md` recording facts, dates and unresolved constraints.

## Phase A — Local correctness and oracles ($0)

### A1. Repository bootstrap

- [ ] Add `pyproject.toml` with pinned Python range, runtime/dev dependency groups and CLI entry point.
- [ ] Generate and commit `uv.lock`.
- [ ] Add `.gitignore`, `README.md`, license decision and CI workflow.
- [ ] Add typed config schema and the five initial TOML configs.
- [ ] Add artifact/incident JSON schemas.

Verification:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

Evidence: `artifacts/gates/phase-a1.txt`.

### A2. Synthetic data and rank coverage

- [ ] Implement stable sample IDs and immutable shard manifests.
- [ ] Implement deterministic shard/sampler ordering.
- [ ] Implement loader state serialization.
- [ ] Implement coverage collector and verifier.
- [ ] Add 1-, 2- and 4-rank Gloo tests.
- [ ] Add duplicate and missing-sample negative tests.
- [ ] Add corrupt-shard checksum rejection test.

Acceptance:

- Exact multiset equality for the clean fixture.
- Duplicate/missing IDs named in negative-test output.
- Corruption detected before optimizer step zero.

Evidence:

- `artifacts/coverage/world-size-{1,2,4}.json`
- `reports/incidents/I-002-corrupt-shard.md`
- `reports/incidents/I-003-rank-coverage.md`

### A3. Model and training loop

- [ ] Implement the minimum spatiotemporal Transformer.
- [ ] Enforce shape/dtype/device and missing-mask contracts.
- [ ] Implement single-process train/evaluate commands.
- [ ] Pass a tiny-batch overfit test.
- [ ] Emit resolved config, parameter count and run manifest.

Acceptance:

- Fixed synthetic batch loss decreases below a predeclared threshold.
- Formal run refuses a dirty or unverifiable data manifest unless explicitly marked nonformal.

Evidence: `artifacts/gates/tiny-overfit.json`.

### A4. Seed-variance oracle

- [ ] Choose and document the comparison statistic and interval rule.
- [ ] Run three uninterrupted control seeds.
- [ ] Freeze the statistical pass rule before any changed-world-size resume.

Acceptance:

- Three traceable run manifests.
- `artifacts/oracles/seed-band.json` includes the exact runs and calculation.

### Phase A exit gate

All A1–A4 checks pass. Only then may the user start/activate GCP credit or schedule shared GPU compute.

## Phase B — GPU FSDP2 and checkpoint correctness

### B1. FSDP2 integration

- [ ] Add explicit DeviceMesh construction.
- [ ] Shard Transformer blocks and root module.
- [ ] Verify effective global batch and gradient accumulation.
- [ ] Capture single-GPU and local multi-GPU memory/throughput where available.

### B2. Distributed checkpoint contract

- [ ] Save/load model and optimizer with DCP.
- [ ] Persist scheduler, step, RNG, sampler cursor and manifest metadata.
- [ ] Add atomic complete-marker semantics.
- [ ] Reject intentionally interrupted/incomplete checkpoint.

### B3. Deterministic same-world-size resume

- [ ] Run uninterrupted deterministic FP32 control for 20 steps.
- [ ] Run 10 steps, checkpoint, resume to step 20.
- [ ] Compare parameters, optimizer, scheduler, losses and sample IDs exactly.

Acceptance: zero differences. Any difference blocks the gate pending diagnosis.

Evidence:

- `artifacts/oracles/exact-resume.json`
- `reports/incidents/I-001a-deterministic-resume.md`

### B4. Changed-world-size resume

- [ ] Verify 1→2 reshard and resume.
- [ ] Verify 2→1 reshard and resume.
- [ ] Verify post-resume rank coverage.
- [ ] Apply the frozen statistical oracle.
- [ ] Attempt 2→4 only if hardware is available without delaying Phase C.

Evidence:

- `artifacts/checkpoints/reshard-matrix.json`
- `reports/incidents/I-001b-world-size-change.md`

### Phase B exit gate

Exact same-world-size resume passes; 1↔2 DCP reshard passes; artifacts validate against schemas.

## Phase C — Real two-host execution

### C0. Cost and safety gate

- [ ] User confirms GCP activation/paid-billing risk or NRP namespace.
- [ ] Confirm two-GPU quota/capacity; do not assume it.
- [ ] Configure budget alerts and document that alerts do not cap spend.
- [ ] Validate VM expiry/deletion and cleanup commands without GPUs.
- [ ] Set `max_steps`, maximum wall time and checkpoint destination.

Evidence: `artifacts/cloud/preflight.json`.

### C1. Two-host NCCL smoke test

- [ ] Provision two physically distinct GPU hosts in the same network boundary.
- [ ] Run NCCL connectivity/all-reduce test.
- [ ] Capture topology and per-host environment.
- [ ] Verify no credentials appear in captured artifacts.

Evidence:

- `artifacts/cloud/topology.json`
- `artifacts/cloud/nccl-test.json`

### C2. Multi-node training baseline

- [ ] Run the formal two-host training configuration.
- [ ] Compare 1-GPU and 2-GPU throughput under the same global workload.
- [ ] Report speedup, scaling efficiency, step p50/p95 and peak memory.
- [ ] Explain network/topology limits rather than overgeneralizing.

Evidence: `artifacts/scaling/strong-scaling.json`.

### C3. Injected rank failure and recovery

- [ ] Terminate one rank at a declared step.
- [ ] Confirm the run fails visibly and ignores incomplete checkpoint output.
- [ ] Resume from the last committed checkpoint.
- [ ] Evaluate with the appropriate deterministic/statistical oracle.

Evidence: `reports/incidents/I-001-rank-termination.md`.

### C4. Straggler diagnosis

- [ ] Inject a 30% delay in one rank for a bounded interval.
- [ ] Capture per-rank step-time histograms.
- [ ] Diagnose global impact and remove the injection.

Evidence: `reports/incidents/I-004-straggler.md`.

### Phase C exit gate

Distinct-host evidence, NCCL evidence, two-host training, failure/recovery and straggler artifacts all exist and validate.

## Phase D — Portfolio release

- [ ] Write the main incident report around the strongest detected failure.
- [ ] Add an architecture diagram grounded in implemented components.
- [ ] Publish the exact test matrix and captured commands.
- [ ] Publish cost ledger and GPU-hours per evidence milestone.
- [ ] Add limitations prominently: portfolio scale, network class, model size and dataset boundaries.
- [ ] Run all release gates from a clean checkout with `uv sync --locked`.
- [ ] Confirm raw data, checkpoints, credentials and large logs are absent from Git.
- [ ] Reconcile every README/resume claim against an artifact.

Release gate:

```bash
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
git diff --check
git status --short
```

Evidence: `artifacts/gates/release.txt` plus a clean, reviewable repository state.

## Optional extensions — only after Phase D

- [ ] Divergence/rewind intervention experiment with predeclared policy.
- [ ] LargeST held-out-region transfer evaluation.
- [ ] Async DCP overhead comparison.
- [ ] NRP Kubernetes deployment if GCP was used for the minimum project.
- [ ] 2→4 world-size resume.
- [ ] JAX implementation of one small correctness experiment.
- [ ] Conservation metric only after directed topology and boundary-flow requirements are met.

## Explicitly deferred decisions

- Exact LargeST subsets and time windows, pending data inspection and license/citation review.
- Final Python/PyTorch versions, pending compatibility check before lockfile creation.
- GCP region, pending L4 quota and capacity.
- NRP PyTorch operator choice, pending namespace capabilities.
- Public license for project code, pending dependency/data compatibility review.

