# Implementation and Verification Plan

**Overall status:** Phase A complete. Phase B *correctness* complete (B1-B4 verified on CPU/Gloo with captured evidence); Phase B *performance* measurements deferred to GPU hardware. Phase C not started.  
**Planning principle:** Finish the three-week defensible minimum before adding model novelty.

## Status vocabulary

- `[ ]` not started
- `[~]` in progress
- `[x]` implemented and verified with named evidence
- `[!]` blocked with a documented reason

Do not mark an item complete based only on code existence. Its verification command and evidence file must exist.

## Track 0 — Access requests in parallel

- [!] Ask the active UCSD collaborator whether NRP/Nautilus namespace access is available. RULED OUT: students are added to a namespace by their supervisor; the author has graduated and does not wish to ask his former advisor.
- [!] Confirm whether the relevant namespace permits multi-node GPU Jobs or a PyTorch operator. N/A — no namespace will be requested.
- [!] Assess current eligibility for an ACCESS Explore request and obtain the required advisor letter if applicable. RULED OUT: a graduate-student PI needs an advisor letter of collaboration — same constraint as NRP.
- [x] Do **not** start the GCP trial clock until Phase A passes. Phases A *and* B were both completed before signup, so the 90-day window is still unspent.

Evidence:

- `reports/access/compute_options.md` recording facts, dates and unresolved constraints.

## Phase A — Local correctness and oracles ($0)

### A1. Repository bootstrap

- [x] Add `pyproject.toml` with pinned Python range, runtime/dev dependency groups and CLI entry point.
- [x] Generate and commit `uv.lock`.
- [~] Add `.gitignore`, `README.md`, license decision and CI workflow. (`.gitignore`, `README.md`, CI and a repo-hygiene gate are in place; license remains a deferred decision.)
- [x] Add typed config schema and the five initial TOML configs.
- [x] Add artifact/incident JSON schemas.

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

- [x] Implement stable sample IDs and immutable shard manifests.
- [x] Implement deterministic shard/sampler ordering.
- [x] Implement loader state serialization.
- [x] Implement coverage collector and verifier.
- [x] Add 1-, 2- and 4-rank Gloo tests.
- [x] Add duplicate and missing-sample negative tests.
- [x] Add corrupt-shard checksum rejection test.

Acceptance:

- Exact multiset equality for the clean fixture.
- Duplicate/missing IDs named in negative-test output.
- Corruption detected before optimizer step zero.

Evidence:

- `artifacts/coverage/world-size-{1,2,4}.json`
- `reports/incidents/I-002-corrupt-shard.md`
- `reports/incidents/I-003-rank-coverage.md`

### A3. Model and training loop

- [x] Implement the minimum spatiotemporal Transformer.
- [x] Enforce shape/dtype/device and missing-mask contracts.
- [x] Implement single-process train/evaluate commands.
- [x] Pass a tiny-batch overfit test.
- [x] Emit resolved config, parameter count and run manifest.

Acceptance:

- Fixed synthetic batch loss decreases below a predeclared threshold.
- Formal run refuses a dirty or unverifiable data manifest unless explicitly marked nonformal.

Evidence: `artifacts/gates/tiny-overfit.json`.

### A4. Seed-variance oracle

- [x] Choose and document the comparison statistic and interval rule.
- [x] Run three uninterrupted control seeds.
- [x] Freeze the statistical pass rule before any changed-world-size resume.

Acceptance:

- Three traceable run manifests.
- `artifacts/oracles/seed-band.json` includes the exact runs and calculation.

### Phase A exit gate

All A1–A4 checks pass. Only then may the user start/activate GCP credit or schedule shared GPU compute.

## Phase B — GPU FSDP2 and checkpoint correctness

### B1. FSDP2 integration

- [x] Add explicit DeviceMesh construction.
- [x] Shard Transformer blocks and root module.
- [x] Verify effective global batch and gradient accumulation.
- [!] Capture single-GPU and local multi-GPU memory/throughput where available. BLOCKED: no GPU available locally. Deferred to Phase C hardware; this is a *measurement*, not a correctness gate, and does not block B2-B4.

### B2. Distributed checkpoint contract

- [x] Save/load model and optimizer with DCP.
- [x] Persist scheduler, step, RNG, sampler cursor and manifest metadata.
- [x] Add atomic complete-marker semantics.
- [x] Reject intentionally interrupted/incomplete checkpoint.

### B3. Deterministic same-world-size resume

- [x] Run uninterrupted deterministic FP32 control for 20 steps.
- [x] Run 10 steps, checkpoint, resume to step 20.
- [x] Compare parameters, optimizer, scheduler, losses and sample IDs exactly.

Acceptance: zero differences. Any difference blocks the gate pending diagnosis.

Evidence:

- `artifacts/oracles/exact-resume.json`
- `reports/incidents/I-001a-deterministic-resume.md`

### B4. Changed-world-size resume

- [x] Verify 1→2 reshard and resume.
- [x] Verify 2→1 reshard and resume.
- [x] Verify post-resume rank coverage.
- [~] Apply the frozen statistical oracle. DEFERRED with cause: resharding N→M at fixed `grad_accum_steps` changes the global batch size, so comparing the resumed loss against a band built at a different global batch would conflate the reshard with the batch-size change. A like-for-like comparison requires compensating `grad_accum_steps` to hold the global batch constant, which is itself a config change and therefore a separate experiment. Recorded in `artifacts/checkpoints/reshard-matrix.json` under `note_loss_comparison`.
- [x] Attempt 2→4 only if hardware is available without delaying Phase C. Done on CPU/Gloo: 2→4 and 4→2 both verified.

Evidence:

- `artifacts/checkpoints/reshard-matrix.json`
- `reports/incidents/I-001b-world-size-change.md`

### Phase B exit gate

Exact same-world-size resume passes; 1↔2 DCP reshard passes; artifacts validate against schemas.

## Phase C — Real two-host execution

### C0. Cost and safety gate

- [ ] User confirms GCP activation/paid-billing risk or NRP namespace.
- [ ] Confirm two-GPU quota/capacity; do not assume it.
- [~] Configure budget alerts and document that alerts do not cap spend. Documented in `reports/access/compute_options.md` and `artifacts/cloud/preflight.json`; *configuring* the alert is a user action and is tracked as UNANSWERED.
- [~] Validate VM expiry/deletion and cleanup commands without GPUs. `infra/gcp/{provision,startup,firewall,teardown}.sh` written and syntax-checked, with GCP-enforced `--max-run-duration` + `--instance-termination-action=DELETE` rather than a guest-side timer. NOT executed against the live API.
- [x] Set `max_steps`, maximum wall time and checkpoint destination. Asserted by `scripts/preflight_cloud.py`.

Evidence: `artifacts/cloud/preflight.json`.

### C-prep. Launch-path rehearsal ($0, done before provisioning)

Phases A and B used `file://` rendezvous, which works only when all ranks share a
filesystem. Real multi-node uses TCP. That path is now built and rehearsed locally so
the first paid hour is spent measuring, not debugging.

- [x] Add `distributed/launch.py`: env:// rendezvous, torchrun-compatible.
- [x] Add `distributed/entrypoint.py` runnable under `torchrun`; asserts topology
      **before** the first optimizer step.
- [x] Rehearse the real launcher locally (`--rehearsal`), honestly labelled as a
      single-host launch-path exercise and never as multi-node evidence.
- [x] Fail fast on an unresolvable hostname rather than hanging — see
      `reports/incidents/I-007-rendezvous-hostname-hang.md`.

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

## Finding: Phase B correctness does not need a GPU

`fully_shard` (FSDP2) works over a Gloo CPU `DeviceMesh` and produces genuinely
sharded DTensor parameters — a (32, 16) weight becomes (16, 16) local on 2 ranks,
and per-rank resident fraction tracks `1 / world_size` exactly. `torch.distributed
.checkpoint` saves and loads that sharded state, including across a world-size
change.

Phase B therefore splits along a line the original plan did not draw:

| | Needs a GPU? | Status |
|---|---|---|
| FSDP2 sharding, DCP save/load, resharding, resume oracles, failure injection | **No** | Complete on CPU/Gloo |
| Throughput, step-time distribution, peak memory, scaling efficiency, MFU | **Yes** | Deferred to Phase C hardware |

This matters for the cost policy: every correctness gate can be passed before the
GCP 90-day credit clock starts, so paid compute is spent only on measurements that
genuinely require hardware.

## Evaluated and rejected

### FlashKDA / Kimi Delta Attention (MoonshotAI/FlashKDA) — rejected 2026-08-30

Considered as a possible model-side enhancement. Rejected on four independent
grounds, any one of which is disqualifying:

1. **It is an inference kernel.** Its documentation directs callers to use
   `torch.inference_mode()`. This project trains.
2. **It requires SM90+** (H100/H20/GB200). The planned hardware is L4, which is
   SM89 (Ada). It cannot build or run there, and moving to H100s contradicts the
   cost model in `intend.md` §7.
3. **CUDA 12.9+ and fixed K=V=128 head dimensions**, neither of which this model
   or the planned images satisfy.
4. **The architectural motivation is absent at these sequence lengths.** KDA is
   linear attention for long context; this model attends over 12 timesteps and
   1024 sensors on separate axes, and quadratic attention over 12 tokens is free.

There is also a thesis argument, which generalises past this specific library:
`intend.md` §1 states that the model exists to make distributed failures real. A
novel attention variant makes the *model* the claim, and a reviewer cannot verify
that without domain expertise — whereas "every sample was consumed exactly once"
is checkable in under a minute.

A kernel-equivalence experiment (fused kernel vs reference implementation, judged
against a declared tolerance) would be a *good* Phase D extension, because it
reuses the oracle machinery already built. It is not a Phase C activity, and it is
not this library.

## Known issues

- **Step-time breakdown is not instrumented.** `spec.md` §10.2 requires data wait,
  forward/backward, optimizer and checkpoint durations; the loop currently records
  only total step time (p50/p95). Phase C exists to measure, so this should be added
  before provisioning rather than after.
- **`determinism.async_checkpoint` is declared and implemented nowhere.** It is
  validated by the config schema and honoured by no code path — the same shape of
  defect as I-008. Either implement it or remove the field; leaving it invites the
  next reader to assume it works.
- **PyTorch wheel pulls the full CUDA stack on linux-aarch64.** The default PyPI
  `torch` wheel declares CUDA runtime dependencies (~4.6 GB) even on hosts with no
  NVIDIA GPU, which also inflates CPU CI. The standard fix is PyTorch's variant
  index (`download.pytorch.org/whl/cpu`) selected by a `cpu`/`cu128` extra; the
  development sandbox cannot reach that host, so the default index is used for now.
  Revisit before Phase C, when GPU hosts need the matching CUDA build anyway.

## Explicitly deferred decisions

- Exact LargeST subsets and time windows, pending data inspection and license/citation review.
- Final Python/PyTorch versions, pending compatibility check before lockfile creation.
- GCP region, pending L4 quota and capacity.
- NRP PyTorch operator choice, pending namespace capabilities.
- Public license for project code, pending dependency/data compatibility review.

