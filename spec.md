# Specification: Multi-node Traffic Pretraining Incident Lab

**Implementation status:** Sections 1-5, 10 and 12 implemented for the local CPU path (Phase A). Sections 6-9 (FSDP2, DCP checkpointing, resume oracles, incident injection beyond I-002/I-003) and 11 (cloud) are NOT implemented.  
**Target:** Three-week defensible minimum, followed by optional extensions  
**Normative language:** MUST/SHALL are acceptance requirements; SHOULD is recommended; MAY is optional.

## 1. System boundary

The system SHALL train a compact spatiotemporal Transformer over sharded traffic-flow sequences using one or more PyTorch distributed ranks. It SHALL produce machine-readable evidence for data coverage, topology, checkpoints, metrics, incidents and cost.

The system has five logical layers:

```text
source files -> verified manifest -> immutable sample shards -> distributed loader
                                                          -> Transformer + objective
                                                          -> FSDP2 training loop
                                                          -> DCP checkpoint/recovery
                                                          -> evidence + incident reports
```

## 2. Technology baseline

- Python 3.11 or 3.12, selected before the first lockfile and then pinned.
- `uv` for dependency resolution, environments and commands.
- Native PyTorch distributed APIs.
- FSDP2 via `torch.distributed.fsdp.fully_shard`.
- `torch.distributed.checkpoint` (DCP) for model and optimizer state.
- NCCL for GPU distributed execution; Gloo for CPU tests.
- TOML configuration, JSON/JSONL evidence and Markdown incident reports.
- Ruff, mypy and pytest for quality gates.

TorchTitan MAY be consulted or adapted, but the core data-state, checkpoint metadata, incident injection and evidence code SHALL live in this repository.

## 3. Proposed repository layout

```text
pretrainmodel/
├── CLAUDE.md
├── AGENTS.md
├── intend.md
├── spec.md
├── plan.md
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── local_smoke.toml
│   ├── deterministic_resume.toml
│   ├── seed_band.toml
│   ├── single_l4.toml
│   └── two_node_l4.toml
├── src/pretrainmodel/
│   ├── config.py
│   ├── data/{manifest,shard,loader,coverage}.py
│   ├── model/{embedding,transformer,objective}.py
│   ├── distributed/{topology,parallel,checkpoint}.py
│   ├── training/{state,loop,interventions}.py
│   ├── observability/{events,metrics,cost}.py
│   ├── incidents/{inject,schemas}.py
│   └── cli.py
├── tests/
│   ├── unit/
│   ├── distributed/
│   ├── integration/
│   └── fixtures/
├── infra/
│   ├── gcp/
│   └── nrp/
├── artifacts/          # small, reviewable evidence only
├── reports/incidents/
└── scripts/
```

## 4. Data contract

### 4.1 Source manifest

Each source SHALL have a manifest with:

- `dataset_name`, `dataset_version`, `source_url`, `retrieved_at`
- license/citation note
- file path, byte size and SHA-256
- semantic fields and units
- time range, sampling interval and timezone
- sensor count and sensor-ID namespace
- missing-value representation
- graph provenance and whether edges are directed

Raw and processed data SHALL be Git-ignored.

### 4.2 Logical sample identity

Every sample SHALL have a stable ID derived from dataset version, region/sensor group, start timestamp, context length and forecast horizon. IDs SHALL not depend on rank, worker or filesystem enumeration order.

### 4.3 Shards

- Shards SHALL have immutable manifests and SHA-256 checksums.
- The loader SHALL verify the manifest before training unless explicitly running a corruption test.
- Shard ordering and within-shard ordering SHALL be reproducible from the run seed.
- Loader state SHALL expose the logical cursor required for resume.

### 4.4 Rank-coverage invariant

For an epoch without replacement:

```text
multiset_union(consumed_ids_by_rank) == expected_epoch_ids
and intersection between ranks is empty
```

The coverage test SHALL run at world sizes 1, 2 and 4 using synthetic fixtures. A deliberately duplicated or missing sample SHALL make the test fail with the offending IDs.

## 5. Model and objective

### 5.1 Minimum model

The minimum model SHALL be a 20M–50M parameter spatiotemporal Transformer with:

- numeric value projection
- sensor/region identity embedding
- temporal position/calendar features
- masked attention appropriate to the forecasting objective
- forecast head for one or more future horizons

The exact parameter count SHALL be emitted in the run manifest.

### 5.2 Objective

The core objective SHALL be autoregressive or direct multi-horizon traffic-flow prediction. MAE and RMSE SHALL be reported at declared horizons. Missing observations SHALL be masked consistently in both loss and evaluation.

Masked reconstruction and held-out-region adaptation MAY be extensions. They are not prerequisites for the distributed-systems evidence.

### 5.3 Physics-aware metrics

Non-negativity and plausible-range violations MAY be reported as domain constraints. Vehicle conservation MUST NOT be reported unless directed junction topology, boundary flows and compatible units are verified in the source manifest.

## 6. Distributed topology

Every distributed run SHALL record:

- hostnames and hashed instance identifiers
- backend, rank, local rank, world size and rendezvous configuration
- GPU model, driver, CUDA, NCCL, PyTorch and Python versions
- node count and GPU count per node
- network zone/region and whether hosts are physically distinct
- effective global batch and gradient accumulation

FSDP2 SHALL shard Transformer blocks and the root module using an explicit `DeviceMesh`. The single-rank path SHALL use the same training loop without pretending that world size one exercises communication.

## 7. Checkpoint contract

Each checkpoint SHALL contain or reference:

- model and optimizer state through DCP
- scheduler and global-step state
- logical epoch and sample cursor
- CPU RNG state
- CUDA RNG state per rank when applicable
- sampler/shuffle state
- run/config/data-manifest hashes
- checkpoint format version
- committed/complete marker written only after all required rank output succeeds

An incomplete checkpoint SHALL never be selected as the latest valid checkpoint.

## 8. Resume oracles

### 8.1 Deterministic same-world-size oracle

Configuration:

- FP32
- same hardware and world size
- deterministic algorithms enabled
- asynchronous checkpointing disabled
- fixed data order and 20-step run
- checkpoint after step 10

Acceptance: uninterrupted and resumed runs SHALL have exactly equal final parameters, optimizer state, scheduler state, consumed sample IDs and loss sequence after the resume boundary. If a supported operation prevents exactness, the gate fails and the limitation must be diagnosed; the test must not be weakened silently.

### 8.2 Statistical oracle

Three uninterrupted control seeds SHALL establish a variance band for the selected loss/evaluation statistic before BF16, async or changed-world-size resume is evaluated. The statistic, interval construction and pass rule SHALL be declared before the resume run.

Acceptance: the resumed trajectory falls within the declared control band. This is statistical equivalence, not bit equality.

### 8.3 Changed-world-size resume

DCP SHALL save under one world size and load under another. The minimum matrix is 1→2 and 2→1 locally or on available GPUs; 2→4 is an extension.

Acceptance:

- model and optimizer load successfully through DCP resharding
- the post-resume logical sample set obeys the rank-coverage invariant
- loss/evaluation behavior passes the statistical oracle
- the artifact records old/new topology and checkpoint metadata

## 9. Incident injection

The minimum suite SHALL include:

### I-001: rank/process termination

Terminate one rank at a declared step. The run SHALL fail visibly, preserve the last committed checkpoint and resume without consuming an incomplete checkpoint.

### I-002: corrupt shard

Modify a fixture shard after manifest creation. Checksum validation SHALL reject it before the first optimizer step and identify the file.

### I-003: duplicated or missing sample

Inject a sampler defect. The rank-coverage assertion SHALL report duplicate/missing IDs.

### I-004: straggler

Delay one rank by a controlled amount. Per-rank step-time histograms SHALL make the straggler visible and quantify its impact on global throughput.

### I-005: divergence and rewind

An extension SHALL induce a loss spike with an intentionally unsafe configuration, apply a predeclared intervention policy, rewind to the prior valid checkpoint and record whether the intervention recovered. This is an experiment, not evidence of an organic expensive-run incident.

## 10. Observability and evidence

### 10.1 Event record

Each rank SHALL write structured JSONL events with run ID, monotonic timestamp, rank, host, step, event type and typed payload.

### 10.2 Required metrics

- training and validation loss
- samples/second globally and by rank where meaningful
- step time p50/p95 and per-rank distribution
- data wait, forward/backward, optimizer and checkpoint durations
- peak allocated/reserved GPU memory
- strong-scaling speedup and efficiency
- checkpoint size and save/load time
- estimated compute cost and GPU-hours

MFU is excluded from the minimum project.

### 10.3 Run manifest

Every formal run SHALL produce an immutable manifest with:

- run ID, start/end status and timestamps
- Git commit and dirty-tree flag
- config hash and full resolved config
- data-manifest hash
- software/hardware/topology
- artifact paths and checksums
- billing SKU/rate assumption, GPU-hours and estimated cost

### 10.4 Incident report schema

Every incident report SHALL contain:

1. Scope and topology
2. Hypothesis and predeclared detection signal
3. Injection or observed failure
4. Detection evidence
5. Diagnosis
6. Intervention
7. Verification oracle and result
8. Cost and time lost
9. Counterfactual: what would happen without the control
10. Limitations and claim boundary

## 11. Cloud and infrastructure constraints

### 11.1 GCP

- Cloud execution SHALL not begin until the local gate in `plan.md` passes.
- GPU quota and availability SHALL be checked before assuming two L4s exist.
- Preferred minimum topology: two `g2-standard-4` Spot VMs, one L4 each, same zone/VPC.
- Every VM SHALL have an expiry mechanism and cleanup command.
- Every job SHALL have `max_steps`, maximum wall time and a durable checkpoint destination.
- GCP activation and quota actions remain user-controlled because activating paid billing enables charges beyond credit.

### 11.2 NRP/Nautilus

NRP manifests MAY use Kubernetes Jobs or an approved distributed-training operator. Namespace access and cluster policy must be confirmed before assuming PyTorchJob support.

### 11.3 Credentials

No credentials, service-account keys or signed URLs may enter Git, logs, manifests or reports. Use application-default or workload identity mechanisms appropriate to the platform.

## 12. Test matrix

| Layer | Required tests |
|---|---|
| Config | schema, unknown keys, cross-field invariants, stable hash |
| Data | source/shard hashes, stable sample IDs, missing masks, 1/2/4-rank coverage |
| Model | shape/dtype/device contracts, causal mask, tiny-batch overfit |
| Checkpoint | complete marker, interrupted save rejection, same-world exact resume |
| Reshard | DCP load 1→2 and 2→1, optimizer included |
| Incidents | corruption, duplicate/missing IDs, rank kill, straggler signal |
| Evidence | run manifest and incident schema validation, artifact checksums |
| CLI | smoke train/evaluate/resume/verify commands |
| Cloud | two distinct hosts, NCCL connectivity, automatic expiry/cleanup |

## 13. Definition of done

The minimum project is done only when all core acceptance criteria in Sections 4–12 pass and the evidence is captured. A README, configuration or unexecuted cloud script does not satisfy a run requirement.
