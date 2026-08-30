# Multi-node Traffic Pretraining Incident Lab

A compact spatiotemporal Transformer **for** traffic-flow forecasting, built to
produce **evidence about distributed-training correctness and recovery** rather
than a competitive forecasting model.

**No traffic data has been trained on yet.** Every artifact in this repository was
produced from synthetic fixtures on CPU. Real-data training is Phase B onward.

The primary deliverable is an evidence-backed incident report. The model and
dataset exist to make distributed failures real and reproducible.

> **Claim boundary.** This is controlled portfolio-scale training. It is not
> frontier or foundation-model scale, and nothing here should be read as such.
> See [`intend.md`](intend.md) §6 for the full non-goals list.

## What this project is trying to prove

| # | Claim | Oracle |
|---|---|---|
| 1 | Every sample is consumed exactly once per logical epoch | Rank-coverage invariant (multiset equality) |
| 2 | Model, optimizer, scheduler, RNG and data cursor all resume | Checkpoint contract test |
| 3 | A deterministic same-world-size resume is **bit-exact** | Exact-equality oracle — **verified**, 186 tensors, 0 differences |
| 4 | A BF16 / changed-world-size resume is **statistically equivalent** | Seed-variance band, declared *before* the experiment |
| 5 | Checkpoints load at a different world size | DCP reshard matrix — **verified** 1↔2, 2↔4 on CPU/Gloo |
| 6 | Injected failures are detected and produce actionable evidence | Incident suite I-001 … I-005 |

Claims 3 and 4 are deliberately separate. Bit-exactness is achievable only under
the declared FP32/deterministic/synchronous constraints; a BF16 or resharded run
is compared against a pre-registered variance band instead. Conflating the two
would make the tolerance claim unfalsifiable.

## Implementation status

Status is tracked against [`plan.md`](plan.md). Nothing is marked complete without
a named artifact under `artifacts/` or `reports/`.

| Phase | Scope | Status |
|---|---|---|
| Track 0 | Compute access requests (NRP / ACCESS / GCP) | Not started |
| A1 | Repo bootstrap, config schema, CI | **Complete except license choice** — `artifacts/gates/phase-a1.txt` |
| A2 | Data layer, rank-coverage invariant, negative tests | **Complete** — `artifacts/coverage/`, `reports/incidents/I-002`, `I-003` |
| A3 | Model, training loop, run manifest | **Complete** — `artifacts/gates/tiny-overfit.json` |
| A4 | Seed-variance oracle | **Complete** — `artifacts/oracles/seed-band.json` |
| B | FSDP2, DCP checkpoint + reshard | **Correctness complete** — `artifacts/oracles/exact-resume.json`, `artifacts/checkpoints/reshard-matrix.json`; GPU throughput/memory deferred to Phase C |
| C | Real two-host execution, failure injection, scaling | Next |
| D | Portfolio release | Not started |

No multi-node claim is made until an artifact captured from **distinct physical
hosts** exists under `artifacts/cloud/`.

## Documents

Read in this precedence order; earlier documents win on conflict.

1. [`intend.md`](intend.md) — why the project exists, evidence boundaries, non-goals
2. [`spec.md`](spec.md) — required architecture, schemas, invariants, acceptance criteria
3. [`plan.md`](plan.md) — ordered implementation and verification work
4. [`CLAUDE.md`](CLAUDE.md) — agent/contributor operating rules

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/). Python and PyTorch versions are pinned
in `uv.lock`.

```bash
uv sync --locked
uv run pytest -q
```

CPU-only development is a first-class path: data coverage, checkpoint metadata,
config validation, incident schemas and the distributed Gloo tests all run without
a GPU.

### Verification gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest -q
```

## Data

The repository does **not** redistribute any dataset. LargeST is the intended
primary source, supplied separately by the user.

*Planned, not implemented.* `SourceManifest` defines the provenance record
(source URL, version, file hashes, units, timezone, graph directedness) and can
verify it, but **no LargeST ingestion or preprocessing exists yet** — nothing in
`src/` constructs a `SourceManifest`. The only dataset any code path reads today is
the synthetic fixture, which is for tests only and never appears in a
model-quality claim.

See [`intend.md`](intend.md) §4 for the dataset rationale and the conditions under
which a conservation metric would be defensible (it is out of scope until directed
junction topology and boundary flows are verified).

## Repository layout

```text
configs/     TOML run configurations
schemas/     JSON Schemas for run manifests, incidents and coverage artifacts
src/         library code
tests/       unit, distributed (Gloo), integration
infra/       GCP and NRP deployment material
artifacts/   small, reviewable evidence only
reports/     incident reports and access notes
```

## License

Not yet selected; pending dependency and dataset compatibility review
(`plan.md`, deferred decisions).
