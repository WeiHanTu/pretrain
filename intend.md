# Intent: Multi-node Traffic Pretraining Incident Lab

**Status:** Approved design intent; implementation not started  
**Last updated:** 2026-08-30

## 1. Why this project exists

The project addresses one narrow portfolio gap: no captured evidence of operating a model-training job across a physical network boundary.

The goal is not to manufacture frontier-model seniority. A side project cannot reproduce the cost, consequence or organizational ownership of a large laboratory pretraining run. It can demonstrate that the engineer understands and has personally implemented the mechanics that make such runs correct, recoverable and inspectable.

The primary deliverable is therefore an evidence-backed incident report. The model and dataset exist to make the distributed failures real.

## 2. Intended audience

- Research engineering hiring managers evaluating distributed-training fundamentals.
- ML infrastructure engineers evaluating data correctness, checkpointing and recovery judgment.
- Recruiters who need a truthful bridge from small-scale implementation to a pretraining role.
- The project author, as a reusable training-systems laboratory.

## 3. Core proposition

Build and operate a compact spatiotemporal Transformer over California traffic-flow data, first locally and then on two distinct GPU hosts. Demonstrate:

1. Every expected training sample is consumed exactly once per logical epoch unless the configured sampler intentionally says otherwise.
2. Model, optimizer, scheduler, random-state and data-position state can be checkpointed and resumed.
3. A deterministic same-world-size resume is bit-exact under its declared constraints.
4. A BF16 or changed-world-size resume is evaluated against a seed-variance band established before the resume experiment.
5. Distributed checkpoints can be loaded with a different world size.
6. Injected rank failure, shard corruption/duplication and straggler behavior are detected and produce actionable evidence.
7. The project reports throughput, step-time distribution, scaling efficiency, peak memory and cost—not decorative scale metrics.

## 4. Domain choice

### Primary data: LargeST traffic flow

LargeST is the preferred dataset because its official repository describes five HDF5 files containing traffic-flow measurements from 2017–2021, sensor metadata and a road-distance adjacency matrix. It is large enough to exercise sharded streaming and naturally connects to prior transportation experience.

The repository must not redistribute LargeST. The user supplies data separately; preprocessing records source, version, file hashes and license/citation metadata.

### Optional evaluation data

METR-LA and PEMS-BAY may be used only for optional speed-forecasting transfer experiments. They are speed datasets in the original DCRNN workflow and do not justify vehicle-count conservation claims.

### Physics/conservation boundary

Traffic flow is physically constrained, but a conservation residual is defensible only when the data and graph encode directed junction topology, boundary flows and compatible count/flow units. A road-distance adjacency matrix alone is insufficient. Conservation evaluation is therefore conditional and out of the minimum viable project unless those inputs are verified.

## 5. Minimum viable evidence

The project is complete enough for interview use only when all of the following exist:

- A public-quality repository with pinned dependencies, CPU CI and data setup instructions.
- A rank-coverage artifact proving exact sample coverage across at least four local ranks.
- Three short control-seed runs defining the statistical variance band.
- A deterministic same-world-size save/resume test with exact equality.
- A distributed-checkpoint load across at least two different world sizes.
- A real two-host run with host identity, rank topology and NCCL evidence.
- At least one injected process/rank failure followed by successful recovery.
- At least one deliberately corrupt or duplicated shard rejected by an invariant.
- A straggler experiment with per-rank step-time evidence.
- A cost ledger tying each cloud experiment to GPU-hours and useful evidence produced.
- An incident report that distinguishes hypothesis, detection, diagnosis, intervention, result and limitations.

## 6. Non-goals

- Training a competitive traffic-forecasting state of the art.
- Training a language model or calling the result a general foundation model.
- Demonstrating frontier-scale compute, InfiniBand performance or large-cluster operations.
- Adding JAX before the PyTorch evidence is complete.
- Tensor, pipeline or context parallelism in the minimum viable project.
- Exact reproducibility for BF16 asynchronous runs or across different collective reduction orders.
- Redistributing third-party datasets.
- Treating an unexecuted configuration, CI simulation or cloud plan as completed evidence.

## 7. Compute strategy

Priority order:

1. **NRP/Nautilus:** free shared research compute if an active UCSD collaborator can add the user to a namespace.
2. **GCP credit:** two same-zone, same-VPC L4 VMs after local gates pass. GCP Free Trial accounts cannot attach GPUs; activating paid billing retains remaining credit for the original 90-day window but enables overage billing.
3. **ACCESS:** pursue an Explore allocation in parallel if advisor collaboration and current eligibility support it.
4. **Out-of-pocket compute:** excluded unless explicitly approved.

The internal GCP target is $120 or less. Availability and GPU quota are risks; quota does not guarantee capacity.

## 8. Success language

After the evidence exists, an acceptable description is:

> Built and operated a portfolio-scale multi-node PyTorch pretraining stack for a spatiotemporal traffic Transformer, including rank-correct sharded loading, FSDP2, distributed checkpoint resharding across world-size changes, failure injection and recovery validated against deterministic and seed-variance oracles.

In conversation, immediately add:

> This was controlled portfolio-scale training, not frontier foundation-model scale.

Do not write this as a completed resume claim until every referenced artifact exists.

## 9. Source anchors

- PyTorch Distributed Checkpoint supports parallel save/load and load-time resharding: <https://docs.pytorch.org/docs/stable/distributed.checkpoint.html>
- TorchTitan is a reference for native PyTorch pretraining patterns: <https://github.com/pytorch/torchtitan>
- LargeST official dataset repository: <https://github.com/liuxu77/LargeST>
- DCRNN dataset/graph reference for METR-LA and PEMS-BAY: <https://github.com/liyaguang/DCRNN>
- Caltrans PeMS source and access: <https://dot.ca.gov/programs/traffic-operations/mpr/pems-source>
- GCP Free Trial limitations: <https://docs.cloud.google.com/free/docs/free-cloud-features>
- NRP/Nautilus documentation: <https://nrp.ai/documentation/>
- ACCESS request preparation: <https://allocations.access-ci.org/prepare-requests>
