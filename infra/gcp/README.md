# Phase C runbook — two-host run on GCP

**STATUS: none of this has been executed.** There is no GCP account behind this
repository yet. The scripts are syntax-checked and the launch path is rehearsed
locally; nothing here is verified against the live API.

## Before you create anything

The account steps are deliberately not automated. Upgrading billing enables charges
beyond the credit, so it is a human decision (`plan.md` C0).

1. Sign up for GCP. **Do not do this until you are ready to run** — the $300 credit
   expires 90 days from *signup*, not from first use.
2. Upgrade the billing account from Free Trial to paid. The Free Trial **cannot
   attach GPUs at all**; upgrading carries the remaining credit over.
3. Request `NVIDIA_L4_GPUS` (or `PREEMPTIBLE_NVIDIA_L4_GPUS`) quota ≥ 2 in one
   region. New paid accounts start at 0. This can take a day and can be refused.
4. Set a budget alert. **Alerts notify; they do not cap.** The enforced caps are
   `--max-run-duration`, the config's step and wall-clock limits, and teardown.
5. Record each answer in `reports/access/gcp_answers.json` by setting its `answer`
   to `true`. An item counts only when the value is exactly `true` — `"yes"`, `1` and
   `null` all leave it unanswered, deliberately. Set it after doing the thing, not
   when you intend to.

Then:

```bash
uv run python scripts/preflight_cloud.py     # must reach ready_to_provision: true
```

## Provision

```bash
export PROJECT=<your-project-id> ZONE=us-central1-a
DRY_RUN=1 bash infra/gcp/provision.sh        # inspect the commands first
bash infra/gcp/provision.sh
bash infra/gcp/firewall.sh
gcloud compute instances list --filter="labels.purpose=pretrainmodel-phase-c"
```

Note the **internal** IP of `ptm-node-0`. That is `MASTER_ADDR`.

## On each node, before launching

```bash
getent hosts "$(hostname)" || echo "127.0.1.1 $(hostname)" | sudo tee -a /etc/hosts
uv run python -c "
from pretrainmodel.distributed.launch import network_preflight
p = network_preflight(); print(p.hostname, p.hostname_resolves, p.problems)"
```

This is not ceremony. A node that cannot resolve its own hostname makes c10d
rendezvous retry DNS **forever** rather than failing — billing the whole time, with
a symptom that points at the firewall instead. See
`reports/incidents/I-007-rendezvous-hostname-hang.md`.

## Launch

Node 0:

```bash
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29500 \
  -m pretrainmodel.distributed.entrypoint --config configs/two_node_l4.toml
```

Node 1: identical, with `--node-rank=1`.

`two_node_l4.toml` sets `expect_distinct_hosts = true`, so the run **aborts before
the first optimizer step** if the ranks turn out to share a host. That check is what
separates a multi-node artifact from a two-process one; do not disable it to get a
run to start.

## Order of experiments — cheapest first

Debug on the cheap thing. Every minute of confusion here costs two GPU-hours.

1. NCCL connectivity smoke test — a few steps, confirm `distinct_hosts=2`.
2. Two-host training baseline (`artifacts/cloud/topology.json`).
3. Strong scaling: 1 GPU, then 2 across hosts, same global workload.
4. Injected rank failure and recovery.
5. Straggler injection.

Do **not** train to convergence here. That was done on CPU; these hours buy
measurements that only real hardware can produce.

## Fallback: two nodes in different regions

If regional L4 quota lands in two *different* regions (say 1 in us-west3 and 1 in
us-west4), a cross-region run is possible. A GCP VPC is a global resource, so
instances in different regions on the same network reach each other over internal
IPs, and NCCL runs over that.

**Check the global cap first.** `GPUS_ALL_REGIONS` limits total concurrent GPUs
across every region. Regional quota of 1 + 1 is worthless if the global quota is 1:

```bash
gcloud compute project-info describe --project=$PROJECT \
  --format="value(quotas.filter(metric:GPUS_ALL_REGIONS))"
```

**What a cross-region run is and is not evidence for:**

| Phase C item | Cross-region? |
|---|---|
| C1 NCCL connectivity, distinct hosts | **Valid** — arguably a stronger boundary than same-zone |
| C3 injected rank failure and recovery | **Valid** — latency does not affect recovery correctness |
| C2 two-host training baseline | Runs, but slowly |
| C2 strong scaling / throughput | **Not valid** — measures the wide-area link, not the code |
| C4 straggler diagnosis | **Confounded** — the WAN itself creates the asymmetry being looked for |

`capture_topology` records each rank's zone and region, and any run spanning regions
sets `throughput_numbers_comparable: false` with an explicit warning in the artifact.
Scaling efficiency from such a run must not be reported as a property of this
implementation.

**Practical differences:**

- Widen the firewall source range to cover both regions' subnets (the default
  auto-mode VPC uses `10.128.0.0/9`, which already spans regions).
- Raise the rendezvous and collective timeouts; the defaults assume datacenter
  latency.
- Consider `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD` to keep a
  high-latency link busy.
- **Inter-region traffic is billed as egress; intra-zone is free.** FSDP2
  communicates roughly twice the model size per step, so keep step counts modest
  and watch the budget.

**Recommendation.** Prefer waiting for quota in one region. Use this path only to
unblock the correctness half (C1, C3) while a same-region request is pending, and
run C2 scaling afterwards in one zone.

## Teardown — always

```bash
bash infra/gcp/teardown.sh
gcloud compute instances list --project=$PROJECT   # verify empty
```

Run it after failed sessions too. The most expensive possible outcome of this
project is a GPU VM left running after a debugging session that went nowhere.

## If a node is preempted

That is a **free organic interruption** and better evidence than the injected one in
I-001a. Do not treat it as a setback:

1. Record the preemption time and the last committed checkpoint step.
2. Re-provision and resume from `latest_committed`.
3. Verify loss continuity and rank coverage after the resume.
4. Write it up as I-001 with `injected: false`.
