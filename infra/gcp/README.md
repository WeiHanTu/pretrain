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
5. Record the answers in `reports/access/compute_options.md` and flip the
   corresponding entries in `scripts/preflight_cloud.py`.

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
