# Phase C runbook — two-host GPU run on GCP

**STATUS: never executed.** Scripts are syntax-checked and the launch path is
rehearsed locally; nothing here has run against the live GCP API. Expect to fix
something on first contact.

Copy-paste from top to bottom. Every block is self-contained.

---

## 0. Set your variables

Run this once per terminal session. Everything below depends on it.

```bash
export PROJECT=project-3fa48bd6-2eaf-4084-85a
export ZONE=us-west4-c
export REGION=us-west4
gcloud config set project "$PROJECT"
gcloud auth login          # skip if already authenticated
```

Check you are pointed at the right place:

```bash
gcloud config list
gcloud billing projects describe "$PROJECT"
```

`billingEnabled: true` is required. If it is false, or the account is still a Free
Trial, stop — **the Free Trial cannot attach GPUs at all.** Upgrade first.

---

## 1. Enable the APIs

```bash
gcloud services enable compute.googleapis.com --project="$PROJECT"
```

---

## 2. Check quota — both of them

```bash
# Global cap on concurrent GPUs across ALL regions
gcloud compute project-info describe --project="$PROJECT" \
  --format="table(quotas.filter(metric:GPUS_ALL_REGIONS).metric,
                  quotas.filter(metric:GPUS_ALL_REGIONS).limit)"

# Regional L4 quota (spot uses the PREEMPTIBLE metric)
gcloud compute regions describe "$REGION" --project="$PROJECT" \
  --format="table(quotas.metric,quotas.limit)" | grep -i l4
```

You need **both** `GPUS_ALL_REGIONS ≥ 2` **and** `PREEMPTIBLE_NVIDIA_L4_GPUS ≥ 2` in
one region. The global cap is the one people miss: regional quota of 1 + 1 in two
regions is useless if the global limit is 1.

Request increases at IAM & Admin → Quotas & System Limits.

---

## 3. Prove capacity with one instance

Quota is permission, not inventory. Spot L4 capacity varies by zone and hour, and
GCP has no true dry-run for instance creation — so create one, watch it come up,
delete it. Costs cents.

```bash
gcloud compute instances create cap-test \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type=g2-standard-4 \
  --provisioning-model=SPOT \
  --maintenance-policy=TERMINATE \
  --instance-termination-action=DELETE \
  --max-run-duration=10m

gcloud compute instances describe cap-test --zone="$ZONE" --format="value(status)"
gcloud compute instances delete cap-test --zone="$ZONE" --quiet
```

`RUNNING` means you have capacity. `ZONE_RESOURCE_POOL_EXHAUSTED` means try another
zone (`us-central1-b`, `us-west1-b`, …) and update `ZONE` above.

---

## 4. Run the preflight

```bash
uv run python scripts/preflight_cloud.py
```

Record each manual answer in `reports/access/gcp_answers.json` by setting its
`answer` to **exactly** `true` — `"yes"`, `1` and `null` all leave it unanswered on
purpose. Set it after doing the thing, not when you intend to. Continue only when it
prints `ready_to_provision: True`.

---

## 5. Provision two nodes

```bash
DRY_RUN=1 bash infra/gcp/provision.sh    # inspect the commands first
bash infra/gcp/provision.sh
bash infra/gcp/firewall.sh
```

Startup takes a few minutes: each node installs uv, clones the repo, runs
`uv sync --locked`, and generates the synthetic fixture. Wait for it:

```bash
for i in 0 1; do
  echo "--- ptm-node-$i ---"
  gcloud compute ssh "ptm-node-$i" --zone="$ZONE" --command \
    'while [ ! -f /opt/pretrainmodel/.startup-complete ]; do sleep 10; done; echo ready'
done
```

If that hangs, read the boot log:

```bash
gcloud compute instances get-serial-port-output ptm-node-0 --zone="$ZONE" | tail -40
```

---

## 6. Get the master's internal IP

```bash
export MASTER_ADDR=$(gcloud compute instances describe ptm-node-0 --zone="$ZONE" \
  --format="value(networkInterfaces[0].networkIP)")
echo "MASTER_ADDR=$MASTER_ADDR"
```

Use the **internal** IP. The external one routes over the public internet and is
firewalled off.

---

## 7. Launch

Two terminals, one per node.

**Node 0:**

```bash
gcloud compute ssh ptm-node-0 --zone="$ZONE"
cd /opt/pretrainmodel
export MASTER_ADDR=<paste from step 6>
uv run torchrun --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  -m pretrainmodel.distributed.entrypoint \
  --config configs/two_node_synthetic.toml --resume
```

**Node 1:** identical, with `--node-rank=1`.

Node 0 waits for node 1, so start both within a minute or two.

The run **aborts before the first optimizer step** if the ranks turn out to share a
host, or if the two nodes hold different data. Do not disable those checks to get a
run to start — they are the difference between multi-node evidence and a two-process
run that looks like one.

---

## 8. Order of experiments — cheapest first

Every minute of confusion here costs two GPU-hours. Debug on the cheap thing.

1. **Connectivity smoke test** — the launch above with `max_steps` cut to ~20.
   Confirm the output says `distinct_hosts=2, multi_node=True`.
2. **Two-host baseline** — the full `two_node_synthetic.toml`.
3. **Strong scaling** — same global workload at 1 GPU, then 2 across hosts.
4. **Injected rank failure** — `kill` the process on node 1 mid-run, then relaunch
   both with `--resume` and confirm loss continuity.
5. **Straggler** — the per-rank medians are already collected; `timing.json` reports
   the ratio.

Do **not** train to convergence. That was done on CPU. These hours buy measurements
only real hardware can produce.

---

## 9. Collect the artifacts before teardown

```bash
gcloud compute scp --recurse --zone="$ZONE" \
  ptm-node-0:/opt/pretrainmodel/runs ./runs-cloud
```

The VMs are deleted in the next step and take everything with them.

---

## 10. Teardown — always

```bash
bash infra/gcp/teardown.sh
gcloud compute instances list --project="$PROJECT"    # must be empty
```

Run it after failed sessions too. **The most expensive outcome of this project is a
GPU VM left running after a debugging session that went nowhere.**

---

## Troubleshooting

**The job hangs at startup with no output.** The node cannot resolve its own
hostname, so c10d retries DNS forever instead of failing — billing the whole time.
`startup.sh` fixes this at boot; to check by hand:

```bash
getent hosts "$(hostname)" || echo "127.0.1.1 $(hostname)" | sudo tee -a /etc/hosts
```

Full write-up: `reports/incidents/I-007-rendezvous-hostname-hang.md`.

**`ranks disagree about the dataset`.** The nodes generated different fixtures.
Regenerate on both from the same seed, or copy one node's `data/` to the other.

**A node was preempted.** That is a **free organic interruption** and better evidence
than the injected one in I-001a. Do not treat it as a setback: record the preemption
time and last committed checkpoint step, re-provision, relaunch with `--resume`,
verify loss continuity and rank coverage, and write it up as I-001 with
`injected: false`.

**`ZONE_RESOURCE_POOL_EXHAUSTED` mid-run.** Spot capacity vanished. Same as
preemption — resume from the last checkpoint.

---

## Fallback: two nodes in different regions

Possible if quota lands in two different regions. A GCP VPC is global, so instances
in different regions on the same network reach each other over internal IPs.

**Check the global cap first** (step 2) — 1 + 1 regional quota is worthless if
`GPUS_ALL_REGIONS` is 1.

| Phase C item | Cross-region? |
|---|---|
| Connectivity, distinct hosts | **Valid** — a stronger boundary than same-zone |
| Injected failure and recovery | **Valid** — latency does not affect correctness |
| Two-host training baseline | Runs, but slowly |
| Strong scaling / throughput | **Not valid** — measures the wide-area link |
| Straggler diagnosis | **Confounded** — the WAN creates the asymmetry itself |

`capture_topology` records each rank's zone and region. Any run spanning regions
sets `throughput_numbers_comparable: false` and writes an explicit warning into the
artifact, so a WAN-bound number cannot later be read as a property of the code.

Practical differences: raise rendezvous and collective timeouts (defaults assume
datacenter latency); consider `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD`; and
note that **inter-region traffic is billed as egress while intra-zone is free** —
FSDP2 moves roughly twice the model size per step, so keep step counts modest.

**Recommendation:** wait for quota in one region. Use this path only to unblock the
correctness half while a same-region request is pending, and run scaling later in
one zone.
