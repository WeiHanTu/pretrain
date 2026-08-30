# Compute access — decisions and open constraints

**Last updated:** 2026-08-30
**Decision:** GCP, using the $300 first-time credit. NRP and ACCESS are ruled out.

## Options considered

| Option | Cost | Multi-node? | Status |
|---|---|---|---|
| **GCP** ($300 first-time credit) | ~$5–20 for the core run | Yes — two VMs, one zone and VPC | **Chosen** |
| NRP / Nautilus | Free | Yes | **Ruled out** — students are added to a namespace by their supervisor; author has graduated and does not wish to ask his former advisor |
| ACCESS Explore | Free | Yes | **Ruled out** — a graduate student PI needs an advisor letter of collaboration; same constraint |
| RunPod Instant Clusters | ~$28/hr | Yes | Rejected — minimum 2 nodes × 8 GPUs (16 GPUs), H100/H200/B200/A100 only. No cheap 2×1 option exists |
| Kaggle / Colab free GPU | Free | **No** | Single node only; cannot produce multi-node evidence |

The RunPod line is worth keeping: an earlier plan for this project assumed "rent two
cheap one-GPU nodes for $5–20," which is not a product that exists. The minimum
purchasable multi-node configuration there is ~10× that estimate.

## Why GCP specifically

Two VMs in the same zone and VPC share a real private network, so the rendezvous
crosses a genuine host boundary with routable internal IPs and no overlay. That is
the cheapest configuration that produces honest multi-node evidence.

## Hard constraints on the GCP path

1. **The Free Trial cannot attach GPUs at all.** Billing must be upgraded to paid
   first. Remaining credit carries over — but see (2).
2. **The credit expires 90 days from *signup*, not from first use.** Do not create
   the account until the code is ready to run, or the window is spent on
   development. (Phase A and B were both completed before signup for this reason.)
3. **GPU quota starts at 0** on a new paid account and must be requested
   separately. Approval takes time and is sometimes refused.
4. **Quota is permission, not inventory.** Spot L4 capacity varies by zone and hour.
5. **Budget alerts notify; they do not cap.** Nothing in GCP halts spend on its own.

## Enforced spending controls

Alerts are not controls. What actually bounds cost here:

- `--max-run-duration` with `--instance-termination-action=DELETE` — enforced by
  GCP, so it survives a hung guest and an unattended session.
- `train.max_steps` and `train.max_wall_seconds` in every config.
- `infra/gcp/teardown.sh`, run unconditionally after every session including
  failed ones.
- Spot provisioning, at roughly a third of on-demand.

## Spot preemption is wanted here

Preemption is normally a cost of spot pricing. For this project it is a benefit: a
preempted run is a **free, organic interruption** to recover from, which is stronger
evidence than the injected failure in I-001a. It only pays off if the checkpoint
cadence is short enough that recovery is cheap, which is why
`checkpoint.every_steps` is a preflight check rather than a preference.

## Open items

Tracked as UNANSWERED in `artifacts/cloud/preflight.json`; the preflight gate cannot
go green until each is answered by a human:

- [ ] Billing upgraded from Free Trial to paid
- [ ] `NVIDIA_L4_GPUS` (or preemptible equivalent) quota ≥ 2 in the target region
- [ ] Dry-run instance create succeeds in the target zone (capacity, not quota)
- [ ] Budget alert configured on the billing account
- [ ] Credit expiry date recorded

## Estimated cost of the Phase C core run

Two `g2-standard-4` spot instances (1× L4 each) for roughly 4–6 hours covers the
NCCL smoke test, the two-host training baseline, strong-scaling measurement, an
injected rank failure and a straggler experiment. **Prices change and none of the
figures below were quoted from a live billing account** — verify against the pricing
calculator before provisioning. On pre-check estimates this lands well inside the
$120 internal target from `intend.md` §7, and inside the $300 credit with margin.
