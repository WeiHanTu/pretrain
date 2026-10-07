#!/usr/bin/env bash
# Provision two single-GPU VMs in one zone and VPC for the Phase C two-host run.
#
# STATUS: WRITTEN, NEVER EXECUTED. There is no GCP account behind this repository
# yet, so every flag below is unverified against the live API. Run with DRY_RUN=1
# first and reconcile against `gcloud compute instances create --help`.
#
# Safety properties, in order of how much money they save:
#
#   --max-run-duration + --instance-termination-action=DELETE
#       GCP deletes the VM after the deadline whether or not the guest is healthy,
#       whether or not you still have a terminal open, and whether or not you
#       remembered. A guest-side `shutdown -h +N` does none of that: it dies with
#       the guest, so the one case it is meant to cover -- a hung job on a machine
#       you forgot -- is exactly the case it misses.
#
#   --provisioning-model=SPOT
#       Roughly a third of on-demand. Preemption is not a downside for this
#       project: a preempted run is a FREE organic interruption to recover from,
#       which is stronger evidence than the injected one, provided checkpoints are
#       frequent enough to make the recovery cheap.
#
#   labels
#       Teardown is by label, so `teardown.sh` cannot miss a VM whose name was
#       typed differently.
set -euo pipefail

PROJECT="${PROJECT:?set PROJECT to your GCP project id}"
ZONE="${ZONE:-us-west4-c}"
MACHINE="${MACHINE:-g2-standard-4}"      # 1x L4
COUNT="${COUNT:-2}"
PREFIX="${PREFIX:-ptm-node}"
MAX_RUN="${MAX_RUN:-6h}"                 # hard deadline enforced by GCP
DISK_GB="${DISK_GB:-100}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu124-ubuntu-2204-py310}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
LABEL="purpose=pretrainmodel-phase-c"
DRY_RUN="${DRY_RUN:-0}"
# Absolute, so the script works from any working directory.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STARTUP="${REPO_ROOT}/infra/gcp/startup.sh"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then printf '[dry-run] %q ' "$@"; echo; else "$@"; fi
}

echo "project=$PROJECT zone=$ZONE machine=$MACHINE count=$COUNT max_run=$MAX_RUN"
echo "Spot instances: expect preemption. That is a feature here; see the runbook."
echo

for i in $(seq 0 $((COUNT - 1))); do
  run gcloud compute instances create "${PREFIX}-${i}" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE" \
    --provisioning-model=SPOT \
    --maintenance-policy=TERMINATE \
    --instance-termination-action=DELETE \
    --max-run-duration="$MAX_RUN" \
    --image-family="$IMAGE_FAMILY" \
    --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="${DISK_GB}GB" \
    --boot-disk-type=pd-balanced \
    --labels="$LABEL" \
    --metadata-from-file=startup-script="$STARTUP" \
    --scopes=storage-ro
done

echo
echo "Next:"
echo "  1. infra/gcp/firewall.sh          # open the rendezvous port inside the VPC"
echo "  2. gcloud compute instances list --filter=\"labels.purpose=pretrainmodel-phase-c\""
echo "  3. Note the INTERNAL ip of ${PREFIX}-0; it is MASTER_ADDR."
echo "  4. Run the hostname preflight on BOTH nodes before launching (see runbook)."
echo "  5. infra/gcp/teardown.sh          # ALWAYS, even if the run failed"
