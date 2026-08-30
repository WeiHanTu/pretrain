#!/usr/bin/env bash
# Delete everything this project created, by label. Idempotent.
# STATUS: WRITTEN, NEVER EXECUTED.
#
# Run this even when the run failed, especially when the run failed. The most
# expensive possible outcome of this project is a GPU VM left running after a
# debugging session that went nowhere.
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
ZONE="${ZONE:-us-central1-a}"
LABEL="${LABEL:-purpose=pretrainmodel-phase-c}"

mapfile -t VMS < <(gcloud compute instances list \
  --project="$PROJECT" --filter="labels.${LABEL/=/:}" --format="value(name)")

if [[ ${#VMS[@]} -eq 0 ]]; then
  echo "no instances with label ${LABEL}"
else
  echo "deleting: ${VMS[*]}"
  gcloud compute instances delete "${VMS[@]}" --project="$PROJECT" --zone="$ZONE" --quiet
fi

gcloud compute firewall-rules delete ptm-rendezvous --project="$PROJECT" --quiet 2>/dev/null || true
echo "teardown complete. Verify with:"
echo "  gcloud compute instances list --project=$PROJECT"
