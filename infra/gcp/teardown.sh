#!/usr/bin/env bash
# Delete everything this project created, by label. Idempotent.
# STATUS: WRITTEN, NEVER EXECUTED.
#
# Run this even when the run failed, especially when the run failed. The most
# expensive possible outcome of this project is a GPU VM left running after a
# debugging session that went nowhere.
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
LABEL="${LABEL:-purpose=pretrainmodel-phase-c}"

# List name AND zone. Deleting with a single --zone would fail for any instance
# outside it -- exactly what happens with the cross-region fallback -- and a failed
# teardown leaves GPUs billing.
mapfile -t ROWS < <(gcloud compute instances list \
  --project="$PROJECT" --filter="labels.${LABEL/=/:}" \
  --format="value(name,zone)")

if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "no instances with label ${LABEL}"
else
  for row in "${ROWS[@]}"; do
    name="$(awk '{print $1}' <<<"$row")"
    zone="$(awk '{print $2}' <<<"$row")"
    echo "deleting ${name} in ${zone}"
    gcloud compute instances delete "$name" --project="$PROJECT" --zone="$zone" --quiet
  done
fi

gcloud compute firewall-rules delete ptm-rendezvous --project="$PROJECT" --quiet 2>/dev/null || true
echo "teardown complete. Verify with:"
echo "  gcloud compute instances list --project=$PROJECT"
