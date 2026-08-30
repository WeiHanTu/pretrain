#!/usr/bin/env bash
# Allow rendezvous and collective traffic between the nodes, and nothing else.
# STATUS: WRITTEN, NEVER EXECUTED.
#
# Scoped to the VPC's internal range on purpose. A rule that opens the rendezvous
# port to 0.0.0.0/0 would work just as well for the run and expose an unauthenticated
# store to the internet.
set -euo pipefail
PROJECT="${PROJECT:?set PROJECT}"
NETWORK="${NETWORK:-default}"
INTERNAL_CIDR="${INTERNAL_CIDR:-10.128.0.0/9}"
PORTS="${PORTS:-tcp:29400-29600}"

if gcloud compute firewall-rules describe ptm-rendezvous \
     --project="$PROJECT" >/dev/null 2>&1; then
  echo "firewall rule ptm-rendezvous already exists; leaving it alone"
  exit 0
fi

gcloud compute firewall-rules create ptm-rendezvous \
  --project="$PROJECT" \
  --network="$NETWORK" \
  --direction=INGRESS \
  --action=ALLOW \
  --rules="$PORTS" \
  --source-ranges="$INTERNAL_CIDR" \
  --description="pretrainmodel Phase C: torchrun rendezvous + collectives, VPC-internal only"
