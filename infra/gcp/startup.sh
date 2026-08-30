#!/usr/bin/env bash
# VM startup script. STATUS: WRITTEN, NEVER EXECUTED against GCP.
#
# Runs as root at first boot. Output goes to the serial console and to
# /var/log/syslog; check it with:
#   gcloud compute instances get-serial-port-output <name> --zone=<zone>
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/WeiHanTu/pretrain.git}"
TARGET="/opt/pretrainmodel"

# A node that cannot resolve its own hostname makes c10d rendezvous retry DNS
# FOREVER rather than failing, billing the whole time with a symptom that points at
# the firewall. See reports/incidents/I-007-rendezvous-hostname-hang.md.
if ! getent hosts "$(hostname)" >/dev/null 2>&1; then
  echo "127.0.1.1 $(hostname)" >> /etc/hosts
  echo "startup: added $(hostname) to /etc/hosts"
fi

export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
echo 'export PATH="/root/.local/bin:$PATH"' > /etc/profile.d/uv.sh

if [[ ! -d "$TARGET" ]]; then
  git clone --depth 1 "$REPO_URL" "$TARGET"
fi
cd "$TARGET"

# The image ships Python 3.10; this project pins >=3.12, so uv fetches its own.
# The default PyPI torch wheel carries CUDA, which is what is wanted here.
uv sync --locked

# Phase C measures the distributed system, not model quality, so it runs on the
# synthetic fixture. Generated identically on every node from a fixed seed; the
# cross-rank content-hash check verifies the nodes actually agree.
uv run pretrainmodel make-fixture --config configs/two_node_synthetic.toml \
  --shards 24 --timesteps 256

touch /opt/pretrainmodel/.startup-complete
echo "startup: ready"
