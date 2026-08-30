#!/usr/bin/env bash
# VM startup script. STATUS: WRITTEN, NEVER EXECUTED.
#
# The hostname line is not boilerplate. A node that cannot resolve its own
# hostname makes c10d rendezvous retry DNS forever instead of failing, which on a
# billed instance is an invisible, open-ended cost. See reports/incidents/I-007.
set -euo pipefail

if ! getent hosts "$(hostname)" >/dev/null 2>&1; then
  echo "127.0.1.1 $(hostname)" >> /etc/hosts
  echo "startup: added $(hostname) to /etc/hosts (c10d rendezvous would otherwise hang)"
fi

curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> /etc/profile.d/uv.sh
