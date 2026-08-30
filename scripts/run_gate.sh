#!/usr/bin/env bash
# Run the standard verification gate and capture its output as evidence.
#
# Usage: scripts/run_gate.sh <gate-name>
# Writes artifacts/gates/<gate-name>.txt containing the environment, the exact
# commands run and their output.  plan.md treats a passing command without
# captured output as not-yet-evidence, so this script is how a phase is closed.

set -uo pipefail
name="${1:?usage: run_gate.sh <gate-name>}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
out="artifacts/gates/${name}.txt"
mkdir -p "$(dirname "$out")"

# Capture git state BEFORE opening the output file. The shell truncates "$out" at
# redirection time, so reading git status inside the block would observe this
# script's own side effect and report a dirty tree on every run -- a gate that
# cannot see a clean checkout is worthless as evidence.
git_commit="$(git rev-parse HEAD 2>/dev/null || echo '(no commit yet)')"
git_dirty="$([ -n "$(git status --porcelain 2>/dev/null)" ] && echo true || echo false)"

{
  echo "gate:        ${name}"
  echo "captured_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git_commit:  ${git_commit}"
  echo "git_dirty:   ${git_dirty}"
  echo "dirty_note:  tree state observed before this file was written"
  echo "python:      $(uv run python -c 'import sys;print(sys.version.split()[0])' 2>/dev/null)"
  echo "torch:       $(uv run python -c 'import torch;print(torch.__version__)' 2>/dev/null)"
  echo "platform:    $(uname -sm)"
  echo
} > "$out"

status=0
for cmd in \
  "uv sync --locked" \
  "uv run ruff check ." \
  "uv run ruff format --check ." \
  "uv run mypy src" \
  "uv run pytest -q" \
  "uv run python scripts/check_repo_hygiene.py"
do
  echo "\$ ${cmd}" >> "$out"
  if ! eval "${cmd}" >> "$out" 2>&1; then
    echo "[FAILED: ${cmd}]" >> "$out"
    status=1
  fi
  echo >> "$out"
done

echo "result: $([ $status -eq 0 ] && echo PASS || echo FAIL)" >> "$out"
echo "wrote ${out} ($([ $status -eq 0 ] && echo PASS || echo FAIL))"
exit $status
