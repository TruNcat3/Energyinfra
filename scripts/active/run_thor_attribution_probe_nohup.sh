#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${EUID}" -ne 0 || $# -eq 0 ]]; then
  printf 'Usage: sudo -E bash %s --parent PATH --run-dir PATH\n' "$0" >&2
  exit 2
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_attribution_probe_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_attribution_v2.py model-probe "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
