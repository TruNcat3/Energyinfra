#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${EUID}" -ne 0 || $# -eq 0 ]]; then
  printf 'Usage: sudo -E bash %s --parent PATH --repair PATH --run-dir NEW_PATH [--trace short_chat --slo 50 --scenario interactive_1s]\n' "$0" >&2
  exit 2
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/energyinfra_v2_serving_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup "$ROOT/.venv-thor/bin/python" -u scripts/active/run_energyinfra_v2_serving.py run "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop (forwards to active experiment): sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
