#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$EUID" -ne 0 || $# -lt 2 ]]; then
  printf 'Usage: sudo -E bash %s run|profile|select|validate|report|serve|serve-report --run-dir PATH [options]\n' "$0" >&2
  exit 2
fi
STAGE="$1"
shift
case "$STAGE" in run|profile|select|validate|report|serve|serve-report) ;; *) printf 'Unknown stage\n' >&2; exit 2;; esac
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/cpu_floor_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup "$ROOT/.venv-thor/bin/python" -u scripts/active/run_cpu_floor_v1.py "$STAGE" "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
