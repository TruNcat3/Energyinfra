#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -lt 1 ]]; then
  printf 'Usage: sudo -E bash %s STAGE --run-dir ABSOLUTE_PATH [options]\n' "$0" >&2
  exit 2
fi
if [[ "${EUID}" -ne 0 ]]; then
  printf 'Run this launcher with foreground sudo; no background password prompt is supported.\n' >&2
  exit 1
fi
mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$ROOT/logs/energyinfra_v2_${1}_${STAMP}.log"
PIDFILE="$ROOT/logs/energyinfra_v2_${1}_${STAMP}.pid"
cd "$ROOT"
nohup "$ROOT/.venv-thor/bin/python" -u "$ROOT/scripts/active/run_energyinfra_v2.py" "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$PIDFILE"
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nStop gracefully: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
