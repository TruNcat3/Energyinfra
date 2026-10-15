#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/active/run_thor_bursty_fixed_trace_nohup.sh"
PY="$ROOT/.venv-thor/bin/python"
PREPARE="$ROOT/scripts/active/prepare_thor_bursty_trace_v1.py"
RUNNER="$ROOT/scripts/active/run_thor_fixed_trace_serving_v1.py"
CONFIG="$ROOT/configs/experiments/thor_q4_bursty_fixed_trace_v1.json"
WORKER=0
if [[ "${1:-}" == "--worker" ]]; then WORKER=1; shift; fi
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

case "$STAGE" in
  prepare|run|report|restore) ;;
  *) printf 'Usage: %s prepare|run|report|restore\n' "$SCRIPT" >&2; exit 2 ;;
esac

cd "$ROOT"
export PYTHONUNBUFFERED=1

if [[ "$STAGE" == "prepare" ]]; then
  exec "$PY" -u "$PREPARE" --config "$CONFIG"
fi
if [[ "$STAGE" == "report" ]]; then
  exec "$PY" -u "$RUNNER" report --config "$CONFIG"
fi
if [[ $WORKER -eq 1 ]]; then
  exec "$PY" -u "$RUNNER" "$STAGE" --config "$CONFIG"
fi
if [[ $EUID -ne 0 ]]; then
  printf '%s\n' 'Run hardware stages with sudo -E; nohup never requests a password.' >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_bursty_fixed_trace_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$SCRIPT" --worker "$STAGE" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$LOG.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %s\nGraceful stop: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
