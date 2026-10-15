#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/active/run_thor_q4_frontier_nohup.sh"
PY="$ROOT/.venv-thor/bin/python"
CONFIG="configs/experiments/thor_q4_full_frontier_v1.json"
RUN_DIR="data/protocol_v2/thor_q4_full_frontier_v1"
WORKER=0
if [[ "${1:-}" == "--worker" ]]; then WORKER=1; shift; fi
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in
  plan|run|profile|select|validate|report|serve|serve-report|restore) ;;
  *) printf 'Usage: sudo -E bash %s plan|run|profile|select|validate|report|serve|serve-report|restore [runner arguments]\n' "$SCRIPT" >&2; exit 2 ;;
esac
cd "$ROOT"
export PYTHONUNBUFFERED=1
if [[ $WORKER -eq 1 ]]; then
  exec "$PY" -u scripts/active/run_cpu_floor_v1.py "$STAGE" \
    --config "$CONFIG" --preset full --run-dir "$RUN_DIR" "$@"
fi
if [[ $EUID -ne 0 && "$STAGE" != plan && "$STAGE" != select && "$STAGE" != report && "$STAGE" != serve-report ]]; then
  printf '%s\n' 'Run hardware stages with sudo -E; nohup never requests a password.' >&2
  exit 1
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_q4_frontier_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$SCRIPT" --worker "$STAGE" "$@" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$LOG.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %s\nGraceful stop: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
