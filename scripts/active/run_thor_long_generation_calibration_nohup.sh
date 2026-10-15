#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/active/run_thor_long_generation_calibration_nohup.sh"
PY="$ROOT/.venv-thor/bin/python"
RUNNER="$ROOT/scripts/active/run_thor_serving_load_calibration_v1.py"
CONFIG="$ROOT/configs/experiments/thor_q4_long_generation_load_calibration_v1.json"
WORKER=0
if [[ "${1:-}" == "--worker" ]]; then WORKER=1; shift; fi
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

case "$STAGE" in
  coarse|refine|confirm|report|restore) ;;
  *) printf 'Usage: %s coarse|refine|confirm|report|restore [--phase coarse|refine|confirm] [--scale N]\n' "$SCRIPT" >&2; exit 2 ;;
esac

cd "$ROOT"
export PYTHONUNBUFFERED=1

if [[ "$STAGE" == "report" ]]; then
  exec "$PY" -u "$RUNNER" report --config "$CONFIG" "$@"
fi

if [[ $WORKER -eq 1 ]]; then
  case "$STAGE" in
    coarse) exec "$PY" -u "$RUNNER" run --config "$CONFIG" --phase coarse "$@" ;;
    refine) exec "$PY" -u "$RUNNER" run --config "$CONFIG" --phase refine "$@" ;;
    confirm) exec "$PY" -u "$RUNNER" run --config "$CONFIG" --phase confirm "$@" ;;
    restore) exec "$PY" -u "$RUNNER" restore --config "$CONFIG" "$@" ;;
  esac
fi

if [[ $EUID -ne 0 ]]; then
  printf '%s\n' 'Run hardware stages with sudo -E; nohup never requests a password.' >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_long_generation_calibration_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$SCRIPT" --worker "$STAGE" "$@" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$LOG.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %s\nGraceful stop: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
