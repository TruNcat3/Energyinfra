#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-}"
PRESET="${2:-}"
STAGE="${3:-}"

case "$MODE" in
  serial)
    CONFIG="$ROOT/configs/experiments/thor_edge_confirmation_serial_v1.json"
    RUN_ROOT="$ROOT/data/protocol_v2/thor_edge_confirmation_v1/serial"
    ;;
  bursty)
    CONFIG="$ROOT/configs/experiments/thor_edge_confirmation_bursty_v1.json"
    RUN_ROOT="$ROOT/data/protocol_v2/thor_edge_confirmation_v1/bursty"
    ;;
  *)
    printf 'Usage: bash %s serial|bursty quick|full prepare|run|report\n' "$0" >&2
    exit 2
    ;;
esac

case "$PRESET" in
  quick|full) ;;
  *) printf 'Usage: bash %s serial|bursty quick|full prepare|run|report\n' "$0" >&2; exit 2 ;;
esac

case "$STAGE" in
  prepare|run|report) ;;
  *) printf 'Usage: bash %s serial|bursty quick|full prepare|run|report\n' "$0" >&2; exit 2 ;;
esac

if [[ "$STAGE" == "run" && "$EUID" -ne 0 ]]; then
  printf 'Run the hardware stage with sudo -E.\n' >&2
  exit 2
fi
if [[ "$STAGE" != "run" && "$EUID" -eq 0 ]]; then
  printf '%s must run without sudo.\n' "$STAGE" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_edge_confirmation_${MODE}_${PRESET}_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
RUN_DIR="$RUN_ROOT/$PRESET"
cd "$ROOT"

if [[ "$STAGE" == "run" ]]; then
  COMMAND=(env ENERGYINFRA_MAX_SAMPLE_GAP_S="${ENERGYINFRA_MAX_SAMPLE_GAP_S:-0.5}" "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_joint_serving_stage1b_v1.py "$STAGE" --config "$CONFIG" --preset "$PRESET" --run-dir "$RUN_DIR")
else
  COMMAND=("$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_joint_serving_stage1b_v1.py "$STAGE" --config "$CONFIG" --preset "$PRESET" --run-dir "$RUN_DIR")
fi

nohup "${COMMAND[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
KILL="kill -TERM $PID"
if [[ "$EUID" -eq 0 ]]; then KILL="sudo kill -TERM $PID"; fi
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: %s\n' "$PID" "$LOG" "$LOG" "$KILL"
