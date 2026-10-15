#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PRESET="${1:-}"
STAGE="${2:-}"
EXPERIMENT="$ROOT/configs/experiments/thor_edge_low_concurrency_v1.json"

case "$PRESET" in
  quick|full)
    RUN_DIR="$ROOT/data/protocol_v2/thor_edge_low_concurrency_v1/$PRESET"
    ANALYSIS="$ROOT/configs/experiments/thor_edge_low_concurrency_${PRESET}_slo_v1.json"
    ;;
  *)
    printf 'Usage: bash %s quick|full prepare|run|report|analyze\n' "$0" >&2
    exit 2
    ;;
esac

if [[ "$STAGE" != "prepare" && "$STAGE" != "run" && "$STAGE" != "report" && "$STAGE" != "analyze" ]]; then
  printf 'Usage: bash %s quick|full prepare|run|report|analyze\n' "$0" >&2
  exit 2
fi
if [[ "$STAGE" == "run" ]]; then
  if [[ "$EUID" -ne 0 ]]; then
    printf 'Run with sudo -E so hardware state can be controlled and restored.\n' >&2
    exit 2
  fi
elif [[ "$EUID" -eq 0 ]]; then
  printf '%s must run without sudo.\n' "$STAGE" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_edge_low_concurrency_${PRESET}_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"

if [[ "$STAGE" == "analyze" ]]; then
  COMMAND=("$ROOT/.venv-thor/bin/python" -u scripts/active/analyze_thor_prompt_aware_slo_v1.py --config "$ANALYSIS")
else
  if [[ "$STAGE" == "run" ]]; then
    COMMAND=(env ENERGYINFRA_MAX_SAMPLE_GAP_S="${ENERGYINFRA_MAX_SAMPLE_GAP_S:-0.5}" "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_joint_serving_stage1b_v1.py "$STAGE" --config "$EXPERIMENT" --preset "$PRESET" --run-dir "$RUN_DIR")
  else
    COMMAND=("$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_joint_serving_stage1b_v1.py "$STAGE" --config "$EXPERIMENT" --preset "$PRESET" --run-dir "$RUN_DIR")
  fi
fi

nohup "${COMMAND[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
KILL="kill -TERM $PID"
if [[ "$EUID" -eq 0 ]]; then KILL="sudo kill -TERM $PID"; fi
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: %s\n' "$PID" "$LOG" "$LOG" "$KILL"
