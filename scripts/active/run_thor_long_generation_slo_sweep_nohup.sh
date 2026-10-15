#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/active/run_thor_long_generation_slo_sweep_nohup.sh"
PY="$ROOT/.venv-thor/bin/python"
RUNNER="$ROOT/scripts/active/run_thor_fixed_trace_serving_v1.py"
CONFIGS=(
  "$ROOT/configs/experiments/thor_q4_long_generation_fixed_trace_2s75_v1.json"
  "$ROOT/configs/experiments/thor_q4_long_generation_fixed_trace_2s100_v1.json"
)
OUTPUTS=(
  "$ROOT/data/protocol_v2/thor_q4_long_generation_fixed_trace_2s75_v1"
  "$ROOT/data/protocol_v2/thor_q4_long_generation_fixed_trace_2s100_v1"
)
WORKER=0
if [[ "${1:-}" == "--worker" ]]; then WORKER=1; shift; fi
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi

case "$STAGE" in
  run|report|restore) ;;
  *) printf 'Usage: %s run|report|restore\n' "$SCRIPT" >&2; exit 2 ;;
esac

cd "$ROOT"
export PYTHONUNBUFFERED=1

run_stage() {
  local index
  for index in "${!CONFIGS[@]}"; do
    if [[ "$STAGE" == "restore" ]]; then
      if [[ ! -f "${OUTPUTS[$index]}/restore_pending.json" ]] ||
         ! rg -q '"pending"[[:space:]]*:[[:space:]]*true' "${OUTPUTS[$index]}/restore_pending.json"; then
        continue
      fi
    fi
    printf '{"sweep_event":"profile_start","config":"%s","stage":"%s"}\n' "${CONFIGS[$index]}" "$STAGE"
    "$PY" -u "$RUNNER" "$STAGE" --config "${CONFIGS[$index]}"
    printf '{"sweep_event":"profile_complete","config":"%s","stage":"%s"}\n' "${CONFIGS[$index]}" "$STAGE"
  done
}

if [[ $WORKER -eq 1 ]]; then
  run_stage
  exit 0
fi

if [[ "$STAGE" == "report" ]]; then
  run_stage
  exit 0
fi

if [[ $EUID -ne 0 ]]; then
  printf '%s\n' 'Run hardware stages with sudo -E; nohup never requests a password.' >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_long_generation_slo_sweep_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$SCRIPT" --worker "$STAGE" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$LOG.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %s\nGraceful stop: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
