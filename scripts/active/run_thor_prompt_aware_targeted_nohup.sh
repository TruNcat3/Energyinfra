#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SCENARIO="${1:-}"
STAGE="${2:-}"

case "$SCENARIO" in
  capacity_upper)
    RUN_DIR="$ROOT/data/protocol_v2/thor_slo_capacity_frontier_v1/coupled_capacity_upper_v1"
    TARGET_CONFIG="$ROOT/configs/experiments/thor_prompt_aware_targeted_v1.json"
    ;;
  capacity_upper2)
    RUN_DIR="$ROOT/data/protocol_v2/thor_slo_capacity_frontier_v1/coupled_capacity_upper2_v1"
    TARGET_CONFIG="$ROOT/configs/experiments/thor_prompt_aware_capacity_upper2_v1.json"
    ;;
  capacity_upper3)
    RUN_DIR="$ROOT/data/protocol_v2/thor_slo_capacity_frontier_v1/coupled_capacity_upper3_v1"
    TARGET_CONFIG="$ROOT/configs/experiments/thor_prompt_aware_capacity_upper3_v1.json"
    ;;
  tpot30_stability)
    RUN_DIR="$ROOT/data/protocol_v2/thor_slo_capacity_frontier_v1/coupled_tpot30_stability_v1"
    TARGET_CONFIG="$ROOT/configs/experiments/thor_prompt_aware_targeted_v1.json"
    ;;
  *)
    printf 'Usage: bash %s capacity_upper|capacity_upper2|capacity_upper3|tpot30_stability prepare|run|report|analyze\n' "$0" >&2
    exit 2
    ;;
esac

if [[ "$STAGE" != "prepare" && "$STAGE" != "run" && "$STAGE" != "report" && "$STAGE" != "analyze" ]]; then
  printf 'Usage: bash %s capacity_upper|capacity_upper2|capacity_upper3|tpot30_stability prepare|run|report|analyze\n' "$0" >&2
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
LOG="$ROOT/logs/thor_prompt_aware_targeted_${SCENARIO}_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"

if [[ "$STAGE" == "prepare" ]]; then
  COMMAND=("$ROOT/.venv-thor/bin/python" -u scripts/active/prepare_thor_prompt_aware_targeted_v1.py --config "$TARGET_CONFIG" --scenario "$SCENARIO")
elif [[ "$STAGE" == "analyze" ]]; then
  if [[ "$SCENARIO" == capacity_upper* ]]; then
    CONFIGS=("$RUN_DIR/analysis_prompt_bucket.json")
  else
    CONFIGS=("$RUN_DIR/analysis_overall_primary.json" "$RUN_DIR/analysis_prompt_bucket_diagnostic.json")
  fi
  {
    for config in "${CONFIGS[@]}"; do
      "$ROOT/.venv-thor/bin/python" -u scripts/active/analyze_thor_prompt_aware_slo_v1.py --config "$config"
    done
  } >"$LOG" 2>&1 < /dev/null &
  PID=$!
  printf '%s\n' "$PID" > "${LOG%.log}.pid"
  printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
  exit 0
else
  COMMAND=("$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_joint_serving_stage1b_v1.py "$STAGE" --config "$RUN_DIR/experiment.json" --preset full --run-dir "$RUN_DIR")
fi

nohup "${COMMAND[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
KILL="kill -TERM $PID"
if [[ "$EUID" -eq 0 ]]; then KILL="sudo kill -TERM $PID"; fi
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: %s\n' "$PID" "$LOG" "$LOG" "$KILL"
