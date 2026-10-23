#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/wt/energyinfra/Energyinfra-main"
cd "$ROOT"

PRESET="${1:-quick}"
STAGE="${2:-run}"
RUN_DIR="${3:-data/protocol_v2/thor_moe_workload_hybrid_v3/${PRESET}}"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$ROOT/logs/thor_moe_workload_hybrid_v3_${PRESET}_${STAGE}_${STAMP}.log"
mkdir -p "$ROOT/logs"

nohup "$ROOT/.venv-thor/bin/python" \
  "$ROOT/scripts/active/run_thor_moe_phase_aware_serving_v3.py" \
  "$STAGE" --preset "$PRESET" --run-dir "$RUN_DIR" \
  >"$LOG" 2>&1 < /dev/null &
PID=$!

echo "PID: $PID"
echo "Log: $LOG"
echo "Follow: tail -f $LOG"
echo "Graceful stop: sudo kill -TERM $PID"
