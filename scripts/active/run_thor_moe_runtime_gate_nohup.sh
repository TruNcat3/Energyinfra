#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:-}"
RUN_DIR="${2:-$ROOT/data/protocol_v2/thor_moe_runtime_gate_v1}"

if [[ -z "$MODEL" ]]; then
  printf 'Usage: bash %s /external/path/model.gguf [run-dir]\n' "$0" >&2
  exit 2
fi
if [[ ! -f "$MODEL" ]]; then
  printf 'Model does not exist: %s\n' "$MODEL" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_moe_runtime_gate_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
COMMAND=("$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_moe_runtime_gate_v1.py --model "$MODEL" --run-dir "$RUN_DIR")
nohup "${COMMAND[@]}" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\n' "$PID" "$LOG" "$LOG"
