#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-${ENERGYINFRA_MOE_MODEL_DIR:-$ROOT/models/moe_v1}}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

stamp="$(date +%Y%m%d_%H%M%S)"
launcher_pid="$$"
log="$LOG_DIR/thor_moe_model_prepare_${stamp}_${launcher_pid}.log"
pid_file="$LOG_DIR/thor_moe_model_prepare_${stamp}_${launcher_pid}.pid"

nohup bash "$ROOT/scripts/active/prepare_thor_moe_model_v1.sh" "$OUTPUT_DIR" \
  >"$log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$pid_file"

printf 'PID: %s\n' "$pid"
printf 'PID file: %s\n' "$pid_file"
printf 'Log: %s\n' "$log"
printf 'Follow: tail -f %s\n' "$log"
printf 'Graceful stop: kill -TERM %s\n' "$pid"
printf 'Output directory: %s\n' "$OUTPUT_DIR"

