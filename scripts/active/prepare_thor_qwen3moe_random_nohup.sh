#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

stamp="$(date +%Y%m%d_%H%M%S)"
launcher_pid="$$"
log="$LOG_DIR/thor_qwen3moe_random_prepare_${stamp}_${launcher_pid}.log"
pid_file="$LOG_DIR/thor_qwen3moe_random_prepare_${stamp}_${launcher_pid}.pid"

nohup "$ROOT/.venv-thor/bin/python" -u \
  "$ROOT/scripts/active/prepare_thor_qwen3moe_random_v1.py" "$@" \
  >"$log" 2>&1 < /dev/null &
pid="$!"
printf '%s\n' "$pid" > "$pid_file"

printf 'PID: %s\n' "$pid"
printf 'PID file: %s\n' "$pid_file"
printf 'Log: %s\n' "$log"
printf 'Follow: tail -f %s\n' "$log"
printf 'Graceful stop: kill -TERM %s\n' "$pid"
printf 'No model download or hardware frequency changes are performed.\n'

