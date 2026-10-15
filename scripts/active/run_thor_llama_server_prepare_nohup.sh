#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$EUID" -eq 0 ]]; then
  printf 'Run without sudo: bash %s\n' "$0" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_llama_server_prepare_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup bash scripts/active/prepare_thor_llama_server_v1.sh "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: kill -TERM %s\n' \
  "$PID" "$LOG" "$LOG" "$PID"
