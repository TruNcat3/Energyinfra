#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PRESET="${1:-}"
if [[ "$PRESET" != "quick" && "$PRESET" != "full" ]]; then
  printf 'Usage: bash %s quick|full [--run-dir PATH] [--config PATH]\n' "$0" >&2
  exit 2
fi
shift
if [[ "$EUID" -eq 0 ]]; then
  printf 'Run the resource screen without sudo; it does not change hardware frequencies.\n' >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_resource_screen_${PRESET}_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_resource_screen_v1.py run \
  --preset "$PRESET" "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: kill -TERM %s\n' \
  "$PID" "$LOG" "$LOG" "$PID"
