#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == '--worker' ]]; then
  shift
  cd "$ROOT"
  "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_attribution_v2.py build "$@"
  exec "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_attribution_v2.py controller-replay "$@"
fi
if [[ $# -eq 0 ]]; then
  printf 'Usage: bash %s --parent PATH --run-dir NEW_PATH\n' "$0" >&2
  exit 2
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_attribution_prepare_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup bash "$ROOT/scripts/active/run_thor_attribution_prepare_nohup.sh" --worker "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %q\nNo hardware frequency changes in preparation.\n' "$PID" "$LOG" "$LOG"
