#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--worker" ]]; then
  shift
  cd "$ROOT"
  "$ROOT/.venv-thor/bin/python" -u scripts/active/run_energyinfra_v2_repair.py screen "$@"
  exec "$ROOT/.venv-thor/bin/python" -u scripts/active/run_energyinfra_v2_repair.py confirm "$@"
fi
if [[ "${EUID}" -ne 0 ]]; then
  printf 'Run with foreground sudo; background password prompts are not supported.\n' >&2
  exit 1
fi
if [[ $# -eq 0 ]]; then
  printf 'Usage: sudo -E bash %s --parent ABSOLUTE_PATH --run-dir NEW_ABSOLUTE_PATH\n' "$0" >&2
  exit 2
fi
mkdir -p "$ROOT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
LOG="$ROOT/logs/energyinfra_v2_repair_${STAMP}.log"
nohup bash "${BASH_SOURCE[0]}" --worker "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$ROOT/logs/energyinfra_v2_repair_${STAMP}.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %q\nStop: terminate the active Python child first, then this supervisor.\n' "$PID" "$LOG" "$LOG"
