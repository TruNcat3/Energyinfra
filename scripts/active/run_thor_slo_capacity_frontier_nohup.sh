#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-}"
PHASE="${2:-pilot}"
if [[ "$STAGE" != "prepare" && "$STAGE" != "run" && "$STAGE" != "report" ]]; then
  printf 'Usage: bash %s prepare|run|report pilot|refine|confirm|strict_confirm\n' "$0" >&2
  exit 2
fi
if [[ "$PHASE" != "pilot" && "$PHASE" != "refine" && "$PHASE" != "confirm" && "$PHASE" != "strict_confirm" ]]; then
  printf 'Phase must be pilot, refine, confirm, or strict_confirm.\n' >&2
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
LOG="$ROOT/logs/thor_slo_capacity_${STAGE}_${PHASE}_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_slo_capacity_frontier_v1.py \
  "$STAGE" --phase "$PHASE" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
KILL="kill -TERM $PID"
if [[ "$EUID" -eq 0 ]]; then KILL="sudo kill -TERM $PID"; fi
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: %s\n' \
  "$PID" "$LOG" "$LOG" "$KILL"
