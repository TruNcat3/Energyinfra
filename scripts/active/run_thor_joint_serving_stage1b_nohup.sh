#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-}"
PRESET="${2:-quick}"
if [[ "$STAGE" != "prepare" && "$STAGE" != "run" && "$STAGE" != "report" && "$STAGE" != "restore" ]]; then
  printf 'Usage: bash %s prepare|run|report|restore quick|full [--run-dir PATH] [--config PATH]\n' "$0" >&2
  exit 2
fi
if [[ "$PRESET" != "quick" && "$PRESET" != "full" ]]; then
  printf 'Preset must be quick or full.\n' >&2
  exit 2
fi
shift 2 || true
if [[ "$STAGE" == "run" || "$STAGE" == "restore" ]]; then
  if [[ "$EUID" -ne 0 ]]; then
    printf 'Run %s with sudo -E so hardware state can be changed and restored.\n' "$STAGE" >&2
    exit 2
  fi
elif [[ "$EUID" -eq 0 ]]; then
  printf 'Run %s without sudo.\n' "$STAGE" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_joint_serving_stage1b_${STAGE}_${PRESET}_$(date +%Y%m%d_%H%M%S)_$$.log"
cd "$ROOT"
nohup "$ROOT/.venv-thor/bin/python" -u scripts/active/run_thor_joint_serving_stage1b_v1.py \
  "$STAGE" --preset "$PRESET" "$@" >"$LOG" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "${LOG%.log}.pid"
KILL="kill -TERM $PID"
if [[ "$EUID" -eq 0 ]]; then
  KILL="sudo kill -TERM $PID"
fi
printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: %s\n' \
  "$PID" "$LOG" "$LOG" "$KILL"
