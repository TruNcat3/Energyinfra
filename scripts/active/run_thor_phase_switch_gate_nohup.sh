#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  printf 'Usage: %s run|quick|report|restore [--run-dir PATH] [--config PATH]\n' "$0" >&2
  exit 2
fi
shift

case "$STAGE" in
  run|quick)
    if [[ "$EUID" -ne 0 ]]; then
      printf 'Hardware run requires: sudo -E bash %s %s [options]\n' "$0" "$STAGE" >&2
      exit 2
    fi
    mkdir -p "$ROOT/logs"
    LOG="$ROOT/logs/thor_phase_switch_gate_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
    PRESET="full"
    [[ "$STAGE" == "quick" ]] && PRESET="quick"
    cd "$ROOT"
    nohup "$ROOT/.venv-thor/bin/python" -u \
      scripts/active/run_thor_phase_switch_gate_v1.py run \
      --preset "$PRESET" "$@" >"$LOG" 2>&1 < /dev/null &
    PID=$!
    printf '%s\n' "$PID" > "${LOG%.log}.pid"
    printf 'PID: %s\nLog: %s\nFollow: tail -f %q\nGraceful stop: sudo kill -TERM %s\n' \
      "$PID" "$LOG" "$LOG" "$PID"
    ;;
  report)
    cd "$ROOT"
    exec "$ROOT/.venv-thor/bin/python" -u \
      scripts/active/run_thor_phase_switch_gate_v1.py report "$@"
    ;;
  restore)
    if [[ "$EUID" -ne 0 ]]; then
      printf 'Restore requires: sudo -E bash %s restore [options]\n' "$0" >&2
      exit 2
    fi
    cd "$ROOT"
    exec "$ROOT/.venv-thor/bin/python" -u \
      scripts/active/run_thor_phase_switch_gate_v1.py restore "$@"
    ;;
  *)
    printf 'Expected run, quick, report, or restore\n' >&2
    exit 2
    ;;
esac
