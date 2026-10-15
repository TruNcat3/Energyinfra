#!/usr/bin/env bash
# Authenticate in the foreground: sudo -E bash scripts/active/run_thor_finegrained_selector_nohup.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv-thor/bin/python"
CONFIG="$ROOT/configs/experiments/thor_finegrained_selector.yaml"
if [[ "${1:-}" == "--worker" ]]; then
    shift
    exec "$PYTHON" -u src/experiments/run_cap_selector_benchmark.py \
        --experiment-config "$CONFIG" "$@"
fi
if [[ "$EUID" -ne 0 ]]; then
    printf 'Run with foreground sudo: sudo -E bash %s\n' "${BASH_SOURCE[0]}" >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    printf 'Missing Python: %s\n' "$PYTHON" >&2
    exit 1
fi
mkdir -p logs
# Lock is inherited by the background worker until it exits.
exec 9>logs/thor_finegrained_selector.lock
if ! flock -n 9; then
    printf 'A finegrained selector campaign is already running.\n' >&2
    exit 1
fi
stamp="$(date +%Y%m%d_%H%M%S)_$$"
log="$ROOT/logs/thor_finegrained_selector_${stamp}.log"
pidfile="${log%.log}.pid"
nohup bash "$ROOT/scripts/active/run_thor_finegrained_selector_nohup.sh" \
    --worker "$@" </dev/null >"$log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$pidfile"
printf 'Started PID: %s\nLog: %s\nPID file: %s\n' "$pid" "$log" "$pidfile"
printf 'Follow: tail -f %q\n' "$log"
printf 'Stop gracefully: sudo kill -TERM %s\n' "$pid"
printf 'Launch does not imply completion; inspect COMPLETE and status.json.\n'
