#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/active/run_thor_model_scale_nohup.sh"
PY="$ROOT/.venv-thor/bin/python"
WORKER=0
if [[ "${1:-}" == "--worker" ]]; then
    WORKER=1
    shift
fi
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in
    prepare|quick|pilot) ;;
    *) printf 'Usage: sudo -E bash %s prepare|quick|pilot [--run-root PATH]\n' "$SCRIPT" >&2; exit 2 ;;
esac
RUN_ROOT="$ROOT/data/protocol_v2/thor_model_scale_v1"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root)
            if [[ $# -lt 2 ]]; then printf '%s\n' 'Missing --run-root value' >&2; exit 2; fi
            RUN_ROOT="$2"; shift 2 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$RUN_ROOT" != /* ]]; then RUN_ROOT="$ROOT/$RUN_ROOT"; fi
if [[ $EUID -ne 0 ]]; then
    printf '%s\n' 'Run this launcher with sudo -E; it never requests a password inside nohup.' >&2
    exit 1
fi
cd "$ROOT"
export PYTHONUNBUFFERED=1

if [[ $WORKER -eq 1 ]]; then
    if [[ "$STAGE" == "prepare" ]]; then
        printf '%s\n' 'Preparing full-shape random Qwen2.5-14B. No frequency-control writes or inference.'
        exec "$PY" -u scripts/active/prepare_thor_qwen14b_random.py
    fi
    if [[ ! -f models/random_scale_v1/Qwen2.5-14B-random-v152064-Q4_K_M.gguf ]]; then
        printf '%s\n' '14B GGUF is missing. Complete the prepare stage first.' >&2
        exit 1
    fi
    if [[ "$STAGE" == "quick" ]]; then
        exec "$PY" -u scripts/active/run_cpu_floor_v1.py run \
            --config configs/experiments/thor_model_scale_14b_pilot_v1.json \
            --preset quick --run-dir "$RUN_ROOT/14b_quick"
    fi
    CHILD=""
    stop_child() {
        trap '' TERM INT
        printf '%s\n' 'Stopping active model campaign; waiting for hardware restoration.'
        if [[ -n "$CHILD" ]]; then
            kill -TERM "$CHILD" 2>/dev/null || true
            wait "$CHILD" || true
        fi
        exit 143
    }
    trap stop_child TERM INT
    # Never overlap model campaigns. Each has its own immutable manifest and
    # uses the existing runner's global hardware lock and restoration path.
    for SIZE in 7b 14b; do
        printf '\nStarting %s matched pilot: profile -> select -> validate -> report\n' "$SIZE"
        "$PY" -u scripts/active/run_cpu_floor_v1.py run \
            --config "configs/experiments/thor_model_scale_${SIZE}_pilot_v1.json" \
            --preset pilot --run-dir "$RUN_ROOT/${SIZE}_pilot" &
        CHILD=$!
        printf 'Active worker PID: %s\n' "$CHILD"
        if wait "$CHILD"; then
            CHILD=""
        else
            STATUS=$?
            CHILD=""
            printf '%s pilot failed (exit %s); later stages were not launched.\n' "$SIZE" "$STATUS" >&2
            exit "$STATUS"
        fi
    done
    printf '\nMatched pilots complete. Reports:\n%s\n%s\n' \
        "$RUN_ROOT/7b_pilot/RESULTS.md" "$RUN_ROOT/14b_pilot/RESULTS.md"
    exit 0
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/model_scale_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$SCRIPT" --worker "$STAGE" --run-root "$RUN_ROOT" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$LOG.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %s\nGraceful stop: sudo kill -TERM %s\n' \
    "$PID" "$LOG" "$LOG" "$PID"
if [[ "$STAGE" == "prepare" ]]; then
    printf '%s\n' 'Preparation only. It does not launch quick, pilot, or serving.'
else
    printf '%s\n' 'No serving or exhaustive frequency sweep is launched.'
fi
