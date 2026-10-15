#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    echo "this launcher must be started directly with sudo:"
    echo "  sudo -E $0 $*"
    exit 2
fi

PYTHON="$ROOT/.venv-thor/bin/python"
MODEL="${ENERGYINFRA_MODEL:-$ROOT/models/gguf/EnergyInfra-random-Qwen2.5-7B-Q4_K_M.gguf}"
LOG_DIR="${ENERGYINFRA_LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MODEL" ]]; then
    echo "model not found: $MODEL"
    echo "start it first with scripts/active/create_random_qwen2_7b_nohup.sh"
    exit 1
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="$LOG_DIR/thor_static_pareto_${STAMP}.log"
PID_FILE="$LOG_DIR/thor_static_pareto.pid"

nohup setsid env PYTHONUNBUFFERED=1 "$PYTHON" scripts/active/run_thor_static_pareto.py \
    --mode all \
    --model "$MODEL" \
    "$@" >"$LOG" 2>&1 < /dev/null &

PID=$!
printf '%s\n' "$PID" > "$PID_FILE"
printf 'started Thor static table + Pareto run\npid=%s\nmodel=%s\nlog=%s\n' \
    "$PID" "$MODEL" "$LOG"
printf 'monitor: tail -f %s\n' "$LOG"
