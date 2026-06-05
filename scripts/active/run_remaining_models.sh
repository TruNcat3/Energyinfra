#!/bin/bash
# Run expanded workload profiling for remaining models (8B + 14B)
# 7B already complete, skip it.
#
# Per model: 11 GPU × 12 WL × 2 phases × 3 repeats = 792 finegrained runs
#           4 caps × 12 WL × 3 reps + 12×3 dynamic + 12×3 MAXN = 216 cap runs
# Total per model: ~1008 runs, ~7-8 hours
# 2 models: ~2016 runs, ~14-16 hours
#
# Supports resume: automatically skips models/steps with valid existing data.
#
# Usage:
#   sudo nohup bash scripts/active/run_remaining_models.sh > logs/run_remaining_models.log 2>&1 &
#   tail -f logs/run_remaining_models.log
#
# Stop:
#   sudo pkill -f run_finegrained_profiling
#   sudo pkill -f run_cap_profiling
#
# Resume (same command, auto-skips completed):
#   sudo nohup bash scripts/active/run_remaining_models.sh >> logs/run_remaining_models.log 2>&1 &

cd /home/wt/work/Energyinfra
PYTHON="/home/wt/work/Energyinfra/jetson_llm_env/bin/python3"
SCRIPTS="src/experiments"

mkdir -p logs

MODELS=(
    "models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "models/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)

START_TIME=$(date +%s)

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting remaining model profiling"
echo "============================================================"

# Check if a finegrained profiling CSV exists with >=750 rows for a model
check_finegrained_done() {
    local MODEL_NAME="$1"
    # Find latest CSV for this model
    local CSV=$(ls -t data/energy_profiling/finegrained_profiling_*.csv 2>/dev/null | head -1)
    if [ -z "$CSV" ]; then
        return 1
    fi
    # Check if it contains this model and has enough rows (792 expected, >=750 for partial tolerance)
    local ROWS=$($PYTHON -c "
import pandas as pd
try:
    df = pd.read_csv('$CSV')
    m = df[df['model'] == '$MODEL_NAME']
    print(len(m))
except: print(0)
" 2>/dev/null)
    if [ "$ROWS" -ge 750 ] 2>/dev/null; then
        return 0
    fi
    return 1
}

# Check if cap profiling CSV exists with >=200 rows for a model
check_cap_done() {
    local MODEL_NAME="$1"
    local CSV=$(ls -t data/cap_profiling/cap_profiling_*.csv 2>/dev/null | head -1)
    if [ -z "$CSV" ]; then
        return 1
    fi
    local ROWS=$($PYTHON -c "
import pandas as pd
try:
    df = pd.read_csv('$CSV')
    m = df[df['model'] == '$MODEL_NAME']
    print(len(m))
except: print(0)
" 2>/dev/null)
    if [ "$ROWS" -ge 200 ] 2>/dev/null; then
        return 0
    fi
    return 1
}

FAILED=()
SKIPPED=()

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL" .gguf)
    MODEL_START=$(date +%s)

    echo ""
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking: $MODEL_NAME"
    echo "============================================================"

    NEED_FINE=1
    NEED_CAP=1

    # Check finegrained
    if check_finegrained_done "$MODEL_NAME"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP finegrained: $MODEL_NAME already has valid data"
        NEED_FINE=0
    fi

    # Check cap
    if check_cap_done "$MODEL_NAME"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP cap: $MODEL_NAME already has valid data"
        NEED_CAP=0
    fi

    if [ $NEED_FINE -eq 0 ] && [ $NEED_CAP -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $MODEL_NAME fully complete, skipping."
        SKIPPED+=("$MODEL_NAME")
        continue
    fi

    # Step 1: Finegrained GPU profiling
    if [ $NEED_FINE -eq 1 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [1/2] Finegrained GPU profiling (792 runs, ~5-6h)..."
        if $PYTHON $SCRIPTS/run_finegrained_profiling.py --all --gpu-only --model "$MODEL"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finegrained profiling DONE for $MODEL_NAME"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finegrained profiling FAILED for $MODEL_NAME"
            FAILED+=("$MODEL_NAME:finegrained")
            continue
        fi
    fi

    # Step 2: Cap profiling
    if [ $NEED_CAP -eq 1 ]; then
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [2/2] Cap profiling (216 runs, ~1.5-2h)..."
        if $PYTHON $SCRIPTS/run_cap_profiling.py --model "$MODEL" --gpu-only; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cap profiling DONE for $MODEL_NAME"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cap profiling FAILED for $MODEL_NAME"
            FAILED+=("$MODEL_NAME:cap")
        fi
    fi

    MODEL_END=$(date +%s)
    MODEL_ELAPSED=$(( (MODEL_END - MODEL_START) / 60 ))
    echo ""
    echo "=== Completed: $MODEL_NAME (${MODEL_ELAPSED} min) ==="
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE! Total: ${TOTAL_ELAPSED} min"
echo "============================================================"

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped (already complete):"
    for s in "${SKIPPED[@]}"; do
        echo "  + $s"
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    for f in "${FAILED[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "Re-run the same command to retry failed steps."
    exit 1
fi

echo ""
echo "All models completed successfully!"
echo "Check data/energy_profiling/ and data/cap_profiling/ for results."
