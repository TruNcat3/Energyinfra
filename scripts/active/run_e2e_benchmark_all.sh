#!/bin/bash
# Run E2E Cap Selector Benchmark for all 3 models
# Validates workload-aware cap selector with real inference
#
# Per model: ~90 runs (6 strategies × 5 WL × 3 reps + baselines), ~1-1.5h
# 3 models: ~270 runs, ~3-5 hours
#
# Supports resume: skips models with valid existing benchmark data.
#
# Usage:
#   sudo nohup bash scripts/active/run_e2e_benchmark_all.sh >> logs/run_e2e_benchmark_all.log 2>&1 &
#   tail -f logs/run_e2e_benchmark_all.log

cd /home/wt/work/Energyinfra
PYTHON="/home/wt/work/Energyinfra/jetson_llm_env/bin/python3"

mkdir -p logs

MODELS=(
    "models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    "models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "models/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)

START_TIME=$(date +%s)

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E2E Cap Selector Benchmark - All Models"
echo "============================================================"

# Check if a model's E2E benchmark data already exists with enough rows
check_benchmark_done() {
    local MODEL_NAME="$1"
    local CSV=$(ls -t data/cap_selector_benchmark/cap_selector_benchmark_*.csv 2>/dev/null | head -5)
    for f in $CSV; do
        local ROWS=$($PYTHON -c "
import pandas as pd
try:
    df = pd.read_csv('$f')
    m = df[df['model'] == '$MODEL_NAME']
    print(len(m))
except: print(0)
" 2>/dev/null)
        if [ "$ROWS" -ge 80 ] 2>/dev/null; then
            return 0
        fi
    done
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

    # Find latest rate tables for this model
    CAP_RT=$(ls -t data/rate_tables/cap_rate_table_with_savings_*.parquet 2>/dev/null | head -5)
    LOCK_RT=$(ls -t data/rate_tables/lock_rate_table_*.parquet 2>/dev/null | head -5)

    # Find the rate table that contains this model
    CAP_RT_PATH=""
    LOCK_RT_PATH=""
    for f in $CAP_RT; do
        HAS_MODEL=$($PYTHON -c "
import pandas as pd
df = pd.read_csv('$f' if '$f' ends with .csv else '/dev/null')
# parquet
" 2>/dev/null)
        # Use python to check
        FOUND=$($PYTHON -c "
import pandas as pd
try:
    df = pd.read_parquet('$f')
    has = '$MODEL_NAME' in df['model'].values
    print('1' if has else '0')
except: print('0')
" 2>/dev/null)
        if [ "$FOUND" = "1" ]; then
            CAP_RT_PATH="$f"
            break
        fi
    done
    for f in $LOCK_RT; do
        FOUND=$($PYTHON -c "
import pandas as pd
try:
    df = pd.read_parquet('$f')
    has = '$MODEL_NAME' in df['model'].values
    print('1' if has else '0')
except: print('0')
" 2>/dev/null)
        if [ "$FOUND" = "1" ]; then
            LOCK_RT_PATH="$f"
            break
        fi
    done

    if [ -z "$CAP_RT_PATH" ] || [ -z "$LOCK_RT_PATH" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: No rate table found for $MODEL_NAME"
        SKIPPED+=("$MODEL_NAME (no rate table)")
        continue
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rate tables found:"
    echo "  Cap: $(basename $CAP_RT_PATH)"
    echo "  Lock: $(basename $LOCK_RT_PATH)"

    # Check if benchmark already done
    if check_benchmark_done "$MODEL_NAME"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $MODEL_NAME benchmark data already exists"
        SKIPPED+=("$MODEL_NAME")
        continue
    fi

    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running E2E benchmark for $MODEL_NAME..."
    if $PYTHON src/experiments/run_cap_selector_benchmark.py \
        --model "$MODEL" \
        --cap-rate-table "$CAP_RT_PATH" \
        --lock-rate-table "$LOCK_RT_PATH"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] E2E benchmark DONE for $MODEL_NAME"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] E2E benchmark FAILED for $MODEL_NAME"
        FAILED+=("$MODEL_NAME")
    fi

    MODEL_END=$(date +%s)
    MODEL_ELAPSED=$(( (MODEL_END - MODEL_START) / 60 ))
    echo "=== $MODEL_NAME: ${MODEL_ELAPSED} min ==="
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] E2E BENCHMARK ALL DONE! Total: ${TOTAL_ELAPSED} min"
echo "============================================================"

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped:"
    for s in "${SKIPPED[@]}"; do echo "  + $s"; done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
    exit 1
fi

echo ""
echo "All models completed successfully!"
