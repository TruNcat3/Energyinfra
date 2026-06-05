#!/bin/bash
# Run expanded workload profiling for all 3 models sequentially
# Usage: sudo bash scripts/active/run_all_models_profiling.sh
#
# Per model: 11 GPU × 12 WL × 2 phases × 3 repeats = 792 runs
# Cap: 4 caps × 12 WL × 3 reps + 12×3 dynamic + 12×3 MAXN = 216 runs
# Total per model: ~1008 runs
# 3 models total: ~3024 runs, estimated ~35-45 hours

set -e
cd /home/wt/work/Energyinfra
PYTHON="/home/wt/work/Energyinfra/jetson_llm_env/bin/python3"
SCRIPTS="src/experiments"

MODELS=(
    "models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    "models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    "models/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME=$(basename "$MODEL" .gguf)
    echo "============================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $MODEL_NAME"
    echo "============================================================"

    echo "[1/2] Finegrained GPU profiling..."
    sudo $PYTHON $SCRIPTS/run_finegrained_profiling.py --all --gpu-only --model "$MODEL"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finegrained profiling done for $MODEL_NAME"

    echo "[2/2] Cap profiling..."
    sudo $PYTHON $SCRIPTS/run_cap_profiling.py --model "$MODEL" --gpu-only
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cap profiling done for $MODEL_NAME"

    echo ""
    echo "=== Completed: $MODEL_NAME ==="
    echo ""
done

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL MODELS COMPLETE!"
echo "============================================================"
