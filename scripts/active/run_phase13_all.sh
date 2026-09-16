#!/bin/bash
# Phase 13 Full Test Suite — All experiments with checkpoint/resume
#
# Runs all Phase 13 experiments in optimal order:
#   1. P3 quick validation (5min) — verify ThermalSLO works
#   2. P1 finegrained cap (30min quick / 3h full)
#   3. P3 full serving benchmark (30min/model × 5 baseline × 3 trace)
#   4. Visualization (generate all charts)
#
# Supports:
#   --quick         P1 quick (2 cap × 2 wl × 2 rep) + P3 quick (5min)
#   --full          P1 full (10 cap × 7 wl × 3 rep) + P3 full (30min)
#   --p1-only       Only finegrained cap profiling
#   --p3-only       Only serving benchmark
#   --viz-only      Only visualization
#   --resume        Skip completed experiments (check data files)
#   --model <path>  Single model (default: all 3 models)
#   --list          List completion status
#
# Checkpoint: each experiment checks for existing output data.
# If data exists with sufficient rows → skip.
#
# Usage:
#   # Quick test (~1h total)
#   sudo nohup bash scripts/active/run_phase13_all.sh --quick >> logs/phase13_all.log 2>&1 &
#   tail -f logs/phase13_all.log
#
#   # Full experiment (~12h for 3 models)
#   sudo nohup bash scripts/active/run_phase13_all.sh --full >> logs/phase13_all.log 2>&1 &
#
#   # Resume interrupted run
#   sudo nohup bash scripts/active/run_phase13_all.sh --full --resume >> logs/phase13_all.log 2>&1 &
#
#   # Check status
#   bash scripts/active/run_phase13_all.sh --list

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../common.sh"
cd "$REPO_ROOT"
PYTHON="$PY"
mkdir -p logs data/serving_benchmark data/cap_profiling figures/phase13_analysis

# ── Defaults ──
MODE="full"         # quick | full
SCOPE="all"         # all | p1-only | p3-only | viz-only
RESUME=false
SINGLE_MODEL=""
LIST_ONLY=false
DURATION_MIN=30

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)   MODE="quick"; DURATION_MIN=5 ;;
        --full)    MODE="full"; DURATION_MIN=30 ;;
        --p1-only) SCOPE="p1-only" ;;
        --p3-only) SCOPE="p3-only" ;;
        --viz-only) SCOPE="viz-only" ;;
        --resume)  RESUME=true ;;
        --model)   SINGLE_MODEL="$2"; shift ;;
        --list)    LIST_ONLY=true ;;
        --duration) DURATION_MIN="$2"; shift ;;
        -h|--help)
            echo "Usage: $0 [--quick|--full] [--p1-only|--p3-only|--viz-only] [--resume] [--model <path>] [--list] [--duration <min>]"
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

# ── Model list ──
if [ -n "$SINGLE_MODEL" ]; then
    MODELS=("$SINGLE_MODEL")
else
    MODELS=(
        "models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
        "models/gguf/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        "models/gguf/Qwen2.5-14B-Instruct-Q4_K_M.gguf"
    )
fi

# ── Status check ──
model_name() {
    basename "$1" .gguf
}

check_p1_done() {
    local MNAME="$1"
    # Check for finegrained cap profiling data with enough rows
    local CKPT="data/cap_profiling/checkpoint_finegrained_${MNAME}.json"
    if [ -f "$CKPT" ]; then
        local DONE=$($PYTHON -c "
import json
with open('$CKPT') as f:
    d = json.load(f)
print(len(d.get('completed', [])))
" 2>/dev/null)
        local TOTAL=$($PYTHON -c "
# quick: 2 cap * 2 wl * 2 rep * 3 mode + 14 baseline = 38
# full:  10 cap * 7 wl * 3 rep * 3 mode + 42 baseline = 672 approx
print('38' if '$MODE' == 'quick' else '210')
" 2>/dev/null)
        if [ "$DONE" -ge "$TOTAL" ] 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

check_p3_done() {
    local MNAME="$1"
    local TRACE="$2"
    local BASELINE="$3"
    # Check for serving window data with enough windows
    local CSV=$(ls -t "data/serving_benchmark/serving_windows_${BASELINE}_${TRACE}_${MNAME}_"*.csv 2>/dev/null | head -1)
    if [ -n "$CSV" ] && [ -f "$CSV" ]; then
        local ROWS=$($PYTHON -c "
import pandas as pd
try:
    df = pd.read_csv('$CSV')
    print(len(df))
except: print(0)
" 2>/dev/null)
        # quick: 5min / 10s = ~30 windows
        # full: 30min / 10s = ~180 windows
        local MIN_WINDOWS=$($PYTHON -c "print('20' if '$MODE' == 'quick' else '100')" 2>/dev/null)
        if [ "$ROWS" -ge "$MIN_WINDOWS" ] 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# ── List mode ──
if $LIST_ONLY; then
    echo "Phase 13 Experiment Status (MODE=$MODE)"
    echo "============================================================"
    for MODEL in "${MODELS[@]}"; do
        MNAME=$(model_name "$MODEL")
        echo ""
        echo "Model: $MNAME"

        # P1 status
        if check_p1_done "$MNAME"; then
            echo "  P1 Finegrained Cap:  ✅ DONE"
        else
            echo "  P1 Finegrained Cap:  🔲 PENDING"
        fi

        # P3 status
        for BL in MAXN Dynamic BestStatic Pareto ThermalSLO; do
            for TRACE in short_chat long_generation bursty_mixed; do
                if check_p3_done "$MNAME" "$TRACE" "$BL"; then
                    echo "  P3 $BL × $TRACE:  ✅"
                else
                    echo "  P3 $BL × $TRACE:  🔲"
                fi
            done
        done
    done

    # Oracle gap
    if ls data/oracle_gap_analysis/oracle_gap_*.csv >/dev/null 2>&1; then
        echo ""
        echo "P0 Oracle Gap:  ✅ DONE"
    else
        echo ""
        echo "P0 Oracle Gap:  🔲 PENDING"
    fi

    exit 0
fi

# ── Main execution ──
START_TIME=$(date +%s)
FAILED=()
SKIPPED=()

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Phase 13 Full Test Suite"
echo "  MODE:   $MODE"
echo "  SCOPE:  $SCOPE"
echo "  RESUME: $RESUME"
echo "  DURATION: ${DURATION_MIN}min"
echo "  MODELS: ${#MODELS[@]} model(s)"
echo "============================================================"

# ──────────────────────────────────────────────
# Stage 0: Oracle Gap (P0)
# ──────────────────────────────────────────────
if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "viz-only" ]; then
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Stage P0: Oracle Gap Analysis ==="

    if $RESUME && ls data/oracle_gap_analysis/oracle_gap_*.csv >/dev/null 2>&1; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: Oracle gap data exists"
        SKIPPED+=("P0 Oracle Gap")
    else
        if $PYTHON src/ratetable/oracle_gap_analysis.py; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] P0 Oracle Gap DONE"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] P0 Oracle Gap FAILED"
            FAILED+=("P0 Oracle Gap")
        fi
    fi
fi

# ──────────────────────────────────────────────
# Stage 1: Finegrained Cap Profiling (P1)
# ──────────────────────────────────────────────
if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "p1-only" ]; then
    for MODEL in "${MODELS[@]}"; do
        MNAME=$(model_name "$MODEL")

        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === P1: Finegrained Cap — $MNAME ==="

        if $RESUME && check_p1_done "$MNAME"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $MNAME cap data already exists"
            SKIPPED+=("P1 $MNAME")
            continue
        fi

        # Quick estimate: 7B ~30min quick, ~3h full; 14B ~3x slower
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running finegrained cap profiling for $MNAME..."
        P1_ARGS="--model $MODEL"
        if [ "$MODE" = "quick" ]; then
            P1_ARGS="$P1_ARGS --quick"
        fi
        if $RESUME; then
            P1_ARGS="$P1_ARGS --resume"
        fi

        if $PYTHON src/experiments/run_finegrained_cap_profiling.py $P1_ARGS; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] P1 $MNAME DONE"
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] P1 $MNAME FAILED"
            FAILED+=("P1 $MNAME")
        fi
    done
fi

# ──────────────────────────────────────────────
# Stage 2: Serving Benchmark (P3)
# ──────────────────────────────────────────────
if [ "$SCOPE" = "all" ] || [ "$SCOPE" = "p3-only" ]; then
    BASELINES=("MAXN" "Dynamic" "BestStatic" "Pareto" "ThermalSLO")
    TRACES=("short_chat" "long_generation" "bursty_mixed")

    for MODEL in "${MODELS[@]}"; do
        MNAME=$(model_name "$MODEL")

        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] === P3: Serving Benchmark — $MNAME ==="

        ALL_P3_DONE=true
        for BL in "${BASELINES[@]}"; do
            for TRACE in "${TRACES[@]}"; do
                if ! check_p3_done "$MNAME" "$TRACE" "$BL"; then
                    ALL_P3_DONE=false
                    break 2
                fi
            done
        done

        if $RESUME && $ALL_P3_DONE; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP: $MNAME all serving data exists"
            SKIPPED+=("P3 $MNAME")
            continue
        fi

        MODEL_START=$(date +%s)

        for BL in "${BASELINES[@]}"; do
            for TRACE in "${TRACES[@]}"; do
                if $RESUME && check_p3_done "$MNAME" "$TRACE" "$BL"; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   SKIP: $BL × $TRACE (done)"
                    continue
                fi

                echo ""
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Running: $BL × $TRACE ($DURATION_MIN min)"

                if $PYTHON src/experiments/run_serving_benchmark.py \
                    --model "$MODEL" \
                    --baseline "$BL" \
                    --trace "$TRACE" \
                    --duration-min "$DURATION_MIN"; then
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   DONE: $BL × $TRACE"
                else
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   FAILED: $BL × $TRACE"
                    FAILED+=("P3 $MNAME $BL×$TRACE")
                fi

                # Cooldown between experiments
                echo "[$(date '+%Y-%m-%d %H:%M:%S')]   Cooldown 30s..."
                sleep 30
            done
        done

        MODEL_END=$(date +%s)
        MODEL_ELAPSED=$(( (MODEL_END - MODEL_START) / 60 ))
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] P3 $MNAME: ${MODEL_ELAPSED} min total"
    done
fi

# ──────────────────────────────────────────────
# Stage 3: Visualization (P4)
# ──────────────────────────────────────────────
if [ "$SCOPE" != "p1-only" ] && [ "$SCOPE" != "p3-only" ]; then
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === P4: Visualization ==="

    if $PYTHON src/visualization/visualize_phase13.py \
        --oracle-data data/oracle_gap_analysis \
        --serving-data data/serving_benchmark \
        --output figures/phase13_analysis; then
        CHART_COUNT=$(ls -1 figures/phase13_analysis/*.png 2>/dev/null | wc -l)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] P4 DONE: ${CHART_COUNT} charts generated"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] P4 FAILED"
        FAILED+=("P4 Visualization")
    fi
fi

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
END_TIME=$(date +%s)
TOTAL_ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
TOTAL_SEC=$(( (END_TIME - START_TIME) % 60 ))

echo ""
echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] PHASE 13 TEST SUITE COMPLETE"
echo "  Total: ${TOTAL_ELAPSED}m ${TOTAL_SEC}s"
echo "============================================================"

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped (already done):"
    for s in "${SKIPPED[@]}"; do echo "  + $s"; done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "FAILURES:"
    for f in "${FAILED[@]}"; do echo "  ✗ $f"; done
    echo ""
    echo "Re-run with --resume to skip completed experiments:"
    echo "  sudo bash scripts/active/run_phase13_all.sh --full --resume"
    exit 1
fi

echo ""
echo "✅ All experiments completed successfully!"
echo ""
echo "Output:"
echo "  P0: data/oracle_gap_analysis/"
echo "  P1: data/cap_profiling/"
echo "  P3: data/serving_benchmark/"
echo "  P4: figures/phase13_analysis/"
echo ""
echo "Next: analyze results and update documentation."
