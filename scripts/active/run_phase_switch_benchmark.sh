#!/bin/bash
# Phase-Switching E2E Benchmark Runner
#
# Usage:
#   sudo bash scripts/active/run_phase_switch_benchmark.sh          # Full run (~4-6 hours)
#   sudo bash scripts/active/run_phase_switch_benchmark.sh --quick  # Quick test (~30 min)
#
# This script must be run with sudo because it needs to control
# GPU/EMC/CPU frequencies via sysfs/debugfs.

set -e
cd "$(dirname "$0")/../.."

ENV_PYTHON="/home/wt/work/Energyinfra/jetson_llm_env/bin/python3"
echo "Phase-Switching E2E Benchmark"
echo "=============================="
echo "Started: $(date)"
echo ""

# Kill stale tegrastats processes
pkill -f tegrastats 2>/dev/null || true
sleep 1

# Run the experiment
if [ "$1" = "--quick" ]; then
    echo "Mode: QUICK (2 workloads, 3 alphas, 2 repeats)"
    echo ""
    sudo "$ENV_PYTHON" src/experiments/run_phase_switch_e2e.py --quick
else
    echo "Mode: FULL (8 workloads, 3 alphas, 3 repeats)"
    echo ""
    sudo "$ENV_PYTHON" src/experiments/run_phase_switch_e2e.py
fi

echo ""
echo "Completed: $(date)"
echo ""
echo "Results saved to: data/e2e_benchmark/"
echo ""
echo "Next steps:"
echo "  1. Check the CSV and report in data/e2e_benchmark/"
echo "  2. Run visualization:"
echo "     $ENV_PYTHON src/visualization/visualize_phase_switch_e2e.py"
