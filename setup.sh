#!/usr/bin/env bash
# Energyinfra environment initialization.
# Creates the directory structure, Python virtual environment and installs
# dependencies. Safe to re-run (idempotent).
#
# Usage: ./setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV_DIR="jetson_llm_env"   # name kept for continuity with existing docs

echo "==> [1/4] Creating directory structure"
mkdir -p \
    data/energy_profiling data/cap_profiling data/rate_tables \
    data/cap_selector_benchmark data/oracle_gap_analysis data/multi_obj_eval \
    data/serving_benchmark data/e2e_benchmark data/analysis data/parsed \
    data/phase_aware_experiment data/real_model_experiment \
    figures/pareto_frontier figures/cross_model_comparison \
    figures/e2e_benchmark figures/phase13_analysis \
    logs models/gguf

echo "==> [2/4] Creating Python virtual environment ($VENV_DIR)"
if [ ! -x "$VENV_DIR/bin/python3" ]; then
    python3 -m venv "$VENV_DIR" 2>/dev/null || python3 -m virtualenv "$VENV_DIR" \
        || { echo "ERROR: could not create a virtual environment (need python3-venv or virtualenv)" >&2; exit 1; }
fi

echo "==> [3/4] Installing Python dependencies"
"$VENV_DIR/bin/pip" install --quiet --upgrade pip
"$VENV_DIR/bin/pip" install --quiet pandas numpy pyarrow matplotlib seaborn scipy pyyaml

echo "==> [4/4] Checking optional runtime (llama-cpp-python)"
if "$VENV_DIR/bin/python3" -c "import llama_cpp" 2>/dev/null; then
    echo "    llama_cpp OK ($("$VENV_DIR/bin/python3" -c 'import llama_cpp; print(llama_cpp.__version__)'))"
else
    echo "    llama-cpp-python not installed."
    echo "    It requires a CUDA build on Jetson — see docs/说明文档/LLAMACPP_INTEGRATION.md"
fi

echo
echo "Setup complete. Next steps:"
echo "  1. Place GGUF models under models/gguf/  (see docs/说明文档/DOCKER_MODEL_SETUP.md)"
echo "  2. Scripts resolve the venv python automatically; or activate manually:"
echo "       source $VENV_DIR/bin/activate"
echo "  3. Run experiments from the repo root, e.g.:"
echo "       sudo scripts/active/run_phase13_all.sh   # or individual src/experiments/*.py entry points"
