#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"

VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-thor}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NVCC="${CUDACXX:-/usr/local/cuda/bin/nvcc}"

"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
export CUDACXX="$NVCC"
export CMAKE_ARGS="${CMAKE_ARGS:--DGGML_CUDA=on}"

"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install --prefer-binary -r requirements-thor.txt
"$VENV_DIR/bin/python" scripts/create_random_gguf.py

cat <<EOF
Thor environment is ready.
Model: $PROJECT_DIR/models/gguf/EnergyInfra-random-Qwen2.5-7B-Q4_K_M.gguf
Read-only smoke test:
  $VENV_DIR/bin/python scripts/active/run_thor_smoke.py
Frequency-cap test (requires root):
  sudo -E $VENV_DIR/bin/python scripts/active/run_thor_smoke.py --apply-cap
EOF
