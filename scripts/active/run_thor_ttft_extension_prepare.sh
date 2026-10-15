#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv-thor/bin/python"
CONFIG="${ROOT}/configs/experiments/thor_q4_ttft_extension_v1.json"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python environment: ${PYTHON}" >&2
  exit 1
fi

exec "${PYTHON}" "${ROOT}/scripts/active/build_thor_ttft_extension_v1.py" \
  --config "${CONFIG}"
