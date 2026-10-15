#!/usr/bin/env bash
# Run the configuration-aware hardware performance and efficiency sweep.
# Invoke this script directly as root when using nohup.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${REPO_DIR}/.venv-thor/bin/python"
TEST_SCRIPT="${REPO_DIR}/scripts/active/run_thor_static_pareto.py"
LOG_DIR="${REPO_DIR}/logs"
mkdir -p "${LOG_DIR}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "error: run this script directly with sudo from the beginning:" >&2
  echo "  sudo -E ${SCRIPT_DIR}/run_thor_hardware_test.sh $*" >&2
  exit 1
fi

if [[ ! -x "${PYTHON}" ]]; then
  echo "error: missing ${PYTHON}" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/thor_hardware_test_${STAMP}.log"
PID_FILE="${LOG_DIR}/thor_hardware_test.pid"

if [[ -s "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}")"
  if kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "error: another hardware test is running (pid ${OLD_PID})" >&2
    exit 1
  fi
fi

echo "$$" > "${PID_FILE}"
cleanup() { rm -f "${PID_FILE}"; }
trap cleanup EXIT INT TERM

cd "${REPO_DIR}"
{
  echo "hardware_test_start=$(date --iso-8601=seconds)"
  echo "repo=${REPO_DIR}"
  echo "config=${ENERGYINFRA_PLATFORM_CONFIG:-${REPO_DIR}/configs/platform_profiles.yaml}"
  "${PYTHON}" - <<'PY'
from src.controller.cap_controller import CapController
import json

controller = CapController()
print(json.dumps(controller.read_actual_state(), ensure_ascii=True, indent=2))
PY
  echo "hardware_sweep_start=$(date --iso-8601=seconds)"
  # The full workload set includes p2048_o512.  The prompt template expands
  # beyond 4096 tokens, so use a larger context by default; user arguments
  # come last and can override it.
  "${PYTHON}" "${TEST_SCRIPT}" --n-ctx 8192 "$@"
} >> "${LOG_FILE}" 2>&1
