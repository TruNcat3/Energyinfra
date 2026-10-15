#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv-thor/bin/python"
PREPARE="${ROOT}/scripts/active/prepare_thor_queue_aware_hardware_validation_v1.py"
RUNNER="${ROOT}/scripts/active/run_thor_fixed_trace_serving_v1.py"
ANALYZER="${ROOT}/scripts/active/analyze_thor_long_generation_ttft_extension_v1.py"
CONFIG="${ROOT}/configs/experiments/thor_queue_aware_hardware_validation_v1.json"
ANALYSIS_CONFIG="${ROOT}/configs/experiments/thor_queue_aware_hardware_validation_reclassification_v1.json"
LOG_DIR="${ROOT}/logs"
stage="${1:-run}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python environment: ${PYTHON}" >&2
  exit 1
fi

case "${stage}" in
  prepare)
    exec "${PYTHON}" "${PREPARE}" --config "${CONFIG}"
    ;;
  analyze)
    exec "${PYTHON}" "${ANALYZER}" --config "${ANALYSIS_CONFIG}"
    ;;
  run|restore)
    if [[ "${EUID}" -ne 0 ]]; then
      echo "Hardware control requires root: sudo -E bash $0 ${stage}" >&2
      exit 1
    fi
    "${PYTHON}" "${PREPARE}" --config "${CONFIG}"
    mkdir -p "${LOG_DIR}"
    stamp="$(date +%Y%m%d_%H%M%S)"
    log="${LOG_DIR}/thor_queue_aware_hardware_validation_${stage}_${stamp}_$$.log"
    pid_file="${LOG_DIR}/thor_queue_aware_hardware_validation_${stage}_${stamp}_$$.pid"
    nohup "${PYTHON}" "${RUNNER}" --config "${CONFIG}" "${stage}" >"${log}" 2>&1 &
    pid=$!
    printf '%s\n' "${pid}" >"${pid_file}"
    echo "PID: ${pid}"
    echo "PID file: ${pid_file}"
    echo "Log: ${log}"
    echo "Follow: tail -f ${log}"
    echo "Graceful stop: sudo kill -TERM ${pid}"
    ;;
  *)
    echo "Usage: $0 {prepare|run|restore|analyze}" >&2
    exit 2
    ;;
esac
