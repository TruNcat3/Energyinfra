#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv-thor/bin/python"
SCRIPT="${ROOT}/scripts/active/explore_thor_queue_aware_pareto_v1.py"
CONFIG="${ROOT}/configs/experiments/thor_queue_aware_pareto_v1.json"
LOG_DIR="${ROOT}/logs"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python environment: ${PYTHON}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${LOG_DIR}/thor_queue_aware_pareto_${stamp}_$$.log"
pid_file="${LOG_DIR}/thor_queue_aware_pareto_${stamp}_$$.pid"

nohup "${PYTHON}" "${SCRIPT}" --config "${CONFIG}" >"${log}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" >"${pid_file}"

echo "PID: ${pid}"
echo "PID file: ${pid_file}"
echo "Log: ${log}"
echo "Follow: tail -f ${log}"
echo "Stop: kill -TERM ${pid}"
echo "No hardware frequency changes are performed by this search."
