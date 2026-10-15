#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv-thor/bin/python"
RUNNER="${ROOT}/scripts/active/run_thor_fixed_trace_serving_v1.py"
CONFIG="${ROOT}/configs/experiments/thor_q4_long_generation_ttft_extension_v1.json"
LOG_DIR="${ROOT}/logs"
SOURCE_STATIC="${ROOT}/data/protocol_v2/thor_q4_full_frontier_v1"
CUSTOM_STATIC="${ROOT}/data/protocol_v2/thor_q4_ttft_extension_v1"

if [[ "${1:-run}" != "run" ]]; then
  echo "Usage: sudo -E bash $0 run" >&2
  exit 2
fi
if [[ "${EUID}" -ne 0 ]]; then
  echo "Hardware control requires root. Run: sudo -E bash $0 run" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing Python environment: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CUSTOM_STATIC}/comparison.json" ]]; then
  echo "Missing custom plans. Run the prepare stage first." >&2
  exit 1
fi

# Preserve the original static campaign as immutable provenance while exposing
# any auxiliary artifacts expected by the generic serving runner.
for source in "${SOURCE_STATIC}"/*; do
  name="$(basename "${source}")"
  [[ -e "${CUSTOM_STATIC}/${name}" || -L "${CUSTOM_STATIC}/${name}" ]] && continue
  ln -s "${source}" "${CUSTOM_STATIC}/${name}"
done

mkdir -p "${LOG_DIR}"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${LOG_DIR}/thor_long_generation_ttft_extension_${stamp}_$$.log"
pid_file="${LOG_DIR}/thor_long_generation_ttft_extension_${stamp}_$$.pid"

nohup "${PYTHON}" "${RUNNER}" --config "${CONFIG}" run >"${log}" 2>&1 &
pid=$!
printf '%s\n' "${pid}" > "${pid_file}"

echo "PID: ${pid}"
echo "PID file: ${pid_file}"
echo "Log: ${log}"
echo "Follow: tail -f ${log}"
echo "Graceful stop: sudo kill -TERM ${pid}"
