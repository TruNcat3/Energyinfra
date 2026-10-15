#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/active/run_thor_precision_nohup.sh"
PY="$ROOT/.venv-thor/bin/python"
WORKER=0
if [[ "${1:-}" == "--worker" ]]; then
  WORKER=1
  shift
fi
STAGE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$STAGE" in
  prepare|quick|gate|pareto) ;;
  *) printf 'Usage: sudo -E bash %s prepare|quick|gate|pareto\n' "$SCRIPT" >&2; exit 2 ;;
esac
if [[ $EUID -ne 0 ]]; then
  printf '%s\n' 'Run with sudo -E; the nohup worker never requests a password.' >&2
  exit 1
fi
cd "$ROOT"
export PYTHONUNBUFFERED=1
if [[ $WORKER -eq 1 ]]; then
  if [[ "$STAGE" == prepare ]]; then
    exec "$PY" -u scripts/active/prepare_thor_precision_v1.py
  fi
  for precision in q4_k_m q5_k_m q8_0; do
    case "$precision" in
      q4_k_m)
        if [[ "$STAGE" == pareto ]]; then
          config="configs/experiments/thor_precision_q4_k_m_pareto_v1.json"
          model="models/precision_v1/Qwen2.5-7B-random-v152064-Q4_K_M_same_source.gguf"
        elif [[ "$STAGE" == gate ]]; then
          config="configs/experiments/thor_precision_q4_k_m_gate_v1.json"
          model="models/precision_v1/Qwen2.5-7B-random-v152064-Q4_K_M_same_source.gguf"
        else
          config="configs/experiments/thor_model_scale_7b_pilot_v1.json"
          model="data/protocol_v2/thor_attribution_v2/models/Qwen2.5-random-paired-v152064-Q4_K_M.gguf"
        fi ;;
      q5_k_m)
        if [[ "$STAGE" == pareto ]]; then config="configs/experiments/thor_precision_q5_k_m_pareto_v1.json"; elif [[ "$STAGE" == gate ]]; then config="configs/experiments/thor_precision_q5_k_m_gate_v1.json"; else config="configs/experiments/thor_precision_q5_k_m_quick_v1.json"; fi
        model="models/precision_v1/Qwen2.5-7B-random-v152064-Q5_K_M.gguf" ;;
      q8_0)
        if [[ "$STAGE" == pareto ]]; then config="configs/experiments/thor_precision_q8_0_pareto_v1.json"; elif [[ "$STAGE" == gate ]]; then config="configs/experiments/thor_precision_q8_0_gate_v1.json"; else config="configs/experiments/thor_precision_q8_0_quick_v1.json"; fi
        model="models/precision_v1/Qwen2.5-7B-random-v152064-Q8_0.gguf" ;;
    esac
    if [[ ! -f "$model" ]]; then
      printf 'Missing model for %s: %s\n' "$precision" "$model" >&2
      exit 1
    fi
    if [[ "$STAGE" == pareto ]]; then
      preset="pilot"
      run_dir="data/protocol_v2/thor_precision_pareto_v1/${precision}_pilot"
    elif [[ "$STAGE" == gate ]]; then
      preset="quick"
      run_dir="data/protocol_v2/thor_precision_v1/${precision}_gate"
    else
      preset="quick"
      run_dir="data/protocol_v2/thor_precision_v1/${precision}_quick"
    fi
    printf 'Starting %s precision quick: %s\n' "$precision" "$run_dir"
    "$PY" -u scripts/active/run_cpu_floor_v1.py run --config "$config" --preset "$preset" --run-dir "$run_dir"
  done
  printf '%s\n' "Precision $STAGE reports:"
  if [[ "$STAGE" == pareto ]]; then
    printf '%s\n' data/protocol_v2/thor_precision_pareto_v1/*_pilot/RESULTS.md
  else
    printf '%s\n' "data/protocol_v2/thor_precision_v1/"*_"$STAGE"/RESULTS.md
  fi
  exit 0
fi
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/thor_precision_${STAGE}_$(date +%Y%m%d_%H%M%S)_$$.log"
nohup bash "$SCRIPT" --worker "$STAGE" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$LOG.pid"
printf 'Supervisor PID: %s\nLog: %s\nFollow: tail -f %s\nGraceful stop: sudo kill -TERM %s\n' "$PID" "$LOG" "$LOG" "$PID"
