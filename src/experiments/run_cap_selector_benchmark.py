#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
E2E Benchmark for Workload-Aware Cap Selector (Phase 12)

Validates the workload-aware cap selector by running real inference with
selected GPU caps and comparing against baselines (dynamic, MAXN).

Experiment matrix:
  Strategies: slo_constrained, min_energy, alpha_weighted, power_budget, pareto
  Workloads: 5 representative
  Repeats: 3
  + baselines: dynamic, MAXN
  Total: ~105 runs

Usage:
    sudo python3 src/experiments/run_cap_selector_benchmark.py
    sudo python3 src/experiments/run_cap_selector_benchmark.py --quick
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.metrics.metrics_collector import MetricsCollector
from src.controller.cap_controller import CapController
from src.controller.workload_cap_selector import WorkloadCapSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPRESENTATIVE_WORKLOADS = {
    'p64_o64':     {'prompt_length': 64,   'output_length': 64},
    'p128_o512':   {'prompt_length': 128,  'output_length': 512},
    'p512_o128':   {'prompt_length': 512,  'output_length': 128},
    'p1024_o512':  {'prompt_length': 1024, 'output_length': 512},
    'p2048_o512':  {'prompt_length': 2048, 'output_length': 512},
}

REPEATS = 3
WARMUP_RUNS = 1
COOLDOWN_S = 3
TEGRASTATS_MS = 500


def run_with_cap(runner, mc, cc, gpu_cap, prompt_length, output_length, tag):
    """Run inference under a specific GPU cap."""
    cap_info = cc.set_cap(gpu_cap, 3199, 1036)
    runner.warmup(WARMUP_RUNS)

    state_before = cc.read_actual_state()
    mc.start_collection(tag=tag)

    result = runner.run_single_inference(
        prompt_length=prompt_length,
        output_length=output_length,
        benchmark_mode=True,
    )

    time.sleep(0.3)
    mc.stop_collection()
    energy = mc.compute_energy()

    state_after = cc.read_actual_state()

    result.update({
        'actual_gpu_mhz': state_after['actual_gpu_mhz'],
        'actual_cpu_mhz': state_after['actual_cpu_mhz'],
        'temperature_start_c': state_before['temperature_c'],
        'temperature_end_c': state_after['temperature_c'],
        'governor_gpu': state_after['governor_gpu'],
        'avg_power_w': energy.get('avg_power_w', 0),
        'max_power_w': energy.get('max_power_w', 0),
        'total_energy_j': energy.get('total_energy_j', 0),
    })
    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
        result['tokens_per_joule'] = result['output_tokens'] / result['total_energy_j']
    else:
        result['energy_per_token_j'] = 0
        result['tokens_per_joule'] = 0

    return result


def run_dynamic(runner, mc, cc, prompt_length, output_length, tag):
    """Run inference under default dynamic governor."""
    cc.restore_dynamic()
    runner.warmup(WARMUP_RUNS)

    state_before = cc.read_actual_state()
    mc.start_collection(tag=tag)

    result = runner.run_single_inference(
        prompt_length=prompt_length,
        output_length=output_length,
        benchmark_mode=True,
    )

    time.sleep(0.3)
    mc.stop_collection()
    energy = mc.compute_energy()

    state_after = cc.read_actual_state()

    result.update({
        'actual_gpu_mhz': state_after['actual_gpu_mhz'],
        'actual_cpu_mhz': state_after['actual_cpu_mhz'],
        'temperature_start_c': state_before['temperature_c'],
        'temperature_end_c': state_after['temperature_c'],
        'governor_gpu': state_after['governor_gpu'],
        'avg_power_w': energy.get('avg_power_w', 0),
        'max_power_w': energy.get('max_power_w', 0),
        'total_energy_j': energy.get('total_energy_j', 0),
    })
    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
        result['tokens_per_joule'] = result['output_tokens'] / result['total_energy_j']
    else:
        result['energy_per_token_j'] = 0
        result['tokens_per_joule'] = 0

    return result


def run_maxn(runner, mc, cc, prompt_length, output_length, tag):
    """Run inference under MAXN (all max, performance governor)."""
    GPU_SYSFS_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
    with open(f'{GPU_SYSFS_PATH}/governor', 'w') as f:
        f.write('performance')
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write('1300500000')
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write('1300500000')

    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov = cpu / 'cpufreq' / 'scaling_governor'
        mx = cpu / 'cpufreq' / 'scaling_max_freq'
        mn = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov.exists():
            try:
                open(gov, 'w').write('performance')
                open(mx, 'w').write('2201600')
                open(mn, 'w').write('2201600')
            except PermissionError:
                pass

    time.sleep(0.3)
    runner.warmup(WARMUP_RUNS)

    state_before = cc.read_actual_state()
    mc.start_collection(tag=tag)

    result = runner.run_single_inference(
        prompt_length=prompt_length,
        output_length=output_length,
        benchmark_mode=True,
    )

    time.sleep(0.3)
    mc.stop_collection()
    energy = mc.compute_energy()

    state_after = cc.read_actual_state()

    result.update({
        'actual_gpu_mhz': state_after['actual_gpu_mhz'],
        'actual_cpu_mhz': state_after['actual_cpu_mhz'],
        'temperature_start_c': state_before['temperature_c'],
        'temperature_end_c': state_after['temperature_c'],
        'governor_gpu': state_after['governor_gpu'],
        'avg_power_w': energy.get('avg_power_w', 0),
        'max_power_w': energy.get('max_power_w', 0),
        'total_energy_j': energy.get('total_energy_j', 0),
    })
    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
        result['tokens_per_joule'] = result['output_tokens'] / result['total_energy_j']
    else:
        result['energy_per_token_j'] = 0
        result['tokens_per_joule'] = 0

    return result


def run_benchmark(model_path: str = 'models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf',
                 quick: bool = False,
                 cap_rate_table: str = None,
                 lock_rate_table: str = None):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/cap_selector_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem

    # Find latest rate tables if not specified
    if not cap_rate_table:
        cap_files = sorted(Path('data/rate_tables').glob('cap_rate_table_with_savings_*.parquet'))
        cap_rate_table = str(cap_files[-1]) if cap_files else None
    if not lock_rate_table:
        lock_files = sorted(Path('data/rate_tables').glob('lock_rate_table_*.parquet'))
        lock_rate_table = str(lock_files[-1]) if lock_files else None

    # Load selector
    selector = WorkloadCapSelector(
        cap_rate_table_path=cap_rate_table,
        lock_rate_table_path=lock_rate_table,
        model=model_name,
    )

    workloads = REPRESENTATIVE_WORKLOADS
    if quick:
        workloads = {k: v for i, (k, v) in enumerate(workloads.items()) if i < 3}
        repeats = 2
    else:
        repeats = REPEATS

    # Determine which strategies to test
    strategies = {
        'pareto': {'strategy': 'pareto'},
        'slo_50ms': {'strategy': 'slo_constrained', 'tpot_slo_ms': 50.0},
        'slo_45ms': {'strategy': 'slo_constrained', 'tpot_slo_ms': 45.0},
        'min_energy': {'strategy': 'min_energy'},
        'alpha_03': {'strategy': 'alpha_weighted', 'alpha': 0.3},
        'alpha_07': {'strategy': 'alpha_weighted', 'alpha': 0.7},
        'pwr_45w': {'strategy': 'power_budget', 'power_budget_w': 45.0},
    }

    # Pre-compute which cap each strategy selects per workload
    plan = {}
    for wl_name, wl_params in workloads.items():
        plan[wl_name] = {}
        for strat_name, strat_kwargs in strategies.items():
            cfg = selector.select(
                wl_params['prompt_length'], wl_params['output_length'],
                **strat_kwargs
            )
            if cfg:
                plan[wl_name][strat_name] = cfg['gpu_cap_mhz']
                logger.info(f"Plan: {wl_name} {strat_name} → cap={cfg['gpu_cap_mhz']}MHz")

    # Count runs
    n_strategy_runs = sum(len(v) for v in plan.values()) * repeats
    n_baseline_runs = len(workloads) * 2 * repeats  # dynamic + maxn
    total = n_strategy_runs + n_baseline_runs
    logger.info(f"E2E Benchmark: {n_strategy_runs} strategy + {n_baseline_runs} baseline = {total} total")

    # Load model
    logger.info("Loading model...")
    runner = LlamaCppRunner(
        model_path=model_path,
        n_gpu_layers=-1, n_ctx=4096, n_threads=4
    )
    logger.info("Model loaded")

    mc = MetricsCollector(interval_ms=TEGRASTATS_MS, output_dir='data/raw_logs')
    cc = CapController()

    results: List[Dict] = []
    run_count = 0

    # Part 1: Strategy runs
    logger.info("\n" + "=" * 60)
    logger.info("PART 1: Strategy-selected caps")
    logger.info("=" * 60)

    for wl_name, wl_params in workloads.items():
        for strat_name, gpu_cap in plan[wl_name].items():
            for rep in range(repeats):
                run_count += 1
                pl = wl_params['prompt_length']
                ol = wl_params['output_length']
                tag = f"{strat_name}_{wl_name}_rep{rep}"

                logger.info(f"\n[{run_count}/{total}] {strat_name} {wl_name} cap={gpu_cap}MHz rep={rep}")

                try:
                    result = run_with_cap(runner, mc, cc, gpu_cap, pl, ol, tag)
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'run_type': 'strategy',
                        'strategy': strat_name,
                        'gpu_cap_mhz': gpu_cap,
                        'workload': wl_name,
                        'prompt_length': pl,
                        'output_length': ol,
                        'repeat': rep,
                        'model': model_name,
                        **result,
                    }
                    results.append(record)
                    logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                               f"E/tok={result.get('energy_per_token_j',0):.4f}J "
                               f"Pwr={result.get('avg_power_w',0):.1f}W "
                               f"actual_GPU={result.get('actual_gpu_mhz',0)}")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    results.append({
                        'timestamp': datetime.now().isoformat(),
                        'run_type': 'strategy', 'strategy': strat_name,
                        'workload': wl_name, 'repeat': rep,
                        'gpu_cap_mhz': gpu_cap, 'error': str(e),
                        'model': model_name,
                    })

                time.sleep(COOLDOWN_S)

    # Part 2: Dynamic baselines
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Dynamic Baseline")
    logger.info("=" * 60)

    dynamic_results = {}
    for wl_name, wl_params in workloads.items():
        for rep in range(repeats):
            run_count += 1
            pl = wl_params['prompt_length']
            ol = wl_params['output_length']
            tag = f"dynamic_{wl_name}_rep{rep}"

            logger.info(f"\n[{run_count}/{total}] DYNAMIC {wl_name} rep={rep}")

            try:
                result = run_dynamic(runner, mc, cc, pl, ol, tag)
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'run_type': 'baseline',
                    'strategy': 'dynamic',
                    'gpu_cap_mhz': 1300,
                    'workload': wl_name,
                    'prompt_length': pl,
                    'output_length': ol,
                    'repeat': rep,
                    'model': model_name,
                    **result,
                }
                results.append(record)
                dynamic_results[wl_name] = result
                logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                           f"E/tok={result.get('energy_per_token_j',0):.4f}J "
                           f"Pwr={result.get('avg_power_w',0):.1f}W")
            except Exception as e:
                logger.error(f"  FAILED: {e}")

            time.sleep(COOLDOWN_S)

    # Part 3: MAXN baselines
    logger.info("\n" + "=" * 60)
    logger.info("PART 3: MAXN Baseline")
    logger.info("=" * 60)

    for wl_name, wl_params in workloads.items():
        for rep in range(repeats):
            run_count += 1
            pl = wl_params['prompt_length']
            ol = wl_params['output_length']
            tag = f"maxn_{wl_name}_rep{rep}"

            logger.info(f"\n[{run_count}/{total}] MAXN {wl_name} rep={rep}")

            try:
                result = run_maxn(runner, mc, cc, pl, ol, tag)
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'run_type': 'baseline',
                    'strategy': 'maxn',
                    'gpu_cap_mhz': 1300,
                    'workload': wl_name,
                    'prompt_length': pl,
                    'output_length': ol,
                    'repeat': rep,
                    'model': model_name,
                    **result,
                }
                results.append(record)
                logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                           f"E/tok={result.get('energy_per_token_j',0):.4f}J "
                           f"Pwr={result.get('avg_power_w',0):.1f}W")
            except Exception as e:
                logger.error(f"  FAILED: {e}")

            time.sleep(COOLDOWN_S)

    # Save
    df = pd.DataFrame(results)
    out_path = output_dir / f'cap_selector_benchmark_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path} ({len(df)} rows)")

    # Restore dynamic
    try:
        cc.restore_dynamic()
    except Exception as e:
        logger.warning(f"restore_dynamic() failed: {e}")

    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='E2E Cap Selector Benchmark')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--model', type=str, default='models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf')
    parser.add_argument('--cap-rate-table', type=str, default=None)
    parser.add_argument('--lock-rate-table', type=str, default=None)
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        quick=args.quick,
        cap_rate_table=args.cap_rate_table,
        lock_rate_table=args.lock_rate_table,
    )
