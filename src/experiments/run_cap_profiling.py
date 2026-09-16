#!/usr/bin/env python3
"""
Cap Profiling Experiment (Phase 10)

Profiles LLM inference under frequency cap mode (dynamic governor + max_freq cap).
Unlike lock-mode profiling which forces exact frequencies, cap profiling lets the
governor adapt within the cap boundary — reflecting real online deployment behavior.

Experiment matrix:
  GPU cap: 612 / 816 / 1020 / 1300
  EMC cap: 204 / 665 / 2133 / 3199
  CPU cap: 1036 (fixed)
  Workloads: 6
  Repeats: 3
  = 288 runs

Plus baseline: default_dynamic (no cap) × 6 workloads × 3 repeats = 18 runs
Total: ~306 runs ≈ 4-5 hours

Usage:
    sudo python3 src/experiments/run_cap_profiling.py
    sudo python3 src/experiments/run_cap_profiling.py --quick
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Experiment config ──
GPU_CAPS = [612, 816, 1020, 1300]
EMC_CAPS = [204, 665, 2133, 3199]
CPU_CAP = 1036  # Fixed for first version

WORKLOADS = {
    'p64_o64':     {'prompt_length': 64,   'output_length': 64},
    'p128_o128':   {'prompt_length': 128,  'output_length': 128},
    'p128_o512':   {'prompt_length': 128,  'output_length': 512},
    'p256_o256':   {'prompt_length': 256,  'output_length': 256},
    'p512_o128':   {'prompt_length': 512,  'output_length': 128},
    'p512_o512':   {'prompt_length': 512,  'output_length': 512},
    'p512_o1024':  {'prompt_length': 512,  'output_length': 1024},
    'p1024_o128':  {'prompt_length': 1024, 'output_length': 128},
    'p1024_o512':  {'prompt_length': 1024, 'output_length': 512},
    'p1024_o1024': {'prompt_length': 1024, 'output_length': 1024},
    'p2048_o128':  {'prompt_length': 2048, 'output_length': 128},
    'p2048_o512':  {'prompt_length': 2048, 'output_length': 512},
}

REPEATS = 3
WARMUP_RUNS = 1
COOLDOWN_S = 3
TEGRASTATS_MS = 500


def run_single_cap(
    runner: LlamaCppRunner,
    mc: MetricsCollector,
    cc: CapController,
    gpu_cap: int, emc_cap: int, cpu_cap: int,
    prompt_length: int, output_length: int,
) -> Dict:
    """Run inference under cap mode with energy measurement."""
    cap_info = cc.set_cap(gpu_cap, emc_cap, cpu_cap)
    runner.warmup(WARMUP_RUNS)

    tag = f"cap_g{gpu_cap}_e{emc_cap}_p{prompt_length}_o{output_length}"
    mc.start_collection(tag=tag)

    # Read state before inference
    state_before = cc.read_actual_state()

    result = runner.run_single_inference(
        prompt_length=prompt_length,
        output_length=output_length,
        benchmark_mode=True,
    )

    time.sleep(0.3)
    mc.stop_collection()
    energy = mc.compute_energy()

    # Read state after inference (actual frequencies during run)
    state_after = cc.read_actual_state()

    result.update({
        'control_mode': 'cap',
        'target_gpu_cap_mhz': gpu_cap,
        'target_emc_cap_mhz': emc_cap,
        'target_cpu_cap_mhz': cpu_cap,
        'actual_gpu_mhz': state_after['actual_gpu_mhz'],
        'actual_emc_mhz': state_after['actual_emc_mhz'],
        'actual_cpu_mhz': state_after['actual_cpu_mhz'],
        'governor_gpu': state_after['governor_gpu'],
        'governor_cpu': state_after['governor_cpu'],
        'temperature_start_c': state_before['temperature_c'],
        'temperature_end_c': state_after['temperature_c'],
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


def run_dynamic_baseline(
    runner: LlamaCppRunner,
    mc: MetricsCollector,
    cc: CapController,
    prompt_length: int, output_length: int,
) -> Dict:
    """Run inference under default dynamic governor (no cap)."""
    cc.restore_dynamic()
    runner.warmup(WARMUP_RUNS)

    state_before = cc.read_actual_state()

    tag = f"dyn_p{prompt_length}_o{output_length}"
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
        'control_mode': 'dynamic',
        'target_gpu_cap_mhz': 1300,
        'target_emc_cap_mhz': 3199,
        'target_cpu_cap_mhz': 1497,
        'actual_gpu_mhz': state_after['actual_gpu_mhz'],
        'actual_emc_mhz': state_after['actual_emc_mhz'],
        'actual_cpu_mhz': state_after['actual_cpu_mhz'],
        'governor_gpu': state_after['governor_gpu'],
        'governor_cpu': state_after['governor_cpu'],
        'temperature_start_c': state_before['temperature_c'],
        'temperature_end_c': state_after['temperature_c'],
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


def run_maxn_baseline(
    runner: LlamaCppRunner,
    mc: MetricsCollector,
    cc: CapController,
    prompt_length: int, output_length: int,
) -> Dict:
    """Run inference under MAXN: all frequencies locked at max, performance governor."""
    # Lock GPU/CPU/EMC to max frequencies
    GPU_SYSFS_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
    with open(f'{GPU_SYSFS_PATH}/governor', 'w') as f:
        f.write('performance')
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write('1300500000')
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write('1300500000')

    from pathlib import Path as PPath
    for cpu in PPath('/sys/devices/system/cpu').glob('cpu[0-9]*'):
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

    tag = f"maxn_p{prompt_length}_o{output_length}"
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
        'control_mode': 'maxn',
        'target_gpu_cap_mhz': 1300,
        'target_emc_cap_mhz': 3199,
        'target_cpu_cap_mhz': 2201,
        'actual_gpu_mhz': state_after['actual_gpu_mhz'],
        'actual_emc_mhz': state_after['actual_emc_mhz'],
        'actual_cpu_mhz': state_after['actual_cpu_mhz'],
        'governor_gpu': state_after['governor_gpu'],
        'governor_cpu': state_after['governor_cpu'],
        'temperature_start_c': state_before['temperature_c'],
        'temperature_end_c': state_after['temperature_c'],
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


def run_experiment(quick: bool = False, model_path: str = 'models/gguf/Phi-3-mini-4k-instruct-q4.gguf', gpu_only: bool = False):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/cap_profiling')
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem

    if quick:
        gpu_caps = [816, 1300]
        emc_caps = [3199] if gpu_only else [204, 3199]
        workloads = {
            'p128_o128':  {'prompt_length': 128, 'output_length': 128},
            'p128_o1024': {'prompt_length': 128, 'output_length': 1024},
        }
        repeats = 2
    else:
        gpu_caps = GPU_CAPS
        emc_caps = [3199] if gpu_only else EMC_CAPS
        workloads = WORKLOADS
        repeats = REPEATS

    cap_runs = len(gpu_caps) * len(emc_caps) * len(workloads) * repeats
    dyn_runs = len(workloads) * repeats
    maxn_runs = len(workloads) * repeats
    total = cap_runs + dyn_runs + maxn_runs
    logger.info(f"Cap Profiling: {cap_runs} cap + {dyn_runs} dynamic + {maxn_runs} MAXN = {total} total runs")

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

    # ── Part 1: Cap profiling ──
    logger.info("\n" + "=" * 60)
    logger.info("PART 1: Cap Profiling")
    logger.info("=" * 60)

    for gpu_cap in gpu_caps:
        for emc_cap in emc_caps:
            for wl_name, wl_params in workloads.items():
                for rep in range(repeats):
                    run_count += 1
                    pl = wl_params['prompt_length']
                    ol = wl_params['output_length']

                    logger.info(f"\n[{run_count}/{total}] CAP GPU<={gpu_cap} EMC<={emc_cap} "
                               f"{wl_name} rep={rep}")

                    try:
                        result = run_single_cap(
                            runner, mc, cc, gpu_cap, emc_cap, CPU_CAP, pl, ol
                        )
                        record = {
                            'timestamp': datetime.now().isoformat(),
                            'control_mode': 'cap',
                            'gpu_cap_mhz': gpu_cap,
                            'emc_cap_mhz': emc_cap,
                            'cpu_cap_mhz': CPU_CAP,
                            'workload': wl_name,
                            'prompt_length': pl,
                            'output_length': ol,
                            'target_output_tokens': ol,
                            'repeat': rep,
                            **{k: v for k, v in result.items()
                               if k not in ('timestamp',)},
                            'model': model_name,
                            'experiment': 'cap_profiling',
                        }
                        results.append(record)
                        logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                                   f"E/tok={result.get('energy_per_token_j',0):.4f}J "
                                   f"actual_gpu={result.get('actual_gpu_mhz',0)} "
                                   f"gov={result.get('governor_gpu','?')}")
                    except Exception as e:
                        logger.error(f"  FAILED: {e}")
                        results.append({
                            'timestamp': datetime.now().isoformat(),
                            'control_mode': 'cap', 'gpu_cap_mhz': gpu_cap,
                            'emc_cap_mhz': emc_cap, 'workload': wl_name,
                            'repeat': rep, 'error': str(e),
                        })

                    time.sleep(COOLDOWN_S)

    # ── Part 2: Dynamic baseline ──
    logger.info("\n" + "=" * 60)
    logger.info("PART 2: Dynamic Baseline (no cap)")
    logger.info("=" * 60)

    for wl_name, wl_params in workloads.items():
        for rep in range(repeats):
            run_count += 1
            pl = wl_params['prompt_length']
            ol = wl_params['output_length']

            logger.info(f"\n[{run_count}/{total}] DYNAMIC {wl_name} rep={rep}")

            try:
                result = run_dynamic_baseline(runner, mc, cc, pl, ol)
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'control_mode': 'dynamic',
                    'gpu_cap_mhz': 1300,  # No cap = max
                    'emc_cap_mhz': 3199,
                    'cpu_cap_mhz': 1497,
                    'workload': wl_name,
                    'prompt_length': pl,
                    'output_length': ol,
                    'target_output_tokens': ol,
                    'repeat': rep,
                    **{k: v for k, v in result.items()
                       if k not in ('timestamp',)},
                    'model': model_name,
                    'experiment': 'cap_profiling',
                }
                results.append(record)
                logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                           f"E/tok={result.get('energy_per_token_j',0):.4f}J "
                           f"actual_gpu={result.get('actual_gpu_mhz',0)}")
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                results.append({
                    'timestamp': datetime.now().isoformat(),
                    'control_mode': 'dynamic', 'workload': wl_name,
                    'repeat': rep, 'error': str(e),
                })

            time.sleep(COOLDOWN_S)

    # ── Part 3: MAXN baseline ──
    logger.info("\n" + "=" * 60)
    logger.info("PART 3: MAXN Baseline (all max, performance governor)")
    logger.info("=" * 60)

    for wl_name, wl_params in workloads.items():
        for rep in range(repeats):
            run_count += 1
            pl = wl_params['prompt_length']
            ol = wl_params['output_length']

            logger.info(f"\n[{run_count}/{total}] MAXN {wl_name} rep={rep}")

            try:
                result = run_maxn_baseline(runner, mc, cc, pl, ol)
                record = {
                    'timestamp': datetime.now().isoformat(),
                    'control_mode': 'maxn',
                    'gpu_cap_mhz': 1300,
                    'emc_cap_mhz': 3199,
                    'cpu_cap_mhz': 2201,
                    'workload': wl_name,
                    'prompt_length': pl,
                    'output_length': ol,
                    'target_output_tokens': ol,
                    'repeat': rep,
                    **{k: v for k, v in result.items()
                       if k not in ('timestamp',)},
                    'model': model_name,
                    'experiment': 'cap_profiling',
                }
                results.append(record)
                logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                           f"E/tok={result.get('energy_per_token_j',0):.4f}J "
                           f"Pwr={result.get('avg_power_w',0):.1f}W")
            except Exception as e:
                logger.error(f"  FAILED: {e}")
                results.append({
                    'timestamp': datetime.now().isoformat(),
                    'control_mode': 'maxn', 'workload': wl_name,
                    'repeat': rep, 'error': str(e),
                })

            time.sleep(COOLDOWN_S)

    # ── Save ──
    df = pd.DataFrame(results)
    out_path = output_dir / f'cap_profiling_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path} ({len(df)} rows)")

    try:
        cc.restore_dynamic()
    except Exception as e:
        logger.warning(f"restore_dynamic() failed: {e} (data saved)")

    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cap Profiling Experiment')
    parser.add_argument('--quick', action='store', help='Quick mode')
    parser.add_argument('--model', type=str, default='models/gguf/Phi-3-mini-4k-instruct-q4.gguf',
                        help='Path to GGUF model file')
    parser.add_argument('--gpu-only', action='store_true',
                        help='GPU-only sweep (skip EMC caps, fix at max)')
    args = parser.parse_args()

    run_experiment(quick=bool(args.quick), model_path=args.model, gpu_only=args.gpu_only)
