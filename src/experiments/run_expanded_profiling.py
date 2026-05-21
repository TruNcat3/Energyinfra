#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Expanded Energy Rate Table Profiling Experiment
Extends the workload matrix for comprehensive rate table coverage and rule mining.

Key differences from run_profiling_experiment.py:
  - Expanded workload grid: 12 workloads covering prompt 32-2048 × output 32-512
  - Reduced frequency configs: 4 GPU × 2 CPU (skip CPU2201 which showed marginal benefit)
  - Two phases: mixed + decode_only (for phase-aware rate table)
  - Designed for rate table construction and Phase-Aware DVFS rule mining
"""

import sys
import os
from pathlib import Path

# Ensure CWD is project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.experiments.run_profiling_experiment import (
    run_profiling_experiment,
    GPU_FREQS, CPU_FREQS, WORKLOADS, PHASES,
    set_gpu_freq, set_cpu_freq, restore_max_freq,
    run_single_with_energy,
    GPU_SYSFS_PATH, CPU_SYSFS_PATH,
    REPEATS, WARMUP_RUNS, COOLDOWN_SECONDS, TEGRASTATS_INTERVAL_MS,
)
import csv
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.metrics.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Expanded Frequency Configurations ──
# Skip CPU 2201 (marginal benefit, ~1W extra power for ~0.3ms TPOT improvement)
EXP_GPU_FREQS = [306, 612, 918, 1300]
EXP_CPU_FREQS = [1036, 1497]

# ── Expanded Workload Grid ──
# Covers prompt_length: 32, 64, 128, 256, 512, 1024, 2048
# Covers output_length: 32, 64, 128, 256, 512
EXP_WORKLOADS = {
    # Row 1: Small prompt (32 tokens)
    'tiny_tiny':      {'prompt_length': 32,   'output_length': 32},
    'tiny_short':     {'prompt_length': 32,   'output_length': 128},
    'tiny_long':      {'prompt_length': 32,   'output_length': 512},
    # Row 2: Medium-short prompt (128 tokens)
    'short_tiny':     {'prompt_length': 128,  'output_length': 32},
    'short_short':    {'prompt_length': 128,  'output_length': 64},
    'short_long':     {'prompt_length': 128,  'output_length': 256},
    # Row 3: Medium prompt (512 tokens)
    'medium_tiny':    {'prompt_length': 512,  'output_length': 32},
    'medium_short':   {'prompt_length': 512,  'output_length': 128},
    'medium_long':    {'prompt_length': 512,  'output_length': 512},
    # Row 4: Large prompt (1024 tokens)
    'long_tiny':      {'prompt_length': 1024, 'output_length': 32},
    'long_medium':    {'prompt_length': 1024, 'output_length': 128},
    # Row 5: Extra-large prompt (2048 tokens)
    'xlong_short':    {'prompt_length': 2048, 'output_length': 64},
    'xlong_medium':   {'prompt_length': 2048, 'output_length': 128},
    'xlong_long':     {'prompt_length': 2048, 'output_length': 512},
}

EXP_PHASES = ['mixed', 'decode_only']
EXP_REPEATS = 3


def run_expanded_profiling():
    """Run expanded workload energy profiling for rate table construction."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/energy_profiling')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model...")
    runner = LlamaCppRunner(
        model_path='models/gguf/Phi-3-mini-4k-instruct-q4.gguf',
        n_gpu_layers=-1, n_ctx=4096, n_threads=4
    )
    logger.info("Model loaded")

    mc = MetricsCollector(interval_ms=TEGRASTATS_INTERVAL_MS, output_dir='data/raw_logs')

    # Set GPU governor
    with open(f'{GPU_SYSFS_PATH}/governor', 'w') as f:
        f.write('userspace')

    results: List[Dict] = []
    total_runs = len(EXP_GPU_FREQS) * len(EXP_CPU_FREQS) * len(EXP_WORKLOADS) * len(EXP_PHASES) * EXP_REPEATS
    run_count = 0

    for gpu_mhz in EXP_GPU_FREQS:
        for cpu_mhz in EXP_CPU_FREQS:
            config_name = f"GPU{gpu_mhz}_CPU{cpu_mhz}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Config: {config_name}")
            logger.info(f"{'='*60}")

            actual_gpu = set_gpu_freq(gpu_mhz)
            actual_cpu = set_cpu_freq(cpu_mhz)
            time.sleep(1)
            logger.info(f"  Set: GPU={actual_gpu}MHz, CPU={actual_cpu}MHz")

            logger.info(f"  Warmup ({WARMUP_RUNS} runs)...")
            runner.warmup(WARMUP_RUNS)

            for phase in EXP_PHASES:
                for wl_name, wl_params in EXP_WORKLOADS.items():
                    for rep in range(EXP_REPEATS):
                        run_count += 1
                        logger.info(f"  [{run_count}/{total_runs}] {config_name} {wl_name} {phase} rep={rep}")

                        try:
                            result = run_single_with_energy(
                                runner, mc,
                                prompt_length=wl_params['prompt_length'],
                                output_length=wl_params['output_length'],
                                phase=phase
                            )
                        except Exception as e:
                            logger.error(f"    FAILED: {e}")
                            result = {
                                'ttft_ms': 0, 'tpot_ms': 0, 'total_time_ms': 0,
                                'tokens_per_second': 0, 'output_tokens': 0,
                                'total_energy_j': 0, 'avg_power_w': 0,
                                'energy_per_token_j': 0, 'tokens_per_joule': 0,
                                'error': str(e), 'phase': phase,
                            }

                        record = {
                            'timestamp': datetime.now().isoformat(),
                            'config_name': config_name,
                            'gpu_freq_mhz': actual_gpu,
                            'cpu_freq_mhz': actual_cpu,
                            'target_gpu_freq_mhz': gpu_mhz,
                            'target_cpu_freq_mhz': cpu_mhz,
                            'workload': wl_name,
                            'phase': result.get('phase', phase),
                            'prompt_length': wl_params['prompt_length'],
                            'output_length': wl_params['output_length'],
                            'repeat': rep,
                            'ttft_ms': result['ttft_ms'],
                            'tpot_ms': result['tpot_ms'],
                            'total_time_ms': result.get('total_time_ms', 0),
                            'output_tokens': result.get('output_tokens', 0),
                            'tokens_per_second': result.get('tokens_per_second', 0),
                            'total_energy_j': result.get('total_energy_j', 0),
                            'avg_power_w': result.get('avg_power_w', 0),
                            'max_power_w': result.get('max_power_w', 0),
                            'energy_per_token_j': result.get('energy_per_token_j', 0),
                            'tokens_per_joule': result.get('tokens_per_joule', 0),
                            'avg_gpu_soc_w': result.get('avg_gpu_soc_w', 0),
                            'avg_cpu_cv_w': result.get('avg_cpu_cv_w', 0),
                            'avg_sys_5v0_w': result.get('avg_sys_5v0_w', 0),
                            'avg_temp_cpu_c': result.get('avg_temp_cpu_c', 0),
                            'max_temp_cpu_c': result.get('max_temp_cpu_c', 0),
                            'power_samples': result.get('power_samples', 0),
                            'model': 'Phi-3-mini-Q4',
                            'runtime': 'llama.cpp',
                            'phase_note': result.get('phase_note', ''),
                            'experiment': 'expanded_rate_table',
                        }
                        if 'error' in result:
                            record['error'] = result['error']
                        results.append(record)

                        ept = result.get('energy_per_token_j', 0)
                        tpj = result.get('tokens_per_joule', 0)
                        logger.info(f"    TTFT={result['ttft_ms']:.1f}ms TPOT={result['tpot_ms']:.1f}ms "
                                   f"tok={result.get('output_tokens', 0)} tps={result.get('tokens_per_second', 0):.1f} "
                                   f"power={result.get('avg_power_w', 0):.2f}W "
                                   f"E/token={ept:.4f}J tok/J={tpj:.2f}")

            logger.info(f"  Cooldown {COOLDOWN_SECONDS}s...")
            time.sleep(COOLDOWN_SECONDS)

    restore_max_freq()

    df = pd.DataFrame(results)
    out_path = output_dir / f'expanded_profiling_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(df)} rows)")

    # Quick summary
    valid = df[df['output_tokens'] > 0]
    logger.info(f"Valid runs: {len(valid)}/{len(df)}")
    if not valid.empty:
        for phase in valid['phase'].unique():
            pv = valid[valid['phase'] == phase]
            best = pv.loc[pv['energy_per_token_j'].idxmin()]
            logger.info(f"  {phase}: best={best['config_name']} E/tok={best['energy_per_token_j']:.3f}J")

    return out_path


if __name__ == '__main__':
    n_workloads = len(EXP_WORKLOADS)
    n_configs = len(EXP_GPU_FREQS) * len(EXP_CPU_FREQS)
    n_phases = len(EXP_PHASES)
    total = n_configs * n_workloads * n_phases * EXP_REPEATS
    logger.info(f"Expanded Rate Table Experiment")
    logger.info(f"  Configs: {n_configs} ({len(EXP_GPU_FREQS)} GPU x {len(EXP_CPU_FREQS)} CPU)")
    logger.info(f"  Workloads: {n_workloads}")
    logger.info(f"  Phases: {n_phases} ({', '.join(EXP_PHASES)})")
    logger.info(f"  Repeats: {EXP_REPEATS}")
    logger.info(f"  Total runs: {total}")
    logger.info(f"  Estimated time: ~{total * 25 // 60} min")
    run_expanded_profiling()
