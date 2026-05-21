#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Fine-Grained GPU×EMC Energy Profiling Experiment

Fixes from previous experiments:
  - Uses ALL 11 available GPU frequencies (not just 4)
  - Includes EMC frequency as a DVFS dimension (4 levels via debugfs)
  - Uses performance governor for reliable GPU frequency locking
  - CPU fixed at 1036 MHz (confirmed optimal from prior data)

Experiment matrix:
  11 GPU × 4 EMC × 1 CPU = 44 configs
  × 5 workloads × 2 phases (decode, mixed) × 3 repeats
  = 1320 runs (~8-9 hours)

  Use --decode-only or --mixed-only to run a subset.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import csv
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.metrics.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Full Hardware Frequency Configurations ──
# All 11 GPU frequencies available on Jetson Orin
GPU_FREQS = [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300]
GPU_SYSFS_HZ = {
    306:  306000000,
    408:  408000000,
    510:  510000000,
    612:  612000000,
    714:  714000000,
    816:  816000000,
    918:  918000000,
    1020: 1020000000,
    1122: 1122000000,
    1224: 1224000000,
    1300: 1300500000,
}

# All 4 EMC frequencies (via debugfs)
EMC_FREQS = [204, 665, 2133, 3199]
EMC_HZ = {204: 204000000, 665: 665600000, 2133: 2133000000, 3199: 3199000000}

# CPU: fixed at 1036 MHz (confirmed optimal for energy efficiency)
CPU_FREQ = 1036
CPU_SYSFS_HZ = {1036: 1036800, 1497: 1497600}

GPU_SYSFS_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
CPU_SYSFS_PATH = '/sys/devices/system/cpu/cpu0/cpufreq'
EMC_DEBUGFS_MAX = '/sys/kernel/debug/emc/max_rate'
EMC_DEBUGFS_MIN = '/sys/kernel/debug/emc/min_rate'
EMC_DEBUGFS_CLK = '/sys/kernel/debug/clk/emc/clk_rate'

WORKLOADS = {
    'p128_o64':   {'prompt_length': 128,  'output_length': 64},
    'p512_o128':  {'prompt_length': 512,  'output_length': 128},
    'p1024_o128': {'prompt_length': 1024, 'output_length': 128},
    'p128_o256':  {'prompt_length': 128,  'output_length': 256},
    'p1024_o512': {'prompt_length': 1024, 'output_length': 512},
}

REPEATS = 3
WARMUP_RUNS = 2
COOLDOWN_S = 3
TEGRASTATS_INTERVAL_MS = 500


def set_gpu_freq(mhz: int) -> int:
    target_hz = GPU_SYSFS_HZ[mhz]
    with open(f'{GPU_SYSFS_PATH}/governor', 'w') as f:
        f.write('performance')
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write(str(target_hz))
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write(str(target_hz))
    time.sleep(0.3)
    actual = int(open(f'{GPU_SYSFS_PATH}/cur_freq').read().strip()) // 1000000
    return actual


def set_cpu_freq(mhz: int) -> int:
    target_hz = CPU_SYSFS_HZ[mhz]
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov = cpu / 'cpufreq' / 'scaling_governor'
        mx = cpu / 'cpufreq' / 'scaling_max_freq'
        mn = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov.exists():
            try:
                open(gov, 'w').write('userspace')
                open(mx, 'w').write(str(target_hz))
                open(mn, 'w').write(str(target_hz))
            except PermissionError:
                pass
    time.sleep(0.1)
    return int(open(f'{CPU_SYSFS_PATH}/scaling_cur_freq').read().strip()) // 1000


def set_emc_freq(mhz: int) -> int:
    target_hz = EMC_HZ[mhz]
    max_emc = max(EMC_HZ.values())
    # Widen range first to avoid max < min constraint
    with open(EMC_DEBUGFS_MAX, 'w') as f:
        f.write(str(max_emc))
    with open(EMC_DEBUGFS_MIN, 'w') as f:
        f.write('204000000')
    # Now set to target
    with open(EMC_DEBUGFS_MIN, 'w') as f:
        f.write(str(target_hz))
    with open(EMC_DEBUGFS_MAX, 'w') as f:
        f.write(str(target_hz))
    time.sleep(0.3)
    actual_hz = int(open(EMC_DEBUGFS_CLK).read().strip())
    return actual_hz // 1000000


def restore_max_freq():
    max_gpu = max(GPU_SYSFS_HZ.values())
    min_gpu = min(GPU_SYSFS_HZ.values())
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write(str(max_gpu))
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write(str(min_gpu))
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov = cpu / 'cpufreq' / 'scaling_governor'
        mx = cpu / 'cpufreq' / 'scaling_max_freq'
        mn = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov.exists():
            try:
                open(gov, 'w').write('schedutil')
                open(mx, 'w').write(str(max(CPU_SYSFS_HZ.values())))
                open(mn, 'w').write(str(min(CPU_SYSFS_HZ.values())))
            except PermissionError:
                pass
    with open(EMC_DEBUGFS_MIN, 'w') as f:
        f.write('204000000')
    with open(EMC_DEBUGFS_MAX, 'w') as f:
        f.write('3199000000')


def run_single_with_energy(runner, mc, prompt_length, output_length, phase='mixed') -> Dict:
    tag = f"p{prompt_length}_o{output_length}_{phase}"
    mc.start_collection(tag=tag)

    if phase == 'decode':
        result = runner.run_single_inference(prompt_length=16, output_length=output_length)
        result['phase'] = 'decode'
        result['phase_note'] = f'decode-only (p=16, o={output_length})'
    else:
        result = runner.run_single_inference(prompt_length=prompt_length, output_length=output_length)
        result['phase'] = 'mixed'

    time.sleep(0.3)
    mc.stop_collection()

    energy = mc.compute_energy()
    result.update({
        'avg_power_w': energy.get('avg_power_w', 0),
        'max_power_w': energy.get('max_power_w', 0),
        'total_energy_j': energy.get('total_energy_j', 0),
        'avg_gpu_soc_w': energy.get('avg_gpu_soc_w', 0),
        'avg_cpu_cv_w': energy.get('avg_cpu_cv_w', 0),
        'avg_sys_5v0_w': energy.get('avg_sys_5v0_w', 0),
        'avg_temp_cpu_c': energy.get('avg_temp_cpu_c', 0),
        'power_samples': energy.get('n_samples', 0),
    })

    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
        result['tokens_per_joule'] = result['output_tokens'] / result['total_energy_j']
    else:
        result['energy_per_token_j'] = 0
        result['tokens_per_joule'] = 0
    return result


def run_finegrained_profiling(phases=('decode',)):
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

    # Set CPU to fixed optimal frequency
    actual_cpu = set_cpu_freq(CPU_FREQ)
    logger.info(f"CPU fixed at {actual_cpu} MHz")

    n_configs = len(GPU_FREQS) * len(EMC_FREQS)
    n_phases = len(phases)
    total_runs = n_configs * len(WORKLOADS) * n_phases * REPEATS
    run_count = 0
    results: List[Dict] = []

    for gpu_mhz in GPU_FREQS:
        for emc_mhz in EMC_FREQS:
            config_name = f"GPU{gpu_mhz}_EMC{emc_mhz}_CPU{CPU_FREQ}"

            logger.info(f"\n{'='*60}")
            logger.info(f"Config: {config_name}  [{run_count}/{total_runs}]")
            logger.info(f"{'='*60}")

            actual_gpu = set_gpu_freq(gpu_mhz)
            actual_emc = set_emc_freq(emc_mhz)
            time.sleep(0.5)
            logger.info(f"  Set: GPU={actual_gpu}MHz, EMC={actual_emc}MHz, CPU={actual_cpu}MHz")

            logger.info(f"  Warmup ({WARMUP_RUNS} runs)...")
            runner.warmup(WARMUP_RUNS)

            for phase in phases:
                for wl_name, wl_params in WORKLOADS.items():
                    for rep in range(REPEATS):
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
                            'emc_freq_mhz': actual_emc,
                            'target_gpu_mhz': gpu_mhz,
                            'target_emc_mhz': emc_mhz,
                            'workload': wl_name,
                            'prompt_length': wl_params['prompt_length'],
                            'output_length': wl_params['output_length'],
                            'phase': result.get('phase', phase),
                            'repeat': rep,
                            'ttft_ms': result.get('ttft_ms', 0),
                            'tpot_ms': result.get('tpot_ms', 0),
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
                            'power_samples': result.get('power_samples', 0),
                            'model': 'Phi-3-mini-Q4',
                            'runtime': 'llama.cpp',
                            'experiment': 'finegrained_gpu_emc',
                        }
                        if 'error' in result:
                            record['error'] = result['error']
                        results.append(record)

                        ept = result.get('energy_per_token_j', 0)
                        logger.info(f"    TPOT={result.get('tpot_ms',0):.1f}ms "
                                   f"tps={result.get('tokens_per_second',0):.1f} "
                                   f"Pwr={result.get('avg_power_w',0):.1f}W "
                                   f"E/tok={ept:.4f}J")

            time.sleep(COOLDOWN_S)

    restore_max_freq()

    df = pd.DataFrame(results)
    out_path = output_dir / f'finegrained_profiling_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path} ({len(df)} rows)")

    valid = df[df['output_tokens'] > 0]
    if not valid.empty:
        report_path = output_dir / f'finegrained_summary_{timestamp}.md'
        with open(report_path, 'w') as f:
            f.write(generate_report(valid, phases))
        logger.info(f"Report: {report_path}")

    return out_path


def generate_report(df: pd.DataFrame, phases) -> str:
    lines = [
        "# Fine-Grained GPU×EMC Profiling Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Phi-3-mini-Q4",
        f"**GPU levels**: {len(df['gpu_freq_mhz'].unique())} (306-1300 MHz, 11 steps)",
        f"**EMC levels**: {len(df['emc_freq_mhz'].unique())} (204-3199 MHz)",
        f"**CPU**: 1036 MHz (fixed)",
        f"**Phases**: {', '.join(phases)}",
        f"**Total valid runs**: {len(df)}",
    ]

    # GPU effect (across all EMC)
    lines.extend(["", "## GPU Frequency Effect (averaged across EMC)", "",
                  "| GPU (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |",
                  "|:---:|:---:|:---:|:---:|:---:|:---:|"])
    for gpu in sorted(df['gpu_freq_mhz'].unique()):
        g = df[df['gpu_freq_mhz'] == gpu]
        lines.append(f"| {gpu} | {g['energy_per_token_j'].mean():.4f} | "
                    f"{g['tpot_ms'].mean():.1f} | {g['tokens_per_second'].mean():.1f} | "
                    f"{g['avg_power_w'].mean():.1f} | {g['tokens_per_joule'].mean():.2f} |")

    # EMC effect (across all GPU)
    lines.extend(["", "## EMC Frequency Effect (averaged across GPU)", "",
                  "| EMC (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |",
                  "|:---:|:---:|:---:|:---:|:---:|:---:|"])
    for emc in sorted(df['emc_freq_mhz'].unique()):
        e = df[df['emc_freq_mhz'] == emc]
        lines.append(f"| {emc} | {e['energy_per_token_j'].mean():.4f} | "
                    f"{e['tpot_ms'].mean():.1f} | {e['tokens_per_second'].mean():.1f} | "
                    f"{e['avg_power_w'].mean():.1f} | {e['tokens_per_joule'].mean():.2f} |")

    # Best config per workload
    lines.extend(["", "## Best E/tok Config per Workload", "",
                  "| Workload | Best Config | E/tok (J) | TPOT (ms) |",
                  "|:---:|:---:|:---:|:---:|"])
    for wl in sorted(df['workload'].unique()):
        w = df[df['workload'] == wl]
        best = w.loc[w['energy_per_token_j'].idxmin()]
        lines.append(f"| {wl} | GPU{int(best['gpu_freq_mhz'])}_EMC{int(best['emc_freq_mhz'])} | "
                    f"{best['energy_per_token_j']:.4f} | {best['tpot_ms']:.1f} |")

    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fine-Grained GPU×EMC Energy Profiling')
    parser.add_argument('--decode-only', action='store_true', help='Only decode phase')
    parser.add_argument('--mixed-only', action='store_true', help='Only mixed phase')
    parser.add_argument('--all', action='store_true', help='Both decode and mixed phases')
    args = parser.parse_args()

    if args.all:
        phases = ('decode', 'mixed')
    elif args.mixed_only:
        phases = ('mixed',)
    else:
        phases = ('decode',)

    n_configs = len(GPU_FREQS) * len(EMC_FREQS)
    total = n_configs * len(WORKLOADS) * len(phases) * REPEATS

    logger.info(f"Fine-Grained GPU×EMC Profiling Experiment")
    logger.info(f"  GPU: {len(GPU_FREQS)} frequencies {GPU_FREQS}")
    logger.info(f"  EMC: {len(EMC_FREQS)} frequencies {EMC_FREQS}")
    logger.info(f"  CPU: {CPU_FREQ} MHz (fixed)")
    logger.info(f"  Workloads: {len(WORKLOADS)}")
    logger.info(f"  Phases: {phases}")
    logger.info(f"  Total configs: {n_configs}")
    logger.info(f"  Total runs: {total}")
    logger.info(f"  Estimated time: ~{total * 25 // 60} min ({total * 25 / 3600:.1f} hours)")

    run_finegrained_profiling(phases=phases)
