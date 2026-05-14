#!/usr/bin/env python3
"""
Real Model Energy Profiling Experiment
Runs llama.cpp inference with tegrastats power measurement under different frequency settings.

Corrected from v1 (which only collected latency):
  - Integrates tegrastats for real power/energy measurement
  - Supports per-phase profiling (prefill-only / decode-only)
  - Records energy/token, tokens/J for rate table construction

Experiment matrix: GPU × CPU × workloads × phase × repeats
"""

import csv
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Frequency Configurations ──
# Use fewer configs for energy profiling (focused sweep)
GPU_FREQS = [306, 612, 918, 1300]
CPU_FREQS = [1036, 1497, 2201]

# sysfs values
GPU_SYSFS = {306: 306000000, 612: 612000000, 918: 918000000, 1300: 1300500000}
CPU_SYSFS = {1036: 1036800, 1497: 1497600, 2201: 2201600}

# ── Workload Definitions ──
WORKLOADS = {
    'short_short':   {'prompt_length': 128,  'output_length': 64},
    'short_long':    {'prompt_length': 128,  'output_length': 256},
    'medium_medium': {'prompt_length': 512,  'output_length': 128},
    'long_medium':   {'prompt_length': 1024, 'output_length': 128},
    'long_long':     {'prompt_length': 1024, 'output_length': 512},
}

# Phase definitions for per-phase profiling
PHASES = ['mixed', 'prefill_only', 'decode_only']

GPU_SYSFS_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
CPU_SYSFS_PATH = '/sys/devices/system/cpu/cpu0/cpufreq'

REPEATS = 3
WARMUP_RUNS = 2
COOLDOWN_SECONDS = 5
TEGRASTATS_INTERVAL_MS = 500


def set_gpu_freq(mhz: int) -> int:
    target_hz = GPU_SYSFS[mhz]
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write(str(target_hz))
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write(str(target_hz))
    time.sleep(0.1)
    return int(open(f'{GPU_SYSFS_PATH}/cur_freq').read().strip()) // 1000000


def set_cpu_freq(mhz: int) -> int:
    target_hz = CPU_SYSFS[mhz]
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


def restore_max_freq():
    max_gpu = max(GPU_SYSFS.values())
    min_gpu = min(GPU_SYSFS.values())
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write(str(max_gpu))
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write(str(min_gpu))
    max_cpu = max(CPU_SYSFS.values())
    min_cpu = min(CPU_SYSFS.values())
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov = cpu / 'cpufreq' / 'scaling_governor'
        mx = cpu / 'cpufreq' / 'scaling_max_freq'
        mn = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov.exists():
            try:
                open(gov, 'w').write('schedutil')
                open(mx, 'w').write(str(max_cpu))
                open(mn, 'w').write(str(min_cpu))
            except PermissionError:
                pass


def run_single_with_energy(runner, mc, prompt_length, output_length, phase='mixed') -> Dict:
    """Run a single inference with tegrastats energy measurement."""
    from src.metrics.metrics_collector import MetricsCollector

    tag = f"p{prompt_length}_o{output_length}_{phase}"
    mc.start_collection(tag=tag)

    if phase == 'prefill_only':
        # Only measure prefill: output 1 token
        result = runner.run_single_inference(prompt_length=prompt_length, output_length=1)
        result['phase'] = 'prefill'
        # Extrapolate to full output_length for fair comparison
        if result['output_tokens'] > 0:
            result['phase_note'] = f'prefill-only (1 output token)'
    elif phase == 'decode_only':
        # Short prompt (minimize prefill), long output
        result = runner.run_single_inference(prompt_length=16, output_length=output_length)
        result['phase'] = 'decode'
        result['phase_note'] = f'decode-only (16 prompt tokens, {output_length} output)'
    else:
        result = runner.run_single_inference(prompt_length=prompt_length, output_length=output_length)
        result['phase'] = 'mixed'

    time.sleep(0.3)
    mc.stop_collection()

    # Get energy data
    energy = mc.compute_energy()
    result.update({
        'avg_power_w': energy.get('avg_power_w', 0),
        'max_power_w': energy.get('max_power_w', 0),
        'total_energy_j': energy.get('total_energy_j', 0),
        'avg_gpu_soc_w': energy.get('avg_gpu_soc_w', 0),
        'avg_cpu_cv_w': energy.get('avg_cpu_cv_w', 0),
        'avg_sys_5v0_w': energy.get('avg_sys_5v0_w', 0),
        'avg_temp_cpu_c': energy.get('avg_temp_cpu_c', 0),
        'max_temp_cpu_c': energy.get('max_temp_cpu_c', 0),
        'avg_temp_tj_c': energy.get('avg_temp_tj_c', 0),
        'power_samples': energy.get('n_samples', 0),
    })

    # Derived energy metrics
    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
        result['tokens_per_joule'] = result['output_tokens'] / result['total_energy_j']
    else:
        result['energy_per_token_j'] = 0
        result['tokens_per_joule'] = 0

    return result


def run_profiling_experiment(phases: Optional[List[str]] = None):
    """
    Run the energy profiling experiment.

    Args:
        phases: Which phases to profile. Default: ['mixed'] for quick run.
                Use ['mixed', 'prefill_only', 'decode_only'] for full profile.
    """
    if phases is None:
        phases = ['mixed']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/energy_profiling')
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.benchmark.llama_cpp_runner import LlamaCppRunner
    from src.metrics.metrics_collector import MetricsCollector

    # Load model
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
    total_runs = len(GPU_FREQS) * len(CPU_FREQS) * len(WORKLOADS) * len(phases) * REPEATS
    run_count = 0

    for gpu_mhz in GPU_FREQS:
        for cpu_mhz in CPU_FREQS:
            config_name = f"GPU{gpu_mhz}_CPU{cpu_mhz}"
            logger.info(f"\n{'='*60}")
            logger.info(f"Config: {config_name}")
            logger.info(f"{'='*60}")

            actual_gpu = set_gpu_freq(gpu_mhz)
            actual_cpu = set_cpu_freq(cpu_mhz)
            time.sleep(1)
            logger.info(f"  Set: GPU={actual_gpu}MHz, CPU={actual_cpu}MHz")

            # Warmup
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
                            'target_gpu_freq_mhz': gpu_mhz,
                            'target_cpu_freq_mhz': cpu_mhz,
                            'workload': wl_name,
                            'phase': result.get('phase', phase),
                            'prompt_length': wl_params['prompt_length'],
                            'output_length': wl_params['output_length'],
                            'repeat': rep,
                            # Latency metrics
                            'ttft_ms': result['ttft_ms'],
                            'tpot_ms': result['tpot_ms'],
                            'total_time_ms': result.get('total_time_ms', 0),
                            'output_tokens': result.get('output_tokens', 0),
                            'tokens_per_second': result.get('tokens_per_second', 0),
                            # Energy metrics
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
                            # Metadata
                            'model': 'Phi-3-mini-Q4',
                            'runtime': 'llama.cpp',
                            'phase_note': result.get('phase_note', ''),
                        }
                        if 'error' in result:
                            record['error'] = result['error']
                        results.append(record)

                        # Log summary
                        ept = result.get('energy_per_token_j', 0)
                        tpj = result.get('tokens_per_joule', 0)
                        logger.info(f"    TTFT={result['ttft_ms']:.1f}ms TPOT={result['tpot_ms']:.1f}ms "
                                   f"tok={result.get('output_tokens', 0)} tps={result.get('tokens_per_second', 0):.1f} "
                                   f"power={result.get('avg_power_w', 0):.2f}W "
                                   f"E/token={ept:.4f}J tok/J={tpj:.2f}")

            # Cooldown
            logger.info(f"  Cooldown {COOLDOWN_SECONDS}s...")
            time.sleep(COOLDOWN_SECONDS)

    # Restore
    restore_max_freq()

    # Save
    df = pd.DataFrame(results)
    out_path = output_dir / f'energy_profiling_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path} ({len(df)} rows)")

    # Summary
    valid = df[df['output_tokens'] > 0]
    if valid.empty:
        logger.warning("No valid results!")
        return out_path

    summary_path = output_dir / f'energy_summary_{timestamp}.md'
    with open(summary_path, 'w') as f:
        f.write(generate_report(valid, timestamp, phases))
    logger.info(f"Saved report: {summary_path}")

    return out_path


def generate_report(df: pd.DataFrame, timestamp: str, phases: List[str]) -> str:
    lines = [
        "# Energy Profiling Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Phi-3-mini-4k-instruct-Q4",
        f"**Platform**: Jetson AGX Orin",
        f"**Phases**: {', '.join(phases)}",
        f"**Total valid runs**: {len(df)}",
        "",
        "## Per-Config Energy Summary (all workloads averaged)",
        "",
        "| Config | TTFT (ms) | TPOT (ms) | TPS | Avg Power (W) | E/token (J) | tok/J |",
        "|--------|-----------|-----------|-----|---------------|-------------|-------|",
    ]

    for config in sorted(df['config_name'].unique()):
        c = df[df['config_name'] == config]
        lines.append(
            f"| {config} | {c['ttft_ms'].mean():.1f} | {c['tpot_ms'].mean():.1f} | "
            f"{c['tokens_per_second'].mean():.1f} | {c['avg_power_w'].mean():.2f} | "
            f"{c['energy_per_token_j'].mean():.4f} | {c['tokens_per_joule'].mean():.2f} |"
        )

    # Best configs
    best_tps = df.groupby('config_name')['tokens_per_second'].mean().idxmax()
    best_eff = df.groupby('config_name')['tokens_per_joule'].mean().idxmax()
    best_ept = df.groupby('config_name')['energy_per_token_j'].mean().idxmin()

    lines.extend([
        "",
        "## Key Findings",
        f"- **Best Throughput**: {best_tps} ({df[df['config_name']==best_tps]['tokens_per_second'].mean():.1f} tok/s)",
        f"- **Best Energy Efficiency**: {best_eff} ({df[df['config_name']==best_eff]['tokens_per_joule'].mean():.2f} tok/J)",
        f"- **Lowest Energy/Token**: {best_ept} ({df[df['config_name']==best_ept]['energy_per_token_j'].mean():.4f} J/token)",
    ])

    # GPU effect on energy
    lines.extend(["", "## GPU Frequency Effect on Energy"])
    for gpu in sorted(df['gpu_freq_mhz'].unique()):
        g = df[df['gpu_freq_mhz'] == gpu]
        lines.append(
            f"- GPU {int(gpu)}MHz: power={g['avg_power_w'].mean():.2f}W, "
            f"E/token={g['energy_per_token_j'].mean():.4f}J, "
            f"TPS={g['tokens_per_second'].mean():.1f}"
        )

    # CPU effect on energy
    lines.extend(["", "## CPU Frequency Effect on Energy"])
    for cpu in sorted(df['cpu_freq_mhz'].unique()):
        c = df[df['cpu_freq_mhz'] == cpu]
        lines.append(
            f"- CPU {int(cpu)}MHz: power={c['avg_power_w'].mean():.2f}W, "
            f"E/token={c['energy_per_token_j'].mean():.4f}J, "
            f"TPS={c['tokens_per_second'].mean():.1f}"
        )

    # Phase comparison if available
    if len(df['phase'].unique()) > 1:
        lines.extend(["", "## Phase Comparison"])
        for phase in df['phase'].unique():
            p = df[df['phase'] == phase]
            lines.append(
                f"- **{phase}**: TTFT={p['ttft_ms'].mean():.1f}ms, "
                f"TPOT={p['tpot_ms'].mean():.1f}ms, "
                f"power={p['avg_power_w'].mean():.2f}W, "
                f"E/token={p['energy_per_token_j'].mean():.4f}J"
            )

    return '\n'.join(lines)


if __name__ == '__main__':
    import os, argparse
    # Ensure CWD is project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    parser = argparse.ArgumentParser(description='Energy Profiling Experiment')
    parser.add_argument('--phases', nargs='+', default=['mixed'],
                        choices=['mixed', 'prefill_only', 'decode_only'],
                        help='Phases to profile (default: mixed only)')
    parser.add_argument('--full', action='store_true',
                        help='Run all phases (mixed + prefill + decode)')
    args = parser.parse_args()

    phases = ['mixed', 'prefill_only', 'decode_only'] if args.full else args.phases
    logger.info(f"Running energy profiling with phases: {phases}")
    run_profiling_experiment(phases=phases)
