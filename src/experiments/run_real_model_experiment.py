#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Real Model Multi-Configuration Experiment
Runs llama.cpp inference under different GPU/CPU frequency settings on Jetson Orin.

Experiment matrix: 4 GPU freqs x 3 CPU freqs x 5 workloads x 3 repeats = 180 runs
"""

import csv
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Frequency Configurations ──
GPU_FREQS = [306, 612, 918, 1300]      # MHz (4 levels)
CPU_FREQS = [1036, 1497, 2201]          # MHz (3 levels: low/mid/max)

# Map to actual sysfs values (Hz)
GPU_SYSFS = {306: 306000000, 612: 612000000, 918: 918000000, 1300: 1300500000}
CPU_SYSFS = {1036: 1036800, 1497: 1497600, 2201: 2201600}

# ── Workload Definitions ──
WORKLOADS = {
    'short_short':  {'prompt_length': 128,  'output_length': 64},
    'short_long':   {'prompt_length': 128,  'output_length': 256},
    'medium_medium': {'prompt_length': 512, 'output_length': 128},
    'long_medium':  {'prompt_length': 1024, 'output_length': 128},
    'long_long':    {'prompt_length': 1024, 'output_length': 512},
}

GPU_SYSFS_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
CPU_SYSFS_PATH = '/sys/devices/system/cpu/cpu0/cpufreq'

REPEATS = 3
WARMUP_RUNS = 2
COOLDOWN_SECONDS = 5


def set_gpu_freq(mhz: int):
    target_hz = GPU_SYSFS[mhz]
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write(str(target_hz))
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write(str(target_hz))
    time.sleep(0.1)
    cur = int(open(f'{GPU_SYSFS_PATH}/cur_freq').read().strip())
    return cur // 1000000


def set_cpu_freq(mhz: int):
    target_hz = CPU_SYSFS[mhz]
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov_path = cpu / 'cpufreq' / 'scaling_governor'
        max_path = cpu / 'cpufreq' / 'scaling_max_freq'
        min_path = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov_path.exists():
            try:
                open(gov_path, 'w').write('userspace')
                open(max_path, 'w').write(str(target_hz))
                open(min_path, 'w').write(str(target_hz))
            except PermissionError:
                pass
    time.sleep(0.1)
    cur = int(open(f'{CPU_SYSFS_PATH}/scaling_cur_freq').read().strip())
    return cur // 1000  # kHz -> MHz


def restore_max_freq():
    """Restore all frequencies to maximum."""
    max_gpu = max(GPU_SYSFS.values())
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write(str(max_gpu))
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write(str(min(GPU_SYSFS.values())))
    max_cpu = max(CPU_SYSFS.values())
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov_path = cpu / 'cpufreq' / 'scaling_governor'
        max_path = cpu / 'cpufreq' / 'scaling_max_freq'
        min_path = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov_path.exists():
            try:
                open(gov_path, 'w').write('schedutil')
                open(max_path, 'w').write(str(max_cpu))
                open(min_path, 'w').write(str(min(CPU_SYSFS.values())))
            except PermissionError:
                pass


def run_experiment():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/real_model_experiment')
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.benchmark.llama_cpp_runner import LlamaCppRunner

    # Load model once
    logger.info("Loading model...")
    runner = LlamaCppRunner(
        model_path='models/gguf/Phi-3-mini-4k-instruct-q4.gguf',
        n_gpu_layers=-1, n_ctx=4096, n_threads=4
    )
    logger.info("Model loaded")

    # Set GPU governor to userspace
    with open(f'{GPU_SYSFS_PATH}/governor', 'w') as f:
        f.write('userspace')

    results: List[Dict] = []
    total_runs = len(GPU_FREQS) * len(CPU_FREQS) * len(WORKLOADS) * REPEATS
    run_count = 0

    for gpu_mhz in GPU_FREQS:
        for cpu_mhz in CPU_FREQS:
            config_name = f"GPU{gpu_mhz}_CPU{cpu_mhz}"
            logger.info(f"\n{'='*50}")
            logger.info(f"Config: {config_name}")
            logger.info(f"{'='*50}")

            # Apply frequencies
            actual_gpu = set_gpu_freq(gpu_mhz)
            actual_cpu = set_cpu_freq(cpu_mhz)
            time.sleep(1)
            logger.info(f"  Set: GPU={actual_gpu}MHz, CPU={actual_cpu}MHz")

            # Warmup for this config
            logger.info(f"  Warmup ({WARMUP_RUNS} runs)...")
            runner.warmup(WARMUP_RUNS)

            for wl_name, wl_params in WORKLOADS.items():
                for rep in range(REPEATS):
                    run_count += 1
                    logger.info(f"  [{run_count}/{total_runs}] {wl_name} rep={rep}")

                    result = runner.run_single_inference(
                        prompt_length=wl_params['prompt_length'],
                        output_length=wl_params['output_length']
                    )

                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'config_name': config_name,
                        'gpu_freq_mhz': actual_gpu,
                        'cpu_freq_mhz': actual_cpu,
                        'target_gpu_freq_mhz': gpu_mhz,
                        'target_cpu_freq_mhz': cpu_mhz,
                        'workload': wl_name,
                        'prompt_length': wl_params['prompt_length'],
                        'output_length': wl_params['output_length'],
                        'repeat': rep,
                        'ttft_ms': result['ttft_ms'],
                        'tpot_ms': result['tpot_ms'],
                        'total_time_ms': result['total_time_ms'],
                        'output_tokens': result['output_tokens'],
                        'tokens_per_second': result['tokens_per_second'],
                        'model': 'Phi-3-mini-Q4',
                        'runtime': 'llama.cpp-0.3.22',
                    }
                    results.append(record)

                    logger.info(f"    TTFT={result['ttft_ms']:.1f}ms TPOT={result['tpot_ms']:.1f}ms "
                               f"tok={result['output_tokens']} tps={result['tokens_per_second']:.1f}")

            # Cooldown between configs
            logger.info(f"  Cooldown {COOLDOWN_SECONDS}s...")
            time.sleep(COOLDOWN_SECONDS)

    # Restore frequencies
    logger.info("Restoring frequencies to max...")
    restore_max_freq()

    # Save results
    df = pd.DataFrame(results)

    detailed_path = output_dir / f'real_experiment_{timestamp}.csv'
    df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed results: {detailed_path}")

    # Summary
    summary = df.groupby(['config_name', 'workload']).agg({
        'ttft_ms': ['mean', 'std'],
        'tpot_ms': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std'],
        'total_time_ms': ['mean', 'std'],
        'output_tokens': 'mean',
    }).round(2)
    summary_path = output_dir / f'real_summary_{timestamp}.csv'
    summary.to_csv(summary_path)
    logger.info(f"Saved summary: {summary_path}")

    # Config-level summary
    config_summary = df.groupby('config_name').agg({
        'ttft_ms': 'mean',
        'tpot_ms': 'mean',
        'tokens_per_second': 'mean',
        'total_time_ms': 'mean',
    }).round(2)
    config_path = output_dir / f'real_config_summary_{timestamp}.csv'
    config_summary.to_csv(config_path)

    # Report
    report = generate_report(df, timestamp)
    report_path = output_dir / f'real_report_{timestamp}.md'
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Saved report: {report_path}")

    return detailed_path, summary_path, report_path


def generate_report(df: pd.DataFrame, timestamp: str) -> str:
    lines = []
    lines.append("# Real Model Experiment Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Model**: Phi-3-mini-4k-instruct-Q4")
    lines.append(f"**Runtime**: llama.cpp 0.3.22")
    lines.append(f"**Platform**: Jetson AGX Orin (MAXN mode)")

    lines.append(f"\n**Experiment Matrix**: {df['config_name'].nunique()} configs x "
                f"{df['workload'].nunique()} workloads x {REPEATS} repeats = {len(df)} runs")

    # Per-config summary
    lines.append("\n## Configuration Performance Summary")
    lines.append("\n| Config | GPU (MHz) | CPU (MHz) | TTFT (ms) | TPOT (ms) | TPS |")
    lines.append("|--------|-----------|-----------|-----------|-----------|-----|")

    for config in df['config_name'].unique():
        cdata = df[df['config_name'] == config]
        gpu = int(cdata['gpu_freq_mhz'].iloc[0])
        cpu = int(cdata['cpu_freq_mhz'].iloc[0])
        lines.append(f"| {config} | {gpu} | {cpu} | "
                    f"{cdata['ttft_ms'].mean():.1f} | {cdata['tpot_ms'].mean():.1f} | "
                    f"{cdata['tokens_per_second'].mean():.1f} |")

    # Best configs
    lines.append("\n## Key Findings")

    best_ttft = df.groupby('config_name')['ttft_ms'].mean().idxmin()
    best_tpot = df.groupby('config_name')['tpot_ms'].mean().idxmin()
    best_tps = df.groupby('config_name')['tokens_per_second'].mean().idxmax()

    lines.append(f"\n**Best TTFT**: {best_ttft} ({df[df['config_name']==best_ttft]['ttft_ms'].mean():.1f}ms)")
    lines.append(f"**Best TPOT**: {best_tpot} ({df[df['config_name']==best_tpot]['tpot_ms'].mean():.1f}ms)")
    lines.append(f"**Best Throughput**: {best_tps} ({df[df['config_name']==best_tps]['tokens_per_second'].mean():.1f} tok/s)")

    # GPU frequency effect
    lines.append("\n## GPU Frequency Effect")
    for gpu in sorted(df['gpu_freq_mhz'].unique()):
        gdata = df[df['gpu_freq_mhz'] == gpu]
        lines.append(f"- GPU {int(gpu)}MHz: TTFT={gdata['ttft_ms'].mean():.1f}ms, "
                    f"TPOT={gdata['tpot_ms'].mean():.1f}ms, TPS={gdata['tokens_per_second'].mean():.1f}")

    # CPU frequency effect
    lines.append("\n## CPU Frequency Effect")
    for cpu in sorted(df['cpu_freq_mhz'].unique()):
        cdata = df[df['cpu_freq_mhz'] == cpu]
        lines.append(f"- CPU {int(cpu)}MHz: TTFT={cdata['ttft_ms'].mean():.1f}ms, "
                    f"TPOT={cdata['tpot_ms'].mean():.1f}ms, TPS={cdata['tokens_per_second'].mean():.1f}")

    # Per-workload breakdown
    lines.append("\n## Per-Workload Breakdown")
    for wl in df['workload'].unique():
        wdata = df[df['workload'] == wl]
        pl = int(wdata['prompt_length'].iloc[0])
        ol = int(wdata['output_length'].iloc[0])
        lines.append(f"\n### {wl} (prompt={pl}, output={ol})")
        best = wdata.loc[wdata['tokens_per_second'].idxmax()]
        lines.append(f"- Best throughput: {best['config_name']} ({best['tokens_per_second']:.1f} tok/s)")
        best_ttft_wl = wdata.loc[wdata['ttft_ms'].idxmin()]
        lines.append(f"- Best TTFT: {best_ttft_wl['config_name']} ({best_ttft_wl['ttft_ms']:.1f}ms)")

    return '\n'.join(lines)


if __name__ == '__main__':
    detailed, summary, report = run_experiment()
    print(f"\nResults:")
    print(f"  Detailed: {detailed}")
    print(f"  Summary: {summary}")
    print(f"  Report: {report}")
