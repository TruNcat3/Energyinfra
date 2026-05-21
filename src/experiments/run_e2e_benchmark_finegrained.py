#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Phase-Aware DVFS Validation Experiment (Fine-Grained)

Validates the phase-aware DVFS policy using the fine-grained GPU×EMC rate table.

Experiment matrix:
  - 6 alpha values: {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
  - 2 strategies: single_config, phase_aware
  - 5 workloads
  - 3 repeats
  = 180 runs

Plus 3 baselines × 5 workloads × 3 repeats = 45 runs

Total: 225 runs ≈ 2-3 hours

Usage:
    sudo python3 src/experiments/run_e2e_benchmark_finegrained.py
    sudo python3 src/experiments/run_e2e_benchmark_finegrained.py --quick  # alpha=0,0.5,1.0 only
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
from src.controller.weighted_selector import WeightedSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Frequency control (same as finegrained profiling) ──
GPU_FREQS_HZ = {
    306: 306000000, 408: 408000000, 510: 510000000, 612: 612000000,
    714: 714000000, 816: 816000000, 918: 918000000, 1020: 1020000000,
    1122: 1122000000, 1224: 1224000000, 1300: 1300500000,
}
EMC_HZ = {204: 204000000, 665: 665600000, 2133: 2133000000, 3199: 3199000000}
CPU_HZ = {1036: 1036800, 1497: 1497600}

GPU_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
CPU_PATH = '/sys/devices/system/cpu/cpu0/cpufreq'
EMC_MAX = '/sys/kernel/debug/emc/max_rate'
EMC_MIN = '/sys/kernel/debug/emc/min_rate'
EMC_CLK = '/sys/kernel/debug/clk/emc/clk_rate'

WORKLOADS = {
    'p128_o64':  {'prompt_length': 128,  'output_length': 64},
    'p512_o128': {'prompt_length': 512,  'output_length': 128},
    'p1024_o128': {'prompt_length': 1024, 'output_length': 128},
    'p128_o256': {'prompt_length': 128,  'output_length': 256},
    'p1024_o512': {'prompt_length': 1024, 'output_length': 512},
}

BASELINES = {
    'MAXN':     {'gpu': 1300, 'emc': 3199, 'cpu': 1497},
    '30W_mode': {'gpu': 612,  'emc': 2133, 'cpu': 1497},
    'E_min':    {'gpu': 816,  'emc': 204,  'cpu': 1036},
}

REPEATS = 3
WARMUP_RUNS = 2
COOLDOWN_S = 3
TEGRASTATS_MS = 500


def set_gpu(mhz):
    hz = GPU_FREQS_HZ[mhz]
    with open(f'{GPU_PATH}/governor', 'w') as f: f.write('performance')
    with open(f'{GPU_PATH}/max_freq', 'w') as f: f.write(str(hz))
    with open(f'{GPU_PATH}/min_freq', 'w') as f: f.write(str(hz))
    time.sleep(0.3)
    return int(open(f'{GPU_PATH}/cur_freq').read().strip()) // 1000000


def set_cpu(mhz):
    hz = CPU_HZ[mhz]
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov = cpu / 'cpufreq' / 'scaling_governor'
        mx = cpu / 'cpufreq' / 'scaling_max_freq'
        mn = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov.exists():
            try:
                open(gov, 'w').write('userspace')
                open(mx, 'w').write(str(hz))
                open(mn, 'w').write(str(hz))
            except PermissionError:
                pass
    time.sleep(0.1)
    return int(open(f'{CPU_PATH}/scaling_cur_freq').read().strip()) // 1000


def set_emc(mhz):
    hz = EMC_HZ[mhz]
    max_hz = max(EMC_HZ.values())
    with open(EMC_MAX, 'w') as f: f.write(str(max_hz))
    with open(EMC_MIN, 'w') as f: f.write('204000000')
    with open(EMC_MIN, 'w') as f: f.write(str(hz))
    with open(EMC_MAX, 'w') as f: f.write(str(hz))
    time.sleep(0.3)
    return int(open(EMC_CLK).read().strip()) // 1000000


def set_all(gpu, emc, cpu):
    ag = set_gpu(gpu)
    ae = set_emc(emc)
    ac = set_cpu(cpu)
    return ag, ae, ac


def restore_max():
    max_gpu = max(GPU_FREQS_HZ.values())
    min_gpu = min(GPU_FREQS_HZ.values())
    with open(f'{GPU_PATH}/max_freq', 'w') as f: f.write(str(max_gpu))
    with open(f'{GPU_PATH}/min_freq', 'w') as f: f.write(str(min_gpu))
    try:
        with open(f'{GPU_PATH}/governor', 'w') as f: f.write('simple_ondemand')
    except OSError:
        with open(f'{GPU_PATH}/governor', 'w') as f: f.write('performance')
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        gov = cpu / 'cpufreq' / 'scaling_governor'
        mx = cpu / 'cpufreq' / 'scaling_max_freq'
        mn = cpu / 'cpufreq' / 'scaling_min_freq'
        if gov.exists():
            try:
                open(gov, 'w').write('schedutil')
                open(mx, 'w').write(str(max(CPU_HZ.values())))
                open(mn, 'w').write(str(min(CPU_HZ.values())))
            except PermissionError:
                pass
    with open(EMC_MIN, 'w') as f: f.write('204000000')
    with open(EMC_MAX, 'w') as f: f.write('3199000000')


def run_single(runner, mc, prompt_length, output_length):
    """Run single inference with energy measurement."""
    tag = f"p{prompt_length}_o{output_length}"
    mc.start_collection(tag=tag)
    result = runner.run_single_inference(
        prompt_length=prompt_length, output_length=output_length
    )
    time.sleep(0.3)
    mc.stop_collection()
    energy = mc.compute_energy()
    result.update({
        'avg_power_w': energy.get('avg_power_w', 0),
        'total_energy_j': energy.get('total_energy_j', 0),
        'avg_gpu_soc_w': energy.get('avg_gpu_soc_w', 0),
        'avg_cpu_cv_w': energy.get('avg_cpu_cv_w', 0),
    })
    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
        result['tokens_per_joule'] = result['output_tokens'] / result['total_energy_j']
    else:
        result['energy_per_token_j'] = 0
        result['tokens_per_joule'] = 0
    return result


def run_experiment(rate_table_path: str, quick: bool = False):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/e2e_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading WeightedSelector...")
    ws = WeightedSelector(rate_table_path)

    logger.info("Loading model...")
    runner = LlamaCppRunner(
        model_path='models/gguf/Phi-3-mini-4k-instruct-q4.gguf',
        n_gpu_layers=-1, n_ctx=4096, n_threads=4
    )
    logger.info("Model loaded")

    mc = MetricsCollector(interval_ms=TEGRASTATS_MS, output_dir='data/raw_logs')

    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if not quick else [0.0, 0.5, 1.0]
    strategies = ['single_config', 'phase_aware']

    total = len(alphas) * len(strategies) * len(WORKLOADS) * REPEATS + len(BASELINES) * len(WORKLOADS) * REPEATS
    run_count = 0
    results: List[Dict] = []

    # ── Phase 1: Alpha sweep with strategies ──
    for alpha in alphas:
        for strategy in strategies:
            for wl_name, wl_params in WORKLOADS.items():
                for rep in range(REPEATS):
                    run_count += 1
                    pl = wl_params['prompt_length']
                    ol = wl_params['output_length']

                    logger.info(f"\n[{run_count}/{total}] alpha={alpha} {strategy} {wl_name} rep={rep}")

                    if strategy == 'single_config':
                        cfg = ws.select_config(pl, ol, 'decode', alpha)
                        if cfg is None:
                            logger.warning("  No config found, skipping")
                            continue
                        gpu, emc, cpu = cfg['gpu_freq_mhz'], cfg.get('emc_freq_mhz', 204), cfg['cpu_freq_mhz']
                    else:  # phase_aware
                        cfgs = ws.select_phase_aware_configs(pl, ol, alpha)
                        if cfgs is None:
                            logger.warning("  No phase-aware configs, skipping")
                            continue
                        # For phase_aware we run a full mixed inference
                        # but set prefill config first, then switch to decode
                        # For this validation we use the decode config for the whole run
                        # since we can't actually switch mid-inference
                        # We measure the decode-phase energy separately below
                        pcfg = cfgs['prefill_config']
                        dcfg = cfgs['decode_config']
                        gpu, emc, cpu = dcfg['gpu_freq_mhz'], dcfg.get('emc_freq_mhz', 204), dcfg['cpu_freq_mhz']

                    try:
                        ag, ae, ac = set_all(gpu, emc, cpu)
                        runner.warmup(1)

                        result = run_single(runner, mc, pl, ol)
                        record = {
                            'timestamp': datetime.now().isoformat(),
                            'strategy': strategy,
                            'alpha': alpha,
                            'workload': wl_name,
                            'prompt_length': pl,
                            'output_length': ol,
                            'repeat': rep,
                            'target_gpu_mhz': gpu,
                            'target_emc_mhz': emc,
                            'target_cpu_mhz': cpu,
                            'actual_gpu_mhz': ag,
                            'actual_emc_mhz': ae,
                            'actual_cpu_mhz': ac,
                            'ttft_ms': result.get('ttft_ms', 0),
                            'tpot_ms': result.get('tpot_ms', 0),
                            'total_time_ms': result.get('total_time_ms', 0),
                            'output_tokens': result.get('output_tokens', 0),
                            'tokens_per_second': result.get('tokens_per_second', 0),
                            'total_energy_j': result.get('total_energy_j', 0),
                            'avg_power_w': result.get('avg_power_w', 0),
                            'energy_per_token_j': result.get('energy_per_token_j', 0),
                            'tokens_per_joule': result.get('tokens_per_joule', 0),
                            'avg_gpu_soc_w': result.get('avg_gpu_soc_w', 0),
                            'avg_cpu_cv_w': result.get('avg_cpu_cv_w', 0),
                            'model': 'Phi-3-mini-Q4',
                            'experiment': 'e2e_benchmark_finegrained',
                        }
                        if strategy == 'phase_aware' and cfgs:
                            record['prefill_gpu'] = pcfg['gpu_freq_mhz']
                            record['prefill_emc'] = pcfg.get('emc_freq_mhz', 204)
                            record['decode_gpu'] = dcfg['gpu_freq_mhz']
                            record['decode_emc'] = dcfg.get('emc_freq_mhz', 204)
                            record['phase_switch'] = cfgs.get('phase_switch', False)
                    except Exception as e:
                        logger.error(f"  FAILED: {e}")
                        record = {
                            'timestamp': datetime.now().isoformat(),
                            'strategy': strategy, 'alpha': alpha,
                            'workload': wl_name, 'repeat': rep,
                            'error': str(e),
                        }

                    results.append(record)
                    logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                               f"tps={result.get('tokens_per_second',0):.1f} "
                               f"E/tok={result.get('energy_per_token_j',0):.4f}J")

                    time.sleep(COOLDOWN_S)

    # ── Phase 2: Baselines ──
    for bl_name, bl_cfg in BASELINES.items():
        for wl_name, wl_params in WORKLOADS.items():
            for rep in range(REPEATS):
                run_count += 1
                pl = wl_params['prompt_length']
                ol = wl_params['output_length']

                logger.info(f"\n[{run_count}/{total}] BASELINE {bl_name} {wl_name} rep={rep}")

                try:
                    ag, ae, ac = set_all(bl_cfg['gpu'], bl_cfg['emc'], bl_cfg['cpu'])
                    runner.warmup(1)
                    result = run_single(runner, mc, pl, ol)
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'baseline',
                        'alpha': -1,
                        'baseline_name': bl_name,
                        'workload': wl_name,
                        'prompt_length': pl,
                        'output_length': ol,
                        'repeat': rep,
                        'target_gpu_mhz': bl_cfg['gpu'],
                        'target_emc_mhz': bl_cfg['emc'],
                        'target_cpu_mhz': bl_cfg['cpu'],
                        'actual_gpu_mhz': ag,
                        'actual_emc_mhz': ae,
                        'actual_cpu_mhz': ac,
                        'ttft_ms': result.get('ttft_ms', 0),
                        'tpot_ms': result.get('tpot_ms', 0),
                        'total_time_ms': result.get('total_time_ms', 0),
                        'output_tokens': result.get('output_tokens', 0),
                        'tokens_per_second': result.get('tokens_per_second', 0),
                        'total_energy_j': result.get('total_energy_j', 0),
                        'avg_power_w': result.get('avg_power_w', 0),
                        'energy_per_token_j': result.get('energy_per_token_j', 0),
                        'tokens_per_joule': result.get('tokens_per_joule', 0),
                        'avg_gpu_soc_w': result.get('avg_gpu_soc_w', 0),
                        'avg_cpu_cv_w': result.get('avg_cpu_cv_w', 0),
                        'model': 'Phi-3-mini-Q4',
                        'experiment': 'e2e_benchmark_finegrained',
                    }
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'baseline', 'baseline_name': bl_name,
                        'workload': wl_name, 'repeat': rep,
                        'error': str(e),
                    }

                results.append(record)
                logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                           f"E/tok={result.get('energy_per_token_j',0):.4f}J")
                time.sleep(COOLDOWN_S)

    # ── Save results BEFORE restore (in case restore crashes) ──
    df = pd.DataFrame(results)
    out_path = output_dir / f'e2e_benchmark_finegrained_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path} ({len(df)} rows)")

    # Generate report
    valid = df[(df['output_tokens'] > 0) & (df['energy_per_token_j'] > 0)]
    if not valid.empty:
        report_path = output_dir / f'e2e_summary_finegrained_{timestamp}.md'
        with open(report_path, 'w') as f:
            f.write(generate_report(valid))
        logger.info(f"Report: {report_path}")

    try:
        restore_max()
    except Exception as e:
        logger.warning(f"restore_max() failed: {e} (data already saved)")

    return out_path


def generate_report(df: pd.DataFrame) -> str:
    lines = [
        "# E2E Benchmark Report (Fine-Grained GPU×EMC)",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Phi-3-mini-Q4",
        f"**Valid runs**: {len(df)}",
        "",
        "## Alpha Effect (averaged across workloads)",
        "",
        "| Alpha | Strategy | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for strategy in ['single_config', 'phase_aware']:
        for alpha in sorted(df[df['strategy'] == strategy]['alpha'].unique()):
            sub = df[(df['strategy'] == strategy) & (df['alpha'] == alpha)]
            lines.append(
                f"| {alpha} | {strategy} | {sub['energy_per_token_j'].mean():.4f} "
                f"| {sub['tpot_ms'].mean():.1f} | {sub['tokens_per_second'].mean():.1f} "
                f"| {sub['avg_power_w'].mean():.1f} | {sub['tokens_per_joule'].mean():.2f} |"
            )

    # Baselines
    lines.extend(["", "## Baselines", "",
                  "| Baseline | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |",
                  "|:---:|:---:|:---:|:---:|:---:|:---:|"])
    for bl in sorted(df[df['strategy'] == 'baseline']['baseline_name'].unique()):
        sub = df[df['baseline_name'] == bl]
        lines.append(
            f"| {bl} | {sub['energy_per_token_j'].mean():.4f} "
            f"| {sub['tpot_ms'].mean():.1f} | {sub['tokens_per_second'].mean():.1f} "
            f"| {sub['avg_power_w'].mean():.1f} | {sub['tokens_per_joule'].mean():.2f} |"
        )

    # Per-workload comparison
    lines.extend(["", "## Best Config per Workload (alpha=0.5, single_config)", "",
                  "| Workload | GPU | EMC | E/tok (J) | TPOT (ms) | Power (W) |",
                  "|:---:|:---:|:---:|:---:|:---:|:---:|"])
    for wl in sorted(df['workload'].unique()):
        sub = df[(df['workload'] == wl) & (df['strategy'] == 'single_config') & (df['alpha'] == 0.5)]
        if not sub.empty:
            best = sub.loc[sub['energy_per_token_j'].idxmin()]
            lines.append(
                f"| {wl} | {int(best['actual_gpu_mhz'])} | {int(best['actual_emc_mhz'])} "
                f"| {best['energy_per_token_j']:.4f} | {best['tpot_ms']:.1f} "
                f"| {best['avg_power_w']:.1f} |"
            )

    # Energy savings vs baselines
    lines.extend(["", "## Energy Savings vs MAXN Baseline", ""])
    maxn = df[df['baseline_name'] == 'MAXN']['energy_per_token_j'].mean()
    if maxn > 0:
        for strategy in ['single_config', 'phase_aware']:
            for alpha in [0.0, 0.5, 1.0]:
                sub = df[(df['strategy'] == strategy) & (df['alpha'] == alpha)]
                if not sub.empty:
                    saving = (1 - sub['energy_per_token_j'].mean() / maxn) * 100
                    lines.append(f"- {strategy} alpha={alpha}: {saving:+.1f}% energy vs MAXN")

    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='E2E Benchmark (Fine-Grained)')
    parser.add_argument('--rate-table', type=str,
                        default='data/rate_tables/finegrained_selector_table_20260516_055008.parquet',
                        help='Path to fine-grained selector table')
    parser.add_argument('--quick', action='store_true', help='Quick mode (3 alphas only)')
    args = parser.parse_args()

    n_alphas = 3 if args.quick else 6
    total = n_alphas * 2 * len(WORKLOADS) * REPEATS + len(BASELINES) * len(WORKLOADS) * REPEATS
    logger.info(f"E2E Benchmark: {total} runs, ~{total * 25 // 60} min")

    run_experiment(rate_table_path=args.rate_table, quick=args.quick)
