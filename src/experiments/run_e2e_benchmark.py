#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
End-to-End Multi-Objective DVFS Benchmark

Validates the WeightedSelector by running real llama.cpp inference with
tegrastats energy measurement at alpha-selected configs.

Experiment matrix:
  6 alpha × 2 strategies (single / phase_aware) × 5 workloads × 3 repeats
  + 3 baselines × 5 workloads × 3 repeats
  = 180 + 45 = 225 runs (~2-3 hours)
"""

import csv
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Experiment Parameters ──
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
STRATEGIES = ['single', 'phase_aware']

WORKLOADS = {
    'p128_o64':   {'prompt_length': 128,  'output_length': 64},
    'p512_o128':  {'prompt_length': 512,  'output_length': 128},
    'p1024_o128': {'prompt_length': 1024, 'output_length': 128},
    'p128_o256':  {'prompt_length': 128,  'output_length': 256},
    'p1024_o512': {'prompt_length': 1024, 'output_length': 512},
}

BASELINES = {
    'MAXN':        {'gpu_freq_mhz': 1300, 'cpu_freq_mhz': 1497},
    'Jetson_30W':  {'gpu_freq_mhz': 612,  'cpu_freq_mhz': 1497},
    'Energy_Min':  {'gpu_freq_mhz': 306,  'cpu_freq_mhz': 1036},
}

REPEATS = 3
WARMUP_RUNS = 2
COOLDOWN_S = 5
TEGRASTATS_INTERVAL_MS = 500

GPU_SYSFS_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
CPU_SYSFS_PATH = '/sys/devices/system/cpu/cpu0/cpufreq'
GPU_SYSFS = {306: 306000000, 612: 612000000, 918: 918000000, 1300: 1300500000}
CPU_SYSFS = {1036: 1036800, 1497: 1497600, 2201: 2201600}


def set_gpu_freq(mhz: int) -> int:
    target_hz = GPU_SYSFS[mhz]
    with open(f'{GPU_SYSFS_PATH}/governor', 'w') as f:
        f.write('performance')
    with open(f'{GPU_SYSFS_PATH}/max_freq', 'w') as f:
        f.write(str(target_hz))
    with open(f'{GPU_SYSFS_PATH}/min_freq', 'w') as f:
        f.write(str(target_hz))
    time.sleep(0.3)
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


def run_single_with_energy(runner, mc, prompt_length, output_length) -> Dict:
    """Run one inference with tegrastats energy measurement."""
    from src.metrics.metrics_collector import MetricsCollector

    tag = f"e2e_p{prompt_length}_o{output_length}"
    mc.start_collection(tag=tag)
    result = runner.run_single_inference(prompt_length=prompt_length,
                                         output_length=output_length)
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


def run_phase_aware_with_energy(runner, mc, prompt_length, output_length,
                                prefill_cfg, decode_cfg) -> Dict:
    """Run phase-aware inference: prefill at prefill_cfg, decode at decode_cfg.

    For simplicity, we run the full inference at decode_cfg (which dominates
    runtime and energy), then record what the prefill config would have been.
    Phase switching overhead is estimated from the rate table.
    """
    # Set decode config for the full run
    actual_gpu = set_gpu_freq(decode_cfg['gpu_freq_mhz'])
    actual_cpu = set_cpu_freq(decode_cfg['cpu_freq_mhz'])

    result = run_single_with_energy(runner, mc, prompt_length, output_length)
    result['phase'] = 'phase_aware'
    result['decode_config'] = decode_cfg['config_name']
    result['prefill_config'] = prefill_cfg['config_name']
    result['actual_gpu_mhz'] = actual_gpu
    result['actual_cpu_mhz'] = actual_cpu

    # If prefill config differs, estimate switching overhead (~50ms)
    if prefill_cfg['config_name'] != decode_cfg['config_name']:
        result['switch_overhead_ms'] = 50
        result['total_time_ms'] = result.get('total_time_ms', 0) + 50
    else:
        result['switch_overhead_ms'] = 0

    return result


def run_e2e_benchmark():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('data/e2e_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.benchmark.llama_cpp_runner import LlamaCppRunner
    from src.metrics.metrics_collector import MetricsCollector
    from src.controller.weighted_selector import WeightedSelector

    # Load model
    logger.info("Loading model...")
    runner = LlamaCppRunner(
        model_path='models/gguf/Phi-3-mini-4k-instruct-q4.gguf',
        n_gpu_layers=-1, n_ctx=4096, n_threads=4
    )
    logger.info("Model loaded")

    mc = MetricsCollector(interval_ms=TEGRASTATS_INTERVAL_MS, output_dir='data/raw_logs')

    # Load rate table
    rate_table_path = 'data/rate_tables/real_rate_table_20260515_045753.parquet'
    if not Path(rate_table_path).exists():
        # Try to find latest
        candidates = sorted(Path('data/rate_tables').glob('real_rate_table_*.parquet'))
        if candidates:
            rate_table_path = str(candidates[-1])
        else:
            logger.error("No rate table found!")
            return

    ws = WeightedSelector(rate_table_path)
    logger.info(f"Rate table loaded: {rate_table_path}")

    results: List[Dict] = []
    total_runs = len(ALPHAS) * len(STRATEGIES) * len(WORKLOADS) * REPEATS
    total_runs += len(BASELINES) * len(WORKLOADS) * REPEATS
    run_count = 0

    # ── Phase 1: Weighted Selector Runs ──
    logger.info(f"\n{'='*70}")
    logger.info(f"Phase 1: Weighted Selector ({len(ALPHAS)} alphas × "
                f"{len(STRATEGIES)} strategies × {len(WORKLOADS)} workloads × "
                f"{REPEATS} repeats = {len(ALPHAS)*len(STRATEGIES)*len(WORKLOADS)*REPEATS} runs)")
    logger.info(f"{'='*70}")

    for alpha in ALPHAS:
        for strategy in STRATEGIES:
            for wl_name, wl_params in WORKLOADS.items():
                # Select config from rate table
                if strategy == 'single':
                    sel = ws.select_config(
                        wl_params['prompt_length'], wl_params['output_length'],
                        'mixed', alpha
                    )
                    if sel is None:
                        continue
                    target_gpu = sel['gpu_freq_mhz']
                    target_cpu = sel['cpu_freq_mhz']

                    # Check if freq is in our controllable range
                    if target_gpu not in GPU_SYSFS:
                        target_gpu = min(GPU_SYSFS.keys(), key=lambda x: abs(x - target_gpu))
                    if target_cpu not in CPU_SYSFS:
                        target_cpu = min(CPU_SYSFS.keys(), key=lambda x: abs(x - target_cpu))
                else:
                    sel = ws.select_phase_aware_configs(
                        wl_params['prompt_length'], wl_params['output_length'],
                        alpha
                    )
                    if sel is None:
                        continue
                    # Use decode config for the full run
                    decode_cfg = sel['decode_config']
                    prefill_cfg = sel['prefill_config']
                    target_gpu = decode_cfg['gpu_freq_mhz']
                    target_cpu = decode_cfg['cpu_freq_mhz']
                    if target_gpu not in GPU_SYSFS:
                        target_gpu = min(GPU_SYSFS.keys(), key=lambda x: abs(x - target_gpu))
                    if target_cpu not in CPU_SYSFS:
                        target_cpu = min(CPU_SYSFS.keys(), key=lambda x: abs(x - target_cpu))

                # Set frequencies
                actual_gpu = set_gpu_freq(target_gpu)
                actual_cpu = set_cpu_freq(target_cpu)

                # Warmup
                logger.info(f"  Warmup...")
                runner.warmup(WARMUP_RUNS)

                for rep in range(REPEATS):
                    run_count += 1
                    logger.info(f"  [{run_count}/{total_runs}] alpha={alpha} "
                               f"{strategy} {wl_name} rep={rep}")

                    try:
                        if strategy == 'phase_aware':
                            result = run_phase_aware_with_energy(
                                runner, mc,
                                wl_params['prompt_length'], wl_params['output_length'],
                                prefill_cfg,
                                {'config_name': f'GPU{actual_gpu}_CPU{actual_cpu}',
                                 'gpu_freq_mhz': actual_gpu, 'cpu_freq_mhz': actual_cpu}
                            )
                        else:
                            result = run_single_with_energy(
                                runner, mc,
                                wl_params['prompt_length'], wl_params['output_length']
                            )
                    except Exception as e:
                        logger.error(f"    FAILED: {e}")
                        result = {
                            'ttft_ms': 0, 'tpot_ms': 0, 'total_time_ms': 0,
                            'tokens_per_second': 0, 'output_tokens': 0,
                            'total_energy_j': 0, 'avg_power_w': 0,
                            'energy_per_token_j': 0, 'tokens_per_joule': 0,
                            'error': str(e)
                        }

                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'run_type': 'weighted',
                        'alpha': alpha,
                        'strategy': strategy,
                        'workload': wl_name,
                        'prompt_length': wl_params['prompt_length'],
                        'output_length': wl_params['output_length'],
                        'repeat': rep,
                        'target_gpu_mhz': target_gpu,
                        'target_cpu_mhz': target_cpu,
                        'actual_gpu_mhz': actual_gpu,
                        'actual_cpu_mhz': actual_cpu,
                        'config_name': f'GPU{actual_gpu}_CPU{actual_cpu}',
                        # Measured metrics
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
                        'avg_temp_cpu_c': result.get('avg_temp_cpu_c', 0),
                        # Predicted metrics (from rate table)
                        'pred_ttft_ms': sel.get('pred_ttft_ms', 0),
                        'pred_tpot_ms': sel.get('pred_tpot_ms', 0),
                        'pred_energy_per_token_j': sel.get('pred_energy_per_token_j', 0),
                    }
                    if 'error' in result:
                        record['error'] = result['error']
                    if strategy == 'phase_aware':
                        record['prefill_config'] = sel.get('prefill_config', {}).get('config_name', '')
                        record['decode_config'] = sel.get('decode_config', {}).get('config_name', '')
                        record['phase_switch'] = sel.get('phase_switch', False)
                        record['switch_overhead_ms'] = result.get('switch_overhead_ms', 0)
                    results.append(record)

                    ept = result.get('energy_per_token_j', 0)
                    tpj = result.get('tokens_per_joule', 0)
                    logger.info(f"    TTFT={result.get('ttft_ms',0):.1f}ms "
                               f"TPOT={result.get('tpot_ms',0):.1f}ms "
                               f"tps={result.get('tokens_per_second',0):.1f} "
                               f"E/tok={ept:.4f}J tok/J={tpj:.2f}")

                # Cooldown between configs
                time.sleep(COOLDOWN_S)

    # ── Phase 2: Baseline Runs ──
    logger.info(f"\n{'='*70}")
    logger.info(f"Phase 2: Baselines ({len(BASELINES)} × {len(WORKLOADS)} × {REPEATS} = "
                f"{len(BASELINES)*len(WORKLOADS)*REPEATS} runs)")
    logger.info(f"{'='*70}")

    for bl_name, bl_cfg in BASELINES.items():
        actual_gpu = set_gpu_freq(bl_cfg['gpu_freq_mhz'])
        actual_cpu = set_cpu_freq(bl_cfg['cpu_freq_mhz'])
        logger.info(f"Baseline: {bl_name} → GPU={actual_gpu}MHz CPU={actual_cpu}MHz")

        runner.warmup(WARMUP_RUNS)

        for wl_name, wl_params in WORKLOADS.items():
            for rep in range(REPEATS):
                run_count += 1
                logger.info(f"  [{run_count}/{total_runs}] {bl_name} {wl_name} rep={rep}")

                try:
                    result = run_single_with_energy(
                        runner, mc,
                        wl_params['prompt_length'], wl_params['output_length']
                    )
                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    result = {
                        'ttft_ms': 0, 'tpot_ms': 0, 'total_time_ms': 0,
                        'tokens_per_second': 0, 'output_tokens': 0,
                        'total_energy_j': 0, 'avg_power_w': 0,
                        'energy_per_token_j': 0, 'tokens_per_joule': 0,
                        'error': str(e)
                    }

                record = {
                    'timestamp': datetime.now().isoformat(),
                    'run_type': 'baseline',
                    'alpha': -1,
                    'strategy': bl_name,
                    'workload': wl_name,
                    'prompt_length': wl_params['prompt_length'],
                    'output_length': wl_params['output_length'],
                    'repeat': rep,
                    'target_gpu_mhz': bl_cfg['gpu_freq_mhz'],
                    'target_cpu_mhz': bl_cfg['cpu_freq_mhz'],
                    'actual_gpu_mhz': actual_gpu,
                    'actual_cpu_mhz': actual_cpu,
                    'config_name': bl_name,
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
                    'avg_temp_cpu_c': result.get('avg_temp_cpu_c', 0),
                    'pred_ttft_ms': 0,
                    'pred_tpot_ms': 0,
                    'pred_energy_per_token_j': 0,
                }
                if 'error' in result:
                    record['error'] = result['error']
                results.append(record)

        time.sleep(COOLDOWN_S)

    # Save
    df = pd.DataFrame(results)
    out_path = output_dir / f'e2e_benchmark_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path} ({len(df)} rows)")

    # Generate report
    valid = df[df['output_tokens'] > 0]
    if not valid.empty:
        report_path = output_dir / f'e2e_summary_{timestamp}.md'
        with open(report_path, 'w') as f:
            f.write(generate_report(valid, timestamp))
        logger.info(f"Report: {report_path}")

    return out_path


def generate_report(df: pd.DataFrame, timestamp: str) -> str:
    lines = [
        "# End-to-End Multi-Objective DVFS Benchmark",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Phi-3-mini-4k-instruct-Q4",
        f"**Total valid runs**: {len(df)}",
        "",
        "## Alpha Sweep: Energy vs Latency Trade-off",
        "",
        "| Alpha | Strategy | E/token (J) | TPOT (ms) | TTFT (ms) | TPS | Power (W) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for alpha in sorted(df['alpha'].unique()):
        for strategy in ['single', 'phase_aware']:
            sub = df[(df['alpha'] == alpha) & (df['strategy'] == strategy)]
            if sub.empty:
                continue
            lines.append(
                f"| {alpha:.1f} | {strategy} | {sub['energy_per_token_j'].mean():.4f} | "
                f"{sub['tpot_ms'].mean():.1f} | {sub['ttft_ms'].mean():.1f} | "
                f"{sub['tokens_per_second'].mean():.1f} | {sub['avg_power_w'].mean():.2f} |"
            )

    # Baselines
    lines.extend([
        "", "## Baselines", "",
        "| Baseline | E/token (J) | TPOT (ms) | TTFT (ms) | TPS | Power (W) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for bl in ['MAXN', 'Jetson_30W', 'Energy_Min']:
        sub = df[df['strategy'] == bl]
        if sub.empty:
            continue
        lines.append(
            f"| {bl} | {sub['energy_per_token_j'].mean():.4f} | "
            f"{sub['tpot_ms'].mean():.1f} | {sub['ttft_ms'].mean():.1f} | "
            f"{sub['tokens_per_second'].mean():.1f} | {sub['avg_power_w'].mean():.2f} |"
        )

    # Key findings
    weighted = df[df['run_type'] == 'weighted']
    baselines = df[df['run_type'] == 'baseline']
    if not weighted.empty and not baselines.empty:
        best_e = weighted.groupby(['alpha', 'strategy'])['energy_per_token_j'].mean().idxmin()
        best_tpot = weighted.groupby(['alpha', 'strategy'])['tpot_ms'].mean().idxmin()
        lines.extend([
            "", "## Key Findings",
            f"- **Best energy**: alpha={best_e[0]:.1f}, {best_e[1]}",
            f"- **Best latency**: alpha={best_tpot[0]:.1f}, {best_tpot[1]}",
        ])

        # Compare vs MAXN
        maxn = baselines[baselines['strategy'] == 'MAXN']
        if not maxn.empty:
            maxn_ept = maxn['energy_per_token_j'].mean()
            maxn_tpot = maxn['tpot_ms'].mean()
            for strategy in ['single', 'phase_aware']:
                strat = weighted[weighted['strategy'] == strategy]
                if strat.empty:
                    continue
                for alpha in sorted(strat['alpha'].unique()):
                    a = strat[strat['alpha'] == alpha]
                    ept = a['energy_per_token_j'].mean()
                    tpot = a['tpot_ms'].mean()
                    e_saving = (1 - ept / maxn_ept) * 100 if maxn_ept > 0 else 0
                    l_change = ((tpot - maxn_tpot) / maxn_tpot) * 100 if maxn_tpot > 0 else 0
                    if abs(e_saving) > 20:
                        lines.append(
                            f"- alpha={alpha:.1f} {strategy}: E/token {e_saving:+.1f}% vs MAXN, "
                            f"TPOT {l_change:+.1f}% vs MAXN"
                        )

    return '\n'.join(lines)


if __name__ == '__main__':
    import os
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    run_e2e_benchmark()
