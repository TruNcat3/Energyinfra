#!/usr/bin/env python3
"""
True Phase-Switching E2E Benchmark

Implements REAL online phase switching: different frequency configs for
prefill and decode, with actual mid-inference frequency switch.

Experiment matrix:
  1. Phase-switching runs: prefill config → switch → decode config
  2. Single-config runs (for comparison)
  3. Extended workloads with longer decode (512/1024/2048 tokens)
  4. Weak baselines (dynamic governor, random config, etc.)

Usage:
    sudo python3 src/experiments/run_phase_switch_e2e.py
    sudo python3 src/experiments/run_phase_switch_e2e.py --quick
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import re
import time
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.metrics.metrics_collector import MetricsCollector
from src.controller.weighted_selector import WeightedSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Frequency control ──
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

# Standard workloads + extended decode workloads
WORKLOADS = {
    'p128_o64':    {'prompt_length': 128,  'output_length': 64},
    'p512_o128':   {'prompt_length': 512,  'output_length': 128},
    'p1024_o128':  {'prompt_length': 1024, 'output_length': 128},
    'p128_o256':   {'prompt_length': 128,  'output_length': 256},
    'p1024_o512':  {'prompt_length': 1024, 'output_length': 512},
    # Extended decode workloads
    'p128_o512':   {'prompt_length': 128,  'output_length': 512},
    'p128_o1024':  {'prompt_length': 128,  'output_length': 1024},
    'p512_o1024':  {'prompt_length': 512,  'output_length': 1024},
}

BASELINES = {
    'MAXN':       {'gpu': 1300, 'emc': 3199, 'cpu': 1497},
    '30W_mode':   {'gpu': 612,  'emc': 2133, 'cpu': 1497},
    'E_min':      {'gpu': 816,  'emc': 204,  'cpu': 1036},
}

REPEATS = 3
WARMUP_RUNS = 1
COOLDOWN_S = 3
TEGRASTATS_MS = 500


def set_gpu(mhz):
    hz = GPU_FREQS_HZ[mhz]
    with open(f'{GPU_PATH}/governor', 'w') as f:
        f.write('performance')
    with open(f'{GPU_PATH}/max_freq', 'w') as f:
        f.write(str(hz))
    with open(f'{GPU_PATH}/min_freq', 'w') as f:
        f.write(str(hz))
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
    with open(EMC_MAX, 'w') as f:
        f.write(str(max_hz))
    with open(EMC_MIN, 'w') as f:
        f.write('204000000')
    with open(EMC_MIN, 'w') as f:
        f.write(str(hz))
    with open(EMC_MAX, 'w') as f:
        f.write(str(hz))
    time.sleep(0.3)
    return int(open(EMC_CLK).read().strip()) // 1000000


def set_all(gpu, emc, cpu):
    ag = set_gpu(gpu)
    ae = set_emc(emc)
    ac = set_cpu(cpu)
    return ag, ae, ac


def set_dynamic_governor():
    """Set GPU to simple_ondemand (dynamic governor) — weak baseline."""
    max_gpu = max(GPU_FREQS_HZ.values())
    min_gpu = min(GPU_FREQS_HZ.values())
    with open(f'{GPU_PATH}/max_freq', 'w') as f:
        f.write(str(max_gpu))
    with open(f'{GPU_PATH}/min_freq', 'w') as f:
        f.write(str(min_gpu))
    try:
        with open(f'{GPU_PATH}/governor', 'w') as f:
            f.write('simple_ondemand')
    except OSError:
        with open(f'{GPU_PATH}/governor', 'w') as f:
            f.write('performance')
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
    with open(EMC_MIN, 'w') as f:
        f.write('204000000')
    with open(EMC_MAX, 'w') as f:
        f.write('3199000000')
    time.sleep(0.5)
    actual_gpu = int(open(f'{GPU_PATH}/cur_freq').read().strip()) // 1000000
    actual_emc = int(open(EMC_CLK).read().strip()) // 1000000
    actual_cpu = int(open(f'{CPU_PATH}/scaling_cur_freq').read().strip()) // 1000
    return actual_gpu, actual_emc, actual_cpu


def restore_max():
    max_gpu = max(GPU_FREQS_HZ.values())
    min_gpu = min(GPU_FREQS_HZ.values())
    with open(f'{GPU_PATH}/max_freq', 'w') as f:
        f.write(str(max_gpu))
    with open(f'{GPU_PATH}/min_freq', 'w') as f:
        f.write(str(min_gpu))
    try:
        with open(f'{GPU_PATH}/governor', 'w') as f:
            f.write('simple_ondemand')
    except OSError:
        with open(f'{GPU_PATH}/governor', 'w') as f:
            f.write('performance')
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
    with open(EMC_MIN, 'w') as f:
        f.write('204000000')
    with open(EMC_MAX, 'w') as f:
        f.write('3199000000')


def parse_tegrastats_with_timestamps(log_path: Path) -> pd.DataFrame:
    """Parse tegrastats log with timestamps for phase-split energy."""
    samples = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line or 'RAM' not in line:
                continue
            m = {}
            ts = re.match(r'(\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})', line)
            if ts:
                try:
                    m['timestamp'] = pd.Timestamp(ts.group(1))
                except ValueError:
                    continue
            else:
                continue

            ram = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
            if ram:
                m['ram_used_mb'] = int(ram.group(1))

            power_total = 0
            for match in re.finditer(r'(VDD_\w+|VIN_\w+)\s+(\d+)mW/(\d+)mW', line):
                rail, avg_mw = match.group(1), int(match.group(2))
                m[f'power_{rail}_mw'] = avg_mw
                power_total += avg_mw
            m['power_total_mw'] = power_total

            for tmatch in re.finditer(r'(\w+)@([\d.]+)C', line):
                m[f'temp_{tmatch.group(1)}_c'] = float(tmatch.group(2))

            if m:
                samples.append(m)

    return pd.DataFrame(samples) if samples else pd.DataFrame()


def compute_phase_split_energy(
    mc: MetricsCollector,
    prefill_start: float,
    prefill_end: float,
    decode_end: float,
) -> Dict:
    """Compute separate energy for prefill and decode phases using timestamps."""
    df = parse_tegrastats_with_timestamps(mc.log_file)
    if df.empty or 'timestamp' not in df.columns:
        energy = mc.compute_energy()
        return {
            'total_energy_j': energy.get('total_energy_j', 0),
            'prefill_energy_j': 0,
            'decode_energy_j': 0,
            'avg_power_w': energy.get('avg_power_w', 0),
            'prefill_avg_power_w': 0,
            'decode_avg_power_w': 0,
        }

    # Convert wall clock times to approximate tegrastats timestamps
    # tegrastats logs wall clock time, so we match by time windows
    ts_min = df['timestamp'].min()
    ts_max = df['timestamp'].max()

    prefill_duration_s = prefill_end - prefill_start
    decode_duration_s = decode_end - prefill_end

    if 'power_total_mw' not in df.columns:
        return {'total_energy_j': 0, 'prefill_energy_j': 0, 'decode_energy_j': 0}

    # Simple split: first N samples are prefill, rest are decode
    total_duration = decode_end - prefill_start
    if total_duration <= 0:
        return {'total_energy_j': 0, 'prefill_energy_j': 0, 'decode_energy_j': 0}

    n_total = len(df)
    n_prefill = max(1, int(n_total * prefill_duration_s / total_duration))
    n_decode = n_total - n_prefill

    prefill_df = df.iloc[:n_prefill]
    decode_df = df.iloc[n_prefill:]

    prefill_avg_mw = prefill_df['power_total_mw'].mean() if not prefill_df.empty else 0
    decode_avg_mw = decode_df['power_total_mw'].mean() if not decode_df.empty else 0

    prefill_energy_j = prefill_avg_mw * prefill_duration_s / 1000.0
    decode_energy_j = decode_avg_mw * decode_duration_s / 1000.0
    total_energy_j = prefill_energy_j + decode_energy_j

    return {
        'total_energy_j': round(total_energy_j, 3),
        'prefill_energy_j': round(prefill_energy_j, 3),
        'decode_energy_j': round(decode_energy_j, 3),
        'avg_power_w': round(df['power_total_mw'].mean() / 1000.0, 3),
        'prefill_avg_power_w': round(prefill_avg_mw / 1000.0, 3),
        'decode_avg_power_w': round(decode_avg_mw / 1000.0, 3),
        'prefill_duration_s': round(prefill_duration_s, 3),
        'decode_duration_s': round(decode_duration_s, 3),
        'prefill_samples': n_prefill,
        'decode_samples': n_decode,
    }


def run_phase_switch(
    runner: LlamaCppRunner,
    mc: MetricsCollector,
    prompt_length: int,
    output_length: int,
    prefill_gpu: int, prefill_emc: int, prefill_cpu: int,
    decode_gpu: int, decode_emc: int, decode_cpu: int,
) -> Dict:
    """
    Run inference with TRUE phase switching:
    1. Set prefill config
    2. Start tegrastats
    3. Run inference, switch frequency at first token (prefill→decode boundary)
    4. Stop tegrastats
    5. Compute separate prefill/decode energy
    """
    # Set prefill config
    ag = set_gpu(prefill_gpu)
    ae = set_emc(prefill_emc)
    ac = set_cpu(prefill_cpu)

    # Phase switch callback: switch to decode config
    switch_info = {}
    def on_phase_switch(info):
        switch_info['prefill_time_ms'] = info['prefill_time_ms']
        switch_info['switch_start'] = time.monotonic()
        set_gpu(decode_gpu)
        set_emc(decode_emc)
        set_cpu(decode_cpu)
        switch_info['switch_end'] = time.monotonic()
        switch_info['switch_overhead_ms'] = (switch_info['switch_end'] - switch_info['switch_start']) * 1000

    # Run with energy measurement
    tag = f"ps_p{prompt_length}_o{output_length}"
    mc.start_collection(tag=tag)
    prefill_start = time.monotonic()

    result = runner.run_phase_split_inference(
        prompt_length=prompt_length,
        output_length=output_length,
        phase_switch_callback=on_phase_switch,
    )

    decode_end = time.monotonic()
    time.sleep(0.3)
    mc.stop_collection()

    first_token_time = prefill_start + result['ttft_ms'] / 1000.0
    phase_energy = compute_phase_split_energy(
        mc, prefill_start, first_token_time, decode_end
    )

    result['actual_prefill_gpu'] = ag
    result['actual_prefill_emc'] = ae
    result['actual_prefill_cpu'] = ac
    result['actual_decode_gpu'] = int(open(f'{GPU_PATH}/cur_freq').read().strip()) // 1000000
    result['actual_decode_emc'] = int(open(EMC_CLK).read().strip()) // 1000000
    result['actual_decode_cpu'] = int(open(f'{CPU_PATH}/scaling_cur_freq').read().strip()) // 1000
    result['switch_overhead_ms'] = switch_info.get('switch_overhead_ms', 0)
    result['total_energy_j'] = phase_energy['total_energy_j']
    result['prefill_energy_j'] = phase_energy['prefill_energy_j']
    result['decode_energy_j'] = phase_energy['decode_energy_j']
    result['prefill_avg_power_w'] = phase_energy.get('prefill_avg_power_w', 0)
    result['decode_avg_power_w'] = phase_energy.get('decode_avg_power_w', 0)
    result['avg_power_w'] = phase_energy.get('avg_power_w', 0)

    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
        result['decode_energy_per_token_j'] = (
            result['decode_energy_j'] / max(result.get('decode_tokens', result['output_tokens']), 1)
        )
    else:
        result['energy_per_token_j'] = 0
        result['decode_energy_per_token_j'] = 0

    return result


def run_single_fixed(
    runner: LlamaCppRunner,
    mc: MetricsCollector,
    prompt_length: int,
    output_length: int,
    gpu: int, emc: int, cpu: int,
) -> Dict:
    """Run single inference with fixed config (no phase switching)."""
    ag = set_gpu(gpu)
    ae = set_emc(emc)
    ac = set_cpu(cpu)

    tag = f"fixed_p{prompt_length}_o{output_length}"
    mc.start_collection(tag=tag)
    result = runner.run_single_inference(
        prompt_length=prompt_length, output_length=output_length
    )
    time.sleep(0.3)
    mc.stop_collection()
    energy = mc.compute_energy()
    result.update({
        'actual_gpu_mhz': ag,
        'actual_emc_mhz': ae,
        'actual_cpu_mhz': ac,
        'avg_power_w': energy.get('avg_power_w', 0),
        'total_energy_j': energy.get('total_energy_j', 0),
    })
    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
    else:
        result['energy_per_token_j'] = 0
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

    # Use subset of workloads in quick mode
    if quick:
        workloads = {
            'p128_o128':  {'prompt_length': 128,  'output_length': 128},
            'p128_o1024': {'prompt_length': 128,  'output_length': 1024},
        }
        alphas = [0.0, 0.5, 1.0]
        repeats = 2
    else:
        workloads = WORKLOADS
        alphas = [0.0, 0.5, 1.0]
        repeats = REPEATS

    results: List[Dict] = []
    run_count = 0

    # ── Part 1: True Phase-Switching Runs ──
    logger.info("\n" + "="*60)
    logger.info("PART 1: True Phase-Switching DVFS")
    logger.info("="*60)

    for alpha in alphas:
        for wl_name, wl_params in workloads.items():
            for rep in range(repeats):
                run_count += 1
                pl = wl_params['prompt_length']
                ol = wl_params['output_length']

                cfgs = ws.select_phase_aware_configs(pl, ol, alpha)
                if cfgs is None:
                    logger.warning(f"  No phase-aware configs for {wl_name} alpha={alpha}")
                    continue

                pcfg = cfgs['prefill_config']
                dcfg = cfgs['decode_config']

                logger.info(f"\n[{run_count}] PS alpha={alpha} {wl_name} rep={rep}")
                logger.info(f"  Prefill: GPU{pcfg['gpu_freq_mhz']} EMC{pcfg.get('emc_freq_mhz',204)}")
                logger.info(f"  Decode:  GPU{dcfg['gpu_freq_mhz']} EMC{dcfg.get('emc_freq_mhz',204)}")

                try:
                    runner.warmup(WARMUP_RUNS)
                    result = run_phase_switch(
                        runner, mc, pl, ol,
                        pcfg['gpu_freq_mhz'], pcfg.get('emc_freq_mhz', 204), pcfg.get('cpu_freq_mhz', 1036),
                        dcfg['gpu_freq_mhz'], dcfg.get('emc_freq_mhz', 204), dcfg.get('cpu_freq_mhz', 1036),
                    )
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'true_phase_switch',
                        'alpha': alpha,
                        'workload': wl_name,
                        'prompt_length': pl,
                        'output_length': ol,
                        'target_output_tokens': ol,
                        'repeat': rep,
                        'prefill_gpu': pcfg['gpu_freq_mhz'],
                        'prefill_emc': pcfg.get('emc_freq_mhz', 204),
                        'prefill_cpu': pcfg.get('cpu_freq_mhz', 1036),
                        'decode_gpu': dcfg['gpu_freq_mhz'],
                        'decode_emc': dcfg.get('emc_freq_mhz', 204),
                        'decode_cpu': dcfg.get('cpu_freq_mhz', 1036),
                        'actual_prefill_gpu': result.get('actual_prefill_gpu', 0),
                        'actual_prefill_emc': result.get('actual_prefill_emc', 0),
                        'actual_decode_gpu': result.get('actual_decode_gpu', 0),
                        'actual_decode_emc': result.get('actual_decode_emc', 0),
                        'ttft_ms': result.get('ttft_ms', 0),
                        'tpot_ms': result.get('tpot_ms', 0),
                        'total_time_ms': result.get('total_time_ms', 0),
                        'output_tokens': result.get('output_tokens', 0),
                        'decode_tokens': result.get('decode_tokens', 0),
                        'tokens_per_second': result.get('tokens_per_second', 0),
                        'switch_overhead_ms': result.get('switch_overhead_ms', 0),
                        'total_energy_j': result.get('total_energy_j', 0),
                        'prefill_energy_j': result.get('prefill_energy_j', 0),
                        'decode_energy_j': result.get('decode_energy_j', 0),
                        'prefill_avg_power_w': result.get('prefill_avg_power_w', 0),
                        'decode_avg_power_w': result.get('decode_avg_power_w', 0),
                        'avg_power_w': result.get('avg_power_w', 0),
                        'energy_per_token_j': result.get('energy_per_token_j', 0),
                        'decode_energy_per_token_j': result.get('decode_energy_per_token_j', 0),
                        'is_complete_output': result.get('output_tokens', 0) >= ol * 0.9,
                        'model': 'Phi-3-mini-Q4',
                        'experiment': 'phase_switch_e2e',
                    }
                    results.append(record)
                    logger.info(f"  TTFT={result.get('ttft_ms',0):.1f}ms "
                               f"TPOT={result.get('tpot_ms',0):.1f}ms "
                               f"switch={result.get('switch_overhead_ms',0):.1f}ms "
                               f"E/tok={result.get('energy_per_token_j',0):.4f}J "
                               f"pre={result.get('prefill_energy_j',0):.3f}J "
                               f"dec={result.get('decode_energy_j',0):.3f}J")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    results.append({
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'true_phase_switch', 'alpha': alpha,
                        'workload': wl_name, 'repeat': rep, 'error': str(e),
                    })

                time.sleep(COOLDOWN_S)

    # ── Part 2: Single Config (decode-only) for Comparison ──
    logger.info("\n" + "="*60)
    logger.info("PART 2: Single Config (No Switch)")
    logger.info("="*60)

    for alpha in alphas:
        for wl_name, wl_params in workloads.items():
            for rep in range(repeats):
                run_count += 1
                pl = wl_params['prompt_length']
                ol = wl_params['output_length']

                cfg = ws.select_config(pl, ol, 'decode', alpha)
                if cfg is None:
                    continue

                gpu = cfg['gpu_freq_mhz']
                emc = cfg.get('emc_freq_mhz', 204)
                cpu = cfg.get('cpu_freq_mhz', 1036)

                logger.info(f"\n[{run_count}] SC alpha={alpha} {wl_name} rep={rep} "
                           f"GPU{gpu} EMC{emc}")

                try:
                    runner.warmup(WARMUP_RUNS)
                    result = run_single_fixed(runner, mc, pl, ol, gpu, emc, cpu)
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'single_config',
                        'alpha': alpha,
                        'workload': wl_name,
                        'prompt_length': pl,
                        'output_length': ol,
                        'target_output_tokens': ol,
                        'repeat': rep,
                        'target_gpu_mhz': gpu,
                        'target_emc_mhz': emc,
                        'target_cpu_mhz': cpu,
                        'actual_gpu_mhz': result.get('actual_gpu_mhz', 0),
                        'actual_emc_mhz': result.get('actual_emc_mhz', 0),
                        'actual_cpu_mhz': result.get('actual_cpu_mhz', 0),
                        'ttft_ms': result.get('ttft_ms', 0),
                        'tpot_ms': result.get('tpot_ms', 0),
                        'total_time_ms': result.get('total_time_ms', 0),
                        'output_tokens': result.get('output_tokens', 0),
                        'tokens_per_second': result.get('tokens_per_second', 0),
                        'total_energy_j': result.get('total_energy_j', 0),
                        'avg_power_w': result.get('avg_power_w', 0),
                        'energy_per_token_j': result.get('energy_per_token_j', 0),
                        'is_complete_output': result.get('output_tokens', 0) >= ol * 0.9,
                        'model': 'Phi-3-mini-Q4',
                        'experiment': 'phase_switch_e2e',
                    }
                    results.append(record)
                    logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                               f"E/tok={result.get('energy_per_token_j',0):.4f}J")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    results.append({
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'single_config', 'alpha': alpha,
                        'workload': wl_name, 'repeat': rep, 'error': str(e),
                    })

                time.sleep(COOLDOWN_S)

    # ── Part 3: Strong Baselines (MAXN, 30W, E_min) ──
    logger.info("\n" + "="*60)
    logger.info("PART 3: Strong Baselines")
    logger.info("="*60)

    for bl_name, bl_cfg in BASELINES.items():
        for wl_name, wl_params in workloads.items():
            for rep in range(repeats):
                run_count += 1
                pl = wl_params['prompt_length']
                ol = wl_params['output_length']

                logger.info(f"\n[{run_count}] BL {bl_name} {wl_name} rep={rep}")

                try:
                    runner.warmup(WARMUP_RUNS)
                    result = run_single_fixed(
                        runner, mc, pl, ol,
                        bl_cfg['gpu'], bl_cfg['emc'], bl_cfg['cpu']
                    )
                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'baseline',
                        'alpha': -1,
                        'baseline_name': bl_name,
                        'workload': wl_name,
                        'prompt_length': pl,
                        'output_length': ol,
                        'target_output_tokens': ol,
                        'repeat': rep,
                        'target_gpu_mhz': bl_cfg['gpu'],
                        'target_emc_mhz': bl_cfg['emc'],
                        'target_cpu_mhz': bl_cfg['cpu'],
                        'actual_gpu_mhz': result.get('actual_gpu_mhz', 0),
                        'actual_emc_mhz': result.get('actual_emc_mhz', 0),
                        'actual_cpu_mhz': result.get('actual_cpu_mhz', 0),
                        'ttft_ms': result.get('ttft_ms', 0),
                        'tpot_ms': result.get('tpot_ms', 0),
                        'total_time_ms': result.get('total_time_ms', 0),
                        'output_tokens': result.get('output_tokens', 0),
                        'tokens_per_second': result.get('tokens_per_second', 0),
                        'total_energy_j': result.get('total_energy_j', 0),
                        'avg_power_w': result.get('avg_power_w', 0),
                        'energy_per_token_j': result.get('energy_per_token_j', 0),
                        'is_complete_output': result.get('output_tokens', 0) >= ol * 0.9,
                        'model': 'Phi-3-mini-Q4',
                        'experiment': 'phase_switch_e2e',
                    }
                    results.append(record)
                    logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                               f"E/tok={result.get('energy_per_token_j',0):.4f}J")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    results.append({
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'baseline', 'baseline_name': bl_name,
                        'workload': wl_name, 'repeat': rep, 'error': str(e),
                    })

                time.sleep(COOLDOWN_S)

    # ── Part 4: Weak Baselines ──
    logger.info("\n" + "="*60)
    logger.info("PART 4: Weak Baselines (Dynamic Governor, Random Config)")
    logger.info("="*60)

    weak_baselines = {
        'dynamic_governor': 'dynamic',
        'min_freq': {'gpu': 306, 'emc': 204, 'cpu': 1036},
        'random_config': 'random',
    }

    for wb_name, wb_cfg in weak_baselines.items():
        # Only test on a subset of workloads for weak baselines
        wb_workloads = {
            'p128_o128':  {'prompt_length': 128,  'output_length': 128},
            'p128_o1024': {'prompt_length': 128,  'output_length': 1024},
        }
        for wl_name, wl_params in wb_workloads.items():
            for rep in range(repeats):
                run_count += 1
                pl = wl_params['prompt_length']
                ol = wl_params['output_length']

                logger.info(f"\n[{run_count}] WEAK {wb_name} {wl_name} rep={rep}")

                try:
                    if wb_name == 'dynamic_governor':
                        ag, ae, ac = set_dynamic_governor()
                    elif wb_name == 'random_config':
                        gpu = random.choice(list(GPU_FREQS_HZ.keys()))
                        emc = random.choice(list(EMC_HZ.keys()))
                        cpu = random.choice(list(CPU_HZ.keys()))
                        ag, ae, ac = set_all(gpu, emc, cpu)
                    else:
                        ag, ae, ac = set_all(wb_cfg['gpu'], wb_cfg['emc'], wb_cfg['cpu'])

                    runner.warmup(WARMUP_RUNS)
                    tag = f"weak_{wb_name}_p{pl}_o{ol}"
                    mc.start_collection(tag=tag)
                    result = runner.run_single_inference(
                        prompt_length=pl, output_length=ol
                    )
                    time.sleep(0.3)
                    mc.stop_collection()
                    energy = mc.compute_energy()
                    result['total_energy_j'] = energy.get('total_energy_j', 0)
                    result['avg_power_w'] = energy.get('avg_power_w', 0)
                    if result['output_tokens'] > 0 and result['total_energy_j'] > 0:
                        result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
                    else:
                        result['energy_per_token_j'] = 0

                    record = {
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'weak_baseline',
                        'alpha': -1,
                        'baseline_name': wb_name,
                        'workload': wl_name,
                        'prompt_length': pl,
                        'output_length': ol,
                        'target_output_tokens': ol,
                        'repeat': rep,
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
                        'is_complete_output': result.get('output_tokens', 0) >= ol * 0.9,
                        'model': 'Phi-3-mini-Q4',
                        'experiment': 'phase_switch_e2e',
                    }
                    results.append(record)
                    logger.info(f"  TPOT={result.get('tpot_ms',0):.1f}ms "
                               f"E/tok={result.get('energy_per_token_j',0):.4f}J")
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    results.append({
                        'timestamp': datetime.now().isoformat(),
                        'strategy': 'weak_baseline', 'baseline_name': wb_name,
                        'workload': wl_name, 'repeat': rep, 'error': str(e),
                    })

                time.sleep(COOLDOWN_S)

    # ── Save results BEFORE restore ──
    df = pd.DataFrame(results)
    out_path = output_dir / f'phase_switch_e2e_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path} ({len(df)} rows)")

    # Summary
    valid = df[(df.get('output_tokens', 0) > 0) & (df.get('energy_per_token_j', 0) > 0)]
    if not valid.empty:
        report_path = output_dir / f'phase_switch_report_{timestamp}.md'
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
        "# Phase-Switching E2E Benchmark Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Model**: Phi-3-mini-Q4",
        f"**Valid runs**: {len(df)}",
        "",
        "## 1. Phase-Switching vs Single Config (E/tok by Alpha)",
        "",
        "| Strategy | Alpha | E/tok (J) | TTFT (ms) | TPOT (ms) | Switch (ms) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for strategy in ['true_phase_switch', 'single_config']:
        for alpha in sorted(df[df['strategy'] == strategy]['alpha'].unique()):
            sub = df[(df['strategy'] == strategy) & (df['alpha'] == alpha)]
            sw = sub.get('switch_overhead_ms', pd.Series([0]*len(sub)))
            lines.append(
                f"| {strategy} | {alpha} | {sub['energy_per_token_j'].mean():.4f} "
                f"| {sub['ttft_ms'].mean():.1f} | {sub['tpot_ms'].mean():.1f} "
                f"| {sw.mean():.1f} |"
            )

    # Baselines
    lines.extend(["", "## 2. Baselines", "",
                  "| Baseline | Type | E/tok (J) | TPOT (ms) | TPS |",
                  "|:---|:---:|:---:|:---:|:---:|"])
    for bl_type in ['baseline', 'weak_baseline']:
        bls = df[df['strategy'] == bl_type]
        for bl in sorted(bls['baseline_name'].unique()):
            sub = bls[bls['baseline_name'] == bl]
            lines.append(
                f"| {bl} | {bl_type} | {sub['energy_per_token_j'].mean():.4f} "
                f"| {sub['tpot_ms'].mean():.1f} | {sub['tokens_per_second'].mean():.1f} |"
            )

    # Phase-split energy breakdown
    ps = df[df['strategy'] == 'true_phase_switch']
    if not ps.empty and 'prefill_energy_j' in ps.columns:
        lines.extend(["", "## 3. Phase Energy Breakdown (Phase-Switching)", "",
                      "| Alpha | Workload | Prefill (J) | Decode (J) | Total (J) | Prefill% | Switch (ms) |",
                      "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"])
        for alpha in sorted(ps['alpha'].unique()):
            for wl in sorted(ps['workload'].unique()):
                sub = ps[(ps['alpha'] == alpha) & (ps['workload'] == wl)]
                if not sub.empty:
                    pf = sub['prefill_energy_j'].mean()
                    dc = sub['decode_energy_j'].mean()
                    tot = pf + dc
                    pf_pct = (pf / tot * 100) if tot > 0 else 0
                    sw = sub.get('switch_overhead_ms', pd.Series([0]*len(sub)))
                    lines.append(
                        f"| {alpha} | {wl} | {pf:.3f} | {dc:.3f} | {tot:.3f} "
                        f"| {pf_pct:.0f}% | {sw.mean():.1f} |"
                    )

    # Extended workload comparison
    lines.extend(["", "## 4. Extended Decode Workloads (Alpha=0.5)", "",
                  "| Workload | Output Tokens | Phase-Switch E/tok | Single E/tok | MAXN E/tok | PS vs MAXN |",
                  "|:---:|:---:|:---:|:---:|:---:|:---:|"])
    maxn = df[df['baseline_name'] == 'MAXN']
    sc = df[(df['strategy'] == 'single_config') & (df['alpha'] == 0.5)]
    ps05 = df[(df['strategy'] == 'true_phase_switch') & (df['alpha'] == 0.5)]
    for wl in sorted(df['workload'].unique()):
        ps_sub = ps05[ps05['workload'] == wl]
        sc_sub = sc[sc['workload'] == wl]
        maxn_sub = maxn[maxn['workload'] == wl]
        if not ps_sub.empty and not maxn_sub.empty:
            ps_ept = ps_sub['energy_per_token_j'].mean()
            sc_ept = sc_sub['energy_per_token_j'].mean() if not sc_sub.empty else 0
            maxn_ept = maxn_sub['energy_per_token_j'].mean()
            saving = (1 - ps_ept / maxn_ept) * 100 if maxn_ept > 0 else 0
            ol = ps_sub['output_length'].iloc[0]
            lines.append(
                f"| {wl} | {ol} | {ps_ept:.4f} | {sc_ept:.4f} | {maxn_ept:.4f} | {saving:+.1f}% |"
            )

    # Complete vs incomplete
    if 'is_complete_output' in df.columns:
        complete = df[df['is_complete_output'] == True]
        incomplete = df[df['is_complete_output'] == False]
        lines.extend([
            "",
            "## 5. Output Completeness",
            "",
            f"- Complete runs: {len(complete)}",
            f"- Incomplete runs: {len(incomplete)}",
        ])
        if not complete.empty:
            lines.append(f"- Complete avg E/tok: {complete['energy_per_token_j'].mean():.4f} J")
        if not incomplete.empty:
            lines.append(f"- Incomplete avg E/tok: {incomplete['energy_per_token_j'].mean():.4f} J")

    return '\n'.join(lines)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Phase-Switching E2E Benchmark')
    parser.add_argument('--rate-table', type=str,
                        default='data/rate_tables/finegrained_selector_table_20260516_071238.parquet',
                        help='Path to fine-grained selector table')
    parser.add_argument('--quick', action='store_true', help='Quick mode (2 workloads, 3 alphas, 2 repeats)')
    args = parser.parse_args()

    logger.info("Phase-Switching E2E Benchmark")
    logger.info("=" * 60)
    logger.info("This experiment implements TRUE online phase switching:")
    logger.info("  1. Set prefill config → run prefill → detect first token")
    logger.info("  2. Switch to decode config (mid-inference)")
    logger.info("  3. Continue decode at new frequency")
    logger.info("  4. Measure separate prefill/decode energy")
    logger.info("=" * 60)

    run_experiment(rate_table_path=args.rate_table, quick=args.quick)
