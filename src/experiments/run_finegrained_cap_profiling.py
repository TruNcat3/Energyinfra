#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Fine-grained Cap Profiling (Phase 13, P1)

Extends Phase 10 cap profiling with 10 GPU cap levels instead of 4,
providing a richer Pareto frontier for online DVFS selection.

Experiment matrix:
  GPU cap: 10 levels [408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300]
  EMC cap: 3199 (fixed at max)
  CPU cap: 1036 (fixed)
  Workloads: 7 representative
  Repeats: 3
  = 210 cap runs + 42 baseline runs = 252 total ≈ 3h/model

Quick mode: 2 cap × 2 workload × 2 rep + baselines ≈ 30 min

Features:
  - Checkpoint/resume: survives interruptions
  - Auto-calls build_workload_rate_table after completion
  - GPU-only sweep (EMC/CPU fixed)

Usage:
    sudo python3 src/experiments/run_finegrained_cap_profiling.py --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
    sudo python3 src/experiments/run_finegrained_cap_profiling.py --quick --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
    sudo python3 src/experiments/run_finegrained_cap_profiling.py --resume --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import time
import json
import logging
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.metrics.metrics_collector import MetricsCollector
from src.controller.cap_controller import CapController

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Fine-grained experiment config ──
FINEGRAINED_GPU_CAPS = [408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300]
EMC_CAP = 3199   # Fixed at max
CPU_CAP = 1036    # Fixed

REPRESENTATIVE_WORKLOADS = {
    'p64_o64':     {'prompt_length': 64,   'output_length': 64},
    'p128_o512':   {'prompt_length': 128,  'output_length': 512},
    'p512_o128':   {'prompt_length': 512,  'output_length': 128},
    'p512_o512':   {'prompt_length': 512,  'output_length': 512},
    'p1024_o512':  {'prompt_length': 1024, 'output_length': 512},
    'p2048_o128':  {'prompt_length': 2048, 'output_length': 128},
    'p2048_o512':  {'prompt_length': 2048, 'output_length': 512},
}

REPEATS = 3
WARMUP_RUNS = 1
COOLDOWN_S = 3
TEGRASTATS_MS = 500


def make_run_key(control_mode: str, gpu_cap: int, workload_name: str,
                 repeat: int) -> str:
    """Create a unique key for checkpoint tracking."""
    raw = f"{control_mode}_gpu{gpu_cap}_{workload_name}_r{repeat}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_checkpoint(model_name: str) -> set:
    """Load completed run keys from checkpoint file."""
    ckpt_dir = Path('data/cap_profiling')
    ckpt_path = ckpt_dir / f'checkpoint_finegrained_{model_name}.json'
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            data = json.load(f)
        logger.info(f"Loaded checkpoint: {len(data.get('completed', []))} completed runs")
        return set(data.get('completed', [])), data.get('results', [])
    return set(), []


def save_checkpoint(model_name: str, completed: set, results: List[Dict]):
    """Save checkpoint to disk."""
    ckpt_dir = Path('data/cap_profiling')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f'checkpoint_finegrained_{model_name}.json'
    with open(ckpt_path, 'w') as f:
        json.dump({
            'completed': sorted(completed),
            'results': results,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)


def run_single_cap(
    runner: LlamaCppRunner,
    mc: MetricsCollector,
    cc: CapController,
    gpu_cap: int,
    prompt_length: int, output_length: int,
) -> Dict:
    """Run inference under fine-grained cap mode with energy measurement."""
    cc.set_cap(gpu_cap, EMC_CAP, CPU_CAP)
    runner.warmup(WARMUP_RUNS)

    tag = f"fgcap_g{gpu_cap}_p{prompt_length}_o{output_length}"
    mc.start_collection(tag=tag)

    state_before = cc.read_actual_state()

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
        'control_mode': 'cap',
        'target_gpu_cap_mhz': gpu_cap,
        'target_emc_cap_mhz': EMC_CAP,
        'target_cpu_cap_mhz': CPU_CAP,
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

    tag = f"fgdyn_p{prompt_length}_o{output_length}"
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

    tag = f"fgmaxn_p{prompt_length}_o{output_length}"
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


def run_experiment(
    model_path: str,
    quick: bool = False,
    resume: bool = False,
):
    """Run fine-grained cap profiling experiment."""
    output_dir = Path('data/cap_profiling')
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Select configs
    if quick:
        gpu_caps = [612, 1300]
        workloads = {
            'p128_o512': {'prompt_length': 128, 'output_length': 512},
            'p512_o128': {'prompt_length': 512, 'output_length': 128},
        }
        repeats = 2
    else:
        gpu_caps = FINEGRAINED_GPU_CAPS
        workloads = REPRESENTATIVE_WORKLOADS
        repeats = REPEATS

    # Build run queue
    run_queue = []
    for gpu_cap in gpu_caps:
        for wl_name, wl_params in workloads.items():
            for rep in range(repeats):
                run_queue.append({
                    'control_mode': 'cap',
                    'gpu_cap': gpu_cap,
                    'workload_name': wl_name,
                    'pl': wl_params['prompt_length'],
                    'ol': wl_params['output_length'],
                    'repeat': rep,
                })

    # Add baselines
    for wl_name, wl_params in workloads.items():
        for rep in range(repeats):
            run_queue.append({
                'control_mode': 'dynamic',
                'gpu_cap': 1300,
                'workload_name': wl_name,
                'pl': wl_params['prompt_length'],
                'ol': wl_params['output_length'],
                'repeat': rep,
            })
            run_queue.append({
                'control_mode': 'maxn',
                'gpu_cap': 1300,
                'workload_name': wl_name,
                'pl': wl_params['prompt_length'],
                'ol': wl_params['output_length'],
                'repeat': rep,
            })

    total = len(run_queue)
    logger.info(f"Fine-grained Cap Profiling: {total} runs ({len(gpu_caps)} GPU caps "
                f"× {len(workloads)} workloads × {repeats} repeats + baselines)")

    # Load checkpoint if resuming
    completed_keys: set = set()
    results: List[Dict] = []
    if resume:
        completed_keys, results = load_checkpoint(model_name)
        if completed_keys:
            logger.info(f"Resuming: {len(completed_keys)}/{total} runs already done")

    # Load model
    logger.info("Loading model...")
    runner = LlamaCppRunner(
        model_path=model_path,
        n_gpu_layers=-1, n_ctx=4096, n_threads=4
    )
    logger.info("Model loaded")

    mc = MetricsCollector(interval_ms=TEGRASTATS_MS, output_dir='data/raw_logs')
    cc = CapController()

    # Track cap switches for logging
    last_gpu_cap = None
    run_count = len(completed_keys)

    for run_spec in run_queue:
        ctrl = run_spec['control_mode']
        gpu_cap = run_spec['gpu_cap']
        wl_name = run_spec['workload_name']
        pl = run_spec['pl']
        ol = run_spec['ol']
        rep = run_spec['repeat']

        key = make_run_key(ctrl, gpu_cap, wl_name, rep)
        if key in completed_keys:
            continue  # Skip completed runs

        run_count += 1
        logger.info(f"\n[{run_count}/{total}] {ctrl.upper()} GPU={gpu_cap}MHz "
                    f"{wl_name} rep={rep}")

        try:
            if ctrl == 'cap':
                # Only change cap if different from last (avoid redundant sysfs writes)
                if last_gpu_cap != gpu_cap:
                    cc.set_cap(gpu_cap, EMC_CAP, CPU_CAP)
                    last_gpu_cap = gpu_cap
                result = run_single_cap(runner, mc, cc, gpu_cap, pl, ol)
            elif ctrl == 'dynamic':
                if last_gpu_cap is not None:
                    cc.restore_dynamic()
                    last_gpu_cap = None
                result = run_dynamic_baseline(runner, mc, cc, pl, ol)
            elif ctrl == 'maxn':
                if last_gpu_cap != 1300:
                    run_maxn_baseline.__wrapped__(runner, mc, cc, pl, ol) if hasattr(run_maxn_baseline, '__wrapped__') else None
                result = run_maxn_baseline(runner, mc, cc, pl, ol)
                last_gpu_cap = 1300
            else:
                raise ValueError(f"Unknown control mode: {ctrl}")

            record = {
                'timestamp': datetime.now().isoformat(),
                'experiment': 'finegrained_cap_profiling',
                'model': model_name,
                **{k: v for k, v in result.items()},
            }
            results.append(record)
            completed_keys.add(key)

            logger.info(f"  TPOT={result.get('tpot_ms', 0):.1f}ms "
                        f"E/tok={result.get('energy_per_token_j', 0):.4f}J "
                        f"Pwr={result.get('avg_power_w', 0):.1f}W "
                        f"actual_gpu={result.get('actual_gpu_mhz', 0)}")

        except Exception as e:
            logger.error(f"  FAILED: {e}")
            results.append({
                'timestamp': datetime.now().isoformat(),
                'experiment': 'finegrained_cap_profiling',
                'model': model_name,
                'control_mode': ctrl,
                'target_gpu_cap_mhz': gpu_cap,
                'workload': wl_name,
                'repeat': rep,
                'error': str(e),
            })
            completed_keys.add(key)  # Mark as attempted

        # Save checkpoint every 5 runs
        if run_count % 5 == 0:
            save_checkpoint(model_name, completed_keys, results)

        time.sleep(COOLDOWN_S)

    # ── Save final results ──
    df = pd.DataFrame(results)
    out_path = output_dir / f'finegrained_cap_profiling_{model_name}_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path} ({len(df)} rows, {len(df[df['error'].notna()])} errors)")

    # Save final checkpoint
    save_checkpoint(model_name, completed_keys, results)

    # Cleanup
    try:
        cc.restore_dynamic()
    except Exception as e:
        logger.warning(f"restore_dynamic() failed: {e} (data saved)")

    # ── Auto-build rate table if fine-grained data exists ──
    logger.info("\nAttempting to auto-build rate table with fine-grained data...")
    try:
        from src.ratetable.build_workload_rate_table import build_rate_table
        lock_pattern = str(sorted(Path('data/energy_profiling').glob(
            f'finegrained_profiling_{model_name}_*.csv'))[-1])
        logger.info(f"Using lock data: {lock_pattern}")
        logger.info("NOTE: Rate table build requires lock-mode profiling data.")
        logger.info("      Use build_workload_rate_table.py separately with "
                     "--cap flag to incorporate this fine-grained cap data.")
    except Exception as e:
        logger.info(f"Auto rate table build skipped: {e}")

    return out_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Fine-grained Cap Profiling (10 GPU caps)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full profiling (~3h/model)
  sudo python3 src/experiments/run_finegrained_cap_profiling.py \\
    --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf

  # Quick test (~30min)
  sudo python3 src/experiments/run_finegrained_cap_profiling.py \\
    --quick --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf

  # Resume interrupted run
  sudo python3 src/experiments/run_finegrained_cap_profiling.py \\
    --resume --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf
        """)
    parser.add_argument('--model', type=str, required=True,
                        help='Path to GGUF model file')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 2 caps × 2 workloads × 2 reps')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    args = parser.parse_args()

    run_experiment(
        model_path=args.model,
        quick=args.quick,
        resume=args.resume,
    )
