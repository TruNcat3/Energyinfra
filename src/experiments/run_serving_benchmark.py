#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Long-running Serving Benchmark (Phase 13, P3)

Evaluates DVFS strategies under sustained inference load with thermal effects.
Simulates a serving scenario where requests arrive continuously over 30-60 minutes.

Three workload traces:
  - short_chat:   70% p64_o64, 20% p128_o128, 10% p512_o128
  - long_gen:     20% p64_o64, 30% p128_o512, 30% p512_o512, 20% p1024_o512
  - bursty_mixed: mixed + periodic bursts of 8-10 p1024_o1024 every 5min

Five baselines:
  - MAXN:          Lock at 1300MHz, no adjustment
  - Dynamic:       Default simple_ondemand governor, no adjustment
  - BestStatic:    Best single cap from rate table, no adjustment
  - Pareto:        Per-request workload-aware cap selection (no thermal feedback)
  - ThermalSLO:    Initial cap + 10s window thermal/SLO feedback control

Per-window metrics:
  window_id, baseline, trace, model, cap_mhz, cap_action,
  n_requests, tpot_p50/p95/p99, avg/max_power,
  temp_mean/max/slope, slo_violation_rate, tokens/J,
  cap_switch_count

Usage:
    sudo python3 src/experiments/run_serving_benchmark.py \
        --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --duration-min 10 --quick

    sudo python3 src/experiments/run_serving_benchmark.py \
        --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
        --duration-min 30 --trace short_chat --baseline ThermalSLO
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import time
import json
import random
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.metrics.metrics_collector import MetricsCollector
from src.controller.cap_controller import CapController
from src.controller.workload_cap_selector import WorkloadCapSelector
from src.controller.thermal_slo_controller import (
    ThermalSLOCapController, ControllerConfig, WindowState,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Workload definitions ──
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

# ── Trace distributions ──
TRACE_DISTRIBUTIONS = {
    'short_chat': [
        ('p64_o64', 0.70), ('p128_o128', 0.20), ('p512_o128', 0.10),
    ],
    'long_generation': [
        ('p64_o64', 0.20), ('p128_o512', 0.30),
        ('p512_o512', 0.30), ('p1024_o512', 0.20),
    ],
    'bursty_mixed': [
        ('p64_o64', 0.25), ('p128_o128', 0.20), ('p128_o512', 0.15),
        ('p512_o128', 0.10), ('p512_o512', 0.10), ('p1024_o512', 0.10),
        ('p2048_o128', 0.10),
    ],
}

BURST_CONFIG = {
    'workload': 'p1024_o1024',
    'count': 8,          # requests per burst
    'interval_min': 300,  # seconds between bursts (5 min)
    'interval_jitter_s': 30,
}


@dataclass
class WindowMetrics:
    """Metrics aggregated over one control window."""
    window_id: int = 0
    timestamp: float = 0.0
    baseline: str = ''
    trace: str = ''
    model: str = ''

    # Cap state
    cap_mhz: int = 1300
    cap_action: str = 'keep'

    # Request stats
    n_requests: int = 0

    # TPOT percentiles
    tpot_p50_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    tpot_p99_ms: float = 0.0

    # Power
    avg_power_w: float = 0.0
    max_power_w: float = 0.0

    # Temperature
    temp_start_c: float = 0.0
    temp_end_c: float = 0.0
    temp_mean_c: float = 0.0
    temp_max_c: float = 0.0
    temp_slope_c_per_s: float = 0.0

    # SLO compliance
    slo_violation_count: int = 0
    slo_violation_rate: float = 0.0

    # Energy efficiency
    total_tokens: int = 0
    total_energy_j: float = 0.0
    tokens_per_joule: float = 0.0

    # Performance
    tokens_per_second: float = 0.0

    # Control overhead
    cap_switch_count: int = 0


class RequestGenerator:
    """Generates request workload sequence for a given trace."""

    def __init__(self, trace_name: str, duration_s: float, seed: int = 42):
        self.trace_name = trace_name
        self.duration_s = duration_s
        self.rng = random.Random(seed)

        self.distribution = TRACE_DISTRIBUTIONS.get(trace_name)
        if not self.distribution:
            raise ValueError(f"Unknown trace: {trace_name}")

        self.workload_names = [w for w, _ in self.distribution]
        self.workload_weights = [p for _, p in self.distribution]

    def generate_requests(self) -> List[Dict]:
        """Generate a list of requests for the entire duration.

        Returns list of dicts with prompt_length, output_length, is_burst flags.
        """
        requests = []
        t = 0.0

        while t < self.duration_s:
            # Check for burst
            if (self.trace_name == 'bursty_mixed' and
                    t > BURST_CONFIG['interval_min'] and
                    self.rng.random() < 0.05):  # 5% chance per check
                burst_wl = WORKLOADS[BURST_CONFIG['workload']]
                for _ in range(BURST_CONFIG['count']):
                    requests.append({
                        'prompt_length': burst_wl['prompt_length'],
                        'output_length': burst_wl['output_length'],
                        'workload_name': BURST_CONFIG['workload'],
                        'is_burst': True,
                        'scheduled_time': t,
                    })
                    t += 2.0  # Back-to-back during burst
                continue

            # Normal request: pick from distribution
            wl_name = self.rng.choices(
                self.workload_names, weights=self.workload_weights, k=1)[0]
            wl = WORKLOADS[wl_name]
            requests.append({
                'prompt_length': wl['prompt_length'],
                'output_length': wl['output_length'],
                'workload_name': wl_name,
                'is_burst': False,
                'scheduled_time': t,
            })
            # Inter-arrival time: ~3-8s between requests
            t += self.rng.uniform(3.0, 8.0)

        return requests


def run_single_inference_with_metrics(
    runner: LlamaCppRunner,
    mc: MetricsCollector,
    cc: CapController,
    prompt_length: int,
    output_length: int,
    collect_metrics: bool = True,
) -> Dict:
    """Run a single inference with energy/thermal metrics collection."""
    if collect_metrics:
        tag = f"srv_p{prompt_length}_o{output_length}"
        mc.start_collection(tag=tag)

    state_before = cc.read_actual_state() if cc else {}

    result = runner.run_single_inference(
        prompt_length=prompt_length,
        output_length=output_length,
        benchmark_mode=True,
    )

    time.sleep(0.2)

    if collect_metrics:
        mc.stop_collection()
        energy = mc.compute_energy()

        state_after = cc.read_actual_state() if cc else {}

        result.update({
            'avg_power_w': energy.get('avg_power_w', 0),
            'max_power_w': energy.get('max_power_w', 0),
            'total_energy_j': energy.get('total_energy_j', 0),
            'temperature_c': state_after.get('temperature_c', 0),
            'actual_gpu_mhz': state_after.get('actual_gpu_mhz', 0),
        })
        if result['output_tokens'] > 0 and result.get('total_energy_j', 0) > 0:
            result['energy_per_token_j'] = result['total_energy_j'] / result['output_tokens']
            result['tokens_per_joule'] = result['output_tokens'] / result['total_energy_j']
    return result


def setup_baseline(
    baseline: str,
    cc: CapController,
    cap_selector: Optional[WorkloadCapSelector],
    thermal_controller: Optional[ThermalSLOCapController],
    tpot_slo_ms: float = 50.0,
) -> Dict:
    """Initialize a baseline's frequency control strategy.

    Returns baseline context dict.
    """
    ctx = {
        'baseline': baseline,
        'current_cap_mhz': 1300,
        'thermal_controller': None,
    }

    if baseline == 'MAXN':
        # Lock GPU/CPU/EMC to max
        GPU_SYSFS = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
        with open(f'{GPU_SYSFS}/governor', 'w') as f:
            f.write('performance')
        with open(f'{GPU_SYSFS}/max_freq', 'w') as f:
            f.write('1300500000')
        with open(f'{GPU_SYSFS}/min_freq', 'w') as f:
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
        ctx['current_cap_mhz'] = 1300
        time.sleep(0.5)

    elif baseline == 'Dynamic':
        cc.restore_dynamic()
        ctx['current_cap_mhz'] = 1300
        time.sleep(0.5)

    elif baseline == 'BestStatic':
        # Find best single cap from rate table (lowest avg E/tok)
        if cap_selector:
            try:
                # Use a medium workload as representative
                best = cap_selector.select(
                    prompt_length=512, output_length=512,
                    strategy='min_energy',
                )
                cap = best.get('gpu_cap_mhz', 918)
            except Exception:
                cap = 918  # Fallback
        else:
            cap = 918
        cc.set_cap(cap, 3199, 1036)
        ctx['current_cap_mhz'] = cap
        time.sleep(0.5)

    elif baseline == 'Pareto':
        cc.restore_dynamic()
        ctx['current_cap_mhz'] = 1300
        time.sleep(0.5)

    elif baseline == 'ThermalSLO':
        if thermal_controller:
            thermal_controller.reset()
            ctx['thermal_controller'] = thermal_controller
        cc.restore_dynamic()
        ctx['current_cap_mhz'] = 1300
        time.sleep(0.5)

    return ctx


def run_serving_benchmark(
    model_path: str,
    trace: str = 'short_chat',
    baseline: str = 'ThermalSLO',
    duration_min: float = 10.0,
    window_interval_s: float = 10.0,
    tpot_slo_ms: float = 50.0,
    quick: bool = False,
    checkpoint: bool = False,
):
    """Run long-serving benchmark with one baseline × one trace.

    Args:
        model_path: Path to GGUF model
        trace: Trace name
        baseline: Baseline strategy name
        duration_min: Experiment duration in minutes
        window_interval_s: Control window interval
        tpot_slo_ms: TPOT SLO threshold
        quick: Quick mode (5min, short_chat only)
        checkpoint: Save intermediate results
    """
    output_dir = Path('data/serving_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(model_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if quick:
        duration_min = 5.0
        trace = 'short_chat'
        duration_s = duration_min * 60
    else:
        duration_s = duration_min * 60

    logger.info(f"{'='*60}")
    logger.info(f"Serving Benchmark: {baseline} | {trace} | {duration_min:.0f}min")
    logger.info(f"Model: {model_name} | SLO: {tpot_slo_ms}ms")
    logger.info(f"{'='*60}")

    # Generate request sequence
    req_gen = RequestGenerator(trace, duration_s)
    requests = req_gen.generate_requests()
    logger.info(f"Generated {len(requests)} requests for {duration_s:.0f}s")

    # Load model
    logger.info("Loading model...")
    runner = LlamaCppRunner(
        model_path=model_path,
        n_gpu_layers=-1, n_ctx=4096, n_threads=4,
    )
    logger.info("Model loaded, warming up...")
    runner.warmup(2)

    # Initialize controllers
    mc = MetricsCollector(interval_ms=500, output_dir='data/raw_logs')
    cc = CapController()

    # Load rate tables for selectors
    cap_files = sorted(Path('data/rate_tables').glob('cap_rate_table_with_savings_*.parquet'))
    lock_files = sorted(Path('data/rate_tables').glob('lock_rate_table_*.parquet'))
    cap_rt = str(cap_files[-1]) if cap_files else ''
    lock_rt = str(lock_files[-1]) if lock_files else ''

    cap_selector = None
    if cap_rt:
        try:
            cap_selector = WorkloadCapSelector(
                cap_rate_table_path=cap_rt,
                lock_rate_table_path=lock_rt,
                model=model_name,
            )
        except Exception as e:
            logger.warning(f"WorkloadCapSelector init failed: {e}")

    thermal_config = ControllerConfig(
        tpot_slo_ms=tpot_slo_ms,
        gpu_caps=[408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300],
    )
    thermal_controller = None
    if cap_rt:
        try:
            thermal_controller = ThermalSLOCapController(
                cap_rate_table_path=cap_rt,
                lock_rate_table_path=lock_rt,
                model=model_name,
                cap_controller=cc,
                config=thermal_config,
            )
        except Exception as e:
            logger.warning(f"ThermalSLOCapController init failed: {e}")

    # Setup baseline
    ctx = setup_baseline(
        baseline, cc, cap_selector, thermal_controller, tpot_slo_ms,
    )
    logger.info(f"Baseline '{baseline}' initialized, cap={ctx['current_cap_mhz']}MHz")

    # ── Run serving loop ──
    window_metrics_list: List[WindowMetrics] = []
    current_window: WindowMetrics = WindowMetrics(
        baseline=baseline, trace=trace, model=model_name,
    )
    request_results: List[Dict] = []

    window_id = 0
    window_start_time = time.time()
    prev_temp_c = None
    req_idx = 0

    start_time = time.time()
    logger.info(f"\nStarting serving loop at {datetime.now().strftime('%H:%M:%S')}...")

    while req_idx < len(requests) and (time.time() - start_time) < duration_s:
        req = requests[req_idx]
        pl = req['prompt_length']
        ol = req['output_length']

        # ── Per-request cap selection for Pareto baseline ──
        if baseline == 'Pareto' and cap_selector:
            try:
                sel = cap_selector.select(
                    prompt_length=pl, output_length=ol,
                    strategy='pareto',
                )
                new_cap = sel.get('gpu_cap_mhz', ctx['current_cap_mhz'])
                if new_cap != ctx['current_cap_mhz']:
                    cc.set_cap(new_cap, 3199, 1036)
                    ctx['current_cap_mhz'] = new_cap
            except Exception:
                pass

        # ── Per-request initial cap for ThermalSLO (first request only) ──
        elif baseline == 'ThermalSLO' and thermal_controller and window_id == 0 and req_idx == 0:
            try:
                init = thermal_controller.select_initial_cap(pl, ol)
                ctx['current_cap_mhz'] = init['gpu_cap_mhz']
            except Exception:
                pass

        # ── Run inference ──
        try:
            result = run_single_inference_with_metrics(
                runner, mc, cc, pl, ol, collect_metrics=True,
            )
            result['baseline'] = baseline
            result['trace'] = trace
            result['workload_name'] = req.get('workload_name', '')
            result['is_burst'] = req.get('is_burst', False)
            result['window_id'] = window_id
            result['request_id'] = req_idx
            request_results.append(result)
        except Exception as e:
            logger.error(f"Request {req_idx} failed: {e}")
            request_results.append({
                'error': str(e), 'baseline': baseline, 'trace': trace,
                'window_id': window_id, 'request_id': req_idx,
            })

        # ── Accumulate window metrics ──
        if not current_window.n_requests:
            current_window.temp_start_c = result.get('temperature_c', 0)

        current_window.n_requests += 1
        current_window.total_tokens += result.get('output_tokens', 0)
        current_window.total_energy_j += result.get('total_energy_j', 0)

        # Track TPOTs
        tpot = result.get('tpot_ms', 0)
        if not hasattr(current_window, '_tpots'):
            current_window._tpots = []
        current_window._tpots.append(tpot)

        # Track SLO violations
        if tpot > tpot_slo_ms:
            current_window.slo_violation_count += 1

        # Track power
        current_window.max_power_w = max(
            current_window.max_power_w,
            result.get('max_power_w', 0),
        )

        req_idx += 1

        # ── Window boundary check ──
        elapsed = time.time() - window_start_time
        if elapsed >= window_interval_s or req_idx >= len(requests):
            # Finalize window
            current_window.window_id = window_id
            current_window.timestamp = time.time()
            current_window.cap_mhz = ctx['current_cap_mhz']
            current_window.slo_violation_rate = (
                current_window.slo_violation_count / max(current_window.n_requests, 1)
            )

            # Compute percentiles
            if hasattr(current_window, '_tpots') and current_window._tpots:
                tpots = current_window._tpots
                current_window.tpot_p50_ms = np.percentile(tpots, 50)
                current_window.tpot_p95_ms = np.percentile(tpots, 95)
                current_window.tpot_p99_ms = np.percentile(tpots, 99)

            # Compute power (average from energy)
            if current_window.total_energy_j > 0 and elapsed > 0:
                current_window.avg_power_w = current_window.total_energy_j / elapsed

            # Energy efficiency
            if current_window.total_energy_j > 0:
                current_window.tokens_per_joule = (
                    current_window.total_tokens / current_window.total_energy_j
                )

            # Temperature tracking
            current_window.temp_end_c = result.get('temperature_c', 0)
            current_window.temp_mean_c = (
                (current_window.temp_start_c + current_window.temp_end_c) / 2
            )
            current_window.temp_max_c = max(
                current_window.temp_start_c, current_window.temp_end_c
            )
            if prev_temp_c is not None:
                current_window.temp_slope_c_per_s = (
                    (current_window.temp_end_c - prev_temp_c) / elapsed
                )
            prev_temp_c = current_window.temp_end_c

            # TPS
            if elapsed > 0 and current_window.total_tokens > 0:
                current_window.tokens_per_second = current_window.total_tokens / elapsed

            # ── ThermalSLO controller update ──
            if baseline == 'ThermalSLO' and thermal_controller:
                try:
                    state = thermal_controller.update_cap(
                        current_temperature_c=current_window.temp_end_c,
                        temperature_slope_c_per_s=current_window.temp_slope_c_per_s,
                        temperature_max_c=current_window.temp_max_c,
                        tpot_p50_ms=current_window.tpot_p50_ms,
                        tpot_p95_ms=current_window.tpot_p95_ms,
                        tpot_p99_ms=current_window.tpot_p99_ms,
                        tokens_per_second=current_window.tokens_per_second,
                        slo_violation_count=current_window.slo_violation_count,
                        total_requests=current_window.n_requests,
                        avg_power_w=current_window.avg_power_w,
                        max_power_w=current_window.max_power_w,
                        prompt_length=pl,
                        output_length=ol,
                    )
                    ctx['current_cap_mhz'] = state.current_cap_mhz
                    current_window.cap_action = state.cap_action.value
                    current_window.cap_switch_count = state.cap_switch_count
                except Exception as e:
                    logger.error(f"ThermalSLO update failed: {e}")

            window_metrics_list.append(current_window)

            # Log window summary
            logger.info(
                f"[W{window_id:3d}] {baseline:12s} cap={current_window.cap_mhz:4d}MHz "
                f"act={current_window.cap_action:8s} "
                f"n_req={current_window.n_requests:2d} "
                f"TPOT_p50={current_window.tpot_p50_ms:5.1f}ms "
                f"pwr={current_window.avg_power_w:4.1f}W "
                f"temp={current_window.temp_end_c:4.1f}°C "
                f"tok/J={current_window.tokens_per_joule:5.1f}"
            )

            # Start new window
            window_id += 1
            current_window = WindowMetrics(
                baseline=baseline, trace=trace, model=model_name,
            )
            window_start_time = time.time()

        # Brief inter-request gap
        time.sleep(0.5)

    elapsed_total = time.time() - start_time
    logger.info(f"\nServing loop complete: {elapsed_total:.0f}s, "
                f"{len(request_results)} requests, {window_id} windows")

    # ── Save results ──
    # Window-level aggregation
    if window_metrics_list:
        wm_df = pd.DataFrame([asdict(w) for w in window_metrics_list])
        # Remove internal _tpots field
        if '_tpots' in wm_df.columns:
            wm_df = wm_df.drop(columns=['_tpots'])
        wm_path = output_dir / f'serving_windows_{baseline}_{trace}_{model_name}_{timestamp}.csv'
        wm_df.to_csv(wm_path, index=False)
        logger.info(f"Saved window metrics: {wm_path} ({len(wm_df)} rows)")

    # Request-level detail
    if request_results:
        req_df = pd.DataFrame(request_results)
        req_path = output_dir / f'serving_requests_{baseline}_{trace}_{model_name}_{timestamp}.csv'
        req_df.to_csv(req_path, index=False)
        logger.info(f"Saved request detail: {req_path} ({len(req_df)} rows)")

    # Cleanup
    try:
        cc.restore_dynamic()
    except Exception:
        pass

    return {
        'baseline': baseline,
        'trace': trace,
        'model': model_name,
        'duration_s': elapsed_total,
        'n_requests': len(request_results),
        'n_windows': window_id,
        'window_csv': str(wm_path) if window_metrics_list else None,
        'request_csv': str(req_path) if request_results else None,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Long-running Serving Benchmark (Phase 13, P3)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick 5-min test with ThermalSLO
  sudo python3 src/experiments/run_serving_benchmark.py \\
    --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf --quick

  # Full 30-min with specific baseline
  sudo python3 src/experiments/run_serving_benchmark.py \\
    --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
    --duration-min 30 --baseline ThermalSLO --trace long_generation

  # Run all baselines × all traces (overnight)
  sudo python3 src/experiments/run_serving_benchmark.py \\
    --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf \\
    --duration-min 30 --all
        """)
    parser.add_argument('--model', type=str, required=True,
                        help='Path to GGUF model file')
    parser.add_argument('--trace', type=str, default='short_chat',
                        choices=list(TRACE_DISTRIBUTIONS.keys()),
                        help='Workload trace')
    parser.add_argument('--baseline', type=str, default='ThermalSLO',
                        choices=['MAXN', 'Dynamic', 'BestStatic', 'Pareto', 'ThermalSLO'],
                        help='Baseline strategy')
    parser.add_argument('--duration-min', type=float, default=10.0,
                        help='Experiment duration in minutes')
    parser.add_argument('--tpot-slo-ms', type=float, default=50.0,
                        help='TPOT SLO threshold in ms')
    parser.add_argument('--window-interval-s', type=float, default=10.0,
                        help='Control window interval in seconds')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 5min short_chat')
    parser.add_argument('--all', action='store_true',
                        help='Run all baselines × all traces (sequential)')
    args = parser.parse_args()

    if args.all:
        baselines = ['MAXN', 'Dynamic', 'BestStatic', 'Pareto', 'ThermalSLO']
        traces = list(TRACE_DISTRIBUTIONS.keys())
        all_results = []
        for bl in baselines:
            for tr in traces:
                logger.info(f"\n{'#'*60}")
                logger.info(f"# Running: {bl} × {tr}")
                logger.info(f"{'#'*60}")
                try:
                    result = run_serving_benchmark(
                        model_path=args.model,
                        trace=tr,
                        baseline=bl,
                        duration_min=args.duration_min,
                        window_interval_s=args.window_interval_s,
                        tpot_slo_ms=args.tpot_slo_ms,
                    )
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed {bl}×{tr}: {e}")
                # Cooldown between experiments
                time.sleep(30)

        # Save combined summary
        if all_results:
            summary_df = pd.DataFrame(all_results)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_path = Path('data/serving_benchmark') / f'serving_summary_{ts}.csv'
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"\nAll experiments complete. Summary: {summary_path}")
    else:
        run_serving_benchmark(
            model_path=args.model,
            trace=args.trace,
            baseline=args.baseline,
            duration_min=args.duration_min,
            window_interval_s=args.window_interval_s,
            tpot_slo_ms=args.tpot_slo_ms,
            quick=args.quick,
        )
