#!/usr/bin/env python3
"""
Phase-Aware DVFS Online Controller
Orchestrates phase-aware frequency switching during LLM inference,
tracking real-time metrics and energy savings vs baselines.
"""

import time
import json
import logging
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
from collections import defaultdict

from synthetic_benchmark import SyntheticBenchmark
from phase_aware_policy import PhaseAwarePolicy
from select_config import ConfigSelector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhaseController:
    """
    Online controller that applies Phase-Aware DVFS during inference.

    Supports multiple scheduling strategies:
    - default: mid-frequency for all phases
    - max_perf: max frequency for all phases
    - energy_efficient: lowest energy config for all phases
    - phase_aware: different configs per phase (our approach)
    - oracle: offline-optimal per-phase config
    """

    STRATEGIES = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'adaptive_phase_aware', 'oracle']

    def __init__(self, config: Dict, strategy: str = 'phase_aware'):
        self.config = config
        self.strategy = strategy
        self.benchmark = SyntheticBenchmark(config)
        self.policy = PhaseAwarePolicy(
            selector_table_path=config.get('selector_table_path',
                                           'data/rate_tables/selector_table.parquet'),
            config_path=config.get('selector_config_path', 'configs/selector.yaml')
        )
        self.selector = ConfigSelector(
            config.get('selector_table_path', 'data/rate_tables/selector_table.parquet')
        )

        self.slo = config.get('slo', {
            'ttft_ms': 1000, 'tpot_ms': 80,
            'max_power_w': 40, 'max_temp_c': 80
        })

        # Frequency presets from config
        self.freq_presets = config.get('frequencies', {
            'gpu': {'low': 378, 'mid': 846, 'high': 1428},
            'cpu': {'low': 1020, 'mid': 1479, 'high': 2015},
            'emc': {'low': 133, 'mid': 1600, 'high': 2133}
        })

        # Switching overhead estimates (ms)
        self.switching_overhead = config.get('switching_overhead_ms', {
            'gpu': 50, 'cpu': 20, 'emc': 80
        })

        # Track history
        self.history: List[Dict] = []
        self.cumulative_energy_j = 0.0
        self.cumulative_tokens = 0
        self.slo_violations = defaultdict(int)

    def _get_strategy_config(self, workload: Dict, phase: str) -> Dict:
        """Get frequency config for a given strategy and phase."""
        f = self.freq_presets

        if self.strategy == 'default':
            return {'gpu_freq': f['gpu']['mid'], 'cpu_freq': f['cpu']['mid'],
                    'emc_freq': f['emc']['mid']}
        elif self.strategy == 'max_perf':
            return {'gpu_freq': f['gpu']['high'], 'cpu_freq': f['cpu']['high'],
                    'emc_freq': f['emc']['high']}
        elif self.strategy == 'energy_efficient':
            return {'gpu_freq': f['gpu']['low'], 'cpu_freq': f['cpu']['mid'],
                    'emc_freq': f['emc']['mid']}
        elif self.strategy == 'phase_aware':
            return self._get_phase_aware_config(workload, phase)
        elif self.strategy == 'oracle':
            return self._get_oracle_config(workload, phase)
        else:
            return {'gpu_freq': f['gpu']['mid'], 'cpu_freq': f['cpu']['mid'],
                    'emc_freq': f['emc']['mid']}

    def _get_phase_aware_config(self, workload: Dict, phase: str) -> Dict:
        """Get phase-aware config: high GPU for prefill, high EMC for decode."""
        f = self.freq_presets
        if phase == 'prefill':
            return {'gpu_freq': f['gpu']['high'], 'cpu_freq': f['cpu']['high'],
                    'emc_freq': f['emc']['mid']}
        elif phase == 'decode':
            return {'gpu_freq': f['gpu']['mid'], 'cpu_freq': f['cpu']['mid'],
                    'emc_freq': f['emc']['high']}
        else:
            return {'gpu_freq': f['gpu']['mid'], 'cpu_freq': f['cpu']['mid'],
                    'emc_freq': f['emc']['mid']}

    def _get_oracle_config(self, workload: Dict, phase: str) -> Dict:
        """Get oracle config from selector table (offline-optimal)."""
        wl = {
            'model': 'synthetic_qwen_7b_int4', 'runtime': 'synthetic',
            'batch_size': workload.get('batch_size', 1),
            'prompt_length': workload.get('prompt_len', 512),
            'output_length': workload.get('output_len', 128),
            'phase': phase, 'concurrency': 1
        }
        result = self.selector.select_config(wl, self.slo)
        cfg = result.get('selected_config', {})
        return {
            'gpu_freq': cfg.get('gpu_freq_mhz', self.freq_presets['gpu']['mid']),
            'cpu_freq': cfg.get('cpu_freq_mhz', self.freq_presets['cpu']['mid']),
            'emc_freq': cfg.get('emc_freq_mhz', self.freq_presets['emc']['mid'])
        }

    def _estimate_switching_cost(self, from_cfg: Dict, to_cfg: Dict) -> Dict:
        """Estimate time and energy cost of frequency switching."""
        overhead_ms = 0.0
        components_changed = []
        for comp in ['gpu', 'cpu', 'emc']:
            key = f'{comp}_freq'
            if from_cfg.get(key) != to_cfg.get(key):
                overhead_ms += self.switching_overhead.get(comp, 50)
                components_changed.append(comp)
        return {
            'overhead_ms': overhead_ms,
            'components_changed': components_changed,
            'energy_overhead_j': overhead_ms * 0.015  # ~15W avg during switch
        }

    def run_single_inference(self, workload: Dict) -> Dict:
        """
        Run a single inference with the configured strategy.

        The workload dict should have: prompt_len, output_len, batch_size
        """
        prompt_len = workload.get('prompt_len', 512)
        output_len = workload.get('output_len', 128)
        batch_size = workload.get('batch_size', 1)

        logger.info(f"[{self.strategy}] Running inference: prompt={prompt_len}, "
                     f"output={output_len}, batch={batch_size}")

        result = {
            'strategy': self.strategy,
            'workload': workload.copy(),
            'phases': {},
            'switching_cost': {},
            'total_energy_j': 0.0,
            'total_time_ms': 0.0,
            'slo_met': True,
            'slo_violations': []
        }

        if self.strategy in ('phase_aware', 'adaptive_phase_aware'):
            # Check adaptive policy decision
            policy_workload = {
                'model': 'synthetic_qwen_7b_int4',
                'runtime': 'synthetic',
                'batch_size': batch_size,
                'prompt_length': prompt_len,
                'output_length': output_len,
                'concurrency': 1
            }
            adaptive_decision = self.policy.should_enable_phase_aware(policy_workload, self.slo)

            if self.strategy == 'adaptive_phase_aware':
                switch_enabled = adaptive_decision.get('switch_enabled', False)
                result['adaptive_decision'] = adaptive_decision
                result['switch_enabled'] = switch_enabled

                if switch_enabled:
                    result = self._run_phase_aware(workload)
                    result['switch_enabled'] = True
                    result['selection_reason'] = adaptive_decision.get('selection_reason', '')
                    result['switching_overhead_ms'] = adaptive_decision.get('switching_cost', {}).get('overhead_ms', 0)
                    result['switching_energy_j'] = adaptive_decision.get('switching_cost', {}).get('energy_overhead_j', 0)
                    result['expected_energy_saving_j'] = adaptive_decision.get('energy_saving', 0)
                    result['break_even_tokens'] = adaptive_decision.get('break_even_tokens', 0)
                else:
                    # Fall back to single config from policy
                    single_cfg = adaptive_decision.get('single_config', {})
                    freq = {
                        'gpu_freq': single_cfg.get('gpu_freq_mhz', self.freq_presets['gpu']['mid']),
                        'cpu_freq': single_cfg.get('cpu_freq_mhz', self.freq_presets['cpu']['mid']),
                        'emc_freq': single_cfg.get('emc_freq_mhz', self.freq_presets['emc']['mid'])
                    }
                    result.update(self._run_single_config_with_freq(workload, freq))
                    result['switch_enabled'] = False
                    result['selection_reason'] = adaptive_decision.get('selection_reason', '')
            else:
                # Original phase_aware: always switch
                result = self._run_phase_aware(workload)
                result['switch_enabled'] = True
                result['adaptive_decision'] = adaptive_decision
        else:
            result = self._run_single_config(workload)

        # Check SLO
        result['slo_met'], result['slo_violations'] = self._check_slo(result)

        # Update cumulative stats
        self.cumulative_energy_j += result['total_energy_j']
        self.cumulative_tokens += output_len
        for v in result['slo_violations']:
            self.slo_violations[v] += 1
        self.history.append(result)

        return result

    def _run_single_config(self, workload: Dict) -> Dict:
        """Run inference with a single fixed config (for non-phase-aware strategies)."""
        freq = self._get_strategy_config(workload, 'mixed')
        return self._run_single_config_with_freq(workload, freq)

    def _run_single_config_with_freq(self, workload: Dict, freq: Dict) -> Dict:
        """Run inference with an explicit frequency config."""
        prompt_len = workload.get('prompt_len', 512)
        output_len = workload.get('output_len', 128)
        batch_size = workload.get('batch_size', 1)

        metrics = self.benchmark.run_benchmark(
            {'prompt_len': prompt_len, 'output_len': output_len,
             'batch_size': batch_size, 'phase': 'mixed'},
            freq
        )

        return {
            'strategy': self.strategy,
            'workload': workload.copy(),
            'config': freq,
            'ttft_ms': metrics.get('ttft_ms', 0),
            'tpot_ms': metrics.get('tpot_ms', 0),
            'total_time_ms': metrics.get('total_time_ms', 0),
            'total_energy_j': metrics.get('total_energy_j', metrics.get('energy_j', 0)),
            'avg_power_w': metrics.get('avg_power_w', 0),
            'max_power_w': metrics.get('max_power_w', 0),
            'temperature_c': metrics.get('temperature_c', 0),
            'tokens_per_second': metrics.get('tokens_per_second', 0),
            'slo_met': True,
            'slo_violations': []
        }

    def _run_phase_aware(self, workload: Dict) -> Dict:
        """Run inference with phase-aware DVFS: separate configs for prefill and decode."""
        prompt_len = workload.get('prompt_len', 512)
        output_len = workload.get('output_len', 128)
        batch_size = workload.get('batch_size', 1)

        # Phase 1: Prefill
        prefill_freq = self._get_strategy_config(workload, 'prefill')
        prefill_metrics = self.benchmark.simulate_prefill(prompt_len)
        # Apply frequency effect
        freq_factor = prefill_freq['gpu_freq'] / 846
        prefill_metrics['prefill_time_ms'] /= freq_factor
        prefill_metrics['avg_power_w'] *= (0.8 + 0.2 * freq_factor)
        prefill_metrics['energy_j'] = (prefill_metrics['avg_power_w'] *
                                        prefill_metrics['prefill_time_ms']) / 1000

        # Switching cost from prefill to decode
        decode_freq = self._get_strategy_config(workload, 'decode')
        switch_cost = self._estimate_switching_cost(prefill_freq, decode_freq)

        # Phase 2: Decode
        decode_metrics = self.benchmark.simulate_decode(output_len, prompt_len)
        emc_factor = decode_freq['emc_freq'] / 1600
        gpu_factor = decode_freq['gpu_freq'] / 846
        decode_metrics['tpot_ms'] /= (0.6 * gpu_factor + 0.4 * emc_factor)
        decode_metrics['total_time_ms'] = decode_metrics['tpot_ms'] * output_len
        decode_metrics['avg_power_w'] *= (0.7 + 0.15 * gpu_factor + 0.15 * emc_factor)
        decode_metrics['energy_j'] = (decode_metrics['avg_power_w'] *
                                       decode_metrics['total_time_ms']) / 1000

        total_energy = (prefill_metrics['energy_j'] + decode_metrics['energy_j'] +
                        switch_cost['energy_overhead_j'])
        total_time = (prefill_metrics['prefill_time_ms'] + decode_metrics['total_time_ms'] +
                      switch_cost['overhead_ms'])

        return {
            'strategy': self.strategy,
            'workload': workload.copy(),
            'prefill_config': prefill_freq,
            'decode_config': decode_freq,
            'config': prefill_freq,  # For SLO check compatibility
            'switching_cost': switch_cost,
            'ttft_ms': prefill_metrics['prefill_time_ms'] + decode_metrics.get('ttft_ms', 0),
            'tpot_ms': decode_metrics['tpot_ms'],
            'total_time_ms': total_time,
            'total_energy_j': total_energy,
            'prefill_energy_j': prefill_metrics['energy_j'],
            'decode_energy_j': decode_metrics['energy_j'],
            'switch_energy_j': switch_cost['energy_overhead_j'],
            'avg_power_w': total_energy / (total_time / 1000) if total_time > 0 else 0,
            'max_power_w': max(prefill_metrics['max_power_w'], decode_metrics['max_power_w']),
            'temperature_c': max(prefill_metrics['temperature_c'], decode_metrics['temperature_c']),
            'tokens_per_second': output_len / (total_time / 1000) if total_time > 0 else 0,
            'slo_met': True,
            'slo_violations': []
        }

    def _check_slo(self, result: Dict) -> tuple:
        """Check if result meets SLO constraints."""
        violations = []
        slo = self.slo

        if result.get('ttft_ms', 0) > slo.get('ttft_ms', 1000):
            violations.append('ttft_ms')
        if result.get('tpot_ms', 0) > slo.get('tpot_ms', 80):
            violations.append('tpot_ms')
        if result.get('max_power_w', 0) > slo.get('max_power_w', 40):
            violations.append('max_power_w')
        if result.get('temperature_c', 0) > slo.get('max_temp_c', 80):
            violations.append('max_temp_c')

        return len(violations) == 0, violations

    def get_summary(self) -> Dict:
        """Get summary statistics from all runs."""
        if not self.history:
            return {}

        n = len(self.history)
        energies = [r['total_energy_j'] for r in self.history]
        times = [r['total_time_ms'] for r in self.history]
        ttfts = [r['ttft_ms'] for r in self.history]
        tpots = [r['tpot_ms'] for r in self.history]
        slo_met_count = sum(1 for r in self.history if r['slo_met'])

        return {
            'strategy': self.strategy,
            'total_runs': n,
            'slo_met_rate': slo_met_count / n,
            'slo_violation_breakdown': dict(self.slo_violations),
            'avg_energy_j': np.mean(energies),
            'std_energy_j': np.std(energies),
            'avg_time_ms': np.mean(times),
            'avg_ttft_ms': np.mean(ttfts),
            'avg_tpot_ms': np.mean(tpots),
            'total_energy_j': self.cumulative_energy_j,
            'total_tokens': self.cumulative_tokens,
            'energy_per_token_j': (self.cumulative_energy_j / self.cumulative_tokens
                                    if self.cumulative_tokens > 0 else 0),
            'avg_tokens_per_second': np.mean([r['tokens_per_second'] for r in self.history]),
        }


def run_comparison_experiment(config: Dict, workloads: List[Dict],
                              repeats: int = 10) -> Dict:
    """
    Run a full comparison experiment across all strategies.

    Returns a dict with per-strategy results and comparison metrics.
    """
    logger.info(f"Running comparison experiment: {len(workloads)} workloads × "
                f"{len(PhaseController.STRATEGIES)} strategies × {repeats} repeats")

    results = {}
    for strategy in PhaseController.STRATEGIES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Strategy: {strategy}")
        logger.info(f"{'='*60}")

        controller = PhaseController(config, strategy=strategy)
        strategy_results = []

        for wl in workloads:
            for rep in range(repeats):
                r = controller.run_single_inference(wl)
                r['repeat'] = rep
                strategy_results.append(r)

        results[strategy] = {
            'summary': controller.get_summary(),
            'detailed': strategy_results
        }

        s = results[strategy]['summary']
        logger.info(f"  Energy/token: {s['energy_per_token_j']:.4f} J/tok | "
                     f"TTFT: {s['avg_ttft_ms']:.1f}ms | TPOT: {s['avg_tpot_ms']:.1f}ms | "
                     f"SLO rate: {s['slo_met_rate']:.1%}")

    # Compute relative improvements
    comparison = _compute_comparison(results)
    return {'strategies': results, 'comparison': comparison}


def _compute_comparison(results: Dict) -> Dict:
    """Compute relative improvement of each strategy vs default baseline."""
    baseline_key = 'default'
    if baseline_key not in results:
        return {}

    baseline = results[baseline_key]['summary']
    comparison = {}

    for strategy, data in results.items():
        s = data['summary']
        comparison[strategy] = {
            'energy_per_token_j': s['energy_per_token_j'],
            'avg_ttft_ms': s['avg_ttft_ms'],
            'avg_tpot_ms': s['avg_tpot_ms'],
            'avg_time_ms': s['avg_time_ms'],
            'slo_met_rate': s['slo_met_rate'],
            'avg_tokens_per_second': s.get('avg_tokens_per_second', 0),
            'energy_reduction_pct': 0.0,
            'ttft_reduction_pct': 0.0,
            'tpot_reduction_pct': 0.0,
        }

        if baseline['energy_per_token_j'] > 0:
            comparison[strategy]['energy_reduction_pct'] = (
                1 - s['energy_per_token_j'] / baseline['energy_per_token_j']) * 100
        if baseline['avg_ttft_ms'] > 0:
            comparison[strategy]['ttft_reduction_pct'] = (
                1 - s['avg_ttft_ms'] / baseline['avg_ttft_ms']) * 100
        if baseline['avg_tpot_ms'] > 0:
            comparison[strategy]['tpot_reduction_pct'] = (
                1 - s['avg_tpot_ms'] / baseline['avg_tpot_ms']) * 100

    return comparison
