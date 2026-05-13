#!/usr/bin/env python3
"""
Baseline Comparison Experiment Orchestrator
Runs llama.cpp inference under different Jetson power configurations,
collects metrics, and produces comparison reports.

Experiment matrix: N workloads × M baselines × R repeats
"""

import csv
import json
import time
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from jetson_power_modes import (
    record_current_power_state,
    restore_power_state,
    apply_baseline_config,
    get_current_nvpmodel,
    _get_frequency_sysfs
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineComparison:
    """Orchestrate baseline comparison experiments on Jetson."""

    def __init__(self, baselines_path: str = 'configs/baselines.yaml',
                 workloads_path: str = 'configs/real_model_workloads.yaml',
                 output_dir: str = 'data/real_baseline_comparison'):
        self.baselines_path = Path(baselines_path)
        self.workloads_path = Path(workloads_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load configs
        with open(self.baselines_path) as f:
            self.baselines_config = yaml.safe_load(f)
        with open(self.workloads_path) as f:
            self.workloads_config = yaml.safe_load(f)

        self.baselines = self.baselines_config.get('baselines', {})
        self.workloads = self.workloads_config.get('workloads', {})
        self.model_config = self.workloads_config.get('model', {})
        self.exp_config = self.workloads_config.get('experiment', {})

        self.cooldown = self.baselines_config.get('cooldown_seconds', 30)
        self.warmup_runs = self.exp_config.get('warmup_runs', 2)
        self.repeats = self.exp_config.get('repeats', 5)
        self.timeout = self.exp_config.get('timeout_seconds', 300)

        self.initial_state = None
        self.runner = None
        self.results: List[Dict] = []

    def _init_runner(self):
        """Initialize llama.cpp runner."""
        from llama_cpp_runner import LlamaCppRunner
        self.runner = LlamaCppRunner(
            model_path=self.model_config['path'],
            n_gpu_layers=self.model_config.get('n_gpu_layers', -1),
            n_ctx=self.model_config.get('n_ctx', 4096),
            n_threads=self.model_config.get('n_threads', 4)
        )

    def run(self) -> Dict:
        """Run the complete baseline comparison experiment."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Starting baseline comparison: {timestamp}")
        logger.info(f"Baselines: {list(self.baselines.keys())}")
        logger.info(f"Workloads: {list(self.workloads.keys())}")
        logger.info(f"Matrix: {len(self.baselines)} × {len(self.workloads)} × {self.repeats}")

        # Record initial state
        self.initial_state = record_current_power_state()

        try:
            self._init_runner()

            for bl_name, bl_config in self.baselines.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Baseline: {bl_name}")
                logger.info(f"{'='*60}")

                # Apply baseline config
                apply_baseline_config(bl_config)
                time.sleep(3)  # Let frequencies settle

                # Record actual system state
                actual_state = {
                    'nvpmodel': get_current_nvpmodel(),
                    'gpu_freq_mhz': _get_frequency_sysfs('gpu'),
                    'cpu_freq_mhz': _get_frequency_sysfs('cpu'),
                    'emc_freq_mhz': _get_frequency_sysfs('emc')
                }

                # Warmup
                self.runner.warmup(self.warmup_runs)

                # Run workloads
                for wl_name, wl_config in self.workloads.items():
                    logger.info(f"  Workload: {wl_name}")
                    for rep in range(self.repeats):
                        result = self._run_single(
                            bl_name, bl_config, wl_name, wl_config,
                            rep, actual_state
                        )
                        self.results.append(result)

                # Cooldown between baselines
                logger.info(f"  Cooldown {self.cooldown}s...")
                time.sleep(self.cooldown)

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            # Always restore initial state
            if self.initial_state:
                restore_power_state(self.initial_state)
                logger.info("Restored initial power state")

        # Save results
        output_files = self._save_results(timestamp)

        logger.info(f"\nExperiment complete! Results saved to: {self.output_dir}")
        return output_files

    def _run_single(self, bl_name: str, bl_config: Dict,
                    wl_name: str, wl_config: Dict,
                    repeat: int, actual_state: Dict) -> Dict:
        """Run a single inference and record all metrics."""
        prompt_len = wl_config.get('prompt_length', 512)
        output_len = wl_config.get('output_length', 128)

        start_ts = time.time()
        inference_result = self.runner.run_single_inference(
            prompt_length=prompt_len,
            output_length=output_len
        )
        end_ts = time.time()

        record = {
            # Identification
            'timestamp': datetime.now().isoformat(),
            'baseline_name': bl_name,
            'baseline_type': bl_config.get('type', 'unknown'),
            'workload_name': wl_name,
            'repeat': repeat,

            # Workload specs
            'prompt_length': prompt_len,
            'output_length': output_len,
            'batch_size': wl_config.get('batch_size', 1),
            'model': self.model_config.get('name', ''),

            # System state during run
            'nvpmodel_actual': actual_state.get('nvpmodel', ''),
            'gpu_freq_mhz_actual': actual_state.get('gpu_freq_mhz', 0),
            'cpu_freq_mhz_actual': actual_state.get('cpu_freq_mhz', 0),
            'emc_freq_mhz_actual': actual_state.get('emc_freq_mhz', 0),

            # Performance metrics
            'ttft_ms': inference_result.get('ttft_ms', 0),
            'tpot_ms': inference_result.get('tpot_ms', 0),
            'total_time_ms': inference_result.get('total_time_ms', 0),
            'output_tokens': inference_result.get('output_tokens', 0),
            'tokens_per_second': inference_result.get('tokens_per_second', 0),

            # Energy metrics (filled by tegrastats integration)
            'total_energy_j': inference_result.get('total_energy_j', 0),
            'avg_power_w': inference_result.get('avg_power_w', 0),
            'max_power_w': inference_result.get('max_power_w', 0),
            'temperature_c': inference_result.get('temperature_c', 0),

            # Derived
            'energy_per_token_j': (
                inference_result.get('total_energy_j', 0) /
                max(inference_result.get('output_tokens', 1), 1)
            ),
            'wall_time_s': end_ts - start_ts,
            'error': inference_result.get('error', '')
        }

        logger.info(f"    [{repeat}] TTFT={record['ttft_ms']:.1f}ms, "
                    f"TPOT={record['tpot_ms']:.1f}ms, "
                    f"E/tok={record['energy_per_token_j']:.4f}J")

        return record

    def _save_results(self, timestamp: str) -> Dict:
        """Save all results to output files."""
        df = pd.DataFrame(self.results)

        # Detailed CSV
        detailed_path = self.output_dir / f'detailed_results_{timestamp}.csv'
        df.to_csv(detailed_path, index=False)

        # Summary CSV (per baseline-workload aggregation)
        summary = df.groupby(['baseline_name', 'workload_name']).agg({
            'ttft_ms': ['mean', 'std', 'median'],
            'tpot_ms': ['mean', 'std', 'median'],
            'total_energy_j': ['mean', 'std'],
            'energy_per_token_j': ['mean', 'std', 'median'],
            'tokens_per_second': ['mean', 'std'],
            'avg_power_w': ['mean', 'max'],
            'temperature_c': ['mean', 'max'],
            'total_time_ms': ['mean', 'std']
        }).round(4)
        summary_path = self.output_dir / f'comparison_summary_{timestamp}.csv'
        summary.to_csv(summary_path)

        # Workload breakdown (per workload, all baselines)
        breakdown_path = self.output_dir / f'workload_breakdown_{timestamp}.csv'
        workload_breakdown = []
        for wl_name in df['workload_name'].unique():
            wl_data = df[df['workload_name'] == wl_name]
            for bl_name in wl_data['baseline_name'].unique():
                bl_data = wl_data[wl_data['baseline_name'] == bl_name]
                workload_breakdown.append({
                    'workload': wl_name,
                    'baseline': bl_name,
                    'prompt_length': bl_data['prompt_length'].iloc[0],
                    'output_length': bl_data['output_length'].iloc[0],
                    'ttft_ms_mean': bl_data['ttft_ms'].mean(),
                    'ttft_ms_std': bl_data['ttft_ms'].std(),
                    'tpot_ms_mean': bl_data['tpot_ms'].mean(),
                    'tpot_ms_std': bl_data['tpot_ms'].std(),
                    'energy_per_token_j_mean': bl_data['energy_per_token_j'].mean(),
                    'energy_per_token_j_std': bl_data['energy_per_token_j'].std(),
                    'tokens_per_second_mean': bl_data['tokens_per_second'].mean(),
                    'avg_power_w_mean': bl_data['avg_power_w'].mean(),
                    'temperature_c_mean': bl_data['temperature_c'].mean(),
                    'n_runs': len(bl_data),
                    'error_count': (bl_data['error'] != '').sum()
                })
        pd.DataFrame(workload_breakdown).to_csv(breakdown_path, index=False)

        # Report
        report_path = self.output_dir / f'baseline_report_{timestamp}.md'
        report = self._generate_report(df, timestamp)
        with open(report_path, 'w') as f:
            f.write(report)

        return {
            'detailed': str(detailed_path),
            'summary': str(summary_path),
            'breakdown': str(breakdown_path),
            'report': str(report_path)
        }

    def _generate_report(self, df: pd.DataFrame, timestamp: str) -> str:
        """Generate comparison report."""
        lines = []
        lines.append("# Baseline Comparison Report")
        lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Model**: {self.model_config.get('name', 'unknown')}")
        lines.append(f"**Baselines**: {len(self.baselines)}")
        lines.append(f"**Workloads**: {len(self.workloads)}")
        lines.append(f"**Repeats**: {self.repeats}")

        # Per-baseline summary
        lines.append("\n## Baseline Performance Summary")
        lines.append("\n| Baseline | Type | Avg TTFT (ms) | Avg TPOT (ms) | "
                    "Avg Energy/Token (J) | Avg TPS | Avg Power (W) |")
        lines.append("|----------|------|---------------|---------------|"
                    "---------------------|---------|---------------|")

        for bl_name in df['baseline_name'].unique():
            bl_data = df[df['baseline_name'] == bl_name]
            bl_type = bl_data['baseline_type'].iloc[0]
            lines.append(
                f"| {bl_name} | {bl_type} | "
                f"{bl_data['ttft_ms'].mean():.1f} | "
                f"{bl_data['tpot_ms'].mean():.1f} | "
                f"{bl_data['energy_per_token_j'].mean():.4f} | "
                f"{bl_data['tokens_per_second'].mean():.1f} | "
                f"{bl_data['avg_power_w'].mean():.1f} |"
            )

        # Find best baselines
        lines.append("\n## Key Comparisons")
        baseline_avgs = df.groupby('baseline_name').agg({
            'energy_per_token_j': 'mean',
            'ttft_ms': 'mean',
            'tpot_ms': 'mean',
            'tokens_per_second': 'mean'
        }).to_dict('index')

        if baseline_avgs:
            best_energy = min(baseline_avgs.items(),
                            key=lambda x: x[1]['energy_per_token_j'])
            best_ttft = min(baseline_avgs.items(),
                          key=lambda x: x[1]['ttft_ms'])
            best_tps = max(baseline_avgs.items(),
                         key=lambda x: x[1]['tokens_per_second'])

            lines.append(f"\n**Best Energy Efficiency**: {best_energy[0]} "
                        f"({best_energy[1]['energy_per_token_j']:.4f} J/token)")
            lines.append(f"\n**Best TTFT**: {best_ttft[0]} "
                        f"({best_ttft[1]['ttft_ms']:.1f} ms)")
            lines.append(f"\n**Best Throughput**: {best_tps[0]} "
                        f"({best_tps[1]['tokens_per_second']:.1f} tok/s)")

        # Errors
        errors = df[df['error'] != '']
        if not errors.empty:
            lines.append(f"\n## Errors ({len(errors)} runs)")
            for _, row in errors.iterrows():
                lines.append(f"- {row['baseline_name']}/{row['workload_name']} "
                            f"rep={row['repeat']}: {row['error']}")

        return '\n'.join(lines)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Run Baseline Comparison Experiment')
    parser.add_argument('--baselines', default='configs/baselines.yaml')
    parser.add_argument('--workloads', default='configs/real_model_workloads.yaml')
    parser.add_argument('--output-dir', default='data/real_baseline_comparison')
    args = parser.parse_args()

    experiment = BaselineComparison(
        baselines_path=args.baselines,
        workloads_path=args.workloads,
        output_dir=args.output_dir
    )

    results = experiment.run()
    print("\nOutput files:")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    import sys
    sys.exit(main())
