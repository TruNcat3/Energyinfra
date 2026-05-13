#!/usr/bin/env python3
"""
Phase-Aware DVFS Validation Experiment
Runs comprehensive comparison of scheduling strategies against baselines.
"""

import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from phase_controller import PhaseController, run_comparison_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


OUTPUT_DIR = Path(__file__).parent.parent / 'data' / 'phase_aware_experiment'


def get_config() -> dict:
    return {
        'selector_table_path': 'data/rate_tables/selector_table.parquet',
        'selector_config_path': 'configs/selector.yaml',
        'frequencies': {
            'gpu': {'low': 378, 'mid': 846, 'high': 1428},
            'cpu': {'low': 1020, 'mid': 1479, 'high': 2015},
            'emc': {'low': 133, 'mid': 1600, 'high': 2133},
        },
        'switching_overhead_ms': {'gpu': 50, 'cpu': 20, 'emc': 80},
        'slo': {
            'ttft_ms': 1000,
            'tpot_ms': 80,
            'max_power_w': 40,
            'max_temp_c': 80,
        },
    }


def get_workloads() -> list:
    """Define representative workloads covering different scenarios."""
    return [
        # Short prompt, short output
        {'prompt_len': 128, 'output_len': 64, 'batch_size': 1,
         'name': 'short_short'},
        # Short prompt, long output
        {'prompt_len': 128, 'output_len': 256, 'batch_size': 1,
         'name': 'short_long'},
        # Medium prompt, medium output
        {'prompt_len': 512, 'output_len': 128, 'batch_size': 1,
         'name': 'medium_medium'},
        # Long prompt, medium output
        {'prompt_len': 1024, 'output_len': 128, 'batch_size': 1,
         'name': 'long_medium'},
        # Long prompt, long output
        {'prompt_len': 1024, 'output_len': 512, 'batch_size': 1,
         'name': 'long_long'},
    ]


def save_results(results: dict, output_dir: Path):
    """Save experiment results to files."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save detailed results as CSV
    rows = []
    for strategy, data in results['strategies'].items():
        for r in data['detailed']:
            row = {
                'strategy': strategy,
                'workload_name': r['workload'].get('name', 'unknown'),
                'prompt_len': r['workload'].get('prompt_len', 0),
                'output_len': r['workload'].get('output_len', 0),
                'batch_size': r['workload'].get('batch_size', 1),
                'repeat': r.get('repeat', 0),
                'ttft_ms': r.get('ttft_ms', 0),
                'tpot_ms': r.get('tpot_ms', 0),
                'total_time_ms': r.get('total_time_ms', 0),
                'total_energy_j': r.get('total_energy_j', 0),
                'avg_power_w': r.get('avg_power_w', 0),
                'max_power_w': r.get('max_power_w', 0),
                'temperature_c': r.get('temperature_c', 0),
                'tokens_per_second': r.get('tokens_per_second', 0),
                'slo_met': r.get('slo_met', False),
                'slo_violations': ','.join(r.get('slo_violations', [])),
            }
            if 'switching_cost' in r:
                row['switch_overhead_ms'] = r['switching_cost'].get('overhead_ms', 0)
                row['switch_energy_j'] = r['switching_cost'].get('energy_overhead_j', 0)
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = output_dir / f'detailed_results_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed results saved: {csv_path}")

    # Save summary comparison as CSV
    comparison_rows = []
    for strategy, metrics in results['comparison'].items():
        comparison_rows.append({
            'strategy': strategy,
            **metrics
        })
    comp_df = pd.DataFrame(comparison_rows)
    comp_path = output_dir / f'comparison_summary_{timestamp}.csv'
    comp_df.to_csv(comp_path, index=False)
    logger.info(f"Comparison summary saved: {comp_path}")

    # Save per-workload breakdown
    workload_rows = []
    for strategy, data in results['strategies'].items():
        for r in data['detailed']:
            wl_name = r['workload'].get('name', 'unknown')
            workload_rows.append({
                'strategy': strategy,
                'workload': wl_name,
                'total_energy_j': r.get('total_energy_j', 0),
                'ttft_ms': r.get('ttft_ms', 0),
                'tpot_ms': r.get('tpot_ms', 0),
                'total_time_ms': r.get('total_time_ms', 0),
            })
    wl_df = pd.DataFrame(workload_rows)

    # Aggregate per workload × strategy
    wl_agg = wl_df.groupby(['strategy', 'workload']).agg(
        mean_energy_j=('total_energy_j', 'mean'),
        std_energy_j=('total_energy_j', 'std'),
        mean_ttft_ms=('ttft_ms', 'mean'),
        mean_tpot_ms=('tpot_ms', 'mean'),
        mean_time_ms=('total_time_ms', 'mean'),
    ).reset_index()
    wl_path = output_dir / f'workload_breakdown_{timestamp}.csv'
    wl_agg.to_csv(wl_path, index=False)
    logger.info(f"Workload breakdown saved: {wl_path}")

    return csv_path, comp_path, wl_path


def generate_report(results: dict, output_dir: Path):
    """Generate a text analysis report."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = output_dir / f'report_{timestamp}.txt'

    lines = []
    lines.append("=" * 70)
    lines.append("Phase-Aware DVFS Validation Experiment Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Strategy comparison table
    lines.append("## Strategy Comparison Summary")
    lines.append("-" * 70)
    lines.append(f"{'Strategy':<20} {'E/tok(J)':<10} {'TTFT(ms)':<10} {'TPOT(ms)':<10} "
                 f"{'E Save%':<10} {'SLO%':<8} {'tok/s':<10}")
    lines.append("-" * 70)

    for strategy in PhaseController.STRATEGIES:
        s = results['strategies'][strategy]['summary']
        c = results['comparison'].get(strategy, {})
        lines.append(
            f"{strategy:<20} {s['energy_per_token_j']:<10.4f} "
            f"{s['avg_ttft_ms']:<10.1f} {s['avg_tpot_ms']:<10.1f} "
            f"{c.get('energy_reduction_pct', 0):<10.1f} "
            f"{s['slo_met_rate']:<8.1%} "
            f"{s.get('avg_tokens_per_second', 0):<10.1f}"
        )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("-" * 70)

    # Compare phase_aware vs default
    pa = results['comparison'].get('phase_aware', {})
    df_comp = results['comparison'].get('default', {})
    if pa and df_comp:
        e_save = pa.get('energy_reduction_pct', 0)
        ttft_change = pa.get('ttft_reduction_pct', 0)
        tpot_change = pa.get('tpot_reduction_pct', 0)
        slo_rate = results['strategies']['phase_aware']['summary']['slo_met_rate']

        lines.append(f"Phase-Aware vs Default Baseline:")
        lines.append(f"  Energy reduction:   {e_save:+.1f}%")
        lines.append(f"  TTFT change:        {ttft_change:+.1f}%")
        lines.append(f"  TPOT change:        {tpot_change:+.1f}%")
        lines.append(f"  SLO satisfaction:    {slo_rate:.1%}")

    # Compare phase_aware vs max_perf
    mp = results['comparison'].get('max_perf', {})
    if pa and mp:
        e_vs_max = pa['energy_per_token_j'] - mp.get('energy_per_token_j', 0)
        lines.append(f"\nPhase-Aware vs Max Performance:")
        lines.append(f"  Energy difference:  {e_vs_max:+.4f} J/tok")

    lines.append("")
    lines.append("## Per-Workload Analysis")
    lines.append("-" * 70)

    # Per-workload energy comparison
    workloads = get_workloads()
    for wl in workloads:
        wl_name = wl['name']
        lines.append(f"\n  Workload: {wl_name} (prompt={wl['prompt_len']}, output={wl['output_len']})")
        for strategy in ['default', 'max_perf', 'phase_aware']:
            wl_results = [
                r for r in results['strategies'][strategy]['detailed']
                if r['workload'].get('name') == wl_name
            ]
            if wl_results:
                avg_e = np.mean([r['total_energy_j'] for r in wl_results])
                avg_ttft = np.mean([r['ttft_ms'] for r in wl_results])
                avg_tpot = np.mean([r['tpot_ms'] for r in wl_results])
                lines.append(f"    {strategy:<20} E={avg_e:.3f}J  TTFT={avg_ttft:.1f}ms  TPOT={avg_tpot:.1f}ms")

    lines.append("")
    lines.append("=" * 70)
    lines.append("End of Report")
    lines.append("=" * 70)

    report_text = '\n'.join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved: {report_path}")

    print(report_text)
    return report_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = get_config()
    workloads = get_workloads()
    repeats = 10

    logger.info("Starting Phase-Aware DVFS Validation Experiment")
    logger.info(f"Workloads: {len(workloads)}, Repeats: {repeats}")
    logger.info(f"Strategies: {PhaseController.STRATEGIES}")

    results = run_comparison_experiment(config, workloads, repeats=repeats)

    csv_path, comp_path, wl_path = save_results(results, OUTPUT_DIR)
    report_path = generate_report(results, OUTPUT_DIR)

    logger.info("Experiment completed successfully!")
    logger.info(f"Results in: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
