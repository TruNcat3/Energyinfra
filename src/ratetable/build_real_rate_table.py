#!/usr/bin/env python3
"""
Build Energy Rate Table from Real GPU Profiling Data

Processes energy_profiling CSV files into a structured rate table with:
  - Workload buckets (prompt_length × output_length × phase)
  - Per-config energy metrics (E/token, TTFT, TPOT, power)
  - Pareto frontier analysis
  - Phase-specific optimal configs for Phase-Aware DVFS
  - Interpolation rules for unseen workload sizes
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RealRateTableBuilder:
    """Build rate tables from real Jetson GPU energy profiling data."""

    def __init__(self, data_files: List[str], output_dir: str = "data/rate_tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_data = self._load_data(data_files)
        logger.info(f"Loaded {len(self.raw_data)} rows from {len(data_files)} files")

    def _load_data(self, data_files: List[str]) -> pd.DataFrame:
        dfs = []
        for f in data_files:
            path = Path(f)
            if path.exists():
                df = pd.read_csv(path)
                df['source_file'] = path.name
                dfs.append(df)
                logger.info(f"  {path.name}: {len(df)} rows")
            else:
                logger.warning(f"  File not found: {f}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def filter_valid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid runs (zero tokens, zero energy)."""
        valid = df[(df['output_tokens'] > 0) & (df['total_energy_j'] > 0)].copy()
        dropped = len(df) - len(valid)
        if dropped > 0:
            logger.info(f"Filtered {dropped} invalid runs (zero tokens/energy)")
        return valid

    def build_bucket_key(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create workload bucket key from prompt_length × output_length × phase."""
        df = df.copy()
        df['bucket_key'] = (
            'p' + df['prompt_length'].astype(str) +
            '_o' + df['output_length'].astype(str) +
            '_' + df['phase'].astype(str)
        )
        return df

    def aggregate_by_config(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate repeat runs by config × workload × phase."""
        group_cols = ['config_name', 'gpu_freq_mhz', 'cpu_freq_mhz',
                      'workload', 'phase', 'prompt_length', 'output_length', 'bucket_key']

        metrics = {
            'ttft_ms': ['mean', 'std', 'min', 'max'],
            'tpot_ms': ['mean', 'std', 'min', 'max'],
            'tokens_per_second': ['mean', 'std'],
            'total_time_ms': ['mean', 'std'],
            'output_tokens': ['mean'],
            'avg_power_w': ['mean', 'std'],
            'max_power_w': ['mean'],
            'total_energy_j': ['mean', 'std'],
            'energy_per_token_j': ['mean', 'std'],
            'tokens_per_joule': ['mean', 'std'],
            'avg_gpu_soc_w': ['mean', 'std'],
            'avg_cpu_cv_w': ['mean', 'std'],
            'avg_sys_5v0_w': ['mean', 'std'],
            'avg_temp_cpu_c': ['mean'],
        }

        grouped = df.groupby(group_cols).agg(metrics).reset_index()
        grouped.columns = ['_'.join(col).rstrip('_') if col[1] else col[0]
                          for col in grouped.columns.values]

        # Calculate CV for key metrics
        for metric in ['ttft_ms', 'tpot_ms', 'energy_per_token_j']:
            std_col = f'{metric}_std'
            mean_col = f'{metric}_mean'
            if std_col in grouped.columns and mean_col in grouped.columns:
                grouped[f'{metric}_cv'] = (
                    grouped[std_col] / grouped[mean_col] * 100
                ).fillna(0).replace([np.inf, -np.inf], 0)

        return grouped

    def find_optimal_configs(self, agg_df: pd.DataFrame) -> Dict:
        """Find optimal config for each workload bucket."""
        results = {}

        for bucket_key in agg_df['bucket_key'].unique():
            bucket = agg_df[agg_df['bucket_key'] == bucket_key]
            if bucket.empty:
                continue

            # Best energy efficiency
            best_energy_idx = bucket['energy_per_token_j_mean'].idxmin()
            best_energy = bucket.loc[best_energy_idx]

            # Best throughput
            best_tps_idx = bucket['tokens_per_second_mean'].idxmax()
            best_tps = bucket.loc[best_tps_idx]

            # Best TTFT
            best_ttft_idx = bucket['ttft_ms_mean'].idxmin()
            best_ttft = bucket.loc[best_ttft_idx]

            results[bucket_key] = {
                'best_energy_config': best_energy['config_name'],
                'best_energy_ept': float(best_energy['energy_per_token_j_mean']),
                'best_energy_tps': float(best_energy['tokens_per_second_mean']),
                'best_tps_config': best_tps['config_name'],
                'best_tps': float(best_tps['tokens_per_second_mean']),
                'best_ttft_config': best_ttft['config_name'],
                'best_ttft': float(best_ttft['ttft_ms_mean']),
                'num_configs': len(bucket),
            }

        return results

    def analyze_gpu_scaling(self, agg_df: pd.DataFrame) -> Dict:
        """Analyze GPU frequency scaling patterns per phase and workload size."""
        results = {}

        for phase in agg_df['phase'].unique():
            phase_data = agg_df[agg_df['phase'] == phase]
            phase_results = {}

            for gpu_freq in sorted(phase_data['gpu_freq_mhz'].unique()):
                gpu_data = phase_data[phase_data['gpu_freq_mhz'] == gpu_freq]
                phase_results[str(gpu_freq)] = {
                    'avg_ttft_ms': float(gpu_data['ttft_ms_mean'].mean()),
                    'avg_tpot_ms': float(gpu_data['tpot_ms_mean'].mean()),
                    'avg_ept_j': float(gpu_data['energy_per_token_j_mean'].mean()),
                    'avg_power_w': float(gpu_data['avg_power_w_mean'].mean()),
                    'avg_gpu_soc_w': float(gpu_data['avg_gpu_soc_w_mean'].mean()),
                }

            results[phase] = phase_results

        return results

    def analyze_workload_scaling(self, agg_df: pd.DataFrame) -> Dict:
        """Analyze how energy scales with prompt_length and output_length."""
        results = {}

        for phase in agg_df['phase'].unique():
            phase_data = agg_df[agg_df['phase'] == phase]
            phase_results = {}

            # Group by prompt_length
            for pl in sorted(phase_data['prompt_length'].unique()):
                pl_data = phase_data[phase_data['prompt_length'] == pl]
                ol_results = {}
                for ol in sorted(pl_data['output_length'].unique()):
                    ol_data = pl_data[pl_data['output_length'] == ol]
                    # Use the most common optimal config (GPU612)
                    best_data = ol_data[ol_data['gpu_freq_mhz'] == 612]
                    if best_data.empty:
                        best_data = ol_data

                    ol_results[str(ol)] = {
                        'ttft_ms': float(best_data['ttft_ms_mean'].mean()),
                        'tpot_ms': float(best_data['tpot_ms_mean'].mean()),
                        'ept_j': float(best_data['energy_per_token_j_mean'].mean()),
                        'power_w': float(best_data['avg_power_w_mean'].mean()),
                    }
                phase_results[str(pl)] = ol_results

            results[phase] = phase_results

        return results

    def mine_dvfs_rules(self, agg_df: pd.DataFrame) -> List[Dict]:
        """Mine Phase-Aware DVFS rules from the rate table data.

        Optimization targets per phase:
          - prefill: minimize TTFT (energy/token unreliable with 1 output token)
          - decode: minimize E/token
          - mixed: minimize E/token
        """
        rules = []

        for phase in agg_df['phase'].unique():
            phase_data = agg_df[agg_df['phase'] == phase]

            # Choose optimization target
            if phase == 'prefill':
                opt_col = 'ttft_ms_mean'
                opt_fn = 'idxmin'
            else:
                opt_col = 'energy_per_token_j_mean'
                opt_fn = 'idxmin'

            for pl in sorted(phase_data['prompt_length'].unique()):
                for ol in sorted(phase_data['output_length'].unique()):
                    sub = phase_data[
                        (phase_data['prompt_length'] == pl) &
                        (phase_data['output_length'] == ol)
                    ]
                    if sub.empty:
                        continue

                    if opt_fn == 'idxmin':
                        best_idx = sub[opt_col].idxmin()
                    else:
                        best_idx = sub[opt_col].idxmax()
                    best = sub.loc[best_idx]

                    rules.append({
                        'phase': phase,
                        'prompt_length': int(pl),
                        'output_length': int(ol),
                        'optimal_gpu_freq': int(best['gpu_freq_mhz']),
                        'optimal_cpu_freq': int(best['cpu_freq_mhz']),
                        'optimal_config': best['config_name'],
                        'ept_j': float(best['energy_per_token_j_mean']),
                        'ttft_ms': float(best['ttft_ms_mean']),
                        'tpot_ms': float(best['tpot_ms_mean']),
                        'power_w': float(best['avg_power_w_mean']),
                    })

        return rules

    def compute_pareto(self, agg_df: pd.DataFrame) -> pd.DataFrame:
        """Compute Pareto frontier within each bucket (minimize E/token + TPOT)."""
        results = []

        for bucket_key in agg_df['bucket_key'].unique():
            bucket = agg_df[agg_df['bucket_key'] == bucket_key].copy()
            if len(bucket) < 2:
                bucket['is_pareto'] = True
                bucket['pareto_rank'] = 1
                results.append(bucket)
                continue

            # Pareto: minimize energy_per_token_j AND minimize tpot_ms
            objectives = ['energy_per_token_j_mean', 'tpot_ms_mean']
            values = bucket[objectives].values

            ranks = np.zeros(len(bucket))
            remaining = set(range(len(bucket)))
            rank = 1

            while remaining:
                frontier = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i == j:
                            continue
                        if (values[j][0] <= values[i][0] and values[j][1] <= values[i][1] and
                            (values[j][0] < values[i][0] or values[j][1] < values[i][1])):
                            dominated = True
                            break
                    if not dominated:
                        frontier.append(i)

                for idx in frontier:
                    ranks[idx] = rank
                    remaining.remove(idx)
                rank += 1

            bucket['pareto_rank'] = ranks
            bucket['is_pareto'] = ranks == 1
            results.append(bucket)

        return pd.concat(results, ignore_index=True)

    def build_rate_table(self):
        """Main entry point: build complete rate table from profiling data."""
        if self.raw_data.empty:
            logger.error("No data loaded")
            return None

        logger.info("Step 1: Filtering valid runs...")
        valid_df = self.filter_valid(self.raw_data)

        logger.info("Step 2: Creating workload buckets...")
        bucketed_df = self.build_bucket_key(valid_df)

        logger.info("Step 3: Aggregating by config...")
        agg_df = self.aggregate_by_config(bucketed_df)
        logger.info(f"  {len(agg_df)} aggregated rows, {agg_df['bucket_key'].nunique()} buckets")

        logger.info("Step 4: Finding optimal configs...")
        optimal = self.find_optimal_configs(agg_df)

        logger.info("Step 5: Analyzing GPU scaling...")
        gpu_scaling = self.analyze_gpu_scaling(agg_df)

        logger.info("Step 6: Analyzing workload scaling...")
        workload_scaling = self.analyze_workload_scaling(agg_df)

        logger.info("Step 7: Mining DVFS rules...")
        dvfs_rules = self.mine_dvfs_rules(agg_df)

        logger.info("Step 8: Computing Pareto frontier...")
        pareto_df = self.compute_pareto(agg_df)

        # Save outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        agg_path = self.output_dir / f'real_rate_table_{timestamp}.parquet'
        agg_df.to_parquet(agg_path, index=False)
        logger.info(f"Saved: {agg_path}")

        pareto_path = self.output_dir / f'real_pareto_{timestamp}.parquet'
        pareto_df.to_parquet(pareto_path, index=False)
        logger.info(f"Saved: {pareto_path}")

        rules_path = self.output_dir / f'dvfs_rules_{timestamp}.json'
        with open(rules_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'model': 'Phi-3-mini-4k-instruct-Q4',
                'platform': 'Jetson AGX Orin',
                'optimal_configs': optimal,
                'gpu_scaling': gpu_scaling,
                'workload_scaling': workload_scaling,
                'dvfs_rules': dvfs_rules,
            }, f, indent=2, default=str)
        logger.info(f"Saved: {rules_path}")

        # Summary
        self._print_summary(agg_df, dvfs_rules, optimal)

        return {
            'rate_table': str(agg_path),
            'pareto': str(pareto_path),
            'dvfs_rules': str(rules_path),
        }

    def _print_summary(self, agg_df, dvfs_rules, optimal):
        """Print rate table summary."""
        print("\n" + "=" * 60)
        print("RATE TABLE SUMMARY")
        print("=" * 60)
        print(f"  Total buckets: {agg_df['bucket_key'].nunique()}")
        print(f"  Total configs per bucket: {agg_df.groupby('bucket_key').size().max()}")
        print(f"  Phases: {agg_df['phase'].unique().tolist()}")
        print(f"  GPU freqs: {sorted(agg_df['gpu_freq_mhz'].unique().tolist())}")
        print(f"  CPU freqs: {sorted(agg_df['cpu_freq_mhz'].unique().tolist())}")
        print(f"  Prompt lengths: {sorted(agg_df['prompt_length'].unique().tolist())}")
        print(f"  Output lengths: {sorted(agg_df['output_length'].unique().tolist())}")
        print(f"  DVFS rules mined: {len(dvfs_rules)}")

        # Show rule consistency
        if dvfs_rules:
            mixed_rules = [r for r in dvfs_rules if r['phase'] == 'mixed']
            decode_rules = [r for r in dvfs_rules if r['phase'] == 'decode']

            if mixed_rules:
                gpu_counts = {}
                for r in mixed_rules:
                    gf = r['optimal_gpu_freq']
                    gpu_counts[gf] = gpu_counts.get(gf, 0) + 1
                print(f"\n  Mixed phase optimal GPU freq distribution: {gpu_counts}")

            if decode_rules:
                gpu_counts = {}
                for r in decode_rules:
                    gf = r['optimal_gpu_freq']
                    gpu_counts[gf] = gpu_counts.get(gf, 0) + 1
                print(f"  Decode phase optimal GPU freq distribution: {gpu_counts}")

        print("=" * 60)


def main():
    import sys

    data_dir = Path('data/energy_profiling')

    # Find all profiling CSV files
    csv_files = sorted(data_dir.glob('energy_profiling_*.csv')) + \
                sorted(data_dir.glob('expanded_profiling_*.csv'))

    if not csv_files:
        logger.error("No profiling data found in data/energy_profiling/")
        return 1

    # Use only GPU-enabled data (from 2026-05-14)
    gpu_files = [str(f) for f in csv_files if '20260514' in f.name or '2026051' in f.name]
    # Also include expanded profiling if available
    gpu_files = [str(f) for f in csv_files]

    logger.info(f"Found {len(gpu_files)} data files:")
    for f in gpu_files:
        logger.info(f"  {f}")

    builder = RealRateTableBuilder(data_files=gpu_files)
    results = builder.build_rate_table()

    if results:
        print("\nGenerated files:")
        for name, path in results.items():
            print(f"  {name}: {path}")
        return 0
    return 1


if __name__ == '__main__':
    import sys
    project_root = Path(__file__).resolve().parent.parent.parent
    import os
    os.chdir(project_root)
    sys.path.insert(0, str(project_root))
    sys.exit(main())
