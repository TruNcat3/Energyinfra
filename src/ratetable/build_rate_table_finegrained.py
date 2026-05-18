#!/usr/bin/env python3
"""
Build Energy Rate Table from Fine-Grained GPU×EMC Profiling Data

Reads real profiling data (11 GPU × 4 EMC × 1 CPU = 44 configs)
and builds a complete rate table with Pareto analysis, SLO constraints,
and DVFS rules.

Usage:
    python src/ratetable/build_rate_table_finegrained.py
    python src/ratetable/build_rate_table_finegrained.py --input data/energy_profiling/finegrained_profiling_*.csv
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import json
import argparse
from datetime import datetime
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinegrainedRateTableBuilder:
    """Build rate table from fine-grained GPU×EMC profiling data."""

    def __init__(self, csv_path: str, output_dir: str = "data/rate_tables"):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.raw_df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.raw_df)} rows from {self.csv_path.name}")

    def filter_bad_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove anomalous runs."""
        before = len(df)
        df = df[df['output_tokens'] > 0].copy()
        df = df[df['energy_per_token_j'] >= 0.1].copy()
        df = df[df['tpot_ms'] > 0].copy()
        after = len(df)
        if before != after:
            logger.info(f"Filtered {before - after} bad rows, {after} remaining")
        return df

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and add missing fields."""
        df = df.rename(columns={
            'avg_temp_cpu_c': 'temperature_c',
        })
        for col in ['batch_size', 'concurrency']:
            if col not in df.columns:
                df[col] = 1
        if 'temperature_c' not in df.columns:
            df['temperature_c'] = np.nan
        return df

    def create_workload_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create workload bucket keys."""
        bucket_cols = ['model', 'runtime', 'batch_size',
                       'prompt_length', 'output_length', 'phase', 'concurrency']
        for col in bucket_cols:
            if col not in df.columns:
                df[col] = 'unknown'
        df['bucket_key'] = df[bucket_cols].astype(str).agg('|'.join, axis=1)
        return df

    def aggregate_by_config(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate repeats by (bucket_key, gpu_freq_mhz, emc_freq_mhz, cpu_freq_mhz)."""
        bucket_cols = ['model', 'runtime', 'batch_size', 'prompt_length',
                       'output_length', 'phase', 'concurrency', 'bucket_key']
        freq_cols = ['gpu_freq_mhz', 'cpu_freq_mhz', 'emc_freq_mhz']
        metric_cols = [
            'ttft_ms', 'tpot_ms', 'total_time_ms', 'tokens_per_second',
            'avg_power_w', 'max_power_w', 'temperature_c',
            'energy_per_token_j', 'tokens_per_joule',
            'avg_gpu_soc_w', 'avg_cpu_cv_w', 'avg_sys_5v0_w',
            'output_tokens',
        ]
        available_metrics = [c for c in metric_cols if c in df.columns]
        group_cols = bucket_cols + freq_cols
        agg_funcs = {col: ['median', 'mean', 'std', 'count'] for col in available_metrics}
        grouped = df.groupby(group_cols, as_index=False).agg(agg_funcs)
        grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                           for col in grouped.columns.values]
        for metric in available_metrics:
            std_col, mean_col = f'{metric}_std', f'{metric}_mean'
            if std_col in grouped.columns and mean_col in grouped.columns:
                grouped[f'{metric}_cv'] = (
                    grouped[std_col] / grouped[mean_col] * 100
                ).fillna(0)
        return grouped

    def calculate_pareto(self, df: pd.DataFrame) -> pd.DataFrame:
        """Non-dominated sorting per bucket."""
        objectives = [
            'energy_per_token_j_median',
            'tpot_ms_median',
            'avg_power_w_median',
        ]
        available = [o for o in objectives if o in df.columns]
        if not available:
            df['is_pareto'] = False
            df['pareto_rank'] = -1
            return df

        results = []
        for bk in df['bucket_key'].unique():
            sub = df[df['bucket_key'] == bk].copy()
            vals = sub[available].apply(pd.to_numeric, errors='coerce').fillna(np.inf)
            ranks = np.full(len(sub), -1.0)
            remaining = set(range(len(sub)))
            cur_rank = 1
            while remaining:
                frontier = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i == j:
                            continue
                        if all(vals.iloc[j][o] <= vals.iloc[i][o] for o in available) and \
                           any(vals.iloc[j][o] < vals.iloc[i][o] for o in available):
                            dominated = True
                            break
                    if not dominated:
                        frontier.append(i)
                for idx in frontier:
                    ranks[idx] = cur_rank
                    remaining.remove(idx)
                cur_rank += 1
            sub['pareto_rank'] = ranks
            sub['is_pareto'] = sub['pareto_rank'] == 1
            results.append(sub)
        return pd.concat(results, ignore_index=True)

    def apply_slo(self, df: pd.DataFrame, slo: Dict = None) -> pd.DataFrame:
        """Apply SLO constraints."""
        if slo is None:
            slo = {'ttft_ms': 2000, 'tpot_ms': 100, 'max_power_w': 45}
        df['slo_ttft_met'] = df['ttft_ms_median'] <= slo['ttft_ms']
        df['slo_tpot_met'] = df['tpot_ms_median'] <= slo['tpot_ms']
        df['slo_power_met'] = df['avg_power_w_median'] <= slo['max_power_w']
        df['slo_all_met'] = df['slo_ttft_met'] & df['slo_tpot_met'] & df['slo_power_met']
        df['ttft_margin'] = slo['ttft_ms'] - df['ttft_ms_median']
        df['tpot_margin'] = slo['tpot_ms'] - df['tpot_ms_median']
        return df

    def mine_dvfs_rules(self, df: pd.DataFrame) -> List[Dict]:
        """Mine DVFS rules: best config per (workload_bucket, optimization_objective)."""
        rules = []
        for bk in df['bucket_key'].unique():
            sub = df[df['bucket_key'] == bk]
            if sub.empty:
                continue
            parts = dict(zip(
                ['model', 'runtime', 'batch_size', 'prompt_length', 'output_length', 'phase', 'concurrency'],
                bk.split('|')
            ))

            slo_ok = sub[sub['slo_all_met']]
            if slo_ok.empty:
                slo_ok = sub

            # Rule 1: min energy (among SLO-feasible)
            best_e = slo_ok.loc[slo_ok['energy_per_token_j_median'].idxmin()]
            rules.append({
                'bucket_key': bk, 'objective': 'min_energy',
                'prompt_length': int(parts.get('prompt_length', 0)),
                'output_length': int(parts.get('output_length', 0)),
                'phase': parts.get('phase', 'unknown'),
                'gpu_freq_mhz': int(best_e['gpu_freq_mhz']),
                'emc_freq_mhz': int(best_e['emc_freq_mhz']),
                'cpu_freq_mhz': int(best_e['cpu_freq_mhz']),
                'energy_per_token_j': round(float(best_e['energy_per_token_j_median']), 4),
                'tpot_ms': round(float(best_e['tpot_ms_median']), 1),
                'tokens_per_second': round(float(best_e['tokens_per_second_median']), 1),
                'avg_power_w': round(float(best_e['avg_power_w_median']), 1),
                'is_pareto': bool(best_e.get('is_pareto', False)),
            })

            # Rule 2: min latency (among SLO-feasible)
            best_l = slo_ok.loc[slo_ok['tpot_ms_median'].idxmin()]
            rules.append({
                'bucket_key': bk, 'objective': 'min_latency',
                'prompt_length': int(parts.get('prompt_length', 0)),
                'output_length': int(parts.get('output_length', 0)),
                'phase': parts.get('phase', 'unknown'),
                'gpu_freq_mhz': int(best_l['gpu_freq_mhz']),
                'emc_freq_mhz': int(best_l['emc_freq_mhz']),
                'cpu_freq_mhz': int(best_l['cpu_freq_mhz']),
                'energy_per_token_j': round(float(best_l['energy_per_token_j_median']), 4),
                'tpot_ms': round(float(best_l['tpot_ms_median']), 1),
                'tokens_per_second': round(float(best_l['tokens_per_second_median']), 1),
                'avg_power_w': round(float(best_l['avg_power_w_median']), 1),
                'is_pareto': bool(best_l.get('is_pareto', False)),
            })

            # Rule 3: best tok/J (energy efficiency)
            if 'tokens_per_joule_median' in sub.columns:
                best_eff = sub.loc[sub['tokens_per_joule_median'].idxmax()]
                rules.append({
                    'bucket_key': bk, 'objective': 'max_efficiency',
                    'prompt_length': int(parts.get('prompt_length', 0)),
                    'output_length': int(parts.get('output_length', 0)),
                    'phase': parts.get('phase', 'unknown'),
                    'gpu_freq_mhz': int(best_eff['gpu_freq_mhz']),
                    'emc_freq_mhz': int(best_eff['emc_freq_mhz']),
                    'cpu_freq_mhz': int(best_eff['cpu_freq_mhz']),
                    'energy_per_token_j': round(float(best_eff['energy_per_token_j_median']), 4),
                    'tpot_ms': round(float(best_eff['tpot_ms_median']), 1),
                    'tokens_per_second': round(float(best_eff['tokens_per_second_median']), 1),
                    'avg_power_w': round(float(best_eff['avg_power_w_median']), 1),
                    'is_pareto': bool(best_eff.get('is_pareto', False)),
                })

        return rules

    def build(self) -> Dict:
        """Full pipeline: filter → normalize → bucket → aggregate → pareto → SLO → rules."""
        df = self.filter_bad_data(self.raw_df)
        df = self.normalize_columns(df)
        df = self.create_workload_buckets(df)

        logger.info(f"Unique buckets: {df['bucket_key'].nunique()}")
        logger.info(f"Unique configs (GPU×EMC): {df.groupby(['gpu_freq_mhz','emc_freq_mhz']).ngroups}")

        agg = self.aggregate_by_config(df)
        logger.info(f"Aggregated: {len(agg)} rows")

        pareto = self.calculate_pareto(agg)
        n_pareto = pareto['is_pareto'].sum()
        logger.info(f"Pareto-optimal configs: {n_pareto} / {len(pareto)}")

        slo = self.apply_slo(pareto)
        n_slo = slo['slo_all_met'].sum()
        logger.info(f"SLO-feasible configs: {n_slo} / {len(slo)}")

        rules = self.mine_dvfs_rules(slo)
        logger.info(f"DVFS rules mined: {len(rules)}")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_files = {}

        # Save aggregated rate table
        p = self.output_dir / f'finegrained_rate_table_{timestamp}.parquet'
        slo.to_parquet(p, index=False)
        out_files['rate_table'] = str(p)
        logger.info(f"Rate table: {p}")

        # Save DVFS rules
        p = self.output_dir / f'finegrained_dvfs_rules_{timestamp}.json'
        with open(p, 'w') as f:
            json.dump(rules, f, indent=2, default=str)
        out_files['dvfs_rules'] = str(p)
        logger.info(f"DVFS rules: {p}")

        # Save selector table (clean subset for runtime use)
        selector_cols = [
            'bucket_key', 'model', 'runtime', 'phase',
            'prompt_length', 'output_length',
            'gpu_freq_mhz', 'cpu_freq_mhz', 'emc_freq_mhz',
            'ttft_ms_median', 'tpot_ms_median',
            'tokens_per_second_median',
            'energy_per_token_j_median', 'tokens_per_joule_median',
            'avg_power_w_median', 'max_power_w_median',
            'avg_gpu_soc_w_median', 'avg_cpu_cv_w_median', 'avg_sys_5v0_w_median',
            'slo_all_met', 'is_pareto', 'pareto_rank',
            'energy_per_token_j_cv', 'tpot_ms_cv',
        ]
        sel_cols = [c for c in selector_cols if c in slo.columns]
        selector = slo[sel_cols].copy()
        p = self.output_dir / f'finegrained_selector_table_{timestamp}.parquet'
        selector.to_parquet(p, index=False)
        out_files['selector_table'] = str(p)
        logger.info(f"Selector table: {p}")

        # Generate summary report
        report = self._generate_report(slo, rules, df)
        p = self.output_dir / f'finegrained_rate_table_report_{timestamp}.md'
        with open(p, 'w') as f:
            f.write(report)
        out_files['report'] = str(p)
        logger.info(f"Report: {p}")

        return out_files

    def _generate_report(self, slo_df: pd.DataFrame, rules: List[Dict], raw: pd.DataFrame) -> str:
        lines = [
            "# Fine-Grained Rate Table Report",
            f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Source**: {self.csv_path.name}",
            f"**Raw rows**: {len(raw)} (after filtering)",
            f"**Unique configs**: {slo_df.groupby(['gpu_freq_mhz','emc_freq_mhz']).ngroups} (GPU×EMC)",
            f"**Unique buckets**: {slo_df['bucket_key'].nunique()}",
            f"**Pareto-optimal**: {int(slo_df['is_pareto'].sum())}",
            f"**SLO-feasible**: {int(slo_df['slo_all_met'].sum())}",
            f"**DVFS rules**: {len(rules)}",
            "",
            "## Config Space",
            f"- GPU: 11 levels (306-1300 MHz)",
            f"- EMC: 4 levels (204-3199 MHz)",
            f"- CPU: 1036 MHz (fixed)",
            "",
            "## DVFS Rules Summary",
            "",
            "| Objective | Phase | Prompt | Output | GPU (MHz) | EMC (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
        for r in sorted(rules, key=lambda x: (x['phase'], x['prompt_length'], x['objective'])):
            lines.append(
                f"| {r['objective']} | {r['phase']} | {r['prompt_length']} | {r['output_length']} "
                f"| {r['gpu_freq_mhz']} | {r['emc_freq_mhz']} "
                f"| {r['energy_per_token_j']:.4f} | {r['tpot_ms']:.1f} "
                f"| {r['tokens_per_second']:.1f} | {r['avg_power_w']:.1f} |"
            )

        # Pareto analysis per bucket
        lines.extend(["", "## Pareto Analysis by Workload Bucket", ""])
        for bk in sorted(slo_df['bucket_key'].unique()):
            sub = slo_df[slo_df['bucket_key'] == bk]
            pareto_sub = sub[sub['is_pareto']]
            lines.append(f"### {bk}")
            lines.append(f"- Total configs: {len(sub)}, Pareto-optimal: {len(pareto_sub)}, SLO-feasible: {int(sub['slo_all_met'].sum())}")
            if not pareto_sub.empty:
                lines.append("")
                lines.append("| GPU | EMC | E/tok (J) | TPOT (ms) | Power (W) | tok/J |")
                lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|")
                for _, row in pareto_sub.sort_values('energy_per_token_j_median').iterrows():
                    lines.append(
                        f"| {int(row['gpu_freq_mhz'])} | {int(row['emc_freq_mhz'])} "
                        f"| {row['energy_per_token_j_median']:.4f} | {row['tpot_ms_median']:.1f} "
                        f"| {row['avg_power_w_median']:.1f} "
                        f"| {row['tokens_per_joule_median']:.2f} |"
                    )
            lines.append("")

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Build rate table from fine-grained profiling data')
    parser.add_argument('--input', type=str,
                        default='data/energy_profiling/finegrained_profiling_20260516_015611.csv',
                        help='Path to fine-grained profiling CSV')
    parser.add_argument('--output-dir', type=str, default='data/rate_tables',
                        help='Output directory')
    args = parser.parse_args()

    builder = FinegrainedRateTableBuilder(csv_path=args.input, output_dir=args.output_dir)
    results = builder.build()

    print("\n" + "=" * 60)
    print("FINE-GRAINED RATE TABLE BUILD COMPLETE")
    print("=" * 60)
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    main()
