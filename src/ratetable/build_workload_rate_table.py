#!/usr/bin/env python3
"""
Workload-Aware Rate Table Builder

Builds a unified rate table from both lock-mode finegrained profiling and
cap-mode profiling data. Produces:
  1. Lock-mode rate table: exact E/tok at each GPU freq per workload (with Pareto ranks)
  2. Cap-mode rate table: E/tok under each GPU cap per workload
  3. Workload-aware DVFS rules: optimal GPU cap per (workload, objective)

Usage:
    python src/ratetable/build_workload_rate_table.py
    python src/ratetable/build_workload_rate_table.py --model Qwen2.5-7B-Instruct-Q4_K_M
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from src.controller.pareto_selector import compute_pareto_frontier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_finegrained_data(path: str, model_filter: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['output_tokens'] > 0].copy()
    df = df[df['energy_per_token_j'] > 0].copy()
    df = df[df['tpot_ms'] > 0].copy()
    if model_filter:
        df = df[df['model'] == model_filter]
    df['source'] = 'lock_mode'
    logger.info(f"Loaded finegrained: {len(df)} rows from {Path(path).name}")
    return df


def load_cap_data(path: str, model_filter: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df['output_tokens'] > 0].copy()
    df = df[df['energy_per_token_j'] > 0].copy()
    df = df[df['tpot_ms'] > 0].copy()
    if model_filter:
        df = df[df['model'] == model_filter]
    df['source'] = 'cap_mode'
    logger.info(f"Loaded cap profiling: {len(df)} rows from {Path(path).name}")
    return df


def aggregate_lock_mode(df: pd.DataFrame) -> pd.DataFrame:
    group = ['model', 'workload', 'prompt_length', 'output_length',
             'phase', 'gpu_freq_mhz', 'emc_freq_mhz', 'cpu_freq_mhz']
    metrics = ['ttft_ms', 'tpot_ms', 'total_time_ms', 'tokens_per_second',
               'energy_per_token_j', 'tokens_per_joule', 'avg_power_w', 'max_power_w',
               'output_tokens']
    avail = [c for c in metrics if c in df.columns]
    agg = {c: ['median', 'mean', 'std', 'count'] for c in avail}
    grouped = df.groupby(group, as_index=False).agg(agg)
    grouped.columns = ['_'.join(c).strip('_') if c[1] else c[0]
                       for c in grouped.columns.values]
    for m in avail:
        std_c, mean_c = f'{m}_std', f'{m}_mean'
        if std_c in grouped.columns and mean_c in grouped.columns:
            grouped[f'{m}_cv'] = (grouped[std_c] / grouped[mean_c] * 100).fillna(0)
    return grouped


def aggregate_cap_mode(df: pd.DataFrame) -> pd.DataFrame:
    group = ['model', 'workload', 'prompt_length', 'output_length',
             'control_mode', 'target_gpu_cap_mhz', 'target_emc_cap_mhz', 'target_cpu_cap_mhz']
    metrics = ['ttft_ms', 'tpot_ms', 'total_time_ms', 'tokens_per_second',
               'energy_per_token_j', 'tokens_per_joule', 'avg_power_w', 'max_power_w',
               'actual_gpu_mhz', 'output_tokens']
    avail = [c for c in metrics if c in df.columns]
    agg = {c: ['median', 'mean', 'std', 'count'] for c in avail}
    grouped = df.groupby(group, as_index=False).agg(agg)
    grouped.columns = ['_'.join(c).strip('_') if c[1] else c[0]
                       for c in grouped.columns.values]
    for m in avail:
        std_c, mean_c = f'{m}_std', f'{m}_mean'
        if std_c in grouped.columns and mean_c in grouped.columns:
            grouped[f'{m}_cv'] = (grouped[std_c] / grouped[mean_c] * 100).fillna(0)
    return grouped


def compute_pareto_ranks(lock_agg: pd.DataFrame) -> pd.DataFrame:
    """Pre-compute Pareto rank per (model, workload, phase) group in lock-mode data.

    Adds columns: pareto_rank, is_pareto, dominates_count, dominated_by_count.
    This allows downstream consumers (ParetoSelector, visualizations) to skip
    recomputing the non-dominated sort.
    """
    objectives = ['energy_per_token_j_median', 'tpot_ms_median', 'avg_power_w_median']
    available_obj = [o for o in objectives if o in lock_agg.columns]
    if len(available_obj) < 2:
        logger.warning(f"Insufficient objective columns ({available_obj}), skipping Pareto rank")
        return lock_agg

    result_parts = []
    grouped = lock_agg.groupby(['model', 'workload', 'phase'])
    for (model, wl, phase), group_df in grouped:
        with_ranks = compute_pareto_frontier(group_df, available_obj)
        result_parts.append(with_ranks)

    result = pd.concat(result_parts, ignore_index=True)
    n_pareto = result['is_pareto'].sum()
    logger.info(f"Pareto ranks computed: {n_pareto} Pareto-optimal / {len(result)} total configs")
    return result


def compute_dvfs_savings(cap_df: pd.DataFrame) -> pd.DataFrame:
    """Compute savings of each cap config vs dynamic and MAXN baselines."""
    results = []
    for model in cap_df['model'].unique():
        for wl in cap_df['workload'].unique():
            base = cap_df[(cap_df['model'] == model) & (cap_df['workload'] == wl)]
            dyn = base[base['control_mode'] == 'dynamic']
            maxn = base[base['control_mode'] == 'maxn']
            caps = base[base['control_mode'] == 'cap']

            if dyn.empty:
                continue
            dyn_ept = dyn['energy_per_token_j_median'].values[0]
            dyn_tpot = dyn['tpot_ms_median'].values[0]
            dyn_pwr = dyn['avg_power_w_median'].values[0]

            maxn_ept = maxn['energy_per_token_j_median'].values[0] if not maxn.empty else np.nan
            maxn_tpot = maxn['tpot_ms_median'].values[0] if not maxn.empty else np.nan

            for _, cap_row in caps.iterrows():
                r = cap_row.to_dict()
                r['savings_ept_vs_dynamic_pct'] = (dyn_ept - r['energy_per_token_j_median']) / dyn_ept * 100
                r['savings_tpot_vs_dynamic_pct'] = (dyn_tpot - r['tpot_ms_median']) / dyn_tpot * 100
                r['savings_pwr_vs_dynamic_pct'] = (dyn_pwr - r['avg_power_w_median']) / dyn_pwr * 100
                if not np.isnan(maxn_ept):
                    r['savings_ept_vs_maxn_pct'] = (maxn_ept - r['energy_per_token_j_median']) / maxn_ept * 100
                    r['savings_tpot_vs_maxn_pct'] = (maxn_tpot - r['tpot_ms_median']) / maxn_tpot * 100
                else:
                    r['savings_ept_vs_maxn_pct'] = np.nan
                    r['savings_tpot_vs_maxn_pct'] = np.nan
                results.append(r)

            # Also include dynamic and maxn rows
            for _, row in dyn.iterrows():
                r = row.to_dict()
                r['savings_ept_vs_dynamic_pct'] = 0.0
                r['savings_tpot_vs_dynamic_pct'] = 0.0
                r['savings_pwr_vs_dynamic_pct'] = 0.0
                if not np.isnan(maxn_ept):
                    r['savings_ept_vs_maxn_pct'] = (maxn_ept - r['energy_per_token_j_median']) / maxn_ept * 100
                    r['savings_tpot_vs_maxn_pct'] = (maxn_tpot - r['tpot_ms_median']) / maxn_tpot * 100
                results.append(r)
            for _, row in maxn.iterrows():
                r = row.to_dict()
                r['savings_ept_vs_dynamic_pct'] = (dyn_ept - r['energy_per_token_j_median']) / dyn_ept * 100
                r['savings_tpot_vs_dynamic_pct'] = (dyn_tpot - r['tpot_ms_median']) / dyn_tpot * 100
                r['savings_pwr_vs_dynamic_pct'] = (dyn_pwr - r['avg_power_w_median']) / dyn_pwr * 100
                r['savings_ept_vs_maxn_pct'] = 0.0
                r['savings_tpot_vs_maxn_pct'] = 0.0
                results.append(r)

    return pd.DataFrame(results)


def mine_workload_rules(lock_agg: pd.DataFrame, cap_savings: pd.DataFrame) -> List[Dict]:
    """Mine DVFS rules: optimal GPU cap per workload × objective."""
    rules = []

    # From lock-mode: best GPU freq per workload
    for model in lock_agg['model'].unique():
        m_lock = lock_agg[lock_agg['model'] == model]
        for phase in m_lock['phase'].unique():
            p_lock = m_lock[m_lock['phase'] == phase]
            for wl in p_lock['workload'].unique():
                w_lock = p_lock[p_lock['workload'] == wl]
                if w_lock.empty:
                    continue
                pl = int(w_lock['prompt_length'].iloc[0])
                ol = int(w_lock['output_length'].iloc[0])

                best_e = w_lock.loc[w_lock['energy_per_token_j_median'].idxmin()]
                best_l = w_lock.loc[w_lock['tpot_ms_median'].idxmin()]
                best_eff = w_lock.loc[w_lock['tokens_per_joule_median'].idxmax()]

                for obj, row in [('min_energy', best_e), ('min_latency', best_l),
                                 ('max_efficiency', best_eff)]:
                    rules.append({
                        'model': model, 'workload': wl,
                        'prompt_length': pl, 'output_length': ol,
                        'phase': phase, 'objective': obj,
                        'source': 'lock_mode',
                        'gpu_freq_mhz': int(row['gpu_freq_mhz']),
                        'emc_freq_mhz': int(row['emc_freq_mhz']),
                        'cpu_freq_mhz': int(row['cpu_freq_mhz']),
                        'energy_per_token_j': round(float(row['energy_per_token_j_median']), 4),
                        'tpot_ms': round(float(row['tpot_ms_median']), 1),
                        'tokens_per_second': round(float(row['tokens_per_second_median']), 1),
                        'avg_power_w': round(float(row['avg_power_w_median']), 1),
                    })

    # From cap-mode: best cap per workload
    cap_only = cap_savings[cap_savings['control_mode'] == 'cap']
    for model in cap_only['model'].unique():
        m_cap = cap_only[cap_only['model'] == model]
        for wl in m_cap['workload'].unique():
            w_cap = m_cap[m_cap['workload'] == wl]
            if w_cap.empty:
                continue
            pl = int(w_cap['prompt_length'].iloc[0])
            ol = int(w_cap['output_length'].iloc[0])

            best_e = w_cap.loc[w_cap['energy_per_token_j_median'].idxmin()]
            best_l = w_cap.loc[w_cap['tpot_ms_median'].idxmin()]

            for obj, row in [('min_energy', best_e), ('min_latency', best_l)]:
                rules.append({
                    'model': model, 'workload': wl,
                    'prompt_length': pl, 'output_length': ol,
                    'phase': 'mixed', 'objective': obj,
                    'source': 'cap_mode',
                    'gpu_cap_mhz': int(row['target_gpu_cap_mhz']),
                    'actual_gpu_mhz': round(float(row.get('actual_gpu_mhz_median', 0)), 0),
                    'energy_per_token_j': round(float(row['energy_per_token_j_median']), 4),
                    'tpot_ms': round(float(row['tpot_ms_median']), 1),
                    'tokens_per_second': round(float(row['tokens_per_second_median']), 1),
                    'avg_power_w': round(float(row['avg_power_w_median']), 1),
                    'savings_ept_vs_dynamic_pct': round(float(row.get('savings_ept_vs_dynamic_pct', 0)), 1),
                    'savings_tpot_vs_dynamic_pct': round(float(row.get('savings_tpot_vs_dynamic_pct', 0)), 1),
                })

    return rules


def build_rate_table(finegrained_csv: str, cap_csv: str,
                     model_filter: str = None,
                     output_dir: str = 'data/rate_tables') -> Dict:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    lock_df = load_finegrained_data(finegrained_csv, model_filter)
    cap_df = load_cap_data(cap_csv, model_filter)

    # Aggregate
    lock_agg = aggregate_lock_mode(lock_df)
    cap_agg = aggregate_cap_mode(cap_df)
    logger.info(f"Lock-mode aggregated: {len(lock_agg)} rows, "
                f"{lock_agg['workload'].nunique()} workloads, "
                f"{lock_agg['gpu_freq_mhz'].nunique()} GPU configs")
    logger.info(f"Cap-mode aggregated: {len(cap_agg)} rows, "
                f"{cap_agg['workload'].nunique()} workloads, "
                f"{cap_agg['control_mode'].nunique()} modes")

    # Compute Pareto ranks for lock-mode data
    lock_agg = compute_pareto_ranks(lock_agg)

    # Compute cap savings
    cap_savings = compute_dvfs_savings(cap_agg)

    # Mine rules
    rules = mine_workload_rules(lock_agg, cap_savings)
    logger.info(f"DVFS rules mined: {len(rules)}")

    # Save outputs
    out_files = {}

    p = out_dir / f'lock_rate_table_{timestamp}.parquet'
    lock_agg.to_parquet(p, index=False)
    out_files['lock_rate_table'] = str(p)
    logger.info(f"Lock rate table: {p}")

    p = out_dir / f'cap_rate_table_with_savings_{timestamp}.parquet'
    cap_savings.to_parquet(p, index=False)
    out_files['cap_rate_table'] = str(p)
    logger.info(f"Cap rate table: {p}")

    p = out_dir / f'workload_dvfs_rules_{timestamp}.json'
    with open(p, 'w') as f:
        json.dump(rules, f, indent=2, default=str)
    out_files['dvfs_rules'] = str(p)
    logger.info(f"DVFS rules: {p}")

    # Generate report
    report = generate_report(lock_agg, cap_savings, rules, lock_df)
    model_name = model_filter or 'all'
    p = out_dir / f'workload_rate_table_report_{model_name}_{timestamp}.md'
    with open(p, 'w') as f:
        f.write(report)
    out_files['report'] = str(p)
    logger.info(f"Report: {p}")

    return out_files


def generate_report(lock_agg: pd.DataFrame, cap_savings: pd.DataFrame,
                    rules: List[Dict], raw_lock: pd.DataFrame) -> str:
    lines = [
        "# Workload-Aware Rate Table Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Lock-mode rows**: {len(lock_agg)}",
        f"**Cap-mode rows**: {len(cap_savings)}",
        f"**DVFS rules**: {len(rules)}",
    ]

    for model in lock_agg['model'].unique():
        m_lock = lock_agg[lock_agg['model'] == model]
        m_cap = cap_savings[cap_savings['model'] == model]

        lines.extend(["", f"## Model: {model}", ""])

        # Workload DVFS space (lock-mode, decode phase)
        decode = m_lock[m_lock['phase'] == 'decode']
        if not decode.empty:
            lines.extend(["### Per-Workload DVFS Space (Lock Mode, Decode)", "",
                          "| Workload | Best GPU | E/tok (J) | TPOT (ms) | Pwr (W) | "
                          "E/tok Range (%) | Best→Worst GPU |",
                          "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"])
            for wl in sorted(decode['workload'].unique()):
                w = decode[decode['workload'] == wl]
                best = w.loc[w['energy_per_token_j_median'].idxmin()]
                worst = w.loc[w['energy_per_token_j_median'].idxmax()]
                rng = (worst['energy_per_token_j_median'] - best['energy_per_token_j_median']) / best['energy_per_token_j_median'] * 100
                lines.append(
                    f"| {wl} | {int(best['gpu_freq_mhz'])} | "
                    f"{best['energy_per_token_j_median']:.4f} | "
                    f"{best['tpot_ms_median']:.1f} | "
                    f"{best['avg_power_w_median']:.1f} | "
                    f"{rng:.1f}% | "
                    f"{int(best['gpu_freq_mhz'])}→{int(worst['gpu_freq_mhz'])} |"
                )

        # Cap-mode comparison
        if not m_cap.empty:
            lines.extend(["", "### Cap Mode: Savings vs Dynamic Baseline", "",
                          "| Workload | GPU Cap | E/tok (J) | TPOT (ms) | Pwr (W) | "
                          "ΔE vs Dyn | ΔTPOT vs Dyn |",
                          "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"])
            caps = m_cap[m_cap['control_mode'] == 'cap']
            for wl in sorted(caps['workload'].unique()):
                w = caps[caps['workload'] == wl]
                for _, row in w.sort_values('target_gpu_cap_mhz').iterrows():
                    lines.append(
                        f"| {wl} | {int(row['target_gpu_cap_mhz'])} | "
                        f"{row['energy_per_token_j_median']:.4f} | "
                        f"{row['tpot_ms_median']:.1f} | "
                        f"{row['avg_power_w_median']:.1f} | "
                        f"{row.get('savings_ept_vs_dynamic_pct', 0):+.1f}% | "
                        f"{row.get('savings_tpot_vs_dynamic_pct', 0):+.1f}% |"
                    )

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Build workload-aware rate table')
    parser.add_argument('--finegrained', type=str, required=True,
                        help='Path to finegrained profiling CSV')
    parser.add_argument('--cap', type=str, required=True,
                        help='Path to cap profiling CSV')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter by model name')
    parser.add_argument('--output-dir', type=str, default='data/rate_tables')
    args = parser.parse_args()

    results = build_rate_table(
        finegrained_csv=args.finegrained,
        cap_csv=args.cap,
        model_filter=args.model,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("WORKLOAD-AWARE RATE TABLE BUILD COMPLETE")
    print("=" * 60)
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == '__main__':
    main()
