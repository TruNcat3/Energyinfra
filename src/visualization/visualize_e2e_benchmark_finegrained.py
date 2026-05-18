#!/usr/bin/env python3
"""
Visualize E2E Benchmark Results (Fine-Grained GPU×EMC)

Generates:
  1. Pareto frontier: E/token vs TPOT, per alpha, with baselines
  2. Alpha effect: E/token, TPOT, Power across alpha values
  3. Strategy comparison: single_config vs phase_aware
  4. Predicted vs actual: rate table predictions vs measured values
  5. Workload heatmap: best config per workload

Usage:
    python src/visualization/visualize_e2e_benchmark_finegrained.py
    python src/visualization/visualize_e2e_benchmark_finegrained.py --input data/e2e_benchmark/e2e_benchmark_*.csv
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_latest_csv(pattern: str) -> Path:
    files = sorted(Path('data/e2e_benchmark').glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return files[-1]


def plot_pareto(df: pd.DataFrame, out_dir: Path):
    """Pareto frontier: E/token vs TPOT."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot alpha points
    strategies = df[df['strategy'].isin(['single_config', 'phase_aware'])]
    for strategy, marker in [('single_config', 'o'), ('phase_aware', 's')]:
        sub = strategies[strategies['strategy'] == strategy]
        for alpha in sorted(sub['alpha'].unique()):
            a = sub[sub['alpha'] == alpha]
            ax.scatter(a['energy_per_token_j'].mean(), a['tpot_ms'].mean(),
                      marker=marker, s=100, label=f'{strategy} α={alpha}', alpha=0.8)

    # Plot baselines
    baselines = df[df['strategy'] == 'baseline']
    for bl_name in sorted(baselines['baseline_name'].unique()):
        bl = baselines[baselines['baseline_name'] == bl_name]
        ax.scatter(bl['energy_per_token_j'].mean(), bl['tpot_ms'].mean(),
                  marker='X', s=200, edgecolors='black', linewidths=2,
                  label=f'Baseline: {bl_name}', zorder=5)

    ax.set_xlabel('Energy per Token (J)', fontsize=12)
    ax.set_ylabel('TPOT (ms)', fontsize=12)
    ax.set_title('Pareto Frontier: Energy vs Latency (Fine-Grained GPU×EMC)', fontsize=13)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'pareto_frontier_finegrained.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: pareto_frontier_finegrained.png")


def plot_alpha_effect(df: pd.DataFrame, out_dir: Path):
    """Alpha effect on E/token, TPOT, Power."""
    strategies = df[df['strategy'].isin(['single_config', 'phase_aware'])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ('energy_per_token_j', 'Energy per Token (J)'),
        ('tpot_ms', 'TPOT (ms)'),
        ('tokens_per_second', 'Throughput (TPS)'),
    ]
    available_metrics = [(m, y) for m, y in metrics if m in strategies.columns]
    if len(available_metrics) < len(metrics):
        axes = axes[:len(available_metrics)]
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 5))
    metrics = available_metrics

    for ax, (metric, ylabel) in zip(axes if hasattr(axes, '__iter__') else [axes], metrics):
        for strategy, marker in [('single_config', 'o'), ('phase_aware', 's')]:
            sub = strategies[strategies['strategy'] == strategy]
            grouped = sub.groupby('alpha')[metric].agg(['mean', 'std']).sort_index()
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker=marker, capsize=5, label=strategy, linewidth=2)

        # Add baseline lines
        baselines = df[df['strategy'] == 'baseline']
        for bl_name in sorted(baselines['baseline_name'].unique()):
            bl = baselines[baselines['baseline_name'] == bl_name]
            ax.axhline(bl[metric].mean(), linestyle='--', alpha=0.5, label=bl_name)

        ax.set_xlabel('Alpha', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Alpha Effect on Energy, Latency, and Power', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'alpha_effect_finegrained.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: alpha_effect_finegrained.png")


def plot_strategy_comparison(df: pd.DataFrame, out_dir: Path):
    """Bar chart comparing single_config vs phase_aware."""
    strategies = df[df['strategy'].isin(['single_config', 'phase_aware'])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ('energy_per_token_j', 'Energy per Token (J)'),
        ('tpot_ms', 'TPOT (ms)'),
        ('tokens_per_second', 'Throughput (TPS)'),
    ]
    available_metrics = [(m, y) for m, y in metrics if m in strategies.columns]
    metrics = available_metrics
    if len(metrics) < 3:
        axes = axes[:len(metrics)]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))

    alphas = sorted(strategies['alpha'].unique())
    x = np.arange(len(alphas))
    width = 0.35

    for ax, (metric, ylabel) in zip(axes, metrics):
        sc_means = []
        pa_means = []
        for a in alphas:
            sc = strategies[(strategies['strategy'] == 'single_config') & (strategies['alpha'] == a)]
            pa = strategies[(strategies['strategy'] == 'phase_aware') & (strategies['alpha'] == a)]
            sc_means.append(sc[metric].mean() if not sc.empty else 0)
            pa_means.append(pa[metric].mean() if not pa.empty else 0)

        ax.bar(x - width/2, sc_means, width, label='single_config', color='steelblue')
        ax.bar(x + width/2, pa_means, width, label='phase_aware', color='coral')
        ax.set_xlabel('Alpha')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a:.1f}' for a in alphas])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Strategy Comparison: Single Config vs Phase-Aware', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'strategy_comparison_finegrained.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: strategy_comparison_finegrained.png")


def plot_workload_heatmap(df: pd.DataFrame, out_dir: Path):
    """Heatmap of best E/token config per workload."""
    strategies = df[df['strategy'].isin(['single_config', 'phase_aware'])]
    fig, ax = plt.subplots(figsize=(12, 6))

    workloads = sorted(strategies['workload'].unique())
    alphas = sorted(strategies['alpha'].unique())

    ept_matrix = np.zeros((len(workloads), len(alphas)))
    for i, wl in enumerate(workloads):
        for j, alpha in enumerate(alphas):
            sub = strategies[(strategies['workload'] == wl) & (strategies['alpha'] == alpha)]
            ept_matrix[i, j] = sub['energy_per_token_j'].mean() if not sub.empty else np.nan

    im = ax.imshow(ept_matrix, cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f'α={a:.1f}' for a in alphas])
    ax.set_yticks(range(len(workloads)))
    ax.set_yticklabels(workloads)
    ax.set_title('Energy per Token (J) by Workload × Alpha', fontsize=13)

    for i in range(len(workloads)):
        for j in range(len(alphas)):
            ax.text(j, i, f'{ept_matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)

    plt.colorbar(im, ax=ax, label='E/token (J)')
    plt.tight_layout()
    fig.savefig(out_dir / 'workload_heatmap_finegrained.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: workload_heatmap_finegrained.png")


def plot_energy_savings(df: pd.DataFrame, out_dir: Path):
    """Energy savings vs MAXN baseline."""
    baselines = df[df['strategy'] == 'baseline']
    maxn = baselines[baselines['baseline_name'] == 'MAXN']
    if maxn.empty:
        logger.warning("No MAXN baseline found")
        return

    maxn_ept = maxn['energy_per_token_j'].mean()
    maxn_tpot = maxn['tpot_ms'].mean()

    strategies = df[df['strategy'].isin(['single_config', 'phase_aware'])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric_col, baseline_val, title in [
        (axes[0], 'energy_per_token_j', maxn_ept, 'Energy Savings vs MAXN'),
        (axes[1], 'tpot_ms', maxn_tpot, 'Latency Change vs MAXN'),
    ]:
        for strategy, marker in [('single_config', 'o'), ('phase_aware', 's')]:
            sub = strategies[strategies['strategy'] == strategy]
            grouped = sub.groupby('alpha')[metric_col].mean().sort_index()
            pct_change = (1 - grouped / baseline_val) * 100
            ax.plot(pct_change.index, pct_change.values, marker=marker,
                   linewidth=2, markersize=8, label=strategy)

        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('Alpha', fontsize=12)
        ax.set_ylabel('% Change vs MAXN', fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'energy_savings_vs_maxn.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: energy_savings_vs_maxn.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize E2E Benchmark (Fine-Grained)')
    parser.add_argument('--input', type=str, default=None, help='CSV path')
    args = parser.parse_args()

    if args.input:
        csv_path = Path(args.input)
    else:
        csv_path = find_latest_csv('e2e_benchmark_finegrained_*.csv')

    logger.info(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[(df['output_tokens'] > 0) & (df['energy_per_token_j'] > 0)]
    logger.info(f"Valid rows: {len(df)}")

    out_dir = Path('figures/e2e_benchmark_finegrained')
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto(df, out_dir)
    plot_alpha_effect(df, out_dir)
    plot_strategy_comparison(df, out_dir)
    plot_workload_heatmap(df, out_dir)
    plot_energy_savings(df, out_dir)

    logger.info(f"\nAll figures saved to: {out_dir}/")


if __name__ == '__main__':
    main()
