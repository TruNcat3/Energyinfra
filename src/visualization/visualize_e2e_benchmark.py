#!/usr/bin/env python3
"""
Visualize End-to-End Multi-Objective DVFS Benchmark

Generates 5 plots:
  1. Pareto front (E/token vs TPOT) with baselines
  2. Alpha effect on metrics
  3. Phase-Aware vs Single Config
  4. Measured vs Predicted (rate table validation)
  5. Workload breakdown heatmap
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional

matplotlib.use('Agg')

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))


def plot_pareto_front(df: pd.DataFrame, output_dir: Path):
    """Core plot: E/token vs TPOT, each alpha a point, baselines marked."""
    fig, ax = plt.subplots(figsize=(10, 7))

    weighted = df[df['run_type'] == 'weighted']
    baselines = df[df['run_type'] == 'baseline']

    # Plot each strategy with different markers
    markers = {'single': 'o', 'phase_aware': 's'}
    colors = plt.cm.viridis(np.linspace(0, 1, len(weighted['alpha'].unique())))

    for strategy in ['single', 'phase_aware']:
        strat = weighted[weighted['strategy'] == strategy]
        if strat.empty:
            continue

        for i, alpha in enumerate(sorted(strat['alpha'].unique())):
            a = strat[strat['alpha'] == alpha]
            ept = a['energy_per_token_j'].mean()
            tpot = a['tpot_ms'].mean()
            ept_std = a['energy_per_token_j'].std()
            tpot_std = a['tpot_ms'].std()

            ax.errorbar(tpot, ept,
                       xerr=tpot_std, yerr=ept_std,
                       fmt=markers[strategy], color=colors[i],
                       markersize=10, capsize=3,
                       label=f'α={alpha:.1f}' if strategy == 'single' else None)

    # Baselines
    bl_markers = {'MAXN': 'X', 'Jetson_30W': 'D', 'Energy_Min': '^'}
    for bl_name, bl_data in baselines.groupby('strategy'):
        if bl_name not in bl_markers:
            continue
        ept = bl_data['energy_per_token_j'].mean()
        tpot = bl_data['tpot_ms'].mean()
        ax.scatter(tpot, ept, marker=bl_markers[bl_name], s=150,
                  c='red', zorder=5, label=bl_name, edgecolors='black', linewidths=1)

    ax.set_xlabel('TPOT (ms) — Latency', fontsize=12)
    ax.set_ylabel('Energy/Token (J) — Energy', fontsize=12)
    ax.set_title('Pareto Front: Energy-Latency Trade-off', fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'pareto_front.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_alpha_effect(df: pd.DataFrame, output_dir: Path):
    """Bar chart: metrics across alpha values."""
    weighted = df[df['run_type'] == 'weighted']
    if weighted.empty:
        return

    metrics = [
        ('energy_per_token_j', 'E/token (J)'),
        ('tpot_ms', 'TPOT (ms)'),
        ('ttft_ms', 'TTFT (ms)'),
        ('avg_power_w', 'Power (W)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    alphas = sorted(weighted['alpha'].unique())
    x = np.arange(len(alphas))
    width = 0.35

    for idx, (col, ylabel) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]

        for j, strategy in enumerate(['single', 'phase_aware']):
            strat = weighted[weighted['strategy'] == strategy]
            vals = [strat[strat['alpha'] == a][col].mean() for a in alphas]
            ax.bar(x + j * width - width / 2, vals, width,
                  label=strategy, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{a:.1f}' for a in alphas])
        ax.set_xlabel('Alpha')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by Alpha')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = output_dir / 'alpha_effect.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_phase_vs_single(df: pd.DataFrame, output_dir: Path):
    """Compare phase_aware vs single at each alpha."""
    weighted = df[df['run_type'] == 'weighted']
    if weighted.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (col, ylabel) in enumerate([
        ('energy_per_token_j', 'E/token (J)'),
        ('tpot_ms', 'TPOT (ms)'),
        ('tokens_per_second', 'Throughput (tok/s)')
    ]):
        ax = axes[idx]
        for strategy in ['single', 'phase_aware']:
            strat = weighted[weighted['strategy'] == strategy]
            grouped = strat.groupby('alpha')[col].mean()
            ax.plot(grouped.index, grouped.values, 'o-', label=strategy, markersize=6)

        ax.set_xlabel('Alpha')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel}: Phase-Aware vs Single')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'phase_vs_single.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_measured_vs_predicted(df: pd.DataFrame, output_dir: Path):
    """Scatter: rate table predictions vs real measurements."""
    weighted = df[df['run_type'] == 'weighted']
    if weighted.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (pred_col, meas_col, label) in enumerate([
        ('pred_tpot_ms', 'tpot_ms', 'TPOT (ms)'),
        ('pred_energy_per_token_j', 'energy_per_token_j', 'E/token (J)'),
    ]):
        ax = axes[idx]
        valid = weighted[(weighted[pred_col] > 0) & (weighted[meas_col] > 0)]
        if valid.empty:
            continue

        # Per-config averages
        grouped = valid.groupby('config_name').agg({
            pred_col: 'mean', meas_col: 'mean'
        }).reset_index()

        ax.scatter(grouped[pred_col], grouped[meas_col], s=60, alpha=0.8)

        # Perfect prediction line
        all_vals = np.concatenate([grouped[pred_col], grouped[meas_col]])
        lo, hi = all_vals.min(), all_vals.max()
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='Perfect prediction')

        ax.set_xlabel(f'Predicted {label}')
        ax.set_ylabel(f'Measured {label}')
        ax.set_title(f'Rate Table Validation: {label}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # R² score
        from sklearn.metrics import r2_score
        r2 = r2_score(grouped[meas_col], grouped[pred_col])
        ax.text(0.05, 0.95, f'R²={r2:.3f}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top')

    plt.tight_layout()
    path = output_dir / 'measured_vs_predicted.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_workload_breakdown(df: pd.DataFrame, output_dir: Path):
    """Heatmap: which config is selected at each alpha × workload."""
    weighted = df[df['run_type'] == 'weighted']
    if weighted.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # For each (alpha, workload), get the selected config's GPU freq
    pivot = weighted.groupby(['alpha', 'workload'])['actual_gpu_mhz'].mean().reset_index()
    pivot_table = pivot.pivot(index='alpha', columns='workload', values='actual_gpu_mhz')

    im = ax.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_xticklabels(pivot_table.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels([f'{a:.1f}' for a in pivot_table.index])

    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            val = pivot_table.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=9)

    ax.set_xlabel('Workload')
    ax.set_ylabel('Alpha')
    ax.set_title('Selected GPU Frequency (MHz) by Alpha and Workload')
    plt.colorbar(im, label='GPU MHz')
    plt.tight_layout()

    path = output_dir / 'workload_breakdown.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    data_dir = Path('data/e2e_benchmark')
    csv_files = sorted(data_dir.glob('e2e_benchmark_*.csv'))

    if not csv_files:
        print("No e2e benchmark data found. Run run_e2e_benchmark.py first.")
        return 1

    latest = csv_files[-1]
    print(f"Loading: {latest.name}")
    df = pd.read_csv(latest)

    output_dir = Path('figures/e2e_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")
    plot_pareto_front(df, output_dir)
    plot_alpha_effect(df, output_dir)
    plot_phase_vs_single(df, output_dir)
    plot_measured_vs_predicted(df, output_dir)
    plot_workload_breakdown(df, output_dir)

    print(f"\nAll figures saved to {output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
