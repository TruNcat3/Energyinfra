#!/usr/bin/env python3
"""
Visualize Phase-Aware DVFS Experiment Results
Generates comparison charts for the validation experiment.
"""

import sys
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent.parent.parent / 'figures' / 'phase_aware_experiment'
DATA_DIR = Path(__file__).parent.parent.parent / 'data' / 'phase_aware_experiment'

STRATEGY_LABELS = {
    'default': 'Default (mid)',
    'max_perf': 'Max Performance',
    'energy_efficient': 'Energy Efficient',
    'phase_aware': 'Phase-Aware (Ours)',
    'oracle': 'Oracle',
}
STRATEGY_COLORS = {
    'default': '#808080',
    'max_perf': '#d62728',
    'energy_efficient': '#2ca02c',
    'phase_aware': '#1f77b4',
    'oracle': '#ff7f0e',
}


def load_latest_results() -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA_DIR / 'detailed_results_*.csv')))
    if not files:
        raise FileNotFoundError("No experiment results found")
    latest = files[-1]
    logger.info(f"Loading: {latest}")
    return pd.read_csv(latest)


def plot_energy_comparison(df: pd.DataFrame):
    """Bar chart: energy per token by strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Energy per output token
    df['energy_per_token'] = df['total_energy_j'] / df['output_len']

    strategies = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'oracle']
    means, stds, labels = [], [], []
    for s in strategies:
        sub = df[df['strategy'] == s]
        means.append(sub['energy_per_token'].mean())
        stds.append(sub['energy_per_token'].std())
        labels.append(STRATEGY_LABELS[s])

    x = np.arange(len(strategies))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[STRATEGY_COLORS[s] for s in strategies],
                  edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Energy per Token (J/tok)', fontsize=12)
    ax.set_title('Energy Efficiency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / 'energy_comparison.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_latency_comparison(df: pd.DataFrame):
    """Grouped bar: TTFT and TPOT by strategy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    strategies = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'oracle']

    # TTFT
    ttft_means, ttft_stds = [], []
    for s in strategies:
        sub = df[df['strategy'] == s]
        ttft_means.append(sub['ttft_ms'].mean())
        ttft_stds.append(sub['ttft_ms'].std())

    x = np.arange(len(strategies))
    ax1.bar(x, ttft_means, yerr=ttft_stds, capsize=5,
            color=[STRATEGY_COLORS[s] for s in strategies],
            edgecolor='black', linewidth=0.5, alpha=0.85)
    ax1.set_ylabel('TTFT (ms)', fontsize=12)
    ax1.set_title('Time to First Token', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=20, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # TPOT
    tpot_means, tpot_stds = [], []
    for s in strategies:
        sub = df[df['strategy'] == s]
        tpot_means.append(sub['tpot_ms'].mean())
        tpot_stds.append(sub['tpot_ms'].std())

    ax2.bar(x, tpot_means, yerr=tpot_stds, capsize=5,
            color=[STRATEGY_COLORS[s] for s in strategies],
            edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.set_ylabel('TPOT (ms)', fontsize=12)
    ax2.set_title('Time Per Output Token', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=20, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / 'latency_comparison.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_workload_breakdown(df: pd.DataFrame):
    """Grouped bar: energy by workload and strategy."""
    fig, ax = plt.subplots(figsize=(14, 6))

    strategies = ['default', 'max_perf', 'phase_aware']
    workloads = sorted(df['workload_name'].unique())

    df['energy_per_token'] = df['total_energy_j'] / df['output_len']

    n_strats = len(strategies)
    n_wl = len(workloads)
    bar_width = 0.25
    x = np.arange(n_wl)

    for i, s in enumerate(strategies):
        means, stds = [], []
        for wl in workloads:
            sub = df[(df['strategy'] == s) & (df['workload_name'] == wl)]
            means.append(sub['energy_per_token'].mean())
            stds.append(sub['energy_per_token'].std())
        offset = (i - n_strats / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds, capsize=3,
               label=STRATEGY_LABELS[s], color=STRATEGY_COLORS[s],
               edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Energy per Token (J/tok)', fontsize=12)
    ax.set_title('Energy Efficiency by Workload', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    wl_labels = [wl.replace('_', '\n') for wl in workloads]
    ax.set_xticklabels(wl_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / 'workload_breakdown.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_slo_compliance(df: pd.DataFrame):
    """Stacked bar: SLO compliance rate."""
    fig, ax = plt.subplots(figsize=(10, 6))

    strategies = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'oracle']
    met_rates, violated_rates = [], []
    for s in strategies:
        sub = df[df['strategy'] == s]
        met = sub['slo_met'].mean()
        met_rates.append(met * 100)
        violated_rates.append((1 - met) * 100)

    x = np.arange(len(strategies))
    bars1 = ax.bar(x, met_rates, color='#2ca02c', alpha=0.8, label='SLO Met')
    bars2 = ax.bar(x, violated_rates, bottom=met_rates, color='#d62728', alpha=0.8, label='SLO Violated')

    for bar, rate in zip(bars1, met_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'{rate:.0f}%', ha='center', va='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('SLO Compliance Rate', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([STRATEGY_LABELS[s] for s in strategies], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / 'slo_compliance.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def plot_pareto_tradeoff(df: pd.DataFrame):
    """Scatter plot: energy vs latency tradeoff."""
    fig, ax = plt.subplots(figsize=(10, 7))

    strategies = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'oracle']
    for s in strategies:
        sub = df[df['strategy'] == s]
        avg_e = (sub['total_energy_j'] / sub['output_len']).mean()
        avg_ttft = sub['ttft_ms'].mean()
        avg_tpot = sub['tpot_ms'].mean()
        ax.scatter(avg_e, avg_ttft, s=200, color=STRATEGY_COLORS[s],
                   label=STRATEGY_LABELS[s], edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(STRATEGY_LABELS[s], (avg_e, avg_ttft),
                     textcoords="offset points", xytext=(10, 8), fontsize=9)

    ax.set_xlabel('Energy per Token (J/tok)', fontsize=12)
    ax.set_ylabel('TTFT (ms)', fontsize=12)
    ax.set_title('Energy-Latency Tradeoff', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / 'pareto_tradeoff.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11})

    df = load_latest_results()
    logger.info(f"Loaded {len(df)} results rows")

    plot_energy_comparison(df)
    plot_latency_comparison(df)
    plot_workload_breakdown(df)
    plot_slo_compliance(df)
    plot_pareto_tradeoff(df)

    logger.info("All visualizations generated!")
    logger.info(f"Output: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
