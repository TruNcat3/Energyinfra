#!/usr/bin/env python3
"""
Visualize Phase-Switching E2E Benchmark Results

Generates:
  1. Phase-switching vs single config vs baselines (E/token)
  2. Phase energy breakdown (prefill vs decode bar chart)
  3. Extended workload comparison (decode length effect)
  4. Switch overhead analysis
  5. Weak baseline comparison

Usage:
    python src/visualization/visualize_phase_switch_e2e.py
    python src/visualization/visualize_phase_switch_e2e.py --input data/e2e_benchmark/phase_switch_e2e_*.csv
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


def plot_phase_switch_comparison(df: pd.DataFrame, out_dir: Path):
    """Phase-switching vs single config vs baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: E/token by strategy
    ax = axes[0]
    strategies_to_plot = ['true_phase_switch', 'single_config']
    colors = {'true_phase_switch': 'coral', 'single_config': 'steelblue'}

    for strategy in strategies_to_plot:
        sub = df[df['strategy'] == strategy]
        if sub.empty:
            continue
        grouped = sub.groupby('alpha')['energy_per_token_j'].agg(['mean', 'std']).sort_index()
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o' if strategy == 'true_phase_switch' else 's',
                   capsize=5, linewidth=2, markersize=8,
                   color=colors.get(strategy, 'gray'), label=strategy)

    # Add baseline lines
    for bl_type in ['baseline', 'weak_baseline']:
        bls = df[df['strategy'] == bl_type]
        for bl_name in sorted(bls['baseline_name'].unique()):
            bl = bls[bls['baseline_name'] == bl_name]
            ax.axhline(bl['energy_per_token_j'].mean(), linestyle='--', alpha=0.5,
                       label=f'{bl_name} ({bl_type})')

    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('Energy per Token (J)', fontsize=12)
    ax.set_title('E/token: Phase-Switch vs Single Config', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: TPOT by strategy
    ax = axes[1]
    for strategy in strategies_to_plot:
        sub = df[df['strategy'] == strategy]
        if sub.empty:
            continue
        grouped = sub.groupby('alpha')['tpot_ms'].agg(['mean', 'std']).sort_index()
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o' if strategy == 'true_phase_switch' else 's',
                   capsize=5, linewidth=2, markersize=8,
                   color=colors.get(strategy, 'gray'), label=strategy)

    for bl_type in ['baseline', 'weak_baseline']:
        bls = df[df['strategy'] == bl_type]
        for bl_name in sorted(bls['baseline_name'].unique()):
            bl = bls[bls['baseline_name'] == bl_name]
            ax.axhline(bl['tpot_ms'].mean(), linestyle='--', alpha=0.5,
                       label=f'{bl_name}')

    ax.set_xlabel('Alpha', fontsize=12)
    ax.set_ylabel('TPOT (ms)', fontsize=12)
    ax.set_title('TPOT: Phase-Switch vs Single Config', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Phase-Switching DVFS: Energy and Latency Trade-off', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'phase_switch_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: phase_switch_comparison.png")


def plot_phase_energy_breakdown(df: pd.DataFrame, out_dir: Path):
    """Prefill vs Decode energy breakdown for phase-switching runs."""
    ps = df[df['strategy'] == 'true_phase_switch']
    if ps.empty or 'prefill_energy_j' not in ps.columns:
        logger.warning("No phase-split energy data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Stacked bar by workload
    ax = axes[0]
    workloads = sorted(ps['workload'].unique())
    alphas = sorted(ps['alpha'].unique())
    x = np.arange(len(workloads))
    width = 0.8 / len(alphas)

    for i, alpha in enumerate(alphas):
        sub = ps[ps['alpha'] == alpha]
        prefill_e = []
        decode_e = []
        for wl in workloads:
            wl_sub = sub[sub['workload'] == wl]
            if not wl_sub.empty:
                prefill_e.append(wl_sub['prefill_energy_j'].mean())
                decode_e.append(wl_sub['decode_energy_j'].mean())
            else:
                prefill_e.append(0)
                decode_e.append(0)

        offset = (i - len(alphas)/2 + 0.5) * width
        ax.bar(x + offset, prefill_e, width, label=f'Prefill α={alpha}',
               alpha=0.8, color=plt.cm.Blues(0.3 + 0.2*i))
        ax.bar(x + offset, decode_e, width, bottom=prefill_e,
               label=f'Decode α={alpha}',
               alpha=0.8, color=plt.cm.Reds(0.3 + 0.2*i))

    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Energy (J)', fontsize=12)
    ax.set_title('Energy Breakdown by Workload', fontsize=13)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Right: Prefill energy percentage vs output length
    ax = axes[1]
    for alpha in alphas:
        sub = ps[ps['alpha'] == alpha]
        pf_pct = []
        ols = []
        for wl in workloads:
            wl_sub = sub[sub['workload'] == wl]
            if not wl_sub.empty:
                pf = wl_sub['prefill_energy_j'].mean()
                dc = wl_sub['decode_energy_j'].mean()
                total = pf + dc
                if total > 0:
                    pf_pct.append(pf / total * 100)
                else:
                    pf_pct.append(0)
                ols.append(wl_sub['output_length'].iloc[0])
            else:
                pf_pct.append(0)
                ols.append(0)

        ax.scatter(ols, pf_pct, marker='o', s=80, label=f'α={alpha}')

    ax.set_xlabel('Output Length (tokens)', fontsize=12)
    ax.set_ylabel('Prefill Energy (%)', fontsize=12)
    ax.set_title('Prefill Energy Fraction vs Output Length', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'phase_energy_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: phase_energy_breakdown.png")


def plot_decode_length_effect(df: pd.DataFrame, out_dir: Path):
    """E/token vs output length showing decode scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect data for different output lengths
    for strategy, marker, color in [
        ('true_phase_switch', 'o', 'coral'),
        ('single_config', 's', 'steelblue'),
    ]:
        sub = df[df['strategy'] == strategy]
        if sub.empty:
            continue

        # Average across alphas for each output length
        for alpha in sorted(sub['alpha'].unique()):
            a_sub = sub[sub['alpha'] == alpha]
            grouped = a_sub.groupby('output_length').agg({
                'energy_per_token_j': 'mean',
                'tpot_ms': 'mean',
            }).sort_index()

            axes[0].plot(grouped.index, grouped['energy_per_token_j'],
                        marker=marker, linewidth=2, alpha=0.7,
                        label=f'{strategy} α={alpha}')
            axes[1].plot(grouped.index, grouped['tpot_ms'],
                        marker=marker, linewidth=2, alpha=0.7,
                        label=f'{strategy} α={alpha}')

    # Add baselines
    for bl_type in ['baseline', 'weak_baseline']:
        bls = df[df['strategy'] == bl_type]
        for bl_name in sorted(bls['baseline_name'].unique()):
            bl = bls[bls['baseline_name'] == bl_name]
            grouped = bl.groupby('output_length').agg({
                'energy_per_token_j': 'mean',
                'tpot_ms': 'mean',
            }).sort_index()
            if not grouped.empty:
                axes[0].plot(grouped.index, grouped['energy_per_token_j'],
                            linestyle='--', alpha=0.5, label=f'{bl_name}')
                axes[1].plot(grouped.index, grouped['tpot_ms'],
                            linestyle='--', alpha=0.5, label=f'{bl_name}')

    axes[0].set_xlabel('Output Length (tokens)', fontsize=12)
    axes[0].set_ylabel('Energy per Token (J)', fontsize=12)
    axes[0].set_title('E/token vs Decode Length', fontsize=13)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Output Length (tokens)', fontsize=12)
    axes[1].set_ylabel('TPOT (ms)', fontsize=12)
    axes[1].set_title('TPOT vs Decode Length', fontsize=13)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Effect of Decode Length on Energy and Latency', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'decode_length_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: decode_length_effect.png")


def plot_switch_overhead(df: pd.DataFrame, out_dir: Path):
    """Switch overhead analysis."""
    ps = df[df['strategy'] == 'true_phase_switch']
    if ps.empty or 'switch_overhead_ms' not in ps.columns:
        logger.warning("No switch overhead data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Switch overhead by workload
    ax = axes[0]
    workloads = sorted(ps['workload'].unique())
    overheads = [ps[ps['workload'] == wl]['switch_overhead_ms'].mean() for wl in workloads]
    ttfts = [ps[ps['workload'] == wl]['ttft_ms'].mean() for wl in workloads]
    overhead_pct = [o/t*100 if t > 0 else 0 for o, t in zip(overheads, ttfts)]

    ax.bar(range(len(workloads)), overheads, color='coral', alpha=0.8)
    ax.set_xticks(range(len(workloads)))
    ax.set_xticklabels(workloads, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Switch Overhead (ms)', fontsize=12)
    ax.set_title('Frequency Switch Overhead', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (ov, pct) in enumerate(zip(overheads, overhead_pct)):
        ax.text(i, ov + 5, f'{pct:.1f}%\nof TTFT', ha='center', fontsize=8)

    # Right: Switch overhead vs prefill config frequency
    ax = axes[1]
    if 'prefill_gpu' in ps.columns:
        for alpha in sorted(ps['alpha'].unique()):
            sub = ps[ps['alpha'] == alpha]
            grouped = sub.groupby('prefill_gpu')['switch_overhead_ms'].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', capsize=5, label=f'α={alpha}')

    ax.set_xlabel('Prefill GPU Freq (MHz)', fontsize=12)
    ax.set_ylabel('Switch Overhead (ms)', fontsize=12)
    ax.set_title('Switch Overhead vs Prefill Frequency', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'switch_overhead.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: switch_overhead.png")


def plot_weak_baseline_comparison(df: pd.DataFrame, out_dir: Path):
    """Weak baselines vs optimized configs."""
    fig, ax = plt.subplots(figsize=(12, 6))

    all_strategies = df.groupby(['strategy', 'baseline_name']).agg({
        'energy_per_token_j': 'mean',
        'tpot_ms': 'mean',
        'tokens_per_second': 'mean',
    }).reset_index()

    labels = []
    epts = []
    tpots = []
    colors = []
    markers = []

    # Phase-switching
    for alpha in sorted(df[df['strategy'] == 'true_phase_switch']['alpha'].unique()):
        sub = df[(df['strategy'] == 'true_phase_switch') & (df['alpha'] == alpha)]
        if not sub.empty:
            labels.append(f'PS α={alpha}')
            epts.append(sub['energy_per_token_j'].mean())
            tpots.append(sub['tpot_ms'].mean())
            colors.append('coral')
            markers.append('o')

    # Single config
    for alpha in sorted(df[df['strategy'] == 'single_config']['alpha'].unique()):
        sub = df[(df['strategy'] == 'single_config') & (df['alpha'] == alpha)]
        if not sub.empty:
            labels.append(f'SC α={alpha}')
            epts.append(sub['energy_per_token_j'].mean())
            tpots.append(sub['tpot_ms'].mean())
            colors.append('steelblue')
            markers.append('s')

    # Strong baselines
    for bl_name in sorted(df[df['strategy'] == 'baseline']['baseline_name'].unique()):
        sub = df[(df['strategy'] == 'baseline') & (df['baseline_name'] == bl_name)]
        if not sub.empty:
            labels.append(bl_name)
            epts.append(sub['energy_per_token_j'].mean())
            tpots.append(sub['tpot_ms'].mean())
            colors.append('green')
            markers.append('D')

    # Weak baselines
    for bl_name in sorted(df[df['strategy'] == 'weak_baseline']['baseline_name'].unique()):
        sub = df[(df['strategy'] == 'weak_baseline') & (df['baseline_name'] == bl_name)]
        if not sub.empty:
            labels.append(f'WEAK: {bl_name}')
            epts.append(sub['energy_per_token_j'].mean())
            tpots.append(sub['tpot_ms'].mean())
            colors.append('red')
            markers.append('X')

    for i, (label, ept, tpot, c, m) in enumerate(zip(labels, epts, tpots, colors, markers)):
        ax.scatter(ept, tpot, s=100, c=c, marker=m, zorder=5)
        ax.annotate(label, (ept, tpot), textcoords="offset points",
                   xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Energy per Token (J)', fontsize=12)
    ax.set_ylabel('TPOT (ms)', fontsize=12)
    ax.set_title('All Strategies and Baselines: Energy vs Latency', fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'weak_baseline_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: weak_baseline_comparison.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize Phase-Switch E2E Benchmark')
    parser.add_argument('--input', type=str, default=None, help='CSV path')
    args = parser.parse_args()

    if args.input:
        csv_path = Path(args.input)
    else:
        csv_path = find_latest_csv('phase_switch_e2e_*.csv')

    logger.info(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    valid_mask = pd.Series([True] * len(df))
    if 'output_tokens' in df.columns:
        valid_mask &= (df['output_tokens'] > 0)
    if 'energy_per_token_j' in df.columns:
        valid_mask &= (df['energy_per_token_j'] > 0)
    df = df[valid_mask]
    logger.info(f"Valid rows: {len(df)}")

    out_dir = Path('figures/phase_switch_e2e')
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_phase_switch_comparison(df, out_dir)
    plot_phase_energy_breakdown(df, out_dir)
    plot_decode_length_effect(df, out_dir)
    plot_switch_overhead(df, out_dir)
    plot_weak_baseline_comparison(df, out_dir)

    logger.info(f"\nAll figures saved to: {out_dir}/")


if __name__ == '__main__':
    main()
