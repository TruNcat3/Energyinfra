#!/usr/bin/env python3
"""
Visualize Cap Profiling Results

Generates:
  1. Cap Pareto frontier (E/token vs TPOT per GPU cap)
  2. GPU cap effect on E/token and TPOT
  3. EMC cap effect
  4. Cap vs dynamic baseline comparison
  5. Temperature effect by cap level
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
    files = sorted(Path('data/cap_profiling').glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {pattern}")
    return files[-1]


def plot_cap_pareto(df: pd.DataFrame, out_dir: Path):
    """Pareto frontier: E/token vs TPOT, colored by GPU cap."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cap_data = df[df['control_mode'] == 'cap']
    dyn_data = df[df['control_mode'] == 'dynamic']

    colors = {612: 'blue', 816: 'green', 1020: 'orange', 1300: 'red'}

    for gpu_cap in sorted(cap_data['gpu_cap_mhz'].unique()):
        sub = cap_data[cap_data['gpu_cap_mhz'] == gpu_cap]
        grouped = sub.groupby('workload').agg({
            'energy_per_token_j': 'mean',
            'tpot_ms': 'mean',
        })
        ax.scatter(grouped['energy_per_token_j'], grouped['tpot_ms'],
                  s=80, color=colors.get(gpu_cap, 'gray'),
                  label=f'GPU cap {gpu_cap}', alpha=0.8, marker='o')

    if not dyn_data.empty:
        dyn_grouped = dyn_data.groupby('workload').agg({
            'energy_per_token_j': 'mean',
            'tpot_ms': 'mean',
        })
        ax.scatter(dyn_grouped['energy_per_token_j'], dyn_grouped['tpot_ms'],
                  s=200, marker='X', color='black', edgecolors='black',
                  linewidths=2, label='dynamic (no cap)', zorder=5)

    ax.set_xlabel('Energy per Token (J)', fontsize=12)
    ax.set_ylabel('TPOT (ms)', fontsize=12)
    ax.set_title('Cap Pareto: Energy vs Latency', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'cap_pareto.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: cap_pareto.png")


def plot_gpu_cap_effect(df: pd.DataFrame, out_dir: Path):
    """GPU cap effect on E/token and TPOT (averaged across EMC caps)."""
    cap_data = df[df['control_mode'] == 'cap']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # E/token
    ax = axes[0]
    for wl in sorted(cap_data['workload'].unique()):
        sub = cap_data[cap_data['workload'] == wl]
        grouped = sub.groupby('gpu_cap_mhz')['energy_per_token_j'].agg(['mean', 'std']).sort_index()
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', capsize=3, linewidth=1.5, label=wl, alpha=0.8)

    dyn = df[df['control_mode'] == 'dynamic']
    if not dyn.empty:
        ax.axhline(dyn['energy_per_token_j'].mean(), linestyle='--', color='black',
                   alpha=0.5, label='dynamic avg')

    ax.set_xlabel('GPU Cap (MHz)', fontsize=12)
    ax.set_ylabel('Energy per Token (J)', fontsize=12)
    ax.set_title('GPU Cap → E/token', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # TPOT
    ax = axes[1]
    for wl in sorted(cap_data['workload'].unique()):
        sub = cap_data[cap_data['workload'] == wl]
        grouped = sub.groupby('gpu_cap_mhz')['tpot_ms'].agg(['mean', 'std']).sort_index()
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', capsize=3, linewidth=1.5, label=wl, alpha=0.8)

    if not dyn.empty:
        ax.axhline(dyn['tpot_ms'].mean(), linestyle='--', color='black',
                   alpha=0.5, label='dynamic avg')

    ax.set_xlabel('GPU Cap (MHz)', fontsize=12)
    ax.set_ylabel('TPOT (ms)', fontsize=12)
    ax.set_title('GPU Cap → TPOT', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('GPU Frequency Cap Effect', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'gpu_cap_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: gpu_cap_effect.png")


def plot_cap_vs_dynamic(df: pd.DataFrame, out_dir: Path):
    """Cap configs vs dynamic baseline savings."""
    cap_data = df[df['control_mode'] == 'cap']
    dyn = df[df['control_mode'] == 'dynamic']

    if dyn.empty:
        logger.warning("No dynamic baseline data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for wl in sorted(cap_data['workload'].unique()):
        cap_sub = cap_data[cap_data['workload'] == wl]
        dyn_sub = dyn[dyn['workload'] == wl]
        if dyn_sub.empty:
            continue
        dyn_ept = dyn_sub['energy_per_token_j'].mean()
        dyn_tpot = dyn_sub['tpot_ms'].mean()

        # Group by GPU cap
        for ax, metric, baseline, ylabel in [
            (axes[0], 'energy_per_token_j', dyn_ept, 'E/token savings vs dynamic (%)'),
            (axes[1], 'tpot_ms', dyn_tpot, 'TPOT change vs dynamic (%)'),
        ]:
            grouped = cap_sub.groupby('gpu_cap_mhz')[metric].mean().sort_index()
            pct_change = (1 - grouped / baseline) * 100
            ax.plot(pct_change.index, pct_change.values, marker='o', linewidth=1.5,
                   label=wl, alpha=0.8)

    for ax, ylabel in zip(axes, ['E/token savings vs dynamic (%)',
                                  'TPOT change vs dynamic (%)']):
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel('GPU Cap (MHz)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cap vs Dynamic Baseline', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / 'cap_vs_dynamic.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: cap_vs_dynamic.png")


def plot_temperature_effect(df: pd.DataFrame, out_dir: Path):
    """Temperature by cap level."""
    cap_data = df[df['control_mode'] == 'cap']
    if 'temperature_end_c' not in cap_data.columns:
        logger.warning("No temperature data")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for wl in sorted(cap_data['workload'].unique()):
        sub = cap_data[cap_data['workload'] == wl]
        grouped = sub.groupby('gpu_cap_mhz')['temperature_end_c'].agg(['mean', 'std']).sort_index()
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                   marker='o', capsize=3, linewidth=1.5, label=wl)

    dyn = df[df['control_mode'] == 'dynamic']
    if not dyn.empty and 'temperature_end_c' in dyn.columns:
        ax.axhline(dyn['temperature_end_c'].mean(), linestyle='--',
                   color='black', alpha=0.5, label='dynamic avg')

    ax.set_xlabel('GPU Cap (MHz)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title('Temperature by GPU Cap Level', fontsize=13)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / 'temperature_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved: temperature_effect.png")


def main():
    parser = argparse.ArgumentParser(description='Visualize Cap Profiling')
    parser.add_argument('--input', type=str, default=None, help='CSV path')
    args = parser.parse_args()

    if args.input:
        csv_path = Path(args.input)
    else:
        csv_path = find_latest_csv('cap_profiling_*.csv')

    logger.info(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    valid = df[(df['output_tokens'] > 0) & (df['energy_per_token_j'] > 0)]
    logger.info(f"Valid rows: {len(valid)}")

    out_dir = Path('figures/cap_profiling')
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cap_pareto(valid, out_dir)
    plot_gpu_cap_effect(valid, out_dir)
    plot_cap_vs_dynamic(valid, out_dir)
    plot_temperature_effect(valid, out_dir)

    logger.info(f"\nAll figures saved to: {out_dir}/")


if __name__ == '__main__':
    main()
