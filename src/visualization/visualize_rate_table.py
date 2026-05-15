#!/usr/bin/env python3
"""
Visualize Energy Rate Table Analysis

Generates plots for:
  1. Optimal GPU freq by workload size (heatmap)
  2. E/token scaling with prompt_length and output_length
  3. GPU freq effect by phase (bar chart)
  4. Power domain breakdown by phase
  5. DVFS rule summary (which config wins where)
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
from typing import Optional

matplotlib.use('Agg')

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))


def plot_optimal_gpu_heatmap(dvfs_rules: list, output_dir: Path):
    """Plot optimal GPU frequency as heatmap for each phase."""
    df = pd.DataFrame(dvfs_rules)

    for phase in df['phase'].unique():
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            continue

        pivot = phase_df.pivot_table(
            index='prompt_length', columns='output_length',
            values='optimal_gpu_freq', aggfunc='first'
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(int(i)) for i in pivot.index])

        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{int(val)}', ha='center', va='center', fontsize=9)

        ax.set_xlabel('Output Length (tokens)')
        ax.set_ylabel('Prompt Length (tokens)')
        ax.set_title(f'Optimal GPU Frequency (MHz) - {phase} phase')
        plt.colorbar(im, label='GPU MHz')
        plt.tight_layout()

        path = output_dir / f'optimal_gpu_heatmap_{phase}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_energy_scaling(rate_table: pd.DataFrame, output_dir: Path):
    """Plot E/token scaling with prompt/output length."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, phase in enumerate(['mixed', 'decode']):
        ax = axes[idx]
        phase_df = rate_table[rate_table['phase'] == phase]
        if phase_df.empty:
            continue

        # Group by prompt_length, plot E/token vs output_length
        for pl in sorted(phase_df['prompt_length'].unique()):
            pl_data = phase_df[phase_df['prompt_length'] == pl]
            # Use best config for each (lowest E/token)
            best_per_ol = pl_data.loc[pl_data.groupby('output_length')['energy_per_token_j_mean'].idxmin()]
            best_per_ol = best_per_ol.sort_values('output_length')

            ax.plot(best_per_ol['output_length'], best_per_ol['energy_per_token_j_mean'],
                   'o-', label=f'prompt={int(pl)}', markersize=4)

        ax.set_xlabel('Output Length (tokens)')
        ax.set_ylabel('E/token (J)')
        ax.set_title(f'{phase} phase: E/token vs output length')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'energy_scaling.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_gpu_freq_comparison(rate_table: pd.DataFrame, output_dir: Path):
    """Bar chart: E/token and TPOT by GPU freq for each phase."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (metric, ylabel) in enumerate([
        ('energy_per_token_j_mean', 'E/token (J)'),
        ('tpot_ms_mean', 'TPOT (ms)')
    ]):
        ax = axes[idx]
        for phase_idx, phase in enumerate(['mixed', 'decode']):
            phase_df = rate_table[rate_table['phase'] == phase]
            if phase_df.empty:
                continue

            gpu_avg = phase_df.groupby('gpu_freq_mhz')[metric].mean()
            bars = ax.bar(
                [f'{int(g)}' for g in gpu_avg.index],
                gpu_avg.values,
                width=0.35,
                label=phase,
                alpha=0.8
            )

        ax.set_xlabel('GPU Frequency (MHz)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} by GPU Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = output_dir / 'gpu_freq_comparison.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_power_breakdown(rate_table: pd.DataFrame, output_dir: Path):
    """Power domain breakdown by phase and config."""
    fig, ax = plt.subplots(figsize=(12, 5))

    phases = ['mixed', 'decode']
    configs = sorted(rate_table['config_name'].unique())

    x = np.arange(len(configs))
    width = 0.35

    for i, phase in enumerate(phases):
        phase_df = rate_table[rate_table['phase'] == phase]
        if phase_df.empty:
            continue

        gpu_power = phase_df.groupby('config_name')['avg_gpu_soc_w_mean'].mean().reindex(configs, fill_value=0)
        cpu_power = phase_df.groupby('config_name')['avg_cpu_cv_w_mean'].mean().reindex(configs, fill_value=0)
        sys_power = phase_df.groupby('config_name')['avg_sys_5v0_w_mean'].mean().reindex(configs, fill_value=0)

        offset = width * i
        ax.bar(x + offset - width/2, gpu_power, width, label=f'{phase} GPU SoC', alpha=0.8)
        ax.bar(x + offset - width/2, cpu_power, width, bottom=gpu_power, label=f'{phase} CPU CV', alpha=0.8)
        ax.bar(x + offset - width/2, sys_power, width, bottom=gpu_power + cpu_power,
               label=f'{phase} SYS 5V0', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Power (W)')
    ax.set_title('Power Domain Breakdown by Config and Phase')
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = output_dir / 'power_breakdown.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_dvfs_rule_summary(dvfs_rules: list, output_dir: Path):
    """Summary of which config wins across all workload sizes."""
    df = pd.DataFrame(dvfs_rules)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, phase in enumerate(['prefill', 'mixed', 'decode']):
        ax = axes[idx]
        phase_df = df[df['phase'] == phase]
        if phase_df.empty:
            ax.set_title(f'{phase} (no data)')
            continue

        config_counts = phase_df['optimal_config'].value_counts()
        bars = ax.bar(range(len(config_counts)), config_counts.values, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(config_counts)))
        ax.set_xticklabels(config_counts.index, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Number of workloads')
        ax.set_title(f'{phase}: optimal config frequency')

        for bar, val in zip(bars, config_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(val), ha='center', fontsize=9)

    plt.tight_layout()
    path = output_dir / 'dvfs_rule_summary.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    # Find latest rate table and rules
    rate_dir = Path('data/rate_tables')
    rules_files = sorted(rate_dir.glob('dvfs_rules_*.json'))
    table_files = sorted(rate_dir.glob('real_rate_table_*.parquet'))

    if not rules_files or not table_files:
        print("No rate table data found. Run build_real_rate_table.py first.")
        return 1

    latest_rules = rules_files[-1]
    latest_table = table_files[-1]

    print(f"Loading: {latest_rules.name}")
    with open(latest_rules) as f:
        data = json.load(f)

    print(f"Loading: {latest_table.name}")
    rate_table = pd.read_parquet(latest_table)

    output_dir = Path('figures/rate_table')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating visualizations...")
    plot_dvfs_rule_summary(data['dvfs_rules'], output_dir)
    plot_optimal_gpu_heatmap(data['dvfs_rules'], output_dir)
    plot_energy_scaling(rate_table, output_dir)
    plot_gpu_freq_comparison(rate_table, output_dir)
    plot_power_breakdown(rate_table, output_dir)

    print(f"\nAll figures saved to {output_dir}/")
    return 0


if __name__ == '__main__':
    sys.exit(main())
