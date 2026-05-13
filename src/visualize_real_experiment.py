#!/usr/bin/env python3
"""
Real Model Experiment Visualization
Generates comparison charts from real llama.cpp experiment data.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import glob

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'axes.grid': True, 'grid.alpha': 0.3,
})

OUTPUT_DIR = Path('figures/real_model_experiment')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_latest_data():
    files = sorted(glob.glob('data/real_model_experiment/real_experiment_*.csv'))
    if not files:
        raise FileNotFoundError("No experiment data found")
    latest = files[-1]
    print(f"Loading: {latest}")
    return pd.read_csv(latest), latest


def plot_gpu_freq_effect(df):
    """Fig 1: Performance vs GPU frequency (averaged over CPU freqs and workloads)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    gpu_groups = df.groupby('gpu_freq_mhz').agg({
        'ttft_ms': 'mean', 'tpot_ms': 'mean', 'tokens_per_second': 'mean'
    }).sort_index()

    colors = ['#dc3545', '#fd7e14', '#198754', '#0d6efd']

    # TTFT
    ax = axes[0]
    ax.bar([f'{int(g)}' for g in gpu_groups.index], gpu_groups['ttft_ms'], color=colors)
    ax.set_xlabel('GPU Frequency (MHz)')
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('(a) TTFT vs GPU Freq')
    for i, v in enumerate(gpu_groups['ttft_ms']):
        ax.text(i, v + 20, f'{v:.0f}', ha='center', fontsize=9)

    # TPOT
    ax = axes[1]
    ax.bar([f'{int(g)}' for g in gpu_groups.index], gpu_groups['tpot_ms'], color=colors)
    ax.set_xlabel('GPU Frequency (MHz)')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('(b) TPOT vs GPU Freq')
    for i, v in enumerate(gpu_groups['tpot_ms']):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=9)

    # Throughput
    ax = axes[2]
    ax.bar([f'{int(g)}' for g in gpu_groups.index], gpu_groups['tokens_per_second'], color=colors)
    ax.set_xlabel('GPU Frequency (MHz)')
    ax.set_ylabel('Tokens/Second')
    ax.set_title('(c) Throughput vs GPU Freq')
    for i, v in enumerate(gpu_groups['tokens_per_second']):
        ax.text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=9)

    fig.suptitle('GPU Frequency Effect on Real Model Inference', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'gpu_freq_effect.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: gpu_freq_effect.png")


def plot_config_heatmap(df):
    """Fig 2: Heatmap of throughput across GPU x CPU configs."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax_idx, (metric, title, fmt) in enumerate([
        ('ttft_ms', 'TTFT (ms)', '.0f'),
        ('tpot_ms', 'TPOT (ms)', '.1f'),
        ('tokens_per_second', 'Throughput (tok/s)', '.1f'),
    ]):
        ax = axes[ax_idx]
        pivot = df.groupby(['gpu_freq_mhz', 'cpu_freq_mhz'])[metric].mean().unstack()
        pivot = pivot.sort_index(ascending=True)

        im = ax.imshow(pivot.values, cmap='RdYlGn' if metric == 'tokens_per_second' else 'RdYlGn_r',
                       aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{int(c)}' for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{int(r)}' for r in pivot.index])
        ax.set_xlabel('CPU Frequency (MHz)')
        ax.set_ylabel('GPU Frequency (MHz)')
        ax.set_title(title)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f'{val:{fmt}}', ha='center', va='center', fontsize=9,
                       color='white' if (metric != 'tokens_per_second' and val > pivot.values.mean()) or
                              (metric == 'tokens_per_second' and val < pivot.values.mean()) else 'black')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Configuration Performance Heatmap (Real Model)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'config_heatmap.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: config_heatmap.png")


def plot_workload_breakdown(df):
    """Fig 3: Per-workload TTFT and TPOT across configs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    workloads = sorted(df['workload'].unique())
    configs = sorted(df['config_name'].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))

    # TTFT by workload
    ax = axes[0]
    x = np.arange(len(workloads))
    width = 0.8 / len(configs)
    for i, config in enumerate(configs):
        vals = []
        for wl in workloads:
            v = df[(df['workload'] == wl) & (df['config_name'] == config)]['ttft_ms'].mean()
            vals.append(v)
        ax.bar(x + i * width, vals, width, label=config, color=colors[i])
    ax.set_xticks(x + width * len(configs) / 2)
    ax.set_xticklabels(workloads, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('(a) TTFT by Workload and Config')
    ax.legend(fontsize=7, ncol=2)

    # TPOT by workload
    ax = axes[1]
    for i, config in enumerate(configs):
        vals = []
        for wl in workloads:
            v = df[(df['workload'] == wl) & (df['config_name'] == config)]['tpot_ms'].mean()
            vals.append(v)
        ax.bar(x + i * width, vals, width, label=config, color=colors[i])
    ax.set_xticks(x + width * len(configs) / 2)
    ax.set_xticklabels(workloads, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('(b) TPOT by Workload and Config')
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'workload_breakdown.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: workload_breakdown.png")


def plot_scaling_analysis(df):
    """Fig 4: Frequency scaling curves for TTFT and throughput."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cpu_freqs = sorted(df['cpu_freq_mhz'].unique())
    colors = ['#dc3545', '#198754', '#0d6efd']

    # TTFT scaling with GPU freq
    ax = axes[0]
    for i, cpu in enumerate(cpu_freqs):
        subset = df[df['cpu_freq_mhz'] == cpu].groupby('gpu_freq_mhz')['ttft_ms'].mean().sort_index()
        ax.plot(subset.index / 1000, subset.values, 'o-', label=f'CPU {int(cpu)}MHz',
               color=colors[i], linewidth=2, markersize=6)
    ax.set_xlabel('GPU Frequency (GHz)')
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('(a) TTFT Scaling')
    ax.legend()

    # Throughput scaling
    ax = axes[1]
    for i, cpu in enumerate(cpu_freqs):
        subset = df[df['cpu_freq_mhz'] == cpu].groupby('gpu_freq_mhz')['tokens_per_second'].mean().sort_index()
        ax.plot(subset.index / 1000, subset.values, 'o-', label=f'CPU {int(cpu)}MHz',
               color=colors[i], linewidth=2, markersize=6)
    ax.set_xlabel('GPU Frequency (GHz)')
    ax.set_ylabel('Tokens/Second')
    ax.set_title('(b) Throughput Scaling')
    ax.legend()

    fig.suptitle('Frequency Scaling Analysis (Real Model)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'scaling_analysis.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: scaling_analysis.png")


def plot_real_vs_synthetic_comparison(df):
    """Fig 5: Compare real model results with synthetic benchmark patterns."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Group by GPU freq, show throughput range
    gpu_groups = df.groupby('gpu_freq_mhz').agg({
        'tokens_per_second': ['mean', 'std'],
        'tpot_ms': ['mean', 'std'],
    })

    gpu_freqs = gpu_groups.index.values
    tps_mean = gpu_groups['tokens_per_second']['mean'].values
    tps_std = gpu_groups['tokens_per_second']['std'].values

    ax.errorbar(gpu_freqs, tps_mean, yerr=tps_std, fmt='o-', linewidth=2,
               markersize=8, capsize=5, label='Real Model (Phi-3-mini Q4)')

    # Normalize to show relative scaling
    base_tps = tps_mean[0]
    ax2 = ax.twinx()
    ax2.plot(gpu_freqs, tps_mean / base_tps * 100, 's--', color='red',
            alpha=0.6, label='Relative Scaling (%)')
    ax2.set_ylabel('Relative Throughput (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    ax.set_xlabel('GPU Frequency (MHz)')
    ax.set_ylabel('Tokens/Second')
    ax.set_title('Real Model Throughput vs GPU Frequency\n(All CPU freqs and workloads averaged)')
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'real_vs_synthetic.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: real_vs_synthetic.png")


if __name__ == '__main__':
    print("Generating real model experiment visualizations...")
    df, source = load_latest_data()
    total = len(df)
    print(f"Loaded {total} records from {source}")
    print(f"Configs: {df['config_name'].nunique()}, Workloads: {df['workload'].nunique()}")

    # Filter 0-token anomaly runs
    zero_mask = df['output_tokens'] == 0
    n_zero = zero_mask.sum()
    if n_zero > 0:
        print(f"\nFiltering {n_zero}/{total} zero-token anomaly runs "
              f"({n_zero/total*100:.1f}%)")
        for _, row in df[zero_mask].iterrows():
            print(f"  - {row['config_name']} {row['workload']} rep={row['repeat']} "
                  f"TTFT={row['ttft_ms']:.0f}ms")
        df = df[~zero_mask].copy()
        print(f"Remaining: {len(df)} valid runs")

    if len(df) == 0:
        print("ERROR: No valid data after filtering!")
        exit(1)

    plot_gpu_freq_effect(df)
    plot_config_heatmap(df)
    plot_workload_breakdown(df)
    plot_scaling_analysis(df)
    plot_real_vs_synthetic_comparison(df)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
