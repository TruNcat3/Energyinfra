#!/usr/bin/env python3
"""
Jetson Power Mode Comparison - Real Model Benchmark
Compares real inference performance across Jetson power configurations.

Config mapping (from 180-run real experiment):
  MAXN (满性能):         GPU1300 + CPU2201  — nvpmodel mode 0 全开
  30W-Default (默认):     GPU612  + CPU1497  — nvpmodel mode 2 (Jetson出厂默认)
  15W-LowPower (低功耗):  GPU306  + CPU1036  — nvpmodel mode 1
  GPU-Min/CPU-Max (我们的策略): GPU306 + CPU2201 — GPU最低频+CPU最高频
  Energy-Focus (能效优先): GPU306 + CPU1497  — 中等CPU+最低GPU
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

OUTPUT_DIR = Path('figures/power_mode_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 统一档位命名 (与 docs/实验文档/README.md 保持一致)
MODE_MAP = {
    'GPU1300_CPU2201': 'MAXN',
    'GPU918_CPU2201':  'GPU918/CPU-Max',
    'GPU612_CPU2201':  'GPU612/CPU-Max',
    'GPU306_CPU2201':  'GPU-Min/CPU-Max (Ours)',
    'GPU612_CPU1497':  '30W (Default)',
    'GPU306_CPU1497':  'Energy-Focus',
    'GPU306_CPU1036':  '15W',
    'GPU1300_CPU1036': 'GPU-Max/CPU-Min',
    'GPU612_CPU1036':  'GPU612/CPU-Min',
    'GPU918_CPU1036':  'GPU918/CPU-Min',
    'GPU918_CPU1497':  'GPU918/CPU-Mid',
    'GPU1300_CPU1497': 'GPU-Max/CPU-Mid',
}

# 核心对比组
CORE_MODES = [
    'MAXN',
    '30W (Default)',
    '15W',
    'GPU-Min/CPU-Max (Ours)',
]


def load_data():
    files = sorted(glob.glob('data/real_model_experiment/real_experiment_*.csv'))
    if not files:
        raise FileNotFoundError("No experiment data found")
    latest = files[-1]
    print(f"Loading: {latest}")
    df = pd.read_csv(latest)
    # Filter zero-token anomalies
    df = df[df['output_tokens'] > 0].copy()
    # Add mode name
    df['mode'] = df['config_name'].map(MODE_MAP)
    return df, latest


def plot_throughput_comparison(df):
    """Bar chart: throughput across core modes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    core_df = df[df['mode'].isin(CORE_MODES)]
    mode_order = CORE_MODES
    colors = ['#dc3545', '#fd7e14', '#6c757d', '#198754']

    x = np.arange(len(mode_order))
    vals, errs, labels = [], [], []

    for mode in mode_order:
        mdata = core_df[core_df['mode'] == mode]['tokens_per_second']
        vals.append(mdata.mean())
        errs.append(mdata.std())
        labels.append(mode.split('(')[0].strip())

    bars = ax.bar(x, vals, yerr=errs, color=colors, capsize=5, width=0.6,
                  edgecolor='black', linewidth=0.5)

    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.15, f'{v:.1f}', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(' (', '\n(') for m in mode_order], fontsize=10)
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('Jetson Power Mode Throughput Comparison\n(Phi-3-mini Q4, 5 workloads × 3 repeats)')
    ax.set_ylim(0, max(vals) * 1.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'throughput_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: throughput_comparison.png")


def plot_tttp_tpot_scatter(df):
    """Scatter: TTFT vs TPOT for all configs, colored by CPU freq."""
    fig, ax = plt.subplots(figsize=(10, 7))

    cpu_colors = {1036: '#dc3545', 1497: '#fd7e14', 2201: '#198754'}
    gpu_markers = {306: 'o', 612: 's', 918: '^', 1300: 'D'}

    for cpu in sorted(df['cpu_freq_mhz'].unique()):
        for gpu in sorted(df['gpu_freq_mhz'].unique()):
            subset = df[(df['cpu_freq_mhz'] == cpu) & (df['gpu_freq_mhz'] == gpu)]
            if len(subset) == 0:
                continue
            ax.scatter(subset['ttft_ms'], subset['tpot_ms'],
                      c=cpu_colors[cpu], marker=gpu_markers[gpu],
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                      label=f'CPU {int(cpu)}MHz, GPU {int(gpu)}MHz')

    # Legend - deduplicate
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=2, loc='upper right')

    ax.set_xlabel('TTFT (ms)')
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('TTFT vs TPOT by Configuration\n(Shape=GPU freq, Color=CPU freq)')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'ttft_tpot_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: ttft_tpot_scatter.png")


def plot_cpu_gpu_effect(df):
    """Grouped bar: CPU and GPU frequency effect on throughput."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # CPU effect
    ax = axes[0]
    cpu_groups = df.groupby('cpu_freq_mhz')['tokens_per_second'].agg(['mean', 'std']).sort_index()
    colors = ['#dc3545', '#fd7e14', '#198754']
    bars = ax.bar([f'{int(c)}' for c in cpu_groups.index],
                  cpu_groups['mean'], yerr=cpu_groups['std'],
                  color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    for i, (idx, row) in enumerate(cpu_groups.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.15, f'{row["mean"]:.1f}',
                ha='center', fontweight='bold')
    ax.set_xlabel('CPU Frequency (MHz)')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('(a) CPU Frequency Effect\n(GPU freq averaged out)')
    # Add scaling factor
    low, high = cpu_groups['mean'].iloc[0], cpu_groups['mean'].iloc[-1]
    ax.text(0.05, 0.95, f'{low/high:.2f}x → {high/high:.2f}x\nScaling: {high/low:.2f}x',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # GPU effect
    ax = axes[1]
    gpu_groups = df.groupby('gpu_freq_mhz')['tokens_per_second'].agg(['mean', 'std']).sort_index()
    colors = ['#6c757d'] * len(gpu_groups)
    bars = ax.bar([f'{int(g)}' for g in gpu_groups.index],
                  gpu_groups['mean'], yerr=gpu_groups['std'],
                  color=colors, capsize=5, edgecolor='black', linewidth=0.5)
    for i, (idx, row) in enumerate(gpu_groups.iterrows()):
        ax.text(i, row['mean'] + row['std'] + 0.15, f'{row["mean"]:.1f}',
                ha='center', fontweight='bold')
    ax.set_xlabel('GPU Frequency (MHz)')
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('(b) GPU Frequency Effect\n(CPU freq averaged out)')
    ax.set_ylim(0, max(cpu_groups['mean']) * 1.25)
    low, high = gpu_groups['mean'].iloc[0], gpu_groups['mean'].iloc[-1]
    ax.text(0.05, 0.95, f'{low/high:.2f}x → {high/high:.2f}x\nScaling: {high/low:.2f}x',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    fig.suptitle('Frequency Scaling: CPU vs GPU (Real Model)', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'cpu_gpu_effect.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: cpu_gpu_effect.png")


def plot_workload_by_mode(df):
    """Grouped bar: per-workload throughput for core modes."""
    fig, ax = plt.subplots(figsize=(12, 6))

    core_df = df[df['mode'].isin(CORE_MODES)]
    workloads = sorted(core_df['workload'].unique())
    modes = CORE_MODES
    colors = ['#dc3545', '#fd7e14', '#6c757d', '#198754']

    x = np.arange(len(workloads))
    width = 0.8 / len(modes)

    for i, mode in enumerate(modes):
        vals = []
        for wl in workloads:
            v = core_df[(core_df['mode'] == mode) & (core_df['workload'] == wl)]['tokens_per_second'].mean()
            vals.append(v)
        ax.bar(x + i * width, vals, width, label=mode, color=colors[i],
               edgecolor='black', linewidth=0.3)

    ax.set_xticks(x + width * len(modes) / 2)
    wl_labels = {
        'short_short': 'S-S\n(128→64)',
        'short_long': 'S-L\n(128→256)',
        'medium_medium': 'M-M\n(512→128)',
        'long_medium': 'L-M\n(1024→128)',
        'long_long': 'L-L\n(1024→512)',
    }
    ax.set_xticklabels([wl_labels.get(wl, wl) for wl in workloads], fontsize=9)
    ax.set_ylabel('Throughput (tokens/s)')
    ax.set_title('Per-Workload Throughput by Power Mode\n(prompt→output tokens)')
    ax.legend(fontsize=9, loc='upper right')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'workload_by_mode.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: workload_by_mode.png")


def plot_energy_efficiency_proxy(df):
    """
    Proxy for energy efficiency: performance/frequency ratio.
    Since we don't have real energy data, use throughput/GPU_freq as proxy.
    Higher = more efficient use of GPU clock cycles.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = df.groupby('config_name').agg({
        'tokens_per_second': 'mean',
        'gpu_freq_mhz': 'first',
        'cpu_freq_mhz': 'first',
    }).sort_values('tokens_per_second', ascending=False)

    # Efficiency = throughput / GPU_freq (higher GPU freq costs more power)
    configs['efficiency'] = configs['tokens_per_second'] / configs['gpu_freq_mhz'] * 1000

    # Normalize to MAXN
    maxn_eff = configs[configs.index == 'GPU1300_CPU2201']['efficiency'].values[0]
    configs['efficiency_ratio'] = configs['efficiency'] / maxn_eff

    colors = ['#198754' if c >= 1.0 else '#0d6efd' if c >= 0.8 else '#fd7e14' if c >= 0.5 else '#dc3545'
              for c in configs['efficiency_ratio']]

    bars = ax.barh(range(len(configs)),
                   configs['efficiency_ratio'],
                   color=colors, edgecolor='black', linewidth=0.3)

    ax.set_yticks(range(len(configs)))
    labels = []
    for idx, row in configs.iterrows():
        mode = MODE_MAP.get(idx, idx)
        labels.append(f'{mode}  (GPU {int(row["gpu_freq_mhz"])}, CPU {int(row["cpu_freq_mhz"])})')
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='MAXN baseline')
    ax.set_xlabel('Relative GPU Efficiency (throughput/GPU_freq, MAXN=1.0)')
    ax.set_title('GPU Energy Efficiency Proxy\n(Higher = more performance per GPU clock cycle)')
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'energy_efficiency_proxy.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: energy_efficiency_proxy.png")


def plot_dashboard(df):
    """Single dashboard with 4 key charts."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Throughput comparison
    ax = axes[0, 0]
    core_df = df[df['mode'].isin(CORE_MODES)]
    colors = ['#dc3545', '#fd7e14', '#6c757d', '#198754']
    mode_order = CORE_MODES
    vals, errs = [], []
    for mode in mode_order:
        mdata = core_df[core_df['mode'] == mode]['tokens_per_second']
        vals.append(mdata.mean())
        errs.append(mdata.std())
    ax.bar(range(len(mode_order)), vals, yerr=errs, color=colors, capsize=4)
    ax.set_xticks(range(len(mode_order)))
    ax.set_xticklabels([m.split('(')[0].strip() for m in mode_order], fontsize=9, rotation=15)
    for i, v in enumerate(vals):
        ax.text(i, v + errs[i] + 0.1, f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('tokens/s')
    ax.set_title('(a) Throughput by Power Mode')

    # 2. CPU vs GPU scaling
    ax = axes[0, 1]
    cpu_data = df.groupby('cpu_freq_mhz')['tokens_per_second'].mean().sort_index()
    gpu_data = df.groupby('gpu_freq_mhz')['tokens_per_second'].mean().sort_index()
    cpu_norm = cpu_data / cpu_data.iloc[0]
    gpu_norm = gpu_data / gpu_data.iloc[0]
    ax.plot(cpu_data.index / 1000, cpu_norm, 'o-', color='#dc3545', linewidth=2.5,
            markersize=8, label=f'CPU scaling ({cpu_norm.iloc[-1]:.2f}x)')
    ax.plot(gpu_data.index / 1000, gpu_norm, 's--', color='#0d6efd', linewidth=2.5,
            markersize=8, label=f'GPU scaling ({gpu_norm.iloc[-1]:.2f}x)')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Relative Throughput')
    ax.set_title('(b) CPU vs GPU Frequency Scaling')
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)

    # 3. TTFT distribution
    ax = axes[1, 0]
    for i, mode in enumerate(mode_order):
        mdata = core_df[core_df['mode'] == mode]['ttft_ms']
        ax.barh(i, mdata.mean(), xerr=mdata.std(), color=colors[i],
                capsize=4, edgecolor='black', linewidth=0.3)
        ax.text(mdata.mean() + mdata.std() + 20, i, f'{mdata.mean():.0f}ms',
                va='center', fontsize=9)
    ax.set_yticks(range(len(mode_order)))
    ax.set_yticklabels([m.split('(')[0].strip() for m in mode_order], fontsize=9)
    ax.set_xlabel('TTFT (ms)')
    ax.set_title('(c) TTFT by Power Mode')

    # 4. TPOT distribution
    ax = axes[1, 1]
    for i, mode in enumerate(mode_order):
        mdata = core_df[core_df['mode'] == mode]['tpot_ms']
        ax.barh(i, mdata.mean(), xerr=mdata.std(), color=colors[i],
                capsize=4, edgecolor='black', linewidth=0.3)
        ax.text(mdata.mean() + mdata.std() + 2, i, f'{mdata.mean():.0f}ms',
                va='center', fontsize=9)
    ax.set_yticks(range(len(mode_order)))
    ax.set_yticklabels([m.split('(')[0].strip() for m in mode_order], fontsize=9)
    ax.set_xlabel('TPOT (ms)')
    ax.set_title('(d) TPOT by Power Mode')

    fig.suptitle('Jetson Power Mode Comparison Dashboard — Real Model (Phi-3-mini Q4)\n'
                 '180 runs: 4 GPU × 3 CPU × 5 workloads × 3 repeats',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'power_mode_dashboard.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: power_mode_dashboard.png")


if __name__ == '__main__':
    print("Generating Jetson Power Mode Comparison...")
    df, source = load_data()
    print(f"Valid runs: {len(df)}")

    plot_throughput_comparison(df)
    plot_tttp_tpot_scatter(df)
    plot_cpu_gpu_effect(df)
    plot_workload_by_mode(df)
    plot_energy_efficiency_proxy(df)
    plot_dashboard(df)

    # Print summary table
    print("\n" + "=" * 70)
    print("POWER MODE COMPARISON SUMMARY")
    print("=" * 70)
    core_df = df[df['mode'].isin(CORE_MODES)]
    summary = core_df.groupby('mode').agg({
        'ttft_ms': 'mean',
        'tpot_ms': 'mean',
        'tokens_per_second': ['mean', 'std'],
    }).round(1)
    summary = summary.loc[[m for m in CORE_MODES if m in summary.index]]
    print(summary.to_string())
    print("=" * 70)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
