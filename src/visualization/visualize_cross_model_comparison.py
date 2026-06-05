#!/usr/bin/env python3
"""
Cross-Model DVFS Comparison Visualization

Generates comprehensive comparison charts across 7B, 8B, 14B models:
  1. GPU frequency effect (E/tok, TPOT, Power) for each model
  2. Per-workload DVFS space comparison
  3. Cap mode savings comparison (vs dynamic, vs MAXN)
  4. Workload-aware optimal GPU frequency map
"""

import sys, os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.figsize'] = (16, 12)

MODEL_COLORS = {
    'Qwen2.5-7B-Instruct-Q4_K_M': '#2196F3',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M': '#4CAF50',
    'Qwen2.5-14B-Instruct-Q4_K_M': '#FF5722',
}
MODEL_LABELS = {
    'Qwen2.5-7B-Instruct-Q4_K_M': '7B (Qwen2.5)',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M': '8B (Llama-3.1)',
    'Qwen2.5-14B-Instruct-Q4_K_M': '14B (Qwen2.5)',
}


def load_all_lock_tables():
    files = sorted(Path('data/rate_tables').glob('lock_rate_table_20260604_*.parquet'))
    tables = {}
    for f in files:
        df = pd.read_parquet(f)
        for model in df['model'].unique():
            tables[model] = df[df['model'] == model]
    return tables


def load_all_cap_tables():
    files = sorted(Path('data/rate_tables').glob('cap_rate_table_with_savings_20260604_*.parquet'))
    tables = {}
    for f in files:
        df = pd.read_parquet(f)
        for model in df['model'].unique():
            tables[model] = df[df['model'] == model]
    return tables


def plot_gpu_frequency_effect(lock_tables, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GPU Frequency Effect Across Models (Decode Phase)', fontsize=14, fontweight='bold')

    for model_name, df in lock_tables.items():
        decode = df[df['phase'] == 'decode']
        gpu_avg = decode.groupby('gpu_freq_mhz').agg({
            'energy_per_token_j_median': 'mean',
            'tpot_ms_median': 'mean',
            'tokens_per_second_median': 'mean',
            'avg_power_w_median': 'mean',
        }).reset_index()

        label = MODEL_LABELS.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, '#999')

        axes[0, 0].plot(gpu_avg['gpu_freq_mhz'], gpu_avg['energy_per_token_j_median'],
                       'o-', color=color, label=label, linewidth=1.5, markersize=4)
        axes[0, 1].plot(gpu_avg['gpu_freq_mhz'], gpu_avg['tpot_ms_median'],
                       'o-', color=color, label=label, linewidth=1.5, markersize=4)
        axes[1, 0].plot(gpu_avg['gpu_freq_mhz'], gpu_avg['tokens_per_second_median'],
                       'o-', color=color, label=label, linewidth=1.5, markersize=4)
        axes[1, 1].plot(gpu_avg['gpu_freq_mhz'], gpu_avg['avg_power_w_median'],
                       'o-', color=color, label=label, linewidth=1.5, markersize=4)

    titles = ['Energy per Token (J)', 'Time per Output Token (ms)',
              'Tokens per Second', 'Average Power (W)']
    for ax, title in zip(axes.flat, titles):
        ax.set_xlabel('GPU Frequency (MHz)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([306, 510, 714, 918, 1122, 1300])

    plt.tight_layout()
    p = out_dir / 'cross_model_gpu_freq_effect.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {p}')


def plot_workload_dvfs_heatmap(lock_tables, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Per-Workload Best E/tok GPU Frequency (Decode Phase)',
                fontsize=14, fontweight='bold')

    for idx, (model_name, df) in enumerate(lock_tables.items()):
        decode = df[df['phase'] == 'decode']
        ax = axes[idx]
        label = MODEL_LABELS.get(model_name, model_name)

        rows = []
        for wl in sorted(decode['workload'].unique()):
            w = decode[decode['workload'] == wl]
            best = w.loc[w['energy_per_token_j_median'].idxmin()]
            rows.append({
                'workload': wl,
                'best_gpu': int(best['gpu_freq_mhz']),
                'ept_range': (w['energy_per_token_j_median'].max() - w['energy_per_token_j_median'].min()) / w['energy_per_token_j_median'].min() * 100,
                'best_ept': best['energy_per_token_j_median'],
            })
        rdf = pd.DataFrame(rows)

        color = MODEL_COLORS.get(model_name, '#2196F3')
        colors = [color if g >= 918 else '#90CAF9' for g in rdf['best_gpu']]
        bars = ax.barh(range(len(rdf)), rdf['best_gpu'], color=colors)
        ax.set_yticks(range(len(rdf)))
        ax.set_yticklabels(rdf['workload'], fontsize=9)
        ax.set_xlabel('Best GPU Frequency (MHz)')
        ax.set_title(label)
        ax.set_xlim(300, 1400)
        ax.axvline(x=918, color='red', linestyle='--', alpha=0.5, label='918 MHz')
        ax.grid(True, axis='x', alpha=0.3)

        for i, row in rdf.iterrows():
            ax.text(row['best_gpu'] + 15, i, f"{row['ept_range']:.0f}%",
                   va='center', fontsize=8, color='#666')

    axes[0].set_ylabel('Workload')
    plt.tight_layout()
    p = out_dir / 'cross_model_best_gpu_per_workload.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {p}')


def plot_cap_savings_comparison(cap_tables, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cap Mode Savings: Best Cap vs Baselines', fontsize=14, fontweight='bold')

    x = np.arange(12)
    workloads = sorted(cap_tables[list(cap_tables.keys())[0]]['workload'].unique())

    # E/tok savings vs dynamic
    for model_name, df in cap_tables.items():
        label = MODEL_LABELS.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, '#999')

        sav_dyn = []
        sav_maxn = []
        for wl in workloads:
            wl_df = df[df['workload'] == wl]
            dyn = wl_df[wl_df['control_mode'] == 'dynamic']
            maxn = wl_df[wl_df['control_mode'] == 'maxn']
            caps = wl_df[wl_df['control_mode'] == 'cap']

            dyn_ept = dyn['energy_per_token_j_median'].mean() if not dyn.empty else np.nan
            maxn_ept = maxn['energy_per_token_j_median'].mean() if not maxn.empty else np.nan

            if not caps.empty and not np.isnan(dyn_ept):
                best_cap = caps.loc[caps['energy_per_token_j_median'].idxmin()]
                sav_dyn.append((dyn_ept - best_cap['energy_per_token_j_median']) / dyn_ept * 100)
            else:
                sav_dyn.append(0)

            if not np.isnan(maxn_ept) and not np.isnan(dyn_ept):
                sav_maxn.append((maxn_ept - dyn_ept) / maxn_ept * 100)
            else:
                sav_maxn.append(0)

        axes[0].bar(x + list(cap_tables.keys()).index(model_name) * 0.25 - 0.25,
                   sav_dyn, width=0.25, color=color, label=label, alpha=0.85)
        axes[1].bar(x + list(cap_tables.keys()).index(model_name) * 0.25 - 0.25,
                   sav_maxn, width=0.25, color=color, label=label, alpha=0.85)

    axes[0].set_ylabel('E/tok Savings vs Dynamic (%)')
    axes[0].set_title('Best Cap E/tok Savings vs Dynamic')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(workloads, rotation=45, ha='right', fontsize=8)
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='y', alpha=0.3)

    axes[1].set_ylabel('E/tok Savings of Dynamic vs MAXN (%)')
    axes[1].set_title('Dynamic E/tok Savings vs MAXN')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(workloads, rotation=45, ha='right', fontsize=8)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'cross_model_cap_savings.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {p}')


def plot_dvfs_space_summary(lock_tables, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('DVFS Optimization Space Summary', fontsize=14, fontweight='bold')

    # Bar chart: DVFS space per model
    model_names = []
    dvfs_spaces = []
    colors = []
    for model_name, df in lock_tables.items():
        decode = df[df['phase'] == 'decode']
        ranges = []
        for wl in decode['workload'].unique():
            w = decode[decode['workload'] == wl]
            mn = w['energy_per_token_j_median'].min()
            mx = w['energy_per_token_j_median'].max()
            ranges.append((mx - mn) / mn * 100)
        model_names.append(MODEL_LABELS.get(model_name, model_name))
        dvfs_spaces.append(np.mean(ranges))
        colors.append(MODEL_COLORS.get(model_name, '#999'))

    bars = axes[0].bar(model_names, dvfs_spaces, color=colors, width=0.5)
    axes[0].set_ylabel('Avg DVFS Space (%)')
    axes[0].set_title('Average E/tok Range Across GPU Frequencies')
    for bar, val in zip(bars, dvfs_spaces):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)

    # Box plot: per-workload DVFS space
    data_for_box = []
    labels_for_box = []
    for model_name, df in lock_tables.items():
        decode = df[df['phase'] == 'decode']
        ranges = []
        for wl in decode['workload'].unique():
            w = decode[decode['workload'] == wl]
            mn = w['energy_per_token_j_median'].min()
            mx = w['energy_per_token_j_median'].max()
            ranges.append((mx - mn) / mn * 100)
        data_for_box.append(ranges)
        labels_for_box.append(MODEL_LABELS.get(model_name, model_name))

    bp = axes[1].boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel('E/tok Range (%)')
    axes[1].set_title('Per-Workload DVFS Space Distribution')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'cross_model_dvfs_space_summary.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {p}')


def main():
    out_dir = Path('figures/cross_model_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    lock_tables = load_all_lock_tables()
    cap_tables = load_all_cap_tables()

    print(f'Loaded {len(lock_tables)} lock tables, {len(cap_tables)} cap tables')

    plot_gpu_frequency_effect(lock_tables, out_dir)
    plot_workload_dvfs_heatmap(lock_tables, out_dir)
    plot_cap_savings_comparison(cap_tables, out_dir)
    plot_dvfs_space_summary(lock_tables, out_dir)

    print('\nAll cross-model comparison figures saved to figures/cross_model_comparison/')


if __name__ == '__main__':
    main()
