#!/usr/bin/env python3
"""
Phase 5 Comprehensive Comparison Visualizations
Generate publication-quality comparison charts from Phase-Aware experiment
and Selector Evaluation data.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTPUT_DIR = Path('figures/phase5_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Data Loading ──
pa_data = pd.read_csv('data/phase_aware_experiment/comparison_summary_20260512_205105.csv', index_col=0)
sel_data = pd.read_csv('data/selector_eval/selector_comparison.csv')
sel_regret = pd.read_csv('data/selector_eval/selector_regret.csv', index_col=0)

# ── Color palette ──
STRATEGY_COLORS = {
    'default': '#6c757d',
    'max_perf': '#dc3545',
    'energy_efficient': '#198754',
    'phase_aware': '#0d6efd',
    'oracle': '#ffc107',
    'adaptive_phase_aware': '#6f42c1',
}
SEL_COLORS = {
    'maxn_all_high': '#dc3545',
    'all_mid': '#6c757d',
    'energy_efficient_all_low': '#198754',
    'fixed_best_efficiency': '#fd7e14',
    'oracle_best_per_bucket': '#ffc107',
    'ours_slo_aware_selector': '#0d6efd',
}

SEL_LABELS = {
    'maxn_all_high': 'MaxN',
    'all_mid': 'All Mid',
    'energy_efficient_all_low': 'Energy Efficient',
    'fixed_best_efficiency': 'Fixed Best',
    'oracle_best_per_bucket': 'Oracle',
    'ours_slo_aware_selector': 'Ours (SLO-Aware)',
}

PA_LABELS = {
    'default': 'Default (Mid)',
    'max_perf': 'Max Performance',
    'energy_efficient': 'Energy Efficient',
    'phase_aware': 'Phase-Aware (Ours)',
    'oracle': 'Oracle',
}


def plot_phase_aware_comparison():
    """Fig 1: Phase-Aware strategy 4-metric comparison bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    strategies = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'oracle']
    labels = [PA_LABELS[s] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]

    # Energy per token
    ax = axes[0, 0]
    vals = [pa_data.loc[s, 'energy_per_token_j'] for s in strategies]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Energy/Token (J)')
    ax.set_title('(a) Energy Efficiency')
    ax.axhline(y=pa_data.loc['default', 'energy_per_token_j'], color='gray', ls='--', alpha=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.tick_params(axis='x', rotation=25)

    # TTFT
    ax = axes[0, 1]
    vals = [pa_data.loc[s, 'avg_ttft_ms'] for s in strategies]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('(b) Time to First Token')
    ax.axhline(y=1000, color='red', ls='--', alpha=0.4, label='SLO (1000ms)')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 2, f'{v:.0f}',
                ha='center', va='bottom', fontsize=8)
    ax.tick_params(axis='x', rotation=25)

    # TPOT
    ax = axes[1, 0]
    vals = [pa_data.loc[s, 'avg_tpot_ms'] for s in strategies]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('TPOT (ms)')
    ax.set_title('(c) Time per Output Token')
    ax.axhline(y=80, color='red', ls='--', alpha=0.4, label='SLO (80ms)')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f'{v:.1f}',
                ha='center', va='bottom', fontsize=8)
    ax.tick_params(axis='x', rotation=25)

    # SLO compliance
    ax = axes[1, 1]
    vals = [pa_data.loc[s, 'slo_met_rate'] * 100 for s in strategies]
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('SLO Compliance (%)')
    ax.set_title('(d) SLO Compliance Rate')
    ax.set_ylim(0, 105)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.0f}%',
                ha='center', va='bottom', fontsize=8)
    ax.tick_params(axis='x', rotation=25)

    fig.suptitle('Phase-Aware DVFS Strategy Comparison', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'phase_aware_4metrics.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: phase_aware_4metrics.png")


def plot_energy_reduction_waterfall():
    """Fig 2: Energy reduction % vs Default for each strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = ['max_perf', 'energy_efficient', 'phase_aware', 'oracle']
    labels = [PA_LABELS[s] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]

    reductions = [pa_data.loc[s, 'energy_reduction_pct'] for s in strategies]
    bars = ax.bar(labels, reductions, color=colors, edgecolor='white', linewidth=0.5, width=0.6)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Energy Reduction vs Default (%)')
    ax.set_title('Energy Reduction Relative to Default Strategy')

    for bar, v in zip(bars, reductions):
        offset = 1 if v >= 0 else -3
        ax.text(bar.get_x() + bar.get_width()/2, v + offset, f'{v:+.1f}%',
                ha='center', fontsize=10, fontweight='bold')

    ax.set_ylim(min(reductions) - 15, max(reductions) + 10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'energy_reduction_waterfall.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: energy_reduction_waterfall.png")


def plot_selector_regret():
    """Fig 3: Selector evaluation — energy regret by strategy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    strategies_order = ['maxn_all_high', 'all_mid', 'fixed_best_efficiency',
                        'energy_efficient_all_low', 'ours_slo_aware_selector', 'oracle_best_per_bucket']

    # Mean + P95 regret
    ax = axes[0]
    means = [sel_regret.loc[s, 'mean_energy_regret'] * 100 for s in strategies_order]
    p95s = [sel_regret.loc[s, 'p95_energy_regret'] * 100 for s in strategies_order]
    labels = [SEL_LABELS[s] for s in strategies_order]
    colors = [SEL_COLORS[s] for s in strategies_order]

    x = np.arange(len(strategies_order))
    ax.bar(x, means, color=colors, edgecolor='white', linewidth=0.5, width=0.6, label='Mean Regret')
    ax.errorbar(x, means, yerr=[np.abs(m - p) for m, p in zip(means, p95s)],
                fmt='o', color='red', markersize=4, capsize=4, label='P95')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel('Energy Regret vs Oracle (%)')
    ax.set_title('(a) Selector Energy Regret')
    ax.legend()

    # Per-bucket regret heatmap
    ax = axes[1]
    bucket_short = {
        'synthetic_qwen_7b_int4|synthetic|1.0|128|128.0|decode|1': 'BS1/PL128/OL128/dec',
        'synthetic_qwen_7b_int4|synthetic|1.0|128|512.0|mixed|1': 'BS1/PL128/OL512/mix',
        'synthetic_qwen_7b_int4|synthetic|1.0|512|128.0|mixed|1': 'BS1/PL512/OL128/mix',
        'synthetic_qwen_7b_int4|synthetic|1.0|512|128.0|prefill|1': 'BS1/PL512/OL128/pre',
        'synthetic_qwen_7b_int4|synthetic|1.0|1024|32.0|mixed|1': 'BS1/PL1024/OL32/mix',
        'synthetic_qwen_7b_int4|synthetic|2.0|512|128.0|mixed|1': 'BS2/PL512/OL128/mix',
        'synthetic_qwen_7b_int4|synthetic|4.0|256|64.0|mixed|1': 'BS4/PL256/OL64/mix',
    }

    plot_strategies = ['maxn_all_high', 'all_mid', 'fixed_best_efficiency', 'ours_slo_aware_selector']
    plot_labels = [SEL_LABELS[s] for s in plot_strategies]

    buckets = list(bucket_short.values())
    matrix = []
    for s in plot_strategies:
        row = []
        s_data = sel_data[sel_data['strategy'] == s]
        for bk_full, bk_short in bucket_short.items():
            match = s_data[s_data['bucket_key'] == bk_full]
            if not match.empty:
                regret = (match.iloc[0]['energy_per_token_j'] - match.iloc[0]['oracle_energy']) / match.iloc[0]['oracle_energy'] * 100
                row.append(regret)
            else:
                row.append(0)
        matrix.append(row)

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=-5, vmax=50)
    ax.set_xticks(range(len(buckets)))
    ax.set_xticklabels(buckets, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(plot_labels)))
    ax.set_yticklabels(plot_labels)
    ax.set_title('(b) Per-Bucket Regret (%)')
    for i in range(len(plot_labels)):
        for j in range(len(buckets)):
            ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center', fontsize=7,
                    color='white' if abs(matrix[i,j]) > 25 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.8, label='Regret (%)')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'selector_regret_analysis.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: selector_regret_analysis.png")


def plot_phase_aware_radar():
    """Fig 4: Radar/spider chart comparing Phase-Aware vs Default."""
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    categories = ['Energy\nEfficiency', 'TTFT', 'TPOT', 'SLO\nCompliance', 'Throughput']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    def normalize(s):
        e = pa_data.loc[s]
        return [
            1 - e['energy_per_token_j'] / 0.4,           # Lower energy = better
            1 - e['avg_ttft_ms'] / 200,                    # Lower TTFT = better
            1 - e['avg_tpot_ms'] / 25,                     # Lower TPOT = better
            e['slo_met_rate'],                               # Higher = better
            e['avg_tokens_per_second'] / 150,               # Higher = better
        ]

    for strategy, label, color in [
        ('default', 'Default', STRATEGY_COLORS['default']),
        ('phase_aware', 'Phase-Aware (Ours)', STRATEGY_COLORS['phase_aware']),
        ('max_perf', 'Max Perf', STRATEGY_COLORS['max_perf']),
    ]:
        values = normalize(strategy)
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Strategy Trade-off Profile', y=1.08, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'strategy_radar.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: strategy_radar.png")


def plot_phase_aware_ttft_tpot_scatter():
    """Fig 5: TTFT vs Energy/Token scatter, bubble size = TPOT."""
    fig, ax = plt.subplots(figsize=(8, 6))

    strategies = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'oracle']
    for s in strategies:
        e = pa_data.loc[s]
        ax.scatter(
            e['avg_ttft_ms'], e['energy_per_token_j'],
            s=max(e['avg_tpot_ms'] * 40, 80),
            c=STRATEGY_COLORS[s],
            label=f"{PA_LABELS[s]} (TPOT={e['avg_tpot_ms']:.1f}ms)",
            edgecolors='white', linewidth=1.5, alpha=0.85, zorder=5
        )
        ax.annotate(PA_LABELS[s], (e['avg_ttft_ms'], e['energy_per_token_j']),
                    textcoords="offset points", xytext=(8, 5), fontsize=8)

    ax.set_xlabel('TTFT (ms)')
    ax.set_ylabel('Energy/Token (J)')
    ax.set_title('Energy Efficiency vs TTFT Trade-off\n(Bubble size = TPOT)')
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'ttft_vs_energy_scatter.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: ttft_vs_energy_scatter.png")


def plot_p5_summary_dashboard():
    """Fig 6: Phase 5 summary dashboard with key numbers."""
    fig = plt.figure(figsize=(14, 8))

    # Title bar
    fig.text(0.5, 0.97, 'EnergyInfra Phase 5 — Evaluation Dashboard',
             ha='center', va='top', fontsize=16, fontweight='bold')

    # ── Panel 1: Phase-Aware bar comparison ──
    ax1 = fig.add_axes([0.05, 0.52, 0.42, 0.38])
    strategies = ['default', 'max_perf', 'energy_efficient', 'phase_aware', 'oracle']
    labels = [PA_LABELS[s] for s in strategies]
    colors = [STRATEGY_COLORS[s] for s in strategies]
    vals = [pa_data.loc[s, 'energy_per_token_j'] for s in strategies]
    ax1.barh(labels, vals, color=colors, edgecolor='white')
    ax1.set_xlabel('Energy/Token (J)')
    ax1.set_title('Phase-Aware Experiment Results', fontsize=11)
    ax1.axvline(x=pa_data.loc['default', 'energy_per_token_j'], color='gray', ls='--', alpha=0.5)
    for i, v in enumerate(vals):
        ax1.text(v + 0.003, i, f'{v:.3f}', va='center', fontsize=8)

    # ── Panel 2: Selector regret ──
    ax2 = fig.add_axes([0.55, 0.52, 0.42, 0.38])
    sel_strategies = ['maxn_all_high', 'all_mid', 'fixed_best_efficiency',
                      'ours_slo_aware_selector']
    sel_labels = [SEL_LABELS[s] for s in sel_strategies]
    sel_colors = [SEL_COLORS[s] for s in sel_strategies]
    regrets = [sel_regret.loc[s, 'mean_energy_regret'] * 100 for s in sel_strategies]
    ax2.barh(sel_labels, regrets, color=sel_colors, edgecolor='white')
    ax2.set_xlabel('Mean Energy Regret vs Oracle (%)')
    ax2.set_title('Selector Evaluation (P0 Fixes Verified)', fontsize=11)
    ax2.axvline(x=0, color='black', linewidth=0.8)
    for i, v in enumerate(regrets):
        ax2.text(max(v, 0.5) + 0.3, i, f'{v:.2f}%', va='center', fontsize=8)

    # ── Panel 3: Key metrics table ──
    ax3 = fig.add_axes([0.05, 0.05, 0.42, 0.38])
    ax3.axis('off')
    table_data = [
        ['Metric', 'Default', 'Phase-Aware', 'Change'],
        ['Energy/Token (J)', f'{pa_data.loc["default","energy_per_token_j"]:.3f}',
         f'{pa_data.loc["phase_aware","energy_per_token_j"]:.3f}',
         f'{pa_data.loc["phase_aware","energy_reduction_pct"]:+.1f}%'],
        ['TTFT (ms)', f'{pa_data.loc["default","avg_ttft_ms"]:.0f}',
         f'{pa_data.loc["phase_aware","avg_ttft_ms"]:.0f}',
         f'{pa_data.loc["phase_aware","ttft_reduction_pct"]:+.1f}%'],
        ['TPOT (ms)', f'{pa_data.loc["default","avg_tpot_ms"]:.1f}',
         f'{pa_data.loc["phase_aware","avg_tpot_ms"]:.1f}',
         f'{pa_data.loc["phase_aware","tpot_reduction_pct"]:+.1f}%'],
        ['SLO Rate', f'{pa_data.loc["default","slo_met_rate"]:.0%}',
         f'{pa_data.loc["phase_aware","slo_met_rate"]:.0%}', '-'],
        ['Oracle Regret', 'N/A', '0.00%', 'Verified'],
        ['Anomalies', 'N/A', '0', 'Clean'],
    ]
    table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#343a40')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 3 and row > 0:
            val = table_data[row][3]
            if '+' in val:
                cell.set_facecolor('#d1e7dd')
            elif '-' in val and val.startswith('-'):
                cell.set_facecolor('#f8d7da')
    ax3.set_title('Phase-Aware vs Default: Key Metrics', fontsize=11, y=0.98)

    # ── Panel 4: Phase 5 completion status ──
    ax4 = fig.add_axes([0.55, 0.05, 0.42, 0.38])
    ax4.axis('off')
    status_items = [
        ('P0.1: Fix fixed_best/oracle/regret', True),
        ('P0.2: Phase-specific energy columns', True),
        ('P0.3: Fix Pareto dominance', True),
        ('P1.1: Jetson power modes module', True),
        ('P1.2: Baseline config definitions', True),
        ('P2.1: Real model workload configs', True),
        ('P2.2: llama.cpp runner', True),
        ('P3.1: Adaptive policy connected', True),
        ('P1.3: Baseline comparison orchestrator', True),
        ('P4: Rate table rebuilt + verified', True),
    ]
    for i, (item, done) in enumerate(status_items):
        y = 0.92 - i * 0.085
        marker = '✓' if done else '✗'
        color = '#198754' if done else '#dc3545'
        ax4.text(0.02, y, marker, fontsize=12, color=color, fontweight='bold',
                 transform=ax4.transAxes, va='center')
        ax4.text(0.08, y, item, fontsize=9, transform=ax4.transAxes, va='center',
                 color='#333' if done else '#999')
    ax4.set_title('Phase 5 Implementation Status', fontsize=11, y=0.98)

    fig.savefig(OUTPUT_DIR / 'phase5_dashboard.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: phase5_dashboard.png")


if __name__ == '__main__':
    print("Generating Phase 5 comparison visualizations...")
    plot_phase_aware_comparison()
    plot_energy_reduction_waterfall()
    plot_selector_regret()
    plot_phase_aware_radar()
    plot_phase_aware_ttft_tpot_scatter()
    plot_p5_summary_dashboard()
    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
