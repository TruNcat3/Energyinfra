#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Phase 13 Visualization (P4)

Generates 9 analysis charts from Phase 13 experiment data:
1. Oracle Gap — Strategies vs Oracle-Energy gap%
2. Temperature over Time — Different baselines' temperature curves
3. Power & TPOT over Time — Dual-axis line + rolling average
4. Cap Adaptation Timeline — ThermalSLO cap selection + temperature overlay
5. Serving Summary Dashboard — Multi-panel: tokens/J, power, temp, SLO, switches
6. Cross-trace Comparison — Radar chart comparing baselines across traces
7. 3D MDR/JIR Heatmap — Multi-objective dominance & joint improvement rates
8. Hypervolume Comparison — Bar chart of HV across strategies/models
9. 4D Serving MDR Matrix — Per-window multi-objective dominance in serving

Usage:
    python3 src/visualization/visualize_phase13.py
    python3 src/visualization/visualize_phase13.py --oracle-only
    python3 src/visualization/visualize_phase13.py --serving-data data/serving_benchmark/
"""

import sys
import os
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
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional

# ── Style constants ──
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})

MODEL_LABELS = {
    'Qwen2.5-7B-Instruct-Q4_K_M': '7B (Qwen2.5)',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M': '8B (Llama-3.1)',
    'Qwen2.5-14B-Instruct-Q4_K_M': '14B (Qwen2.5)',
}

BASELINE_COLORS = {
    'MAXN': '#e74c3c',
    'Dynamic': '#3498db',
    'BestStatic': '#2ecc71',
    'Pareto': '#9b59b6',
    'ThermalSLO': '#f39c12',
}

BASELINE_MARKERS = {
    'MAXN': 's',
    'Dynamic': 'o',
    'BestStatic': '^',
    'Pareto': 'D',
    'ThermalSLO': '*',
}

TRACE_COLORS = {
    'short_chat': '#3498db',
    'long_generation': '#e74c3c',
    'bursty_mixed': '#f39c12',
}


def find_latest_file(pattern: str, directory: str = 'data') -> Optional[str]:
    """Find latest file matching glob pattern."""
    files = sorted(Path(directory).rglob(pattern))
    return str(files[-1]) if files else None


# ═══════════════════════════════════════════════════════════
# Chart 1: Oracle Gap
# ═══════════════════════════════════════════════════════════
def plot_oracle_gap(df: pd.DataFrame, output_dir: Path):
    """Plot oracle gap analysis: strategy vs Oracle-Energy gap%."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Per-model gap bars
    ax = axes[0]
    strategies = ['Dynamic', 'MAXN', 'Best Static', 'Pareto (knee)']
    gap_cols = [
        'dynamic_vs_oracle_energy_gap_pct',
        'maxn_vs_oracle_energy_gap_pct',
        'best_static_vs_oracle_energy_gap_pct',
        'pareto_vs_oracle_energy_gap_pct',
    ]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    models = sorted(df['model'].unique())
    x = np.arange(len(models))
    width = 0.18

    for i, (strat, col, color) in enumerate(zip(strategies, gap_cols, colors)):
        vals = [df[df['model'] == m][col].mean() for m in models]
        labels = [MODEL_LABELS.get(m, m) for m in models]
        ax.bar(x + i * width, vals, width, label=strat, color=color, alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.3, f'{v:+.1f}%', ha='center', fontsize=7)

    ax.set_xlabel('Model')
    ax.set_ylabel('Gap vs Oracle-Energy (%)')
    ax.set_title('Oracle Gap by Model & Strategy')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models])
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Right: Overall summary
    ax = axes[1]
    avg_gaps = [df[col].mean() for col in gap_cols]
    bars = ax.barh(strategies, avg_gaps, color=colors, alpha=0.85)
    for bar, val in zip(bars, avg_gaps):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{val:+.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Average Gap vs Oracle-Energy (%)')
    ax.set_title('Average Oracle Gap (All Models)')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Add conclusion annotation
    pareto_avg = avg_gaps[-1]
    conclusion = ('DVFS space limited\n(Pareto → Oracle < 5%)'
                  if pareto_avg < 5
                  else 'Moderate DVFS space\nexists for optimization')
    ax.annotate(conclusion, xy=(pareto_avg, len(strategies) - 1),
                xytext=(pareto_avg + 8, len(strategies) - 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=8, color='red', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow'))

    plt.tight_layout()
    path = output_dir / 'oracle_gap.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════
# Chart 2: Temperature over Time
# ═══════════════════════════════════════════════════════════
def plot_temperature_over_time(wm_df: pd.DataFrame, output_dir: Path):
    """Plot temperature curves for different baselines."""
    if wm_df.empty:
        print('  Skipping temperature plot: no serving data')
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    for baseline in sorted(wm_df['baseline'].unique()):
        bl_data = wm_df[wm_df['baseline'] == baseline].sort_values('window_id')
        if bl_data.empty:
            continue
        ax.plot(bl_data['window_id'], bl_data['temp_end_c'],
                label=baseline, marker='o', markersize=3,
                color=BASELINE_COLORS.get(baseline, 'gray'),
                alpha=0.8)

    # Threshold lines
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Warning (75°C)')
    ax.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='Critical (85°C)')

    ax.set_xlabel('Window ID (10s each)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature over Time by Baseline')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'temperature_over_time.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════
# Chart 3: Power & TPOT over Time
# ═══════════════════════════════════════════════════════════
def plot_power_tpot_over_time(wm_df: pd.DataFrame, output_dir: Path):
    """Plot dual-axis power and TPOT over time with rolling average."""
    if wm_df.empty:
        print('  Skipping power/tpot plot: no serving data')
        return

    baselines = sorted(wm_df['baseline'].unique())
    n_bl = len(baselines)

    fig, axes = plt.subplots(n_bl, 1, figsize=(14, 4 * n_bl), sharex=True)
    if n_bl == 1:
        axes = [axes]

    for i, baseline in enumerate(baselines):
        bl = wm_df[wm_df['baseline'] == baseline].sort_values('window_id')
        if bl.empty:
            continue

        ax = axes[i]
        ax2 = ax.twinx()

        # Power line
        ax.plot(bl['window_id'], bl['avg_power_w'],
                color='#e74c3c', label='Power (W)', linewidth=1.5, alpha=0.8)
        # Rolling average
        if len(bl) >= 3:
            ax.plot(bl['window_id'], bl['avg_power_w'].rolling(3, center=True).mean(),
                    color='#e74c3c', linestyle='--', linewidth=2, alpha=0.5)

        # TPOT line
        ax2.plot(bl['window_id'], bl['tpot_p50_ms'],
                 color='#3498db', label='TPOT p50 (ms)', linewidth=1.5, alpha=0.8)
        ax2.plot(bl['window_id'], bl['tpot_p95_ms'],
                 color='#3498db', linestyle=':', linewidth=1, alpha=0.5,
                 label='TPOT p95 (ms)')
        # SLO line
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3)

        ax.set_ylabel('Power (W)', color='#e74c3c')
        ax2.set_ylabel('TPOT (ms)', color='#3498db')
        ax.set_title(f'{baseline}', fontsize=11)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

    axes[-1].set_xlabel('Window ID (10s each)')
    fig.suptitle('Power & TPOT over Time by Baseline', fontsize=14, y=1.01)
    plt.tight_layout()
    path = output_dir / 'power_tpot_over_time.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════
# Chart 4: Cap Adaptation Timeline
# ═══════════════════════════════════════════════════════════
def plot_cap_adaptation_timeline(wm_df: pd.DataFrame, output_dir: Path):
    """Plot ThermalSLO cap adaptation with temperature overlay."""
    thermal = wm_df[wm_df['baseline'] == 'ThermalSLO'].sort_values('window_id')
    if thermal.empty:
        print('  Skipping cap adaptation: no ThermalSLO data')
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Cap selection
    ax1.step(thermal['window_id'], thermal['cap_mhz'],
             where='post', color='#9b59b6', linewidth=2)
    ax1.fill_between(thermal['window_id'], thermal['cap_mhz'],
                     alpha=0.15, color='#9b59b6', step='post')
    ax1.set_ylabel('GPU Cap (MHz)')
    ax1.set_title('Cap Adaptation Timeline (ThermalSLO)')
    ax1.set_ylim(300, 1400)

    # Annotate cap changes
    cap_changes = thermal[thermal['cap_action'] != 'keep']
    for _, row in cap_changes.iterrows():
        ax1.annotate(
            f'{row["cap_action"]}',
            xy=(row['window_id'], row['cap_mhz']),
            xytext=(row['window_id'], row['cap_mhz'] + 60),
            fontsize=7, ha='center', color='red',
            arrowprops=dict(arrowstyle='->', color='red', lw=0.8),
        )

    # Panel 2: Temperature
    ax2.plot(thermal['window_id'], thermal['temp_end_c'],
             color='#e74c3c', linewidth=1.5, label='Temperature')
    ax2.axhline(y=75, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax2.axhline(y=85, color='red', linestyle='--', alpha=0.5, label='Critical')
    ax2.fill_between(thermal['window_id'], thermal['temp_end_c'],
                     75, alpha=0.1, color='orange', where=thermal['temp_end_c'] > 75)
    ax2.fill_between(thermal['window_id'], thermal['temp_end_c'],
                     85, alpha=0.1, color='red', where=thermal['temp_end_c'] > 85)
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend(fontsize=8)

    # Panel 3: SLO violation rate
    ax3.bar(thermal['window_id'], thermal['slo_violation_rate'] * 100,
            color='#e74c3c', alpha=0.6, width=0.8)
    ax3.set_ylabel('SLO Violation Rate (%)')
    ax3.set_xlabel('Window ID (10s each)')
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.3, label='10% threshold')

    plt.tight_layout()
    path = output_dir / 'cap_adaptation_timeline.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════
# Chart 5: Serving Summary Dashboard
# ═══════════════════════════════════════════════════════════
def plot_serving_summary_dashboard(wm_df: pd.DataFrame, output_dir: Path):
    """Multi-panel dashboard: tokens/J, power, temp, SLO, switches."""
    if wm_df.empty:
        print('  Skipping dashboard: no serving data')
        return

    baselines = sorted(wm_df['baseline'].unique())
    metrics_to_plot = [
        ('tokens_per_joule', 'Tokens/Joule', 'higher_better'),
        ('avg_power_w', 'Average Power (W)', 'lower_better'),
        ('temp_max_c', 'Peak Temperature (°C)', 'lower_better'),
        ('slo_violation_rate', 'SLO Violation Rate', 'lower_better'),
        ('cap_switch_count', 'Cap Switch Count', 'info'),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for i, (metric, title, orientation) in enumerate(metrics_to_plot):
        ax = axes[i]
        data = wm_df.groupby('baseline')[metric].agg(['mean', 'std']).reindex(baselines)

        means = data['mean'].values
        stds = data['std'].values
        x = np.arange(len(baselines))
        colors = [BASELINE_COLORS.get(b, 'gray') for b in baselines]

        bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8,
                      capsize=3, error_kw={'linewidth': 1})

        # Value labels
        for j, (m, s) in enumerate(zip(means, stds)):
            fmt = f'{m:.1f}' if m < 100 else f'{m:.0f}'
            ax.text(j, m + s + 0.5, fmt, ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(baselines, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)

        # Highlight best
        if orientation == 'lower_better' and len(means) > 0:
            best_idx = np.argmin(means)
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(2)
        elif orientation == 'higher_better' and len(means) > 0:
            best_idx = np.argmax(means)
            bars[best_idx].set_edgecolor('green')
            bars[best_idx].set_linewidth(2)

    fig.suptitle('Serving Benchmark Summary Dashboard', fontsize=14, y=1.02)
    plt.tight_layout()
    path = output_dir / 'serving_summary_dashboard.png'
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════
# Chart 6: Cross-trace Radar Comparison
# ═══════════════════════════════════════════════════════════
def plot_cross_trace_radar(wm_df: pd.DataFrame, output_dir: Path):
    """Radar chart comparing baselines across traces on multiple metrics."""
    if wm_df.empty or 'trace' not in wm_df.columns:
        print('  Skipping radar: no trace data')
        return

    baselines = sorted(wm_df['baseline'].unique())
    traces = sorted(wm_df['trace'].unique())

    # Aggregate metrics per (baseline, trace)
    agg = wm_df.groupby(['baseline', 'trace']).agg(
        tokens_per_joule=('tokens_per_joule', 'mean'),
        avg_power_w=('avg_power_w', 'mean'),
        temp_max_c=('temp_max_c', 'mean'),
        slo_violation_rate=('slo_violation_rate', 'mean'),
        tpot_p50_ms=('tpot_p50_ms', 'mean'),
    ).reset_index()

    # Normalize to 0-1 range across all baselines for radar
    metrics_for_radar = ['tokens_per_joule', 'avg_power_w', 'temp_max_c',
                         'slo_violation_rate', 'tpot_p50_ms']
    # Invert metrics where lower is better
    invert_metrics = ['avg_power_w', 'temp_max_c', 'slo_violation_rate', 'tpot_p50_ms']

    for metric in metrics_for_radar:
        min_val = agg[metric].min()
        max_val = agg[metric].max()
        if max_val > min_val:
            if metric in invert_metrics:
                agg[f'{metric}_norm'] = 1.0 - (agg[metric] - min_val) / (max_val - min_val)
            else:
                agg[f'{metric}_norm'] = (agg[metric] - min_val) / (max_val - min_val)
        else:
            agg[f'{metric}_norm'] = 0.5

    # Plot one radar per trace
    for trace in traces:
        trace_data = agg[agg['trace'] == trace]
        if trace_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        categories = ['Tokens/J', 'Power\n(inverted)', 'Temp\n(inverted)',
                      'SLO Compl.\n(inverted)', 'TPOT\n(inverted)']
        norm_cols = [f'{m}_norm' for m in metrics_for_radar]
        n_cats = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
        angles += angles[:1]

        for _, row in trace_data.iterrows():
            values = [row[c] for c in norm_cols]
            values += values[:1]
            bl = row['baseline']
            ax.plot(angles, values, 'o-', linewidth=2,
                    label=bl, color=BASELINE_COLORS.get(bl, 'gray'),
                    marker=BASELINE_MARKERS.get(bl, 'o'), markersize=5)
            ax.fill(angles, values, alpha=0.05,
                    color=BASELINE_COLORS.get(bl, 'gray'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f'Cross-Baseline Comparison\nTrace: {trace}', fontsize=12, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)

        plt.tight_layout()
        path = output_dir / f'cross_trace_radar_{trace}.png'
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved: {path}')


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# Chart 7: 3D MDR/JIR Heatmap
# ══════════════════════════════════════════════════════════════════

def plot_3d_mdr_jir_heatmap(eval_dir: str, output_dir: Path):
    """
    Multi-objective MDR and JIR heatmap from E2E benchmark data.

    Two side-by-side heatmaps showing pairwise MDR and JIR between
    all strategies, averaged across models and workloads.
    """
    # Load offline evaluation CSV
    eval_files = sorted(Path(eval_dir).glob('multi_obj_3d_offline_*.csv'))
    if not eval_files:
        print('    No 3D offline evaluation data found')
        return

    df = pd.read_csv(eval_files[-1])

    strategies = sorted(df['strategy_a'].unique())
    # Limit to main strategies for clarity
    main_strats = [s for s in strategies if s in
                   ['pareto', 'dynamic', 'maxn', 'min_energy', 'slo_50ms',
                    'alpha_07', 'best_static', 'pwr_45w']]

    models = sorted(df['model'].unique())
    model_label = {m: m.replace('-Instruct-Q4_K_M', '').replace('Meta-Llama-3.1-', 'Llama-')
                   for m in models}

    for metric_name, metric_col, cmap, label in [
        ('MDR', 'mdr', 'YlOrRd', 'Multi-Objective Dominance Rate'),
        ('JIR', 'jir', 'YlGn', 'Joint Improvement Ratio'),
    ]:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models) + 1, 4.5),
                                  squeeze=False)
        fig.suptitle(f'3D {label}: E/tok × TPOT × Power (minimize)',
                     fontsize=13, fontweight='bold')

        for j, model in enumerate(models):
            ax = axes[0, j]
            model_df = df[df['model'] == model]

            n = len(main_strats)
            matrix = np.zeros((n, n))
            for i, sa in enumerate(main_strats):
                for k, sb in enumerate(main_strats):
                    row = model_df[(model_df['strategy_a'] == sa) &
                                   (model_df['strategy_b'] == sb)]
                    if len(row) > 0:
                        matrix[i, k] = row.iloc[0][metric_col]

            # Mask diagonal
            mask = np.eye(n, dtype=bool)
            masked = np.ma.array(matrix, mask=mask)

            im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=0)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(main_strats, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(main_strats, fontsize=8)
            ax.set_title(model_label[model], fontsize=11)

            # Annotate cells
            for i in range(n):
                for k in range(n):
                    if i != k:
                        val = matrix[i, k]
                        color = 'white' if val > np.nanmax(masked) * 0.6 else 'black'
                        ax.text(k, i, f'{val:.0%}', ha='center', va='center',
                                fontsize=7, color=color)

        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=metric_name)

        # Row label = "dominates column"
        fig.text(0.01, 0.5, 'Row dominates → Column',
                 rotation=90, va='center', fontsize=9, fontstyle='italic')

        plt.tight_layout(rect=[0.03, 0, 1, 0.93])
        path = output_dir / f'multi_obj_3d_{metric_name.lower()}_heatmap.png'
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f'    Saved: {path}')


# ══════════════════════════════════════════════════════════════════
# Chart 8: Hypervolume Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════

def plot_hypervolume_comparison(eval_dir: str, output_dir: Path):
    """
    Bar chart comparing Hypervolume across strategies and models.

    Shows both 3D offline HV (from E2E benchmark) and 4D serving HV
    (from serving windows), highlighting Pareto's multi-objective advantage.
    """
    hv_files = sorted(Path(eval_dir).glob('multi_obj_hv_summary_*.csv'))
    if not hv_files:
        print('    No HV summary data found')
        return

    hv_df = pd.read_csv(hv_files[-1])

    model_label = {
        'Qwen2.5-7B-Instruct-Q4_K_M': '7B',
        'Meta-Llama-3.1-8B-Instruct-Q4_K_M': '8B',
        'Qwen2.5-14B-Instruct-Q4_K_M': '14B',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Hypervolume Indicator (Higher = Better)',
                 fontsize=14, fontweight='bold')

    # ── Left: 3D Offline ──
    ax1 = axes[0]
    offline_hv = hv_df[hv_df['trace'] == 'offline'].copy()
    if len(offline_hv) > 0:
        offline_hv['model_short'] = offline_hv['model'].map(model_label)
        strats_3d = sorted(offline_hv['strategy'].unique())

        x = np.arange(len(strats_3d))
        width = 0.25
        models_3d = sorted(offline_hv['model_short'].unique())
        colors_3d = ['#3498db', '#e74c3c', '#2ecc71']

        for i, m in enumerate(models_3d):
            vals = [offline_hv[(offline_hv['strategy'] == s) &
                              (offline_hv['model_short'] == m)]['hv_3d'].values
                    for s in strats_3d]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            bars = ax1.bar(x + i * width, vals, width, label=m, color=colors_3d[i])

        ax1.set_xticks(x + width)
        ax1.set_xticklabels(strats_3d, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Hypervolume (3D)')
        ax1.set_title('3D: E/tok × TPOT × Power')
        ax1.legend(title='Model')
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3)

    # ── Right: 4D Serving ──
    ax2 = axes[1]
    serving_hv = hv_df[(hv_df['trace'] != 'offline') &
                        (hv_df['hv_4d'] > 0)].copy()
    if len(serving_hv) > 0:
        serving_hv['model_short'] = serving_hv['model'].map(model_label)
        # For serving, use the entry with max points per (baseline, model)
        serving_hv = serving_hv.loc[
            serving_hv.groupby(['strategy', 'model'])['n_points'].idxmax()
        ]

        strats_4d = sorted(serving_hv['strategy'].unique())
        x4 = np.arange(len(strats_4d))
        width4 = 0.25
        models_4d = sorted(serving_hv['model_short'].unique())
        colors_4d = ['#3498db', '#e74c3c', '#2ecc71']

        for i, m in enumerate(models_4d):
            vals = [serving_hv[(serving_hv['strategy'] == s) &
                               (serving_hv['model_short'] == m)]['hv_4d'].values
                    for s in strats_4d]
            vals = [v[0] if len(v) > 0 else 0 for v in vals]
            bars = ax2.bar(x4 + i * width4, vals, width4, label=m, color=colors_4d[i])

            # Highlight Pareto bars
            for si, s in enumerate(strats_4d):
                if s == 'Pareto':
                    ax2.bar(x4[si] + i * width4, vals[si], width4,
                            edgecolor='black', linewidth=2, color=colors_4d[i])

        ax2.set_xticks(x4 + width4)
        ax2.set_xticklabels(strats_4d, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Hypervolume (4D)')
        ax2.set_title('4D: E/tok × TPOT × Power × Peak Temp')
        ax2.legend(title='Model')
        ax2.set_yscale('log')
        ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'multi_obj_hypervolume_comparison.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


# ══════════════════════════════════════════════════════════════════
# Chart 9: 4D Serving MDR Matrix
# ══════════════════════════════════════════════════════════════════

def plot_4d_serving_mdr_matrix(eval_dir: str, output_dir: Path):
    """
    Per-window MDR matrix for 4D serving evaluation.

    Shows MDR for each (baseline_a, baseline_b) pair per model,
    averaged across traces. Highlights Pareto/ThermalSLO vs MAXN/Dynamic.
    """
    eval_files = sorted(Path(eval_dir).glob('multi_obj_4d_serving_*.csv'))
    if not eval_files:
        print('    No 4D serving evaluation data found')
        return

    df = pd.read_csv(eval_files[-1])

    # Use per-window results only (total_count > 1)
    win_df = df[df['total_count'] > 1].copy()

    baselines = sorted(['MAXN', 'Dynamic', 'BestStatic', 'Pareto', 'ThermalSLO'])
    models = sorted(win_df['model'].unique())
    model_label = {m: m.replace('-Instruct-Q4_K_M', '').replace('Meta-Llama-3.1-', 'Llama-')
                   for m in models}

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models) + 1, 4.5),
                              squeeze=False)
    fig.suptitle('4D Serving MDR: E/tok × TPOT × Power × Peak Temp (per-window)',
                 fontsize=13, fontweight='bold')

    for j, model in enumerate(models):
        ax = axes[0, j]
        model_df = win_df[win_df['model'] == model]

        n = len(baselines)
        matrix = np.zeros((n, n))
        for i, ba in enumerate(baselines):
            for k, bb in enumerate(baselines):
                rows = model_df[(model_df['strategy_a'] == ba) &
                                (model_df['strategy_b'] == bb)]
                if len(rows) > 0:
                    matrix[i, k] = rows['mdr'].mean()

        mask = np.eye(n, dtype=bool)
        masked = np.ma.array(matrix, mask=mask)

        im = ax.imshow(masked, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=0.3)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(baselines, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(baselines, fontsize=8)
        ax.set_title(model_label[model], fontsize=11)

        for i in range(n):
            for k in range(n):
                if i != k:
                    val = matrix[i, k]
                    color = 'white' if val > 0.15 else 'black'
                    ax.text(k, i, f'{val:.0%}', ha='center', va='center',
                            fontsize=7, color=color)

    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='MDR')
    fig.text(0.01, 0.5, 'Row dominates → Column',
             rotation=90, va='center', fontsize=9, fontstyle='italic')

    plt.tight_layout(rect=[0.03, 0, 1, 0.93])
    path = output_dir / 'multi_obj_4d_serving_mdr_matrix.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {path}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 13 Visualization')
    parser.add_argument('--oracle-only', action='store_true',
                        help='Only generate oracle gap chart')
    parser.add_argument('--serving-data', type=str,
                        default='data/serving_benchmark',
                        help='Directory with serving benchmark data')
    parser.add_argument('--oracle-data', type=str,
                        default='data/oracle_gap_analysis',
                        help='Directory with oracle gap analysis data')
    parser.add_argument('--output', type=str,
                        default='figures/phase13_analysis',
                        help='Output directory')
    parser.add_argument('--eval-data', type=str,
                        default='data/multi_obj_eval',
                        help='Directory with multi-objective evaluation results')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('  Phase 13 Visualization (9 charts)')
    print(f'  Output: {output_dir}')
    print('=' * 60)

    # ── Chart 1: Oracle Gap ──
    oracle_files = sorted(Path(args.oracle_data).glob('oracle_gap_*.csv'))
    if oracle_files:
        oracle_df = pd.read_csv(oracle_files[-1])
        # Ensure numeric columns
        for col in oracle_df.columns:
            if 'gap_pct' in col or 'ept' in col.lower() or 'tpot' in col.lower():
                oracle_df[col] = pd.to_numeric(oracle_df[col], errors='coerce')
        print(f'\n[1/9] Oracle Gap: {len(oracle_df)} rows from {oracle_files[-1].name}')
        plot_oracle_gap(oracle_df, output_dir)
    else:
        print('\n[1/9] Oracle Gap: no data found, skipping')

    if args.oracle_only:
        print('\n  Oracle-only mode complete')
        return

    # ── Load serving data ──
    serving_dir = Path(args.serving_data)
    serving_files = sorted(serving_dir.glob('serving_windows_*.csv'))
    if serving_files:
        wm_df = pd.concat([pd.read_csv(f) for f in serving_files], ignore_index=True)
        # Clean columns
        for col in wm_df.columns:
            wm_df[col] = pd.to_numeric(wm_df[col], errors='coerce')
        print(f'\nServing data: {len(serving_files)} files, {len(wm_df)} window rows')
    else:
        wm_df = pd.DataFrame()
        print(f'\nServing data: no files found in {args.serving_data}')

    # ── Charts 2-6 (require serving data) ──
    if not wm_df.empty:
        print('\n[2/9] Temperature over Time')
        plot_temperature_over_time(wm_df, output_dir)

        print('\n[3/9] Power & TPOT over Time')
        plot_power_tpot_over_time(wm_df, output_dir)

        print('\n[4/9] Cap Adaptation Timeline')
        plot_cap_adaptation_timeline(wm_df, output_dir)

        print('\n[5/9] Serving Summary Dashboard')
        plot_serving_summary_dashboard(wm_df, output_dir)

        print('\n[6/9] Cross-trace Radar')
        plot_cross_trace_radar(wm_df, output_dir)
    else:
        for i in range(2, 7):
            print(f'\n[{i}/9] Skipped: no serving data')

    # ── Charts 7-9 (require multi-objective evaluation data) ──
    eval_dir = Path(args.eval_data)
    if eval_dir.exists():
        print('\n[7/9] 3D MDR/JIR Heatmap')
        plot_3d_mdr_jir_heatmap(str(eval_dir), output_dir)

        print('\n[8/9] Hypervolume Comparison')
        plot_hypervolume_comparison(str(eval_dir), output_dir)

        print('\n[9/9] 4D Serving MDR Matrix')
        plot_4d_serving_mdr_matrix(str(eval_dir), output_dir)
    else:
        print(f'\nMulti-objective eval data not found in {args.eval_data}')
        print('Run: python3 src/ratetable/pareto_multi_objective_evaluation.py')
        for i in range(7, 10):
            print(f'\n[{i}/9] Skipped: no eval data')

    print(f'\n  Phase 13 visualization complete ({len(list(output_dir.glob("*.png")))} charts)')


if __name__ == '__main__':
    main()
