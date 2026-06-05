#!/usr/bin/env python3
"""
E2E Benchmark Analysis: Workload-Aware DVFS vs Baselines

Analyzes cap_selector_benchmark data and generates comparison charts
showing the advantage of our Pareto-based DVFS selection vs baselines.
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
from matplotlib.lines import Line2D

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

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
STRATEGY_COLORS = {
    'pareto': '#E91E63',
    'min_energy': '#9C27B0',
    'slo_50ms': '#3F51B5',
    'slo_45ms': '#3F51B5',
    'alpha_03': '#FF9800',
    'alpha_07': '#FF9800',
    'pwr_45w': '#00BCD4',
    'dynamic': '#666666',
    'maxn': '#333333',
}
STRATEGY_LABELS = {
    'pareto': 'Pareto (ours)',
    'min_energy': 'Min Energy',
    'slo_50ms': 'SLO 50ms',
    'slo_45ms': 'SLO 45ms',
    'alpha_03': 'α=0.3',
    'alpha_07': 'α=0.7',
    'pwr_45w': 'Pwr≤45W',
    'dynamic': 'Dynamic (default)',
    'maxn': 'MAXN (all-max)',
}

OUT_DIR = Path('figures/e2e_benchmark')


def load_data():
    dfs = []
    for f in sorted(Path('data/cap_selector_benchmark').glob('*.csv')):
        dfs.append(pd.read_csv(str(f)))
    return pd.concat(dfs, ignore_index=True)


def compute_stats(df):
    """Compute per-(model, strategy, workload) aggregated stats."""
    valid = df[df['energy_per_token_j'] > 0].copy()
    grouped = valid.groupby(['model', 'strategy', 'workload']).agg({
        'energy_per_token_j': ['mean', 'std'],
        'tpot_ms': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std'],
        'avg_power_w': ['mean', 'std'],
        'total_energy_j': 'mean',
        'output_tokens': 'mean',
        'gpu_cap_mhz': 'first',
    })
    grouped.columns = ['_'.join(c).strip('_') if c[1] else c[0] for c in grouped.columns.values]
    return grouped.reset_index()


def compute_savings(stats, baseline='dynamic'):
    """Compute savings vs baseline for each strategy."""
    results = []
    for model in stats['model'].unique():
        m = stats[stats['model'] == model]
        base = m[m['strategy'] == baseline]
        if base.empty:
            continue
        for _, base_row in base.iterrows():
            wl = base_row['workload']
            for _, row in m[m['workload'] == wl].iterrows():
                if row['strategy'] == baseline:
                    results.append({
                        **row.to_dict(),
                        f'savings_ept_vs_{baseline}': 0.0,
                        f'savings_tpot_vs_{baseline}': 0.0,
                        f'savings_pwr_vs_{baseline}': 0.0,
                    })
                else:
                    results.append({
                        **row.to_dict(),
                        f'savings_ept_vs_{baseline}': (base_row['energy_per_token_j_mean'] - row['energy_per_token_j_mean']) / base_row['energy_per_token_j_mean'] * 100,
                        f'savings_tpot_vs_{baseline}': (base_row['tpot_ms_mean'] - row['tpot_ms_mean']) / base_row['tpot_ms_mean'] * 100,
                        f'savings_pwr_vs_{baseline}': (base_row['avg_power_w_mean'] - row['avg_power_w_mean']) / base_row['avg_power_w_mean'] * 100,
                    })
    return pd.DataFrame(results)


def plot_e2e_summary_bars(stats, sav_dyn, sav_maxn, out_dir):
    """Main summary: grouped bar chart of E/tok savings vs dynamic and MAXN."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('E2E Benchmark: Energy-per-Token Savings vs Baselines',
                fontsize=14, fontweight='bold')

    # Which strategies to show
    show_strategies = ['pareto', 'min_energy', 'slo_50ms', 'alpha_07', 'pwr_45w']
    models = sorted(stats['model'].unique())
    x = np.arange(len(models))
    width = 0.15

    for ax, sav_df, baseline_name in [(axes[0], sav_dyn, 'dynamic'), (axes[1], sav_maxn, 'maxn')]:
        for i, strat in enumerate(show_strategies):
            vals = []
            for model in models:
                s = sav_df[(sav_df['model'] == model) & (sav_df['strategy'] == strat)]
                vals.append(s[f'savings_ept_vs_{baseline_name}'].mean() if not s.empty else 0)
            color = STRATEGY_COLORS.get(strat, '#999')
            label = STRATEGY_LABELS.get(strat, strat)
            bars = ax.bar(x + i * width - 2 * width, vals, width=width,
                         color=color, alpha=0.85, label=label, edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                if abs(val) > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                           f'{val:+.1f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], fontsize=11)
        ax.set_ylabel('E/tok Savings (%)', fontsize=11)
        ax.set_title(f'vs {baseline_name.upper()}', fontsize=12)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'e2e_savings_summary.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_e2e_per_workload_heatmap(stats, sav_dyn, out_dir):
    """Per-workload heatmap: E/tok savings for Pareto vs dynamic."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('E2E Benchmark: Per-Workload E/tok Savings — Pareto vs Dynamic',
                fontsize=14, fontweight='bold')

    models = sorted(stats['model'].unique())

    for col_idx, model in enumerate(models):
        ax = axes[col_idx]
        ml = MODEL_LABELS.get(model, model)

        # Get Pareto and dynamic per workload
        pareto = sav_dyn[(sav_dyn['model'] == model) & (sav_dyn['strategy'] == 'pareto')]
        dyn = sav_dyn[(sav_dyn['model'] == model) & (sav_dyn['strategy'] == 'dynamic')]
        min_e = sav_dyn[(sav_dyn['model'] == model) & (sav_dyn['strategy'] == 'min_energy')]

        workloads = sorted(stats[stats['model'] == model]['workload'].unique())
        n_wl = len(workloads)

        x = np.arange(n_wl)
        width = 0.35

        pareto_vals = [pareto[pareto['workload'] == w]['savings_ept_vs_dynamic'].values[0]
                       for w in workloads if len(pareto[pareto['workload'] == w]) > 0]
        pareto_vals = pareto_vals[:n_wl]
        min_e_vals = [min_e[min_e['workload'] == w]['savings_ept_vs_dynamic'].values[0]
                      for w in workloads if len(min_e[min_e['workload'] == w]) > 0]
        min_e_vals = min_e_vals[:n_wl]

        color_p = MODEL_COLORS.get(model, '#2196F3')
        bars1 = ax.bar(x - width/2, pareto_vals, width, color='#E91E63', alpha=0.85,
                      label='Pareto (ours)', edgecolor='white')
        bars2 = ax.bar(x + width/2, min_e_vals, width, color='#9C27B0', alpha=0.85,
                      label='Min Energy', edgecolor='white')

        for bar, val in zip(bars1, pareto_vals):
            if abs(val) > 0.3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                       f'{val:+.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold',
                       color='#E91E63')
        for bar, val in zip(bars2, min_e_vals):
            if abs(val) > 0.3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                       f'{val:+.1f}%', ha='center', va='bottom', fontsize=7, color='#9C27B0')

        ax.set_xticks(x)
        ax.set_xticklabels([w.replace('p', '').replace('o', '→') for w in workloads],
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('E/tok Savings (%)', fontsize=10)
        ax.set_title(ml, fontsize=12, color=MODEL_COLORS.get(model, 'black'))
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'e2e_per_workload_savings.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_e2e_power_vs_tpot(stats, out_dir):
    """Power vs TPOT scatter: Pareto vs baselines."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('E2E Benchmark: Power vs TPOT Trade-off',
                fontsize=14, fontweight='bold')

    highlight = ['pareto', 'dynamic', 'maxn']

    for col_idx, model in enumerate(sorted(stats['model'].unique())):
        ax = axes[col_idx]
        ml = MODEL_LABELS.get(model, model)
        m = stats[stats['model'] == model]

        for strat in sorted(m['strategy'].unique()):
            s = m[m['strategy'] == strat]
            color = STRATEGY_COLORS.get(strat, '#bbb')
            alpha = 1.0 if strat in highlight else 0.4
            size = 120 if strat in highlight else 40
            marker = '*' if strat == 'pareto' else ('D' if strat == 'dynamic' else ('X' if strat == 'maxn' else 'o'))
            label = STRATEGY_LABELS.get(strat, strat) if strat in highlight else None
            zorder = 5 if strat in highlight else 2
            edgecolor = 'black' if strat in highlight else color

            ax.scatter(s['avg_power_w_mean'], s['tpot_ms_mean'],
                      c=color, s=size, marker=marker, alpha=alpha,
                      label=label, zorder=zorder, edgecolors=edgecolor if strat != 'dynamic' else color,
                      linewidths=0.8 if strat == 'pareto' else 0.5)

        ax.set_xlabel('Avg Power (W)', fontsize=11)
        ax.set_ylabel('TPOT (ms)', fontsize=11)
        ax.set_title(ml, fontsize=12, color=MODEL_COLORS.get(model, 'black'))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'e2e_power_vs_tpot.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_e2e_all_strategies_radar(stats, out_dir):
    """Radar/parallel coordinate chart: Pareto vs other strategies, normalized."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    fig.suptitle('E2E Benchmark: Multi-Objective Comparison (Normalized)',
                fontsize=14, fontweight='bold')

    objectives = ['E/tok (↓)', 'TPOT (↓)', 'Power (↓)', 'TPS (↑)']
    n_obj = len(objectives)

    for col_idx, model in enumerate(sorted(stats['model'].unique())):
        ax = axes[col_idx]
        ml = MODEL_LABELS.get(model, model)
        m = stats[stats['model'] == model].groupby('strategy').mean(numeric_only=True)

        # Normalize each objective to [0, 1] (0 = best)
        metrics = {
            'E/tok (↓)': 'energy_per_token_j_mean',
            'TPOT (↓)': 'tpot_ms_mean',
            'Power (↓)': 'avg_power_w_mean',
            'TPS (↑)': 'tokens_per_second_mean',
        }
        normed = {}
        for label, col in metrics.items():
            vals = m[col]
            mn, mx = vals.min(), vals.max()
            if '↑' in label:
                # Higher is better → 1 - (val-min)/(max-min) so min maps to 0
                normed[label] = 1.0 - (vals - mn) / (mx - mn + 1e-12)
            else:
                # Lower is better → (val-min)/(max-min) so min maps to 0
                normed[label] = (vals - mn) / (mx - mn + 1e-12)

        for strat in ['pareto', 'min_energy', 'slo_50ms', 'alpha_07', 'dynamic', 'maxn']:
            if strat not in m.index:
                continue
            vals = [normed[obj][strat] for obj in objectives]
            vals.append(vals[0])  # close the polygon

            angles = [i / n_obj * 2 * np.pi for i in range(n_obj)]
            angles.append(angles[0])

            color = STRATEGY_COLORS.get(strat, '#999')
            lw = 2.5 if strat == 'pareto' else 1.0
            label = STRATEGY_LABELS.get(strat, strat)
            ax.plot(angles, vals, 'o-', color=color, linewidth=lw, markersize=4,
                   label=label)
            ax.fill(angles, vals, color=color, alpha=0.08)

        ax.set_xticks([i / n_obj * 2 * np.pi for i in range(n_obj)])
        ax.set_xticklabels(objectives, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(['', '', ''], fontsize=7)
        ax.set_title(ml, fontsize=11, pad=15, color=MODEL_COLORS.get(model, 'black'))
        ax.legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    p = out_dir / 'e2e_multi_objective_radar.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_e2e_cap_selection_map(stats, out_dir):
    """Show which GPU cap each strategy selects per workload."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('E2E Benchmark: GPU Cap Selection per Workload (MHz)',
                fontsize=14, fontweight='bold')

    show_strats = ['pareto', 'min_energy', 'slo_50ms', 'alpha_07']
    workloads_order = ['p64_o64', 'p128_o512', 'p512_o128', 'p1024_o512', 'p2048_o512']

    for col_idx, model in enumerate(sorted(stats['model'].unique())):
        ax = axes[col_idx]
        ml = MODEL_LABELS.get(model, model)
        m = stats[stats['model'] == model]

        x = np.arange(len(workloads_order))
        width = 0.2

        for i, strat in enumerate(show_strats):
            s = m[m['strategy'] == strat]
            if s.empty:
                continue
            caps = []
            for wl in workloads_order:
                w = s[s['workload'] == wl]
                caps.append(w['gpu_cap_mhz_first'].values[0] if not w.empty else 0)

            color = STRATEGY_COLORS.get(strat, '#999')
            label = STRATEGY_LABELS.get(strat, strat)
            bars = ax.bar(x + i * width - 1.5 * width, caps, width,
                         color=color, alpha=0.85, label=label, edgecolor='white')
            for bar, val in zip(bars, caps):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                           str(int(val)), ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([w.replace('p', '').replace('o', '→') for w in workloads_order],
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('GPU Cap (MHz)', fontsize=10)
        ax.set_title(ml, fontsize=12, color=MODEL_COLORS.get(model, 'black'))
        ax.set_ylim(0, 1500)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'e2e_cap_selection_map.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def generate_report(df, stats, sav_dyn, sav_maxn, out_dir):
    """Generate markdown report."""
    lines = ['# E2E Cap Selector Benchmark Report',
             f'\n**Generated**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}',
             f'**Total runs**: {len(df)}',
             f'**Models**: {len(df["model"].unique())}',
             f'**Strategies**: {len(df["strategy"].unique())}',
             f'**Workloads**: {len(df["workload"].unique())}',
             '']

    # Per-model summary table
    for model in sorted(stats['model'].unique()):
        ml = MODEL_LABELS.get(model, model)
        m = stats[stats['model'] == model]
        lines.extend(['', f'## {ml}', ''])
        lines.append('| Strategy | Cap (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | ΔE vs Dyn | ΔE vs MAXN |')
        lines.append('|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|')

        for strat in ['pareto', 'min_energy', 'slo_50ms', 'slo_45ms', 'alpha_03', 'alpha_07', 'pwr_45w', 'dynamic', 'maxn']:
            s = m[m['strategy'] == strat]
            if s.empty:
                continue
            marker = '**⭐**' if strat == 'pareto' else ''
            cap = s['gpu_cap_mhz_first'].values[0]
            ept = s['energy_per_token_j_mean'].mean()
            tpot = s['tpot_ms_mean'].mean()
            tps = s['tokens_per_second_mean'].mean()
            pwr = s['avg_power_w_mean'].mean()

            dyn_ept = m[m['strategy'] == 'dynamic']['energy_per_token_j_mean'].mean()
            maxn_ept = m[m['strategy'] == 'maxn']['energy_per_token_j_mean'].mean()
            de = (dyn_ept - ept) / dyn_ept * 100 if dyn_ept > 0 else 0
            me = (maxn_ept - ept) / maxn_ept * 100 if maxn_ept > 0 else 0

            lines.append(
                f'| {marker} {strat} | {cap:.0f} | {ept:.4f} | {tpot:.1f} | {tps:.1f} | {pwr:.1f} | {de:+.1f}% | {me:+.1f}% |'
            )

    # Key findings
    lines.extend(['', '## Key Findings', ''])

    # Pareto vs dynamic for each model
    lines.append('### Pareto vs Dynamic (Default Governor)')
    lines.append('| Model | E/tok Δ | TPOT Δ | Power Δ |')
    lines.append('|:---|:---:|:---:|:---:|')
    for model in sorted(stats['model'].unique()):
        ml = MODEL_LABELS.get(model, model)
        p = sav_dyn[(sav_dyn['model'] == model) & (sav_dyn['strategy'] == 'pareto')]
        if not p.empty:
            lines.append(f'| {ml} | {p["savings_ept_vs_dynamic"].mean():+.1f}% | '
                        f'{p["savings_tpot_vs_dynamic"].mean():+.1f}% | '
                        f'{p["savings_pwr_vs_dynamic"].mean():+.1f}% |')

    # Pareto vs MAXN
    lines.append('')
    lines.append('### Pareto vs MAXN (Performance Mode)')
    lines.append('| Model | E/tok Δ | TPOT Δ | Power Δ |')
    lines.append('|:---|:---:|:---:|:---:|')
    for model in sorted(stats['model'].unique()):
        ml = MODEL_LABELS.get(model, model)
        p = sav_maxn[(sav_maxn['model'] == model) & (sav_maxn['strategy'] == 'pareto')]
        if not p.empty:
            lines.append(f'| {ml} | {p["savings_ept_vs_maxn"].mean():+.1f}% | '
                        f'{p["savings_tpot_vs_maxn"].mean():+.1f}% | '
                        f'{p["savings_pwr_vs_maxn"].mean():+.1f}% |')

    report = '\n'.join(lines)
    p = out_dir / 'e2e_benchmark_report.md'
    p.write_text(report)
    print(f'  Saved: {p}')
    return report


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading benchmark data...')
    df = load_data()
    print(f'  {len(df)} rows, {df["model"].nunique()} models, {df["strategy"].nunique()} strategies')

    print('Computing stats...')
    stats = compute_stats(df)
    sav_dyn = compute_savings(stats, 'dynamic')
    sav_maxn = compute_savings(stats, 'maxn')

    print('Generating figures...')
    plot_e2e_summary_bars(stats, sav_dyn, sav_maxn, out_dir)
    plot_e2e_per_workload_heatmap(stats, sav_dyn, out_dir)
    plot_e2e_power_vs_tpot(stats, out_dir)
    plot_e2e_all_strategies_radar(stats, out_dir)
    plot_e2e_cap_selection_map(stats, out_dir)
    report = generate_report(df, stats, sav_dyn, sav_maxn, out_dir)

    print(f'\nAll E2E benchmark figures saved to {out_dir}/')
    print('\n' + report)


if __name__ == '__main__':
    main()
