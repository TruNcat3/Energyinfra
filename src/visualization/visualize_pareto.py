#!/usr/bin/env python3
"""
Pareto Frontier Visualization

Generates 6 charts showing the multi-objective Pareto frontier
across models, workloads, and objectives.

Usage:
    python3 src/visualization/visualize_pareto.py
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

OUT_DIR = Path('figures/pareto_frontier')


def load_all_tables():
    lock_files = sorted(Path('data/rate_tables').glob('lock_rate_table_20260604_*.parquet'))
    cap_files = sorted(Path('data/rate_tables').glob('cap_rate_table_with_savings_20260604_*.parquet'))
    lock_tables, cap_tables = {}, {}
    for f in lock_files:
        df = pd.read_parquet(f)
        for m in df['model'].unique():
            lock_tables[m] = df[df['model'] == m]
    for f in cap_files:
        df = pd.read_parquet(f)
        for m in df['model'].unique():
            cap_tables[m] = df[df['model'] == m]
    return lock_tables, cap_tables


def compute_all_frontiers(lock_tables, cap_tables, phase='decode'):
    """Pre-compute Pareto frontiers for all (model, workload, source) combos.

    Returns dict: (model_name, wl, source) -> {
        'all_df': DataFrame with pareto_rank/is_pareto columns,
        'frontier_df': Pareto-optimal subset,
        'suffix': '_median' or '_mean',
    }
    """
    from src.controller.pareto_selector import compute_pareto_frontier

    results = {}
    for model_name in lock_tables:
        for source_name, table in [('lock', lock_tables[model_name]),
                                   ('cap', cap_tables.get(model_name, pd.DataFrame()))]:
            if table.empty:
                continue
            phase_df = table if 'phase' not in table.columns else table[table['phase'] == phase]
            for wl in phase_df['workload'].unique():
                key = (model_name, wl, source_name)
                df = phase_df[phase_df['workload'] == wl].copy()

                # For cap mode, compute frontier on caps only
                if source_name == 'cap' and 'control_mode' in df.columns:
                    cap_only = df[df['control_mode'] == 'cap']
                else:
                    cap_only = df

                if cap_only.empty:
                    results[key] = {
                        'all_df': df, 'frontier_df': pd.DataFrame(), 'suffix': '_median',
                    }
                    continue

                suffix = '_median' if 'energy_per_token_j_median' in cap_only.columns else '_mean'
                obj_cols = [f'{o}{suffix}' for o in ['energy_per_token_j', 'tpot_ms', 'avg_power_w']]
                avail = [c for c in obj_cols if c in cap_only.columns]
                if len(avail) >= 2:
                    ranked = compute_pareto_frontier(cap_only, avail)
                else:
                    ranked = cap_only.copy()
                    ranked['is_pareto'] = False
                    ranked['pareto_rank'] = -1

                results[key] = {
                    'all_df': ranked,
                    'frontier_df': ranked[ranked['is_pareto'] == True].copy(),
                    'suffix': suffix,
                }
    return results


def _get_model_workloads(frontiers):
    """Extract sorted unique models and workloads from frontiers dict."""
    models = sorted(set(k[0] for k in frontiers))
    workloads = sorted(set(k[1] for k in frontiers))
    return models, workloads


def plot_pareto_2d(frontiers, out_dir):
    """Main 2D plot: E/tok vs TPOT with Pareto frontiers (lock-mode)."""
    models, all_wls = _get_model_workloads(frontiers)
    # Pick representative workloads that exist in all models
    lock_keys = [k for k in frontiers if k[2] == 'lock']
    repr_wls = sorted(set(k[1] for k in lock_keys))[:6]

    n_rows = len(repr_wls)
    n_cols = len(models)
    if n_rows == 0 or n_cols == 0:
        print('  Skipped plot_pareto_2d: no data')
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle('Pareto Frontier: Energy per Token vs Time per Output Token (Lock Mode)',
                 fontsize=14, fontweight='bold', y=1.01)

    from src.controller.pareto_selector import find_knee_point

    for row_idx, wl in enumerate(repr_wls):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            key = (model_name, wl, 'lock')
            if key not in frontiers:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(wl if col_idx == 0 else '', fontsize=9)
                continue

            data = frontiers[key]
            suffix = data['suffix']
            ept_col = f'energy_per_token_j{suffix}'
            tpot_col = f'tpot_ms{suffix}'

            all_df = data['all_df']
            frontier_df = data['frontier_df']

            if all_df.empty or ept_col not in all_df.columns:
                continue

            # All configs as grey dots
            valid = all_df[(all_df[ept_col] > 0) & (all_df[tpot_col] > 0)]
            ax.scatter(valid[ept_col], valid[tpot_col], c='#ddd', s=30, zorder=1,
                      edgecolors='#999', linewidths=0.5, alpha=0.6)

            # Pareto frontier as colored connected line
            if not frontier_df.empty and ept_col in frontier_df.columns:
                color = MODEL_COLORS.get(model_name, '#2196F3')
                valid_f = frontier_df[(frontier_df[ept_col] > 0) & (frontier_df[tpot_col] > 0)]
                if len(valid_f) >= 2:
                    sorted_f = valid_f.sort_values(ept_col)
                    ax.plot(sorted_f[ept_col], sorted_f[tpot_col], 'o-', color=color,
                           markersize=7, linewidth=2, zorder=3)
                    # Knee point
                    knee_idx = find_knee_point(sorted_f, ept_col, tpot_col)
                    knee = sorted_f.iloc[knee_idx]
                    ax.scatter([knee[ept_col]], [knee[tpot_col]], marker='*', s=150,
                              color='#E91E63', zorder=5, edgecolors='black', linewidths=0.8)
                elif len(valid_f) == 1:
                    ax.scatter(valid_f[ept_col], valid_f[tpot_col], marker='o', s=80,
                              color=color, zorder=3, edgecolors='black')

            # Baselines from cap data
            cap_key = (model_name, wl, 'cap')
            if cap_key in frontiers:
                cap_data = frontiers[cap_key]['all_df']
                for baseline, marker, lbl in [('dynamic', 'D', 'dynamic'), ('maxn', 'X', 'MAXN')]:
                    if 'control_mode' in cap_data.columns:
                        bl = cap_data[cap_data['control_mode'] == baseline]
                        if not bl.empty and ept_col in bl.columns:
                            ax.scatter(bl[ept_col], bl[tpot_col], marker=marker, s=80,
                                      c='none', edgecolors='#666', linewidths=1.5, zorder=4)

            ax.set_xlabel('Energy per Token (J)', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('TPOT (ms)', fontsize=9)
            ax.set_title(wl if col_idx == 0 else '', fontsize=9)
            ax.grid(True, alpha=0.2)

            if row_idx == 0 and col_idx == n_cols - 1:
                legend_elements = [
                    Line2D([0], [0], marker='o', color='#2196F3', linestyle='-', markersize=6, label='Pareto frontier'),
                    Line2D([0], [0], marker='*', color='#E91E63', linestyle='', markersize=10, label='Knee point'),
                    Line2D([0], [0], marker='D', color='none', markeredgecolor='#666', markersize=8, label='Dynamic baseline'),
                    Line2D([0], [0], marker='X', color='none', markeredgecolor='#666', markersize=8, label='MAXN baseline'),
                ]
                ax.legend(handles=legend_elements, fontsize=7, loc='upper right')

    plt.tight_layout()
    p = out_dir / 'pareto_2d_ept_vs_tpot.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_pareto_power_vs_tpot(frontiers, out_dir):
    """Power vs TPOT projection of Pareto frontier."""
    models, _ = _get_model_workloads(frontiers)
    lock_keys = [k for k in frontiers if k[2] == 'lock']
    repr_wls = sorted(set(k[1] for k in lock_keys))[:4]

    n_rows = len(repr_wls)
    n_cols = len(models)
    if n_rows == 0 or n_cols == 0:
        print('  Skipped plot_pareto_power_vs_tpot: no data')
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle('Pareto Frontier: Power vs TPOT (Lock Mode)', fontsize=14, fontweight='bold')

    for row_idx, wl in enumerate(repr_wls):
        for col_idx, model_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            key = (model_name, wl, 'lock')
            if key not in frontiers:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            data = frontiers[key]
            suffix = data['suffix']
            pwr_col = f'avg_power_w{suffix}'
            tpot_col = f'tpot_ms{suffix}'

            all_df = data['all_df']
            frontier_df = data['frontier_df']
            if all_df.empty or pwr_col not in all_df.columns:
                continue

            color = MODEL_COLORS.get(model_name, '#2196F3')
            valid = all_df[(all_df[pwr_col] > 0) & (all_df[tpot_col] > 0)]
            ax.scatter(valid[pwr_col], valid[tpot_col], c='#ddd', s=30, zorder=1,
                      edgecolors='#999', linewidths=0.5, alpha=0.6)

            if not frontier_df.empty:
                valid_f = frontier_df[(frontier_df[pwr_col] > 0) & (frontier_df[tpot_col] > 0)]
                if len(valid_f) >= 2:
                    sorted_f = valid_f.sort_values(pwr_col)
                    ax.plot(sorted_f[pwr_col], sorted_f[tpot_col], 'o-', color=color,
                           markersize=7, linewidth=2, zorder=3)

            ax.set_xlabel('Power (W)', fontsize=9)
            if col_idx == 0:
                ax.set_ylabel('TPOT (ms)', fontsize=9)
            ax.set_title(wl if col_idx == 0 else '', fontsize=9)
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = out_dir / 'pareto_power_vs_tpot.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_cross_model_frontier(frontiers, out_dir):
    """Overlay Pareto frontiers from different models on one plot for a representative workload."""
    # Find a workload that exists in all models
    lock_keys = [k for k in frontiers if k[2] == 'lock']
    wl_counts = {}
    for k in lock_keys:
        wl_counts[k[1]] = wl_counts.get(k[1], 0) + 1
    # Pick the workload with most models
    target_wl = max(wl_counts, key=wl_counts.get)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f'Cross-Model Pareto Frontier Comparison: {target_wl}', fontsize=14, fontweight='bold')

    projections = [
        ('energy_per_token_j', 'tpot_ms', 'Energy per Token (J)', 'TPOT (ms)'),
        ('avg_power_w', 'tpot_ms', 'Power (W)', 'TPOT (ms)'),
        ('energy_per_token_j', 'avg_power_w', 'Energy per Token (J)', 'Power (W)'),
    ]

    for col_idx, (x_base, y_base, x_label, y_label) in enumerate(projections):
        ax = axes[col_idx]
        for model_name in sorted(set(k[0] for k in lock_keys)):
            key = (model_name, target_wl, 'lock')
            if key not in frontiers:
                continue
            data = frontiers[key]
            frontier_df = data['frontier_df']
            suffix = data['suffix']
            x_col = f'{x_base}{suffix}'
            y_col = f'{y_base}{suffix}'

            if frontier_df.empty or x_col not in frontier_df.columns:
                continue

            label = MODEL_LABELS.get(model_name, model_name)
            color = MODEL_COLORS.get(model_name, '#999')
            valid_f = frontier_df[(frontier_df[x_col] > 0) & (frontier_df[y_col] > 0)]

            if len(valid_f) >= 2:
                sorted_f = valid_f.sort_values(x_col)
                ax.plot(sorted_f[x_col], sorted_f[y_col], 'o-', color=color,
                       markersize=8, linewidth=2, label=label)
            elif len(valid_f) == 1:
                ax.scatter(valid_f[x_col], valid_f[y_col], marker='o', s=80, color=color,
                          edgecolors='black', label=label)

            # Dynamic baseline
            cap_key = (model_name, target_wl, 'cap')
            if cap_key in frontiers:
                cap_data = frontiers[cap_key]['all_df']
                if 'control_mode' in cap_data.columns:
                    dyn = cap_data[cap_data['control_mode'] == 'dynamic']
                    if not dyn.empty and x_col in dyn.columns and y_col in dyn.columns:
                        ax.scatter(dyn[x_col], dyn[y_col], marker='D', s=100, c='none',
                                  edgecolors='#333', linewidths=1.5, zorder=4)

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'pareto_cross_model_comparison.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_alpha_sweep(frontiers, out_dir):
    """Show how alpha maps to frontier points for each model."""
    models = sorted(set(k[0] for k in frontiers if k[2] == 'lock'))
    if not models:
        print('  Skipped plot_alpha_sweep: no data')
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models) + 1, 5), squeeze=False)
    fig.suptitle('Alpha Sweep on Pareto Frontier (p512_o128)', fontsize=14, fontweight='bold')

    alphas = np.linspace(0, 1, 25)
    target_wl = 'p512_o128'

    for col_idx, model_name in enumerate(models):
        ax = axes[0, col_idx]
        label = MODEL_LABELS.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, '#999')

        key = (model_name, target_wl, 'lock')
        if key not in frontiers:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        data = frontiers[key]
        frontier_df = data['frontier_df']
        suffix = data['suffix']
        ept_col = f'energy_per_token_j{suffix}'
        tpot_col = f'tpot_ms{suffix}'
        pwr_col = f'avg_power_w{suffix}'

        if frontier_df.empty or ept_col not in frontier_df.columns:
            continue

        sorted_f = frontier_df[(frontier_df[ept_col] > 0) & (frontier_df[tpot_col] > 0)].sort_values(ept_col).reset_index(drop=True)
        n = len(sorted_f)
        if n == 0:
            continue

        ept_vals, tpot_vals, pwr_vals = [], [], []
        for a in alphas:
            idx = min(int(round(a * (n - 1))), n - 1)
            row = sorted_f.iloc[idx]
            ept_vals.append(row[ept_col])
            tpot_vals.append(row[tpot_col])
            pwr_vals.append(row[pwr_col])

        ax.plot(alphas, ept_vals, '-', color=color, linewidth=2, label='E/tok (J)')
        ax2 = ax.twinx()
        ax2.plot(alphas, tpot_vals, '--', color='#FF9800', linewidth=1.5, label='TPOT (ms)')
        ax2.plot(alphas, pwr_vals, ':', color='#9C27B0', linewidth=1.5, label='Power (W)')

        ax.set_xlabel('alpha (0=energy, 1=latency)', fontsize=11)
        ax.set_ylabel('E/tok (J)', fontsize=11, color=color)
        ax2.set_ylabel('TPOT (ms) / Power (W)', fontsize=9, color='#666')
        ax.set_title(f'{label}\n{n} Pareto points', fontsize=11, color=color)
        ax.grid(True, alpha=0.2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

    plt.tight_layout()
    p = out_dir / 'pareto_alpha_sweep.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_frontier_size_summary(frontiers, out_dir):
    """Bar chart: number of Pareto-optimal configs per (model, workload, source)."""
    rows = []
    for (model_name, wl, source), data in sorted(frontiers.items()):
        n_frontier = len(data['frontier_df'])
        n_total = len(data['all_df'])
        rows.append({
            'model': MODEL_LABELS.get(model_name, model_name),
            'source': source,
            'workload': wl,
            'n_frontier': n_frontier,
            'n_total': n_total,
            'color': MODEL_COLORS.get(model_name, '#999'),
        })

    if not rows:
        print('  Skipped plot_frontier_size_summary: no data')
        return

    df = pd.DataFrame(rows)

    # Separate lock and cap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Pareto Frontier Size: Non-Dominated Configs per Workload',
                fontsize=14, fontweight='bold')

    for ax, source, title in [(ax1, 'lock', 'Lock Mode (11 GPU freqs)'),
                               (ax2, 'cap', 'Cap Mode (4 caps)')]:
        sub = df[df['source'] == source].copy()
        if sub.empty:
            continue

        models_in = sorted(sub['model'].unique())
        workloads = sorted(sub['workload'].unique())
        n_models = len(models_in)
        n_wls = len(workloads)
        width = 0.8 / n_models

        for m_idx, model in enumerate(models_in):
            m_data = sub[sub['model'] == model]
            x = [workloads.index(w) + m_idx * width - 0.4 + width / 2 for w in m_data['workload']]
            color = m_data['color'].iloc[0] if not m_data['color'].empty else '#999'
            bars = ax.bar(x, m_data['n_frontier'], width=width * 0.9, color=color,
                        alpha=0.8, label=model, edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, m_data['n_frontier']):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                           str(val), ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_xticks(range(n_wls))
        ax.set_xticklabels([w.replace('p', '').replace('o', '→') for w in workloads],
                          rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Pareto-Optimal Configs', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    p = out_dir / 'pareto_frontier_size_summary.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def plot_workload_pareto_grids(frontiers, out_dir):
    """Per-workload Pareto grid for one model (default 7B), showing 3 projections."""
    target_model = 'Qwen2.5-7B-Instruct-Q4_K_M'
    if not any(k[0] == target_model and k[2] == 'lock' for k in frontiers):
        # Fall back to first available model
        target_model = next((k[0] for k in frontiers if k[2] == 'lock'), None)
    if target_model is None:
        print('  Skipped plot_workload_pareto_grids: no lock data')
        return

    model_wls = sorted(k[1] for k in frontiers if k[0] == target_model and k[2] == 'lock')
    n_wl = len(model_wl := model_wls[:12])
    if n_wl == 0:
        print('  Skipped plot_workload_pareto_grids: no workloads')
        return

    n_cols = 4
    n_rows = (n_wl + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows), squeeze=False)
    fig.suptitle(f'Pareto Frontier per Workload — {MODEL_LABELS.get(target_model, target_model)} (Lock Mode)',
                fontsize=14, fontweight='bold', y=1.01)

    from src.controller.pareto_selector import find_knee_point

    for idx, wl in enumerate(model_wl):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        key = (target_model, wl, 'lock')
        if key not in frontiers:
            ax.text(0.5, 0.5, f'{wl}\nno data', ha='center', va='center', transform=ax.transAxes)
            continue

        data = frontiers[key]
        suffix = data['suffix']
        ept_col = f'energy_per_token_j{suffix}'
        tpot_col = f'tpot_ms{suffix}'
        pwr_col = f'avg_power_w{suffix}'
        all_df = data['all_df']
        frontier_df = data['frontier_df']

        if all_df.empty or ept_col not in all_df.columns:
            continue

        valid = all_df[(all_df[ept_col] > 0) & (all_df[tpot_col] > 0)]
        ax.scatter(valid[ept_col], valid[tpot_col], c='#ddd', s=25, zorder=1,
                  edgecolors='#bbb', linewidths=0.3, alpha=0.6)

        if not frontier_df.empty:
            valid_f = frontier_df[(frontier_df[ept_col] > 0) & (frontier_df[tpot_col] > 0)]
            color = MODEL_COLORS.get(target_model, '#2196F3')
            if len(valid_f) >= 2:
                sorted_f = valid_f.sort_values(ept_col)
                ax.plot(sorted_f[ept_col], sorted_f[tpot_col], '-', color=color,
                       linewidth=1.5, zorder=3)
                # Knee point
                knee_idx = find_knee_point(sorted_f, ept_col, tpot_col)
                knee = sorted_f.iloc[knee_idx]
                ax.scatter([knee[ept_col]], [knee[tpot_col]], marker='*', s=100,
                          color='#E91E63', zorder=5, edgecolors='black', linewidths=0.6)
            elif len(valid_f) == 1:
                ax.scatter(valid_f[ept_col], valid_f[tpot_col], marker='o', s=60,
                          color=color, zorder=3, edgecolors='black')

        # Annotate frontier size
        n_f = len(frontier_df)
        ax.set_title(f'{wl} ({n_f} Pareto)', fontsize=10, pad=3)
        ax.set_xlabel('E/tok (J)', fontsize=8)
        ax.set_ylabel('TPOT (ms)', fontsize=8)
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(n_wl, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    p = out_dir / 'pareto_workload_grid.png'
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {p}')


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading rate tables...')
    lock_tables, cap_tables = load_all_tables()
    print(f'  Models: {list(lock_tables.keys())}')

    print('Computing Pareto frontiers...')
    frontiers = compute_all_frontiers(lock_tables, cap_tables)
    n_front = sum(1 for v in frontiers.values() if len(v['frontier_df']) > 0)
    print(f'  {len(frontiers)} (model, workload, source) combos, {n_front} with non-empty frontiers')

    print('Generating figures...')
    plot_pareto_2d(frontiers, out_dir)
    plot_pareto_power_vs_tpot(frontiers, out_dir)
    plot_cross_model_frontier(frontiers, out_dir)
    plot_alpha_sweep(frontiers, out_dir)
    plot_frontier_size_summary(frontiers, out_dir)
    plot_workload_pareto_grids(frontiers, out_dir)

    print(f'\nAll Pareto figures saved to {out_dir}/')


if __name__ == '__main__':
    main()
