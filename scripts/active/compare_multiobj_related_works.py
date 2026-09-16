#!/usr/bin/env python3
"""
Multi-objective comparison: EnergyInfra vs EdgeShark/FlashFlow in EMO framework.

Extends our MDR/JIR/HV/Composite Waste evaluation to include:
  - EdgeShark* (simulated = best_static from oracle gap data)
  - FlashFlow* (simulated from lock rate table + 715ms switching overhead)

3D objectives (all minimize): E/tok, TPOT, Power
"""

import sys, os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
import pandas as pd
import numpy as np

MODEL_ORDER = [
    'Qwen2.5-7B-Instruct-Q4_K_M',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M',
    'Qwen2.5-14B-Instruct-Q4_K_M',
]
MODEL_SHORT = {
    'Qwen2.5-7B-Instruct-Q4_K_M': '7B (Qwen2.5)',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M': '8B (Llama-3.1)',
    'Qwen2.5-14B-Instruct-Q4_K_M': '14B (Qwen2.5)',
}
SWITCH_OVERHEAD_MS = 715.0
SWITCH_POWER_SPIKE_W = 5.0

# ============================================================
# Load Data
# ============================================================
print("Loading data...")

# Lock rate tables
lock_rts = {}
for f in sorted(os.popen('ls data/rate_tables/lock_rate_table_*.parquet').read().strip().split('\n')):
    if not f: continue
    rt = pd.read_parquet(f)
    mk = rt['model'].iloc[0]
    if mk not in lock_rts:
        lock_rts[mk] = rt

# Oracle gap (has best_static, pareto, dynamic, maxn per-workload E2E results)
oracle_files = sorted(os.popen('ls data/oracle_gap_analysis/oracle_gap_*.csv').read().strip().split('\n'))
oracle = pd.concat([pd.read_csv(f) for f in oracle_files if f], ignore_index=True).drop_duplicates(
    subset=['model', 'workload', 'prompt_length', 'output_length'])

# Load E2E benchmark for more strategies
e2e_files = sorted(os.popen('ls data/cap_selector_benchmark/cap_selector_benchmark_*.csv').read().strip().split('\n'))
e2e = pd.concat([pd.read_csv(f) for f in e2e_files if f], ignore_index=True)

print(f"  Oracle: {len(oracle)} rows, E2E: {len(e2e)} rows")

# ============================================================
# Build unified per-workload dataset with all strategies
# ============================================================

def build_strategy_points():
    """
    For each model × workload, collect the 3D point (ept, tpot, power) for each strategy.
    Returns: list of dicts {model, workload, strategy, ept, tpot, power}
    """
    points = []

    for mk in MODEL_ORDER:
        oracle_sub = oracle[oracle['model'] == mk].copy()
        e2e_sub = e2e[e2e['model'] == mk].copy()

        for _, row in oracle_sub.iterrows():
            wl = row['workload']
            pl = row['prompt_length']
            ol = row['output_length']

            base = {'model': mk, 'workload': wl, 'pl': pl, 'ol': ol}

            # Dynamic
            points.append({**base, 'strategy': 'dynamic',
                           'ept': row['dynamic_ept'], 'tpot': row.get('dynamic_tpot', row.get('dynamic_tpms', 0)),
                           'power': row['dynamic_power']})
            # MAXN
            points.append({**base, 'strategy': 'maxn',
                           'ept': row['maxn_ept'], 'tpot': row.get('maxn_tpot', row.get('maxn_tpms', 0)),
                           'power': row['maxn_power']})
            # EdgeShark* = best_static
            points.append({**base, 'strategy': 'edgeshark',
                           'ept': row['best_static_ept'], 'tpot': row.get('best_static_tpot', row.get('best_static_tpms', 0)),
                           'power': row['best_static_power']})
            # EnergyInfra = pareto
            points.append({**base, 'strategy': 'pareto',
                           'ept': row['pareto_ept'], 'tpot': row.get('pareto_tpot', row.get('pareto_tpms', 0)),
                           'power': row['pareto_power']})

        # FlashFlow simulation
        if mk in lock_rts:
            rt = lock_rts[mk]
            for _, row in oracle_sub.iterrows():
                wl = row['workload']
                pl = row['prompt_length']
                ol = row['output_length']

                wl_data = rt[(rt['workload'] == wl) &
                              (rt['prompt_length'] == pl) &
                              (rt['output_length'] == ol)]
                if len(wl_data) == 0:
                    wl_data = rt[rt['workload'] == wl]
                if len(wl_data) == 0:
                    continue

                prefill_data = wl_data[wl_data['phase'] == 'mixed']
                decode_data = wl_data[wl_data['phase'] == 'decode']

                if len(prefill_data) == 0 or len(decode_data) == 0:
                    prefill_data = wl_data.copy()
                    decode_data = wl_data.copy()

                best_pf = prefill_data.loc[prefill_data['energy_per_token_j_median'].idxmin()]
                best_dc = decode_data.loc[decode_data['energy_per_token_j_median'].idxmin()]

                pf_time = best_pf['ttft_ms_median'] / 1000.0 if best_pf['ttft_ms_median'] > 0 else 0.5
                dc_tpms = best_dc['tpot_ms_median']
                base_power = row.get('maxn_power', 45)

                pf_energy = pf_time * best_pf['avg_power_w_median']
                dc_energy = (dc_tpms * ol / 1000.0) * best_dc['avg_power_w_median']
                sw_energy = (SWITCH_OVERHEAD_MS / 1000.0) * (base_power + SWITCH_POWER_SPIKE_W)

                total_e = pf_energy + dc_energy + sw_energy
                total_t = pf_time + SWITCH_OVERHEAD_MS / 1000.0 + dc_tpms * ol / 1000.0

                points.append({
                    'model': mk, 'workload': wl, 'pl': pl, 'ol': ol,
                    'strategy': 'flashflow',
                    'ept': total_e / max(ol, 1),
                    'tpot': total_t * 1000 / max(ol, 1),
                    'power': total_e / total_t if total_t > 0 else 0,
                })

    return pd.DataFrame(points)

# Also load all strategies from E2E benchmark
def build_e2e_points():
    """Get additional strategies from E2E benchmark (min_energy, slo_50ms, etc.)"""
    points = []
    for _, row in e2e.iterrows():
        points.append({
            'model': row['model'],
            'workload': row['workload'],
            'pl': row['prompt_length'],
            'ol': row['output_length'],
            'strategy': row['strategy'],
            'ept': row['energy_per_token_j'],
            'tpot': row['tpot_ms'],
            'power': row['avg_power_w'],
        })
    return pd.DataFrame(points)

all_points = build_strategy_points()
e2e_points = build_e2e_points()

print(f"  Strategy points: {len(all_points)} ({all_points['strategy'].unique().tolist()})")
print(f"  E2E points: {len(e2e_points)}")

# ============================================================
# Multi-Objective Metrics Computation
# ============================================================

def _merge_on_common(df_base, df_cmp):
    merge_keys = [k for k in ['model', 'workload', 'pl', 'ol'] if k in df_base.columns and k in df_cmp.columns]
    if not merge_keys:
        merge_keys = ['model', 'workload']
    return df_base.merge(df_cmp, on=merge_keys, suffixes=('_b', '_c'))


def compute_mdr(df_base, df_cmp, base_name='pareto', cmp_name='edgeshark'):
    """
    Multi-Objective Dominance Rate:
    For each workload, check if base dominates cmp in ALL 3 dimensions.
    MDR = fraction of workloads where base strictly dominates cmp.
    """
    merge_keys = [k for k in ['model', 'workload', 'pl', 'ol'] if k in df_base.columns and k in df_cmp.columns]
    if not merge_keys:
        merge_keys = ['model', 'workload']
    merged = df_base.merge(df_cmp, on=merge_keys, suffixes=('_b', '_c'))
    if len(merged) == 0:
        return 0.0, 0, 0

    # Strict dominance: base <= cmp in all 3 dims, strict in at least 1
    dominates = (
        (merged['ept_b'] <= merged['ept_c']) &
        (merged['tpot_b'] <= merged['tpot_c']) &
        (merged['power_b'] <= merged['power_c']) &
        (
            (merged['ept_b'] < merged['ept_c']) |
            (merged['tpot_b'] < merged['tpot_c']) |
            (merged['power_b'] < merged['power_c'])
        )
    )
    total = len(merged)
    count = dominates.sum()
    return count / total if total > 0 else 0.0, count, total


def compute_jir(df_base, df_cmp, base_name='pareto', cmp_name='edgeshark'):
    """
    Joint Improvement Ratio:
    Geometric mean of improvement ratios when base dominates cmp in all dims.
    """
    merged = _merge_on_common(df_base, df_cmp)
    if len(merged) == 0:
        return 0.0

    dominates = (
        (merged['ept_b'] <= merged['ept_c']) &
        (merged['tpot_b'] <= merged['tpot_c']) &
        (merged['power_b'] <= merged['power_c']) &
        (
            (merged['ept_b'] < merged['ept_c']) |
            (merged['tpot_b'] < merged['tpot_c']) |
            (merged['power_b'] < merged['power_c'])
        )
    )

    if dominates.sum() == 0:
        return 0.0

    dom = merged[dominates].copy()
    # Geometric mean of relative improvements (positive when base is better)
    eps = 1e-10
    ept_ratio = (dom['ept_c'] - dom['ept_b']) / dom['ept_c']
    tpot_ratio = (dom['tpot_c'] - dom['tpot_b']) / dom['tpot_c']
    power_ratio = (dom['power_c'] - dom['power_b']) / dom['power_c']

    # Only compute geometric mean where strictly better in all 3
    strict = (ept_ratio > 0) & (tpot_ratio > 0) & (power_ratio > 0)
    if strict.sum() == 0:
        return 0.0

    log_gm = np.mean(np.log(ept_ratio[strict] + 1) * np.log(tpot_ratio[strict] + 1) * np.log(power_ratio[strict] + 1))
    return np.exp(log_gm ** (1/3)) - 1  # Convert back from log space


def compute_composite_waste(df_base, df_cmp, ideal_points):
    """
    Composite Waste: Average normalized distance to ideal point.
    Lower is better (0 = on ideal point).
    """
    merged = _merge_on_common(df_base, df_cmp)
    if len(merged) == 0:
        return 0.0

    wastes = []
    for _, row in merged.iterrows():
        wl = row['workload']
        mk = row['model']

        # Find ideal point for this workload
        key = (mk, wl)
        if key not in ideal_points:
            continue
        ideal = ideal_points[key]

        # Normalized distance for each strategy
        def dist_to_ideal(ept, tpot, power):
            ranges = ideal['ranges']
            if ranges['ept'] == 0 or ranges['tpot'] == 0 or ranges['power'] == 0:
                return 0
            de = (ept - ideal['ept']) / ranges['ept']
            dt = (tpot - ideal['tpot']) / ranges['tpot']
            dp = (power - ideal['power']) / ranges['power']
            return np.sqrt(de**2 + dt**2 + dp**2)

        d_base = dist_to_ideal(row['ept_b'], row['tpot_b'], row['power_b'])
        d_cmp = dist_to_ideal(row['ept_c'], row['tpot_c'], row['power_c'])
        wastes.append(d_base - d_cmp)  # negative = base is closer to ideal

    return np.mean(wastes) * 100 if wastes else 0.0  # as percentage


def compute_hypervolume(df, strategy, ideal_points, all_maxes):
    """
    Compute approximate Hypervolume for a strategy's solution set.
    HV = volume of objective space dominated by the strategy.
    Reference point = worst-case across all strategies per model.
    """
    sub = df[df['strategy'] == strategy].copy()
    if len(sub) == 0:
        return 0.0

    hv_per_model = {}
    for mk in sub['model'].unique():
        mk_sub = sub[sub['model'] == mk]
        ref = all_maxes.get(mk, {})
        if not ref:
            continue

        # Normalized coordinates (0-1, where 0 = best)
        ref_ept = ref['ept'] * 1.1
        ref_tpms = ref['tpot'] * 1.1
        ref_pwr = ref['power'] * 1.1

        # For each workload, compute dominated volume
        total_vol = 0
        for _, row in mk_sub.iterrows():
            wl = row['workload']
            key = (mk, wl)

            if key in ideal_points:
                ideal = ideal_points[key]
                ranges = ideal['ranges']
                if ranges['ept'] > 0 and ranges['tpot'] > 0 and ranges['power'] > 0:
                    # Contribution = volume between ideal and this point
                    x = (ref_ept - max(row['ept'], ideal['ept'])) / ref_ept
                    y = (ref_tpms - max(row['tpot'], ideal['tpot'])) / ref_tpms
                    z = (ref_pwr - max(row['power'], ideal['power'])) / ref_pwr
                    total_vol += max(0, x) * max(0, y) * max(0, z)
                else:
                    total_vol += 0
            else:
                total_vol += 0

        hv_per_model[mk] = total_vol

    return hv_per_model

# ============================================================
# Compute Ideal Points & Ranges
# ============================================================

print("\nComputing multi-objective metrics...")

# Combine all_points and E2E points
combined = pd.concat([all_points, e2e_points], ignore_index=True)

ideal_points = {}
all_maxes = {}

for (mk, wl), grp in combined.groupby(['model', 'workload']):
    if 'pl' in grp.columns and 'ol' in grp.columns:
        # Use the finest granularity
        sub = grp.groupby(['model', 'workload', 'pl', 'ol', 'strategy']).agg({
            'ept': 'mean', 'tpot': 'mean', 'power': 'mean'
        }).reset_index()
        # Ideal = best across all strategies
        ideal_ept = sub['ept'].min()
        ideal_tpms = sub['tpot'].min()
        ideal_pwr = sub['power'].min()
        max_ept = sub['ept'].max()
        max_tpms = sub['tpot'].max()
        max_pwr = sub['power'].max()
    else:
        sub = grp.groupby(['model', 'workload', 'strategy']).agg({
            'ept': 'mean', 'tpot': 'mean', 'power': 'mean'
        }).reset_index()
        ideal_ept = sub['ept'].min()
        ideal_tpms = sub['tpot'].min()
        ideal_pwr = sub['power'].min()
        max_ept = sub['ept'].max()
        max_tpms = sub['tpot'].max()
        max_pwr = sub['power'].max()

    ideal_points[(mk, wl)] = {
        'ept': ideal_ept, 'tpot': ideal_tpms, 'power': ideal_pwr,
        'ranges': {
            'ept': max_ept - ideal_ept if max_ept > ideal_ept else 1,
            'tpot': max_tpms - ideal_tpms if max_tpms > ideal_tpms else 1,
            'power': max_pwr - ideal_pwr if max_pwr > ideal_pwr else 1,
        }
    }

for mk in combined['model'].unique():
    mk_sub = combined[combined['model'] == mk]
    all_maxes[mk] = {
        'ept': mk_sub['ept'].max(),
        'tpot': mk_sub['tpot'].max(),
        'power': mk_sub['power'].max(),
    }

# ============================================================
# Strategy-level aggregations (per workload mean)
# ============================================================

# Aggregate all_points per model/workload/strategy (take mean across repeats)
strat_agg = all_points.groupby(['model', 'workload', 'strategy']).agg(
    ept=('ept', 'mean'),
    tpot=('tpot', 'mean'),
    power=('power', 'mean'),
).reset_index()

# Also aggregate E2E for additional strategies
e2e_agg = e2e_points.groupby(['model', 'workload', 'strategy']).agg(
    ept=('ept', 'mean'),
    tpot=('tpot', 'mean'),
    power=('power', 'mean'),
).reset_index()

# ============================================================
# Compute MDR, JIR, Waste for each pairwise comparison
# ============================================================

print('\n' + '=' * 90)
print('  MULTI-OBJECTIVE COMPARISON: EnergyInfra (Pareto) vs Related Works')
print('  Objectives: E/tok, TPOT, Power (all minimize)')
print('=' * 90)

strategies_to_compare = ['dynamic', 'maxn', 'edgeshark', 'flashflow']
strategy_labels = {
    'dynamic': 'Dynamic (默认)',
    'maxn': 'MAXN (全频)',
    'edgeshark': 'EdgeShark* (静态cap)',
    'flashflow': 'FlashFlow* (phase切频)',
}

# --- Per-model MDR table ---
print()
print('  Multi-Objective Dominance Rate (MDR): Pareto dominates other?')
print('  (Fraction of workloads where Pareto ≤ other in ALL 3 dimensions)')
print()
print('  %-22s | %10s | %10s | %10s' % ('Comparison', '7B', '8B', '14B'))
print('  %-22s | %10s | %10s | %10s' % ('-' * 22, '-' * 10, '-' * 10, '-' * 10))

def get_strat_df(strat, mk):
    """Get strategy dataframe for a model, trying strat_agg then e2e_agg."""
    cols_a = ['model', 'workload', 'ept', 'tpot', 'power']
    cols_b = ['model', 'workload', 'ept', 'tpot', 'power']
    if 'pl' in strat_agg.columns:
        cols_a = ['model', 'workload', 'pl', 'ol', 'ept', 'tpot', 'power']
    if 'pl' in e2e_agg.columns:
        cols_b = ['model', 'workload', 'pl', 'ol', 'ept', 'tpot', 'power']

    df = strat_agg[(strat_agg['model'] == mk) & (strat_agg['strategy'] == strat)][cols_a]
    if len(df) == 0:
        df = e2e_agg[(e2e_agg['model'] == mk) & (e2e_agg['strategy'] == strat)][cols_b]
    return df

for strat in strategies_to_compare:
    rates = []
    for mk in MODEL_ORDER:
        df_b = get_strat_df('pareto', mk)
        df_c = get_strat_df(strat, mk)

        if len(df_b) == 0 or len(df_c) == 0:
            rates.append('--')
            continue

        mdr, cnt, total = compute_mdr(df_b, df_c)
        rates.append(f'{mdr*100:.1f}% ({cnt}/{total})')

    label = strategy_labels.get(strat, strat)
    print('  %-22s | %10s | %10s | %10s' % (
        f'Pareto vs {label}', rates[0], rates[1], rates[2]))

# --- Reverse: other dominates Pareto? ---
print()
print('  Reverse MDR: Other dominates Pareto?')
print('  %-22s | %10s | %10s | %10s' % ('Comparison', '7B', '8B', '14B'))
print('  %-22s | %10s | %10s | %10s' % ('-' * 22, '-' * 10, '-' * 10, '-' * 10))

for strat in strategies_to_compare:
    rates = []
    for mk in MODEL_ORDER:
        df_b = get_strat_df(strat, mk)
        df_c = get_strat_df('pareto', mk)

        if len(df_b) == 0 or len(df_c) == 0:
            rates.append('--')
            continue

        mdr, cnt, total = compute_mdr(df_b, df_c)
        rates.append(f'{mdr*100:.1f}% ({cnt}/{total})')

    label = strategy_labels.get(strat, strat)
    print('  %-22s | %10s | %10s | %10s' % (
        f'{label} vs Pareto', rates[0], rates[1], rates[2]))

# --- Composite Waste comparison ---
print()
print('  Composite Waste Distance to Ideal Point:')
print('  (Negative = Pareto is closer to ideal; more negative = better)')
print()
print('  %-22s | %10s | %10s | %10s' % ('Comparison', '7B', '8B', '14B'))
print('  %-22s | %10s | %10s | %10s' % ('-' * 22, '-' * 10, '-' * 10, '-' * 10))

for strat in strategies_to_compare:
    wastes = []
    for mk in MODEL_ORDER:
        df_b = get_strat_df('pareto', mk)
        df_c = get_strat_df(strat, mk)

        if len(df_b) == 0 or len(df_c) == 0:
            wastes.append('--')
            continue

        waste = compute_composite_waste(df_b, df_c, ideal_points)
        wastes.append(f'{waste:+.1f}pp')

    label = strategy_labels.get(strat, strat)
    print('  %-22s | %10s | %10s | %10s' % (
        f'Pareto vs {label}', wastes[0], wastes[1], wastes[2]))

# --- Hypervolume comparison ---
print()
print('  Hypervolume Indicator (HV) — dominated objective space volume')
print('  (Higher = better; computed per-model across all workloads)')
print()

# Use the full combined data with all strategies
hv_strategies = ['pareto', 'dynamic', 'maxn', 'edgeshark', 'flashflow',
                 'min_energy', 'slo_50ms', 'slo_45ms', 'alpha_03', 'alpha_07', 'pwr_45w']

# Merge strat_agg with e2e_agg
full_agg = strat_agg.copy()
for _, row in e2e_agg.iterrows():
    mask = (full_agg['model'] == row['model']) & (full_agg['workload'] == row['workload']) & (full_agg['strategy'] == row['strategy'])
    if mask.any():
        continue
    full_agg = pd.concat([full_agg, pd.DataFrame([row])], ignore_index=True)

# Build per-strategy per-model point sets
hv_table = []
for mk in MODEL_ORDER:
    mk_all = full_agg[full_agg['model'] == mk].copy()

    for strat in hv_strategies:
        strat_sub = mk_all[mk_all['strategy'] == strat]
        if len(strat_sub) == 0:
            hv_table.append({'model': mk, 'strategy': strat, 'hv': 0, 'n_points': 0})
            continue

        # Compute HV using reference point = worst across all strategies
        ref_ept = mk_all['ept'].max() * 1.05
        ref_tpms = mk_all['tpot'].max() * 1.05
        ref_pwr = mk_all['power'].max() * 1.05

        # Simple HV: sum of box volumes for each point
        total_hv = 0
        for _, row in strat_sub.iterrows():
            x = max(0, ref_ept - row['ept'])
            y = max(0, ref_tpms - row['tpot'])
            z = max(0, ref_pwr - row['power'])
            total_hv += x * y * z

        hv_table.append({'model': mk, 'strategy': strat, 'hv': total_hv, 'n_points': len(strat_sub)})

hv_df = pd.DataFrame(hv_table)

# Print HV table
print('  %-14s' % 'Strategy', end='')
for mk in MODEL_ORDER:
    print(' | %-18s' % MODEL_SHORT[mk], end='')
print()
print('  ' + '-' * 78)

pareto_hv = {}
for mk in MODEL_ORDER:
    row = hv_df[(hv_df['model'] == mk) & (hv_df['strategy'] == 'pareto')]
    pareto_hv[mk] = row['hv'].values[0] if len(row) > 0 else 0

# Sort strategies by average HV ratio
strat_order = ['pareto', 'edgeshark', 'flashflow', 'dynamic', 'maxn',
               'min_energy', 'slo_50ms', 'alpha_07', 'pwr_45w', 'slo_45ms', 'alpha_03']

for strat in strat_order:
    label = strat + (' <<<' if strat == 'pareto' else '')
    if strat in ['edgeshark', 'flashflow']:
        label = strat + '*' + (' <<<' if strat == 'pareto' else '')
    print('  %-14s' % label, end='')
    for mk in MODEL_ORDER:
        row = hv_df[(hv_df['model'] == mk) & (hv_df['strategy'] == strat)]
        if len(row) > 0 and row['hv'].values[0] > 0:
            h = row['hv'].values[0]
            ratio = pareto_hv[mk] / h if h > 0 else 0
            if ratio > 10:
                r_str = f'{ratio:.0f}×'
            elif ratio > 1.5:
                r_str = f'{ratio:.1f}×'
            else:
                r_str = f'{ratio:.2f}×'
            print(f' | %10.1f (%6s)' % (h, r_str), end='')
        else:
            print(' | %10s (%6s)' % ('--', '--'), end='')
    print()

print()
print('  Note: numbers = HV value; () = Pareto HV / Strategy HV ratio')

# --- Per-model pairwise Pareto vs EdgeShark/FlashFlow detailed ---
print()
print('=' * 90)
print('  PER-WORKLOAD 3D POINTS: Pareto vs EdgeShark* vs FlashFlow*')
print('=' * 90)

for mk in MODEL_ORDER:
    pareto_sub = strat_agg[(strat_agg['model'] == mk) & (strat_agg['strategy'] == 'pareto')].copy()
    edge_sub = strat_agg[(strat_agg['model'] == mk) & (strat_agg['strategy'] == 'edgeshark')].copy()
    flash_sub = strat_agg[(strat_agg['model'] == mk) & (strat_agg['strategy'] == 'flashflow')].copy()

    if len(pareto_sub) == 0 or len(edge_sub) == 0:
        continue

    merged = pareto_sub.merge(edge_sub, on=['model', 'workload'], suffixes=('_p', '_e'))
    if len(flash_sub) > 0:
        merged = merged.merge(flash_sub, on=['model', 'workload'])
        merged.rename(columns={'ept': 'ept_f', 'tpot': 'tpot_f', 'power': 'power_f'}, inplace=True)

    print()
    print(f'  [{MODEL_SHORT[mk]}]')
    print('  %-14s | %8s | %8s | %8s | %7s | %7s | %7s | %7s' % (
        'Workload', 'P_E/tok', 'E_E/tok', 'F_E/tok', 'P_Tpot', 'E_Tpot', 'F_Tpot', 'P_Pwr'))
    print('  %-14s | %8s | %8s | %8s | %7s | %7s | %7s | %7s' % (
        '-' * 14, '-' * 8, '-' * 8, '-' * 8, '-' * 7, '-' * 7, '-' * 7, '-' * 7))

    for _, row in merged.sort_values('workload').iterrows():
        f_ept = f"{row.get('ept_f', 0):.4f}" if 'ept_f' in row and pd.notna(row.get('ept_f')) else '--'
        f_tpms = f"{row.get('tpot_f', 0):.1f}" if 'tpot_f' in row and pd.notna(row.get('tpot_f')) else '--'
        print('  %-14s | %8.4f | %8.4f | %8s | %7.1f | %7.1f | %7s | %7.1f' % (
            row['workload'],
            row['ept_p'], row['ept_e'], f_ept,
            row['tpot_p'], row['tpot_e'], f_tpms,
            row['power_p']))

# ============================================================
# Summary
# ============================================================
print()
print('=' * 90)
print('  MULTI-OBJECTIVE SUMMARY')
print('=' * 90)
print()
print('  Key findings:')
print()

# Count Pareto dominance
for mk in MODEL_ORDER:
    pareto = strat_agg[(strat_agg['model'] == mk) & (strat_agg['strategy'] == 'pareto')]
    edge = strat_agg[(strat_agg['model'] == mk) & (strat_agg['strategy'] == 'edgeshark')]
    flash = strat_agg[(strat_agg['model'] == mk) & (strat_agg['strategy'] == 'flashflow')]

    if len(pareto) > 0 and len(edge) > 0:
        merged = pareto.merge(edge, on=['model', 'workload'], suffixes=('_p', '_e'))
        dom_all3 = ((merged['ept_p'] <= merged['ept_e']) &
                     (merged['tpot_p'] <= merged['tpot_e']) &
                     (merged['power_p'] <= merged['power_e']))
        dom_strict = dom_all3 & (
            (merged['ept_p'] < merged['ept_e']) |
            (merged['tpot_p'] < merged['tpot_e']) |
            (merged['power_p'] < merged['power_e']))
        rev_strict = ((merged['ept_e'] <= merged['ept_p']) &
                       (merged['tpot_e'] <= merged['tpot_p']) &
                       (merged['power_e'] <= merged['power_p'])) & (
            (merged['ept_e'] < merged['ept_p']) |
            (merged['tpot_e'] < merged['tpot_p']) |
            (merged['power_e'] < merged['power_p']))
        print(f'  [{MODEL_SHORT[mk]}] Pareto vs EdgeShark*:')
        print(f'    Pareto dominates:  {dom_strict.sum()}/{len(merged)} workloads')
        print(f'    EdgeShark dom.:    {rev_strict.sum()}/{len(merged)} workloads')

    if len(pareto) > 0 and len(flash) > 0:
        merged = pareto.merge(flash, on=['model', 'workload'], suffixes=('_p', '_f'))
        dom_strict = ((merged['ept_p'] <= merged['ept_f']) &
                       (merged['tpot_p'] <= merged['tpot_f']) &
                       (merged['power_p'] <= merged['power_f'])) & (
            (merged['ept_p'] < merged['ept_f']) |
            (merged['tpot_p'] < merged['tpot_f']) |
            (merged['power_p'] < merged['power_f']))
        rev_strict = ((merged['ept_f'] <= merged['ept_p']) &
                       (merged['tpot_f'] <= merged['tpot_p']) &
                       (merged['power_f'] <= merged['power_p'])) & (
            (merged['ept_f'] < merged['ept_p']) |
            (merged['tpot_f'] < merged['tpot_p']) |
            (merged['power_f'] < merged['power_p']))
        print(f'    Pareto vs FlashFlow*: Pareto dom. {dom_strict.sum()}/{len(merged)}, '
              f'FlashFlow dom. {rev_strict.sum()}/{len(merged)}')
    print()
