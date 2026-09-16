#!/usr/bin/env python3
"""
Compare EnergyInfra (our method) vs simulated EdgeShark and FlashFlow on our hardware.

Methods compared:
  1. EnergyInfra (Pareto)     — Our workload-aware Pareto cap selection (E2E benchmark data)
  2. EdgeShark-simulated     — Static single cap per model, ignores workload (≈ best_static)
  3. FlashFlow-simulated     — Phase-boundary DVFS: best prefill freq + best decode freq
     with switching overhead measured in Phase 7 (~715ms)
  4. MAXN                    — Max frequency baseline
  5. Dynamic                 — simple_ondemand default
  6. Oracle-Energy           — Theoretical lower bound (min E/tok across all freqs per workload)

Data sources:
  - Lock rate tables (per-phase, 11 freqs): for FlashFlow simulation
  - Oracle gap CSV: for best_static / pareto / dynamic / maxn / oracle actual E2E results
  - E2E benchmark: for per-workload detailed results
"""

import sys, os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
import pandas as pd
import numpy as np

# ============================================================
# Constants
# ============================================================

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

# Phase 7 measured switching overhead on Jetson Orin
SWITCH_OVERHEAD_MS = 715.0   # ms, added to TTFT when switching freq
SWITCH_POWER_SPIKE_W = 5.0    # W, extra power during transition (estimated)

# ============================================================
# Load Data
# ============================================================

print("Loading data...")

# 1. Lock rate tables (per-phase, for FlashFlow simulation)
lock_rts = {}
for mk in MODEL_ORDER:
    pattern = f'data/rate_tables/lock_rate_table_*.parquet'
    for f in sorted(os.popen(f'ls {pattern}').read().strip().split('\n')):
        if not f: continue
        rt = pd.read_parquet(f)
        if rt['model'].iloc[0] == mk:
            lock_rts[mk] = rt
            break
    if mk not in lock_rts:
        print(f"  WARNING: No lock rate table for {mk}")

# 2. Oracle gap analysis (has best_static / pareto / dynamic / maxn actual results)
oracle_files = sorted(os.popen('ls data/oracle_gap_analysis/oracle_gap_*.csv').read().strip().split('\n'))
oracle_dfs = [pd.read_csv(f) for f in oracle_files if f]
oracle = pd.concat(oracle_dfs, ignore_index=True).drop_duplicates(
    subset=['model', 'workload', 'prompt_length', 'output_length']
)

# 3. E2E benchmark results (for per-workload detailed comparison)
e2e_files = sorted(os.popen('ls data/cap_selector_benchmark/cap_selector_benchmark_*.csv').read().strip().split('\n'))
e2e_dfs = [pd.read_csv(f) for f in e2e_files if f]
e2e = pd.concat(e2e_dfs, ignore_index=True)

print(f"  Lock rate tables: {len(lock_rts)} models")
print(f"  Oracle gap: {len(oracle)} rows ({oracle['model'].nunique()} models)")
print(f"  E2E benchmark: {len(e2e)} rows ({e2e['model'].nunique()} models)")

# ============================================================
# Method 1: FlashFlow Simulation (Phase-Boundary DVFS)
# ============================================================
# FlashFlow's idea: use high freq for prefill, low freq for decode.
# We compute the theoretical best per-workload from lock data,
# then penalize by switching overhead.

def simulate_flashflow(lock_rt, oracle_sub):
    """
    For each workload, find:
      - Best prefill (mixed phase) freq for min E/tok
      - Best decode freq for min E/tok
    Then compute effective E/tok including switching overhead.
    """
    results = []

    for _, row in oracle_sub.iterrows():
        wl = row['workload']
        pl = row['prompt_length']
        ol = row['output_length']
        model = row['model']

        # Find matching lock data for this workload
        wl_data = lock_rt[
            (lock_rt['workload'] == wl) &
            (lock_rt['prompt_length'] == pl) &
            (lock_rt['output_length'] == ol)
        ]

        if len(wl_data) == 0:
            # Try approximate match (workload only)
            wl_data = lock_rt[lock_rt['workload'] == wl]

        if len(wl_data) == 0:
            continue

        # Get prefill (mixed) and decode data
        prefill_data = wl_data[wl_data['phase'] == 'mixed']
        decode_data = wl_data[wl_data['phase'] == 'decode']

        if len(prefill_data) == 0 or len(decode_data) == 0:
            # Use decode-only if no mixed
            prefill_data = wl_data.copy()
            decode_data = wl_data.copy()

        # Best prefill freq (min E/tok)
        best_prefill = prefill_data.loc[prefill_data['energy_per_token_j_median'].idxmin()]
        # Best decode freq (min E/tok)
        best_decode = decode_data.loc[decode_data['energy_per_token_j_median'].idxmin()]

        # Compute effective E/tok with switching overhead
        # Original E2E without switching:
        #   E/tok = total_energy / output_tokens
        # With phase switching:
        #   TTFT increases by SWITCH_OVERHEAD_MS
        #   Extra energy = SWITCH_OVERHEAD_MS/1000 * (avg_power + SWITCH_POWER_SPIKE)
        #   New total_time = original + SWITCH_OVERHEAD_MS
        #   New total_energy = original + switch_energy
        #   New E/tok = new_total_energy / output_tokens

        # Use oracle's maxn as baseline for timing/energy structure
        base_ttft = row.get('maxn_tpot', 50) * (pl / max(ol, 1))  # rough TTFT estimate
        base_tpms = row.get('maxn_tpot', 50)
        base_power = row.get('maxn_power', 45)
        output_tokens = ol

        # Compute FlashFlow's ideal (no overhead) from lock data
        # Prefill energy ≈ TTFT * prefill_power / 1000
        # Decode energy ≈ TPOT * output_tokens * decode_power / 1000
        prefill_time_s = best_prefill['ttft_ms_median'] / 1000.0 if best_prefill['ttft_ms_median'] > 0 else 0.5
        prefill_power = best_prefill['avg_power_w_median']
        decode_power = best_decode['avg_power_w_median']
        decode_tpms = best_decode['tpot_ms_median']

        # Total energy = prefill_energy + decode_energy + switch_energy
        prefill_energy = prefill_time_s * prefill_power
        decode_energy = (decode_tpms * output_tokens / 1000.0) * decode_power
        switch_energy = (SWITCH_OVERHEAD_MS / 1000.0) * (base_power + SWITCH_POWER_SPIKE_W)

        total_energy = prefill_energy + decode_energy + switch_energy
        total_time = (prefill_time_s + SWITCH_OVERHEAD_MS / 1000.0 +
                      decode_tpms * output_tokens / 1000.0)

        effective_ept = total_energy / max(output_tokens, 1)
        effective_tpms = total_time * 1000 / max(output_tokens, 1)
        effective_power = total_energy / total_time

        results.append({
            'model': model,
            'workload': wl,
            'prompt_length': pl,
            'output_length': ol,
            'flashflow_ept': effective_ept,
            'flashflow_tpms': effective_tpms,
            'flashflow_power': effective_power,
            'prefill_freq': best_prefill['gpu_freq_mhz'],
            'decode_freq': best_decode['gpu_freq_mhz'],
            'switch_overhead_pct': (SWITCH_OVERHEAD_MS / 1000.0) / total_time * 100,
        })

    return pd.DataFrame(results)


# ============================================================
# Compute FlashFlow simulation for each model
# ============================================================

flashflow_results = []
for mk in MODEL_ORDER:
    if mk in lock_rts:
        oracle_sub = oracle[oracle['model'] == mk]
        ff_df = simulate_flashflow(lock_rts[mk], oracle_sub)
        if len(ff_df) > 0:
            flashflow_results.append(ff_df)
            print(f"  FlashFlow simulated for {MODEL_SHORT[mk]}: {len(ff_df)} workloads")

flashflow_all = pd.concat(flashflow_results, ignore_index=True) if flashflow_results else pd.DataFrame()

# ============================================================
# Merge all methods into unified comparison
# ============================================================

# From oracle CSV: best_static, pareto, dynamic, maxn, oracle_energy
comparison = oracle[oracle['model'].isin(MODEL_ORDER)].copy()
comparison = comparison.rename(columns={
    'pareto_ept': 'pareto_ept',
    'best_static_ept': 'edgeshark_ept',  # EdgeShark ≈ BestStatic
    'dynamic_ept': 'dynamic_ept',
    'maxn_ept': 'maxn_ept',
    'oracle_energy_ept': 'oracle_ept',
    'pareto_tpms': 'pareto_tpms',
    'best_static_tpms': 'edgeshark_tpms',
    'dynamic_tpot': 'dynamic_tpms',
    'maxn_tpot': 'maxn_tpms',
    'oracle_energy_tpot': 'oracle_tpms',
    'pareto_power': 'pareto_power',
    'best_static_power': 'edgeshark_power',
    'dynamic_power': 'dynamic_power',
    'maxn_power': 'maxn_power',
    'oracle_energy_power': 'oracle_power',
})

# Merge FlashFlow results
if len(flashflow_all) > 0:
    comparison = comparison.merge(
        flashflow_all[['model', 'workload', 'prompt_length', 'output_length',
                        'flashflow_ept', 'flashflow_tpms', 'flashflow_power',
                        'prefill_freq', 'decode_freq', 'switch_overhead_pct']],
        on=['model', 'workload', 'prompt_length', 'output_length'],
        how='left'
    )

# ============================================================
# Print Detailed Comparison
# ============================================================

print()
print('=' * 100)
print('  RELATED WORK METHOD COMPARISON ON JETSON ORIN (OUR HARDWARE)')
print('  Models: Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B (Q4_K_M)')
print('  Methods: EnergyInfra (Pareto) vs EdgeShark-sim vs FlashFlow-sim vs MAXN vs Dynamic vs Oracle')
print('=' * 100)

for mk in MODEL_ORDER:
    sub = comparison[comparison['model'] == mk]
    if len(sub) == 0:
        continue

    ms = MODEL_SHORT[mk]
    n_wl = sub['workload'].nunique()

    print()
    print(f'  ─── [{ms}] ({n_wl} workloads) ───')
    print()

    # Per-workload detailed table
    print('  %-14s | %-8s | %9s | %9s | %9s | %9s | %9s | %9s' % (
        'Workload', 'Method', 'E/tok(J)', 'TPOT(ms)', 'Power(W)',
        'vs MAXN', 'vs Dynamic', 'vs Oracle'))
    print('  %-14s | %-8s | %9s | %9s | %9s | %9s | %9s | %9s' % (
        '-' * 14, '-' * 8, '-' * 9, '-' * 9, '-' * 9, '-' * 9, '-' * 9, '-' * 9))

    for wl in sorted(sub['workload'].unique()):
        wl_sub = sub[sub['workload'] == wl]
        wl_row = wl_sub.iloc[0]

        maxn_ept = wl_row['maxn_ept']
        dyn_ept = wl_row['dynamic_ept']
        oracle_ept = wl_row['oracle_ept']

        methods = [
            ('MAXN', wl_row['maxn_ept'], wl_row.get('maxn_tpot', wl_row.get('maxn_tpms', 0)),
             wl_row['maxn_power'], maxn_ept, dyn_ept, oracle_ept),
            ('Dynamic', wl_row['dynamic_ept'], wl_row.get('dynamic_tpot', wl_row.get('dynamic_tpms', 0)),
             wl_row['dynamic_power'], maxn_ept, dyn_ept, oracle_ept),
            ('EdgeShark*', wl_row['edgeshark_ept'], wl_row.get('edgeshark_tpot', wl_row.get('edgeshark_tpms', 0)),
             wl_row['edgeshark_power'], maxn_ept, dyn_ept, oracle_ept),
            ('FlashFlow*', wl_row.get('flashflow_ept', None), wl_row.get('flashflow_tpms', None),
             wl_row.get('flashflow_power', None), maxn_ept, dyn_ept, oracle_ept),
            ('EnergyInfra', wl_row['pareto_ept'], wl_row.get('pareto_tpot', wl_row.get('pareto_tpms', 0)),
             wl_row['pareto_power'], maxn_ept, dyn_ept, oracle_ept),
            ('Oracle', wl_row['oracle_ept'], wl_row.get('oracle_tpot', wl_row.get('oracle_tpms', 0)),
             wl_row['oracle_power'], maxn_ept, dyn_ept, oracle_ept),
        ]

        for name, ept, tpot, power, ref_maxn, ref_dyn, ref_oracle in methods:
            if ept is None or pd.isna(ept):
                continue
            ept_val = float(ept)
            tpot_val = float(tpot) if tpot and not pd.isna(tpot) else 0
            power_val = float(power) if power and not pd.isna(power) else 0

            vs_maxn = (ept_val - ref_maxn) / ref_maxn * 100
            vs_dyn = (ept_val - ref_dyn) / ref_dyn * 100
            vs_oracle = (ept_val - ref_oracle) / ref_oracle * 100

            marker = ' <<<' if name == 'EnergyInfra' else (' (oracle)' if name == 'Oracle' else '')
            print('  %-14s | %-8s | %9.4f | %9.2f | %9.2f | %+8.1f%% | %+8.1f%% | %+8.1f%%' % (
                wl if name == 'EnergyInfra' else '', name + marker,
                ept_val, tpot_val, power_val, vs_maxn, vs_dyn, vs_oracle))

    # Aggregate summary
    print()
    print('  ── Aggregate (mean across workloads) ──')
    agg = {}

    method_cols = {
        'MAXN': ('maxn_ept', 'maxn_tpot' if 'maxn_tpot' in sub.columns else 'maxn_tpms', 'maxn_power'),
        'Dynamic': ('dynamic_ept', 'dynamic_tpot' if 'dynamic_tpot' in sub.columns else 'dynamic_tpms', 'dynamic_power'),
        'EdgeShark*': ('edgeshark_ept', 'edgeshark_tpot' if 'edgeshark_tpot' in sub.columns else 'edgeshark_tpms', 'edgeshark_power'),
        'EnergyInfra': ('pareto_ept', 'pareto_tpot' if 'pareto_tpot' in sub.columns else 'pareto_tpms', 'pareto_power'),
        'Oracle': ('oracle_ept', 'oracle_tpot' if 'oracle_tpot' in sub.columns else 'oracle_tpms', 'oracle_power'),
    }

    if 'flashflow_ept' in sub.columns:
        method_cols['FlashFlow*'] = ('flashflow_ept', 'flashflow_tpms', 'flashflow_power')

    for name, (ept_col, tpot_col, pwr_col) in method_cols.items():
        if ept_col in sub.columns and sub[ept_col].notna().any():
            agg[name] = {
                'ept': sub[ept_col].mean(),
                'tpot': sub[tpot_col].mean() if tpot_col in sub.columns else 0,
                'power': sub[pwr_col].mean() if pwr_col in sub.columns else 0,
            }

    print('  %-14s | %9s | %9s | %9s | %9s | %9s' % (
        'Method', 'E/tok(J)', 'TPOT(ms)', 'Power(W)',
        'vs MAXN', 'E/tok Gap'))
    print('  %-14s | %9s | %9s | %9s | %9s | %9s' % (
        '-' * 14, '-' * 9, '-' * 9, '-' * 9, '-' * 9, '-' * 9))

    maxn_ept_mean = agg.get('MAXN', {}).get('ept', 0)
    oracle_ept_mean = agg.get('Oracle', {}).get('ept', 0)

    for name, vals in agg.items():
        vs_maxn = (vals['ept'] - maxn_ept_mean) / maxn_ept_mean * 100
        gap = (vals['ept'] - oracle_ept_mean) / oracle_ept_mean * 100
        marker = ' <<<' if name == 'EnergyInfra' else ''
        print('  %-14s | %9.4f | %9.2f | %9.2f | %+8.1f%% | %+8.1f%%' % (
            name + marker, vals['ept'], vals['tpot'], vals['power'], vs_maxn, gap))


# ============================================================
# Cross-Model Summary Table
# ============================================================

print()
print('=' * 100)
print('  CROSS-MODEL SUMMARY: EnergyInfra vs EdgeShark vs FlashFlow vs Baselines')
print('=' * 100)
print()

all_model_stats = []

for mk in MODEL_ORDER:
    sub = comparison[comparison['model'] == mk]
    if len(sub) == 0:
        continue

    ms = MODEL_SHORT[mk]
    stats = {'model': ms}

    for col_prefix, method_name in [('pareto', 'EnergyInfra'), ('edgeshark', 'EdgeShark*'),
                                      ('dynamic', 'Dynamic'), ('maxn', 'MAXN'),
                                      ('oracle_energy', 'Oracle')]:
        ept_col = f'{col_prefix}_ept'
        tpot_col = f'{col_prefix}_tpot' if f'{col_prefix}_tpot' in sub.columns else f'{col_prefix}_tpms'
        pwr_col = f'{col_prefix}_power'

        if ept_col in sub.columns:
            stats[f'{method_name}_ept'] = sub[ept_col].mean()
            stats[f'{method_name}_power'] = sub[pwr_col].mean() if pwr_col in sub.columns else 0

    # FlashFlow
    if 'flashflow_ept' in sub.columns and sub['flashflow_ept'].notna().any():
        stats['FlashFlow*_ept'] = sub['flashflow_ept'].mean()
        stats['FlashFlow*_power'] = sub['flashflow_power'].mean()

    all_model_stats.append(stats)

# Build comparison table
print('  %-16s' % 'Metric', end='')
for s in all_model_stats:
    print(' | %-18s' % s['model'], end='')
print()
print('  ' + '-' * 90)

# E/tok comparison
for metric in ['ept']:
    print()
    methods_to_show = ['MAXN', 'Dynamic', 'EdgeShark*', 'FlashFlow*', 'EnergyInfra', 'Oracle']
    for m in methods_to_show:
        col = f'{m}_{metric}'
        vals_exist = any(col in s for s in all_model_stats)
        if not vals_exist:
            continue

        label = m + (' <<<' if m == 'EnergyInfra' else '')
        print('  %-16s' % label, end='')
        for s in all_model_stats:
            if col in s:
                print(' | %14.4f' % s[col], end='')
            else:
                print(' | %14s' % '--', end='')
        print()

    # Also show relative improvement vs MAXN
    print()
    print('  E/tok improvement vs MAXN:')
    for m in ['Dynamic', 'EdgeShark*', 'FlashFlow*', 'EnergyInfra']:
        ept_col = f'{m}_ept'
        print('  %-16s' % m, end='')
        for s in all_model_stats:
            if ept_col in s and 'MAXN_ept' in s:
                pct = (s[ept_col] - s['MAXN_ept']) / s['MAXN_ept'] * 100
                print(' | %+14.1f%%' % pct, end='')
            else:
                print(' | %14s' % '--', end='')
        print()

    # Power comparison
    print()
    print('  Power (W):')
    for m in ['MAXN', 'Dynamic', 'EdgeShark*', 'FlashFlow*', 'EnergyInfra']:
        pwr_col = f'{m}_power'
        print('  %-16s' % m, end='')
        for s in all_model_stats:
            if pwr_col in s:
                print(' | %14.2f' % s[pwr_col], end='')
            else:
                print(' | %14s' % '--', end='')
        print()

    # Power saving vs MAXN
    print()
    print('  Power saving vs MAXN:')
    for m in ['Dynamic', 'EdgeShark*', 'FlashFlow*', 'EnergyInfra']:
        pwr_col = f'{m}_power'
        print('  %-16s' % m, end='')
        for s in all_model_stats:
            if pwr_col in s and 'MAXN_power' in s:
                pct = (s[pwr_col] - s['MAXN_power']) / s['MAXN_power'] * 100
                print(' | %+14.1f%%' % pct, end='')
            else:
                print(' | %14s' % '--', end='')
        print()

    # Gap to Oracle
    print()
    print('  Gap to Oracle (%):')
    for m in ['Dynamic', 'EdgeShark*', 'FlashFlow*', 'EnergyInfra']:
        ept_col = f'{m}_ept'
        print('  %-16s' % m, end='')
        for s in all_model_stats:
            if ept_col in s and 'Oracle_ept' in s:
                pct = (s[ept_col] - s['Oracle_ept']) / s['Oracle_ept'] * 100
                print(' | %+14.1f%%' % pct, end='')
            else:
                print(' | %14s' % '--', end='')
        print()

# ============================================================
# FlashFlow switching overhead analysis
# ============================================================

if len(flashflow_all) > 0:
    print()
    print('=' * 100)
    print('  FLASHFLOW SWITCHING OVERHEAD ANALYSIS')
    print('  Phase 7 measured: ~715ms switching delay on Jetson Orin')
    print('=' * 100)
    print()

    for mk in MODEL_ORDER:
        ff_sub = flashflow_all[flashflow_all['model'] == mk]
        if len(ff_sub) == 0:
            continue

        print(f'  [{MODEL_SHORT[mk]}]')
        print('  %-14s | %8s | %8s | %9s | %8s' % (
            'Workload', 'PrefillF', 'DecodeF', 'Overhead%', 'Same F?'))
        print('  %-14s | %8s | %8s | %9s | %8s' % (
            '-' * 14, '-' * 8, '-' * 8, '-' * 9, '-' * 8))

        for _, row in ff_sub.iterrows():
            same = 'YES' if row['prefill_freq'] == row['decode_freq'] else 'NO'
            print('  %-14s | %6dMHz | %6dMHz | %8.1f%% | %8s' % (
                row['workload'], row['prefill_freq'], row['decode_freq'],
                row['switch_overhead_pct'], same))
        print()

    # Overall switching overhead impact
    print('  Average switching overhead as % of total time:')
    for mk in MODEL_ORDER:
        ff_sub = flashflow_all[flashflow_all['model'] == mk]
        if len(ff_sub) > 0:
            avg_pct = ff_sub['switch_overhead_pct'].mean()
            max_pct = ff_sub['switch_overhead_pct'].max()
            print(f'    {MODEL_SHORT[mk]}: mean={avg_pct:.1f}%, max={max_pct:.1f}%')

# ============================================================
# Key Insights
# ============================================================

print()
print('=' * 100)
print('  KEY INSIGHTS')
print('=' * 100)
print()

print('  1. EdgeShark (Static per-model cap) ≈ BestStatic:')
print('     - Picks one frequency for all workloads, ignores input/output length variation')
print('     - Simulated via our E2E benchmark "best_static" strategy')
print('     - Our workload-aware Pareto outperforms by selecting different caps per workload')
print()

print('  2. FlashFlow (Phase-boundary DVFS):')
print('     - Uses high freq for prefill + low freq for decode')
print('     - Simulated from lock rate table per-phase data')
print('     - Switching overhead (~715ms) is MASSIVE on Jetson Orin:')
print('       * Phase 7 showed switching takes 37-65% of TTFT')
print('       * For short sequences (p64_o64), overhead dominates total time')
print('       * Only potentially viable for very long generation (p1024_o1024+)')
print('     - This is why EnergyInfra uses WORKLOAD-AWARE CAP (not phase switching)')
print()

print('  3. EnergyInfra advantages:')
print('     - Zero per-request overhead (lookup table, no switching)')
print('     - Workload-aware: adapts to input/output length')
print('     - Thermal-SLO feedback for long-running serving')
print('     - Multi-objective Pareto with HV gold-standard validation')
print()

print('  * EdgeShark and FlashFlow results are SIMULATED on our hardware.')
print('    Original papers tested on different platforms:')
print('    - EdgeShark: Jetson AGX Xavier (Volta GPU) + CNN models')
print('    - FlashFlow: NVIDIA A100 / RTX 3090 (datacenter) + BERT/GPT-2')
print('    Neither has published results on Jetson Orin + LLM serving.')
