#!/usr/bin/env python3
"""Detailed HV statistical analysis for paper narrative."""

import sys, os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
import pandas as pd, numpy as np

hv = pd.read_csv('data/multi_obj_eval/multi_obj_hv_summary_20260614_181323.csv')
offline = hv[hv['trace'] == 'offline'].copy()

MODELS = {
    'Qwen2.5-7B-Instruct-Q4_K_M': '7B (Qwen2.5)',
    'Meta-Llama-3.1-8B-Instruct-Q4_K_M': '8B (Llama-3.1)',
    'Qwen2.5-14B-Instruct-Q4_K_M': '14B (Qwen2.5)',
}

print('=' * 82)
print('  3D Hypervolume (HV) Detailed Comparison')
print('  Objectives: E/tok, TPOT, Power (all minimize)')
print('  HV = volume of dominated objective space (Zitzler 1999), higher = better')
print('  Reference point = worst-case across all strategies per dimension')
print('=' * 82)

all_stats = []

for mk, ml in sorted(MODELS.items(), key=lambda x: x[1]):
    mdf = offline[offline['model'] == mk].sort_values('hv_3d', ascending=False)
    pareto_hv = mdf[mdf['strategy'] == 'pareto']['hv_3d'].values[0]

    print()
    print('  [%s]' % ml)
    print('  Pareto HV = %.6f (15 observations: 5 workloads x 3 repeats)' % pareto_hv)
    print('  ' + '-' * 68)
    print('  %-12s | %12s | %14s | %10s' % ('Strategy', 'HV', 'Gap to Pareto', 'Ratio'))
    print('  %-12s | %12s | %14s | %10s' % ('-' * 12, '-' * 12, '-' * 14, '-' * 10))

    for _, r in mdf.iterrows():
        s = r['strategy']
        h = r['hv_3d']
        if s == 'pareto':
            print('  %-12s | %12.6f | %14s | %10s' % ('Pareto <<<', h, '(baseline)', '1.00x'))
        else:
            gap_abs = pareto_hv - h
            gap_rel = (pareto_hv - h) / h * 100
            ratio = pareto_hv / h
            print('  %-12s | %12.6f | +%11.4f (+%4.1f%%) | %9.1fx' % (
                s, h, gap_abs, gap_rel, ratio))

    others = mdf[mdf['strategy'] != 'pareto']['hv_3d'].values
    second = others[0]
    median_hv = np.median(others)
    mean_hv = np.mean(others)
    min_hv = np.min(others)
    second_name = mdf.iloc[1]['strategy']
    worst_idx = np.argmin(mdf['hv_3d'].values)
    worst_name = mdf.iloc[worst_idx]['strategy']

    print('  ' + '-' * 68)
    print('  Summary:')
    print('    Pareto vs 2nd place (%-10s): +%.4f abs, +%.1f%% rel, %.1fx ratio' % (
        second_name, pareto_hv - second, (pareto_hv - second) / second * 100, pareto_hv / second))
    print('    Pareto vs median (all others):   +%.4f abs, +%.1f%% rel, %.1fx ratio' % (
        pareto_hv - median_hv, (pareto_hv - median_hv) / median_hv * 100, pareto_hv / median_hv))
    print('    Pareto vs worst   (%-10s): +%.6f abs, +%.1f%% rel, %.0fx ratio' % (
        worst_name, pareto_hv - min_hv, (pareto_hv - min_hv) / min_hv * 100, pareto_hv / min_hv))

    std_hv = np.std(others)
    z_score = (pareto_hv - mean_hv) / std_hv if std_hv > 0 else float('inf')
    print('    Pareto z-score vs others:       %.1f sigma above mean' % z_score)

    all_stats.append({
        'model': ml, 'pareto_hv': pareto_hv, 'second': second_name, 'second_hv': second,
        'median_hv': median_hv, 'mean_hv': mean_hv, 'worst_hv': min_hv,
        'ratio_2nd': pareto_hv / second, 'ratio_median': pareto_hv / median_hv,
        'z_score': z_score
    })

# Cross-model table
print()
print('=' * 82)
print('  Cross-Model HV Comparison Table')
print('=' * 82)
print()

strategies_order = ['pareto', 'pwr_45w', 'slo_50ms', 'alpha_07', 'alpha_03',
                   'min_energy', 'slo_45ms', 'maxn', 'dynamic']

pareto_hvs = {}
for mk in MODELS:
    mdf = offline[offline['model'] == mk]
    pareto_hvs[mk] = mdf[mdf['strategy'] == 'pareto']['hv_3d'].values[0]

mk_keys = sorted(MODELS.keys())

print('  %-12s' % 'Strategy', end='')
for mk in mk_keys:
    ml = MODELS[mk]
    print(' | %-16s' % ml, end='')
print()
print('  ' + '-' * 78)

for s in strategies_order:
    label = 'Pareto <<<' if s == 'pareto' else s
    print('  %-12s' % label, end='')
    for mk in mk_keys:
        mdf = offline[offline['model'] == mk]
        val = mdf[mdf['strategy'] == s]['hv_3d']
        if len(val) > 0 and val.values[0] > 0.001:
            h = val.values[0]
            ratio = pareto_hvs[mk] / h
            if ratio > 100:
                r = '%.0fx' % ratio
            elif ratio > 10:
                r = '%.1fx' % ratio
            else:
                r = '%.2fx' % ratio
            print(' | %8.4f (%5s)' % (h, r), end='')
        else:
            print(' | %8s (%5s)' % ('--', '--'), end='')
    print()

print()
print('  Note: numbers in () = Pareto_HV / Strategy_HV')

# Paper-ready summary
print()
print('=' * 82)
print('  Paper-Ready Summary')
print('=' * 82)
print()
print('  "Our Pareto strategy achieves the highest Hypervolume Indicator (HV)')
print('   across all three models, dominating substantially more objective space')
print('   than any single-objective or constrained baseline:"')
print()
for st in all_stats:
    print('   - %s: Pareto HV=%.2f, %.1fx the 2nd-best (%s HV=%.2f), %.1fx the median,' % (
        st['model'], st['pareto_hv'], st['ratio_2nd'], st['second'], st['second_hv'],
        st['ratio_median']))
    print('     %.1f sigma above the mean of all other strategies' % st['z_score'])
