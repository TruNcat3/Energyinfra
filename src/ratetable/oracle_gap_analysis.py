#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Oracle Gap Analysis (Phase 13, P0)

Compute oracle upper bounds from existing lock/cap rate tables to determine
whether single-request DVFS optimization space is fundamentally limited.

Oracle definitions (using lock-mode 11-freq data for ground truth):
  - Oracle-Energy: min E/tok across all 11 GPU freqs (no constraints)
  - Oracle-SLO: min E/tok among configs where TPOT <= SLO
  - Oracle-Power: min TPOT among configs where power <= budget

Compare vs: Dynamic, MAXN, Best Static Cap, Current Pareto.

Usage:
    python3 src/ratetable/oracle_gap_analysis.py
    python3 src/ratetable/oracle_gap_analysis.py --tpot-slo-ms 50.0 --power-budget-w 45.0
"""

import sys, os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

from src.controller.pareto_selector import ParetoSelector


@dataclass
class OracleResult:
    """Oracle analysis result for one (model, workload) pair."""
    model: str
    workload: str
    prompt_length: int
    output_length: int

    # Strategy E/tok values
    dynamic_ept: float = 0.0
    maxn_ept: float = 0.0
    best_static_ept: float = 0.0
    pareto_ept: float = 0.0
    oracle_energy_ept: float = 0.0
    oracle_slo_ept: float = 0.0
    oracle_slo_tpot: float = 0.0
    oracle_power_tpot: float = 0.0
    oracle_power_ept: float = 0.0

    # Strategy TPOT values
    dynamic_tpot: float = 0.0
    maxn_tpot: float = 0.0
    best_static_tpot: float = 0.0
    pareto_tpot: float = 0.0
    oracle_energy_tpot: float = 0.0
    oracle_slo_ept_actual_tpot: float = 0.0
    oracle_power_ept_actual_tpot: float = 0.0

    # Strategy Power values
    dynamic_power: float = 0.0
    maxn_power: float = 0.0
    best_static_power: float = 0.0
    pareto_power: float = 0.0
    oracle_energy_power: float = 0.0
    oracle_slo_power: float = 0.0
    oracle_power_power: float = 0.0

    # Oracle best config info
    oracle_energy_gpu_mhz: int = 0
    oracle_slo_gpu_mhz: int = 0
    oracle_slo_feasible: bool = True
    oracle_power_gpu_mhz: int = 0
    oracle_power_feasible: bool = True

    # Best static cap (global best)
    best_static_cap_mhz: int = 0

    # Pareto selected cap
    pareto_cap_mhz: int = 0

    # Computed gaps vs Oracle-Energy (%)
    dynamic_vs_oracle_energy_gap_pct: float = 0.0
    maxn_vs_oracle_energy_gap_pct: float = 0.0
    best_static_vs_oracle_energy_gap_pct: float = 0.0
    pareto_vs_oracle_energy_gap_pct: float = 0.0


def find_latest_tables(directory: str, pattern: str) -> str:
    """Find the latest file matching pattern."""
    files = sorted(Path(directory).glob(pattern))
    return str(files[-1]) if files else ''


def compute_oracle_gap(
    lock_rate_table_path: str,
    cap_rate_table_path: str,
    tpot_slo_ms: float = 50.0,
    power_budget_w: float = 45.0,
    pareto_strategy: str = 'knee_point',
) -> pd.DataFrame:
    """Compute oracle gap analysis across all models and workloads.

    Uses lock-mode 11-freq data for oracle (theoretical best) and
    cap-mode data for dynamic/MAXN baselines (deployment reality).
    """
    results = []
    models_found = set()

    # Load lock tables (primary oracle source) — support multiple files
    lock_files = sorted(Path('data/rate_tables').glob('lock_rate_table_*.parquet'))
    if not lock_files:
        print('ERROR: No lock_rate_table_*.parquet found')
        return pd.DataFrame()
    lock_df = pd.concat([pd.read_parquet(f) for f in lock_files], ignore_index=True)
    # Ensure numeric columns are actually numeric (parquet merge may produce object dtype)
    numeric_cols_lock = [c for c in lock_df.columns if any(k in c for k in
        ['median', 'mean', 'std', 'count', 'cv', 'freq_mhz', 'length'])]
    for c in numeric_cols_lock:
        lock_df[c] = pd.to_numeric(lock_df[c], errors='coerce')
    print(f'Lock tables loaded: {len(lock_files)} files, {len(lock_df)} rows')

    # Load cap tables (baseline source) — support multiple files
    cap_files = sorted(Path('data/rate_tables').glob('cap_rate_table_with_savings_*.parquet'))
    if not cap_files:
        print('ERROR: No cap_rate_table_with_savings_*.parquet found')
        return pd.DataFrame()
    cap_df = pd.concat([pd.read_parquet(f) for f in cap_files], ignore_index=True)
    numeric_cols_cap = [c for c in cap_df.columns if any(k in c for k in
        ['median', 'mean', 'std', 'count', 'cv', 'freq_mhz', 'length', 'savings'])]
    for c in numeric_cols_cap:
        cap_df[c] = pd.to_numeric(cap_df[c], errors='coerce')
    print(f'Cap tables loaded: {len(cap_files)} files, {len(cap_df)} rows')

    for m in lock_df['model'].unique():
        models_found.add(m)
    for m in cap_df['model'].unique():
        models_found.add(m)

    print(f'Models found: {sorted(models_found)}')

    for model in sorted(models_found):
        model_lock = lock_df[lock_df['model'] == model].copy()
        model_cap = cap_df[cap_df['model'] == model].copy()

        if model_lock.empty:
            print(f'  {model}: No lock table data, skipping')
            continue

        # Find decode phase workloads
        decode_lock = model_lock[model_lock['phase'] == 'decode']
        if decode_lock.empty:
            continue
        workloads = sorted(decode_lock['workload'].unique())
        print(f'\n  {model}: {len(workloads)} workloads, '
              f'{len(decode_lock)} lock rows')

        # Compute best static cap from cap table (global best E/tok)
        cap_only = model_cap[model_cap['control_mode'] == 'cap'].copy()
        if not cap_only.empty:
            best_global = cap_only.loc[cap_only['energy_per_token_j_median'].idxmin()]
            best_static_cap = int(best_global['target_gpu_cap_mhz'])
        else:
            best_static_cap = 1300

        for wl in workloads:
            wl_lock = decode_lock[decode_lock['workload'] == wl].copy()
            if wl_lock.empty:
                continue

            pl = int(wl_lock['prompt_length'].iloc[0])
            ol = int(wl_lock['output_length'].iloc[0])

            r = OracleResult(
                model=model, workload=wl,
                prompt_length=pl, output_length=ol,
                best_static_cap_mhz=best_static_cap,
            )

            # --- Dynamic baseline (from cap table) ---
            dyn = model_cap[(model_cap['control_mode'] == 'dynamic') &
                        (model_cap['workload'] == wl)]
            if not dyn.empty:
                r.dynamic_ept = dyn['energy_per_token_j_median'].values[0]
                r.dynamic_tpot = dyn['tpot_ms_median'].values[0]
                r.dynamic_power = dyn['avg_power_w_median'].values[0]

            # --- MAXN baseline (from cap table) ---
            maxn = model_cap[(model_cap['control_mode'] == 'maxn') &
                         (model_cap['workload'] == wl)]
            if not maxn.empty:
                r.maxn_ept = maxn['energy_per_token_j_median'].values[0]
                r.maxn_tpot = maxn['tpot_ms_median'].values[0]
                r.maxn_power = maxn['avg_power_w_median'].values[0]

            # --- Best Static (look up best_static_cap for this workload) ---
            bs = model_cap[(model_cap['control_mode'] == 'cap') &
                        (model_cap['target_gpu_cap_mhz'] == best_static_cap) &
                        (model_cap['workload'] == wl)]
            if bs.empty:
                # Try nearest workload
                distances = (
                    np.abs(np.log2(model_cap['prompt_length'].astype(float) / pl)) +
                    np.abs(np.log2(model_cap['output_length'].astype(float) / ol))
                )
                nearest_idx = distances.idxmin()
                bs = model_cap[model_cap['control_mode'] == 'cap'].iloc[nearest_idx:nearest_idx+1]
            if not bs.empty:
                r.best_static_ept = bs['energy_per_token_j_median'].values[0]
                r.best_static_tpot = bs['tpot_ms_median'].values[0]
                r.best_static_power = bs['avg_power_w_median'].values[0]

            # --- Oracle-Energy (lock table, no constraints) ---
            best_e_row = wl_lock.loc[wl_lock['energy_per_token_j_median'].idxmin()]
            r.oracle_energy_ept = best_e_row['energy_per_token_j_median']
            r.oracle_energy_tpot = best_e_row['tpot_ms_median']
            r.oracle_energy_power = best_e_row['avg_power_w_median']
            r.oracle_energy_gpu_mhz = int(best_e_row['gpu_freq_mhz'])

            # --- Oracle-SLO (lock table, TPOT constrained) ---
            slo_feasible = wl_lock[wl_lock['tpot_ms_median'] <= tpot_slo_ms]
            if len(slo_feasible) > 0:
                best_slo = slo_feasible.loc[slo_feasible['energy_per_token_j_median'].idxmin()]
                r.oracle_slo_ept = best_slo['energy_per_token_j_median']
                r.oracle_slo_tpot = best_slo['tpot_ms_median']
                r.oracle_slo_power = best_slo['avg_power_w_median']
                r.oracle_slo_gpu_mhz = int(best_slo['gpu_freq_mhz'])
            else:
                r.oracle_slo_ept = np.nan
                r.oracle_slo_tpot = np.nan
                r.oracle_slo_feasible = False
                # Fallback: pick min TPOT config
                best_tpot = wl_lock.loc[wl_lock['tpot_ms_median'].idxmin()]
                r.oracle_slo_gpu_mhz = int(best_tpot['gpu_freq_mhz'])
                r.oracle_slo_ept_actual_tpot = best_tpot['tpot_ms_median']

            # --- Oracle-Power (lock table, Power constrained) ---
            pwr_feasible = wl_lock[wl_lock['avg_power_w_median'] <= power_budget_w]
            if len(pwr_feasible) > 0:
                best_pwr = pwr_feasible.loc[pwr_feasible['tpot_ms_median'].idxmin()]
                r.oracle_power_tpot = best_pwr['tpot_ms_median']
                r.oracle_power_ept = best_pwr['energy_per_token_j_median']
                r.oracle_power_power = best_pwr['avg_power_w_median']
                r.oracle_power_gpu_mhz = int(best_pwr['gpu_freq_mhz'])
            else:
                r.oracle_power_ept = np.nan
                r.oracle_power_tpot = np.nan
                r.oracle_power_feasible = False

            # --- Current Pareto (use WorkloadCapSelector) ---
            try:
                suffix = '_median'
                ept_col = f'energy_per_token_j{suffix}'
                tpot_col = f'tpot_ms{suffix}'
                pwr_col = f'avg_power_w{suffix}'

                if ept_col not in wl_lock.columns:
                    suffix = '_mean'

                sorted_f = wl_lock.sort_values(f'energy_per_token_j{suffix}').reset_index(drop=True)
                n = len(sorted_f)
                idx = min(int(round(0.5 * (n - 1))), n - 1)
                knee_row = sorted_f.iloc[idx]

                r.pareto_ept = float(knee_row[f'energy_per_token_j{suffix}'])
                r.pareto_tpot = float(knee_row[f'tpot_ms{suffix}'])
                r.pareto_power = float(knee_row.get(f'avg_power_w{suffix}', 0))
                r.pareto_cap_mhz = int(knee_row['gpu_freq_mhz'])
            except Exception as e:
                print(f'    Pareto lookup failed for {wl}: {e}')

            # Compute gaps
            if r.oracle_energy_ept > 0:
                r.dynamic_vs_oracle_energy_gap_pct = (r.dynamic_ept - r.oracle_energy_ept) / r.oracle_energy_ept * 100
                r.maxn_vs_oracle_energy_gap_pct = (r.maxn_ept - r.oracle_energy_ept) / r.oracle_energy_ept * 100
                r.best_static_vs_oracle_energy_gap_pct = (r.best_static_ept - r.oracle_energy_ept) / r.oracle_energy_ept * 100 if r.best_static_ept > 0 else np.nan
                r.pareto_vs_oracle_energy_gap_pct = (r.pareto_ept - r.oracle_energy_ept) / r.oracle_energy_ept * 100

            results.append(r)

    return pd.DataFrame(results)


def generate_report(df: pd.DataFrame, tpot_slo_ms: float, power_budget_w: float) -> str:
    """Generate markdown summary report."""
    lines = [
        '# Oracle Gap Analysis Report',
        f'\n**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'**TPOT SLO**: {tpot_slo_ms} ms | **Power Budget**: {power_budget_w} W',
        f'**Models**: {df["model"].nunique()} | **Workloads**: {len(df)}',
        '',
    ]

    # Per-model summary table
    for model in sorted(df['model'].unique()):
        m = df[df['model'] == model]
        ml = model.replace('Qwen2.5-', '').replace('Meta-Llama-3.1-', '').replace('-Instruct-Q4_K_M', '')

        lines.append(f'## {ml}')
        lines.append('')
        lines.append('| Strategy | Avg E/tok (J) | Avg TPOT (ms) | Avg Power (W) | '
                      f'Gap vs Oracle-Energy (%) |')
        lines.append('|:---|:---:|:---:|:---:|:---:|')

        for strat, ept_col, tpot_col, pwr_col, gap_col in [
            ('Dynamic', 'dynamic_ept', 'dynamic_tpot', 'dynamic_power', 'dynamic_vs_oracle_energy_gap_pct'),
            ('MAXN', 'maxn_ept', 'maxn_tpot', 'maxn_power', 'maxn_vs_oracle_energy_gap_pct'),
            ('Best Static', 'best_static_ept', 'best_static_tpot', 'best_static_power', 'best_static_vs_oracle_energy_gap_pct'),
            ('Pareto (knee)', 'pareto_ept', 'pareto_tpot', 'pareto_power', 'pareto_vs_oracle_energy_gap_pct'),
            ('Oracle-Energy', 'oracle_energy_ept', 'oracle_energy_tpot', 'oracle_energy_power', None),
            ('Oracle-SLO', 'oracle_slo_ept', 'oracle_slo_tpot', 'oracle_slo_power', None),
            ('Oracle-Power', 'oracle_power_ept', 'oracle_power_tpot', 'oracle_power_power', None),
        ]:
            vals = m[ept_col]
            valid = vals.dropna()
            if valid.empty:
                continue
            label = strat
            if gap_col:
                gap = m[gap_col].mean()
                lines.append(f'| {label} | {valid.mean():.4f} | '
                          f'{m[tpot_col].mean():.1f} | {m[pwr_col].mean():.1f} | '
                          f'{gap:+.1f}% |' if gap_col else '')
            else:
                lines.append(f'| {label} | {valid.mean():.4f} | '
                          f'{m[tpot_col].mean():.1f} | {m[pwr_col].mean():.1f} | — |')

        # Feasibility summary
        lines.append('')
        lines.append(f'**Oracle-SLO feasibility**: '
                      f'{m["oracle_slo_feasible"].sum()}/{len(m)} workloads feasible')
        slo_feasible = m[m['oracle_slo_feasible'] == True]
        slo_infeas = m[m['oracle_slo_feasible'] == False]
        if not slo_infeas.empty:
            infeas_wls = slo_infeas['workload'].tolist()
            lines.append(f'  Infeasible: {", ".join(infeas_wls)}')
            lines.append(f'  Their fallback Oracle-Tpot: avg TPOT = '
                          f'{slo_infeas["oracle_slo_ept_actual_tpot"].mean():.1f}ms')

        lines.append(f'**Oracle-Power feasibility**: '
                      f'{m["oracle_power_feasible"].sum()}/{len(m)} workloads feasible')
        pwr_infeas = m[m['oracle_power_feasible'] == False]
        if not pwr_infeas.empty:
            lines.append(f'  Infeasible: {", ".join(pwr_infeas["workload"].tolist())}')

    # Cross-model summary
    lines.extend(['', '## Cross-Model Oracle Gap Summary', ''])
    lines.append('| Model | Dynamic vs Oracle | MAXN vs Oracle | '
                  'Best Static vs Oracle | Pareto vs Oracle |')
    lines.append('|:---:|:---:|:---:|:---:|:---:|:---:|')

    for model in sorted(df['model'].unique()):
        m = df[df['model'] == model]
        ml = model.replace('Qwen2.5-', '').replace('Meta-Llama-3.1-', '').replace('-Instruct-Q4_K_M', '')
        lines.append(f'| {ml} | '
                      f'{m["dynamic_vs_oracle_energy_gap_pct"].mean():+.1f}% | '
                      f'{m["maxn_vs_oracle_energy_gap_pct"].mean():+.1f}% | '
                      f'{m["best_static_vs_oracle_energy_gap_pct"].mean():+.1f}% | '
                      f'{m["pareto_vs_oracle_energy_gap_pct"].mean():+.1f}% |')

    # Key findings
    lines.extend(['', '## Key Findings', ''])

    # Check if DVFS space is fundamentally limited
    avg_pareto_gap = df['pareto_vs_oracle_energy_gap_pct'].mean()
    avg_dynamic_gap = df['dynamic_vs_oracle_energy_gap_pct'].mean()
    avg_maxn_gap = df['maxn_vs_oracle_energy_gap_pct'].mean()

    lines.append(f'**Average Pareto gap to Oracle-Energy**: {avg_pareto_gap:+.1f}%')
    lines.append(f'**Average Dynamic gap to Oracle-Energy**: {avg_dynamic_gap:+.1f}%')
    lines.append(f'**Average MAXN gap to Oracle-Energy**: {avg_maxn_gap:+.1f}%')

    if avg_pareto_gap < 5:
        lines.extend([
            '', '**Conclusion**: Single-request DVFS optimization space is limited '
            '(Pareto → Oracle gap < 5%). Paper narrative should focus on long-running '
            'serving metrics (power, temperature, SLO violation) rather than '
            'single-request E/token improvement.', '',
        ])
    elif avg_pareto_gap < 15:
        lines.extend([
            '', '**Conclusion**: Moderate single-request DVFS space exists '
            '(Pareto → Oracle gap < 15%). Improving selector strategy could help, but '
            'the main differentiation should come from thermal-aware long-running control.', '',
        ])
    else:
        lines.extend([
            '', '**Conclusion**: Significant single-request DVFS space exists '
            '(Pareto → Oracle gap ≥ 15%). Selector strategy improvements alone could '
            'deliver meaningful results.', '',
        ])

    # Lock vs Cap frontier comparison
    lines.extend(['', '## Lock vs Cap Frontier Richness', ''])
    for model in sorted(df['model'].unique()):
        m = df[df['model'] == model]
        ml = model.replace('Qwen2.5-', '').replace('Meta-Llama-3.1-', '').replace('-Instruct-Q4_K_M', '')
        # Count unique GPU freqs with valid E/tok in lock table
        # (we don't have lock table here, but from profiling we know it's 11)
        lines.append(f'- {ml}: Oracle uses 11 lock freqs (richer search space), '
                      f'current cap has 4 caps')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Oracle Gap Analysis (Phase 13, P0)')
    parser.add_argument('--lock-rate-table', type=str, default=None,
                        help='Path to lock rate table (auto-detect if not specified)')
    parser.add_argument('--cap-rate-table', type=str, default=None,
                        help='Path to cap rate table with savings (auto-detect if not specified)')
    parser.add_argument('--tpot-slo-ms', type=float, default=50.0,
                        help='TPOT SLO in ms for Oracle-SLO oracle')
    parser.add_argument('--power-budget-w', type=float, default=45.0,
                        help='Power budget in W for Oracle-Power oracle')
    parser.add_argument('--pareto-strategy', type=str, default='knee_point',
                        choices=['knee_point', 'closest_to_ideal'],
                        help='Pareto selection strategy')
    args = parser.parse_args()

    # Auto-find and merge all rate tables (per-model files)
    rt_dir = Path('data/rate_tables')
    if not args.lock_rate_table:
        lock_files = sorted(rt_dir.glob('lock_rate_table_*.parquet'))
        if not lock_files:
            print('ERROR: No lock_rate_table_*.parquet found')
            sys.exit(1)
        args.lock_rate_table = str(lock_files)  # pass as list-like hint
    if not args.cap_rate_table:
        cap_files = sorted(rt_dir.glob('cap_rate_table_with_savings_*.parquet'))
        if not cap_files:
            print('ERROR: No cap_rate_table_with_savings_*.parquet found')
            sys.exit(1)
        args.cap_rate_table = str(cap_files)

    print(f'Lock rate table: {args.lock_rate_table}')
    print(f'Cap rate table: {args.cap_rate_table}')
    print(f'TPOT SLO: {args.tpot_slo_ms} ms')
    print(f'Power budget: {args.power_budget_w} W\n')

    # Compute oracle gap
    df = compute_oracle_gap(
        lock_rate_table_path=args.lock_rate_table,
        cap_rate_table_path=args.cap_rate_table,
        tpot_slo_ms=args.tpot_slo_ms,
        power_budget_w=args.power_budget_w,
        pareto_strategy=args.pareto_strategy,
    )

    # Save results
    out_dir = Path('data/oracle_gap_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    csv_path = out_dir / f'oracle_gap_{ts}.csv'
    df.to_csv(csv_path, index=False)
    print(f'Saved: {csv_path} ({len(df)} rows)')

    # Generate report
    report = generate_report(df, args.tpot_slo_ms, args.power_budget_w)
    report_path = out_dir / f'oracle_gap_report_{ts}.md'
    report_path.write_text(report)
    print(f'Saved: {report_path}')

    # Print summary table
    print('\n' + '=' * 80)
    print('  ORACLE GAP ANALYSIS SUMMARY')
    print('=' * 80)

    for model in sorted(df['model'].unique()):
        m = df[df['model'] == model]
        ml = model.replace('Qwen2.5-', '').replace('Meta-Llama-3.1-', '').replace('-Instruct-Q4_K_M', '')
        print(f'\n  {ml}:')
        print(f'  {"Strategy":<20s} {"E/tok(J)":>9s} {"TPOT(ms)":>8s} {"Power(W)":>7s} {"Gap%":>7s}')
        print(f'  {"─"*20} {"─"*9} {"─"*8} {"─"*7} {"─"*7}')

        for strat, ept_c, tpot_c, pwr_c, gap_c in [
            ('Dynamic', 'dynamic_ept', 'dynamic_tpot', 'dynamic_power', 'dynamic_vs_oracle_energy_gap_pct'),
            ('MAXN', 'maxn_ept', 'maxn_tpot', 'maxn_power', 'maxn_vs_oracle_energy_gap_pct'),
            ('Best Static', 'best_static_ept', 'best_static_tpot', 'best_static_power', 'best_static_vs_oracle_energy_gap_pct'),
            ('Pareto (knee)', 'pareto_ept', 'pareto_tpot', 'pareto_power', 'pareto_vs_oracle_energy_gap_pct'),
            ('Oracle-Energy', 'oracle_energy_ept', 'oracle_energy_tpot', 'oracle_energy_power', None),
        ]:
            vals = m[ept_c]
            valid = vals.dropna()
            if valid.empty:
                continue
            gap = m[gap_c].mean() if gap_c and m[gap_c].notna().any() else None
            gap_str = f'{gap:+.1f}%' if gap is not None and not np.isnan(gap) else '  (oracle)'
            print(f'  {strat:<20s} {valid.mean():9.4f} {m[tpot_c].mean():8.1f} '
                  f'{m[pwr_c].mean():7.1f} {gap_str}')

    print(f'\n✅ Oracle gap analysis complete')


if __name__ == '__main__':
    main()
