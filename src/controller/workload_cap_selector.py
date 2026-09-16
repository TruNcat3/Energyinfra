#!/usr/bin/env python3
"""
Workload-Aware GPU Cap Selector

Selects optimal GPU frequency cap based on workload characteristics and
optimization objective. Uses both lock-mode profiling data (fine-grained
GPU frequency effects) and cap-mode profiling data (actual governor behavior
under caps).

Selection strategies:
  - slo_constrained: Find lowest GPU cap meeting TPOT SLO → maximizes power savings
  - min_energy: Select GPU cap that minimizes E/tok
  - alpha_weighted: Continuous trade-off between energy and latency
  - power_budget: Find highest GPU cap within power budget

Usage:
    from src.controller.workload_cap_selector import WorkloadCapSelector
    sel = WorkloadCapSelector('data/rate_tables/cap_rate_table_with_savings_*.parquet',
                              'data/rate_tables/lock_rate_table_*.parquet')
    cfg = sel.select(prompt_length=512, output_length=128,
                     strategy='slo_constrained', tpot_slo_ms=50)
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.controller.pareto_selector import ParetoSelector, ParetoFrontierResult

logger = logging.getLogger(__name__)

# Available GPU cap values (MHz)
GPU_CAPS = [612, 816, 1020, 1300]

# GPU frequencies available in lock mode
GPU_FREQS = [306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300]


class WorkloadCapSelector:
    """Select optimal GPU cap based on workload and optimization objective."""

    def __init__(self, cap_rate_table_path: str = None,
                 lock_rate_table_path: str = None,
                 model: str = None):
        self.model = model

        if cap_rate_table_path:
            self.cap_table = pd.read_parquet(cap_rate_table_path)
            if model:
                self.cap_table = self.cap_table[self.cap_table['model'] == model]
            logger.info(f"Cap rate table: {len(self.cap_table)} rows")
        else:
            self.cap_table = pd.DataFrame()

        if lock_rate_table_path:
            self.lock_table = pd.read_parquet(lock_rate_table_path)
            if model:
                self.lock_table = self.lock_table[self.lock_table['model'] == model]
            logger.info(f"Lock rate table: {len(self.lock_table)} rows")
        else:
            self.lock_table = pd.DataFrame()

        # Create ParetoSelector if lock table is available (for multi-objective)
        self._pareto = None
        if not self.lock_table.empty or not self.cap_table.empty:
            self._pareto = ParetoSelector(
                lock_rate_table_path=lock_rate_table_path if not self.lock_table.empty else None,
                cap_rate_table_path=cap_rate_table_path if not self.cap_table.empty else None,
                model=model,
            )
            # Share the already-loaded tables instead of re-reading
            if not self.lock_table.empty:
                self._pareto.lock_table = self.lock_table
            if not self.cap_table.empty:
                self._pareto.cap_table = self.cap_table

    def select(self, prompt_length: int, output_length: int,
               strategy: str = 'slo_constrained',
               alpha: float = 0.5,
               tpot_slo_ms: float = None,
               ttft_slo_ms: float = None,
               power_budget_w: float = None,
               baseline: str = 'dynamic') -> Optional[Dict]:
        """
        Select optimal GPU cap for a workload.

        Args:
            prompt_length: Input token count
            output_length: Output token count
            strategy: 'slo_constrained', 'min_energy', 'alpha_weighted',
                      'power_budget', 'pareto'
            alpha: Energy-latency trade-off (0=energy, 1=latency) for alpha_weighted
            tpot_slo_ms: TPOT SLO in ms (for slo_constrained)
            ttft_slo_ms: TTFT SLO in ms (optional, for slo_constrained)
            power_budget_w: Max power budget in W (for power_budget)
            baseline: Compare vs 'dynamic' or 'maxn'

        Returns:
            Dict with selected config and predicted metrics, or None.
        """
        # Find nearest workload in cap table
        cap_configs = self._find_workload_caps(prompt_length, output_length)
        if cap_configs.empty:
            logger.warning(f"No cap data for p={prompt_length}, o={output_length}")
            return None

        if strategy == 'slo_constrained':
            return self._select_slo_constrained(cap_configs, prompt_length, output_length,
                                                tpot_slo_ms, ttft_slo_ms, baseline)
        elif strategy == 'min_energy':
            return self._select_min_energy(cap_configs, prompt_length, output_length, baseline)
        elif strategy == 'alpha_weighted':
            return self._select_alpha(cap_configs, prompt_length, output_length,
                                      alpha, baseline)
        elif strategy == 'power_budget':
            return self._select_power_budget(cap_configs, prompt_length, output_length,
                                             power_budget_w, baseline)
        elif strategy == 'pareto':
            return self._select_pareto(prompt_length, output_length, baseline,
                                       tpot_slo_ms=tpot_slo_ms,
                                       power_budget_w=power_budget_w,
                                       alpha=alpha)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _find_workload_caps(self, prompt_length: int, output_length: int) -> pd.DataFrame:
        """Find cap configs for exact or nearest workload."""
        if self.cap_table.empty:
            return pd.DataFrame()

        # Exact match
        mask = (
            (self.cap_table['prompt_length'] == prompt_length) &
            (self.cap_table['output_length'] == output_length)
        )
        caps = self.cap_table[mask]
        if not caps.empty:
            return caps

        # Nearest workload by log-distance
        distances = (
            np.abs(np.log2(self.cap_table['prompt_length'].astype(float) / prompt_length)) +
            np.abs(np.log2(self.cap_table['output_length'].astype(float) / output_length))
        )
        min_idx = distances.idxmin()
        best_pl = self.cap_table.loc[min_idx, 'prompt_length']
        best_ol = self.cap_table.loc[min_idx, 'output_length']
        logger.info(f"Nearest workload: p={best_pl}, o={best_ol} for p={prompt_length}, o={output_length}")
        return self.cap_table[
            (self.cap_table['prompt_length'] == best_pl) &
            (self.cap_table['output_length'] == best_ol)
        ]

    def _select_slo_constrained(self, caps: pd.DataFrame, pl: int, ol: int,
                                 tpot_slo_ms: float, ttft_slo_ms: float,
                                 baseline: str) -> Optional[Dict]:
        """Find lowest GPU cap that meets TPOT SLO → maximizes power savings."""
        if tpot_slo_ms is None:
            tpot_slo_ms = 60.0  # Default SLO

        # Filter to cap configs only (not dynamic/maxn)
        cap_only = caps[caps['control_mode'] == 'cap'].copy()
        if cap_only.empty:
            return None

        # Filter by SLO
        feasible = cap_only[cap_only['tpot_ms_median'] <= tpot_slo_ms].copy()
        if ttft_slo_ms is not None:
            feasible = feasible[feasible['ttft_ms_median'] <= ttft_slo_ms]

        if feasible.empty:
            # No cap meets SLO, return cap 1300 (highest)
            logger.warning(f"No cap meets SLO {tpot_slo_ms}ms for p={pl},o={ol}")
            fallback = cap_only[cap_only['target_gpu_cap_mhz'] == cap_only['target_gpu_cap_mhz'].max()]
            if fallback.empty:
                return None
            return self._build_result(fallback.iloc[0], pl, ol, 'slo_constrained',
                                      baseline, slo_met=False)

        # Among feasible, pick lowest cap (most power savings)
        best = feasible.loc[feasible['target_gpu_cap_mhz'].idxmin()]
        return self._build_result(best, pl, ol, 'slo_constrained', baseline, slo_met=True)

    def _select_min_energy(self, caps: pd.DataFrame, pl: int, ol: int,
                           baseline: str) -> Optional[Dict]:
        """Select cap with minimum E/tok."""
        cap_only = caps[caps['control_mode'] == 'cap']
        if cap_only.empty:
            return None
        best = cap_only.loc[cap_only['energy_per_token_j_median'].idxmin()]
        return self._build_result(best, pl, ol, 'min_energy', baseline)

    def _select_alpha(self, caps: pd.DataFrame, pl: int, ol: int,
                      alpha: float, baseline: str) -> Optional[Dict]:
        """Alpha-weighted selection between energy and latency."""
        cap_only = caps[caps['control_mode'] == 'cap'].copy()
        if cap_only.empty:
            return None

        e_col = 'energy_per_token_j_median'
        l_col = 'tpot_ms_median'

        # Normalize
        e_vals = cap_only[e_col]
        l_vals = cap_only[l_col]
        e_norm = (e_vals - e_vals.min()) / (e_vals.max() - e_vals.min() + 1e-12)
        l_norm = (l_vals - l_vals.min()) / (l_vals.max() - l_vals.min() + 1e-12)

        scores = alpha * l_norm + (1 - alpha) * e_norm
        best_idx = scores.idxmin()
        best = cap_only.loc[best_idx]
        result = self._build_result(best, pl, ol, 'alpha_weighted', baseline)
        result['alpha'] = alpha
        result['score'] = float(scores[best_idx])
        return result

    def _select_power_budget(self, caps: pd.DataFrame, pl: int, ol: int,
                             power_budget_w: float, baseline: str) -> Optional[Dict]:
        """Find highest-performing cap within power budget."""
        if power_budget_w is None:
            power_budget_w = 45.0

        cap_only = caps[caps['control_mode'] == 'cap'].copy()
        if cap_only.empty:
            return None

        feasible = cap_only[cap_only['avg_power_w_median'] <= power_budget_w]
        if feasible.empty:
            return None

        # Within budget, pick lowest TPOT
        best = feasible.loc[feasible['tpot_ms_median'].idxmin()]
        return self._build_result(best, pl, ol, 'power_budget', baseline)

    def _select_pareto(self, pl: int, ol: int, baseline: str,
                       tpot_slo_ms: float = None,
                       power_budget_w: float = None,
                       alpha: float = None) -> Optional[Dict]:
        """Multi-objective Pareto selection. Uses cap-mode frontier if available,
        falls back to lock-mode frontier for richer analysis."""
        if self._pareto is None:
            logger.warning("ParetoSelector not available, falling back to min_energy")
            caps = self._find_workload_caps(pl, ol)
            return self._select_min_energy(caps, pl, ol, baseline)

        # Try cap-mode first (deployment-relevant), then lock-mode
        frontier = self._pareto.compute_frontier(pl, ol, phase='decode', source='cap')
        if frontier.n_frontier == 0:
            frontier = self._pareto.compute_frontier(pl, ol, phase='decode', source='lock')

        if frontier.n_frontier == 0:
            logger.warning(f"No Pareto frontier for p={pl}, o={ol}")
            caps = self._find_workload_caps(pl, ol)
            return self._select_min_energy(caps, pl, ol, baseline)

        # Select from frontier (default knee_point for best compromise)
        selected = self._pareto.select_from_frontier(
            frontier,
            strategy='knee_point',
            tpot_slo_ms=tpot_slo_ms,
            power_budget_w=power_budget_w,
            alpha=alpha,
        )

        if selected is None:
            return None

        # Build result in the same format as other strategies
        result = {
            'prompt_length': pl,
            'output_length': ol,
            'strategy': 'pareto',
            'pareto_strategy': selected.get('pareto_strategy', 'knee_point'),
            'pareto_frontier_size': selected.get('pareto_frontier_size', frontier.n_frontier),
            'pareto_source': frontier.source,
        }

        # Map to the standard result keys
        if frontier.source == 'cap':
            result['gpu_cap_mhz'] = selected.get('gpu_freq_mhz', 0)
            result['control_mode'] = 'cap'
        else:
            result['gpu_cap_mhz'] = selected.get('gpu_freq_mhz', 0)
            result['control_mode'] = selected.get('control_mode', 'lock')

        result['pred_tpot_ms'] = selected.get('tpot_ms', 0)
        result['pred_energy_per_token_j'] = selected.get('energy_per_token_j', 0)
        result['pred_avg_power_w'] = selected.get('avg_power_w', 0)
        result['pred_tokens_per_second'] = selected.get('tokens_per_second', 0)
        result['pred_ttft_ms'] = selected.get('ttft_ms', 0)
        result['slo_met'] = True

        # Add savings vs baseline
        caps = self._find_workload_caps(pl, ol)
        base_row = caps[caps['control_mode'] == baseline]
        if not base_row.empty:
            b = base_row.iloc[0]
            for metric, key in [('energy_per_token_j', 'energy_per_token_j'),
                                ('tpot_ms', 'tpot_ms'), ('avg_power_w', 'avg_power_w')]:
                b_val = float(b[f'{metric}_median']) if f'{metric}_median' in b.index else 0
                s_val = result[f'pred_{key.replace("_ms","_ms")}'] if key == metric else selected.get(metric, 0)
                if b_val > 0:
                    result[f'savings_{key}_vs_{baseline}_pct'] = round((b_val - s_val) / b_val * 100, 1)

        return result

    def get_pareto_frontier(self, prompt_length: int, output_length: int,
                             source: str = 'cap', phase: str = 'decode') -> Optional[ParetoFrontierResult]:
        """Get the full Pareto frontier for a workload (for visualization or custom selection)."""
        if self._pareto is None:
            return None
        return self._pareto.compute_frontier(prompt_length, output_length, phase=phase, source=source)

    def _build_result(self, row: pd.Series, pl: int, ol: int,
                      strategy: str, baseline: str,
                      slo_met: bool = True) -> Dict:
        """Build result dict from a rate table row."""
        result = {
            'prompt_length': pl,
            'output_length': ol,
            'strategy': strategy,
            'gpu_cap_mhz': int(row['target_gpu_cap_mhz']),
            'control_mode': row['control_mode'],
            'pred_tpot_ms': round(float(row['tpot_ms_median']), 1),
            'pred_ttft_ms': round(float(row.get('ttft_ms_median', 0)), 1),
            'pred_energy_per_token_j': round(float(row['energy_per_token_j_median']), 4),
            'pred_avg_power_w': round(float(row['avg_power_w_median']), 1),
            'pred_tokens_per_second': round(float(row.get('tokens_per_second_median', 0)), 1),
            'slo_met': slo_met,
        }

        # Add actual GPU freq if available
        if 'actual_gpu_mhz_median' in row.index:
            result['actual_gpu_mhz'] = round(float(row['actual_gpu_mhz_median']), 0)

        # Add savings vs baseline
        sav_ept = row.get(f'savings_ept_vs_{baseline}_pct', np.nan)
        sav_tpot = row.get(f'savings_tpot_vs_{baseline}_pct', np.nan)
        sav_pwr = row.get(f'savings_pwr_vs_{baseline}_pct', np.nan)
        if not np.isnan(sav_ept) if isinstance(sav_ept, float) else True:
            result[f'savings_ept_vs_{baseline}_pct'] = round(float(sav_ept), 1)
        if not np.isnan(sav_tpot) if isinstance(sav_tpot, float) else True:
            result[f'savings_tpot_vs_{baseline}_pct'] = round(float(sav_tpot), 1)
        if not np.isnan(sav_pwr) if isinstance(sav_pwr, float) else True:
            result[f'savings_pwr_vs_{baseline}_pct'] = round(float(sav_pwr), 1)

        return result

    def get_baseline_metrics(self, prompt_length: int, output_length: int,
                             baseline: str = 'dynamic') -> Optional[Dict]:
        """Get baseline metrics for a workload."""
        caps = self._find_workload_caps(prompt_length, output_length)
        base = caps[caps['control_mode'] == baseline]
        if base.empty:
            return None
        row = base.iloc[0]
        return {
            'prompt_length': prompt_length,
            'output_length': output_length,
            'control_mode': baseline,
            'tpot_ms': round(float(row['tpot_ms_median']), 1),
            'ttft_ms': round(float(row.get('ttft_ms_median', 0)), 1),
            'energy_per_token_j': round(float(row['energy_per_token_j_median']), 4),
            'avg_power_w': round(float(row['avg_power_w_median']), 1),
        }

    def sweep_strategies(self, prompt_length: int, output_length: int) -> List[Dict]:
        """Run all selection strategies for a workload and return comparison."""
        results = []
        for strategy in ['slo_constrained', 'min_energy', 'alpha_weighted', 'power_budget', 'pareto']:
            kwargs = {}
            if strategy == 'slo_constrained':
                kwargs['tpot_slo_ms'] = 50.0
            elif strategy == 'alpha_weighted':
                kwargs['alpha'] = 0.5
            elif strategy == 'power_budget':
                kwargs['power_budget_w'] = 45.0

            cfg = self.select(prompt_length, output_length, strategy=strategy, **kwargs)
            if cfg:
                results.append(cfg)

        # Add baselines
        for baseline in ['dynamic', 'maxn']:
            base = self.get_baseline_metrics(prompt_length, output_length, baseline)
            if base:
                results.append(base)

        return results

    def summarize_workload_space(self) -> pd.DataFrame:
        """Summarize DVFS space per workload from lock-mode data."""
        if self.lock_table.empty:
            return pd.DataFrame()

        rows = []
        for phase in self.lock_table['phase'].unique():
            phase_df = self.lock_table[self.lock_table['phase'] == phase]
            for wl in phase_df['workload'].unique():
                wl_df = phase_df[phase_df['workload'] == wl]
                if wl_df.empty:
                    continue
                ept_min = wl_df['energy_per_token_j_median'].min()
                ept_max = wl_df['energy_per_token_j_median'].max()
                tpot_min = wl_df['tpot_ms_median'].min()
                tpot_max = wl_df['tpot_ms_median'].max()
                best_e = wl_df.loc[wl_df['energy_per_token_j_median'].idxmin()]
                rows.append({
                    'workload': wl,
                    'prompt_length': int(wl_df['prompt_length'].iloc[0]),
                    'output_length': int(wl_df['output_length'].iloc[0]),
                    'phase': phase,
                    'best_gpu_for_ept': int(best_e['gpu_freq_mhz']),
                    'ept_range_pct': round((ept_max - ept_min) / ept_min * 100, 1),
                    'tpot_range_pct': round((tpot_max - tpot_min) / tpot_min * 100, 1),
                    'ept_best': round(ept_min, 4),
                    'ept_worst': round(ept_max, 4),
                    'tpot_best': round(tpot_min, 1),
                    'tpot_worst': round(tpot_max, 1),
                })
        return pd.DataFrame(rows)
