#!/usr/bin/env python3
"""
Multi-Objective Pareto DVFS Selector

Computes the full Pareto frontier across 3+ objectives for each workload,
and supports constraint-based selection to pick a single operating point.

Objectives (all minimize):
  1. energy_per_token_j  — energy efficiency
  2. tpot_ms             — latency
  3. avg_power_w         — power consumption

Data sources:
  - lock_mode: 11 GPU frequencies (richer frontier, offline analysis)
  - cap_mode:  4 GPU caps + dynamic + MAXN (deployment)

Usage:
    from src.controller.pareto_selector import ParetoSelector
    ps = ParetoSelector(
        lock_rate_table_path='data/rate_tables/lock_rate_table_*.parquet',
        cap_rate_table_path='data/rate_tables/cap_rate_table_with_savings_*.parquet',
        model='Qwen2.5-7B-Instruct-Q4_K_M')
    )
    frontier = ps.compute_frontier(512, 128, phase='decode', source='lock')
    knee = ps.select_from_frontier(frontier, strategy='knee_point')
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_OBJECTIVES = ['energy_per_token_j', 'tpot_ms', 'avg_power_w']


@dataclass
class ParetoFrontierResult:
    """Result of Pareto frontier computation for a single workload."""
    frontier_df: pd.DataFrame
    all_configs_df: pd.DataFrame
    workload_info: Dict
    source: str
    objectives: List[str]
    n_frontier: int
    n_total: int


def compute_pareto_frontier(
    df: pd.DataFrame,
    objectives: List[str],
    minimize: List[bool] = None,
) -> pd.DataFrame:
    """
    Compute Pareto frontier via non-dominated sorting.

    Args:
        df: DataFrame of candidate configs (one row per config).
        objectives: Column names of objectives.
        minimize: Per-objective direction (True=minimize). Default all True.

    Returns:
        DataFrame with added columns:
          pareto_rank, is_pareto, dominates_count, dominated_by_count
    """
    if minimize is None:
        minimize = [True] * len(objectives)

    available = [o for o in objectives if o in df.columns]
    if not available:
        df['pareto_rank'] = -1
        df['is_pareto'] = False
        df['dominates_count'] = 0
        df['dominated_by_count'] = 0
        return df

    obj_cols = available
    mins = [True] * len(obj_cols)

    vals = df[obj_cols].apply(pd.to_numeric, errors='coerce').fillna(np.inf).copy()
    for i, (col, is_min) in enumerate(zip(obj_cols, mins)):
        if not is_min:
            vals[col] = -vals[col]

    n = len(vals)
    ranks = np.full(n, -1, dtype=float)
    dominates_count = np.zeros(n, dtype=int)
    dominated_by_count = np.zeros(n, dtype=int)
    remaining = set(range(n))
    cur_rank = 1

    while remaining:
        frontier = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if (all(vals.iloc[j][o] <= vals.iloc[i][o] for o in obj_cols) and
                        any(vals.iloc[j][o] < vals.iloc[i][o] for o in obj_cols)):
                    dominated = True
                    dominated_by_count[i] += 1
                    dominates_count[j] += 1
            if not dominated:
                frontier.append(i)
        for idx in frontier:
            ranks[idx] = cur_rank
            remaining.remove(idx)
        cur_rank += 1

    result = df.copy()
    result['pareto_rank'] = ranks
    result['is_pareto'] = ranks == 1
    result['dominates_count'] = dominates_count
    result['dominated_by_count'] = dominated_by_count
    return result


def find_knee_point(
    frontier_df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> int:
    """
    Find knee point on a 2D Pareto frontier using maximum angle change.

    Sorts frontier by x_col, computes angle between consecutive segments,
    returns index where the cumulative direction change is sharpest.

    Returns:
        Index of knee point in the original frontier_df.
    """
    sorted_df = frontier_df.sort_values(x_col).reset_index(drop=True)
    n = len(sorted_df)
    if n <= 2:
        return 0

    x = sorted_df[x_col].values.astype(float)
    y = sorted_df[y_col].values.astype(float)

    best_angle = -1
    best_idx = 0
    for i in range(1, n - 1):
        v1 = np.array([x[i] - x[i-1], y[i] - y[i-1]])
        v2 = np.array([x[i+1] - x[i], y[i+1] - y[i]])
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            angle = 0.0
        else:
            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        if angle > best_angle:
            best_angle = angle
            best_idx = i

    return int(sorted_df.iloc[best_idx].name) if 'name' in sorted_df.columns else best_idx


class ParetoSelector:
    """Multi-objective Pareto DVFS selector.

    Computes the full Pareto frontier across objectives for each workload,
    then supports constraint-based filtering and strategy-based selection.
    """

    def __init__(
        self,
        lock_rate_table_path: str = None,
        cap_rate_table_path: str = None,
        model: str = None,
        objectives: List[str] = None,
    ):
        self.model = model
        self.objectives = objectives or DEFAULT_OBJECTIVES

        self.lock_table = pd.DataFrame()
        self.cap_table = pd.DataFrame()

        if lock_rate_table_path:
            self.lock_table = pd.read_parquet(lock_rate_table_path)
            if model:
                self.lock_table = self.lock_table[self.lock_table['model'] == model]
            logger.info(f"ParetoSelector: lock table {len(self.lock_table)} rows")

        if cap_rate_table_path:
            self.cap_table = pd.read_parquet(cap_rate_table_path)
            if model:
                self.cap_table = self.cap_table[self.cap_table['model'] == model]
            logger.info(f"ParetoSelector: cap table {len(self.cap_table)} rows")

    def _detect_suffix(self, df: pd.DataFrame) -> str:
        for col in df.columns:
            if col == 'energy_per_token_j_median':
                return '_median'
            if col == 'energy_per_token_j_mean':
                return '_mean'
        return '_median'

    def _suffix_cols(self, base: str, suffix: str) -> str:
        return f'{base}{suffix}'

    def compute_frontier(
        self,
        prompt_length: int,
        output_length: int,
        phase: str = 'decode',
        source: str = 'lock',
    ) -> ParetoFrontierResult:
        """
        Compute the Pareto frontier for a workload.

        Args:
            prompt_length: Input token count.
            output_length: Output token count.
            phase: 'decode' or 'mixed'.
            source: 'lock' (11 freqs) or 'cap' (4 caps + baselines).

        Returns:
            ParetoFrontierResult with frontier and all configs with Pareto ranks.
        """
        table = self.lock_table if source == 'lock' else self.cap_table
        if table.empty:
            logger.warning(f"No {source} data available")
            return ParetoFrontierResult(
                frontier_df=pd.DataFrame(), all_configs_df=pd.DataFrame(),
                workload_info={}, source=source, objectives=self.objectives,
                n_frontier=0, n_total=0,
            )

        suffix = self._detect_suffix(table)

        if source == 'lock':
            mask = (
                (table['prompt_length'] == prompt_length) &
                (table['output_length'] == output_length) &
                (table['phase'] == phase)
            )
        else:
            mask = (
                (table['prompt_length'] == prompt_length) &
                (table['output_length'] == output_length)
            )

        configs = table[mask].copy()

        # For cap mode, separate caps from baselines for frontier computation
        if source == 'cap' and 'control_mode' in configs.columns:
            cap_only = configs[configs['control_mode'] == 'cap'].copy()
            baselines = configs[configs['control_mode'].isin(['dynamic', 'maxn'])].copy()
        else:
            cap_only = configs
            baselines = pd.DataFrame()

        if cap_only.empty:
            return ParetoFrontierResult(
                frontier_df=pd.DataFrame(), all_configs_df=configs,
                workload_info={}, source=source, objectives=self.objectives,
                n_frontier=0, n_total=len(configs),
            )

        obj_cols = [self._suffix_cols(o, suffix) for o in self.objectives]
        available = [c for c in obj_cols if c in cap_only.columns]

        with_ranks = compute_pareto_frontier(cap_only, available)

        # Add baselines back (rank > 1) — for lock mode, no baselines exist
        if not baselines.empty:
            baselines['pareto_rank'] = 99
            baselines['is_pareto'] = False
            baselines['dominates_count'] = 0
            baselines['dominated_by_count'] = 0
            with_ranks = pd.concat([with_ranks, baselines], ignore_index=True)

        frontier_df = with_ranks[with_ranks['is_pareto'] == True].copy()
        n_frontier = len(frontier_df)

        workload_info = {
            'prompt_length': prompt_length,
            'output_length': output_length,
            'phase': phase,
            'model': self.model or 'unknown',
        }

        return ParetoFrontierResult(
            frontier_df=frontier_df,
            all_configs_df=with_ranks,
            workload_info=workload_info,
            source=source,
            objectives=self.objectives,
            n_frontier=n_frontier,
            n_total=len(configs),
        )

    def select_from_frontier(
        self,
        frontier: ParetoFrontierResult,
        strategy: str = 'knee_point',
        tpot_slo_ms: float = None,
        power_budget_w: float = None,
        alpha: float = None,
        weights: Dict[str, float] = None,
    ) -> Optional[Dict]:
        """
        Select a single config from the Pareto frontier.

        Strategies:
          - knee_point: Maximum curvature point (best compromise)
          - slo_constrained: Lowest E/tok meeting TPOT SLO
          - power_budget: Lowest TPOT within power budget
          - weighted: Scalarized weighted sum
          - closest_to_ideal: Closest to ideal point

        Returns:
            Dict with selected config + predicted metrics, or None.
        """
        if frontier.n_frontier == 0:
            logger.warning("Empty frontier, cannot select")
            return None

        filtered = frontier

        # Apply constraints
        if tpot_slo_ms is not None:
            filtered = self.filter_frontier(frontier, tpot_max_ms=tpot_slo_ms)
            if filtered.n_frontier == 0:
                logger.warning(f"No frontier configs meet TPOT SLO {tpot_slo_ms}ms, "
                               f"falling back to unconstrained")
                filtered = frontier

        if power_budget_w is not None:
            filtered = self.filter_frontier(filtered, power_max_w=power_budget_w)
            if filtered.n_frontier == 0:
                logger.warning(f"No frontier configs meet power budget {power_budget_w}W, "
                               f"falling back to unconstrained")
                filtered = frontier

        suffix = self._detect_suffix(filtered.frontier_df)

        if strategy == 'knee_point':
            return self._select_knee_point(filtered, suffix)
        elif strategy == 'slo_constrained':
            return self._select_slo_constrained(filtered, suffix)
        elif strategy == 'power_budget':
            return self._select_power_budget(filtered, suffix)
        elif strategy == 'weighted':
            return self._select_weighted(filtered, suffix, alpha, weights)
        elif strategy == 'closest_to_ideal':
            return self._select_closest_to_ideal(filtered, suffix)
        else:
            logger.warning(f"Unknown Pareto strategy: {strategy}, using knee_point")
            return self._select_knee_point(filtered, suffix)

    def _select_knee_point(self, frontier: ParetoFrontierResult, suffix: str) -> Optional[Dict]:
        """Select the knee point (maximum curvature) on the frontier."""
        ept_col = self._suffix_cols('energy_per_token_j', suffix)
        tpot_col = self._suffix_cols('tpot_ms', suffix)

        if ept_col not in frontier.frontier_df.columns or tpot_col not in frontier.frontier_df.columns:
            logger.warning("Missing columns for knee detection, returning first frontier point")
            return self._row_to_dict(frontier.frontier_df.iloc[0]) if not frontier.frontier_df.empty else None

        if len(frontier.frontier_df) <= 2:
            return self._row_to_dict(frontier.frontier_df.iloc[0])

        sorted_df = frontier.frontier_df.sort_values(ept_col).reset_index(drop=True)
        idx = find_knee_point(sorted_df, ept_col, tpot_col)

        # idx is the position in sorted_df; get the original index
        original_idx = sorted_df.iloc[idx].name if 'name' in sorted_df.columns else idx
        knee_row = sorted_df.iloc[idx]

        result = self._row_to_dict(knee_row)
        result['pareto_strategy'] = 'knee_point'
        result['pareto_frontier_size'] = frontier.n_frontier
        return result

    def _select_slo_constrained(self, frontier: ParetoFrontierResult, suffix: str) -> Optional[Dict]:
        """Among Pareto-optimal configs, pick the one with lowest E/tok (already filtered by SLO)."""
        ept_col = self._suffix_cols('energy_per_token_j', suffix)
        if ept_col not in frontier.frontier_df.columns:
            return self._row_to_dict(frontier.frontier_df.iloc[0])
        best = frontier.frontier_df.loc[frontier.frontier_df[ept_col].idxmin()]
        result = self._row_to_dict(best)
        result['pareto_strategy'] = 'slo_constrained'
        result['pareto_frontier_size'] = frontier.n_frontier
        return result

    def _select_power_budget(self, frontier: ParetoFrontierResult, suffix: str) -> Optional[Dict]:
        """Among Pareto-optimal configs, pick the one with lowest TPOT (already filtered by power)."""
        tpot_col = self._suffix_cols('tpot_ms', suffix)
        if tpot_col not in frontier.frontier_df.columns:
            return self._row_to_dict(frontier.frontier_df.iloc[0])
        best = frontier.frontier_df.loc[frontier.frontier_df[tpot_col].idxmin()]
        result = self._row_to_dict(best)
        result['pareto_strategy'] = 'power_budget'
        result['pareto_frontier_size'] = frontier.n_frontier
        return result

    def _select_weighted(self, frontier: ParetoFrontierResult, suffix: str,
                         alpha: float = 0.5,
                         weights: Dict[str, float] = None) -> Optional[Dict]:
        """Scalarized weighted sum across objectives."""
        obj_cols = [self._suffix_cols(o, suffix) for o in self.objectives]
        available = [c for c in obj_cols if c in frontier.frontier_df.columns]
        if len(available) < 2:
            return self._row_to_dict(frontier.frontier_df.iloc[0])

        if weights:
            w = [weights.get(o, 1.0) for o in self.objectives]
        else:
            # Default: equal weights, alpha on latency
            w = [1.0] * len(self.objectives)
            w[1] = alpha
            w[0] = 1 - alpha  # energy gets (1-alpha), latency gets alpha

        # Normalize each objective
        scores = np.zeros(len(frontier.frontier_df))
        total_weight = sum(w)
        for i, (col, wi) in enumerate(zip(available, w)):
            vals = frontier.frontier_df[col].astype(float)
            mn, mx = vals.min(), vals.max()
            if mx - mn > 1e-12:
                scores += wi / total_weight * (vals - mn) / (mx - mn)
        best_idx = scores.argmin()
        best = frontier.frontier_df.iloc[best_idx]
        result = self._row_to_dict(best)
        result['pareto_strategy'] = 'weighted'
        result['pareto_frontier_size'] = frontier.n_frontier
        return result

    def _select_closest_to_ideal(self, frontier: ParetoFrontierResult, suffix: str) -> Optional[Dict]:
        """Pick config closest to the ideal point (min in all objectives)."""
        obj_cols = [self._suffix_cols(o, suffix) for o in self.objectives]
        available = [c for c in obj_cols if c in frontier.frontier_df.columns]
        if len(available) < 2:
            return self._row_to_dict(frontier.frontier_df.iloc[0])

        vals = frontier.frontier_df[available].astype(float)
        # Ideal: min of each objective
        ideal = vals.min().values
        dists = np.sqrt(((vals.values - ideal) ** 2).sum(axis=1))
        best_idx = dists.argmin()
        best = frontier.frontier_df.iloc[best_idx]
        result = self._row_to_dict(best)
        result['pareto_strategy'] = 'closest_to_ideal'
        result['pareto_frontier_size'] = frontier.n_frontier
        return result

    def filter_frontier(
        self,
        frontier: ParetoFrontierResult,
        tpot_max_ms: float = None,
        power_max_w: float = None,
        ept_max_j: float = None,
    ) -> ParetoFrontierResult:
        """Filter Pareto frontier by hard constraints."""
        suffix = self._detect_suffix(frontier.all_configs_df)
        filtered = frontier.all_configs_df.copy()

        if tpot_max_ms is not None:
            tpot_col = self._suffix_cols('tpot_ms', suffix)
            if tpot_col in filtered.columns:
                filtered = filtered[filtered[tpot_col] <= tpot_max_ms]

        if power_max_w is not None:
            pwr_col = self._suffix_cols('avg_power_w', suffix)
            if pwr_col in filtered.columns:
                filtered = filtered[filtered[pwr_col] <= power_max_w]

        if ept_max_j is not None:
            ept_col = self._suffix_cols('energy_per_token_j', suffix)
            if ept_col in filtered.columns:
                filtered = filtered[filtered[ept_col] <= ept_max_j]

        frontier_df = filtered[filtered['is_pareto'] == True].copy()

        return ParetoFrontierResult(
            frontier_df=frontier_df,
            all_configs_df=filtered,
            workload_info=frontier.workload_info,
            source=frontier.source,
            objectives=frontier.objectives,
            n_frontier=len(frontier_df),
            n_total=len(filtered),
        )

    def alpha_to_frontier_point(
        self,
        frontier: ParetoFrontierResult,
        alpha: float,
    ) -> Optional[Dict]:
        """
        Map alpha in [0, 1] to a point on the Pareto frontier.

        alpha=0 -> best E/tok, alpha=1 -> best TPOT.
        """
        ept_col = self._suffix_cols('energy_per_token_j', self._detect_suffix(frontier.frontier_df))
        tpot_col = self._suffix_cols('tpot_ms', self._detect_suffix(frontier.frontier_df))

        if ept_col not in frontier.frontier_df.columns or tpot_col not in frontier.frontier_df.columns:
            return None

        sorted_df = frontier.frontier_df.sort_values(ept_col).reset_index(drop=True)
        n = len(sorted_df)
        if n == 0:
            return None

        idx = min(int(round(alpha * (n - 1))), n - 1)
        best = sorted_df.iloc[idx]
        result = self._row_to_dict(best)
        result['pareto_strategy'] = 'alpha_mapped'
        result['alpha'] = alpha
        result['pareto_frontier_size'] = frontier.n_frontier
        return result

    def compare_frontiers(
        self,
        prompt_length: int,
        output_length: int,
        phase: str = 'decode',
    ) -> pd.DataFrame:
        """Compare Pareto frontiers across loaded models for the same workload."""
        rows = []
        for source_name, table in [('lock', self.lock_table), ('cap', self.cap_table)]:
            if table.empty:
                continue
            if 'phase' in table.columns:
                t = table[table['phase'] == phase]
            else:
                t = table
            mask = (t['prompt_length'] == prompt_length) & (t['output_length'] == output_length)
            subset = t[mask]
            if subset.empty:
                continue

            suffix = self._detect_suffix(subset)
            ept_col = self._suffix_cols('energy_per_token_j', suffix)
            tpot_col = self._suffix_cols('tpot_ms', suffix)
            pwr_col = self._suffix_cols('avg_power_w', suffix)

            obj_cols = [c for c in [ept_col, tpot_col, pwr_col] if c in subset.columns]
            available = [c for c in obj_cols]

            with_ranks = compute_pareto_frontier(subset, available) if len(available) >= 2 else subset.copy()
            with_ranks['pareto_rank'] = with_ranks.get('pareto_rank', pd.Series([-1]*len(with_ranks)))

            for _, row in with_ranks.iterrows():
                rows.append({
                    'source': source_name,
                    'model': row.get('model', 'unknown'),
                    'is_pareto': row.get('is_pareto', False),
                    'pareto_rank': row.get('pareto_rank', -1),
                    **{c: row.get(c) for c in available},
                })
        return pd.DataFrame(rows)

    def _row_to_dict(self, row: pd.Series) -> Dict:
        """Convert a DataFrame row to a config dict."""
        suffix = self._detect_suffix(row.to_frame())
        result = {
            'gpu_freq_mhz': int(row.get('gpu_freq_mhz', row.get('target_gpu_cap_mhz', 0))),
            'control_mode': row.get('control_mode', 'lock'),
            'energy_per_token_j': round(float(row.get(self._suffix_cols('energy_per_token_j', suffix), 0)), 4),
            'tpot_ms': round(float(row.get(self._suffix_cols('tpot_ms', suffix), 0)), 1),
            'avg_power_w': round(float(row.get(self._suffix_cols('avg_power_w', suffix), 0)), 1),
            'tokens_per_second': round(float(row.get(self._suffix_cols('tokens_per_second', suffix), 0)), 1),
        }
        if 'ttft_ms' in row.index or self._suffix_cols('ttft_ms', suffix) in row.index:
            result['ttft_ms'] = round(float(row.get(self._suffix_cols('ttft_ms', suffix), 0)), 1)
        return result
