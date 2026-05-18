#!/usr/bin/env python3
"""
Weighted Multi-Objective Config Selector

Selects GPU/CPU/EMC frequency configs from the energy rate table using a
single knob `alpha` that trades off energy efficiency vs. latency:

  alpha = 0.0  → pure energy optimization (lowest frequency)
  alpha = 1.0  → pure latency optimization (highest frequency)

Supports both legacy rate tables (GPU×CPU only) and fine-grained rate tables
(GPU×EMC×CPU). Auto-detects column suffixes (_mean or _median).

For phase-aware DVFS, prefill and decode configs are selected independently
using phase-appropriate metrics (TTFT for prefill, TPOT+E/token for decode).
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class WeightedSelector:
    """Select configs from rate table with adjustable energy/latency trade-off."""

    def __init__(self, rate_table_path: str):
        self.rate_table_path = rate_table_path
        self.rate_table = pd.read_parquet(rate_table_path)
        self._suffix = self._detect_suffix()
        self._has_emc = 'emc_freq_mhz' in self.rate_table.columns
        logger.info(f"WeightedSelector: loaded {len(self.rate_table)} rows "
                     f"(suffix={self._suffix}, has_emc={self._has_emc})")

    def _detect_suffix(self) -> str:
        """Auto-detect whether rate table uses _mean or _median suffixes."""
        for col in self.rate_table.columns:
            if col == 'energy_per_token_j_mean':
                return '_mean'
            if col == 'energy_per_token_j_median':
                return '_median'
        return '_median'

    def _col(self, base: str) -> str:
        """Get the suffixed column name."""
        return f'{base}{self._suffix}'

    def _normalize(self, series: pd.Series) -> pd.Series:
        """Min-max normalize a series to [0, 1]."""
        mn, mx = series.min(), series.max()
        if mx - mn < 1e-12:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - mn) / (mx - mn)

    def _score(self, bucket_df: pd.DataFrame, alpha: float,
               energy_col: str, latency_col: str) -> pd.Series:
        """Compute weighted score for each config. Lower is better."""
        return alpha * self._normalize(bucket_df[latency_col]) + \
               (1 - alpha) * self._normalize(bucket_df[energy_col])

    def _config_dict(self, row: pd.Series, alpha: float, phase: str,
                     score: float) -> Dict:
        """Build a config result dict from a rate table row."""
        result = {
            'gpu_freq_mhz': int(row['gpu_freq_mhz']),
            'cpu_freq_mhz': int(row['cpu_freq_mhz']),
            'alpha': alpha,
            'phase': phase,
            'score': float(score),
            'pred_ttft_ms': float(row[self._col('ttft_ms')]),
            'pred_tpot_ms': float(row[self._col('tpot_ms')]),
            'pred_tps': float(row[self._col('tokens_per_second')]),
            'pred_energy_per_token_j': float(row[self._col('energy_per_token_j')]),
            'pred_avg_power_w': float(row[self._col('avg_power_w')]),
        }
        if self._has_emc:
            result['emc_freq_mhz'] = int(row['emc_freq_mhz'])
        if 'config_name' in row.index:
            result['config_name'] = row['config_name']
        elif self._has_emc:
            result['config_name'] = (
                f"GPU{int(row['gpu_freq_mhz'])}_EMC{int(row['emc_freq_mhz'])}"
                f"_CPU{int(row['cpu_freq_mhz'])}"
            )
        else:
            result['config_name'] = (
                f"GPU{int(row['gpu_freq_mhz'])}_CPU{int(row['cpu_freq_mhz'])}"
            )
        return result

    def select_config(self, prompt_length: int, output_length: int,
                      phase: str, alpha: float) -> Optional[Dict]:
        """
        Select best config for a (workload, phase, alpha) combination.

        Args:
            prompt_length: input token count
            output_length: output token count
            phase: 'mixed' or 'decode'
            alpha: energy(0)-latency(1) trade-off knob

        Returns:
            Dict with config info and predicted metrics, or None.
        """
        bucket_df = self._get_bucket(prompt_length, output_length, phase)
        if bucket_df.empty:
            return None

        e_col = self._col('energy_per_token_j')
        l_col = self._col('tpot_ms')
        scores = self._score(bucket_df, alpha, e_col, l_col)
        best_idx = scores.idxmin()
        best = bucket_df.loc[best_idx]

        return self._config_dict(best, alpha, phase, float(scores[best_idx]))

    def select_phase_aware_configs(self, prompt_length: int, output_length: int,
                                   alpha: float) -> Optional[Dict]:
        """
        Select separate prefill and decode configs for phase-aware DVFS.

        Prefill: scored on TTFT + power (latency=TTFT, energy=power).
        Decode: scored on TPOT + E/token (latency=TPOT, energy=E/token).

        Returns dict with both configs, or None.
        """
        prefill_df = self._get_bucket(prompt_length, output_length, 'mixed')
        decode_df = self._get_bucket(prompt_length, output_length, 'decode')
        if prefill_df.empty or decode_df.empty:
            return None

        power_col = self._col('avg_power_w')
        ttft_col = self._col('ttft_ms')
        tpot_col = self._col('tpot_ms')
        e_col = self._col('energy_per_token_j')

        prefill_scores = self._score(prefill_df, alpha, power_col, ttft_col)
        prefill_best = prefill_df.loc[prefill_scores.idxmin()]

        decode_scores = self._score(decode_df, alpha, e_col, tpot_col)
        decode_best = decode_df.loc[decode_scores.idxmin()]

        prefill_cfg = {
            'gpu_freq_mhz': int(prefill_best['gpu_freq_mhz']),
            'cpu_freq_mhz': int(prefill_best['cpu_freq_mhz']),
        }
        decode_cfg = {
            'gpu_freq_mhz': int(decode_best['gpu_freq_mhz']),
            'cpu_freq_mhz': int(decode_best['cpu_freq_mhz']),
        }
        if self._has_emc:
            prefill_cfg['emc_freq_mhz'] = int(prefill_best['emc_freq_mhz'])
            decode_cfg['emc_freq_mhz'] = int(decode_best['emc_freq_mhz'])

        prefill_name = (
            f"GPU{prefill_cfg['gpu_freq_mhz']}"
            + (f"_EMC{prefill_cfg['emc_freq_mhz']}" if 'emc_freq_mhz' in prefill_cfg else "")
            + f"_CPU{prefill_cfg['cpu_freq_mhz']}"
        )
        decode_name = (
            f"GPU{decode_cfg['gpu_freq_mhz']}"
            + (f"_EMC{decode_cfg['emc_freq_mhz']}" if 'emc_freq_mhz' in decode_cfg else "")
            + f"_CPU{decode_cfg['cpu_freq_mhz']}"
        )

        return {
            'alpha': alpha,
            'prefill_config': prefill_cfg,
            'decode_config': decode_cfg,
            'pred_ttft_ms': float(prefill_best[ttft_col]),
            'pred_tpot_ms': float(decode_best[tpot_col]),
            'pred_energy_per_token_j': float(decode_best[e_col]),
            'pred_avg_power_w': float(prefill_best[power_col]),
            'phase_switch': prefill_name != decode_name,
        }

    def sweep_alpha(self, prompt_length: int, output_length: int,
                    phase: str = 'mixed',
                    alphas: Optional[List[float]] = None) -> List[Dict]:
        """Sweep alpha values for offline Pareto analysis."""
        if alphas is None:
            alphas = [round(a * 0.1, 1) for a in range(11)]
        results = []
        for alpha in alphas:
            cfg = self.select_config(prompt_length, output_length, phase, alpha)
            if cfg:
                cfg['prompt_length'] = prompt_length
                cfg['output_length'] = output_length
                results.append(cfg)
        return results

    def sweep_alpha_phase_aware(self, prompt_length: int, output_length: int,
                                alphas: Optional[List[float]] = None) -> List[Dict]:
        """Sweep alpha for phase-aware configs."""
        if alphas is None:
            alphas = [round(a * 0.1, 1) for a in range(11)]
        results = []
        for alpha in alphas:
            cfg = self.select_phase_aware_configs(prompt_length, output_length, alpha)
            if cfg:
                cfg['prompt_length'] = prompt_length
                cfg['output_length'] = output_length
                results.append(cfg)
        return results

    def _get_bucket(self, prompt_length: int, output_length: int,
                    phase: str) -> pd.DataFrame:
        """Get exact or nearest bucket from rate table."""
        mask = (
            (self.rate_table['prompt_length'] == prompt_length) &
            (self.rate_table['output_length'] == output_length) &
            (self.rate_table['phase'] == phase)
        )
        bucket_df = self.rate_table[mask]
        if not bucket_df.empty:
            return bucket_df

        # Nearest bucket by log-distance
        phase_df = self.rate_table[self.rate_table['phase'] == phase]
        if phase_df.empty:
            return pd.DataFrame()
        distances = (
            np.abs(np.log2(phase_df['prompt_length'].astype(float) / prompt_length)) +
            np.abs(np.log2(phase_df['output_length'].astype(float) / output_length))
        )
        min_idx = distances.idxmin()
        best_pl = phase_df.loc[min_idx, 'prompt_length']
        best_ol = phase_df.loc[min_idx, 'output_length']
        return phase_df[
            (phase_df['prompt_length'] == best_pl) &
            (phase_df['output_length'] == best_ol)
        ]

    def get_all_configs_for_workload(self, prompt_length: int, output_length: int,
                                     phase: str) -> pd.DataFrame:
        """Return all configs for a workload bucket (for plotting)."""
        return self._get_bucket(prompt_length, output_length, phase)
