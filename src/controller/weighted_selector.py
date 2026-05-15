#!/usr/bin/env python3
"""
Weighted Multi-Objective Config Selector

Selects GPU/CPU frequency configs from the energy rate table using a
single knob `alpha` that trades off energy efficiency vs. latency:

  alpha = 0.0  → pure energy optimization (lowest frequency)
  alpha = 1.0  → pure latency optimization (highest frequency)

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
        self.rate_table = pd.read_parquet(rate_table_path)

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

        scores = self._score(bucket_df, alpha,
                             'energy_per_token_j_mean', 'tpot_ms_mean')
        best_idx = scores.idxmin()
        best = bucket_df.loc[best_idx]

        return {
            'config_name': best['config_name'],
            'gpu_freq_mhz': int(best['gpu_freq_mhz']),
            'cpu_freq_mhz': int(best['cpu_freq_mhz']),
            'alpha': alpha,
            'phase': phase,
            'score': float(scores[best_idx]),
            'pred_ttft_ms': float(best['ttft_ms_mean']),
            'pred_tpot_ms': float(best['tpot_ms_mean']),
            'pred_tps': float(best['tokens_per_second_mean']),
            'pred_energy_per_token_j': float(best['energy_per_token_j_mean']),
            'pred_avg_power_w': float(best['avg_power_w_mean']),
        }

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

        prefill_scores = self._score(prefill_df, alpha,
                                     'avg_power_w_mean', 'ttft_ms_mean')
        prefill_best = prefill_df.loc[prefill_scores.idxmin()]

        decode_scores = self._score(decode_df, alpha,
                                    'energy_per_token_j_mean', 'tpot_ms_mean')
        decode_best = decode_df.loc[decode_scores.idxmin()]

        return {
            'alpha': alpha,
            'prefill_config': {
                'config_name': prefill_best['config_name'],
                'gpu_freq_mhz': int(prefill_best['gpu_freq_mhz']),
                'cpu_freq_mhz': int(prefill_best['cpu_freq_mhz']),
            },
            'decode_config': {
                'config_name': decode_best['config_name'],
                'gpu_freq_mhz': int(decode_best['gpu_freq_mhz']),
                'cpu_freq_mhz': int(decode_best['cpu_freq_mhz']),
            },
            'pred_ttft_ms': float(prefill_best['ttft_ms_mean']),
            'pred_tpot_ms': float(decode_best['tpot_ms_mean']),
            'pred_energy_per_token_j': float(decode_best['energy_per_token_j_mean']),
            'pred_avg_power_w': float(decode_best['avg_power_w_mean']),
            'phase_switch': prefill_best['config_name'] != decode_best['config_name'],
        }

    def sweep_alpha(self, prompt_length: int, output_length: int,
                    phase: str = 'mixed',
                    alphas: Optional[List[float]] = None) -> List[Dict]:
        """Sweep alpha values for offline Pareto analysis (no real inference)."""
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