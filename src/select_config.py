#!/usr/bin/env python3
"""
SLO-Aware Configuration Selector
给定 workload + SLO，从 selector_table.parquet 中选择满足 SLO 的最低能耗配置
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigSelector:
    """SLO-aware configuration selector for energy optimization"""

    def __init__(self, selector_table_path: str = "data/rate_tables/selector_table.parquet"):
        self.selector_table_path = Path(selector_table_path)

        # Load selector table
        try:
            self.selector_df = pd.read_parquet(self.selector_table_path)
            logger.info(f"Loaded selector table: {len(self.selector_df)} configurations")
        except Exception as e:
            logger.error(f"Failed to load selector table: {e}")
            self.selector_df = pd.DataFrame()

        # Distance calculation weights for nearest bucket
        self.distance_weights = {
            'batch_size': 1.0,
            'prompt_length': 1.0,
            'output_length': 1.0
        }

        # Maximum distance for nearest bucket matching
        self.max_nearest_distance = 2.0

    def create_bucket_key(self, workload: Dict) -> str:
        """Create bucket key from workload specifications"""

        # Extract workload parameters with defaults
        model = workload.get('model', 'synthetic_qwen_7b_int4')
        runtime = workload.get('runtime', 'synthetic')
        batch_size = workload.get('batch_size', 1)
        prompt_length = workload.get('prompt_length', workload.get('prompt_len', 512))
        output_length = workload.get('output_length', workload.get('output_len', 128))
        phase = workload.get('phase', 'mixed')
        concurrency = workload.get('concurrency', 1)

        # Create bucket key
        bucket_key = f"{model}|{runtime}|{batch_size}|{prompt_length}|{output_length}|{phase}|{concurrency}"

        return bucket_key

    def find_exact_bucket(self, workload: Dict) -> pd.DataFrame:
        """Find configurations for exact workload bucket match"""

        bucket_key = self.create_bucket_key(workload)
        exact_bucket = self.selector_df[self.selector_df['bucket_key'] == bucket_key]

        return exact_bucket

    def calculate_bucket_distance(self, workload: Dict, bucket_row: pd.Series) -> float:
        """Calculate distance between query workload and bucket"""

        distance = 0.0

        # Batch size distance
        query_batch = workload.get('batch_size', 1)
        bucket_batch = bucket_row.get('batch_size', 1)
        if query_batch > 0 and bucket_batch > 0:
            distance += self.distance_weights['batch_size'] * abs(np.log(query_batch / bucket_batch))

        # Prompt length distance
        query_prompt = workload.get('prompt_length', workload.get('prompt_len', 512))
        bucket_prompt = bucket_row.get('prompt_length', 512)
        if query_prompt > 0 and bucket_prompt > 0:
            distance += self.distance_weights['prompt_length'] * abs(np.log(query_prompt / bucket_prompt))

        # Output length distance
        query_output = workload.get('output_length', workload.get('output_len', 128))
        bucket_output = bucket_row.get('output_length', 128)
        if query_output > 0 and bucket_output > 0:
            distance += self.distance_weights['output_length'] * abs(np.log(query_output / bucket_output))

        # Phase mismatch penalty
        query_phase = workload.get('phase', 'mixed')
        bucket_phase = bucket_row.get('phase', 'mixed')
        if query_phase != bucket_phase:
            distance += 5.0  # Large penalty for phase mismatch

        return distance

    def find_nearest_bucket(self, workload: Dict) -> Tuple[pd.DataFrame, str]:
        """Find nearest workload bucket and return match reason"""

        if self.selector_df.empty:
            return pd.DataFrame(), "no_data"

        # Calculate distances to all buckets
        distances = []
        for idx, row in self.selector_df.iterrows():
            distance = self.calculate_bucket_distance(workload, row)
            distances.append((idx, distance, row['bucket_key']))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Get nearest bucket
        if distances:
            nearest_idx, nearest_distance, nearest_bucket_key = distances[0]

            if nearest_distance <= self.max_nearest_distance:
                nearest_bucket = self.selector_df[self.selector_df['bucket_key'] == nearest_bucket_key]
                match_reason = f"nearest_bucket_distance_{nearest_distance:.2f}"
                return nearest_bucket, match_reason
            else:
                return pd.DataFrame(), "nearest_too_far"

        return pd.DataFrame(), "no_suitable_bucket"

    def filter_by_slo(self, bucket_df: pd.DataFrame, slo: Dict) -> pd.DataFrame:
        """Filter configurations by SLO constraints"""

        if bucket_df.empty:
            return bucket_df

        # Start with all configurations
        filtered_df = bucket_df.copy()

        # Apply SLO filters (only hard constraints)
        if 'ttft_ms' in slo and 'ttft_ms_median' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['ttft_ms_median'] <= slo['ttft_ms']]

        if 'tpot_ms' in slo and 'tpot_ms_median' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['tpot_ms_median'] <= slo['tpot_ms']]

        if 'max_power_w' in slo and 'avg_power_w_median' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['avg_power_w_median'] <= slo['max_power_w']]

        if 'max_temp_c' in slo and 'temperature_c_median' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['temperature_c_median'] <= slo['max_temp_c']]

        return filtered_df

    def select_min_energy_config(self, configs_df: pd.DataFrame) -> pd.Series:
        """Select configuration with minimum energy per token"""

        if configs_df.empty:
            return pd.Series()

        # Filter to only SLO-feasible configs
        slo_feasible = configs_df[configs_df['slo_all_met'] == True]

        if not slo_feasible.empty:
            # Among SLO-feasible, select minimum energy
            min_energy_idx = slo_feasible['energy_per_token_j_median'].idxmin()
            return slo_feasible.loc[min_energy_idx]
        else:
            # No SLO-feasible configs, select one with minimum SLO violations
            # Use energy as secondary criterion
            min_energy_idx = configs_df['energy_per_token_j_median'].idxmin()
            return configs_df.loc[min_energy_idx]

    def find_fallback_safe_config(self) -> pd.Series:
        """Find known safe fallback configuration"""

        if self.selector_df.empty:
            return pd.Series()

        # Look for marked fallback safe configs
        fallback_configs = self.selector_df[self.selector_df.get('is_fallback_safe', False) == True]

        if not fallback_configs.empty:
            # Return first fallback config
            return fallback_configs.iloc[0]

        # If no marked fallback, use mid-range frequencies as safe default
        safe_config = self.selector_df[
            (self.selector_df['gpu_freq_mhz'] == 846) &
            (self.selector_df['cpu_freq_mhz'] == 1479) &
            (self.selector_df['emc_freq_mhz'] == 1600)
        ]

        if not safe_config.empty:
            return safe_config.iloc[0]

        # Ultimate fallback: return first configuration
        return self.selector_df.iloc[0]

    def select_config(self, workload: Dict, slo: Dict) -> Dict:
        """Main selection logic"""

        result = {
            'selected_config': {},
            'predicted_metrics': {},
            'selection_reason': '',
            'fallback_used': False,
            'matched_bucket': ''
        }

        if self.selector_df.empty:
            result['selection_reason'] = 'no_available_config'
            return result

        # Try exact bucket match first
        exact_bucket = self.find_exact_bucket(workload)

        if not exact_bucket.empty:
            # Filter by SLO
            slo_filtered = self.filter_by_slo(exact_bucket, slo)

            if not slo_filtered.empty:
                # Select minimum energy config
                selected = self.select_min_energy_config(slo_filtered)
                result.update(self.format_selection_result(selected, 'exact_min_energy_under_slo', 'exact'))
                return result
            else:
                # No SLO-feasible configs in exact bucket
                selected = self.select_min_energy_config(exact_bucket)
                result.update(self.format_selection_result(selected, 'closest_slo_feasible', 'exact'))
                result['fallback_used'] = True
                return result

        # Try nearest bucket
        nearest_bucket, match_reason = self.find_nearest_bucket(workload)

        if not nearest_bucket.empty:
            # Filter by SLO
            slo_filtered = self.filter_by_slo(nearest_bucket, slo)

            if not slo_filtered.empty:
                # Select minimum energy config
                selected = self.select_min_energy_config(slo_filtered)
                result.update(self.format_selection_result(selected, f'nearest_min_energy_under_slo_{match_reason}', 'nearest'))
                return result
            else:
                # No SLO-feasible configs in nearest bucket
                selected = self.select_min_energy_config(nearest_bucket)
                result.update(self.format_selection_result(selected, f'nearest_closest_slo_feasible_{match_reason}', 'nearest'))
                result['fallback_used'] = True
                return result

        # Fallback to known safe config
        fallback_config = self.find_fallback_safe_config()

        if not fallback_config.empty:
            result.update(self.format_selection_result(fallback_config, 'fallback_known_safe', 'fallback'))
            result['fallback_used'] = True
            return result

        # No available config
        result['selection_reason'] = 'no_available_config'
        return result

    def format_selection_result(self, config: pd.Series, reason: str, bucket_type: str) -> Dict:
        """Format selection result"""

        result = {
            'selected_config': {
                'gpu_freq_mhz': int(config.get('gpu_freq_mhz', 846)),
                'cpu_freq_mhz': int(config.get('cpu_freq_mhz', 1479)),
                'emc_freq_mhz': int(config.get('emc_freq_mhz', 1600))
            },
            'predicted_metrics': {
                'ttft_ms': float(config.get('ttft_ms_median', 0)),
                'tpot_ms': float(config.get('tpot_ms_median', 0)),
                'energy_per_token_j': float(config.get('energy_per_token_j_median', 0)),
                'tokens_per_joule': float(config.get('tokens_per_joule_median', 0)),
                'avg_power_w': float(config.get('avg_power_w_median', 0)),
                'temperature_c': float(config.get('temperature_c', 50))
            },
            'selection_reason': reason,
            'fallback_used': False,
            'matched_bucket': bucket_type,
            'slo_feasible': bool(config.get('slo_all_met', True)),
            'is_pareto': bool(config.get('is_pareto', False)),
            'pareto_rank': int(config.get('pareto_rank', 1))
        }

        return result

    def batch_select(self, workloads: list, slo: Dict) -> list:
        """Batch select configurations for multiple workloads"""

        results = []
        for workload in workloads:
            result = self.select_config(workload, slo)
            results.append({
                'workload': workload,
                'selection': result
            })

        return results


def create_example_workload_query() -> Dict:
    """Create example workload query for testing"""

    return {
        "model": "synthetic_qwen_7b_int4",
        "runtime": "synthetic",
        "batch_size": 1,
        "prompt_length": 512,
        "output_length": 128,
        "phase": "mixed",
        "concurrency": 1,
        "slo": {
            "ttft_ms": 1000,
            "tpot_ms": 80,
            "max_power_w": 40,
            "max_temp_c": 80
        }
    }


def main():
    """Main function for command-line usage"""

    parser = argparse.ArgumentParser(description='SLO-Aware Configuration Selector')
    parser.add_argument('--workload', type=str, help='Path to workload query JSON file')
    parser.add_argument('--selector-table', type=str,
                       default='data/rate_tables/selector_table.parquet',
                       help='Path to selector table parquet file')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--example', action='store_true', help='Use example workload query')

    args = parser.parse_args()

    # Initialize selector
    selector = ConfigSelector(args.selector_table)

    # Load or create workload query
    if args.example:
        workload_query = create_example_workload_query()
        logger.info("Using example workload query")
    elif args.workload:
        with open(args.workload, 'r') as f:
            workload_query = json.load(f)
        logger.info(f"Loaded workload query from: {args.workload}")
    else:
        # Default to example
        workload_query = create_example_workload_query()
        logger.info("No workload specified, using example query")

    # Extract SLO from workload
    slo = workload_query.pop('slo', {})

    # Select configuration
    result = selector.select_config(workload_query, slo)

    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to: {args.output}")
    else:
        print(json.dumps(result, indent=2))

    return 0 if result['selection_reason'] != 'no_available_config' else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())