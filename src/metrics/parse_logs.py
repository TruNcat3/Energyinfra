#!/usr/bin/env python3
"""
Log Parser for Energy Profiling
Aligns benchmark and tegrastats logs, calculates derived metrics.
"""

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re
from datetime import datetime
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogParser:
    """
    Parser for aligning and processing benchmark and metrics logs.

    Features:
    - Aligns benchmark logs with tegrastats logs by timestamp
    - Extracts performance metrics (TTFT, TPOT, ITL, throughput)
    - Calculates derived energy metrics (energy/token, tokens/J)
    - Generates structured CSV/Parquet output
    - Provides data quality checks and outlier detection
    """

    def __init__(self, output_dir: str = "data/parsed"):
        """
        Initialize log parser.

        Args:
            output_dir: Directory for parsed output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Parsed data storage
        self.parsed_data = []

        # Data quality metrics
        self.quality_metrics = {}

    def parse_sweep_results(self, sweep_results_file: str,
                          metrics_dir: str = "data/raw_logs") -> pd.DataFrame:
        """
        Parse complete sweep results and align with metrics.

        Args:
            sweep_results_file: Path to sweep results JSON file
            metrics_dir: Directory containing tegrastats log files

        Returns:
            DataFrame with parsed and aligned results
        """
        logger.info(f"Parsing sweep results from: {sweep_results_file}")

        try:
            # Load sweep results
            with open(sweep_results_file, 'r') as f:
                sweep_results = json.load(f)

            if not sweep_results:
                logger.warning("No sweep results found")
                return pd.DataFrame()

            parsed_configs = []

            # Parse each configuration result
            for i, config_result in enumerate(sweep_results):
                logger.info(f"Parsing configuration {i+1}/{len(sweep_results)}")

                try:
                    parsed_config = self._parse_single_config(config_result, metrics_dir)
                    if parsed_config:
                        parsed_configs.append(parsed_config)
                except Exception as e:
                    logger.error(f"Error parsing config {i+1}: {e}")
                    continue

            # Create DataFrame
            if parsed_configs:
                df = pd.DataFrame(parsed_configs)
                logger.info(f"Successfully parsed {len(parsed_configs)} configurations")

                # Save to file
                self._save_parsed_data(df, sweep_results_file)

                return df
            else:
                logger.warning("No valid configurations parsed")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to parse sweep results: {e}")
            return pd.DataFrame()

    def _parse_single_config(self, config_result: Dict, metrics_dir: str) -> Optional[Dict]:
        """
        Parse single configuration result and align with metrics.

        Args:
            config_result: Single configuration result from sweep
            metrics_dir: Directory containing metrics files

        Returns:
            Parsed and aligned configuration data
        """
        try:
            # Extract benchmark results
            benchmark_results = config_result.get('benchmark_results', [])
            benchmark_stats = config_result.get('benchmark_statistics', {})
            energy_metrics = config_result.get('energy_metrics', {})
            config = config_result.get('config', {})
            workload = config.get('workload', {})
            frequency = config.get('frequency', {})

            if not benchmark_results:
                logger.warning("No benchmark results found in configuration")
                return None

            # Calculate aggregated benchmark metrics
            aggregated_metrics = self._aggregate_benchmark_results(benchmark_results)

            # Calculate energy metrics
            calculated_energy = self._calculate_energy_metrics(
                aggregated_metrics, energy_metrics, workload
            )

            # Create parsed record
            parsed_record = {
                # Configuration info
                'config_num': config_result.get('config_num'),
                'experiment_type': config_result.get('experiment_type'),
                'rep': workload.get('rep', 1),
                'timestamp': config_result.get('timestamp'),

                # Workload parameters
                'model': workload.get('model'),
                'batch_size': workload.get('batch_size'),
                'prompt_len': workload.get('prompt_len'),
                'output_len': workload.get('output_len'),
                'phase': workload.get('phase'),
                'concurrency': workload.get('concurrency'),

                # Frequency parameters
                'gpu_freq': frequency.get('gpu_freq'),
                'cpu_freq': frequency.get('cpu_freq'),
                'emc_freq': frequency.get('emc_freq'),

                # Performance metrics
                **aggregated_metrics,

                # Energy metrics
                **calculated_energy,

                # Additional metadata
                'benchmark_log_file': benchmark_results[0].get('log_file') if benchmark_results else None,
                'tegrastats_log_file': config_result.get('metrics_summary', {}).get('log_file')
            }

            return parsed_record

        except Exception as e:
            logger.error(f"Error parsing single configuration: {e}")
            return None

    def _aggregate_benchmark_results(self, benchmark_results: List[Dict]) -> Dict:
        """
        Aggregate multiple benchmark runs into statistics.

        Args:
            benchmark_results: List of individual benchmark results

        Returns:
            Dictionary with aggregated performance metrics
        """
        if not benchmark_results:
            return {}

        aggregated = {}

        # Extract numeric fields
        numeric_fields = ['ttft_ms', 'tpot_ms', 'itl_ms', 'throughput',
                         'latency_p50_ms', 'latency_p95_ms', 'latency_p99_ms',
                         'gpu_utilization_percent', 'gpu_memory_used_gb']

        for field in numeric_fields:
            values = [r.get(field) for r in benchmark_results if field in r and r.get(field) is not None]

            if values:
                aggregated[f'{field}_mean'] = statistics.mean(values)
                aggregated[f'{field}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                aggregated[f'{field}_min'] = min(values)
                aggregated[f'{field}_max'] = max(values)
                aggregated[f'{field}_median'] = statistics.median(values)

                # Calculate coefficient of variation for stability assessment
                if aggregated[f'{field}_mean'] > 0:
                    aggregated[f'{field}_cv'] = (aggregated[f'{field}_std'] / aggregated[f'{field}_mean']) * 100

        # Also include count
        aggregated['num_runs'] = len(benchmark_results)

        return aggregated

    def _calculate_energy_metrics(self, benchmark_metrics: Dict,
                               energy_metrics: Dict, workload: Dict) -> Dict:
        """
        Calculate derived energy metrics.

        Args:
            benchmark_metrics: Aggregated benchmark metrics
            energy_metrics: Raw energy metrics from metrics collector
            workload: Workload configuration

        Returns:
            Dictionary with calculated energy metrics
        """
        calculated = {}

        try:
            # Get basic energy metrics
            duration_sec = energy_metrics.get('duration_sec', 0)
            avg_power_w = energy_metrics.get('avg_power_w', 0)
            total_energy_j = energy_metrics.get('total_energy_j', 0)

            # Get throughput metrics
            throughput = benchmark_metrics.get('throughput_mean', 0)
            output_tokens = workload.get('output_len', 0)
            batch_size = workload.get('batch_size', 1)
            num_runs = benchmark_metrics.get('num_runs', 1)

            # Calculate total tokens generated
            # For a single run: output_len * batch_size
            # For multiple runs: output_len * batch_size * num_runs
            total_tokens = output_tokens * batch_size * num_runs

            if total_tokens > 0 and total_energy_j > 0:
                # Energy per token
                energy_per_token = total_energy_j / total_tokens
                calculated['energy_per_token_j'] = energy_per_token

                # Tokens per joule (energy efficiency)
                tokens_per_joule = total_tokens / total_energy_j
                calculated['tokens_per_joule'] = tokens_per_joule

            # Energy per request
            total_requests = batch_size * num_runs
            if total_requests > 0:
                calculated['energy_per_request_j'] = total_energy_j / total_requests

            # Performance per watt
            if avg_power_w > 0:
                calculated['performance_per_watt'] = throughput / avg_power_w

            # Energy efficiency score (higher is better)
            # Normalize to reasonable range
            calculated['efficiency_score'] = calculated.get('tokens_per_joule', 0) * 100

            # Include raw energy metrics for reference
            calculated['duration_sec'] = duration_sec
            calculated['avg_power_w'] = avg_power_w
            calculated['max_power_w'] = energy_metrics.get('avg_power_w', 0) * 1.5  # Estimate
            calculated['total_energy_j'] = total_energy_j

        except Exception as e:
            logger.error(f"Error calculating energy metrics: {e}")

        return calculated

    def _save_parsed_data(self, df: pd.DataFrame, source_file: str):
        """
        Save parsed data to multiple formats.

        Args:
            df: DataFrame with parsed data
            source_file: Original results file for naming
        """
        try:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(source_file).stem

            # Save as CSV
            csv_file = self.output_dir / f"{base_name}_parsed_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved CSV: {csv_file}")

            # Save as Parquet
            parquet_file = self.output_dir / f"{base_name}_parsed_{timestamp}.parquet"
            df.to_parquet(parquet_file, index=False)
            logger.info(f"Saved Parquet: {parquet_file}")

            # Save summary statistics
            summary_file = self.output_dir / f"{base_name}_summary_{timestamp}.json"
            summary = self._generate_summary_statistics(df)
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Saved summary: {summary_file}")

        except Exception as e:
            logger.error(f"Failed to save parsed data: {e}")

    def _generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the parsed data.

        Args:
            df: DataFrame with parsed data

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_configs': len(df),
            'successful_configs': len(df),
            'parsing_timestamp': datetime.now().isoformat()
        }

        # Performance summary
        performance_fields = ['ttft_ms_mean', 'tpot_ms_mean', 'itl_ms_mean', 'throughput_mean']
        for field in performance_fields:
            if field in df.columns:
                summary[f'{field}_stats'] = {
                    'mean': float(df[field].mean()),
                    'std': float(df[field].std()),
                    'min': float(df[field].min()),
                    'max': float(df[field].max()),
                    'median': float(df[field].median())
                }

        # Energy summary
        energy_fields = ['energy_per_token_j', 'tokens_per_joule', 'avg_power_w']
        for field in energy_fields:
            if field in df.columns:
                summary[f'{field}_stats'] = {
                    'mean': float(df[field].mean()),
                    'std': float(df[field].std()),
                    'min': float(df[field].min()),
                    'max': float(df[field].max()),
                    'median': float(df[field].median())
                }

        # Configuration diversity
        if 'gpu_freq' in df.columns:
            summary['gpu_freq_values'] = df['gpu_freq'].unique().tolist()
        if 'batch_size' in df.columns:
            summary['batch_size_values'] = df['batch_size'].unique().tolist()

        return summary

    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr',
                      multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers in the parsed data.

        Args:
            df: DataFrame with parsed data
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            multiplier: Multiplier for IQR method

        Returns:
            DataFrame with outlier flags
        """
        logger.info(f"Detecting outliers using {method} method")

        df_copy = df.copy()

        if method == 'iqr':
            # IQR method
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col.endswith('_mean') or col in ['energy_per_token_j', 'tokens_per_joule']:
                    Q1 = df_copy[col].quantile(0.25)
                    Q3 = df_copy[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - multiplier * IQR
                    upper_bound = Q3 + multiplier * IQR

                    df_copy[f'{col}_is_outlier'] = (
                        (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
                    )

        elif method == 'zscore':
            # Z-score method
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col.endswith('_mean') or col in ['energy_per_token_j', 'tokens_per_joule']:
                    mean = df_copy[col].mean()
                    std = df_copy[col].std()
                    if std > 0:
                        z_scores = np.abs((df_copy[col] - mean) / std)
                        df_copy[f'{col}_is_outlier'] = z_scores > 3  # 3-sigma rule

        # Count outliers
        outlier_cols = [col for col in df_copy.columns if col.endswith('_is_outlier')]
        if outlier_cols:
            df_copy['has_outlier'] = df_copy[outlier_cols].any(axis=1)

        return df_copy

    def filter_valid_data(self, df: pd.DataFrame,
                       max_ttft_ms: Optional[float] = None,
                       max_tpot_ms: Optional[float] = None,
                       min_throughput: Optional[float] = None,
                       max_energy_token_j: Optional[float] = None) -> pd.DataFrame:
        """
        Filter data to remove invalid or unreasonable results.

        Args:
            df: DataFrame with parsed data
            max_ttft_ms: Maximum allowed TTFT
            max_tpot_ms: Maximum allowed TPOT
            min_throughput: Minimum allowed throughput
            max_energy_token_j: Maximum allowed energy per token

        Returns:
            Filtered DataFrame
        """
        logger.info("Filtering invalid data")

        df_filtered = df.copy()

        # Apply filters
        if max_ttft_ms and 'ttft_ms_mean' in df_filtered.columns:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['ttft_ms_mean'] <= max_ttft_ms]
            removed = before - len(df_filtered)
            logger.info(f"Removed {removed} configs with TTFT > {max_ttft_ms}ms")

        if max_tpot_ms and 'tpot_ms_mean' in df_filtered.columns:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['tpot_ms_mean'] <= max_tpot_ms]
            removed = before - len(df_filtered)
            logger.info(f"Removed {removed} configs with TPOT > {max_tpot_ms}ms")

        if min_throughput and 'throughput_mean' in df_filtered.columns:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['throughput_mean'] >= min_throughput]
            removed = before - len(df_filtered)
            logger.info(f"Removed {removed} configs with throughput < {min_throughput} tokens/s")

        if max_energy_token_j and 'energy_per_token_j' in df_filtered.columns:
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered['energy_per_token_j'] <= max_energy_token_j]
            removed = before - len(df_filtered)
            logger.info(f"Removed {removed} configs with energy/token > {max_energy_token_j}J")

        logger.info(f"Filtered {len(df)} -> {len(df_filtered)} configurations")

        return df_filtered

    def calculate_marginal_costs(self, df: pd.DataFrame,
                               group_by: List[str] = None) -> pd.DataFrame:
        """
        Calculate marginal costs of performance improvements.

        Args:
            df: DataFrame with parsed data
            group_by: Columns to group by for marginal cost calculation

        Returns:
            DataFrame with marginal cost metrics
        """
        if group_by is None:
            group_by = ['gpu_freq', 'cpu_freq', 'emc_freq']

        logger.info("Calculating marginal costs")

        # Sort by energy efficiency
        df_sorted = df.sort_values('tokens_per_joule', ascending=False)

        marginal_costs = []

        for i in range(len(df_sorted) - 1):
            current = df_sorted.iloc[i]
            next_config = df_sorted.iloc[i + 1]

            # Calculate deltas
            delta_tokens_per_joule = current['tokens_per_joule'] - next_config['tokens_per_joule']
            delta_ttft = next_config['ttft_ms_mean'] - current['ttft_ms_mean']
            delta_tpot = next_config['tpot_ms_mean'] - current['tpot_ms_mean']

            marginal_cost = {
                'config_num_current': current['config_num'],
                'config_num_next': next_config['config_num'],
                'delta_tokens_per_joule': delta_tokens_per_joule,
                'delta_ttft_ms': delta_ttft,
                'delta_tpot_ms': delta_tpot,
                'tokens_per_joule_current': current['tokens_per_joule'],
                'tokens_per_joule_next': next_config['tokens_per_joule'],
                'ttft_current': current['ttft_ms_mean'],
                'ttft_next': next_config['ttft_ms_mean']
            }

            # Calculate cost-benefit ratios
            if delta_ttft > 0:
                marginal_cost['cost_per_ms_ttft'] = delta_tokens_per_joule / delta_ttft
            if delta_tpot > 0:
                marginal_cost['cost_per_ms_tpot'] = delta_tokens_per_joule / delta_tpot

            marginal_costs.append(marginal_cost)

        return pd.DataFrame(marginal_costs)

    def generate_pareto_frontier(self, df: pd.DataFrame,
                                objectives: List[str] = None) -> pd.DataFrame:
        """
        Generate Pareto frontier from data.

        Args:
            df: DataFrame with parsed data
            objectives: List of columns for Pareto analysis

        Returns:
            DataFrame with Pareto frontier configs and dominance info
        """
        if objectives is None:
            objectives = ['energy_per_token_j', 'ttft_ms_mean', 'tpot_ms_mean']

        logger.info(f"Generating Pareto frontier for objectives: {objectives}")

        df_copy = df.copy()

        # Check if objectives exist
        valid_objectives = [obj for obj in objectives if obj in df_copy.columns]
        if not valid_objectives:
            logger.error("No valid objectives found for Pareto analysis")
            return df_copy

        # Add dominance info
        df_copy['is_pareto'] = True
        df_copy['dominance_count'] = 0
        df_copy['pareto_rank'] = 0

        # Simple Pareto frontier calculation
        for i in range(len(df_copy)):
            for j in range(len(df_copy)):
                if i != j:
                    config_i = df_copy.iloc[i]
                    config_j = df_copy.iloc[j]

                    # Check if config_j dominates config_i
                    # (all objectives are better or equal, at least one is strictly better)
                    dominates = True
                    is_strictly_better = False

                    for obj in valid_objectives:
                        if config_j[obj] > config_i[obj]:  # Assuming lower is better for energy/latency
                            dominates = False
                            break
                        elif config_j[obj] < config_i[obj]:
                            is_strictly_better = True

                    if dominates and is_strictly_better:
                        df_copy.at[i, 'is_pareto'] = False
                        df_copy.at[i, 'dominance_count'] += 1

        # Calculate Pareto rank
        pareto_configs = df_copy[df_copy['is_pareto']].copy()

        for rank in range(1, 10):  # Max 10 ranks
            remaining_non_pareto = df_copy[~df_copy['is_pareto']].copy()

            if len(remaining_non_pareto) == 0:
                break

            # Calculate rank for remaining configs
            for i in remaining_non_pareto.index:
                if df_copy.at[i, 'pareto_rank'] == 0:
                    # Count how many Pareto configs dominate this one
                    dominated_count = 0
                    for j in pareto_configs.index:
                        dominates = True
                        is_strictly_better = False

                        for obj in valid_objectives:
                            if pareto_configs.at[j, obj] > df_copy.at[i, obj]:
                                dominates = False
                                break
                            elif pareto_configs.at[j, obj] < df_copy.at[i, obj]:
                                is_strictly_better = True

                        if dominates and is_strictly_better:
                            dominated_count += 1

                    if dominated_count == 0:
                        df_copy.at[i, 'pareto_rank'] = rank

            # Update Pareto configs for next rank
            pareto_configs = df_copy[df_copy['pareto_rank'] == rank].copy()

        logger.info(f"Pareto frontier: {len(df_copy[df_copy['is_pareto']])} configs")

        return df_copy


def main():
    """
    Test function for log parser.
    """
    parser = LogParser()

    # Test with example data (would need actual results file)
    logger.info("Testing log parser functionality")

    # Create example DataFrame
    example_data = [
        {
            'config_num': 1,
            'ttft_ms_mean': 245.3,
            'ttft_ms_std': 2.1,
            'tpot_ms_mean': 45.2,
            'tpot_ms_std': 1.8,
            'throughput_mean': 22.1,
            'energy_per_token_j': 9.8,
            'tokens_per_joule': 0.102,
            'avg_power_w': 35.4,
            'gpu_freq': 'high',
            'cpu_freq': 'high',
            'emc_freq': 'high',
            'batch_size': 1,
            'model': 'qwen-7b-int4'
        },
        {
            'config_num': 2,
            'ttft_ms_mean': 310.5,
            'ttft_ms_std': 3.2,
            'tpot_ms_mean': 58.7,
            'tpot_ms_std': 2.5,
            'throughput_mean': 17.0,
            'energy_per_token_j': 7.2,
            'tokens_per_joule': 0.139,
            'avg_power_w': 25.1,
            'gpu_freq': 'mid',
            'cpu_freq': 'mid',
            'emc_freq': 'mid',
            'batch_size': 1,
            'model': 'qwen-7b-int4'
        }
    ]

    df = pd.DataFrame(example_data)

    # Test outlier detection
    df_with_outliers = parser.detect_outliers(df)
    logger.info(f"Outlier detection completed")

    # Test Pareto frontier
    df_with_pareto = parser.generate_pareto_frontier(df)
    logger.info(f"Pareto frontier completed")
    logger.info(f"Pareto configs: {df_with_pareto[df_with_pareto['is_pareto']]}")

    logger.info("Log parser tests completed successfully")


if __name__ == "__main__":
    main()