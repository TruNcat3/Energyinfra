#!/usr/bin/env python3
"""
Build Energy Rate Table from Experimental Data
将已有实验数据转换为 bucket-level energy rate table，并生成 Pareto table 和 selector table
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateTableBuilder:
    """Build energy rate tables from experimental data"""

    def __init__(self, data_dir="data/experiments_4_1_to_4_7", output_dir="data/rate_tables"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default values for missing fields (synthetic data)
        self.defaults = {
            'model': 'synthetic_qwen_7b_int4',
            'runtime': 'synthetic',
            'concurrency': 1,
            'power_source': 'synthetic',
            'power_statistics_limited': True
        }

        # Load all experimental data
        self.raw_data = self.load_all_experiments()

    def load_all_experiments(self) -> pd.DataFrame:
        """Load and merge all experimental data"""
        all_data = []

        csv_files = sorted(self.data_dir.glob("experiment_4_*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                all_data.append(df)
                logger.info(f"Loaded {csv_file.name}: {len(df)} rows")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total combined data: {len(combined_df)} rows")
            return combined_df
        else:
            logger.warning("No experimental data found")
            return pd.DataFrame()

    def normalize_data_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and add required fields"""

        # Add default fields
        for field, default_value in self.defaults.items():
            if field not in df.columns:
                df[field] = default_value

        # Phase normalization
        if 'phase' not in df.columns:
            if 'scenario' in df.columns:
                # Map scenario to phase
                phase_mapping = {
                    'prefill': 'prefill',
                    'prefill_heavy': 'prefill',
                    'decode': 'decode',
                    'decode_heavy': 'decode'
                }
                df['phase'] = df['scenario'].map(phase_mapping).fillna('mixed')
            else:
                df['phase'] = 'mixed'

        # Calculate phase-specific energy metrics
        df = self.calculate_phase_energy_metrics(df)

        # Add total tokens
        if 'total_tokens' not in df.columns:
            df['total_tokens'] = df.get('prompt_length', 0) + df.get('output_length', 0)

        # Calculate tokens per joule
        if 'tokens_per_joule' not in df.columns:
            df['tokens_per_joule'] = 1.0 / df['energy_per_token_j'].replace(0, np.nan)

        return df

    def calculate_phase_energy_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate phase-specific energy per token metrics"""

        # Initialize new columns
        df['energy_per_input_token_j'] = np.nan
        df['energy_per_output_token_j'] = np.nan
        df['energy_per_total_token_j'] = np.nan

        for idx, row in df.iterrows():
            phase = row.get('phase', 'mixed')
            total_energy = row.get('total_energy_j', row.get('energy_j', 0))
            prompt_len = row.get('prompt_length', row.get('prompt_len', 0))
            output_len = row.get('output_length', row.get('output_len', 0))

            if phase == 'prefill' and prompt_len > 0:
                df.at[idx, 'energy_per_input_token_j'] = total_energy / prompt_len
                df.at[idx, 'energy_per_total_token_j'] = total_energy / prompt_len

            elif phase == 'decode' and output_len > 0:
                df.at[idx, 'energy_per_output_token_j'] = total_energy / output_len
                df.at[idx, 'energy_per_total_token_j'] = total_energy / output_len

            elif phase == 'mixed':
                if output_len > 0:
                    df.at[idx, 'energy_per_output_token_j'] = total_energy / output_len
                total_tokens = prompt_len + output_len
                if total_tokens > 0:
                    df.at[idx, 'energy_per_total_token_j'] = total_energy / total_tokens

        return df

    def create_workload_buckets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create workload buckets and add bucket identifiers"""

        # Define bucket key columns
        bucket_columns = [
            'model', 'runtime', 'batch_size',
            'prompt_length', 'output_length', 'phase', 'concurrency'
        ]

        # Ensure all bucket columns exist and fill missing values
        for col in bucket_columns:
            if col not in df.columns:
                if col == 'batch_size':
                    df[col] = 1
                elif col == 'prompt_length':
                    df[col] = df.get('prompt_len', 512)
                elif col == 'output_length':
                    df[col] = df.get('output_len', 128)
                else:
                    df[col] = self.defaults.get(col, 'unknown')

            # Fill missing values with defaults
            if col == 'batch_size':
                df[col] = df[col].fillna(1)
            elif col == 'prompt_length':
                df[col] = df[col].fillna(512)
            elif col == 'output_length':
                df[col] = df[col].fillna(128)
            elif col == 'phase':
                df[col] = df[col].fillna('mixed')
            elif col in ['model', 'runtime']:
                df[col] = df[col].fillna(self.defaults.get(col, 'unknown'))
            elif col == 'concurrency':
                df[col] = df[col].fillna(1)

        # Create bucket key string
        df['bucket_key'] = df[bucket_columns].apply(
            lambda row: '|'.join([str(x) for x in row]), axis=1
        )

        return df

    def aggregate_by_config(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by workload bucket + frequency configuration"""

        # Define aggregation columns
        bucket_cols = ['model', 'runtime', 'batch_size', 'prompt_length',
                      'output_length', 'phase', 'concurrency', 'bucket_key']

        freq_cols = ['gpu_freq_mhz', 'cpu_freq_mhz', 'emc_freq_mhz']

        # Metrics to aggregate
        metric_cols = [
            'ttft_ms', 'tpot_ms', 'total_time_ms', 'tokens_per_second',
            'avg_power_w', 'max_power_w', 'temperature_c',
            'energy_per_input_token_j', 'energy_per_output_token_j', 'energy_per_total_token_j',
            'tokens_per_joule', 'energy_per_token_j'
        ]

        # Group by bucket and frequency
        group_cols = bucket_cols + freq_cols
        grouped = df.groupby(group_cols)

        # Calculate statistics
        agg_funcs = {col: ['median', 'mean', 'std', 'count'] for col in metric_cols}

        # Special handling for energy_per_token_j (use existing field)
        if 'energy_per_token_j' in df.columns:
            agg_funcs['energy_per_token_j'] = ['median', 'mean', 'std']

        agg_df = grouped.agg(agg_funcs).reset_index()

        # Flatten column names
        agg_df.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                         for col in agg_df.columns.values]

        # Calculate CV for key metrics
        for metric in ['ttft_ms', 'tpot_ms', 'energy_per_token_j']:
            if f'{metric}_std' in agg_df.columns and f'{metric}_mean' in agg_df.columns:
                agg_df[f'{metric}_cv'] = (
                    agg_df[f'{metric}_std'] / agg_df[f'{metric}_mean'] * 100
                ).fillna(0)

        # Count configurations per bucket
        bucket_config_counts = df.groupby('bucket_key').size()
        agg_df['num_configs_in_bucket'] = agg_df['bucket_key'].map(bucket_config_counts)

        return agg_df

    def calculate_pareto_frontier(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pareto frontier within each workload bucket"""

        # Define Pareto objectives (all minimize)
        pareto_objectives = [
            'energy_per_token_j_median',
            'ttft_ms_median',
            'tpot_ms_median',
            'avg_power_w_median'
        ]

        # Filter to only available objectives
        available_objectives = [obj for obj in pareto_objectives if obj in df.columns]

        if not available_objectives:
            logger.warning("No Pareto objectives available in data")
            df['is_pareto'] = False
            df['pareto_rank'] = len(df)
            return df

        # Calculate Pareto frontier per bucket
        pareto_results = []

        for bucket_key in df['bucket_key'].unique():
            bucket_data = df[df['bucket_key'] == bucket_key].copy()

            if len(bucket_data) == 0:
                continue

            # Calculate Pareto ranks
            bucket_data['pareto_rank'] = self._calculate_pareto_ranks(
                bucket_data, available_objectives
            )

            # Mark Pareto frontier (rank == 1)
            bucket_data['is_pareto'] = (bucket_data['pareto_rank'] == 1)

            # Set Pareto confidence
            if len(bucket_data) == 1:
                bucket_data['pareto_confidence'] = 'low'
            elif bucket_data['is_pareto'].sum() > 1:
                bucket_data['pareto_confidence'] = 'high'
            else:
                bucket_data['pareto_confidence'] = 'medium'

            pareto_results.append(bucket_data)

        if pareto_results:
            return pd.concat(pareto_results, ignore_index=True)
        else:
            df['is_pareto'] = False
            df['pareto_rank'] = len(df)
            df['pareto_confidence'] = 'low'
            return df

    def _calculate_pareto_ranks(self, df: pd.DataFrame, objectives: List[str]) -> pd.Series:
        """Calculate Pareto ranks using non-dominated sorting"""

        if len(df) == 0:
            return pd.Series()

        # Convert objectives to numeric, handle missing values
        obj_values = df[objectives].apply(pd.to_numeric, errors='coerce').fillna(np.inf)

        ranks = np.zeros(len(df))
        remaining_indices = set(range(len(df)))
        current_rank = 1

        while remaining_indices:
            # Find current frontier
            current_frontier = []

            for i in remaining_indices:
                is_dominated = False
                for j in remaining_indices:
                    if i == j:
                        continue

                    # Check if j dominates i (strict Pareto: all not worse + at least one strictly better)
                    all_not_worse = all(
                        obj_values.iloc[j][obj] <= obj_values.iloc[i][obj]
                        for obj in objectives
                    )
                    any_strictly_better = any(
                        obj_values.iloc[j][obj] < obj_values.iloc[i][obj]
                        for obj in objectives
                    )

                    if all_not_worse and any_strictly_better:
                        is_dominated = True
                        break

                if not is_dominated:
                    current_frontier.append(i)

            # Assign current rank
            for idx in current_frontier:
                ranks[idx] = current_rank
                remaining_indices.remove(idx)

            current_rank += 1

        return pd.Series(ranks, index=df.index)

    def apply_slo_constraints(self, df: pd.DataFrame, slo_config: Dict = None) -> pd.DataFrame:
        """Apply SLO constraints and calculate margins"""

        if slo_config is None:
            slo_config = {
                'ttft_ms': 1000,
                'tpot_ms': 80,
                'max_power_w': 40,
                'max_temp_c': 80
            }

        # Check SLO compliance
        df['slo_ttft_met'] = df['ttft_ms_median'] <= slo_config['ttft_ms']
        df['slo_tpot_met'] = df['tpot_ms_median'] <= slo_config['tpot_ms']
        df['slo_power_met'] = df['avg_power_w_median'] <= slo_config['max_power_w']

        # Temperature may not be available in all data
        if 'temperature_c_median' in df.columns:
            df['slo_temp_met'] = df['temperature_c_median'] <= slo_config['max_temp_c']
        else:
            df['slo_temp_met'] = True  # Assume compliant if not available

        # Overall SLO compliance
        df['slo_all_met'] = (
            df['slo_ttft_met'] & df['slo_tpot_met'] &
            df['slo_power_met'] & df['slo_temp_met']
        )

        # Calculate SLO margins
        df['ttft_margin'] = slo_config['ttft_ms'] - df['ttft_ms_median']
        df['tpot_margin'] = slo_config['tpot_ms'] - df['tpot_ms_median']
        df['power_margin'] = slo_config['max_power_w'] - df['avg_power_w_median']

        if 'temperature_c_median' in df.columns:
            df['temp_margin'] = slo_config['max_temp_c'] - df['temperature_c_median']
        else:
            df['temp_margin'] = np.nan

        return df

    def build_selector_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build final selector table with all required fields"""

        # Select and rename key fields
        selector_columns = {
            'model': 'model',
            'runtime': 'runtime',
            'batch_size': 'batch_size',
            'prompt_length': 'prompt_length',
            'output_length': 'output_length',
            'phase': 'phase',
            'scenario': 'scenario',  # Keep original scenario if available
            'concurrency': 'concurrency',
            'gpu_freq_mhz': 'gpu_freq_mhz',
            'cpu_freq_mhz': 'cpu_freq_mhz',
            'emc_freq_mhz': 'emc_freq_mhz',
            'ttft_ms_median': 'ttft_ms_median',
            'tpot_ms_median': 'tpot_ms_median',
            'energy_per_token_j_median': 'energy_per_token_j_median',
            'energy_per_input_token_j_median': 'energy_per_input_token_j_median',
            'energy_per_output_token_j_median': 'energy_per_output_token_j_median',
            'energy_per_total_token_j_median': 'energy_per_total_token_j_median',
            'tokens_per_joule_median': 'tokens_per_joule_median',
            'avg_power_w_median': 'avg_power_w_median',
            'max_power_w_median': 'max_power_w_median',
            'temperature_c_median': 'temperature_c_median',
            'slo_ttft_met': 'slo_ttft_met',
            'slo_tpot_met': 'slo_tpot_met',
            'slo_power_met': 'slo_power_met',
            'slo_temp_met': 'slo_temp_met',
            'slo_all_met': 'slo_all_met',
            'ttft_margin': 'ttft_margin',
            'tpot_margin': 'tpot_margin',
            'power_margin': 'power_margin',
            'temp_margin': 'temp_margin',
            'is_pareto': 'is_pareto',
            'pareto_rank': 'pareto_rank',
            'energy_per_token_j_cv': 'energy_per_token_j_cv',
            'pareto_confidence': 'pareto_confidence',
            'bucket_key': 'bucket_key'
        }

        # Select available columns
        available_selector_cols = {k: v for k, v in selector_columns.items() if v in df.columns}
        selector_df = df[list(available_selector_cols.values())].copy()
        selector_df.columns = list(available_selector_cols.keys())

        # Add energy_objective_used column based on phase
        if 'phase' in selector_df.columns:
            selector_df['energy_objective_used'] = selector_df['phase'].map({
                'prefill': 'energy_per_input_token_j_median',
                'decode': 'energy_per_output_token_j_median',
                'mixed': 'energy_per_output_token_j_median'
            }).fillna('energy_per_output_token_j_median')
        else:
            selector_df['energy_objective_used'] = 'energy_per_output_token_j_median'

        # Fill missing phase-specific energy columns with fallback
        for col in ['energy_per_input_token_j_median', 'energy_per_output_token_j_median',
                     'energy_per_total_token_j_median']:
            if col not in selector_df.columns:
                selector_df[col] = selector_df.get('energy_per_token_j_median', 0)
            else:
                selector_df[col] = selector_df[col].fillna(selector_df.get('energy_per_token_j_median', 0))

        # Add measurement CV as overall CV
        if 'energy_per_token_j_cv' in selector_df.columns:
            selector_df['measurement_cv'] = selector_df['energy_per_token_j_cv']
        else:
            selector_df['measurement_cv'] = 0.0

        # Ensure all required columns exist
        required_cols = [
            'model', 'runtime', 'batch_size', 'prompt_length', 'output_length',
            'phase', 'concurrency', 'gpu_freq_mhz', 'cpu_freq_mhz', 'emc_freq_mhz',
            'ttft_ms_median', 'tpot_ms_median', 'energy_per_token_j_median',
            'tokens_per_joule_median', 'avg_power_w_median', 'max_power_w',
            'temperature_c', 'slo_ttft_met', 'slo_tpot_met', 'slo_power_met',
            'slo_temp_met', 'slo_all_met', 'ttft_margin', 'tpot_margin',
            'power_margin', 'temp_margin', 'is_pareto', 'pareto_rank',
            'measurement_cv', 'pareto_confidence'
        ]

        for col in required_cols:
            if col not in selector_df.columns:
                if col == 'max_power_w':
                    selector_df[col] = selector_df.get('max_power_w_median', selector_df.get('avg_power_w_median', 0))
                elif col == 'temperature_c':
                    selector_df[col] = selector_df.get('temperature_c_median', 50.0)
                elif col in ['slo_ttft_met', 'slo_tpot_met', 'slo_power_met', 'slo_temp_met', 'slo_all_met']:
                    selector_df[col] = True
                elif col in ['ttft_margin', 'tpot_margin', 'power_margin', 'temp_margin']:
                    selector_df[col] = 100.0  # Large margin
                elif col in ['is_pareto']:
                    selector_df[col] = True
                elif col in ['pareto_rank']:
                    selector_df[col] = 1
                elif col in ['measurement_cv']:
                    selector_df[col] = 0.0
                elif col in ['pareto_confidence']:
                    selector_df[col] = 'medium'
                else:
                    selector_df[col] = 0

        # Add fallback safe config flag (simple heuristic: mid frequencies)
        selector_df['is_fallback_safe'] = (
            (selector_df['gpu_freq_mhz'] == 846) &
            (selector_df['cpu_freq_mhz'] == 1479) &
            (selector_df['emc_freq_mhz'] == 1600)
        )

        return selector_df

    def generate_quality_report(self, raw_df: pd.DataFrame, final_df: pd.DataFrame) -> Dict:
        """Generate data quality report"""

        # Convert numpy/pandas types to Python native types for JSON serialization
        def convert_to_native(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        missing_values = raw_df.isnull().sum().to_dict()
        missing_values = {k: int(v) for k, v in missing_values.items()}

        data_types = raw_df.dtypes.apply(str).to_dict()

        quality_report = {
            'data_source': str(self.data_dir),
            'generation_time': datetime.now().isoformat(),
            'raw_data': {
                'total_rows': int(len(raw_df)),
                'total_columns': int(len(raw_df.columns)),
                'missing_values': missing_values,
                'data_types': data_types
            },
            'final_selector_table': {
                'total_rows': int(len(final_df)),
                'total_buckets': int(final_df['bucket_key'].nunique()),
                'total_configs': int(len(final_df)),
                'slo_feasible_configs': int(final_df['slo_all_met'].sum()),
                'pareto_configs': int(final_df['is_pareto'].sum()),
                'avg_measurement_cv': float(final_df['measurement_cv'].mean())
            },
            'data_completeness': {
                'has_power_stats': bool('max_power_w' in raw_df.columns or 'max_power_w_median' in final_df.columns),
                'has_temp_stats': bool('temperature_c' in raw_df.columns),
                'has_phase_info': bool('phase' in raw_df.columns or 'scenario' in raw_df.columns),
                'has_frequency_info': bool(all(col in raw_df.columns for col in ['gpu_freq_mhz', 'cpu_freq_mhz', 'emc_freq_mhz']))
            },
            'limitations': [
                'Data from synthetic benchmark, not real Jetson Orin measurements',
                'Power statistics may be limited (synthetic modeling)',
                'CPU/EMC frequency effects may be underestimated',
                'Switching overhead not included in current data',
                'Thermal effects simplified in synthetic data'
            ],
            'recommendations': [
                'Validate with real Jetson Orin + llama.cpp experiments',
                'Expand workload matrix for better coverage',
                'Add comprehensive power and thermal measurements',
                'Measure actual frequency switching overhead',
                'Implement real-time SLO monitoring'
            ]
        }

        return convert_to_native(quality_report)

    def generate_summary_markdown(self, quality_report: Dict) -> str:
        """Generate markdown summary report"""

        lines = []
        lines.append("# Energy Rate Table Build Summary")
        lines.append(f"\n**Generated**: {quality_report['generation_time']}")
        lines.append(f"**Data Source**: {quality_report['data_source']}")

        lines.append("\n## Data Overview")
        lines.append(f"\n**Raw Data**:")
        lines.append(f"- Total Rows: {quality_report['raw_data']['total_rows']}")
        lines.append(f"- Total Columns: {quality_report['raw_data']['total_columns']}")

        lines.append(f"\n**Final Selector Table**:")
        lines.append(f"- Total Rows: {quality_report['final_selector_table']['total_rows']}")
        lines.append(f"- Unique Buckets: {quality_report['final_selector_table']['total_buckets']}")
        lines.append(f"- Total Configurations: {quality_report['final_selector_table']['total_configs']}")
        lines.append(f"- SLO-Feasible Configs: {quality_report['final_selector_table']['slo_feasible_configs']}")
        lines.append(f"- Pareto Configs: {quality_report['final_selector_table']['pareto_configs']}")
        lines.append(f"- Avg Measurement CV: {quality_report['final_selector_table']['avg_measurement_cv']:.2f}%")

        lines.append("\n## Data Completeness")
        completeness = quality_report['data_completeness']
        lines.append(f"- Power Statistics: {'✅' if completeness['has_power_stats'] else '❌'}")
        lines.append(f"- Temperature Statistics: {'✅' if completeness['has_temp_stats'] else '❌'}")
        lines.append(f"- Phase Information: {'✅' if completeness['has_phase_info'] else '❌'}")
        lines.append(f"- Frequency Information: {'✅' if completeness['has_frequency_info'] else '❌'}")

        lines.append("\n## ⚠️ Data Limitations")
        for limitation in quality_report['limitations']:
            lines.append(f"- {limitation}")

        lines.append("\n## Recommendations")
        for recommendation in quality_report['recommendations']:
            lines.append(f"- {recommendation}")

        lines.append("\n## Output Files")
        lines.append("- `profile_raw.parquet`: Raw experimental data with normalized fields")
        lines.append("- `profile_agg_by_config.parquet`: Aggregated data by bucket + config")
        lines.append("- `pareto_by_bucket.parquet`: Pareto frontier analysis")
        lines.append("- `selector_table.parquet`: Final selector table for config selection")
        lines.append("- `rate_table_quality_report.json`: Data quality assessment")
        lines.append("- `rate_table_summary.md`: This summary report")

        return '\n'.join(lines)

    def build_rate_tables(self):
        """Main function to build all rate tables"""

        if self.raw_data.empty:
            logger.error("No raw data available for rate table construction")
            return None

        logger.info("Starting rate table construction...")

        # Step 1: Normalize data fields
        logger.info("Step 1: Normalizing data fields...")
        normalized_df = self.normalize_data_fields(self.raw_data.copy())

        # Step 2: Create workload buckets
        logger.info("Step 2: Creating workload buckets...")
        bucketed_df = self.create_workload_buckets(normalized_df)

        # Step 3: Save raw profile
        logger.info("Step 3: Saving raw profile...")
        raw_profile_path = self.output_dir / 'profile_raw.parquet'
        bucketed_df.to_parquet(raw_profile_path, index=False)
        logger.info(f"Saved raw profile to: {raw_profile_path}")

        # Step 4: Aggregate by configuration
        logger.info("Step 4: Aggregating by configuration...")
        agg_df = self.aggregate_by_config(bucketed_df)

        agg_profile_path = self.output_dir / 'profile_agg_by_config.parquet'
        agg_df.to_parquet(agg_profile_path, index=False)
        logger.info(f"Saved aggregated profile to: {agg_profile_path}")

        # Step 5: Apply SLO constraints
        logger.info("Step 5: Applying SLO constraints...")
        slo_df = self.apply_slo_constraints(agg_df)

        # Step 6: Calculate Pareto frontier
        logger.info("Step 6: Calculating Pareto frontier...")
        pareto_df = self.calculate_pareto_frontier(slo_df)

        pareto_path = self.output_dir / 'pareto_by_bucket.parquet'
        pareto_df.to_parquet(pareto_path, index=False)
        logger.info(f"Saved Pareto analysis to: {pareto_path}")

        # Step 7: Build selector table
        logger.info("Step 7: Building selector table...")
        selector_df = self.build_selector_table(pareto_df)

        selector_path = self.output_dir / 'selector_table.parquet'
        selector_df.to_parquet(selector_path, index=False)
        logger.info(f"Saved selector table to: {selector_path}")

        # Step 8: Generate quality report
        logger.info("Step 8: Generating quality report...")
        quality_report = self.generate_quality_report(bucketed_df, selector_df)

        quality_path = self.output_dir / 'rate_table_quality_report.json'
        with open(quality_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        logger.info(f"Saved quality report to: {quality_path}")

        # Step 9: Generate summary markdown
        logger.info("Step 9: Generating summary report...")
        summary_md = self.generate_summary_markdown(quality_report)

        summary_path = self.output_dir / 'rate_table_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary_md)
        logger.info(f"Saved summary report to: {summary_path}")

        logger.info("✅ Rate table construction completed successfully!")

        return {
            'raw_profile': str(raw_profile_path),
            'agg_profile': str(agg_profile_path),
            'pareto_analysis': str(pareto_path),
            'selector_table': str(selector_path),
            'quality_report': str(quality_path),
            'summary': str(summary_path)
        }


def main():
    """Main function to build rate tables"""
    builder = RateTableBuilder()
    results = builder.build_rate_tables()

    if results:
        print("\n" + "="*80)
        print("ENERGY RATE TABLE CONSTRUCTION COMPLETED")
        print("="*80)

        print(f"\n📊 Generated Files:")
        for name, path in results.items():
            print(f"  ✅ {name}: {path}")

        print(f"\n📁 Output Directory: {builder.output_dir}/")
        return 0
    else:
        print("\n❌ Rate table construction failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())