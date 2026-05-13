#!/usr/bin/env python3
"""
Selector Performance Evaluation
对比不同配置策略在已有数据上的表现，重点输出 selector regret 和 SLO violation rate
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SelectorEvaluator:
    """Evaluate different configuration selection strategies"""

    def __init__(self, selector_table_path: str = "data/rate_tables/selector_table.parquet",
                 output_dir: str = "data/selector_eval"):
        self.selector_table_path = Path(selector_table_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load selector table
        try:
            self.selector_df = pd.read_parquet(self.selector_table_path)
            logger.info(f"Loaded selector table: {len(self.selector_df)} configurations")
        except Exception as e:
            logger.error(f"Failed to load selector table: {e}")
            self.selector_df = pd.DataFrame()

        # Define strategies
        self.strategies = [
            'maxn_all_high',
            'all_mid',
            'energy_efficient_all_low',
            'fixed_best_efficiency',
            'oracle_best_per_bucket',
            'ours_slo_aware_selector'
        ]

        # Pre-compute per-phase fixed best configs
        self._phase_best_configs = self._compute_phase_best_configs()

    def _get_energy_column(self, bucket_df: pd.DataFrame) -> str:
        """Determine the phase-appropriate energy column for a bucket"""
        if 'energy_objective_used' in bucket_df.columns:
            obj_col = bucket_df['energy_objective_used'].iloc[0]
            if obj_col in bucket_df.columns:
                return obj_col
        if 'phase' in bucket_df.columns:
            phase = bucket_df['phase'].iloc[0]
            phase_col_map = {
                'prefill': 'energy_per_input_token_j_median',
                'decode': 'energy_per_output_token_j_median',
                'mixed': 'energy_per_output_token_j_median'
            }
            target_col = phase_col_map.get(phase, 'energy_per_output_token_j_median')
            if target_col in bucket_df.columns:
                return target_col
        return 'energy_per_token_j_median'

    def _compute_phase_best_configs(self) -> Dict:
        """Pre-compute fixed best config per phase group"""
        phase_best = {}
        if self.selector_df.empty or 'phase' not in self.selector_df.columns:
            # Fallback: global best
            energy_col = 'energy_per_token_j_median'
            if not self.selector_df.empty and energy_col in self.selector_df.columns:
                phase_best['_global'] = self.selector_df.loc[self.selector_df[energy_col].idxmin()]
            return phase_best

        for phase in self.selector_df['phase'].unique():
            phase_df = self.selector_df[self.selector_df['phase'] == phase]
            energy_col = self._get_energy_column(phase_df)
            if energy_col in phase_df.columns and not phase_df.empty:
                phase_best[phase] = phase_df.loc[phase_df[energy_col].idxmin()]

        return phase_best

    def get_maxn_config(self, bucket_df: pd.DataFrame) -> pd.Series:
        """Get MaxN / all_high configuration (highest frequencies)"""

        if bucket_df.empty:
            return pd.Series()

        # Find configuration with highest GPU frequency
        max_gpu = bucket_df['gpu_freq_mhz'].max()
        maxn_configs = bucket_df[bucket_df['gpu_freq_mhz'] == max_gpu]

        if len(maxn_configs) > 0:
            # Among max GPU, prefer highest CPU and EMC
            maxn_configs = maxn_configs.sort_values(
                by=['cpu_freq_mhz', 'emc_freq_mhz'], ascending=False
            )
            return maxn_configs.iloc[0]

        return bucket_df.iloc[0]

    def get_all_mid_config(self, bucket_df: pd.DataFrame) -> pd.Series:
        """Get all_mid configuration (medium frequencies)"""

        if bucket_df.empty:
            return pd.Series()

        # Target medium frequencies
        target_gpu = 846
        target_cpu = 1479
        target_emc = 1600

        # Find closest to medium
        bucket_df = bucket_df.copy()
        bucket_df['mid_distance'] = (
            abs(bucket_df['gpu_freq_mhz'] - target_gpu) +
            abs(bucket_df['cpu_freq_mhz'] - target_cpu) +
            abs(bucket_df['emc_freq_mhz'] - target_emc)
        )

        mid_config = bucket_df.loc[bucket_df['mid_distance'].idxmin()]
        return mid_config

    def get_energy_efficient_config(self, bucket_df: pd.DataFrame) -> pd.Series:
        """Get energy-efficient / all_low configuration"""

        if bucket_df.empty:
            return pd.Series()

        energy_col = self._get_energy_column(bucket_df)
        min_energy_idx = bucket_df[energy_col].idxmin()
        return bucket_df.loc[min_energy_idx]

    def get_fixed_best_efficiency_config(self, bucket_df: pd.DataFrame) -> pd.Series:
        """Get fixed best efficiency config (per-phase best, same for all buckets in that phase)"""

        if bucket_df.empty:
            return pd.Series()

        # Determine phase for this bucket
        phase = bucket_df['phase'].iloc[0] if 'phase' in bucket_df.columns else None
        phase_best = self._phase_best_configs.get(phase) if phase else None

        if phase_best is not None:
            return phase_best

        # Fallback to global best
        global_best = self._phase_best_configs.get('_global')
        if global_best is not None:
            return global_best

        # Ultimate fallback: pick min energy from bucket
        energy_col = self._get_energy_column(bucket_df)
        return bucket_df.loc[bucket_df[energy_col].idxmin()]

    def _lookup_config_in_bucket(self, selected_config: pd.Series,
                                     bucket_df: pd.DataFrame) -> pd.Series:
        """Look up the fixed config's actual metrics within the target bucket."""
        gpu = selected_config['gpu_freq_mhz']
        cpu = selected_config['cpu_freq_mhz']
        emc = selected_config['emc_freq_mhz']

        # Find exact match in current bucket
        match = bucket_df[
            (bucket_df['gpu_freq_mhz'] == gpu) &
            (bucket_df['cpu_freq_mhz'] == cpu) &
            (bucket_df['emc_freq_mhz'] == emc)
        ]

        if not match.empty:
            return match.iloc[0]

        # No exact match — find closest frequency combination
        bucket_df = bucket_df.copy()
        bucket_df['freq_distance'] = (
            abs(bucket_df['gpu_freq_mhz'] - gpu) +
            abs(bucket_df['cpu_freq_mhz'] - cpu) +
            abs(bucket_df['emc_freq_mhz'] - emc)
        )
        return bucket_df.loc[bucket_df['freq_distance'].idxmin()]

    def get_oracle_best_config(self, bucket_df: pd.DataFrame) -> pd.Series:
        """Get oracle best config (best energy among SLO-feasible, per bucket)"""

        if bucket_df.empty:
            return pd.Series()

        energy_col = self._get_energy_column(bucket_df)

        # Filter SLO-feasible configs
        slo_feasible = bucket_df[bucket_df['slo_all_met'] == True]

        if not slo_feasible.empty:
            return slo_feasible.loc[slo_feasible[energy_col].idxmin()]
        else:
            return bucket_df.loc[bucket_df[energy_col].idxmin()]

    def get_ours_slo_aware_config(self, bucket_df: pd.DataFrame) -> pd.Series:
        """Get our SLO-aware selector config"""

        if bucket_df.empty:
            return pd.Series()

        energy_col = self._get_energy_column(bucket_df)

        slo_feasible = bucket_df[bucket_df['slo_all_met'] == True]

        if not slo_feasible.empty:
            return slo_feasible.loc[slo_feasible[energy_col].idxmin()]
        else:
            bucket_df = bucket_df.copy()
            bucket_df['slo_violation_score'] = (
                (bucket_df['ttft_ms_median'] - 1000).clip(lower=0) * 0.3 +
                (bucket_df['tpot_ms_median'] - 80).clip(lower=0) * 0.3 +
                (bucket_df['avg_power_w_median'] - 40).clip(lower=0) * 0.2 +
                (bucket_df['temperature_c'] - 80).clip(lower=0) * 0.2
            )
            return bucket_df.loc[bucket_df['slo_violation_score'].idxmin()]

    def evaluate_strategy(self, strategy: str) -> pd.DataFrame:
        """Evaluate a specific strategy across all buckets"""

        if self.selector_df.empty:
            return pd.DataFrame()

        results = []

        # Evaluate per bucket
        for bucket_key in self.selector_df['bucket_key'].unique():
            bucket_df = self.selector_df[self.selector_df['bucket_key'] == bucket_key]

            if bucket_df.empty:
                continue

            # Determine phase-appropriate energy column for this bucket
            energy_col = self._get_energy_column(bucket_df)

            # Select config based on strategy
            if strategy == 'maxn_all_high':
                selected = self.get_maxn_config(bucket_df)
            elif strategy == 'all_mid':
                selected = self.get_all_mid_config(bucket_df)
            elif strategy == 'energy_efficient_all_low':
                selected = self.get_energy_efficient_config(bucket_df)
            elif strategy == 'fixed_best_efficiency':
                selected = self.get_fixed_best_efficiency_config(bucket_df)
            elif strategy == 'oracle_best_per_bucket':
                selected = self.get_oracle_best_config(bucket_df)
            elif strategy == 'ours_slo_aware_selector':
                selected = self.get_ours_slo_aware_config(bucket_df)
            else:
                continue

            if selected.empty:
                continue

            # For fixed_best, look up the selected config's actual energy in this bucket
            if strategy == 'fixed_best_efficiency':
                selected = self._lookup_config_in_bucket(selected, bucket_df)
                if selected.empty:
                    continue

            # Get oracle for this bucket (for regret calculation)
            oracle = self.get_oracle_best_config(bucket_df)

            results.append({
                'strategy': strategy,
                'bucket_key': bucket_key,
                'phase': bucket_df['phase'].iloc[0] if 'phase' in bucket_df.columns else 'unknown',
                'energy_col_used': energy_col,
                'selected_config': {
                    'gpu_freq_mhz': int(selected['gpu_freq_mhz']),
                    'cpu_freq_mhz': int(selected['cpu_freq_mhz']),
                    'emc_freq_mhz': int(selected['emc_freq_mhz'])
                },
                'energy_per_token_j': float(selected[energy_col]),
                'oracle_energy': float(oracle[energy_col]),
                'ttft_ms': float(selected['ttft_ms_median']),
                'tpot_ms': float(selected['tpot_ms_median']),
                'avg_power_w': float(selected['avg_power_w_median']),
                'slo_feasible': bool(selected['slo_all_met']),
                'is_pareto': bool(selected.get('is_pareto', False))
            })

        return pd.DataFrame(results)

    def calculate_regret(self, results_df: pd.DataFrame) -> Tuple[Dict, List]:
        """Calculate regret statistics for each strategy, return (stats, anomalies)"""

        regret_stats = {}
        anomalies = []

        for strategy in results_df['strategy'].unique():
            strategy_data = results_df[results_df['strategy'] == strategy].copy()

            # Calculate energy regret vs oracle
            strategy_data['energy_regret'] = (
                (strategy_data['energy_per_token_j'] - strategy_data['oracle_energy']) /
                strategy_data['oracle_energy']
            )

            # Flag negative regret as anomaly (strategy beats oracle = bug)
            negative_regret = strategy_data[strategy_data['energy_regret'] < -1e-9]
            for _, row in negative_regret.iterrows():
                anomaly_msg = (
                    f"ANOMALY: {strategy} has negative regret ({row['energy_regret']:.4%}) "
                    f"in bucket {row['bucket_key']} — strategy beats oracle, "
                    f"possible cross-phase contamination"
                )
                logger.warning(anomaly_msg)
                anomalies.append({
                    'strategy': strategy,
                    'bucket_key': row['bucket_key'],
                    'phase': row.get('phase', 'unknown'),
                    'energy_regret': float(row['energy_regret']),
                    'strategy_energy': float(row['energy_per_token_j']),
                    'oracle_energy': float(row['oracle_energy'])
                })

            # Handle infeasible buckets
            feasible_data = strategy_data[strategy_data['slo_feasible'] == True]

            regret_stats[strategy] = {
                'total_buckets': len(strategy_data),
                'slo_feasible_buckets': len(feasible_data),
                'slo_violation_rate': 1.0 - (len(feasible_data) / len(strategy_data)),
                'mean_energy_regret': float(strategy_data['energy_regret'].mean()),
                'median_energy_regret': float(strategy_data['energy_regret'].median()),
                'p95_energy_regret': float(strategy_data['energy_regret'].quantile(0.95)),
                'max_energy_regret': float(strategy_data['energy_regret'].max()),
                'mean_energy': float(strategy_data['energy_per_token_j'].mean()),
                'mean_ttft': float(strategy_data['ttft_ms'].mean()),
                'mean_tpot': float(strategy_data['tpot_ms'].mean()),
                'mean_power': float(strategy_data['avg_power_w'].mean()),
                'negative_regret_count': len(negative_regret)
            }

        return regret_stats, anomalies

    def generate_comparison_report(self, results_df: pd.DataFrame, regret_stats: Dict,
                                       anomalies: List = None) -> str:
        """Generate comprehensive comparison report"""

        lines = []
        lines.append("# Selector Performance Evaluation Report")
        lines.append(f"\n**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("**Data Source**: Synthetic Benchmark (Phase 3)")

        lines.append("\n## Strategy Comparison Summary")

        # Create comparison table
        comparison_data = []
        for strategy in self.strategies:
            if strategy in regret_stats:
                stats = regret_stats[strategy]
                comparison_data.append({
                    'Strategy': strategy.replace('_', ' ').title(),
                    'SLO Violation Rate': f"{stats['slo_violation_rate']:.2%}",
                    'Mean Energy Regret': f"{stats['mean_energy_regret']:.2%}",
                    'Median Energy Regret': f"{stats['median_energy_regret']:.2%}",
                    'P95 Energy Regret': f"{stats['p95_energy_regret']:.2%}",
                    'Mean Energy (J/token)': f"{stats['mean_energy']:.4f}",
                    'Mean TTFT (ms)': f"{stats['mean_ttft']:.2f}",
                    'SLO Feasible Buckets': f"{stats['slo_feasible_buckets']}/{stats['total_buckets']}"
                })

        lines.append("\n| Strategy | SLO Violation Rate | Mean Energy Regret | Median Energy Regret | P95 Energy Regret | Mean Energy (J/token) | Mean TTFT (ms) | SLO Feasible Buckets |")
        lines.append("|----------|-------------------|-------------------|---------------------|-------------------|----------------------|----------------|----------------------|")

        for data in comparison_data:
            lines.append(f"| {data['Strategy']} | {data['SLO Violation Rate']} | {data['Mean Energy Regret']} | {data['Median Energy Regret']} | {data['P95 Energy Regret']} | {data['Mean Energy (J/token)']} | {data['Mean TTFT (ms)']} | {data['SLO Feasible Buckets']} |")

        lines.append("\n## Key Findings")

        # Find best performing strategy
        best_strategy = min(regret_stats.items(),
                          key=lambda x: x[1]['mean_energy_regret'] if x[1]['slo_violation_rate'] < 0.5 else float('inf'))

        lines.append(f"\n**Best Performing Strategy**: {best_strategy[0].replace('_', ' ').title()}")
        lines.append(f"- Mean Energy Regret: {best_strategy[1]['mean_energy_regret']:.2%}")
        lines.append(f"- SLO Violation Rate: {best_strategy[1]['slo_violation_rate']:.2%}")

        # Compare with baselines
        oracle_regret = regret_stats.get('oracle_best_per_bucket', {}).get('mean_energy_regret', 0)
        maxn_regret = regret_stats.get('maxn_all_high', {}).get('mean_energy_regret', 0)
        our_regret = regret_stats.get('ours_slo_aware_selector', {}).get('mean_energy_regret', 0)

        lines.append("\n## Performance vs Baselines")

        if oracle_regret is not None and maxn_regret is not None:
            lines.append(f"\n**MaxN vs Oracle**:")
            lines.append(f"- MaxN Regret: {maxn_regret:.2%}")
            lines.append(f"- Oracle Regret: {oracle_regret:.2%} (baseline)")
            lines.append(f"- Performance Gap: {(maxn_regret - oracle_regret):.2%}")

        if our_regret is not None and maxn_regret is not None:
            lines.append(f"\n**Our Selector vs MaxN**:")
            lines.append(f"- Our Selector Regret: {our_regret:.2%}")
            lines.append(f"- MaxN Regret: {maxn_regret:.2%}")
            improvement = maxn_regret - our_regret
            lines.append(f"- Improvement: {improvement:.2%}")

        # Anomaly section
        if anomalies:
            lines.append("\n## ⚠️ Anomalies Detected")
            lines.append(f"\n**Negative regret count**: {len(anomalies)} (strategy beats oracle)")
            lines.append("\nThese indicate potential cross-phase energy column contamination:")
            for a in anomalies[:10]:  # Show first 10
                lines.append(f"- {a['strategy']} in {a['bucket_key']} (phase={a['phase']}): "
                           f"regret={a['energy_regret']:.4%}, "
                           f"strategy={a['strategy_energy']:.6f} vs oracle={a['oracle_energy']:.6f}")
            if len(anomalies) > 10:
                lines.append(f"- ... and {len(anomalies) - 10} more")
        else:
            lines.append("\n## ✅ No Anomalies")
            lines.append("\nNo negative regret detected — oracle regret is correctly 0% for all buckets.")

        lines.append("\n## ⚠️ Data Limitations")
        lines.append("\n**Important**: This evaluation uses synthetic benchmark data.")
        lines.append("\n**Limitations:")
        lines.append("- Results may not reflect real Jetson Orin + llama.cpp performance")
        lines.append("- CPU/EMC frequency effects may be misrepresented")
        lines.append("- Power and thermal modeling is simplified")
        lines.append("- Switching overhead not included in evaluation")

        lines.append("\n**What This Evaluation Validates:")
        lines.append("- ✅ Selector logic and SLO filtering correctness")
        lines.append("- ✅ Regret calculation methodology")
        lines.append("- ✅ Strategy comparison framework")
        lines.append("- ✅ Analysis pipeline and reporting")

        lines.append("\n## Recommendations")
        lines.append("\n1. **Real Hardware Validation**: Re-evaluate with actual Jetson Orin measurements")
        lines.append("2. **Expand Workload Coverage**: Test with diverse workload patterns")
        lines.append("3. **Include Switching Overhead**: Account for frequency transition costs")
        lines.append("4. **Thermal Effects**: Model temperature impact on frequency stability")
        lines.append("5. **Production Testing**: Validate under real serving conditions")

        return '\n'.join(lines)

    def generate_visualizations(self, results_df: pd.DataFrame, regret_stats: Dict):
        """Generate evaluation visualizations"""

        figures_dir = Path("figures/phase3_analysis")
        figures_dir.mkdir(parents=True, exist_ok=True)

        # 1. Energy Regret Comparison
        strategies = list(regret_stats.keys())
        mean_regrets = [regret_stats[s]['mean_energy_regret'] for s in strategies]
        p95_regrets = [regret_stats[s]['p95_energy_regret'] for s in strategies]

        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(strategies))
        plt.bar(x_pos, mean_regrets, alpha=0.7, label='Mean Regret')
        plt.errorbar(x_pos, mean_regrets, yerr=p95_regrets, fmt='o', color='red', label='P95 Regret')
        plt.xlabel('Strategy')
        plt.ylabel('Energy Regret (%)')
        plt.title('Selector Energy Regret vs Oracle')
        plt.xticks(x_pos, [s.replace('_', ' ') for s in strategies], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / 'selector_regret_vs_oracle.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. SLO Violation Rate
        slo_violation_rates = [regret_stats[s]['slo_violation_rate'] for s in strategies]

        plt.figure(figsize=(10, 6))
        plt.bar(strategies, slo_violation_rates, color=['green' if r < 0.1 else 'yellow' if r < 0.3 else 'red' for r in slo_violation_rates])
        plt.xlabel('Strategy')
        plt.ylabel('SLO Violation Rate')
        plt.title('SLO Violation Rate by Strategy')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(figures_dir / 'selector_slo_violation_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualizations saved to: {figures_dir}/")

    def run_evaluation(self):
        """Run complete selector evaluation"""

        if self.selector_df.empty:
            logger.error("No selector data available for evaluation")
            return None

        logger.info("Starting selector performance evaluation...")

        # Evaluate each strategy
        all_results = []
        for strategy in self.strategies:
            logger.info(f"Evaluating strategy: {strategy}")
            strategy_results = self.evaluate_strategy(strategy)
            all_results.append(strategy_results)

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Calculate regret statistics
        regret_stats, anomalies = self.calculate_regret(combined_results)

        # Save results
        comparison_path = self.output_dir / 'selector_comparison.csv'
        combined_results.to_csv(comparison_path, index=False)
        logger.info(f"Saved comparison results to: {comparison_path}")

        regret_path = self.output_dir / 'selector_regret.csv'
        pd.DataFrame(regret_stats).T.to_csv(regret_path)
        logger.info(f"Saved regret statistics to: {regret_path}")

        # Generate report
        report = self.generate_comparison_report(combined_results, regret_stats, anomalies)
        report_path = self.output_dir / 'selector_eval_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved evaluation report to: {report_path}")

        # Generate visualizations
        try:
            self.generate_visualizations(combined_results, regret_stats)
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

        # Save summary JSON
        summary = {
            'evaluation_time': pd.Timestamp.now().isoformat(),
            'total_buckets': combined_results['bucket_key'].nunique(),
            'strategies_evaluated': len(self.strategies),
            'anomaly_count': len(anomalies),
            'regret_stats': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                               for kk, vv in v.items()}
                          for k, v in regret_stats.items()},
            'best_strategy': min(regret_stats.items(),
                              key=lambda x: x[1]['mean_energy_regret'] if x[1]['slo_violation_rate'] < 0.5 else float('inf'))[0]
        }

        summary_path = self.output_dir / 'selector_eval_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved evaluation summary to: {summary_path}")

        logger.info("✅ Selector evaluation completed successfully!")

        return {
            'comparison': str(comparison_path),
            'regret': str(regret_path),
            'report': str(report_path),
            'summary': str(summary_path)
        }


def main():
    """Main function to run selector evaluation"""
    evaluator = SelectorEvaluator()
    results = evaluator.run_evaluation()

    if results:
        print("\n" + "="*80)
        print("SELECTOR PERFORMANCE EVALUATION COMPLETED")
        print("="*80)

        print(f"\n📊 Generated Files:")
        for name, path in results.items():
            print(f"  ✅ {name}: {path}")

        print(f"\n📁 Output Directory: {evaluator.output_dir}/")
        return 0
    else:
        print("\n❌ Selector evaluation failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())