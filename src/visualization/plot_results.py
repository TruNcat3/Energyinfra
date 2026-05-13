#!/usr/bin/env python3
"""
Visualization Module for Energy Profiling Experiments
Generates plots and charts for experiment analysis.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualizationGenerator:
    """
    Generator for visualizing energy profiling experiment results.

    Creates:
    - Heatmaps for frequency sweeps
    - Energy-latency tradeoff curves
    - Tokens/J comparison plots
    - Pareto frontier plots
    - SLO satisfaction analysis
    - Ablation study visualizations
    """

    def __init__(self, output_dir: str = "figures"):
        """
        Initialize visualization generator.

        Args:
            output_dir: Directory for generated plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for different plot types
        self.heatmap_dir = self.output_dir / "heatmaps"
        self.pareto_dir = self.output_dir / "pareto_curves"
        self.tradeoff_dir = self.output_dir / "tradeoffs"
        self.ablation_dir = self.output_dir / "ablations"

        for dir_path in [self.heatmap_dir, self.pareto_dir, self.tradeoff_dir, self.ablation_dir]:
            dir_path.mkdir(exist_ok=True)

        # Plotting settings
        self.fig_dpi = 150
        self.fig_size = (10, 6)
        self.font_size = 10

    def generate_all_visualizations(self, results_df: pd.DataFrame,
                               experiment_name: str = "experiment") -> Dict[str, str]:
        """
        Generate all visualization types for experiment results.

        Args:
            results_df: DataFrame with experiment results
            experiment_name: Name of experiment for file naming

        Returns:
            Dictionary mapping visualization types to file paths
        """
        logger.info(f"Generating all visualizations for: {experiment_name}")
        logger.info(f"Input data shape: {results_df.shape}")

        generated_plots = {}

        try:
            # 1. Heatmaps
            heatmaps = self.generate_heatmaps(results_df, experiment_name)
            generated_plots.update(heatmaps)

            # 2. Pareto frontier
            pareto_plots = self.generate_pareto_plots(results_df, experiment_name)
            generated_plots.update(pareto_plots)

            # 3. Tradeoff curves
            tradeoff_plots = self.generate_tradeoff_curves(results_df, experiment_name)
            generated_plots.update(tradeoff_plots)

            # 4. Performance comparison
            comparison_plots = self.generate_comparison_plots(results_df, experiment_name)
            generated_plots.update(comparison_plots)

            # 5. Energy analysis
            energy_plots = self.generate_energy_plots(results_df, experiment_name)
            generated_plots.update(energy_plots)

            # 6. SLO analysis
            slo_plots = self.generate_slo_plots(results_df, experiment_name)
            generated_plots.update(slo_plots)

            logger.info(f"Generated {len(generated_plots)} visualization files")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

        return generated_plots

    def generate_heatmaps(self, df: pd.DataFrame, experiment_name: str) -> Dict[str, str]:
        """
        Generate heatmaps for frequency sweeps.

        Args:
            df: DataFrame with experiment results
            experiment_name: Name of experiment

        Returns:
            Dictionary mapping heatmap types to file paths
        """
        heatmaps = {}

        # Filter data for heatmap generation
        df_plot = df.copy()

        # 1. GPU frequency sweep heatmap (TTFT vs GPU freq)
        if 'gpu_freq' in df_plot.columns and 'ttft_ms_mean' in df_plot.columns:
            fig, ax = plt.subplots(figsize=self.fig_size)
            gpu_data = df_plot.groupby('gpu_freq')['ttft_ms_mean'].mean()
            gpu_data_sorted = gpu_data.sort_index()

            im = ax.imshow(gpu_data_sorted.values.reshape(1, -1), cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(len(gpu_data_sorted)))
            ax.set_xticklabels([f'{f}MHz' for f in gpu_data_sorted.index])
            ax.set_yticks([0])
            ax.set_yticklabels(['TTFT (ms)'])
            plt.setpcolor(im, edgecolors='k', linewidth=1)
            plt.colorbar(im, ax=ax, label='TTFT (ms)')
            ax.set_title('GPU Frequency Impact on TTFT')
            ax.set_xlabel('GPU Frequency (MHz)')

            file_path = self.heatmap_dir / f"{experiment_name}_gpu_freq_ttft_heatmap.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            heatmaps['gpu_freq_ttft_heatmap'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 2. CPU frequency sweep heatmap
        if 'cpu_freq' in df_plot.columns and 'ttft_ms_mean' in df_plot.columns:
            fig, ax = plt.subplots(figsize=self.fig_size)
            cpu_data = df_plot.groupby('cpu_freq')['ttft_ms_mean'].mean()
            cpu_data_sorted = cpu_data.sort_index()

            im = ax.imshow(cpu_data_sorted.values.reshape(1, -1), cmap='YlGnBu', aspect='auto')
            ax.set_xticks(range(len(cpu_data_sorted)))
            ax.set_xticklabels([f'{f}MHz' for f in cpu_data_sorted.index])
            ax.set_yticks([0])
            ax.set_yticklabels(['TTFT (ms)'])
            plt.setpcolor(im, edgecolors='k', linewidth=1)
            plt.colorbar(im, ax=ax, label='TTFT (ms)')
            ax.set_title('CPU Frequency Impact on TTFT')
            ax.set_xlabel('CPU Frequency (MHz)')

            file_path = self.heatmap_dir / f"{experiment_name}_cpu_freq_ttft_heatmap.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            heatmaps['cpu_freq_ttft_heatmap'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 3. EMC frequency sweep heatmap (TPOT vs EMC freq)
        if 'emc_freq' in df_plot.columns and 'tpot_ms_mean' in df_plot.columns:
            fig, ax = plt.subplots(figsize=self.fig_size)
            emc_data = df_plot.groupby('emc_freq')['tpot_ms_mean'].mean()
            emc_data_sorted = emc_data.sort_index()

            im = ax.imshow(emc_data_sorted.values.reshape(1, -1), cmap='YlOrBr', aspect='auto')
            ax.set_xticks(range(len(emc_data_sorted)))
            ax.set_xticklabels([f'{f}MHz' for f in emc_data_sorted.index])
            ax.set_yticks([0])
            ax.set_yticklabels(['TPOT (ms)'])
            plt.setpcolor(im, edgecolors='k', linewidth=1)
            plt.colorbar(im, ax=ax, label='TPOT (ms)')
            ax.set_title('EMC Frequency Impact on TPOT')
            ax.set_xlabel('EMC Frequency (MHz)')

            file_path = self.heatmap_dir / f"{experiment_name}_emc_freq_tpot_heatmap.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            heatmaps['emc_freq_tpot_heatmap'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 4. Combined frequency heatmap (Energy/Token vs frequencies)
        if all(col in df_plot.columns for col in ['gpu_freq', 'cpu_freq', 'emc_freq', 'energy_per_token_j']):
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create pivot table
            freq_data = df_plot.groupby(['gpu_freq', 'cpu_freq', 'emc_freq'])['energy_per_token_j'].mean()
            freq_data = freq_data.reset_index()

            # Use GPU freq for x-axis, CPU freq for y-axis
            pivot_data = df_plot.pivot_table(
                values='energy_per_token_j',
                index='gpu_freq',
                columns='cpu_freq',
                aggfunc='mean'
            )

            # Create heatmap
            sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt='.3f',
                       cbar_kws={'label': 'Energy/Token (J)'}, ax=ax)
            ax.set_title('Energy per Token (J) - GPU vs CPU Frequency')
            ax.set_xlabel('CPU Frequency (MHz)')
            ax.set_ylabel('GPU Frequency (MHz)')

            file_path = self.heatmap_dir / f"{experiment_name}_energy_freq_heatmap.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            heatmaps['energy_freq_heatmap'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        return heatmaps

    def generate_pareto_plots(self, df: pd.DataFrame, experiment_name: str) -> Dict[str, str]:
        """
        Generate Pareto frontier plots.

        Args:
            df: DataFrame with experiment results
            experiment_name: Name of experiment

        Returns:
            Dictionary mapping Pareto plot types to file paths
        """
        pareto_plots = {}

        # Check if Pareto data exists
        if 'is_pareto' not in df.columns:
            logger.warning("No Pareto frontier data found")
            return pareto_plots

        df_plot = df.copy()

        # 1. Energy vs Latency Pareto frontier
        if all(col in df_plot.columns for col in ['energy_per_token_j', 'ttft_ms_mean']):
            fig, ax = plt.subplots(figsize=self.fig_size)

            # Separate Pareto and non-Pareto points
            pareto_df = df_plot[df_plot['is_pareto']]
            non_pareto_df = df_plot[~df_plot['is_pareto']]

            # Plot all points
            ax.scatter(non_pareto_df['energy_per_token_j'], non_pareto_df['ttft_ms_mean'],
                      c='gray', alpha=0.5, s=50, label='Dominated')
            ax.scatter(pareto_df['energy_per_token_j'], pareto_df['ttft_ms_mean'],
                      c='red', s=100, edgecolors='black', linewidths=2, label='Pareto Frontier')

            # Add labels for Pareto points
            for idx, row in pareto_df.iterrows():
                ax.annotate(f'{row["energy_per_token_j"]:.2f}J, {row["ttft_ms_mean"]:.1f}ms',
                          (row['energy_per_token_j'], row['ttft_ms_mean']),
                          fontsize=8, alpha=0.7)

            ax.set_xlabel('Energy per Token (J)')
            ax.set_ylabel('TTFT (ms)')
            ax.set_title('Pareto Frontier: Energy vs Latency')
            ax.legend()
            ax.grid(True, alpha=0.3)

            file_path = self.pareto_dir / f"{experiment_name}_pareto_energy_latency.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            pareto_plots['pareto_energy_latency'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 2. Tokens/J vs Throughput Pareto frontier
        if all(col in df_plot.columns for col in ['tokens_per_joule', 'throughput_mean']):
            fig, ax = plt.subplots(figsize=self.fig_size)

            pareto_df = df_plot[df_plot['is_pareto']]
            non_pareto_df = df_plot[~df_plot['is_pareto']]

            ax.scatter(non_pareto_df['throughput_mean'], non_pareto_df['tokens_per_joule'],
                      c='gray', alpha=0.5, s=50, label='Dominated')
            ax.scatter(pareto_df['throughput_mean'], pareto_df['tokens_per_joule'],
                      c='green', s=100, edgecolors='black', linewidths=2, label='Pareto Frontier')

            ax.set_xlabel('Throughput (tokens/s)')
            ax.set_ylabel('Tokens per Joule')
            ax.set_title('Pareto Frontier: Throughput vs Efficiency')
            ax.legend()
            ax.grid(True, alpha=0.3)

            file_path = self.pareto_dir / f"{experiment_name}_pareto_throughput_efficiency.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            pareto_plots['pareto_throughput_efficiency'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 3. Multi-objective Pareto radar plot
        if all(col in df_plot.columns for col in ['tokens_per_joule', 'ttft_ms_mean', 'tpot_ms_mean', 'is_pareto']):
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

            # Normalize objectives (higher is better)
            pareto_df = df_plot[df_plot['is_pareto']].copy()

            objectives = {
                'Efficiency': 'tokens_per_joule',
                'TTFT': lambda x: 1000 / x,  # Lower is better
                'TPOT': lambda x: 100 / x,   # Lower is better
                'Power': lambda x: 100 / x    # Lower is better
            }

            # Prepare data for radar chart
            categories = ['Efficiency', 'TTFT', 'TPOT', 'Power']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]

            for i, cat in enumerate(categories):
                if cat == 'Efficiency':
                    values = pareto_df['tokens_per_joule'].values
                elif cat == 'TTFT':
                    values = 1000 / pareto_df['ttft_ms_mean'].values
                elif cat == 'TPOT':
                    values = 100 / pareto_df['tpot_ms_mean'].values
                else:  # Power
                    values = 100 / pareto_df['avg_power_w'].values if 'avg_power_w' in pareto_df.columns else np.ones_like(pareto_df['tokens_per_joule'].values)

            # Plot each Pareto point
            for idx, row in pareto_df.iterrows():
                values_row = [row['tokens_per_joule'], 1000/row['ttft_ms_mean'], 1000/row['tpot_ms_mean'], 1000/row['avg_power_w']]
                ax.plot(angles, values_row + [values_row[0]], 'o-', linewidth=2)
                ax.fill(angles, values_row + [values_row[0]], alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_title(f'Pareto Frontier Analysis ({len(pareto_df)} Configurations)')
            ax.grid(True)

            file_path = self.pareto_dir / f"{experiment_name}_pareto_radar.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            pareto_plots['pareto_radar'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        return pareto_plots

    def generate_tradeoff_curves(self, df: pd.DataFrame, experiment_name: str) -> Dict[str, str]:
        """
        Generate energy-latency tradeoff curves.

        Args:
            df: DataFrame with experiment results
            experiment_name: Name of experiment

        Returns:
            Dictionary mapping tradeoff plot types to file paths
        """
        tradeoff_plots = {}

        df_plot = df.copy()

        # 1. Energy vs TTFT tradeoff
        if all(col in df_plot.columns for col in ['energy_per_token_j', 'ttft_ms_mean']):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Sort by energy
            df_sorted = df_plot.sort_values('energy_per_token_j')

            # Left plot: scatter
            scatter = ax1.scatter(df_sorted['energy_per_token_j'], df_sorted['ttft_ms_mean'],
                              c=df_sorted.get('gpu_freq', range(len(df_sorted))),
                              cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidths=1)
            ax1.set_xlabel('Energy per Token (J)')
            ax1.set_ylabel('TTFT (ms)')
            ax1.set_title('Energy vs Latency Tradeoff')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='GPU Frequency (MHz)')

            # Right plot: line with moving average
            window = max(3, len(df_sorted) // 10)
            df_sorted['energy_ma'] = df_sorted['energy_per_token_j'].rolling(window=window, center=True).mean()
            df_sorted['ttft_ma'] = df_sorted['ttft_ms_mean'].rolling(window=window, center=True).mean()

            ax2.plot(df_sorted['energy_per_token_j'], df_sorted['ttft_ms_mean'],
                    'o-', alpha=0.5, markersize=4, label='Individual Points')
            ax2.plot(df_sorted['energy_ma'], df_sorted['ttft_ma'],
                    '-', linewidth=2, markersize=0, label=f'Moving Average (w={window})')
            ax2.set_xlabel('Energy per Token (J)')
            ax2.set_ylabel('TTFT (ms)')
            ax2.set_title('Energy-Latency Trend')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            file_path = self.tradeoff_dir / f"{experiment_name}_energy_ttft_tradeoff.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            tradeoff_plots['energy_ttft_tradeoff'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 2. Tokens/J vs Throughput tradeoff
        if all(col in df_plot.columns for col in ['tokens_per_joule', 'throughput_mean']):
            fig, ax = plt.subplots(figsize=self.fig_size)

            ax.scatter(df_plot['tokens_per_joule'], df_plot['throughput_mean'],
                      c=df_plot.get('emc_freq', range(len(df_plot))),
                      cmap='plasma', s=100, alpha=0.7, edgecolors='black', linewidths=1)
            ax.set_xlabel('Tokens per Joule')
            ax.set_ylabel('Throughput (tokens/s)')
            ax.set_title('Efficiency vs Throughput')
            ax.grid(True, alpha=0.3)
            plt.colorbar(ax=ax, label='EMC Frequency (MHz)')

            file_path = self.tradeoff_dir / f"{experiment_name}_efficiency_throughput_tradeoff.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            tradeoff_plots['efficiency_throughput_tradeoff'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 3. Energy consumption vs Frequency combination
        if 'config_num' in df_plot.columns and 'energy_per_token_j' in df_plot.columns:
            fig, ax = plt.subplots(figsize=(14, 6))

            df_plot_sorted = df_plot.sort_values('config_num')

            # Create frequency combination labels
            df_plot_sorted['freq_combo'] = (df_plot_sorted['gpu_freq'].astype(str) + '-' +
                                          df_plot_sorted['cpu_freq'].astype(str) + '-' +
                                          df_plot_sorted['emc_freq'].astype(str))

            # Plot line chart
            ax.plot(range(len(df_plot_sorted)), df_plot_sorted['energy_per_token_j'],
                    marker='o', linewidth=2, markersize=8)

            # Add trend line
            z = np.polyfit(range(len(df_plot_sorted)), df_plot_sorted['energy_per_token_j'], 1)
            p = np.poly1d(z)
            ax.plot(range(len(df_plot_sorted)), p(range(len(df_plot_sorted))),
                    "--", alpha=0.7, linewidth=2, label='Trend')

            ax.set_xticks(range(0, len(df_plot_sorted), max(1, len(df_plot_sorted)//5)))
            ax.set_xticklabels([df_plot_sorted.iloc[i]['freq_combo'] for i in range(0, len(df_plot_sorted), max(1, len(df_plot_sorted)//5))],
                            rotation=45, ha='right')
            ax.set_ylabel('Energy per Token (J)')
            ax.set_title('Energy Consumption by Configuration')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            file_path = self.tradeoff_dir / f"{experiment_name}_energy_by_config.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            tradeoff_plots['energy_by_config'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        return tradeoff_plots

    def generate_comparison_plots(self, df: pd.DataFrame, experiment_name: str) -> Dict[str, str]:
        """
        Generate performance comparison plots.

        Args:
            df: DataFrame with experiment results
            experiment_name: Name of experiment

        Returns:
            Dictionary mapping comparison plot types to file paths
        """
        comparison_plots = {}

        df_plot = df.copy()

        # 1. TTFT comparison by frequency
        if 'gpu_freq' in df_plot.columns and 'ttft_ms_mean' in df_plot.columns:
            fig, ax = plt.subplots(figsize=self.fig_size)

            freq_groups = df_plot.groupby('gpu_freq')['ttft_ms_mean']
            x_pos = np.arange(len(freq_groups))

            bars = ax.bar(x_pos, [freq_groups.get(x, 0).mean() for x in freq_groups.groups],
                         yerr=[freq_groups.get(x, 0).std() for x in freq_groups.groups],
                         capsize=5, alpha=0.7, edgecolor='black', linewidth=1.5)

            ax.set_xlabel('GPU Frequency (MHz)')
            ax.set_ylabel('TTFT (ms)')
            ax.set_title('TTFT Comparison by GPU Frequency')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(freq_groups.groups)
            ax.legend(['Mean ± Std'])
            ax.grid(True, alpha=0.3, axis='y')

            file_path = self.pareto_dir / f"{experiment_name}_ttft_comparison.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            comparison_plots['ttft_comparison'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 2. Energy comparison by configuration
        if 'config_num' in df_plot.columns and 'energy_per_token_j' in df_plot.columns:
            fig, ax = plt.subplots(figsize=(16, 6))

            df_plot_sorted = df_plot.sort_values('energy_per_token_j')

            # Create grouped bar chart
            x = np.arange(len(df_plot_sorted))
            width = 0.6

            # Primary bars - energy per token
            ax.bar(x, df_plot_sorted['energy_per_token_j'], width,
                   label='Energy/Token (J)', alpha=0.8, edgecolor='black', linewidth=1.5)

            # Secondary bars - throughput (normalized)
            if 'throughput_mean' in df_plot.columns:
                max_throughput = df_plot_sorted['throughput_mean'].max()
                ax2 = ax.twinx()
                ax2.plot(x, df_plot_sorted['throughput_mean'] / max_throughput * 100,
                         'o-', color='red', linewidth=2, markersize=8, label='Relative Throughput (%)')
                ax2.set_ylabel('Relative Throughput (%)')
                ax2.legend(loc='upper right')

            ax.set_xlabel('Configuration Number')
            ax.set_ylabel('Energy per Token (J)')
            ax.set_title('Energy Efficiency Comparison')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3, axis='y')

            file_path = self.pareto_dir / f"{experiment_name}_energy_comparison.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            comparison_plots['energy_comparison'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        return comparison_plots

    def generate_energy_plots(self, df: pd.DataFrame, experiment_name: str) -> Dict[str, str]:
        """
        Generate energy-specific analysis plots.

        Args:
            df: DataFrame with experiment results
            experiment_name: Name of experiment

        Returns:
            Dictionary mapping energy plot types to file paths
        """
        energy_plots = {}

        df_plot = df.copy()

        # 1. Energy consumption distribution
        if 'energy_per_token_j' in df_plot.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Histogram
            ax1.hist(df_plot['energy_per_token_j'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
            ax1.set_xlabel('Energy per Token (J)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Energy Consumption Distribution')
            ax1.grid(True, alpha=0.3)

            # Box plot
            df_box = df_plot[['energy_per_token_j', 'gpu_freq']].copy()
            bp = df_box.boxplot(by='gpu_freq', ax=ax2)
            ax2.set_xlabel('GPU Frequency (MHz)')
            ax2.set_ylabel('Energy per Token (J)')
            ax2.set_title('Energy Distribution by Frequency')
            ax2.grid(True, alpha=0.3, axis='y')

            plt.suptitle('')
            plt.tight_layout()
            file_path = self.tradeoff_dir / f"{experiment_name}_energy_distribution.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            energy_plots['energy_distribution'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        # 2. Power consumption over time
        if 'avg_power_w' in df_plot.columns and 'config_num' in df_plot.columns:
            fig, ax = plt.subplots(figsize=self.fig_size)

            df_plot_sorted = df_plot.sort_values('config_num')

            ax.plot(df_plot_sorted['config_num'], df_plot_sorted['avg_power_w'],
                    marker='o', linewidth=2, markersize=8, color='darkorange')
            ax.fill_between(df_plot_sorted['config_num'], df_plot_sorted['avg_power_w'],
                             df_plot_sorted['avg_power_w'].min(), alpha=0.3, color='orange')

            ax.set_xlabel('Configuration Number')
            ax.set_ylabel('Average Power (W)')
            ax.set_title('Power Consumption Trend')
            ax.grid(True, alpha=0.3)

            file_path = self.tradeoff_dir / f"{experiment_name}_power_trend.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            energy_plots['power_trend'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        return energy_plots

    def generate_slo_plots(self, df: pd.DataFrame, experiment_name: str) -> Dict[str, str]:
        """
        Generate SLO satisfaction analysis plots.

        Args:
            df: DataFrame with experiment results
            experiment_name: Name of experiment

        Returns:
            Dictionary mapping SLO plot types to file paths
        """
        slo_plots = {}

        df_plot = df.copy()

        # Check if SLO data exists (this would require SLO config)
        # For now, create example SLO analysis plots

        # 1. Latency performance relative to typical SLO thresholds
        if all(col in df_plot.columns for col in ['ttft_ms_mean', 'tpot_ms_mean']):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Example SLO thresholds (these would come from slo.yaml)
            slo_ttft = 1000  # ms
            slo_tpot = 80     # ms

            # Calculate SLO satisfaction
            df_plot['ttft_slo_met'] = df_plot['ttft_ms_mean'] <= slo_ttft
            df_plot['tpot_slo_met'] = df_plot['tpot_ms_mean'] <= slo_tpot

            # Left plot: TTFT SLO satisfaction
            slo_counts = df_plot['ttft_slo_met'].value_counts()
            ax1.pie([slo_counts.get(True, 0), slo_counts.get(False, 0)],
                     labels=[f'Satisfied (≤{slo_ttft}ms)', f'violated (>{slo_ttft}ms)'],
                     autopct='%1.1f', colors=['lightgreen', 'lightcoral'])
            ax1.set_title('TTFT SLO Satisfaction')

            # Right plot: Latency distribution with SLO lines
            ax2.hist(df_plot['ttft_ms_mean'], bins=20, edgecolor='black', alpha=0.7)
            ax2.axvline(x=slo_ttft, color='green', linestyle='--', linewidth=2, label='SLO Threshold')
            ax2.set_xlabel('TTFT (ms)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Latency Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            file_path = self.tradeoff_dir / f"{experiment_name}_slo_analysis.png"
            plt.savefig(file_path, dpi=self.fig_dpi, bbox_inches='tight')
            plt.close()
            slo_plots['slo_analysis'] = str(file_path)
            logger.info(f"Generated: {file_path}")

        return slo_plots

    def generate_summary_report(self, generated_plots: Dict[str, str],
                              experiment_name: str = "experiment") -> str:
        """
        Generate summary report of generated visualizations.

        Args:
            generated_plots: Dictionary mapping plot types to file paths
            experiment_name: Name of experiment

        Returns:
            Path to summary report file
        """
        logger.info("Generating summary report...")

        report_path = self.output_dir / f"{experiment_name}_visualization_summary.md"

        try:
            with open(report_path, 'w') as f:
                f.write(f"# Visualization Summary - {experiment_name}\n\n")
                f.write(f"Generated: {pd.Timestamp.now()}\n\n")

                f.write("## Generated Visualizations\n\n")

                # Group by category
                categories = {
                    'Heatmaps': [],
                    'Pareto Frontier': [],
                    'Tradeoff Curves': [],
                    'Performance Comparisons': [],
                    'Energy Analysis': [],
                    'SLO Analysis': []
                }

                for plot_type, file_path in generated_plots.items():
                    if 'heatmap' in plot_type.lower():
                        categories['Heatmaps'].append((plot_type, file_path))
                    elif 'pareto' in plot_type.lower():
                        categories['Pareto Frontier'].append((plot_type, file_path))
                    elif 'tradeoff' in plot_type.lower():
                        categories['Tradeoff Curves'].append((plot_type, file_path))
                    elif 'comparison' in plot_type.lower():
                        categories['Performance Comparisons'].append((plot_type, file_path))
                    elif 'energy' in plot_type.lower():
                        categories['Energy Analysis'].append((plot_type, file_path))
                    elif 'slo' in plot_type.lower():
                        categories['SLO Analysis'].append((plot_type, file_path))

                # Write categories
                for category, plots in categories.items():
                    if plots:
                        f.write(f"### {category}\n\n")
                        for plot_type, file_path in plots:
                            f.write(f"- **{plot_type}**: `{file_path}`\n")
                        f.write("\n")

                f.write("## File Statistics\n\n")
                f.write(f"- Total visualizations: {len(generated_plots)}\n")
                f.write(f"- Output directory: `{self.output_dir}`\n")

            logger.info(f"Summary report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return None


def main():
    """
    Test function for visualization generator.
    """
    # Create sample data for testing
    sample_data = [
        {
            'config_num': 1,
            'gpu_freq': 1428,
            'cpu_freq': 2015,
            'emc_freq': 2133,
            'ttft_ms_mean': 245.3,
            'tpot_ms_mean': 45.2,
            'energy_per_token_j': 9.8,
            'tokens_per_joule': 0.102,
            'throughput_mean': 22.1,
            'avg_power_w': 35.4,
            'is_pareto': True
        },
        {
            'config_num': 2,
            'gpu_freq': 846,
            'cpu_freq': 1479,
            'emc_freq': 1600,
            'ttft_ms_mean': 310.5,
            'tpot_ms_mean': 58.7,
            'energy_per_token_j': 7.2,
            'tokens_per_joule': 0.139,
            'throughput_mean': 17.0,
            'avg_power_w': 25.1,
            'is_pareto': True
        },
        {
            'config_num': 3,
            'gpu_freq': 378,
            'cpu_freq': 1020,
            'emc_freq': 133,
            'ttft_ms_mean': 420.8,
            'tpot_ms_mean': 85.2,
            'energy_per_token_j': 12.5,
            'tokens_per_joule': 0.080,
            'throughput_mean': 12.3,
            'avg_power_w': 15.2,
            'is_pareto': False
        }
    ]

    df = pd.DataFrame(sample_data)

    # Create visualization generator
    viz_gen = VisualizationGenerator()

    logger.info("Testing visualization generator with sample data...")

    # Generate all visualizations
    generated_plots = viz_gen.generate_all_visualizations(df, "test_experiment")

    # Generate summary report
    summary_report = viz_gen.generate_summary_report(generated_plots, "test_experiment")

    if summary_report:
        logger.info(f"Visualization test completed. Summary: {summary_report}")
    else:
        logger.info("Visualization test completed")


if __name__ == "__main__":
    main()
