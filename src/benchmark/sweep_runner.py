#!/usr/bin/env python3
"""
Sweep Runner for Experiments
Orchestrates workload × frequency sweep experiments.
"""

import logging
import time
import json
import copy
from typing import Dict, List, Optional
from pathlib import Path
import pickle
from datetime import datetime

from src.controller.freq_controller import FrequencyController
from src.metrics.metrics_collector import MetricsCollector
from src.benchmark.benchmark_runner import BenchmarkRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SweepRunner:
    """
    Orchestrator for running workload × frequency sweep experiments.

    Manages:
    - Workload and frequency configuration combinations
    - Warmup, measurement, and cooldown phases
    - Repetitions for statistical significance
    - Checkpointing and resume capability
    - Progress tracking and reporting
    """

    def __init__(self, config: Dict, output_dir: str = "data/raw_logs"):
        """
        Initialize sweep runner.

        Args:
            config: Full configuration dictionary
            output_dir: Directory for all output files
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-components
        self.freq_controller = FrequencyController(config)
        self.benchmark_runner = BenchmarkRunner(config)

        # Sweep configuration
        self.sweep_config = config.get('sweep', {})
        self.workloads_config = config.get('workloads', {})
        self.frequencies_config = config.get('frequencies', {})

        # Experiment tracking
        self.experiment_id = f"sweep_{int(time.time())}"
        self.results = []
        self.checkpoints = []
        self.current_checkpoint = None

        # Progress tracking
        self.total_configs = 0
        self.completed_configs = 0
        self.start_time = None

        # Resume configuration
        self.resume_config = self.sweep_config.get('resume', {})
        self.resume_enabled = self.resume_config.get('enable', False)

        # Metrics collector (will be created per run)
        self.metrics_collector = None

    def generate_configurations(self, experiment_name: str) -> List[Dict]:
        """
        Generate all workload × frequency configurations for an experiment.

        Args:
            experiment_name: Name of the experiment from workloads config

        Returns:
            List of configuration dictionaries
        """
        logger.info(f"Generating configurations for experiment: {experiment_name}")

        try:
            # Get experiment configuration
            exp_config = self.workloads_config.get('preliminary_experiments', {}).get(experiment_name, {})

            if not exp_config:
                logger.error(f"Experiment configuration not found: {experiment_name}")
                return []

            configurations = []

            # Handle different experiment types
            if experiment_name == 'measurement_stability':
                # Single configuration, multiple repetitions
                configurations = self._generate_single_config(exp_config, repetitions=exp_config.get('repeats', 10))

            elif experiment_name == 'single_knob_sensitivity':
                # Multiple configurations for each knob
                configurations.extend(self._generate_single_knob_configs(exp_config))

            elif experiment_name == 'frequency_combinations':
                # Full factorial design
                configurations.extend(self._generate_combination_configs(exp_config))

            elif experiment_name == 'phase_differences':
                # Different phase configurations
                configurations.extend(self._generate_phase_configs(exp_config))

            elif experiment_name == 'workload_predictability':
                # Wide range of workload variations
                configurations.extend(self._generate_workload_variation_configs(exp_config))

            elif experiment_name == 'switching_overhead':
                # Frequency switching configurations
                configurations.extend(self._generate_switching_configs(exp_config))

            elif experiment_name == 'slo_constrained':
                # Baseline comparisons
                configurations.extend(self._generate_baseline_configs(exp_config))

            else:
                logger.error(f"Unknown experiment type: {experiment_name}")
                return []

            self.total_configs = len(configurations)
            logger.info(f"Generated {self.total_configs} configurations")
            return configurations

        except Exception as e:
            logger.error(f"Failed to generate configurations: {e}")
            return []

    def _generate_single_config(self, exp_config: Dict, repetitions: int = 1) -> List[Dict]:
        """Generate single configuration with multiple repetitions."""
        configs = []

        for i in range(repetitions):
            config = {
                'workload': {
                    'model': exp_config.get('model'),
                    'batch_size': exp_config.get('batch_size'),
                    'prompt_len': exp_config.get('prompt_len'),
                    'output_len': exp_config.get('output_len'),
                    'phase': exp_config.get('phase'),
                    'concurrency': exp_config.get('concurrency'),
                    'warmup_runs': exp_config.get('warmup_runs', 1),
                    'repeats': 1  # Each config is one run
                },
                'frequency': {
                    'gpu_freq': exp_config.get('gpu_freq'),
                    'cpu_freq': exp_config.get('cpu_freq'),
                    'emc_freq': exp_config.get('emc_freq')
                },
                'rep': i + 1,
                'total_reps': repetitions
            }
            configs.append(config)

        return configs

    def _generate_single_knob_configs(self, exp_config: Dict) -> List[Dict]:
        """Generate configurations for single-knob sensitivity experiment."""
        configs = []
        base_workload = {
            'model': exp_config.get('model'),
            'batch_size': exp_config.get('batch_size'),
            'prompt_len': exp_config.get('prompt_len'),
            'output_len': exp_config.get('output_len'),
            'phase': exp_config.get('phase'),
            'concurrency': exp_config.get('concurrency'),
            'warmup_runs': exp_config.get('warmup_runs', 1),
            'repeats': exp_config.get('repeats', 5)
        }

        # GPU sweep
        gpu_sweep = exp_config.get('gpu_sweep', {})
        for i, gpu_freq in enumerate(gpu_sweep.get('gpu_freqs', [])):
            for rep in range(exp_config.get('repeats', 5)):
                config = {
                    'workload': base_workload.copy(),
                    'frequency': {
                        'gpu_freq': gpu_freq,
                        'cpu_freq': gpu_sweep.get('cpu_freq'),
                        'emc_freq': gpu_sweep.get('emc_freq')
                    },
                    'experiment_type': 'gpu_sweep',
                    'rep': rep + 1
                }
                configs.append(config)

        # CPU sweep
        cpu_sweep = exp_config.get('cpu_sweep', {})
        for i, cpu_freq in enumerate(cpu_sweep.get('cpu_freqs', [])):
            for rep in range(exp_config.get('repeats', 5)):
                config = {
                    'workload': base_workload.copy(),
                    'frequency': {
                        'gpu_freq': cpu_sweep.get('gpu_freq'),
                        'cpu_freq': cpu_freq,
                        'emc_freq': cpu_sweep.get('emc_freq')
                    },
                    'experiment_type': 'cpu_sweep',
                    'rep': rep + 1
                }
                configs.append(config)

        # EMC sweep
        emc_sweep = exp_config.get('emc_sweep', {})
        for i, emc_freq in enumerate(emc_sweep.get('emc_freqs', [])):
            for rep in range(exp_config.get('repeats', 5)):
                config = {
                    'workload': base_workload.copy(),
                    'frequency': {
                        'gpu_freq': emc_sweep.get('gpu_freq'),
                        'cpu_freq': emc_sweep.get('cpu_freq'),
                        'emc_freq': emc_freq
                    },
                    'experiment_type': 'emc_sweep',
                    'rep': rep + 1
                }
                configs.append(config)

        return configs

    def _generate_combination_configs(self, exp_config: Dict) -> List[Dict]:
        """Generate all frequency combination configurations."""
        configs = []
        base_workload = {
            'model': exp_config.get('model'),
            'batch_size': exp_config.get('batch_size'),
            'prompt_len': exp_config.get('prompt_len'),
            'output_len': exp_config.get('output_len'),
            'phase': exp_config.get('phase'),
            'concurrency': exp_config.get('concurrency'),
            'warmup_runs': exp_config.get('warmup_runs', 1),
            'repeats': exp_config.get('repeats', 5)
        }

        gpu_freqs = exp_config.get('gpu_freqs', [])
        cpu_freqs = exp_config.get('cpu_freqs', [])
        emc_freqs = exp_config.get('emc_freqs', [])
        repeats = exp_config.get('repeats', 5)

        # Full factorial design
        for gpu_freq in gpu_freqs:
            for cpu_freq in cpu_freqs:
                for emc_freq in emc_freqs:
                    for rep in range(repeats):
                        config = {
                            'workload': base_workload.copy(),
                            'frequency': {
                                'gpu_freq': gpu_freq,
                                'cpu_freq': cpu_freq,
                                'emc_freq': emc_freq
                            },
                            'experiment_type': 'frequency_combinations',
                            'rep': rep + 1
                        }
                        configs.append(config)

        return configs

    def _generate_phase_configs(self, exp_config: Dict) -> List[Dict]:
        """Generate phase-specific configurations."""
        configs = []
        base_workload_base = {
            'model': exp_config.get('model'),
            'batch_size': exp_config.get('batch_size'),
            'concurrency': exp_config.get('concurrency'),
            'warmup_runs': exp_config.get('warmup_runs', 1),
            'repeats': exp_config.get('repeats', 5)
        }

        gpu_freqs = exp_config.get('gpu_freqs', [])
        repeats = exp_config.get('repeats', 5)

        # Generate configs for each phase type
        phase_types = ['only_prefill', 'only_decode', 'prefill_heavy', 'decode_heavy']

        for phase_type in phase_types:
            phase_config = exp_config.get(phase_type, {})
            workload = base_workload_base.copy()
            workload.update({
                'prompt_len': phase_config.get('prompt_len'),
                'output_len': phase_config.get('output_len'),
                'phase': phase_config.get('phase')
            })

            for gpu_freq in gpu_freqs:
                for rep in range(repeats):
                    config = {
                        'workload': workload.copy(),
                        'frequency': {
                            'gpu_freq': gpu_freq,
                            'cpu_freq': exp_config.get('cpu_freq'),
                            'emc_freq': exp_config.get('emc_freq')
                        },
                        'experiment_type': 'phase_differences',
                        'phase_type': phase_type,
                        'rep': rep + 1
                    }
                    configs.append(config)

        return configs

    def _generate_workload_variation_configs(self, exp_config: Dict) -> List[Dict]:
        """Generate workload variation configurations for predictability experiment."""
        configs = []

        models = exp_config.get('models', [])
        batch_sizes = exp_config.get('batch_sizes', [])
        prompt_lengths = exp_config.get('prompt_lengths', [])
        output_lengths = exp_config.get('output_lengths', [])
        phases = exp_config.get('phases', [])
        concurrency_levels = exp_config.get('concurrency_levels', [])
        repeats = exp_config.get('repeats', 3)

        # Use all frequency combinations (would be large, so maybe sample)
        gpu_freqs = self.frequencies_config.get('gpu', {}).get('presets', {}).keys()
        cpu_freqs = self.frequencies_config.get('cpu', {}).get('presets', {}).keys()
        emc_freqs = self.frequencies_config.get('emc', {}).get('presets', {}).keys()

        # Generate a representative subset to keep experiment manageable
        # In practice, you might want to sample strategically
        selected_freqs = [
            {'gpu': 'low', 'cpu': 'low', 'emc': 'low'},
            {'gpu': 'mid', 'cpu': 'mid', 'emc': 'mid'},
            {'gpu': 'high', 'cpu': 'high', 'emc': 'high'}
        ]

        for model in models[:1]:  # Limit to one model initially
            for batch_size in batch_sizes:
                for prompt_len in prompt_lengths[:2]:  # Sample prompt lengths
                    for output_len in output_lengths[:2]:  # Sample output lengths
                        for phase in phases[:2]:  # Sample phases
                            for concurrency in concurrency_levels[:2]:  # Sample concurrency
                                for freq_combo in selected_freqs:
                                    for rep in range(repeats):
                                        workload = {
                                            'model': model,
                                            'batch_size': batch_size,
                                            'prompt_len': prompt_len,
                                            'output_len': output_len,
                                            'phase': phase,
                                            'concurrency': concurrency,
                                            'warmup_runs': 1,
                                            'repeats': 1
                                        }
                                        config = {
                                            'workload': workload,
                                            'frequency': {
                                                'gpu_freq': freq_combo['gpu'],
                                                'cpu_freq': freq_combo['cpu'],
                                                'emc_freq': freq_combo['emc']
                                            },
                                            'experiment_type': 'workload_predictability',
                                            'rep': rep + 1
                                        }
                                        configs.append(config)

        return configs

    def _generate_switching_configs(self, exp_config: Dict) -> List[Dict]:
        """Generate frequency switching overhead configurations."""
        configs = []
        base_workload = {
            'model': exp_config.get('model'),
            'batch_size': exp_config.get('batch_size'),
            'prompt_len': exp_config.get('prompt_len'),
            'output_len': exp_config.get('output_len'),
            'phase': exp_config.get('phase'),
            'concurrency': exp_config.get('concurrency'),
            'warmup_runs': exp_config.get('warmup_runs', 1),
            'repeats': 1
        }
        repeats = exp_config.get('repeats', 10)

        # GPU switches
        gpu_switches = exp_config.get('gpu_switches', [])
        for switch_config in gpu_switches:
            for rep in range(repeats):
                config = {
                    'workload': base_workload.copy(),
                    'frequency': {
                        'gpu_freq': switch_config.get('to'),
                        'cpu_freq': 'high',  # Keep other frequencies constant
                        'emc_freq': 'high'
                    },
                    'experiment_type': 'switching_overhead',
                    'switch_type': 'gpu',
                    'switch_from': switch_config.get('from'),
                    'switch_to': switch_config.get('to'),
                    'rep': rep + 1
                }
                configs.append(config)

        # Similar for EMC and CPU switches...
        # (omitted for brevity)

        return configs

    def _generate_baseline_configs(self, exp_config: Dict) -> List[Dict]:
        """Generate baseline comparison configurations."""
        configs = []
        base_workload = {
            'model': exp_config.get('model'),
            'batch_size': exp_config.get('batch_size'),
            'prompt_len': exp_config.get('prompt_len'),
            'output_len': exp_config.get('output_len'),
            'phase': exp_config.get('phase'),
            'concurrency': exp_config.get('concurrency'),
            'warmup_runs': exp_config.get('warmup_runs', 1),
            'repeats': exp_config.get('repeats', 5)
        }

        # Generate configs for each baseline
        baselines = exp_config.get('baselines', [])
        for baseline in baselines:
            baseline_name = baseline.get('name')

            for rep in range(exp_config.get('repeats', 5)):
                if baseline.get('use_selector'):
                    # Use adaptive selector (to be implemented)
                    config = {
                        'workload': base_workload.copy(),
                        'frequency': {'method': 'adaptive'},
                        'experiment_type': 'slo_constrained',
                        'baseline': baseline_name,
                        'rep': rep + 1
                    }
                else:
                    config = {
                        'workload': base_workload.copy(),
                        'frequency': {
                            'gpu_freq': baseline.get('gpu_freq'),
                            'cpu_freq': baseline.get('cpu_freq'),
                            'emc_freq': baseline.get('emc_freq')
                        },
                        'experiment_type': 'slo_constrained',
                        'baseline': baseline_name,
                        'rep': rep + 1
                    }
                configs.append(config)

        return configs

    def run_sweep(self, experiment_name: str) -> bool:
        """
        Run complete sweep for an experiment.

        Args:
            experiment_name: Name of experiment to run

        Returns:
            True if sweep completed successfully, False otherwise
        """
        logger.info(f"Starting sweep for experiment: {experiment_name}")
        self.start_time = time.time()

        # Generate configurations
        configurations = self.generate_configurations(experiment_name)

        if not configurations:
            logger.error("No configurations to run")
            return False

        # Check for resume
        start_index = 0
        if self.resume_enabled and self.resume_config.get('resume_from'):
            start_index = self._load_checkpoint(self.resume_config.get('resume_from'))
            if start_index > 0:
                logger.info(f"Resuming from checkpoint: configuration {start_index}")

        # Save default frequencies
        self.freq_controller.save_defaults()

        try:
            # Run each configuration
            for i, config in enumerate(configurations[start_index:], start=start_index):
                logger.info(f"\n{'='*60}")
                logger.info(f"Configuration {i+1}/{self.total_configs}")
                logger.info(f"{'='*60}")

                success = self._run_single_config(config, i+1, self.total_configs)

                if success:
                    self.completed_configs += 1
                else:
                    logger.error(f"Configuration {i+1} failed")

                    # Check if we should skip or continue
                    max_failures = self.sweep_config.get('error_handling', {}).get('max_failures', 3)
                    consecutive_failures = self._count_consecutive_failures()
                    if consecutive_failures >= max_failures:
                        logger.error(f"Too many consecutive failures ({consecutive_failures}), stopping sweep")
                        return False

                    if self.sweep_config.get('error_handling', {}).get('skip_on_error', False):
                        logger.info("Skipping failed configuration and continuing")
                    else:
                        logger.error("Stopping sweep due to configuration failure")
                        return False

                # Save checkpoint periodically
                checkpoint_interval = self.sweep_config.get('monitoring', {}).get('checkpoint_interval', 10)
                if (i + 1) % checkpoint_interval == 0:
                    self._save_checkpoint(i + 1)

            # Restore default frequencies
            self.freq_controller.restore_defaults()

            # Save final results
            self._save_results()

            total_time = time.time() - self.start_time
            logger.info(f"\n{'='*60}")
            logger.info(f"Sweep completed successfully")
            logger.info(f"Total configurations: {self.total_configs}")
            logger.info(f"Completed: {self.completed_configs}")
            logger.info(f"Failed: {self.total_configs - self.completed_configs}")
            logger.info(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.2f} minutes)")
            logger.info(f"{'='*60}\n")

            return True

        except KeyboardInterrupt:
            logger.info("Sweep interrupted by user")
            self._save_checkpoint(self.completed_configs)
            self.freq_controller.restore_defaults()
            return False
        except Exception as e:
            logger.error(f"Sweep failed with error: {e}")
            self.freq_controller.restore_defaults()
            return False

    def _run_single_config(self, config: Dict, config_num: int, total_configs: int) -> bool:
        """
        Run a single configuration with warmup, measurement, and cooldown.

        Args:
            config: Configuration dictionary
            config_num: Configuration number
            total_configs: Total number of configurations

        Returns:
            True if successful, False otherwise
        """
        workload = config['workload']
        frequency = config['frequency']
        experiment_type = config.get('experiment_type', 'unknown')

        # Print progress
        progress_pct = (config_num / total_configs) * 100
        logger.info(f"Progress: {config_num}/{total_configs} ({progress_pct:.1f}%)")
        logger.info(f"Experiment type: {experiment_type}")
        logger.info(f"Workload: {workload}")
        logger.info(f"Frequency: {frequency}")

        # Set frequencies
        logger.info("Setting frequencies...")
        success = self._set_frequencies(frequency)
        if not success:
            logger.error("Failed to set frequencies")
            return False

        # Start metrics collection
        self.metrics_collector = MetricsCollector(
            interval_ms=self.sweep_config.get('monitoring', {}).get('interval_sec', 10) * 1000,
            output_dir=str(self.output_dir)
        )
        metrics_log_file = self.metrics_collector.start_collection()
        if not metrics_log_file:
            logger.error("Failed to start metrics collection")
            return False

        try:
            # Run warmup if specified
            warmup_runs = workload.get('warmup_runs', 1)
            if warmup_runs > 0:
                logger.info(f"Running {warmup_runs} warmup run(s)...")
                warmup_success = self.benchmark_runner.run_warmup(
                    workload, frequency, warmup_runs=warmup_runs,
                    timeout_sec=self.sweep_config.get('timeouts', {}).get('warmup_run_sec', 300)
                )
                if not warmup_success:
                    logger.error("Warmup failed")
                    return False

            # Run measurement
            logger.info("Running measurement...")
            measurement_results = self.benchmark_runner.run_measurement(
                workload, frequency, measurement_runs=workload.get('repeats', 1),
                timeout_sec=self.sweep_config.get('timeouts', {}).get('measurement_run_sec', 600)
            )

            if not measurement_results:
                logger.error("Measurement failed")
                return False

            # Stop metrics collection
            self.metrics_collector.stop_collection()

            # Calculate energy consumption
            logger.info("Calculating energy consumption...")
            energy_metrics = self.metrics_collector.calculate_energy_consumption()

            # Calculate benchmark statistics
            logger.info("Calculating benchmark statistics...")
            benchmark_stats = self.benchmark_runner.calculate_statistics(measurement_results)

            # Get metrics summary
            metrics_summary = self.metrics_collector.get_summary_statistics()

            # Combine results
            combined_result = {
                'config_num': config_num,
                'experiment_type': experiment_type,
                'config': config,
                'benchmark_results': measurement_results,
                'benchmark_statistics': benchmark_stats,
                'energy_metrics': energy_metrics,
                'metrics_summary': metrics_summary,
                'timestamp': time.time()
            }

            self.results.append(combined_result)

            logger.info(f"Configuration {config_num} completed successfully")
            if benchmark_stats:
                logger.info(f"Key metrics: TTFT={benchmark_stats.get('ttft_ms_mean', 0):.2f}ms, "
                          f"TPOT={benchmark_stats.get('tpot_ms_mean', 0):.2f}ms, "
                          f"Energy={energy_metrics.get('avg_power_w', 0):.2f}W")

            # Cooldown period
            cooldown_sec = self.sweep_config.get('runs', {}).get('cooldown', 30)
            if cooldown_sec > 0 and config_num < total_configs:
                logger.info(f"Cooldown for {cooldown_sec} seconds...")
                time.sleep(cooldown_sec)

            return True

        except Exception as e:
            logger.error(f"Error running configuration: {e}")
            self.metrics_collector.stop_collection()
            return False

    def _set_frequencies(self, frequency: Dict) -> bool:
        """
        Set frequencies based on configuration.

        Args:
            frequency: Frequency configuration dict

        Returns:
            True if successful, False otherwise
        """
        if frequency.get('method') == 'adaptive':
            # Use adaptive selector (to be implemented)
            logger.info("Using adaptive frequency selection")
            return True

        gpu_freq = frequency.get('gpu_freq')
        cpu_freq = frequency.get('cpu_freq')
        emc_freq = frequency.get('emc_freq')

        # Handle null frequencies (use system default)
        if gpu_freq is None or cpu_freq is None or emc_freq is None:
            logger.info("Some frequencies are null, using system defaults")
            return True

        # Convert presets to actual frequencies
        gpu_freq_value = self.freq_controller.presets.get('gpu', {}).get(gpu_freq, gpu_freq)
        cpu_freq_value = self.freq_controller.presets.get('cpu', {}).get(cpu_freq, cpu_freq)
        emc_freq_value = self.freq_controller.presets.get('emc', {}).get(emc_freq, emc_freq)

        # Set frequencies
        if all([gpu_freq_value, cpu_freq_value, emc_freq_value]):
            return self.freq_controller.set_all_frequencies(gpu_freq_value, cpu_freq_value, emc_freq_value)
        else:
            logger.error("Invalid frequency values")
            return False

    def _count_consecutive_failures(self) -> int:
        """Count consecutive configuration failures."""
        consecutive = 0
        for result in reversed(self.results):
            if not result.get('benchmark_results'):
                consecutive += 1
            else:
                break
        return consecutive

    def _save_checkpoint(self, completed_configs: int):
        """
        Save checkpoint for resume capability.

        Args:
            completed_configs: Number of completed configurations
        """
        checkpoint_data = {
            'experiment_id': self.experiment_id,
            'completed_configs': completed_configs,
            'total_configs': self.total_configs,
            'timestamp': time.time(),
            'results': self.results
        }

        checkpoint_file = self.output_dir / f"checkpoint_{self.experiment_id}_{completed_configs}.pkl"
        try:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            logger.info(f"Checkpoint saved: {checkpoint_file}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self, checkpoint_file: str) -> int:
        """
        Load checkpoint and resume from saved state.

        Args:
            checkpoint_file: Path to checkpoint file

        Returns:
            Index of configuration to resume from
        """
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            self.results = checkpoint_data.get('results', [])
            self.completed_configs = checkpoint_data.get('completed_configs', 0)
            self.total_configs = checkpoint_data.get('total_configs', 0)

            logger.info(f"Loaded checkpoint: {checkpoint_file}")
            logger.info(f"Resuming from configuration {self.completed_configs}")

            return self.completed_configs

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 0

    def _save_results(self):
        """Save complete sweep results."""
        results_file = self.output_dir / f"results_{self.experiment_id}.json"

        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info(f"Results saved to: {results_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """
    Test function for sweep runner.
    """
    # Load configuration files (simplified example)
    import yaml

    try:
        with open('configs/platform.yaml') as f:
            platform_config = yaml.safe_load(f)
        with open('configs/workloads.yaml') as f:
            workloads_config = yaml.safe_load(f)
        with open('configs/frequencies.yaml') as f:
            frequencies_config = yaml.safe_load(f)
        with open('configs/sweep.yaml') as f:
            sweep_config = yaml.safe_load(f)

        # Combine configs
        full_config = {
            'platform': platform_config.get('platform', {}),
            'frequencies': frequencies_config.get('frequencies', {}),
            'sweep': sweep_config.get('sweep', {}),
            'workloads': workloads_config
        }

        # Create sweep runner
        runner = SweepRunner(full_config)

        # Run a small test experiment
        experiment_name = "measurement_stability"
        logger.info(f"Testing sweep runner with experiment: {experiment_name}")

        success = runner.run_sweep(experiment_name)

        if success:
            logger.info("Test sweep completed successfully")
        else:
            logger.error("Test sweep failed")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        logger.info("This is expected if config files don't exist yet")


if __name__ == "__main__":
    main()