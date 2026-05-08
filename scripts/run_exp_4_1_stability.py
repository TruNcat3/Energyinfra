#!/usr/bin/env python3
"""
Experiment 4.1: Measurement Stability Test
Tests measurement stability using synthetic benchmark.

Purpose: Validate that measurements are stable across repeated runs.
"""

import sys
sys.path.insert(0, 'src')

import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from synthetic_benchmark import SyntheticBenchmark
from freq_controller import FrequencyController

def run_stability_experiment(repeats: int = 10, config: dict = None) -> dict:
    """
    Run stability experiment with fixed workload and frequency.

    Args:
        repeats: Number of repeated measurements
        config: Configuration dictionary

    Returns:
        Dictionary with experiment results
    """
    # Initialize controllers
    benchmark = SyntheticBenchmark(config)
    freq_controller = FrequencyController(config, control_method='hybrid')

    # Fixed test configuration
    workload = {
        'batch_size': 1,
        'prompt_len': 512,
        'output_len': 128,
        'phase': 'mixed'
    }

    frequency = {
        'gpu_freq': 846,    # Mid frequency
        'cpu_freq': 1479,   # Mid-high frequency
        'emc_freq': 1600    # Mid-high frequency
    }

    print(f"Running stability experiment with {repeats} repeats...")
    print(f"Workload: {workload}")
    print(f"Frequency: {frequency}")

    # Collect results
    results = []
    for i in range(repeats):
        print(f"\n--- Run {i+1}/{repeats} ---")

        # Get current frequencies
        current_freqs = freq_controller.get_all_frequencies()
        print(f"Current frequencies: {current_freqs}")

        # Run benchmark
        result = benchmark.run_benchmark(workload, frequency)
        result['run_number'] = i + 1

        # Add timing metadata
        result['timestamp'] = datetime.now().isoformat()
        result['wall_clock_time'] = time.time()

        results.append(result)
        print(f"TTFT: {result['ttft_ms']:.2f}ms, TPOT: {result['tpot_ms']:.2f}ms, "
              f"Energy: {result['energy_per_token_j']:.4f}J/token")

        # Small cooldown between runs
        time.sleep(1)

    # Analyze results
    df = pd.DataFrame(results)

    print("\n=== Stability Analysis ===")

    # Calculate statistics for key metrics
    key_metrics = ['ttft_ms', 'tpot_ms', 'total_time_ms', 'energy_per_token_j']
    analysis = {}

    for metric in key_metrics:
        values = df[metric].values
        analysis[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'cv': float(np.std(values) / np.mean(values)),  # Coefficient of variation
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'range': float(np.max(values) - np.min(values))
        }

        print(f"\n{metric}:")
        print(f"  Mean: {analysis[metric]['mean']:.4f}")
        print(f"  Std:  {analysis[metric]['std']:.4f}")
        print(f"  CV:   {analysis[metric]['cv']*100:.2f}%")
        print(f"  Range: {analysis[metric]['range']:.4f}")

    # Stability criteria
    stability_thresholds = {
        'ttft_ms': 0.05,      # 5% CV acceptable
        'tpot_ms': 0.05,
        'total_time_ms': 0.05,
        'energy_per_token_j': 0.08  # 8% CV acceptable (higher variance in power)
    }

    stability_results = {}
    for metric, threshold in stability_thresholds.items():
        is_stable = analysis[metric]['cv'] <= threshold
        stability_results[metric] = {
            'stable': is_stable,
            'cv': analysis[metric]['cv'],
            'threshold': threshold,
            'passed': is_stable
        }

        status = "✅ STABLE" if is_stable else "❌ UNSTABLE"
        print(f"  {status} (CV: {analysis[metric]['cv']*100:.2f}% <= {threshold*100:.1f}%)")

    # Overall stability assessment
    all_stable = all(result['stable'] for result in stability_results.values())
    print(f"\n{'='*40}")
    print(f"Overall Stability: {'✅ PASSED' if all_stable else '❌ FAILED'}")
    print(f"{'='*40}")

    # Return comprehensive results
    return {
        'workload': workload,
        'frequency': frequency,
        'results': results,
        'analysis': analysis,
        'stability': stability_results,
        'all_stable': all_stable,
        'repeats': repeats,
        'timestamp': datetime.now().isoformat()
    }


def save_results(results: dict, output_dir: str = 'data/parsed/'):
    """Save experiment results to files."""

    # Create output directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save raw results as CSV
    df = pd.DataFrame(results['results'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    raw_file = f"{output_dir}exp_4_1_stability_raw_{timestamp}.csv"
    df.to_csv(raw_file, index=False)
    print(f"\nRaw results saved to: {raw_file}")

    # Save analysis as JSON
    analysis_file = f"{output_dir}exp_4_1_stability_analysis_{timestamp}.json"
    with open(analysis_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Analysis saved to: {analysis_file}")

    return raw_file, analysis_file


def main():
    """Main function for stability experiment."""

    # Configuration
    config = {
        'synthetic_benchmark': {
            'base_time_per_token_ms': 0.2,
            'base_tpot_ms': 8.0,
            'variability_factor': 0.1
        },
        'frequencies': {
            'gpu': {
                'presets': {'low': 378, 'mid': 846, 'high': 1428}
            }
        }
    }

    print("=========================================")
    print("Experiment 4.1: Measurement Stability")
    print("=========================================")
    print()

    # Run stability experiment
    results = run_stability_experiment(repeats=10, config=config)

    # Save results
    raw_file, analysis_file = save_results(results)

    # Print summary
    print("\n=========================================")
    print("Experiment 4.1 Completed")
    print("=========================================")
    print(f"Repeats: {results['repeats']}")
    print(f"All stable: {results['all_stable']}")
    print(f"Results saved: {raw_file}")
    print(f"Analysis saved: {analysis_file}")

    if results['all_stable']:
        print("\n✅ System is stable and ready for full experiments!")
    else:
        print("\n⚠️  System shows measurement instability.")
        print("   Consider: increasing repeats, checking system load, reviewing configuration")

    return results['all_stable']


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)