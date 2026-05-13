#!/usr/bin/env python3
"""
Synthetic Benchmark for System Validation
Simulates LLM inference workload for testing core modules without actual LLM models.
"""

import subprocess
import time
import json
import logging
import numpy as np
from typing import Dict, Optional
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticBenchmark:
    """
    Synthetic benchmark that simulates LLM inference for testing core modules.

    This benchmark simulates:
    - Prefill phase (computation-intensive)
    - Decode phase (memory-intensive)
    - Realistic timing patterns
    """

    def __init__(self, config: Dict):
        """Initialize synthetic benchmark with configuration."""
        self.config = config
        self.benchmark_config = config.get('synthetic_benchmark', {})

    def simulate_prefill(self, prompt_len: int) -> Dict:
        """
        Simulate prefill phase (prompt processing).

        Args:
            prompt_len: Length of prompt in tokens

        Returns:
            Dictionary with simulated metrics
        """
        logger.info(f"Simulating prefill phase with {prompt_len} tokens")

        # Simulate computation time based on prompt length
        # Typical prefill time: ~0.1-0.5ms per token depending on model
        base_time_ms = 0.2  # Base time per token
        pref_time_ms = base_time_ms * prompt_len

        # Add some variability
        variability = np.random.normal(0, 0.1 * pref_time_ms)
        pref_time_ms = max(0.1, pref_time_ms + variability)

        # Simulate power consumption during prefill
        # Prefill is compute-intensive: higher GPU power
        avg_power_w = 25 + np.random.normal(0, 5)
        max_power_w = avg_power_w + np.random.uniform(5, 15)

        # Simulate temperature
        temp_c = 45 + np.random.normal(0, 5)

        # Calculate energy
        energy_j = (avg_power_w * pref_time_ms) / 1000

        metrics = {
            'phase': 'prefill',
            'prompt_length': prompt_len,
            'prefill_time_ms': pref_time_ms,
            'total_time_ms': pref_time_ms,  # Add this field for consistency
            'avg_power_w': avg_power_w,
            'max_power_w': max_power_w,
            'temperature_c': temp_c,
            'energy_j': energy_j,
            'tokens_per_second': prompt_len / (pref_time_ms / 1000),
            'ttft_ms': pref_time_ms / prompt_len if prompt_len > 0 else 0,  # Approximate TTFT
            'tpot_ms': 0,  # No output tokens
            'energy_per_token_j': energy_j / prompt_len if prompt_len > 0 else 0
        }

        logger.info(f"Prefill metrics: {pref_time_ms:.2f}ms, {avg_power_w:.1f}W, {energy_j:.2f}J")
        return metrics

    def simulate_decode(self, output_len: int, prompt_len: int = 128) -> Dict:
        """
        Simulate decode phase (token generation).

        Args:
            output_len: Number of output tokens
            prompt_len: Length of prompt (affects cache size)

        Returns:
            Dictionary with simulated metrics
        """
        logger.info(f"Simulating decode phase generating {output_len} tokens")

        # Simulate time per output token (TPOT)
        # Decode is memory-intensive: slower than prefill per token
        base_tpot_ms = 8.0  # Base time per output token

        # Longer prompt means larger KV cache, slightly faster decoding
        cache_effect = min(0.2, prompt_len / 1024 * 0.1)
        tpot_ms = base_tpot_ms * (1 - cache_effect)

        # Add variability between tokens
        tpot_times = np.random.normal(tpot_ms, 1.0, output_len)
        tpot_times = np.maximum(tpot_times, 2.0)  # Minimum 2ms per token

        total_decode_time_ms = np.sum(tpot_times)

        # First token latency (TTFT)
        # First token is usually slower than subsequent tokens
        ttft_ms = tpot_times[0] * 1.5

        # Simulate power consumption during decode
        # Decode is memory-intensive: lower GPU power than prefill
        avg_power_w = 15 + np.random.normal(0, 3)
        max_power_w = avg_power_w + np.random.uniform(2, 8)

        # Simulate temperature
        temp_c = 48 + np.random.normal(0, 4)

        # Calculate energy
        energy_j = (avg_power_w * total_decode_time_ms) / 1000

        metrics = {
            'phase': 'decode',
            'prompt_length': prompt_len,
            'output_length': output_len,
            'ttft_ms': ttft_ms,
            'tpot_ms': np.mean(tpot_times[1:]) if len(tpot_times) > 1 else tpot_times[0],  # Exclude first token
            'total_time_ms': total_decode_time_ms,
            'avg_power_w': avg_power_w,
            'max_power_w': max_power_w,
            'temperature_c': temp_c,
            'energy_j': energy_j,
            'tokens_per_second': output_len / (total_decode_time_ms / 1000),
            'energy_per_token_j': energy_j / output_len if output_len > 0 else 0
        }

        logger.info(f"Decode metrics: TTFT={ttft_ms:.2f}ms, TPOT={metrics['tpot_ms']:.2f}ms, {avg_power_w:.1f}W, {energy_j:.2f}J")
        return metrics

    def simulate_full_inference(self, prompt_len: int, output_len: int, batch_size: int = 1) -> Dict:
        """
        Simulate full inference pipeline (prefill + decode).

        Args:
            prompt_len: Length of prompt in tokens
            output_len: Number of output tokens
            batch_size: Batch size (affects performance)

        Returns:
            Dictionary with complete inference metrics
        """
        logger.info(f"Simulating full inference: prompt={prompt_len}, output={output_len}, batch={batch_size}")

        # Adjust for batch size
        # Larger batches can be more efficient but consume more power
        batch_efficiency = 1.0 + (batch_size - 1) * 0.1  # 10% efficiency gain per additional request
        batch_power_multiplier = 1.0 + (batch_size - 1) * 0.15  # 15% power increase per additional request

        # Simulate prefill phase
        prefill_metrics = self.simulate_prefill(prompt_len * batch_size)

        # Simulate decode phase
        decode_metrics = self.simulate_decode(output_len, prompt_len)

        # Apply batch adjustments
        prefill_metrics['prefill_time_ms'] /= batch_efficiency
        prefill_metrics['energy_j'] *= batch_power_multiplier
        decode_metrics['total_time_ms'] /= batch_efficiency
        decode_metrics['energy_j'] *= batch_power_multiplier

        # Calculate combined metrics
        total_energy_j = prefill_metrics['energy_j'] + decode_metrics['energy_j']
        total_time_ms = prefill_metrics['prefill_time_ms'] + decode_metrics['total_time_ms']

        combined_metrics = {
            'batch_size': batch_size,
            'prompt_length': prompt_len,
            'output_length': output_len,
            'ttft_ms': prefill_metrics['prefill_time_ms'] + decode_metrics['ttft_ms'],
            'tpot_ms': decode_metrics['tpot_ms'],
            'total_time_ms': total_time_ms,
            'avg_power_w': (prefill_metrics['avg_power_w'] + decode_metrics['avg_power_w']) / 2,
            'max_power_w': max(prefill_metrics['max_power_w'], decode_metrics['max_power_w']),
            'temperature_c': max(prefill_metrics['temperature_c'], decode_metrics['temperature_c']),
            'total_energy_j': total_energy_j,
            'energy_per_token_j': total_energy_j / output_len,
            'tokens_per_second': output_len / (total_time_ms / 1000),
            'prefill_time_ms': prefill_metrics['prefill_time_ms'],
            'decode_time_ms': decode_metrics['total_time_ms']
        }

        logger.info(f"Full inference completed: {total_time_ms:.2f}ms total, {total_energy_j:.2f}J total")
        return combined_metrics

    def run_benchmark(self, workload: Dict, frequency: Dict) -> Dict:
        """
        Run synthetic benchmark with specified workload and frequency.

        Args:
            workload: Dictionary containing batch_size, prompt_len, output_len, phase
            frequency: Dictionary containing gpu_freq, cpu_freq, emc_freq

        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Running synthetic benchmark with workload: {workload}")
        logger.info(f"Frequencies: GPU={frequency.get('gpu_freq')}MHz, CPU={frequency.get('cpu_freq')}MHz, EMC={frequency.get('emc_freq')}MHz")

        # Simulate frequency effects
        # Higher frequencies = better performance but more power
        gpu_freq = frequency.get('gpu_freq', 846)
        freq_factor = gpu_freq / 846  # Normalize to mid-frequency

        # Apply frequency effects to timing
        # Higher GPU frequency -> faster computation, lower power efficiency
        self.benchmark_config['freq_factor'] = freq_factor

        # Run appropriate benchmark based on phase
        phase = workload.get('phase', 'mixed')

        if phase == 'prefill':
            result = self.simulate_prefill(workload['prompt_len'])
        elif phase == 'decode':
            result = self.simulate_decode(workload['output_len'], workload['prompt_len'])
        else:  # mixed
            result = self.simulate_full_inference(
                workload['prompt_len'],
                workload['output_len'],
                workload['batch_size']
            )

        # Apply frequency effects to results
        if 'freq_factor' in self.benchmark_config:
            freq_factor = self.benchmark_config['freq_factor']

            # Adjust timing based on frequency
            for time_key in ['prefill_time_ms', 'tpot_ms', 'total_time_ms']:
                if time_key in result:
                    result[time_key] /= freq_factor

            # Adjust power consumption based on frequency
            result['avg_power_w'] *= (0.8 + 0.2 * freq_factor)
            result['max_power_w'] *= (0.8 + 0.2 * freq_factor)

            # Recalculate energy
            if 'total_energy_j' in result:
                result['total_energy_j'] = (result['avg_power_w'] * result['total_time_ms']) / 1000
            elif 'energy_j' in result:
                result['energy_j'] = (result['avg_power_w'] * result['total_time_ms']) / 1000

        # Add frequency information to result
        result['gpu_freq_mhz'] = frequency.get('gpu_freq')
        result['cpu_freq_mhz'] = frequency.get('cpu_freq')
        result['emc_freq_mhz'] = frequency.get('emc_freq')

        return result


def main():
    """Main function for testing synthetic benchmark."""

    # Example configuration
    config = {
        'synthetic_benchmark': {
            'base_time_per_token_ms': 0.2,
            'base_tpot_ms': 8.0,
            'variability_factor': 0.1
        }
    }

    # Create synthetic benchmark
    benchmark = SyntheticBenchmark(config)

    # Test different scenarios
    print("=== Testing Synthetic Benchmark ===")

    # Scenario 1: Prefill only
    print("\n1. Testing Prefill Phase")
    prefill_result = benchmark.simulate_prefill(prompt_len=512)
    print(f"Prefill result: {prefill_result}")

    # Scenario 2: Decode only
    print("\n2. Testing Decode Phase")
    decode_result = benchmark.simulate_decode(output_len=128, prompt_len=128)
    print(f"Decode result: {decode_result}")

    # Scenario 3: Full inference
    print("\n3. Testing Full Inference")
    workload = {
        'batch_size': 1,
        'prompt_len': 512,
        'output_len': 128,
        'phase': 'mixed'
    }
    frequency = {
        'gpu_freq': 846,
        'cpu_freq': 1479,
        'emc_freq': 1600
    }

    full_result = benchmark.run_benchmark(workload, frequency)
    print(f"Full inference result: {json.dumps(full_result, indent=2)}")

    # Scenario 4: Different frequencies
    print("\n4. Testing Different Frequencies")
    for gpu_freq in [378, 846, 1428]:
        freq = {
            'gpu_freq': gpu_freq,
            'cpu_freq': 1479,
            'emc_freq': 1600
        }
        result = benchmark.run_benchmark(workload, freq)
        print(f"GPU={gpu_freq}MHz: TTFT={result['ttft_ms']:.2f}ms, TPOT={result['tpot_ms']:.2f}ms, Energy={result['energy_per_token_j']:.3f}J/token")

    print("\n=== Synthetic Benchmark Tests Completed ===")


if __name__ == "__main__":
    main()