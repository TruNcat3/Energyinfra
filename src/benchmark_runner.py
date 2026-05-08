#!/usr/bin/env python3
"""
Benchmark Runner for LLM Inference
Executes LLM benchmarks and parses performance metrics.
"""

import subprocess
import logging
import time
import re
from typing import Dict, Optional, List
from pathlib import Path
import json
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for executing LLM benchmarks and parsing results.

    Supports multiple runtimes:
    - TensorRT-LLM (primary)
    - llama.cpp (extensible)
    - vLLM (extensible)
    """

    def __init__(self, config: Dict, output_dir: str = "data/raw_logs"):
        """
        Initialize benchmark runner.

        Args:
            config: Configuration dictionary
            output_dir: Directory for log files
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Benchmark template from config
        self.benchmark_config = config.get('benchmark_template', {})
        self.runtime = self.benchmark_config.get('runtime', 'tensorrt_llm')
        self.command_template = self.benchmark_config.get('command_template', '')

        # TensorRT-LLM specific config
        self.trt_llm_config = self.benchmark_config.get('tensorrt_llm', {})

        # Store run results
        self.run_results = []

    def generate_command(self, workload: Dict, frequency: Dict) -> List[str]:
        """
        Generate benchmark command based on workload and frequency configuration.

        Args:
            workload: Workload configuration dict (model, batch_size, prompt_len, etc.)
            frequency: Frequency configuration dict (gpu_freq, cpu_freq, emc_freq)

        Returns:
            List of command parts for subprocess
        """
        try:
            # Extract workload parameters
            model = workload.get('model', 'qwen-7b-int4')
            model_config = self._get_model_config(model)

            batch_size = workload.get('batch_size', 1)
            prompt_len = workload.get('prompt_len', 512)
            output_len = workload.get('output_len', 128)
            phase = workload.get('phase', 'mixed')
            warmup_runs = workload.get('warmup_runs', 1)
            repeats = workload.get('repeats', 5)

            # Extract frequency parameters
            gpu_freq = frequency.get('gpu_freq', 'high')
            cpu_freq = frequency.get('cpu_freq', 'high')
            emc_freq = frequency.get('emc_freq', 'high')

            # Convert frequency presets to actual values
            gpu_freq_value = self._get_frequency_value('gpu', gpu_freq)
            cpu_freq_value = self._get_frequency_value('cpu', cpu_freq)
            emc_freq_value = self._get_frequency_value('emc', emc_freq)

            # Generate command using template
            if self.runtime == 'tensorrt_llm':
                return self._generate_trt_llm_command(
                    model_config, batch_size, prompt_len, output_len,
                    phase, warmup_runs, repeats,
                    gpu_freq_value, cpu_freq_value, emc_freq_value
                )
            else:
                logger.error(f"Unsupported runtime: {self.runtime}")
                return []

        except Exception as e:
            logger.error(f"Failed to generate command: {e}")
            return []

    def _get_model_config(self, model_name: str) -> Dict:
        """
        Get model configuration from config.

        Args:
            model_name: Name of the model

        Returns:
            Model configuration dict
        """
        models_config = self.config.get('models', [])
        for model in models_config:
            if model.get('name') == model_name:
                return model

        # Return default config if not found
        logger.warning(f"Model config not found for {model_name}, using default")
        return {
            'name': model_name,
            'engine_path': f"/path/to/{model_name}.engine",
            'params': '7B',
            'quantization': 'int4'
        }

    def _get_frequency_value(self, target: str, freq_preset: str) -> Optional[int]:
        """
        Convert frequency preset to actual frequency value.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')
            freq_preset: Frequency preset ('low', 'mid', 'high') or integer

        Returns:
            Frequency in MHz, or None if invalid
        """
        try:
            # If already an integer, return it
            if isinstance(freq_preset, int):
                return freq_preset

            # Try to convert string to integer
            try:
                return int(freq_preset)
            except ValueError:
                pass

            # Look up preset value from frequencies config
            frequencies_config = self.config.get('frequencies', {})
            if target in frequencies_config and 'presets' in frequencies_config[target]:
                presets = frequencies_config[target]['presets']
                if freq_preset in presets:
                    return presets[freq_preset]

            logger.warning(f"Could not find frequency value for {target}:{freq_preset}")
            return None

        except Exception as e:
            logger.error(f"Error getting frequency value: {e}")
            return None

    def _generate_trt_llm_command(self, model_config: Dict, batch_size: int,
                                  prompt_len: int, output_len: int, phase: str,
                                  warmup_runs: int, repeats: int,
                                  gpu_freq: int, cpu_freq: int, emc_freq: int) -> List[str]:
        """
        Generate TensorRT-LLM benchmark command.

        Args:
            model_config: Model configuration
            batch_size: Batch size
            prompt_len: Prompt length in tokens
            output_len: Output length in tokens
            phase: Inference phase
            warmup_runs: Number of warmup runs
            repeats: Number of measurement runs
            gpu_freq: GPU frequency in MHz
            cpu_freq: CPU frequency in MHz
            emc_freq: EMC frequency in MHz

        Returns:
            List of command parts
        """
        # Base command
        command = [
            'python', 'run_benchmark.py',
            '--model', model_config.get('name'),
            '--engine', model_config.get('engine_path'),
            '--batch-size', str(batch_size),
            '--input-len', str(prompt_len),
            '--output-len', str(output_len),
            '--warmup', str(warmup_runs),
            '--repeats', str(repeats),
            '--gpu-freq', str(gpu_freq),
            '--cpu-freq', str(cpu_freq),
            '--emc-freq', str(emc_freq)
        ]

        # TensorRT-LLM specific options
        if self.trt_llm_config.get('use_fp16'):
            command.append('--use-fp16')
        if self.trt_llm_config.get('use_graph'):
            command.append('--use-graph')

        # Phase-specific options
        if phase == 'prefill':
            command.append('--prefill-only')
        elif phase == 'decode':
            command.append('--decode-only')

        # Additional options from config
        num_beams = self.trt_llm_config.get('num_beams', 1)
        if num_beams > 1:
            command.extend(['--num-beams', str(num_beams)])

        return command

    def run_benchmark(self, workload: Dict, frequency: Dict,
                     timeout_sec: int = 600) -> Optional[Dict]:
        """
        Run a single benchmark and parse results.

        Args:
            workload: Workload configuration
            frequency: Frequency configuration
            timeout_sec: Timeout in seconds

        Returns:
            Dictionary with benchmark results, or None if failed
        """
        # Generate command
        command = self.generate_command(workload, frequency)
        if not command:
            logger.error("Could not generate benchmark command")
            return None

        # Generate log filename
        timestamp = int(time.time())
        log_filename = f"benchmark_{timestamp}.log"
        log_path = self.output_dir / log_filename

        logger.info(f"Running benchmark: {' '.join(command)}")
        logger.info(f"Logging to: {log_path}")

        try:
            # Record start time
            start_time = time.time()

            # Run benchmark
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=self.output_dir.parent  # Run from project root
            )

            # Record end time
            end_time = time.time()
            total_time = end_time - start_time

            # Save logs
            with open(log_path, 'w') as f:
                f.write(f"Command: {' '.join(command)}\n")
                f.write(f"Start time: {start_time}\n")
                f.write(f"End time: {end_time}\n")
                f.write(f"Total time: {total_time:.2f}s\n\n")
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)

            if result.returncode != 0:
                logger.error(f"Benchmark failed with return code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return None

            # Parse results
            parsed_results = self.parse_output(result.stdout, result.stderr)

            if parsed_results:
                parsed_results['total_time'] = total_time
                parsed_results['start_time'] = start_time
                parsed_results['end_time'] = end_time
                parsed_results['log_file'] = str(log_path)
                parsed_results['command'] = ' '.join(command)

                # Add workload and frequency info
                parsed_results['workload'] = workload.copy()
                parsed_results['frequency'] = frequency.copy()

                logger.info(f"Benchmark completed successfully")
                logger.info(f"Results: {parsed_results}")

                self.run_results.append(parsed_results)
                return parsed_results
            else:
                logger.error("Failed to parse benchmark output")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"Benchmark timed out after {timeout_sec} seconds")
            return None
        except Exception as e:
            logger.error(f"Error running benchmark: {e}")
            return None

    def parse_output(self, stdout: str, stderr: str) -> Optional[Dict]:
        """
        Parse benchmark output to extract metrics.

        Args:
            stdout: Standard output from benchmark
            stderr: Standard error from benchmark

        Returns:
            Dictionary with parsed metrics, or None if parsing failed
        """
        try:
            output = stdout + stderr
            results = {}

            # Parse based on runtime
            if self.runtime == 'tensorrt_llm':
                return self._parse_trt_llm_output(output)
            else:
                logger.error(f"Unsupported runtime: {self.runtime}")
                return None

        except Exception as e:
            logger.error(f"Error parsing output: {e}")
            return None

    def _parse_trt_llm_output(self, output: str) -> Optional[Dict]:
        """
        Parse TensorRT-LLM benchmark output.

        Args:
            output: Combined stdout/stderr from benchmark

        Returns:
            Dictionary with parsed metrics
        """
        results = {}

        # Common patterns to extract from TensorRT-LLM output
        # These patterns may need adjustment based on actual TensorRT-LLM output format

        # Time to First Token (TTFT)
        ttft_patterns = [
            r'Time to first token[:\s]+(\d+\.?\d*)\s*ms',
            r'TTFT[:\s]+(\d+\.?\d*)\s*ms',
            r'First token latency[:\s]+(\d+\.?\d*)\s*ms',
            r'latency.*first.*token[:\s]+(\d+\.?\d*)\s*ms'
        ]

        for pattern in ttft_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results['ttft_ms'] = float(match.group(1))
                logger.info(f"Extracted TTFT: {results['ttft_ms']:.2f}ms")
                break

        # Time Per Output Token (TPOT)
        tpot_patterns = [
            r'Time per output token[:\s]+(\d+\.?\d*)\s*ms',
            r'TPOT[:\s]+(\d+\.?\d*)\s*ms',
            r'Output token latency[:\s]+(\d+\.?\d*)\s*ms',
            r'average.*time.*per.*token[:\s]+(\d+\.?\d*)\s*ms'
        ]

        for pattern in tpot_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results['tpot_ms'] = float(match.group(1))
                logger.info(f"Extracted TPOT: {results['tpot_ms']:.2f}ms")
                break

        # Inter-Token Latency (ITL)
        itl_patterns = [
            r'Inter-token latency[:\s]+(\d+\.?\d*)\s*ms',
            r'ITL[:\s]+(\d+\.?\d*)\s*ms',
            r'token.*time.*std[:\s]+(\d+\.?\d*)\s*ms'
        ]

        for pattern in itl_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results['itl_ms'] = float(match.group(1))
                logger.info(f"Extracted ITL: {results['itl_ms']:.2f}ms")
                break

        # Throughput (tokens per second)
        tps_patterns = [
            r'Tokens per second[:\s]+(\d+\.?\d*)',
            r'TPS[:\s]+(\d+\.?\d*)',
            r'Throughput[:\s]+(\d+\.?\d*)\s*tokens/s',
            r'tokens.*per.*second[:\s]+(\d+\.?\d*)'
        ]

        for pattern in tps_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results['throughput'] = float(match.group(1))
                logger.info(f"Extracted Throughput: {results['throughput']:.2f} tokens/s")
                break

        # Latency distribution (if available)
        latency_patterns = [
            r'P50.*latency[:\s]+(\d+\.?\d*)\s*ms',
            r'P95.*latency[:\s]+(\d+\.?\d*)\s*ms',
            r'P99.*latency[:\s]+(\d+\.?\d*)\s*ms'
        ]

        for pattern in latency_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                percentile = pattern.split('P')[1].split('*')[0] if 'P' in pattern else 'unknown'
                results[f'latency_p{percentile}_ms'] = float(match.group(1))

        # GPU utilization and memory
        gpu_patterns = [
            r'GPU utilization[:\s]+(\d+\.?\d*)\s*%',
            r'GPU.*used.*memory[:\s]+(\d+\.?\d*)\s*GB'
        ]

        for pattern in gpu_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                if 'utilization' in pattern.lower():
                    results['gpu_utilization_percent'] = float(matches[0])
                if 'memory' in pattern.lower():
                    results['gpu_memory_used_gb'] = float(matches[0])

        # Total output tokens (for energy calculations)
        output_tokens_patterns = [
            r'Total output tokens[:\s]+(\d+)',
            r'Output tokens[:\s]+(\d+)',
            r'Generated.*tokens[:\s]+(\d+)'
        ]

        for pattern in output_tokens_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                results['output_tokens'] = int(match.group(1))
                logger.info(f"Extracted output tokens: {results['output_tokens']}")

        return results if results else None

    def run_warmup(self, workload: Dict, frequency: Dict,
                   warmup_runs: int = 1, timeout_sec: int = 300) -> bool:
        """
        Run warmup benchmarks to initialize system state.

        Args:
            workload: Workload configuration
            frequency: Frequency configuration
            warmup_runs: Number of warmup runs
            timeout_sec: Timeout per run

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Running {warmup_runs} warmup run(s)...")

        for i in range(warmup_runs):
            logger.info(f"Warmup run {i+1}/{warmup_runs}")

            result = self.run_benchmark(workload, frequency, timeout_sec)

            if result is None:
                logger.error(f"Warmup run {i+1} failed")
                return False

            # Small delay between warmup runs
            if i < warmup_runs - 1:
                time.sleep(5)

        logger.info("Warmup completed successfully")
        return True

    def run_measurement(self, workload: Dict, frequency: Dict,
                     measurement_runs: int = 5, timeout_sec: int = 600) -> List[Dict]:
        """
        Run measurement benchmarks and collect results.

        Args:
            workload: Workload configuration
            frequency: Frequency configuration
            measurement_runs: Number of measurement runs
            timeout_sec: Timeout per run

        Returns:
            List of measurement results
        """
        logger.info(f"Running {measurement_runs} measurement run(s)...")

        results = []
        successful_runs = 0

        for i in range(measurement_runs):
            logger.info(f"Measurement run {i+1}/{measurement_runs}")

            result = self.run_benchmark(workload, frequency, timeout_sec)

            if result:
                results.append(result)
                successful_runs += 1
            else:
                logger.error(f"Measurement run {i+1} failed")

            # Delay between measurement runs
            if i < measurement_runs - 1:
                time.sleep(2)

        logger.info(f"Measurement completed: {successful_runs}/{measurement_runs} successful runs")

        if successful_runs == 0:
            logger.error("All measurement runs failed")
            return []

        return results

    def calculate_statistics(self, results: List[Dict]) -> Dict:
        """
        Calculate statistics from multiple benchmark results.

        Args:
            results: List of benchmark result dictionaries

        Returns:
            Dictionary with calculated statistics
        """
        if not results:
            return {}

        stats = {}

        # Calculate statistics for numeric fields
        numeric_fields = ['ttft_ms', 'tpot_ms', 'itl_ms', 'throughput']

        for field in numeric_fields:
            values = [r.get(field) for r in results if field in r and r.get(field) is not None]

            if values:
                stats[f'{field}_mean'] = statistics.mean(values)
                stats[f'{field}_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
                stats[f'{field}_min'] = min(values)
                stats[f'{field}_max'] = max(values)
                stats[f'{field}_median'] = statistics.median(values)

                # Calculate coefficient of variation
                if stats[f'{field}_mean'] > 0:
                    stats[f'{field}_cv'] = (stats[f'{field}_std'] / stats[f'{field}_mean']) * 100

        # Add summary info
        stats['total_runs'] = len(results)
        stats['successful_runs'] = sum(1 for r in results if r.get('ttft_ms') is not None)
        stats['success_rate'] = stats['successful_runs'] / len(results) if results else 0.0

        return stats

    def save_results(self, results: List[Dict], filename: Optional[str] = None) -> Optional[str]:
        """
        Save benchmark results to file.

        Args:
            results: List of benchmark results
            filename: Optional filename. If None, auto-generate.

        Returns:
            Path to saved file, or None if failed
        """
        if not results:
            logger.warning("No results to save")
            return None

        if filename is None:
            timestamp = int(time.time())
            filename = f"benchmark_results_{timestamp}.json"

        output_path = self.output_dir / filename

        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Results saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None


def main():
    """
    Test function for benchmark runner.
    """
    # Example configuration
    config = {
        'benchmark_template': {
            'runtime': 'tensorrt_llm',
            'command_template': 'python run_benchmark.py --model {model} --engine {engine_path} --batch-size {batch_size} --input-len {prompt_len} --output-len {output_len}',
            'tensorrt_llm': {
                'use_fp16': True,
                'use_graph': True,
                'num_beams': 1
            }
        },
        'models': [
            {
                'name': 'qwen-7b-int4',
                'engine_path': '/path/to/qwen-7b-int4.engine',
                'params': '7B',
                'quantization': 'int4'
            }
        ],
        'frequencies': {
            'gpu': {
                'presets': {
                    'low': 378,
                    'mid': 846,
                    'high': 1428
                }
            },
            'cpu': {
                'presets': {
                    'low': 1020,
                    'mid': 1479,
                    'high': 2015
                }
            },
            'emc': {
                'presets': {
                    'low': 133,
                    'mid': 1600,
                    'high': 2133
                }
            }
        }
    }

    # Create runner
    runner = BenchmarkRunner(config)

    # Example workload
    workload = {
        'model': 'qwen-7b-int4',
        'batch_size': 1,
        'prompt_len': 512,
        'output_len': 128,
        'phase': 'mixed',
        'warmup_runs': 1,
        'repeats': 3
    }

    # Example frequency configuration
    frequency = {
        'gpu_freq': 'high',
        'cpu_freq': 'high',
        'emc_freq': 'high'
    }

    logger.info("Testing benchmark runner with example workload...")
    logger.info(f"Workload: {workload}")
    logger.info(f"Frequency: {frequency}")

    # Run a single benchmark (this will fail without actual TensorRT-LLM setup)
    result = runner.run_benchmark(workload, frequency, timeout_sec=300)

    if result:
        logger.info(f"Benchmark result: {result}")
    else:
        logger.info("Benchmark failed (expected without actual TensorRT-LLM setup)")


if __name__ == "__main__":
    main()