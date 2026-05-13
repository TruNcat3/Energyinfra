#!/usr/bin/env python3
"""
llama.cpp Inference Runner
Runs real LLM inference using llama-cpp-python, capturing per-token timing
for TTFT/TPOT extraction and energy metrics.
"""

import time
import logging
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class LlamaCppRunner:
    """Run llama.cpp inference with detailed per-token timing."""

    def __init__(self, model_path: str, n_gpu_layers: int = -1,
                 n_ctx: int = 4096, n_threads: int = 4):
        self.model_path = Path(model_path)
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.n_threads = n_threads

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the GGUF model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
            logger.info(f"Loading model: {self.model_path}")
            self.model = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False
            )
            logger.info("Model loaded successfully")
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed. "
                "Install with: pip install llama-cpp-python"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _generate_prompt(self, target_tokens: int) -> str:
        """Generate a prompt of approximately target_tokens length."""
        # Use repetitive padding to reach target token count
        # ~1 token per 4 characters for English text
        base = "The following is a discussion about technology and science. "
        repeats = max(1, (target_tokens * 4) // len(base))
        prompt = (base * repeats)[:target_tokens * 4]
        return prompt

    def run_single_inference(self, prompt_length: int = 512,
                             output_length: int = 128) -> Dict:
        """
        Run a single inference and capture detailed timing.

        Returns dict compatible with SyntheticBenchmark.run_benchmark() format:
            ttft_ms, tpot_ms, total_time_ms, tokens_per_second, output_tokens,
            total_energy_j, avg_power_w, max_power_w, temperature_c
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        prompt = self._generate_prompt(prompt_length)

        start_time = time.monotonic()
        first_token_time = None
        token_times: List[float] = []
        output_tokens = 0

        # Use streaming to capture per-token timing
        try:
            for chunk in self.model.create_completion(
                prompt,
                max_tokens=output_length,
                temperature=0.7,
                stream=True,
                echo=False
            ):
                now = time.monotonic()
                delta = chunk['choices'][0].get('text', '')
                if delta:
                    token_times.append(now)
                    output_tokens += 1
                    if first_token_time is None:
                        first_token_time = now
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                'ttft_ms': 0, 'tpot_ms': 0, 'total_time_ms': 0,
                'tokens_per_second': 0, 'output_tokens': 0,
                'total_energy_j': 0, 'avg_power_w': 0,
                'max_power_w': 0, 'temperature_c': 0,
                'error': str(e)
            }

        end_time = time.monotonic()

        # Calculate metrics
        total_time_ms = (end_time - start_time) * 1000

        if first_token_time is not None:
            ttft_ms = (first_token_time - start_time) * 1000
        else:
            ttft_ms = total_time_ms

        if len(token_times) >= 2:
            # TPOT: average inter-token time
            inter_token_times = [
                (token_times[i+1] - token_times[i]) * 1000
                for i in range(len(token_times) - 1)
            ]
            tpot_ms = np.mean(inter_token_times)
        else:
            tpot_ms = total_time_ms / max(output_tokens, 1)

        tokens_per_second = output_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

        result = {
            'ttft_ms': float(ttft_ms),
            'tpot_ms': float(tpot_ms),
            'total_time_ms': float(total_time_ms),
            'tokens_per_second': float(tokens_per_second),
            'output_tokens': output_tokens,
            'prompt_tokens': len(prompt) // 4,  # Approximate
            # Energy fields filled by MetricsCollector integration
            'total_energy_j': 0.0,
            'avg_power_w': 0.0,
            'max_power_w': 0.0,
            'temperature_c': 0.0
        }

        logger.info(f"Inference: TTFT={ttft_ms:.1f}ms, TPOT={tpot_ms:.1f}ms, "
                    f"tokens={output_tokens}, tps={tokens_per_second:.1f}")

        return result

    def run_with_metrics(self, prompt_length: int, output_length: int,
                         metrics_collector=None) -> Dict:
        """Run inference with optional tegrastats energy measurement."""
        if metrics_collector:
            metrics_collector.start_collection()

        result = self.run_single_inference(prompt_length, output_length)

        if metrics_collector:
            time.sleep(0.5)  # Let final tegrastats sample arrive
            metrics_collector.stop_collection()
            energy_data = metrics_collector.parse_last_run()
            if energy_data is not None and not energy_data.empty:
                result['total_energy_j'] = float(energy_data.get('energy_j', 0))
                result['avg_power_w'] = float(energy_data.get('avg_power_w', 0))
                result['max_power_w'] = float(energy_data.get('max_power_w', 0))
                result['temperature_c'] = float(energy_data.get('temperature_c', 0))

        return result

    def warmup(self, n_runs: int = 2):
        """Run warmup inferences to stabilize GPU/CPU caches."""
        logger.info(f"Running {n_runs} warmup inferences...")
        for i in range(n_runs):
            self.run_single_inference(prompt_length=128, output_length=32)
        logger.info("Warmup complete")
