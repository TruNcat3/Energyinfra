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
                             output_length: int = 128,
                             benchmark_mode: bool = False) -> Dict:
        """
        Run a single inference and capture detailed timing.

        Args:
            prompt_length: Target prompt token count
            output_length: Target output token count
            benchmark_mode: If True, use deterministic generation (temp=0,
                repeat_penalty=1.0) and suppress EOS via logit_bias to ensure
                the model generates exactly output_length tokens.

        Returns dict with:
            ttft_ms, tpot_ms, total_time_ms, tokens_per_second, output_tokens,
            total_energy_j, avg_power_w, max_power_w, temperature_c,
            benchmark_mode, is_complete_output, completion_ratio
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        prompt = self._generate_prompt(prompt_length)

        # Generation params
        if benchmark_mode:
            gen_params = {
                'temperature': 0.0,
                'top_p': 1.0,
                'repeat_penalty': 1.0,
                'stop': [],
                'logit_bias': {self.model.token_eos(): -100},
            }
        else:
            gen_params = {
                'temperature': 0.7,
            }

        start_time = time.monotonic()
        first_token_time = None
        token_times: List[float] = []
        output_tokens = 0

        try:
            for chunk in self.model.create_completion(
                prompt,
                max_tokens=output_length,
                stream=True,
                echo=False,
                **gen_params
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
            'benchmark_mode': benchmark_mode,
            'is_complete_output': output_tokens >= output_length * 0.9,
            'completion_ratio': output_tokens / output_length if output_length > 0 else 0,
            # Energy fields filled by MetricsCollector integration
            'total_energy_j': 0.0,
            'avg_power_w': 0.0,
            'max_power_w': 0.0,
            'temperature_c': 0.0
        }

        logger.info(f"Inference: TTFT={ttft_ms:.1f}ms, TPOT={tpot_ms:.1f}ms, "
                    f"tokens={output_tokens}/{output_length}, tps={tokens_per_second:.1f}"
                    f"{' [BM]' if benchmark_mode else ''}")

        return result

    def run_phase_split_inference(self, prompt_length: int = 512,
                                   output_length: int = 128,
                                   phase_switch_callback=None) -> Dict:
        """
        Run inference with phase boundary detection and optional frequency switch.

        After the first token (end of prefill), calls phase_switch_callback()
        which can switch GPU/EMC/CPU frequencies. Returns separate metrics
        for prefill and decode phases.

        Args:
            prompt_length: Target prompt token count
            output_length: Target output token count
            phase_switch_callback: Callable invoked at prefill→decode boundary.
                Receives dict with 'prefill_time_ms', 'prompt_tokens'.

        Returns:
            Dict with prefill_* and decode_* metrics plus combined totals.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        prompt = self._generate_prompt(prompt_length)

        start_time = time.monotonic()
        first_token_time = None
        token_times: List[float] = []
        output_tokens = 0
        switch_done = False
        switch_overhead_ms = 0.0

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
                    if first_token_time is None:
                        first_token_time = now
                        # Phase boundary: prefill complete
                        if phase_switch_callback and not switch_done:
                            preflight_info = {
                                'prefill_time_ms': (now - start_time) * 1000,
                                'prompt_tokens': len(prompt) // 4,
                            }
                            switch_start = time.monotonic()
                            phase_switch_callback(preflight_info)
                            switch_overhead_ms = (time.monotonic() - switch_start) * 1000
                            switch_done = True
                    token_times.append(now)
                    output_tokens += 1
        except Exception as e:
            logger.error(f"Phase-split inference failed: {e}")
            return {
                'ttft_ms': 0, 'tpot_ms': 0, 'total_time_ms': 0,
                'tokens_per_second': 0, 'output_tokens': 0,
                'prefill_time_ms': 0, 'decode_time_ms': 0,
                'switch_overhead_ms': 0, 'error': str(e)
            }

        end_time = time.monotonic()

        # Calculate combined metrics
        total_time_ms = (end_time - start_time) * 1000
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else total_time_ms

        if len(token_times) >= 2:
            inter_token_times = [
                (token_times[i+1] - token_times[i]) * 1000
                for i in range(len(token_times) - 1)
            ]
            tpot_ms = np.mean(inter_token_times)
        else:
            tpot_ms = total_time_ms / max(output_tokens, 1)

        tokens_per_second = output_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

        # Decode phase metrics (tokens after first)
        decode_start = first_token_time if first_token_time else start_time
        decode_time_ms = (end_time - decode_start) * 1000
        decode_tpot_ms = tpot_ms  # Same as combined since only decode tokens

        result = {
            'ttft_ms': float(ttft_ms),
            'tpot_ms': float(tpot_ms),
            'total_time_ms': float(total_time_ms),
            'tokens_per_second': float(tokens_per_second),
            'output_tokens': output_tokens,
            'prompt_tokens': len(prompt) // 4,
            # Phase-split metrics
            'prefill_time_ms': float(ttft_ms),
            'decode_time_ms': float(decode_time_ms),
            'decode_tokens': output_tokens - 1 if output_tokens > 1 else 0,
            'switch_overhead_ms': float(switch_overhead_ms),
            # Energy fields filled by caller
            'total_energy_j': 0.0,
            'prefill_energy_j': 0.0,
            'decode_energy_j': 0.0,
            'avg_power_w': 0.0,
            'max_power_w': 0.0,
            'temperature_c': 0.0
        }

        logger.info(f"Phase-split: TTFT={ttft_ms:.1f}ms, TPOT={tpot_ms:.1f}ms, "
                    f"switch={switch_overhead_ms:.1f}ms, tokens={output_tokens}")
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
