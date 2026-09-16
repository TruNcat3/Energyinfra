#!/usr/bin/env python3
"""
Simple and reliable llama.cpp benchmark
简单可靠的llama.cpp基准测试
"""

import sys
import os
import time

# Add jetson_llm_env to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jetson_env_path = os.path.join(REPO_ROOT, "jetson_llm_env/lib/python3.10/site-packages")
if jetson_env_path not in sys.path:
    sys.path.insert(0, jetson_env_path)

def run_benchmark(model_path, prompt, max_tokens=20):
    """Run simple benchmark"""

    try:
        import llama_cpp

        # Suppress warnings
        import logging
        logging.getLogger('llama_cpp').setLevel(logging.ERROR)

        print(f"Loading model...", end='', flush=True)
        start_time = time.time()

        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=99,
            n_threads=4,
            verbose=False
        )

        load_time = time.time() - start_time
        print(f" {load_time:.2f}s")

        print(f"Running inference...", end='', flush=True)
        start_time = time.time()

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )

        inference_time = time.time() - start_time
        print(f" {inference_time:.2f}s")

        # Extract results
        generated_text = output['choices'][0]['text']
        tokens = output['usage']['completion_tokens'] if 'usage' in output else len(generated_text.split())
        tps = tokens / inference_time if inference_time > 0 else 0

        print(f"Generated: {tokens} tokens")
        print(f"Performance: {tps:.1f} tokens/s")
        print(f"Output: {generated_text[:50]}...")

        # Return CSV format for easy parsing (only this line will be captured)
        return f"{load_time:.3f},{inference_time:.3f},{tokens},{tps:.2f},{generated_text[:20]}"

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return f"ERROR,{str(e)}"

if __name__ == "__main__":
    model_path = os.path.join(REPO_ROOT, "models/gguf/Phi-3-mini-4k-instruct-q4.gguf")
    prompt = "What is 2+2? Give brief answer."

    result = run_benchmark(model_path, prompt)
    # Print result to stderr for display, and stdout for capture
    print(f"BENCHMARK_RESULT:{result}", file=sys.stderr)
    print(result)  # This line will be captured by shell script