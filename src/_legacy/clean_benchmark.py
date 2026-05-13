#!/usr/bin/env python3
"""
Clean llama.cpp benchmark - outputs only CSV to stdout
纯净的llama.cpp基准测试 - 只输出CSV到标准输出
"""

import sys
import os
import time

# Redirect all print statements to stderr
original_print = print
def print_to_stderr(*args, **kwargs):
    kwargs['file'] = sys.stderr
    original_print(*args, **kwargs)

print = print_to_stderr

# Add jetson_llm_env to path
jetson_env_path = "/home/wt/work/Energyinfra/jetson_llm_env/lib/python3.10/site-packages"
if jetson_env_path not in sys.path:
    sys.path.insert(0, jetson_env_path)

def run_benchmark(model_path, prompt, max_tokens=20):
    """Run simple benchmark"""

    try:
        import llama_cpp

        # Suppress llama.cpp warnings
        import warnings
        warnings.filterwarnings('ignore')

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

        # Output CSV to stdout (this will be captured by shell script)
        sys.stdout.write(f"{load_time:.3f},{inference_time:.3f},{tokens},{tps:.2f},{generated_text[:20]}\n")
        sys.stdout.flush()

        return True

    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.stdout.write(f"ERROR,{str(e)}\n")
        sys.stdout.flush()
        return False

if __name__ == "__main__":
    model_path = "/home/wt/work/Energyinfra/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"
    prompt = "What is 2+2? Give brief answer."

    success = run_benchmark(model_path, prompt)
    sys.exit(0 if success else 1)