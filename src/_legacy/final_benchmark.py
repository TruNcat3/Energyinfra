#!/usr/bin/env python3
"""
Final clean benchmark - guaranteed CSV output to stdout
最终纯净基准测试 - 保证CSV输出到标准输出
"""

import sys
import os
import time

# Add jetson_llm_env to path
jetson_env_path = "/home/wt/work/Energyinfra/jetson_llm_env/lib/python3.10/site-packages"
if jetson_env_path not in sys.path:
    sys.path.insert(0, jetson_env_path)

def run_benchmark(model_path, prompt, max_tokens=20):
    """Run simple benchmark"""

    try:
        import llama_cpp

        # Load model silently
        start_time = time.time()
        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=99,
            n_threads=4,
            verbose=False
        )
        load_time = time.time() - start_time

        # Run inference
        start_time = time.time()
        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )
        inference_time = time.time() - start_time

        # Extract results
        generated_text = output['choices'][0]['text']
        tokens = output['usage']['completion_tokens'] if 'usage' in output else len(generated_text.split())
        tps = tokens / inference_time if inference_time > 0 else 0

        # Clean output text for CSV
        clean_text = generated_text[:20].replace(',', ' ').replace('\n', ' ').strip()

        # Output CSV to stdout
        sys.stdout.write(f"{load_time:.3f},{inference_time:.3f},{tokens},{tps:.2f},{clean_text}\n")
        sys.stdout.flush()

        return True

    except Exception as e:
        sys.stdout.write(f"ERROR,{str(e)}\n")
        sys.stdout.flush()
        return False

if __name__ == "__main__":
    model_path = "/home/wt/work/Energyinfra/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"
    prompt = "What is 2+2? Give brief answer."

    success = run_benchmark(model_path, prompt)
    sys.exit(0 if success else 1)