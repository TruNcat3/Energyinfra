#!/usr/bin/env python3
"""
Baseline Performance Test for llama.cpp
llama.cpp基准性能测试（无需sudo权限）
"""

import sys
import os
import time
import subprocess
import json
from pathlib import Path

# Add jetson_llm_env to path (self-locating relative to this file)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
jetson_env_path = os.path.join(REPO_ROOT, "jetson_llm_env", "lib", "python3.10", "site-packages")
if jetson_env_path not in sys.path:
    sys.path.insert(0, jetson_env_path)

def check_file_size(filepath):
    """Check if file exists and has content"""
    if not os.path.exists(filepath):
        return False, "File not found"
    size = os.path.getsize(filepath)
    if size < 1000:  # Less than 1KB is likely empty
        return False, f"File too small: {size} bytes"
    return True, f"Size: {size / (1024*1024):.2f} MB"

def run_llama_cpp_benchmark(model_path, prompt="Hello, how are you?", max_tokens=50):
    """Run a simple llama.cpp benchmark using Python API"""

    try:
        import llama_cpp

        print(f"  Loading model: {model_path}")
        model_load_start = time.time()

        # Create Llama model with GPU layers
        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=512,  # Small context for faster test
            n_gpu_layers=99,  # Use GPU as much as possible
            n_threads=4,
            verbose=False
        )

        model_load_time = time.time() - model_load_start
        print(f"  Model loaded in {model_load_time:.2f}s")

        # Run inference
        print(f"  Running inference...")
        inference_start = time.time()

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False,
            seed=42  # For reproducibility
        )

        inference_time = time.time() - inference_start

        # Extract generated text
        generated_text = output['choices'][0]['text']

        # Debug: print raw output structure
        print(f"  Debug: output keys = {list(output.keys())}")
        print(f"  Debug: generated text length = {len(generated_text)}")
        if generated_text:
            print(f"  Debug: generated text preview = '{generated_text[:100]}'")

        # Try to get token count from different fields
        token_count = 0
        if 'usage' in output and 'completion_tokens' in output['usage']:
            token_count = output['usage']['completion_tokens']
        elif 'logprobs' in output['choices'][0] and output['choices'][0]['logprobs'] is not None:
            token_count = len(output['choices'][0]['logprobs'])
        else:
            # Fallback: estimate from text length (rough approximation)
            token_count = len(generated_text.split())

        # Calculate metrics
        metrics = {
            'model_load_time_sec': model_load_time,
            'inference_time_sec': inference_time,
            'total_time_sec': model_load_time + inference_time,
            'elapsed_time_sec': inference_time,  # For compatibility with main function
            'prompt': prompt,
            'max_tokens': max_tokens,
            'generated_tokens': token_count,
            'generated_text': generated_text,
            'tokens_per_second': token_count / inference_time if inference_time > 0 else 0,
            'output_length': len(generated_text),
            'raw_output': output  # Include raw output for debugging
        }

        print(f"  ✅ Generated {token_count} tokens in {inference_time:.2f}s ({metrics['tokens_per_second']:.1f} tokens/s)")

        return metrics, None

    except ImportError:
        return None, "llama_cpp package not available"
    except Exception as e:
        return None, f"Error running benchmark: {str(e)}"

def get_system_info():
    """Get basic system information without sudo"""
    info = {
        'platform': 'Jetson Orin',
        'python_version': sys.version,
        'llama_cpp_version': 'unknown'
    }

    # Try to get llama.cpp version
    try:
        import llama_cpp
        info['llama_cpp_version'] = llama_cpp.__version__
    except:
        pass

    # Check GPU info (no sudo needed for nvidia-smi)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            info['gpu_info'] = result.stdout.strip()
    except:
        info['gpu_info'] = 'nvidia-smi not available'

    return info

def main():
    print("=" * 60)
    print("llama.cpp Baseline Performance Test")
    print("=" * 60)

    # System information
    print("\n📊 System Information:")
    sys_info = get_system_info()
    for key, value in sys_info.items():
        print(f"  {key}: {value}")

    # Check model
    model_path = "models/gguf/Phi-3-mini-4k-instruct-q4.gguf"
    print(f"\n📁 Model Check: {model_path}")

    model_ok, model_info = check_file_size(model_path)
    if model_ok:
        print(f"  ✅ {model_info}")
    else:
        print(f"  ❌ {model_info}")
        print("  ⏳ Model might still be downloading...")
        return 1

    # Run benchmark
    print("\n🚀 Running Performance Benchmark...")
    print("  This may take 1-2 minutes...")

    test_prompts = [
        ("Hello, how are you today?", 20),
        ("What is 2+2? Give a brief answer.", 10),
        ("Explain what a computer is in one sentence.", 30)
    ]

    results = []
    for prompt, max_tokens in test_prompts:
        print(f"\n  Testing: '{prompt}' (max {max_tokens} tokens)")
        metrics, error = run_llama_cpp_benchmark(model_path, prompt, max_tokens)

        if error:
            print(f"    ❌ Error: {error}")
        else:
            print(f"    ✅ Completed in {metrics['elapsed_time_sec']:.2f}s")
            if metrics.get('has_performance_info'):
                print(f"    📊 Performance data available in output")
            results.append(metrics)

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Total tests: {len(test_prompts)}")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(test_prompts) - len(results)}")

    if results:
        avg_time = sum(r['elapsed_time_sec'] for r in results) / len(results)
        print(f"  Average time: {avg_time:.2f}s")
        print(f"  ✅ Baseline test completed successfully!")

        # Save results
        output_path = Path("data/baseline_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': sys_info,
            'model_path': model_path,
            'model_info': model_info,
            'results': results,
            'summary': {
                'total_tests': len(test_prompts),
                'successful': len(results),
                'average_time_sec': avg_time
            }
        }

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  📁 Results saved to: {output_path}")
        return 0
    else:
        print("  ❌ No successful tests")
        return 1

if __name__ == "__main__":
    sys.exit(main())