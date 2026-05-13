#!/usr/bin/env python3
"""
Frequency Benchmark for llama.cpp
在不同GPU频率下测试llama.cpp性能
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add jetson_llm_env to path
jetson_env_path = "/home/wt/work/Energyinfra/jetson_llm_env/lib/python3.10/site-packages"
if jetson_env_path not in sys.path:
    sys.path.insert(0, jetson_env_path)

def get_current_frequencies():
    """获取当前系统频率（需要root权限或预先设置）"""

    frequencies = {
        'gpu_mhz': 0,
        'cpu_mhz': [],
        'emc_mhz': 0,
        'timestamp': datetime.now().isoformat()
    }

    try:
        # GPU frequency
        with open('/sys/class/devfreq/17000000.gpu/cur_freq', 'r') as f:
            frequencies['gpu_mhz'] = int(f.read().strip()) // 1000000

        # CPU frequencies
        for cpu_path in Path('/sys/devices/system/cpu').glob('cpu[0-9]*/cpufreq/scaling_cur_freq'):
            cpu_num = cpu_path.parent.name
            with open(cpu_path, 'r') as f:
                freq_mhz = int(f.read().strip()) // 1000
                frequencies['cpu_mhz'].append({
                    'cpu': cpu_num,
                    'freq_mhz': freq_mhz
                })

        # EMC frequency
        emc_path = Path('/sys/class/devfreq/17000000.emc/cur_freq')
        if emc_path.exists():
            with open(emc_path, 'r') as f:
                frequencies['emc_mhz'] = int(f.read().strip()) // 1000000

    except Exception as e:
        print(f"Warning: Could not read frequencies: {e}")

    return frequencies

def run_llama_benchmark(model_path, prompt, max_tokens=50):
    """运行llama.cpp基准测试"""

    try:
        import llama_cpp

        print(f"  Loading model...")
        start_time = time.time()

        model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=99,
            n_threads=4,
            verbose=False
        )

        load_time = time.time() - start_time

        print(f"  Running inference ({max_tokens} tokens)...")
        inference_start = time.time()

        output = model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            echo=False
        )

        inference_time = time.time() - inference_start

        # 提取性能指标
        generated_text = output['choices'][0]['text']
        token_count = output['usage']['completion_tokens'] if 'usage' in output else len(generated_text.split())

        return {
            'load_time_sec': load_time,
            'inference_time_sec': inference_time,
            'total_time_sec': load_time + inference_time,
            'generated_tokens': token_count,
            'tokens_per_second': token_count / inference_time if inference_time > 0 else 0,
            'generated_text': generated_text,
            'success': True
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'load_time_sec': 0,
            'inference_time_sec': 0,
            'total_time_sec': 0,
            'generated_tokens': 0,
            'tokens_per_second': 0
        }

def main():
    print("=" * 60)
    print("Frequency Benchmark for llama.cpp")
    print("=" * 60)

    model_path = "/home/wt/work/Energyinfra/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"

    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return 1

    # 测试提示词
    test_prompts = [
        ("What is the capital of France? Give a brief answer.", 30),
        ("Explain machine learning in one sentence.", 40),
        ("What are the primary colors? List them.", 25)
    ]

    results = []

    # 运行测试
    for i, (prompt, max_tokens) in enumerate(test_prompts, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_prompts)}: '{prompt[:50]}...'")
        print(f"{'='*60}")

        # 获取当前频率
        print("\n📊 Current System Frequencies:")
        freqs = get_current_frequencies()
        print(f"  GPU: {freqs['gpu_mhz']} MHz")

        if freqs['cpu_mhz']:
            cpu_avg = sum(c['freq_mhz'] for c in freqs['cpu_mhz']) / len(freqs['cpu_mhz'])
            print(f"  CPU: {cpu_avg:.0f} MHz (avg)")
            print(f"  CPU Cores: {len(freqs['cpu_mhz'])}")

        if freqs['emc_mhz'] > 0:
            print(f"  EMC: {freqs['emc_mhz']} MHz")

        # 运行基准测试
        print(f"\n🚀 Running Benchmark...")
        benchmark_result = run_llama_benchmark(model_path, prompt, max_tokens)

        if benchmark_result['success']:
            print(f"  ✅ Load time: {benchmark_result['load_time_sec']:.2f}s")
            print(f"  ✅ Inference time: {benchmark_result['inference_time_sec']:.2f}s")
            print(f"  ✅ Generated: {benchmark_result['generated_tokens']} tokens")
            print(f"  ✅ Performance: {benchmark_result['tokens_per_second']:.1f} tokens/s")
            print(f"  📝 Output: '{benchmark_result['generated_text'][:100]}...'")
        else:
            print(f"  ❌ Error: {benchmark_result['error']}")

        # 保存结果
        result = {
            'test_id': i,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'frequencies': freqs,
            'benchmark': benchmark_result,
            'timestamp': datetime.now().isoformat()
        }
        results.append(result)

        # 冷却时间
        if i < len(test_prompts):
            print(f"\n⏱️  Cooling for 5 seconds...")
            time.sleep(5)

    # 保存结果
    output_dir = Path("data/frequency_benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"frequency_benchmark_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'total_tests': len(results),
            'successful_tests': sum(1 for r in results if r['benchmark']['success']),
            'results': results
        }, f, indent=2)

    # 生成摘要
    print(f"\n{'='*60}")
    print("Benchmark Summary")
    print(f"{'='*60}")

    successful_results = [r for r in results if r['benchmark']['success']]

    if successful_results:
        gpu_freqs = [r['frequencies']['gpu_mhz'] for r in successful_results]
        perf_data = [(r['frequencies']['gpu_mhz'], r['benchmark']['tokens_per_second'])
                    for r in successful_results]

        print(f"\n📊 Performance by GPU Frequency:")
        for gpu_freq, tps in perf_data:
            print(f"  {gpu_freq} MHz: {tps:.1f} tokens/s")

        # 计算性能差异
        if len(perf_data) >= 2:
            min_perf = min(tps for _, tps in perf_data)
            max_perf = max(tps for _, tps in perf_data)
            improvement = ((max_perf - min_perf) / min_perf) * 100
            print(f"\n📈 Performance Range:")
            print(f"  Min: {min_perf:.1f} tokens/s")
            print(f"  Max: {max_perf:.1f} tokens/s")
            print(f"  Improvement: {improvement:.1f}%")

    print(f"\n💾 Results saved to: {output_file}")
    print(f"✅ Frequency benchmark completed!")

    return 0

if __name__ == "__main__":
    sys.exit(main())