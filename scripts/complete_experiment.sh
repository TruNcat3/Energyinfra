#!/bin/bash
# Complete Frequency Experiment Script
# 在有sudo权限的终端中执行此脚本

echo "🚀 Starting Complete Frequency Experiment"
echo "=========================================="

# 模型路径
MODEL_PATH="/home/wt/work/Energyinfra/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"
OUTPUT_DIR="data/frequency_experiment_results"
mkdir -p "$OUTPUT_DIR"

# 频率配置
FREQUENCIES=("306000000" "918000000" "1300500000")
FREQ_NAMES=("306 MHz" "918 MHz" "1300 MHz")

# 结果文件
RESULT_FILE="$OUTPUT_DIR/complete_results_$(date +%Y%m%d_%H%M%S).txt"
JSON_FILE="$OUTPUT_DIR/complete_results_$(date +%Y%m%d_%H%M%S).json"

echo "📊 Experiment Configuration:" | tee "$RESULT_FILE"
echo "  Model: $MODEL_PATH" | tee -a "$RESULT_FILE"
echo "  Frequencies to test: ${FREQ_NAMES[*]}" | tee -a "$RESULT_FILE"
echo "  Results will be saved to: $OUTPUT_DIR" | tee -a "$RESULT_FILE"

# JSON开始
echo '{"experiment_start": "'$(date +%Y-%m-%d_%H:%M:%S)'", "results": [' > "$JSON_FILE"

for i in "${!FREQUENCIES[@]}"; do
    freq="${FREQUENCIES[$i]}"
    name="${FREQ_NAMES[$i]}"
    freq_mhz=$((freq / 1000000))

    echo ""
    echo "========================================" | tee -a "$RESULT_FILE"
    echo "Test $((i+1))/${#FREQUENCIES[@]}: $name" | tee -a "$RESULT_FILE"
    echo "========================================" | tee -a "$RESULT_FILE"

    # 设置频率
    echo "Setting GPU to $name..."
    echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/max_freq > /dev/null
    echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/min_freq > /dev/null

    # 等待频率稳定
    sleep 3

    # 验证频率
    actual_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
    actual_mhz=$((actual_freq / 1000000))
    temp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{print $1/1000}')

    echo "✅ Frequency set to: $name" | tee -a "$RESULT_FILE"
    echo "📊 Actual frequency: $actual_mhz MHz" | tee -a "$RESULT_FILE"
    echo "🌡️  Temperature: ${temp}°C" | tee -a "$RESULT_FILE"

    # 运行基准测试
    echo "🚀 Running benchmark at $name..." | tee -a "$RESULT_FILE"
    benchmark_result=$(python3 -c "
import sys
sys.path.insert(0, '/home/wt/work/Energyinfra/jetson_llm_env/lib/python3.10/site-packages')
import llama_cpp
import time

model_path = '$MODEL_PATH'
prompt = 'What is 2+2? Give a brief answer.'
max_tokens = 20

start_load = time.time()
model = llama_cpp.Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=99, n_threads=4, verbose=False)
load_time = time.time() - start_load

start_inference = time.time()
output = model(prompt, max_tokens=max_tokens, temperature=0.7, top_p=0.9, echo=False)
inference_time = time.time() - start_inference

tokens = output['usage']['completion_tokens'] if 'usage' in output else 0
tps = tokens / inference_time if inference_time > 0 else 0

print(f'{load_time:.2f},{inference_time:.2f},{tokens},{tps:.1f}')
" 2>&1)

    # 解析基准测试结果
    if [ $? -eq 0 ]; then
        IFS=',' read -r load_time inference_time tokens tps <<< "$benchmark_result"
        echo "📊 Benchmark Results:" | tee -a "$RESULT_FILE"
        echo "  Load time: ${load_time}s" | tee -a "$RESULT_FILE"
        echo "  Inference time: ${inference_time}s" | tee -a "$RESULT_FILE"
        echo "  Generated tokens: ${tokens}" | tee -a "$RESULT_FILE"
        echo "  Performance: ${tps} tokens/s" | tee -a "$RESULT_FILE"

        # 添加到JSON
        if [ $i -gt 0 ]; then
            echo "," >> "$JSON_FILE"
        fi
        echo '{"frequency_mhz": '$freq_mhz', "actual_mhz": '$actual_mhz', "temperature": '$temp', "load_time": '$load_time', "inference_time": '$inference_time', "tokens": '$tokens', "tokens_per_second": '$tps'}' >> "$JSON_FILE"
    else
        echo "❌ Benchmark failed: $benchmark_result" | tee -a "$RESULT_FILE"
    fi

    # 冷却
    echo "⏱️  Cooling for 5 seconds..."
    sleep 5
done

# JSON结束
echo ']}' >> "$JSON_FILE"

# 恢复默认设置
echo ""
echo "========================================" | tee -a "$RESULT_FILE"
echo "Restoring default settings..." | tee -a "$RESULT_FILE"
echo "========================================" | tee -a "$RESULT_FILE"

echo "1300500000" | sudo tee /sys/class/devfreq/17000000.gpu/max_freq > /dev/null
echo "306000000" | sudo tee /sys/class/devfreq/17000000.gpu/min_freq > /dev/null

sleep 2
final_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq | awk '{print $1/1000000 " MHz"}')
echo "✅ Restored to: $final_freq" | tee -a "$RESULT_FILE"

# 摘要
echo ""
echo "========================================" | tee -a "$RESULT_FILE"
echo "🎉 Experiment Completed!" | tee -a "$RESULT_FILE"
echo "========================================" | tee -a "$RESULT_FILE"
echo "📁 Results:" | tee -a "$RESULT_FILE"
echo "  Text: $RESULT_FILE" | tee -a "$RESULT_FILE"
echo "  JSON: $JSON_FILE" | tee -a "$RESULT_FILE"

echo ""
echo "📊 Performance Summary:"
echo "  Frequency  |  Tokens/s  |  Temp  |  Load Time"
echo "  -----------|------------|--------|-----------"
grep "Performance:" "$RESULT_FILE" | awk -F'Performance: ' '{print $2}' | awk -F' tokens/s' '{print $1 " tokens/s"}' > /tmp/tps.txt
grep "Temperature:" "$RESULT_FILE" | awk -F'Temperature: ' '{print $2}' | awk -F'°C' '{print $1}' > /tmp/temp.txt
grep "Load time:" "$RESULT_FILE" | awk -F'Load time: ' '{print $2}' | awk -F's' '{print $1}' > /tmp/load.txt

paste /tmp/tps.txt /tmp/temp.txt /tmp/load.txt | awk '{
    printf "  %-9s | %-10s | %-6s | %s\n", freq, tps, temp, load
}'

echo ""
echo "✅ You can now analyze the results!"