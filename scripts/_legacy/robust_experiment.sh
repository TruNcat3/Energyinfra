#!/bin/bash
# 改进的频率实验脚本 - 更好的错误处理和输出

echo "🚀 改进的频率实验脚本"
echo "=========================================="

# 检查sudo权限
echo "🔐 检查sudo权限..."
if ! sudo -n true 2>/dev/null; then
    echo "❌ 需要sudo权限来设置GPU频率"
    echo "   请先在终端中运行: sudo -v"
    echo "   然后再执行此脚本"
    exit 1
fi

echo "✅ Sudo权限确认"

# 创建输出目录
OUTPUT_DIR="data/frequency_experiment_results"
mkdir -p "$OUTPUT_DIR"
RESULT_FILE="$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S).txt"
JSON_FILE="$OUTPUT_DIR/results_$(date +%Y%m%d_%H%M%S).json"

# 实验配置
FREQUENCIES=("306000000" "612000000" "918000000" "1300500000")
FREQ_NAMES=("306 MHz" "612 MHz" "918 MHz" "1300 MHz")
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
MODEL_PATH="$REPO_ROOT/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"

echo "📊 实验配置:"
echo "  模型: $MODEL_PATH"
echo "  测试频率: ${FREQ_NAMES[*]}"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 开始记录结果
{
    echo "频率实验结果"
    echo "===================="
    echo "开始时间: $(date)"
    echo "模型: $MODEL_PATH"
    echo ""
} > "$RESULT_FILE"

# JSON开始
echo '{"experiment_info": {"start_time": "'$(date +%Y-%m-%d_%H:%M:%S)'", "model": "'"$MODEL_PATH"'"}, "results": [' > "$JSON_FILE"

success_count=0
total_tests=${#FREQUENCIES[@]}

for i in "${!FREQUENCIES[@]}"; do
    freq="${FREQUENCIES[$i]}"
    name="${FREQ_NAMES[$i]}"
    freq_mhz=$((freq / 1000000))

    echo ""
    echo "=========================================="
    echo "测试 $((i+1))/$total_tests: $name"
    echo "=========================================="

    # 添加到文本结果
    {
        echo "测试 $((i+1)): $name"
        echo "--------------------"
    } >> "$RESULT_FILE"

    # 设置GPU频率
    echo "🔄 设置GPU频率为 $name..."

    if echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/max_freq > /dev/null 2>&1; then
        echo "   ✅ 最大频率设置成功"
    else
        echo "   ❌ 最大频率设置失败"
        {
            echo "❌ 频率设置失败"
        } >> "$RESULT_FILE"
        continue
    fi

    if echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/min_freq > /dev/null 2>&1; then
        echo "   ✅ 最小频率设置成功"
    else
        echo "   ❌ 最小频率设置失败"
        {
            echo "❌ 频率设置失败"
        } >> "$RESULT_FILE"
        continue
    fi

    # 等待频率稳定
    sleep 3

    # 读取实际频率和温度
    actual_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
    actual_mhz=$((actual_freq / 1000000))
    temp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{print $1/1000}')

    echo "📊 实际频率: $actual_mhz MHz"
    echo "🌡️  当前温度: ${temp}°C"

    {
        echo "设置频率: $name"
        echo "实际频率: $actual_mhz MHz"
        echo "温度: ${temp}°C"
    } >> "$RESULT_FILE"

    # 运行基准测试
    echo "🚀 运行基准测试..."

    benchmark_output=$(python3 -c "
import sys
import os
import time

# 添加路径
sys.path.insert(0, os.environ.get('REPO_ROOT', '.') + '/jetson_llm_env/lib/python3.10/site-packages')

try:
    import llama_cpp

    model_path = '$MODEL_PATH'
    prompt = 'What is the capital of France? Give brief answer.'
    max_tokens = 25

    # 加载模型
    load_start = time.time()
    model = llama_cpp.Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=99, n_threads=4, verbose=False)
    load_time = time.time() - load_start

    # 运行推理
    infer_start = time.time()
    output = model(prompt, max_tokens=max_tokens, temperature=0.7, top_p=0.9, echo=False)
    infer_time = time.time() - infer_start

    # 提取结果
    tokens = output['usage']['completion_tokens'] if 'usage' in output else 0
    tps = tokens / infer_time if infer_time > 0 else 0
    text = output['choices'][0]['text'][:50]

    # 输出格式：load_time,infer_time,tokens,tps,text
    print(f'{load_time:.3f},{infer_time:.3f},{tokens},{tps:.2f},{text}')

except Exception as e:
    print(f'ERROR,{str(e)}')
    sys.exit(1)
" 2>&1)

    # 检查基准测试是否成功
    if echo "$benchmark_output" | grep -q "ERROR"; then
        echo "❌ 基准测试失败"
        error_msg=$(echo "$benchmark_output" | grep "ERROR" | cut -d',' -f2-)
        echo "   错误: $error_msg"

        {
            echo "基准测试: 失败"
            echo "错误: $error_msg"
        } >> "$RESULT_FILE"
        continue
    fi

    # 解析结果
    IFS=',' read -r load_time infer_time tokens tps text <<< "$benchmark_output"

    echo "📊 基准测试结果:"
    echo "   模型加载: ${load_time}s"
    echo "   推理时间: ${infer_time}s"
    echo "   生成tokens: ${tokens}"
    echo "   性能: ${tps} tokens/s"
    echo "   输出: ${text}..."

    {
        echo "模型加载: ${load_time}s"
        echo "推理时间: ${infer_time}s"
        echo "生成tokens: ${tokens}"
        echo "性能: ${tps} tokens/s"
        echo "输出: ${text}..."
        echo ""
    } >> "$RESULT_FILE"

    # 添加到JSON
    if [ $success_count -gt 0 ]; then
        echo "," >> "$JSON_FILE"
    fi

    echo '{"frequency_mhz": '$freq_mhz', "actual_mhz": '$actual_mhz', "temperature": '$temp', "load_time": '$load_time', "inference_time": '$infer_time', "tokens": '$tokens', "tokens_per_second": '$tps', "output": "'"$text"'"}' >> "$JSON_FILE"

    success_count=$((success_count + 1))

    # 冷却时间
    echo "⏱️  冷却5秒..."
    sleep 5
done

# JSON结束
echo ']}' >> "$JSON_FILE"

# 恢复默认频率
echo ""
echo "=========================================="
echo "恢复默认设置"
echo "=========================================="

echo "1300500000" | sudo tee /sys/class/devfreq/17000000.gpu/max_freq > /dev/null
echo "306000000" | sudo tee /sys/class/devfreq/17000000.gpu/min_freq > /dev/null

sleep 2
final_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq | awk '{print $1/1000000 " MHz"}')
echo "✅ 恢复完成: $final_freq"

# 最终总结
echo ""
echo "=========================================="
echo "🎉 实验完成！"
echo "=========================================="

{
    echo "完成时间: $(date)"
    echo "成功测试: $success_count/$total_tests"
} >> "$RESULT_FILE"

echo "📁 结果文件:"
echo "  文本格式: $RESULT_FILE"
echo "  JSON格式: $JSON_FILE"

echo ""
echo "📊 测试摘要:"
echo "  成功: $success_count/$total_tests"

if [ $success_count -gt 0 ]; then
    echo ""
    echo "频率    |  实际频率  |  tokens/s  |  温度  |  加载时间"
    echo "---------|------------|------------|--------|----------"

    grep "性能:" "$RESULT_FILE" | awk -F'性能: ' '{print $2}' | awk -F' tokens/s' '{print $1}' > /tmp/tps.txt
    grep "实际频率:" "$RESULT_FILE" | awk -F'实际频率: ' '{print $2}' | awk -F' MHz' '{print $1}' > /tmp/actual.txt
    grep "温度:" "$RESULT_FILE" | awk -F'温度: ' '{print $2}' | awk -F'°C' '{print $1}' > /tmp/temp.txt
    grep "模型加载:" "$RESULT_FILE" | awk -F'模型加载: ' '{print $2}' | awk -F's' '{print $1}' > /tmp/load.txt

    line_num=1
    while IFS= read -r tps; do
        if [ -f "/tmp/actual.txt" ] && [ -f "/temp.txt" ] && [ -f "/load.txt" ]; then
            actual=$(sed -n "${line_num}p" /tmp/actual.txt)
            temp=$(sed -n "${line_num}p" /tmp/temp.txt)
            load=$(sed -n "${line_num}p" /tmp/load.txt)

            printf "%-8s | %-10s | %-10s | %-6s | %s\n" "${FREQ_NAMES[$((line_num-1))]}" "${actual}MHz" "${tps}tokens/s" "${temp}°C" "${load}s"
        fi
        line_num=$((line_num + 1))
    done < /tmp/tps.txt

    # 清理临时文件
    rm -f /tmp/tps.txt /tmp/actual.txt /tmp/temp.txt /tmp/load.txt

    echo ""
    echo "✅ 实验成功完成！可以分析结果了。"
else
    echo ""
    echo "⚠️  没有成功的测试，请检查错误信息。"
fi