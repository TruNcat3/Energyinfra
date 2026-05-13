#!/bin/bash
# Final Frequency Experiment Script - with reliable benchmark
最终频率实验脚本 - 使用可靠的基准测试

echo "🚀 Final Frequency Experiment"
echo "=========================================="

# 检查sudo权限
if ! sudo -n true 2>/dev/null; then
    echo "❌ 需要sudo权限"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="data/frequency_experiment_results"
mkdir -p "$OUTPUT_DIR"

# 实验配置
FREQUENCIES=("306000000" "612000000" "918000000" "1300500000")
FREQ_NAMES=("306 MHz" "612 MHz" "918 MHz" "1300 MHz")
MODEL_PATH="/home/wt/work/Energyinfra/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"
BENCHMARK_SCRIPT="/home/wt/work/Energyinfra/src/final_benchmark.py"

# 结果文件
RESULT_FILE="$OUTPUT_DIR/final_results_$(date +%Y%m%d_%H%M%S).txt"
JSON_FILE="$OUTPUT_DIR/final_results_$(date +%Y%m%d_%H%M%S).json"

echo "📊 实验配置:"
echo "  模型: $MODEL_PATH"
echo "  基准脚本: $BENCHMARK_SCRIPT"
echo "  测试频率: ${FREQ_NAMES[*]}"
echo ""

# 开始文本结果
{
    echo "频率实验结果 (最终版本)"
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

    # 文本记录
    {
        echo "测试 $((i+1)): $name"
        echo "--------------------"
    } >> "$RESULT_FILE"

    # 设置GPU频率
    echo "🔄 设置GPU频率为 $name..."

    if echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/max_freq > /dev/null 2>&1 && \
       echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/min_freq > /dev/null 2>&1; then

        echo "   ✅ 频率设置成功"
    else
        echo "   ❌ 频率设置失败"
        echo "频率设置失败" >> "$RESULT_FILE"
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

    # 文本记录
    {
        echo "设置频率: $name"
        echo "实际频率: $actual_mhz MHz"
        echo "温度: ${temp}°C"
    } >> "$RESULT_FILE"

    # 运行基准测试（重定向stderr到/dev/null以获取纯净输出）
    echo "🚀 运行基准测试..."
    benchmark_result=$(python3 "$BENCHMARK_SCRIPT" 2>/dev/null)

    # 检查是否出错
    if echo "$benchmark_result" | grep -q "^ERROR"; then
        echo "❌ 基准测试失败"
        error_msg=$(echo "$benchmark_result" | cut -d',' -f2-)
        echo "   错误: $error_msg"

        {
            echo "基准测试: 失败"
            echo "错误: $error_msg"
            echo ""
        } >> "$RESULT_FILE"
        continue
    fi

    # 解析CSV结果
    IFS=',' read -r load_time infer_time tokens tps text <<< "$benchmark_result"

    echo "📊 基准测试结果:"
    echo "   模型加载: ${load_time}s"
    echo "   推理时间: ${infer_time}s"
    echo "   生成tokens: ${tokens}"
    echo "   性能: ${tps} tokens/s"
    echo "   输出: ${text}..."

    # 文本记录
    {
        echo "模型加载: ${load_time}s"
        echo "推理时间: ${infer_time}s"
        echo "生成tokens: ${tokens}"
        echo "性能: ${tps} tokens/s"
        echo "输出: ${text}..."
        echo ""
    } >> "$RESULT_FILE"

    # JSON记录
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

    # 提取数据
    grep "性能:" "$RESULT_FILE" | awk -F'性能: ' '{print $2}' | awk -F' tokens/s' '{print $1}' > /tmp/tps.txt
    grep "实际频率:" "$RESULT_FILE" | awk -F'实际频率: ' '{print $2}' | awk -F' MHz' '{print $1}' > /tmp/actual.txt
    grep "温度:" "$RESULT_FILE" | awk -F'温度: ' '{print $2}' | awk -F'°C' '{print $1}' > /tmp/temp.txt
    grep "模型加载:" "$RESULT_FILE" | awk -F'模型加载: ' '{print $2}' | awk -F's' '{print $1}' > /tmp/load.txt

    line_num=1
    while IFS= read -r tps; do
        if [ -f "/tmp/actual.txt" ] && [ -f "/tmp/temp.txt" ] && [ -f "/tmp/load.txt" ]; then
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
    echo "✅ 实验成功完成！结果已保存。"
else
    echo ""
    echo "⚠️  没有成功的测试，请检查错误信息。"
fi