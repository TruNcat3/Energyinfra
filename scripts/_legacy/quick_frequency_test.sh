#!/bin/bash
# Quick Frequency Test - Execute this in your sudo terminal

echo "🚀 Starting Quick Frequency Test for Jetson Orin"
echo "=========================================="

# 频率配置
FREQUENCIES=("306000000" "612000000" "918000000" "1122000000" "1300500000")
FREQ_NAMES=("306 MHz" "612 MHz" "918 MHz" "1122 MHz" "1300 MHz")

# 模型路径
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL_PATH="$REPO_ROOT/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"

# 输出目录
OUTPUT_DIR="data/frequency_test_results"
mkdir -p "$OUTPUT_DIR"

# 结果文件
RESULT_FILE="$OUTPUT_DIR/frequency_test_$(date +%Y%m%d_%H%M%S).csv"

# CSV header
echo "Frequency_MHz,Timestamp,GPU_Freq_Hz,Test_Result,Notes" > "$RESULT_FILE"

echo "📊 Available GPU Frequencies:"
cat /sys/class/devfreq/17000000.gpu/available_frequencies | tr ' ' '\n' | awk '{print "  " $1/1000000 " MHz"}'

echo ""
echo "🔄 Testing frequency control..."

for i in "${!FREQUENCIES[@]}"; do
    freq="${FREQUENCIES[$i]}"
    name="${FREQ_NAMES[$i]}"

    echo ""
    echo "----------------------------------------"
    echo "Test $((i+1))/${#FREQUENCIES[@]}: $name"
    echo "----------------------------------------"

    # 设置频率
    echo "Setting GPU to $name..."
    sudo sh -c "echo '$freq' > /sys/class/devfreq/17000000.gpu/max_freq"
    sudo sh -c "echo '$freq' > /sys/class/devfreq/17000000.gpu/min_freq"

    # 等待频率稳定
    sleep 2

    # 读取实际频率
    actual_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
    actual_mhz=$((actual_freq / 1000000))

    echo "✅ Set to: $name"
    echo "📊 Actual: $actual_mhz MHz"

    # 读取温度
    temp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{print $1/1000}')

    # 记录结果
    timestamp=$(date +%Y-%m-%d_%H:%M:%S)
    echo "$actual_mhz,$timestamp,$actual_freq,Frequency_Set_Success,Temp:${temp}C" >> "$RESULT_FILE"

    echo "🌡️  Temperature: ${temp}°C"
    echo "⏱️  Holding for 3 seconds..."

    # 保持频率3秒
    sleep 3

    echo "✅ Test completed"
done

echo ""
echo "=========================================="
echo "🎉 Frequency control test completed!"
echo "=========================================="

# 恢复默认设置
echo "🔄 Restoring default settings..."
sudo sh -c 'echo "1300500000" > /sys/class/devfreq/17000000.gpu/max_freq'
sudo sh -c 'echo "306000000" > /sys/class/devfreq/17000000.gpu/min_freq'

sleep 2
final_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq | awk '{print $1/1000000 " MHz"}')
echo "✅ Restored to: $final_freq"

echo ""
echo "📁 Results saved to: $RESULT_FILE"
echo ""
echo "📊 Test Summary:"
cat "$RESULT_FILE"

echo ""
echo "💡 Now you can run the benchmark script in another terminal:"
echo "   cd \"$REPO_ROOT\""
echo "   python3 src/_legacy/frequency_benchmark.py"
echo ""
echo "   Make sure to set the desired frequency first using:"
echo "   sudo sh -c 'echo \"<FREQUENCY>\" > /sys/class/devfreq/17000000.gpu/max_freq'"
echo "   sudo sh -c 'echo \"<FREQUENCY>\" > /sys/class/devfreq/17000000.gpu/min_freq'"