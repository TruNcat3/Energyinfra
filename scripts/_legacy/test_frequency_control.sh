#!/bin/bash
# Frequency Control Test Script for Jetson Orin
# 需要在有sudo权限的终端中执行

echo "=========================================="
echo "Jetson Orin Frequency Control Test"
echo "=========================================="

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (use sudo)"
    exit 1
fi

echo "✅ Running with root privileges"

# 当前频率状态
echo ""
echo "📊 Current Frequency Status:"
echo "GPU:"
cat /sys/class/devfreq/17000000.gpu/cur_freq | awk '{print "  Current: " $1/1000000 " MHz"}'
cat /sys/class/devfreq/17000000.gpu/min_freq | awk '{print "  Min: " $1/1000000 " MHz"}'
cat /sys/class/devfreq/17000000.gpu/max_freq | awk '{print "  Max: " $1/1000000 " MHz"}'
cat /sys/class/devfreq/17000000.gpu/governor | awk '{print "  Governor: " $1}'

# CPU频率
echo ""
echo "CPU:"
for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do
    cpu_num=$(basename $(dirname $cpu))
    freq=$(cat $cpu)
    echo "  $cpu_num: $((freq/1000)) MHz"
done

# EMC频率
echo ""
echo "EMC:"
cat /sys/class/devfreq/17000000.emc/cur_freq 2>/dev/null | awk '{print "  Current: " $1/1000000 " MHz"}' || echo "  EMC not found"

echo ""
echo "=========================================="
echo "Available GPU Frequencies:"
cat /sys/class/devfreq/17000000.gpu/available_frequencies | tr ' ' '\n' | awk '{print "  " $1/1000000 " MHz"}'

echo ""
echo "=========================================="
echo "Setting Different Frequency Configurations"
echo "=========================================="

# 测试不同的GPU频率配置
configs=(
    "306000000:GPU Min"
    "612000000:GPU Low-Mid"
    "918000000:GPU Mid"
    "1122000000:GPU High"
    "1300500000:GPU Max"
)

for config in "${configs[@]}"; do
    IFS=':' read -r freq label <<< "$config"
    echo ""
    echo "🔄 Testing: $label ($((freq/1000000)) MHz)"

    # 设置GPU最大频率
    echo "$freq" > /sys/class/devfreq/17000000.gpu/max_freq
    echo "$freq" > /sys/class/devfreq/17000000.gpu/min_freq

    # 等待频率稳定
    sleep 2

    # 读取实际频率
    actual_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
    echo "  ✅ Set to: $((freq/1000000)) MHz"
    echo "  📊 Actual: $((actual_freq/1000000)) MHz"

    # 获取当前系统状态
    temp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{print $1/1000}')
    power=$(cat /sys/class/power_supply/battery/current_now 2>/dev/null || echo "0")

    echo "  🌡️  Temperature: ${temp}°C"
    echo "  ⚡ Power: Reading from tegrastats recommended"

    # 保持这个频率一段时间
    echo "  ⏱️  Holding for 3 seconds..."
    sleep 3
done

echo ""
echo "=========================================="
echo "Restoring Default Settings"
echo "=========================================="

# 恢复默认设置
echo "1300500000" > /sys/class/devfreq/17000000.gpu/max_freq
echo "306000000" > /sys/class/devfreq/17000000.gpu/min_freq

echo "✅ Frequency control test completed!"
echo ""
echo "💡 You can now run the benchmark script to test different frequencies"
echo "   Example: python3 src/_legacy/frequency_benchmark.py"