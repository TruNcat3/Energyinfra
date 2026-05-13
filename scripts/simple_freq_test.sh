#!/bin/bash
echo "🚀 Simple Frequency Test"
echo "Testing GPU frequency control..."

# 测试当前频率
echo ""
echo "📊 Current GPU Frequency:"
current_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
echo "  Current: $((current_freq / 1000000)) MHz"

# 测试设置不同频率
echo ""
echo "🔄 Testing frequency settings..."

test_freqs=("306000000" "918000000" "1300500000")
freq_names=("306 MHz" "918 MHz" "1300 MHz")

for i in "${!test_freqs[@]}"; do
    freq="${test_freqs[$i]}"
    name="${freq_names[$i]}"

    echo ""
    echo "Setting GPU to $name..."

    echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/max_freq > /dev/null
    echo "$freq" | sudo tee /sys/class/devfreq/17000000.gpu/min_freq > /dev/null

    sleep 2

    actual_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
    echo "  ✅ Set to: $name"
    echo "  📊 Actual: $((actual_freq / 1000000)) MHz"

    sleep 2
done

echo ""
echo "🔄 Restoring defaults..."
echo "1300500000" | sudo tee /sys/class/devfreq/17000000.gpu/max_freq > /dev/null
echo "306000000" | sudo tee /sys/class/devfreq/17000000.gpu/min_freq > /dev/null

final_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
echo "✅ Restored to: $((final_freq / 1000000)) MHz"

echo ""
echo "✅ Frequency test completed!"