#!/bin/bash
echo "🔍 诊断脚本 - 逐步测试实验各个部分"
echo "=========================================="

# 1. 检查目录创建
echo "1️⃣  测试目录创建..."
OUTPUT_DIR="data/frequency_experiment_results"
mkdir -p "$OUTPUT_DIR"
if [ -d "$OUTPUT_DIR" ]; then
    echo "✅ 目录创建成功: $OUTPUT_DIR"
else
    echo "❌ 目录创建失败"
    exit 1
fi

# 2. 检查模型文件
echo ""
echo "2️⃣  检查模型文件..."
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_ROOT
MODEL_PATH="$REPO_ROOT/models/gguf/Phi-3-mini-4k-instruct-q4.gguf"
if [ -f "$MODEL_PATH" ]; then
    size=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
    echo "✅ 模型文件存在: $MODEL_PATH ($size)"
else
    echo "❌ 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 3. 测试频率读取
echo ""
echo "3️⃣  测试频率读取..."
if [ -f "/sys/class/devfreq/17000000.gpu/cur_freq" ]; then
    current_freq=$(cat /sys/class/devfreq/17000000.gpu/cur_freq)
    echo "✅ 当前GPU频率: $((current_freq / 1000000)) MHz"
else
    echo "❌ 无法读取GPU频率"
    exit 1
fi

# 4. 测试温度读取
echo ""
echo "4️⃣  测试温度读取..."
temp=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{print $1/1000}')
if [ ! -z "$temp" ] && [ "$temp" != "0" ]; then
    echo "✅ 当前温度: ${temp}°C"
else
    echo "⚠️  温度读取异常: ${temp}°C"
fi

# 5. 测试简单的Python基准测试
echo ""
echo "5️⃣  测试Python基准测试..."
python_test_result=$(python3 -c "
import sys
import os

# 添加llama_cpp路径
jetson_env_path = os.environ.get('REPO_ROOT', '.') + '/jetson_llm_env/lib/python3.10/site-packages'
if jetson_env_path not in sys.path:
    sys.path.insert(0, jetson_env_path)

try:
    import llama_cpp
    print('✅ llama_cpp导入成功')
    print(f'   版本: {llama_cpp.__version__}')

    # 测试简单推理
    model_path = '$MODEL_PATH'
    if os.path.exists(model_path):
        print(f'✅ 模型文件存在')
        print('   开始简单推理测试...')

        import time
        start = time.time()
        model = llama_cpp.Llama(model_path=model_path, n_ctx=512, n_gpu_layers=99, n_threads=4, verbose=False)
        load_time = time.time() - start
        print(f'   模型加载时间: {load_time:.2f}s')

        start = time.time()
        output = model('Hello', max_tokens=5, temperature=0.7, top_p=0.9, echo=False)
        inference_time = time.time() - start
        print(f'   推理时间: {inference_time:.2f}s')
        print(f'   生成文本: {output[\"choices\"][0][\"text\"]}')

        tokens = output['usage']['completion_tokens'] if 'usage' in output else 0
        tps = tokens / inference_time if inference_time > 0 else 0
        print(f'   性能: {tps:.1f} tokens/s')

        print(f'{load_time:.2f},{inference_time:.2f},{tokens},{tps:.1f}')
    else:
        print('❌ 模型文件不存在')
        sys.exit(1)

except Exception as e:
    print(f'❌ Python测试失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1)

python_exit_code=$?
if [ $python_exit_code -eq 0 ]; then
    echo "✅ Python基准测试成功"
    # 提取性能数据
    if echo "$python_test_result" | grep -q "tokens/s"; then
        perf_data=$(echo "$python_test_result" | tail -1)
        IFS=',' read -r load_time inference_time tokens tps <<< "$perf_data"
        echo "   📊 性能数据: 加载${load_time}s, 推理${inference_time}s, ${tokens}tokens, ${tps}tokens/s"
    fi
else
    echo "❌ Python基准测试失败"
    echo "   错误输出:"
    echo "$python_test_result" | grep "❌"
fi

# 6. 测试文件写入
echo ""
echo "6️⃣  测试文件写入..."
TEST_FILE="$OUTPUT_DIR/test_write.txt"
echo "Test content" > "$TEST_FILE"
if [ -f "$TEST_FILE" ]; then
    echo "✅ 文件写入成功: $TEST_FILE"
    cat "$TEST_FILE"
    rm "$TEST_FILE"
else
    echo "❌ 文件写入失败"
    exit 1
fi

# 7. 测试sudo权限（不实际使用，只是检查）
echo ""
echo "7️⃣  检查sudo权限..."
if sudo -n true 2>/dev/null; then
    echo "✅ Sudo权限可用"
else
    echo "⚠️  Sudo权限需要密码或不可用"
    echo "   这可能会影响频率设置"
fi

# 总结
echo ""
echo "=========================================="
echo "🔍 诊断总结"
echo "=========================================="
echo "基本系统检查:"
echo "  ✅ 目录创建: 正常"
echo "  ✅ 模型文件: 正常"
echo "  ✅ 频率读取: 正常"
echo "  ✅ 温度读取: 正常"

if [ $python_exit_code -eq 0 ]; then
    echo "  ✅ Python测试: 正常"
    echo ""
    echo "🎉 所有基本功能正常！可以运行完整实验。"
else
    echo "  ❌ Python测试: 失败"
    echo ""
    echo "⚠️  Python基准测试有问题，需要先解决。"
    echo "   查看上面的错误信息进行调试。"
fi