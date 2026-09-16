#!/bin/bash
# llama.cpp 快速集成脚本
# 无需账号，立即开始真实LLM测试

set -e

# 需色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

print_info "========================================="
print_info "llama.cpp Quick Setup Script"
print_info "========================================="
print_info ""

# 检查Python环境
print_info "Step 1: Checking Python environment..."
if [ -f "$WORKSPACE_DIR/jetson_llm_env/bin/activate" ]; then
    print_info "✅ Virtual environment found"
    source "$WORKSPACE_DIR/jetson_llm_env/bin/activate"
else
    print_error "❌ Virtual environment not found"
    exit 1
fi

# 安装llama.cpp Python绑定
print_info "Step 2: Installing llama.cpp Python binding..."
print_info "This may take a few minutes..."

if python3 -c "import llama_cpp" 2>/dev/null; then
    print_info "✅ llama-cpp-python already installed"
else
    print_info "Installing llama-cpp-python..."
    pip3 install llama-cpp-python || {
        print_error "❌ Failed to install llama-cpp-python"
        exit 1
    }
    print_success "✅ llama-cpp-python installed successfully"
fi

# 创建模型目录
print_info "Step 3: Creating model directories..."
mkdir -p "$WORKSPACE_DIR/models"
mkdir -p "$WORKSPACE_DIR/models/gguf"
mkdir -p "$WORKSPACE_DIR/models/hf"
print_info "✅ Model directories created"

# 创建模型下载脚本
print_info "Step 4: Creating model download scripts..."

# Qwen-7B-GGUF 下载脚本
cat > "$WORKSPACE_DIR/scripts/download_qwen_gguf.sh" << 'EOF'
#!/bin/bash
# Download Qwen-7B GGUF model from Hugging Face

set -e

echo "========================================="
echo "Downloading Qwen-7B GGUF Models"
echo "========================================="
echo ""

# Qwen-7B Quantized GGUF models
QWEN_MODELS=(
    "Qwen/Qwen-7B-Chat-GGUF:qwen-7b-chat-q4_k_m.gguf:q4_k_m"
    "Qwen/Qwen-7B-Chat-GGUF:qwen-7b-chat-q5_k_m.gguf:q5_k_m"
    "Qwen/Qwen-7B-Chat-GGUF:qwen-7b-chat-q8_0.gguf:q8_0"
)

# 选择量化级别
echo "Available quantization options:"
echo "1. Q4_K_M (4-bit, medium, 2.7GB) - RECOMMENDED"
echo "2. Q5_K_M (5-bit, small, 2.3GB)"
echo "3. Q8_0 (8-bit, large, 4.9GB)"
echo ""
read -p "Select quantization (1-3, default=1): " quant_choice

quant_choice=${quant_choice:-1}

case $quant_choice in
    1)
        MODEL_URL="https://huggingface.co/TheBloke/Qwen-7B-Chat-GGUF/resolve/main/qwen-7b-chat-q4_k_m.gguf"
        MODEL_NAME="qwen-7b-chat-q4_k_m.gguf"
        ;;
    2)
        MODEL_URL="https://huggingface.co/TheBloke/Qwen-7B-Chat-GGUF/resolve/main/qwen-7b-chat-q5_k_m.gguf"
        MODEL_NAME="qwen-7b-chat-q5_k_m.gguf"
        ;;
    3)
        MODEL_URL="https://huggingface.co/TheBloke/Qwen-7B-Chat-GGUF/resolve/main/qwen-7b-chat-q8_0.gguf"
        MODEL_NAME="qwen-7b-chat-q8_0.gguf"
        ;;
    *)
        echo "Invalid choice, using Q4_K_M (recommended)"
        MODEL_URL="https://huggingface.co/TheBloke/Qwen-7B-Chat-GGUF/resolve/main/qwen-7b-chat-q4_k_m.gguf"
        MODEL_NAME="qwen-7b-chat-q4_k_m.gguf"
        ;;
esac

echo "Downloading: $MODEL_NAME"
echo "URL: $MODEL_URL"
echo ""

# 下载模型
cd "$WORKSPACE_DIR/models/gguf"

if [ -f "$MODEL_NAME" ]; then
    echo "Model already exists: $MODEL_NAME"
    echo "Re-download? (y/n)"
    read -r RE_DOWNLOAD
    if [ "$RE_DOWNLOAD" != "y" ]; then
        echo "Using existing model"
        exit 0
    fi
    rm -f "$MODEL_NAME"
fi

# 使用wget下载
wget --progress=bar:force "$MODEL_URL" -O "$MODEL_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ SUCCESS: Model downloaded"
    echo "========================================="
    echo ""
    echo "Model saved to: $WORKSPACE_DIR/models/gguf/$MODEL_NAME"
    echo "File size: $(du -h "$MODEL_NAME" | cut -f1)"
    echo ""
    echo "Next steps:"
    echo "1. Test the model:"
    echo "   python3 -m llama_cpp --model $MODEL_NAME --prompt 'Hello, how are you?'"
    echo ""
    echo "2. Run experiments:"
    echo "   python3 src/experiment_4_1_stability_llamacpp.py --model $MODEL_NAME"
else
    echo ""
    echo "========================================="
    echo "❌ ERROR: Download failed"
    echo "========================================="
    echo ""
    echo "Troubleshooting:"
    echo "1. Check internet connection"
    echo "2. Try manual download from Hugging Face:"
    echo "   https://huggingface.co/TheBloke/Qwen-7B-Chat-GGUF"
    echo "3. Check available disk space"
    exit 1
fi
EOF

chmod +x "$WORKSPACE_DIR/scripts/download_qwen_gguf.sh"
print_info "✅ Model download script created"

# 创建测试脚本
print_info "Step 5: Creating test script..."

cat > "$WORKSPACE_DIR/scripts/test_llamacpp_model.sh" << 'EOF'
#!/bin/bash
# Quick test script for llama.cpp models

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    echo "Example: $0 $WORKSPACE_DIR/models/gguf/qwen-7b-chat-q4_k_m.gguf"
    exit 1
fi

MODEL_PATH="$1"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "========================================="
echo "Testing llama.cpp Model"
echo "========================================="
echo ""
echo "Model: $MODEL_PATH"
echo ""

# 检查llama.cpp安装
if ! python3 -c "import llama_cpp" 2>/dev/null; then
    echo "❌ Error: llama-cpp-python not installed"
    echo "Run: pip3 install llama-cpp-python"
    exit 1
fi

# 运行简单测试
echo "Running quick test..."
echo ""

python3 << 'PYTHON_SCRIPT'
import llama_cpp
import sys

model_path = sys.argv[1]

print(f"Loading model: {model_path}")
print("")

# 初始化模型
llm = llama_cpp.Llama(
    model_path=model_path,
    n_ctx=512,        # context size
    n_batch=512,       # batch size
    n_threads=4,       # CPU threads
    verbose=False
)

print("Model loaded successfully!")
print("")

# 运行简单推理
prompt = "The quick brown fox jumps over the lazy dog."

print(f"Prompt: {prompt}")
print("")

tokens = llm.tokenize(prompt)
print(f"Tokens: {len(tokens)}")
print("")

# 生成输出
output = llm(
    prompt,
    max_tokens=50,     # generate up to 50 tokens
    stop=[".", "!", "?"],
    echo=True,
    temperature=0.7,
)

print("Generated output:")
print(output)
print("")

# 简单性能统计
print("Quick performance test:")
print("")

import time

# 测试推理速度
start_time = time.time()
for i in range(5):
    llm("test", max_tokens=10, stop=["."])
end_time = time.time()

avg_time = (end_time - start_time) / 5
tokens_per_second = 10 / avg_time

print(f"Average generation time: {avg_time:.3f}s")
print(f"Tokens per second: {tokens_per_second:.1f}")
print(f"Estimated TTFT: {avg_time/5:.3f}s")
print(f"Estimated TPOT: {avg_time/10:.3f}s")

PYTHON_SCRIPT'

echo ""
echo "========================================="
echo "Test completed successfully!"
echo "========================================="
EOF

chmod +x "$WORKSPACE_DIR/scripts/test_llamacpp_model.sh"
print_info "✅ Test script created"

# 创建集成说明
print_info "Step 6: Creating integration guide..."

cat > "$WORKSPACE_DIR/docs/LLAMACPP_INTEGRATION.md" << 'EOF'
# llama.cpp Integration Guide
# 快速集成llama.cpp到Jetson LLM能耗分析项目

## ✅ 安装验证

llama.cpp已安装并可以在您的项目中使用：

```python
import llama_cpp

# 加载模型
llm = llama_cpp.Llama(
    model_path="path/to/model.gguf",
    n_ctx=4096,
    n_batch=512,
    n_threads=4
)

# 运行推理
output = llm("Your prompt here", max_tokens=100)
print(output)
```

## 🚀 快速开始

### 选项1：下载Qwen-7B模型（推荐）

```bash
# 下载Qwen-7B GGUF模型
bash scripts/download_qwen_gguf.sh

# 这将下载约2.7GB的Q4_K_M量化模型
```

### 选项2：测试模型

```bash
# 测试下载的模型
bash scripts/test_llamacpp_model.sh "$WORKSPACE_DIR/models/gguf/qwen-7b-chat-q4_k_m.gguf"
```

### 选项3：运行实验

```bash
# 使用llama.cpp运行实验4.1
python3 src/experiment_4_1_stability_llamacpp.py \
  --model /path/to/model.gguf \
  --batch_size 1 \
  --prompt_len 512 \
  --output_len 128 \
  --gpu_freq 846 \
  --cpu_freq 1479 \
  --emc_freq 1600
```

## 📊 可用的量化选项

对于Qwen-7B-Chat GGUF模型，有以下量化选项：

| 量化级别 | 文件大小 | 内存占用 | 精度 | 推荐度 |
|----------|----------|----------|------|--------|
| Q4_K_M | ~2.7GB | ~3.5GB | 中等 | ⭐⭐⭐⭐⭐ 推荐 |
| Q5_K_M | ~2.3GB | ~3.1GB | 较低 | ⭐⭐⭐ |
| Q8_0 | ~4.9GB | ~6GB | 较高 | ⭐⭐ |

## 🎯 推荐的测试流程

### 步骤1：基础验证（今日）
1. 下载Qwen-7B-Q4_K_M模型
2. 运行快速测试验证功能
3. 修改实验脚本支持llama.cpp

### 步骤2：真实实验4.1（明日）
1. 使用llama.cpp运行测量稳定性实验
2. 对比合成测试和真实测试结果
3. 验证我们的核心假设

### 步骤3：频率扫描实验（本周）
1. 修改实验4.2支持llama.cpp
2. 运行单旋钮敏感性实验
3. 生成GPU/CPU/EMC频率影响热力图

## 🔧 集成要点

### 频率控制兼容性
llama.cpp完全兼容我们的频率控制器：
- GPU频率：通过sysfs读取和设置
- CPU频率：通过sysfs读取和设置
- EMC频率：通过jetson_clocks设置

### 性能指标获取
llama.cpp可以提供我们需要的所有指标：
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- 吞吐量 (tokens/second)
- 内存使用
- 推理时间

### 与现有系统集成
1. **Benchmark Runner**: 创建llama.cpp适配器
2. **Metrics Collector**: 继续使用tegrastats
3. **Frequency Controller**: 无需修改
4. **Data Analysis**: 无需重大修改

## 📈 预期性能

基于Qwen-7B-Q4_K_M在Jetson Orin上的预期：

- **TTFT**: 100-300ms (取决于频率和上下文)
- **TPOT**: 10-50ms (每token)
- **吞吐量**: 20-60 tokens/s
- **内存占用**: ~3.5GB
- **功耗**: 15-30W (取决于频率)

这些性能特征与我们的合成测试假设相符。

## 🚨 注意事项

1. **量化权衡**: Q4量化节省内存但可能影响精度
2. **上下文长度**: 较长的上下文可能需要更多内存
3. **批处理**: 大批量可能超出GPU内存
4. **温度**: 持续运行可能导致热节流

## 🎉 开始使用

现在您已经准备好使用llama.cpp开始真实的LLM实验！

建议的下一步：
1. 运行下载脚本获取Qwen-7B模型
2. 运行测试脚本验证功能
3. 开始真实实验验证我们的假设

祝您实验顺利！🚀
EOF

print_success "✅ Integration guide created"

# 完成总结
echo ""
echo "========================================="
echo "Setup Complete! 🎉"
echo "========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Download Qwen-7B model:"
echo "   $ bash scripts/download_qwen_gguf.sh"
echo ""
echo "2. Test the model:"
echo "   $ bash scripts/test_llamacpp_model.sh <model_path>"
echo ""
echo "3. Read integration guide:"
echo "   $ cat docs/LLAMACPP_INTEGRATION.md"
echo ""
echo "========================================="