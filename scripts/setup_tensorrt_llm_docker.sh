#!/bin/bash
# TensorRT-LLM Docker 容器设置脚本
# 用于配置官方TensorRT-LLM环境并准备模型

set -e  # 遇到错误时退出

# 颜色输出
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

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 配置变量
CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt-llm:v0.12.0-nightly-trtllm-python-py3"
WORKSPACE_DIR="/home/wt/work/Energyinfra"
DOCKER_VOLUME_MOUNT="$WORKSPACE_DIR:/workspace"

# 模型配置
MODELS=(
    "qwen:7b-int4"
    "llama-3:8b-fp16"
)

print_info "========================================="
print_info "TensorRT-LLM Docker Setup Script"
print_info "========================================="
print_info ""

# 步骤1：检查Docker安装
print_info "Step 1: Checking Docker installation..."
if command -v docker &> /dev/null; then
    print_info "✅ Docker is installed: $(docker --version)"
else
    print_error "❌ Docker is not installed"
    print_info "Please install Docker first: https://docs.docker.com/engine/install/"
    exit 1
fi

# 步骤2：检查Docker服务状态
print_info "Step 2: Checking Docker service status..."
if sudo systemctl is-active --quiet docker; then
    print_info "✅ Docker service is running"
else
    print_warning "⚠️  Docker service is not running"
    print_info "Attempting to start Docker service..."
    sudo systemctl start docker || {
        print_error "❌ Failed to start Docker service"
        print_info "You may need to start Docker manually: sudo systemctl start docker"
        exit 1
    }
    print_info "✅ Docker service started"
fi

# 步骤3：检查NVIDIA Docker运行时支持
print_info "Step 3: Checking NVIDIA Docker runtime..."
if docker info | grep -q "nvidia"; then
    print_info "✅ NVIDIA Docker runtime is configured"
else
    print_warning "⚠️  NVIDIA Docker runtime may not be properly configured"
    print_info "Checking for nvidia-docker2..."
    if ! command -v nvidia-docker &> /dev/null; then
        print_info "Installing nvidia-docker2..."
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | \
          sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
        print_info "✅ nvidia-docker2 installed"
    else
        print_info "✅ nvidia-docker2 is available"
    fi
fi

# 步骤4：拉取TensorRT-LLM Docker镜像
print_info "Step 4: Pulling TensorRT-LLM Docker image..."
print_info "This may take a while (several minutes)..."
print_info "Image: $CONTAINER_IMAGE"

if sudo docker images | grep -q "tensorrt-llm"; then
    print_info "✅ TensorRT-LLM image already exists locally"
else
    print_info "Pulling image (this may take 10-20 minutes)..."
    sudo docker pull "$CONTAINER_IMAGE" || {
        print_error "❌ Failed to pull TensorRT-LLM image"
        print_info "Possible solutions:"
        print_info "  1. Check internet connection"
        print_info "  2. Check NVIDIA NGC account (may require login)"
        print_info "  3. Try different image version"
        exit 1
    }
    print_info "✅ TensorRT-LLM image pulled successfully"
fi

# 步骤5：验证Jetson环境支持
print_info "Step 5: Verifying Jetson environment..."

if [ -f /etc/nv_tegra_release ]; then
    DEVICE_INFO=$(cat /etc/nv_tegra_release | head -n1)
    print_info "✅ Jetson device detected: $DEVICE_INFO"
else
    print_error "❌ Not running on a Jetson device"
    print_warning "This script is designed for Jetson devices"
    exit 1
fi

# 步骤6：创建工作空间目录
print_info "Step 6: Creating workspace directories..."
mkdir -p "$WORKSPACE_DIR/models"
mkdir -p "$WORKSPACE_DIR/engines"
mkdir -p "$WORKSPACE_DIR/hf_models"
print_info "✅ Workspace directories created"

# 步骤7：准备模型下载和转换说明
print_info "Step 7: Preparing model setup instructions..."

for model in "${MODELS[@]}"; do
    IFS=':' read -r MODEL_NAME MODEL_QUANT <<< "$model"

    case "$MODEL_NAME" in
        "qwen")
            HF_MODEL="Qwen/Qwen-7B"
            ENGINE_NAME="qwen-7b-int4.engine"
            ;;
        "llama-3")
            HF_MODEL="meta-llama/Meta-Llama-3-8B"
            ENGINE_NAME="llama-3-8b-fp16.engine"
            ;;
        *)
            print_error "❌ Unknown model: $MODEL_NAME"
            continue
            ;;
    esac

    print_info ""
    print_info "Model setup for: $MODEL_NAME ($MODEL_QUANT)"
    print_info "  Hugging Face Model: $HF_MODEL"
    print_info "  Target Engine: $ENGINE_NAME"
    print_info "  Engine Path: $WORKSPACE_DIR/engines/$ENGINE_NAME"

done

# 步骤8：创建容器启动脚本
print_info "Step 8: Creating Docker container launch script..."

cat > "$WORKSPACE_DIR/scripts/launch_tensorrt_container.sh" << 'EOF'
#!/bin/bash
# TensorRT-LLM Container Launch Script

CONTAINER_IMAGE="nvcr.io/nvidia/tensorrt-llm:v0.12.0-nightly-trtllm-python-py3"
WORKSPACE_DIR="$WORKSPACE_DIR"

echo "========================================="
echo "Launching TensorRT-LLM Container"
echo "========================================="
echo ""

# 检查是否已有运行中的容器
if docker ps | grep -q tensorrt; then
    echo "TensorRT-LLM container is already running"
    echo "To attach: docker exec -it tensorrt_llm bash"
    exit 0
fi

# 启动新容器
echo "Starting TensorRT-LLM container..."
docker run --rm \
    --name tensorrt_llm \
    --gpus all \
    --privileged \
    -it \
    -v "$WORKSPACE_DIR:/workspace" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -e DISPLAY=$DISPLAY \
    $CONTAINER_IMAGE \
    bash

echo "Container exited"
EOF

chmod +x "$WORKSPACE_DIR/scripts/launch_tensorrt_container.sh"
print_info "✅ Container launch script created: scripts/launch_tensorrt_container.sh"

# 步骤9：创建模型转换脚本
print_info "Step 9: Creating model conversion scripts..."

# Qwen-7B-INT4 转换脚本
cat > "$WORKSPACE_DIR/scripts/convert_qwen_int4.sh" << 'EOF'
#!/bin/bash
# Qwen-7B INT4 Model Conversion Script

set -e

echo "========================================="
echo "Converting Qwen-7B to INT4 TensorRT Engine"
echo "========================================="

HF_MODEL="Qwen/Qwen-7B"
OUTPUT_DIR="/workspace/engines/qwen-7b-int4"
HF_CACHE_DIR="/workspace/hf_models"

echo "Step 1: Downloading Hugging Face model..."
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

python3 << 'PYTHON_SCRIPT'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen-7B"
print(f"Loading model: {model_name}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save to local cache
model.save_pretrained(f"{HF_CACHE_DIR}/qwen-7b-hf")
tokenizer.save_pretrained(f"{HF_CACHE_DIR}/qwen-7b-hf")

print(f"Model saved to: {HF_CACHE_DIR}/qwen-7b-hf")
PYTHON_SCRIPT

echo "✅ Model downloaded to: $HF_CACHE_DIR/qwen-7b-hf"
echo ""
echo "Step 2: Converting to TensorRT-LLM format..."

cd /workspace/TensorRT-LLM
python3 examples/llama/convert_checkpoint.py \
    --model_dir "$HF_CACHE_DIR/qwen-7b-hf" \
    --output_dir "$OUTPUT_DIR" \
    --dtype float16 \
    --calibrate_checkpoint \
    --use_gpt_attention_plugin \
    --use_inflight_batching \
    --max_input_len 4096 \
    --max_output_len 512

echo "✅ Model conversion completed"
echo "✅ Engine saved to: $OUTPUT_DIR"
EOF

chmod +x "$WORKSPACE_DIR/scripts/convert_qwen_int4.sh"

# Llama-3-8B-FP16 转换脚本
cat > "$WORKSPACE_DIR/scripts/convert_llama3_fp16.sh" << 'EOF'
#!/bin/bash
# Llama-3-8B FP16 Model Conversion Script

set -e

echo "========================================="
echo "Converting Llama-3-8B to FP16 TensorRT Engine"
echo "========================================="

HF_MODEL="meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR="/workspace/engines/llama-3-8b-fp16"
HF_CACHE_DIR="/workspace/hf_models"

echo "Step 1: Downloading Hugging Face model..."
mkdir -p "$HF_CACHE_DIR"
export HF_HOME="$HF_CACHE_DIR"

python3 << 'PYTHON_SCRIPT'
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Meta-Llama-3-8B"
print(f"Loading model: {model_name}")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save to local cache
model.save_pretrained(f"{HF_CACHE_DIR}/llama-3-8b-hf")
tokenizer.save_pretrained(f"{HF_CACHE_DIR}/llama-3-8b-hf")

print(f"Model saved to: {HF_CACHE_DIR}/llama-3-8b-hf")
PYTHON_SCRIPT

echo "✅ Model downloaded to: $HF_CACHE_DIR/llama-3-8b-hf"
echo ""
echo "Step 2: Converting to TensorRT-LLM format..."

cd /workspace/TensorRT-LLM
python3 examples/llama/convert_checkpoint.py \
    --model_dir "$HF_CACHE_DIR"/llama-3-8b-hf" \
    --output_dir "$OUTPUT_DIR" \
    --dtype float16 \
    --use_gpt_attention_plugin \
    --use_inflight_batching \
    --max_input_len 8192 \
    --max_output_len 1024

echo "✅ Model conversion completed"
echo "✅ Engine saved to: $OUTPUT_DIR"
EOF

chmod +x "$WORKSPACE_DIR/scripts/convert_llama3_fp16.sh"

print_info "✅ Model conversion scripts created"

# 完成总结
echo ""
echo "========================================="
echo "Setup Complete! 🎉"
echo "========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Launch TensorRT-LLM container:"
echo "   $ bash scripts/launch_tensorrt_container.sh"
echo ""
echo "2. Inside container, convert models:"
echo "   # Qwen-7B INT4:"
echo "   $ bash scripts/convert_qwen_int4.sh"
echo ""
echo "   # Llama-3-8B FP16:"
echo "   $ bash scripts/convert_llama3_fp16.sh"
echo ""
echo "3. After conversion, run benchmarks:"
echo "   # Using the TensorRT-LLM benchmark tools"
echo "   $ python3 run.py --engine /workspace/engines/qwen-7b-int4.engine ..."
echo ""
echo "Important Notes:"
echo "  - Model conversion may take several hours per model"
echo "  - Ensure sufficient disk space (models can be 10-30GB each)"
echo "  - Monitor GPU memory usage during conversion"
echo "  - Check NVIDIA NGC for pre-built engines to save time"
echo ""
echo "========================================="