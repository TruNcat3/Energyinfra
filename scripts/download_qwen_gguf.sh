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
cd /home/wt/work/Energyinfra/models/gguf

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
    echo "Model saved to: /home/wt/work/Energyinfra/models/gguf/$MODEL_NAME"
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
