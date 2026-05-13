#!/bin/bash
# Quick test script for llama.cpp models

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_path>"
    echo "Example: $0 /home/wt/work/Energyinfra/models/gguf/qwen-7b-chat-q4_k_m.gguf"
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
