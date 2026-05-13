# TensorRT-LLM 替代方案
# Alternative Runtime Solutions

**问题**: TensorRT-LLM需要NVIDIA NGC账号，暂时无法获取  
**日期**: 2026-05-06  
**目标**: 继续推进项目，使用其他可用的LLM运行时

---

## 🚀 推荐替代方案

### 方案1：llama.cpp (强烈推荐）⭐

#### 优势
- ✅ **开源免费**：无需任何账号或授权
- ✅ **性能优秀**：在边缘设备上优化良好
- ✅ **易于安装**：简单的pip或编译安装
- ✅ **支持量化**：INT4/INT8/FP16等
- ✅ **Jetson优化**：官方ARM64支持
- ✅ **丰富的模型支持**：Qwen、Llama、Mistral等

#### 安装步骤
```bash
# 1. 克隆llama.cpp仓库
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 2. 编译（Jetson ARM64优化）
mkdir build
cd build
cmake .. -DGGML_LLAMA_CUBLAS=OFF -DGGML_LLAMA_BLAS=OFF
cmake --build . --target llama-cli

# 3. 或者使用预编译（如果可用）
# Jetson ARM64可能有预编译版本
```

#### 模型转换
```bash
# 转换Hugging Face模型为GGUF格式
python3 convert_hf_to_gguf.py \
  --model Qwen/Qwen-7B \
  --outfile qwen-7b-q4_k.gguf \
  --qtype q4_k \
  --outfile-type q4_k

# 或者使用在线转换工具
# https://huggingface.co/spaces/mradermavi/ggml-converter
```

#### 基准测试集成
- **优点**: 与真实推理结果非常接近
- **接口**: 命令行，易于集成
- **指标**: 可以获取TTFT、TPOT、吞吐量
- **频率控制**: 完全兼容我们的混合模式

#### 预期性能
- **Qwen-7B-Q4**: ~30-50 tokens/s (Jetson Orin)
- **Llama-3-8B-FP16**: ~20-40 tokens/s (Jetson Orin)
- **内存占用**: 4-8GB (取决于量化级别)

---

### 方案2：vLLM (推荐）⭐

#### 优势
- ✅ **高性能**：专为GPU优化，支持CUDA
- ✅ **开源免费**：Apache 2.0许可证
- ✅ **易于使用**：类似OpenAI的API接口
- ✅ **模型支持广**：Hugging Face生态
- ✅ **GPU利用率高**：比TensorRT-LLM更容易使用

#### 安装步骤
```bash
# 1. 安装依赖
pip3 install vllm

# 或者从源码编译（Jetson优化）
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip3 install -e .

# 2. 启动服务器
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen-7B \
  --quantization bitsandbytes \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9
```

#### 基准测试集成
- **接口**: HTTP API，RESTful
- **优点**: 完全兼容我们的系统设计
- **指标**: 可以获取详细的性能数据
- **频率控制**: 与我们的系统完全兼容

---

### 方案3：DeepSpeed + Transformers

#### 优势
- ✅ **Hugging Face集成**: 直接使用HF模型
- ✅ **易于使用**: 简单的Python API
- ✅ **支持分布式**: 可以扩展到多GPU
- ✅ **优化工具**: 深度优化和量化

#### 安装步骤
```bash
# 1. 安装依赖
pip3 install deepspeed transformers accelerate

# 2. 运行推理
python3 -m deepspeed.launcher \
  --model Qwen/Qwen-7B \
  --dtype float16 \
  --batch_size 1
```

---

### 方案4：保持TensorRT-LLM，使用公开资源

#### 公开Docker Hub镜像
```bash
# 检查Docker Hub上的TensorRT-LLM镜像
docker search tensorrt-llm

# 可能的替代
# 1. 社区维护的TensorRT-LLM Docker镜像
# 2. 包含TensorRT-LLM的通用AI镜像
# 3. Jetson专用的AI推理镜像
```

#### 预编译模型
- **NVIDIA NGC**: 某些模型可能是公开的
- **Hugging Face**: TensorRT-LLM引擎格式模型
- **社区贡献**: GitHub上的TensorRT-LLM项目

---

### 方案5：本地编译TensorRT-LLM（复杂但最灵活）

#### 优势
- ✅ **完全控制**: 可以自定义所有参数
- ✅ **无需Docker**: 直接在系统上编译
- ✅ **最新版本**: 可以获取最新的改进

#### 挑战
- ❌ **编译时间长**: 可能需要数小时
- ❌ **依赖复杂**: 需要TensorRT、CUDA等
- ❌ **可能失败**: Jetson特定的编译问题

#### 安装步骤
```bash
# 1. 获取TensorRT-LLM源码
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# 2. 安装TensorRT和CUDA
# 需要从NVIDIA开发者网站下载

# 3. 构建TensorRT-LLM
python3 scripts/build.py \
  --build_dir=build \
  --trt_root=/path/to/tensorrt \
  --cuda_root=/path/to/cuda
```

---

## 📊 方案对比分析

| 方案 | 安装难度 | 性能 | Jetson兼容 | 模型支持 | 无需账号 | 推荐度 |
|------|----------|------|-----------|---------|---------|--------|
| **llama.cpp** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| **vLLM** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ✅ | ⭐⭐⭐⭐ |
| **DeepSpeed** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⭐⭐ |
| **公开TensorRT** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ❌ | ⭐⭐ |
| **本地编译** | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ | ⭐ |

---

## 🎯 立即行动计划

### 推荐：llama.cpp (立即可行）

#### 为什么选择llama.cpp？
1. **无需任何账号**: 完全开源，MIT许可证
2. **Jetson优化**: 官方ARM64支持和SSE/NEON优化
3. **安装简单**: 一行命令即可安装或编译
4. **性能优秀**: 在边缘设备上与TensorRT-LLM性能相当
5. **丰富的量化支持**: INT4/INT8等，适合Jetson的内存限制
6. **活跃的社区**: 大量的模型和工具支持

#### 具体执行步骤
```bash
# 1. 安装llama.cpp (选择其中一种方式)

# 方式A: 使用pip安装（最简单）
pip3 install llama-cpp-python

# 方式B: 从源码编译（最优化）
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=OFF
cmake --build . --target llama-server

# 方式C: 使用预编译二进制（如果可用）
# 从releases页面下载ARM64版本

# 2. 下载并转换模型
# 从Hugging Face下载Qwen-7B
python3 scripts/download_and_convert_qwen.py

# 3. 运行基准测试
python3 src/benchmark_runner_llamacpp.py \
  --model /models/qwen-7b-q4.gguf \
  --batch_size 1 \
  --prompt_len 512 \
  --output_len 128 \
  --gpu_freq 846 \
  --cpu_freq 1479 \
  --emc_freq 1600

# 4. 运行实验4.1（真实测量）
python3 src/experiment_4_1_stability_real.py
```

---

## 🔄 临时替代策略

### 继续使用合成测试 + 深度分析

如果我们决定暂时不安装真实模型，可以：

#### 策略A：深化合成测试
1. **完善合成测试参数**：
   - 调整合成测试参数，使其更接近真实行为
   - 添加更多系统噪声和变化因素
   - 模拟Jetson特定的性能特征

2. **运行完整实验集**：
   - 实验4.2：单旋钮敏感性（GPU/CPU/EMC sweep）
   - 实验4.3：频率组合交互（Pareto前沿分析）
   - 实验4.4：Prefill/Decode阶段差异
   - 生成完整的数据集和分析

3. **优化系统参数**：
   - 基于合成测试结果优化调度策略
   - 验证我们的Phase-Aware DVFS算法
   - 测试不同的频率切换策略

#### 策略B：寻找社区资源
1. **查找公开的TensorRT-LLM资源**：
   - GitHub上的开源项目
   - Hugging Face上的预转换模型
   - 学术界的公开模型和数据集

2. **联系社区**：
   - Jetson论坛和社区
   - NVIDIA开发者社区
   - 学术合作网络

---

## 🎓 模型资源推荐

### Qwen-7B相关资源

#### Hugging Face官方仓库
- **模型**: https://huggingface.co/Qwen/Qwen-7B
- **聊天模型**: https://huggingface.co/Qwen/Qwen-7B-Chat
- **GGUF版本**: https://huggingface.co/TheBloke/Qwen-7B-GGUF

#### 其他运行时支持的Qwen模型
- **llama.cpp**: 官方支持Qwen系列
- **vLLM**: 支持Hugging Face格式的Qwen
- **ollama**: 集成了Qwen模型

### Llama-3-8B相关资源

#### Hugging Face官方仓库
- **模型**: https://huggingface.co/meta-llama/Meta-Llama-3-8B
- **GGUF版本**: https://huggingface.co/TheBloke/Meta-Llama-3-8B-GGUF

#### 其他运行时支持的Llama模型
- **llama.cpp**: 完美支持Llama系列（原项目）
- **vLLM**: 官方支持Llama系列
- **ollama**: 完整的Llama模型生态

---

## 💡 建议的决策路径

### 短期决策（今日）
**推荐选择**: llama.cpp

**理由**：
1. 可以立即开始，无需任何外部依赖
2. 与我们的系统完全兼容
3. Jetson Orin性能优秀
4. 支持我们需要的所有模型
5. 开发和调试成本低

### 中期决策（本周）
**如果llama.cpp工作良好**：
- 继续使用llama.cpp完成所有前置实验
- 基于真实数据验证我们的假设
- 生成Phase 3完成报告

**如果需要更高性能**：
- 考虑vLLM作为替代
- 评估性能差异
- 选择最佳方案继续

### 长期决策（后续）
**如果获得NGC账号**：
- 重新评估TensorRT-LLM方案
- 对比llama.cpp和TensorRT-LLM的性能
- 选择最适合生产环境的方案

---

## 🚀 下一步行动建议

### 选项A：选择llama.cpp（强烈推荐）⭐
**立即执行**：
1. 安装llama.cpp（pip或编译）
2. 下载Qwen-7B-GGUF模型
3. 修改benchmark_runner支持llama.cpp
4. 运行真实的实验4.1
5. 对比合成测试和真实测试结果

**预期时间**：
- 安装和配置：1-2小时
- 模型下载：10-30分钟
- 集成和测试：2-3小时
- **总计**：4-6小时可以开始真实测试

### 选项B：继续深化合成测试
**立即执行**：
1. 完善合成测试参数
2. 运行实验4.2、4.3、4.4
3. 生成完整的分析报告
4. 优化调度算法
5. 准备详细的实验文档

**预期时间**：
- 测试开发和执行：2-3小时
- 数据分析和可视化：1-2小时
- 报告撰写：1小时
- **总计**：4-6小时完成深入分析

### 选项C：并行进行（最佳）🎯
**同时进行**：
1. 开始llama.cpp安装和集成（主要路径）
2. 同时继续合成测试完善（备用路径）
3. 比较两种方法的结果
4. 选择最佳方案继续

**优势**：
- 最大化时间利用
- 降低项目风险
- 获得更全面的数据
- 验证我们的合成测试方法

---

**推荐决策**: **选择llama.cpp方案** ⭐

**理由**：
- 立即可行，无需任何外部资源
- 性能优秀，与TensorRT-LLM相当
- 完全开源，社区活跃
- 支持我们的所有目标模型
- 可以立即验证我们的核心假设

---

**文档状态**: 替代方案分析完成，等待用户决策  
**推荐**: llama.cpp作为主要替代方案  
**预期**: 4-6小时内可以开始真实测试