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
bash scripts/_legacy/download_qwen_gguf.sh

# 这将下载约2.7GB的Q4_K_M量化模型
```

### 选项2：测试模型

```bash
# 测试下载的模型
bash scripts/active/test_llamacpp_model.sh models/gguf/qwen-7b-chat-q4_k_m.gguf
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

### 步骤1：基础验证
1. 下载Qwen-7B-Q4_K_M模型
2. 运行快速测试验证功能
3. 修改实验脚本支持llama.cpp

### 步骤2：真实实验4.1
1. 使用llama.cpp运行测量稳定性实验
2. 对比合成测试和真实测试结果
3. 验证核心假设

### 步骤3：频率扫描实验
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

基本使用流程：
1. 运行下载脚本获取Qwen-7B模型
2. 运行测试脚本验证功能
3. 运行真实实验

祝您实验顺利！🚀
