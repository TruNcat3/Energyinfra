# LLM 运行时设置指南
# LLM Runtime Setup Guide

## 🎯 当前运行时配置

### 主要运行时：llama.cpp
- **状态**: ✅ 已安装
- **版本**: b4234
- **原因**: TensorRT-LLM账号未获取，使用llama.cpp作为替代方案
- **优势**: 开源、易安装、支持多平台、社区活跃

### 备选运行时
- **TensorRT-LLM**: 需要账号访问，暂不可用
- **vLLM**: 未来可考虑的替代方案

---

## 📦 llama.cpp 安装状态

### 已完成
✅ **llama.cpp 已安装**
- 位置: `/home/wt/work/llama.cpp`
- 可执行文件: `llama-cli`
- 支持的模型格式: GGUF

### 安装验证
```bash
# 检查 llama.cpp 版本
llama-cli --version

# 检查支持的操作
llama-cli --help
```

---

## 🔧 当前工作模式

### Phase 3: Synthetic Benchmark (已完成)
- **数据来源**: 合成测试框架
- **目的**: 验证分析管道和系统架构
- **状态**: ✅ 完成并验证所有7个前置实验

### Phase 4: Real Model Integration (准备中)
- **数据来源**: llama.cpp 真实模型推理
- **目的**: 获取真实的Jetson Orin性能数据
- **状态**: ⏳ 等待模型下载和配置

---

## 🚀 从 Synthetic 到 Real Model 迁移

### 当前数据资产 (Synthetic)
- ✅ 45个实验数据点
- ✅ 8个CSV数据文件
- ✅ 完整的分析和可视化报告
- ✅ 验证的分析管道

### 待生成数据资产 (Real llama.cpp)
- ⏳ 真实Jetson Orin性能测量
- ⏳ 实际频率控制效果
- ⏳ 真实功耗和温度数据
- ⏣ llama.cpp特定优化配置

---

## 📋 需要更新的配置

### 1. 运行时配置更新
**已完成**:
- `configs/platform.yaml`: 主要运行时已更新为llama.cpp
- `configs/selector.yaml`: 运行时选项已包含llama.cpp
- `configs/phase3_experiments.yaml`: 描述已更新

**待更新**:
- 实验配置中的llama.cpp特定参数
- benchmark runner中的llama.cpp命令生成

### 2. benchmark_runner.py 适配
**当前状态**: 主要支持TensorRT-LLM
**需要修改**:
- 添加llama.cpp命令生成逻辑
- 适配llama.cpp输出格式解析
- 调整参数和配置结构

### 3. 工作负载配置更新
**当前配置**: 针对TensorRT-LLM优化的workloads
**需要调整**:
- 模型格式: TensorRT engine → GGUF
- 推理参数: 适配llama.cpp参数
- 批处理支持: 调整llama.cpp的批处理实现

---

## 🔧 llama.cpp 实验配置

### 基本命令格式
```bash
llama-cli \
  --model /path/to/model.gguf \
  --prompt "Your prompt here" \
  --n-predict 128 \
  --ctx-size 2048 \
  --n-gpu-layers 99 \
  --threads 4 \
  --temp 0.7 \
  --top-p 0.9 \
  --repeat-penalty 1.1
```

### 频率控制集成
```bash
# 设置频率
sudo jetson_clocks --set <freq_config>

# 运行llama.cpp benchmark
sudo tegrastats --interval 1000 --logfile power.log &
TEGRASTATS_PID=$!

llama-cli --model model.gguf --benchmark

# 停止tegrastats
kill $TEGRASTATS_PID
```

### 性能指标收集
- **TTFT**: 首token生成时间 (llama.cpp 输出中提取)
- **TPOT**: 输出token平均时间 (llama.cpp 输出中提取)
- **Throughput**: tokens/second
- **Power**: 从tegrastats日志中解析
- **Temperature**: 从tegrastats日志中解析

---

## 📊 数据格式转换

### TensorRT-LLM 输出 → llama.cpp 输出
**TensorRT-LLM 格式**:
```
[2024-05-08 12:00:00] [INFO] TTFT: 10.5 ms
[2024-05-08 12:00:00] [INFO] TPOT: 7.2 ms
[2024-05-08 12:00:00] [INFO] Throughput: 120.5 tok/s
```

**llama.cpp 格式**:
```
llama_perf: 10.5 ms / 99 tokens ( 1.13 ms per token,   88.0 tokens/s)
```

**转换逻辑**:
```python
# 提取TTFT (首token时间)
ttft = match_first_token_time

# 提取TPOT (平均per-token时间)
tpot = average_time_per_token

# 计算throughput
throughput = 1000 / average_time_per_token  # tokens/s
```

---

## 🎯 迁移计划

### 阶段1: llama.cpp 环境验证 (当前)
- ✅ llama.cpp 安装
- ✅ 基本命令测试
- ⏳ 模型下载和配置
- ⏳ 简单性能基准测试

### 阶段2: benchmark_runner.py 适配
- ⏳ 添加llama.cpp命令生成
- ⏳ 实现llama.cpp输出解析
- ⏳ 更新workload配置参数
- ⏳ 集成tegrastats数据收集

### 阶段3: 真实实验执行
- ⏳ 运行频率敏感性实验
- ⏳ 运行配置组合实验
- ⏳ 运行phase-aware实验
- ⏳ 生成真实性能数据

### 阶段4: 数据分析和验证
- ⏳ 对比synthetic vs real数据
- ⏳ 验证分析管道兼容性
- ⏳ 更新rate table和selector
- ⏳ 生成真实性能报告

---

## 🚨 注意事项

### llama.cpp 特定限制
1. **模型格式**: 必须使用GGUF格式
2. **批处理**: llama.cpp的批处理实现可能与TensorRT-LLM不同
3. **内存使用**: GGUF格式可能有不同的内存占用特性
4. **精度**: GGUF通常使用INT4/INT8量化，TensorRT-LLM支持FP16

### Jetson Orin 优化
1. **GPU利用**: 确保n-gpu-layers参数正确
2. **线程配置**: 根据CPU核心数调整threads参数
3. **内存管理**: 监控VRAM使用，避免OOM
4. **散热管理**: 注意温度和功耗限制

---

## 📈 预期差异

### Synthetic vs Real Data
| 方面 | Synthetic | llama.cpp Real |
|------|-----------|----------------|
| **GPU影响** | 主要体现GPU频率 | 真实GPU计算性能 |
| **CPU影响** | 简化建模 | 实际CPU开销 |
| **EMC影响** | 低估 | 真实内存带宽 |
| **切换开销** | 模拟 | 实际测量值 |
| **热效应** | 简化 | 真实温度变化 |

### 验证重点
1. **Phase-Aware DVFS**: Real data是否支持不同阶段不同频率
2. **SLO约束**: 真实数据下的SLO满足率
3. **优化空间**: 实际能效提升潜力
4. **系统稳定性**: 真实硬件下的可靠性

---

## 🔄 回退策略

### 如果llama.cpp遇到问题
1. **vLLM**: 考虑使用vLLM作为替代
2. **TensorRT-LLM**: 等待账号获取后使用原始方案
3. **框架优化**: 自定义简单的benchmark程序

### 如果数据质量不佳
1. **扩展实验**: 增加更多配置和工作负载
2. **优化测量**: 改进数据收集方法
3. **模型调整**: 尝试不同模型或量化级别

---

## 📞 支持和故障排除

### 常见问题
1. **模型下载**: 从Hugging Face下载GGUF模型
2. **权限问题**: 确保对GPU设备的访问权限
3. **内存不足**: 调整ctx-size或使用更小的模型
4. **速度慢**: 检查GPU频率和n-gpu-layers设置

### 调试工具
```bash
# 检查GPU状态
nvidia-smi

# 检查Jetson频率
sudo jetson_clocks --show

# 监控系统状态
tegrastats

# 测试llama.cpp
llama-cli --model tinyllama-1.1b-chat.Q5_K_M.gguf --prompt "test" --n-predict 10
```

---

**更新时间**: 2026-05-08
**维护状态**: Active
**负责人**: Energyinfra Team