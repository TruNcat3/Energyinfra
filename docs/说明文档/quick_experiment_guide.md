# 🚀 最终频率实验指南

## 📋 问题诊断

之前实验结果显示频率设置成功了，但基准测试数据没有正确记录。主要问题是：
- ✅ GPU频率控制工作正常（306, 612, 918, 1300 MHz都设置成功）
- ✅ 温度读取正常（48-49°C）
- ❌ Python基准测试输出格式有问题，导致数据解析失败

## 🔧 解决方案

最终版本的实验脚本使用改进的基准测试：
- 📝 `scripts/_legacy/final_freq_experiment.sh` - 最终实验脚本
- 📝 `src/_legacy/final_benchmark.py` - 改进的基准测试（纯净CSV输出）

## 🚀 运行步骤

### 1. 在有sudo权限的终端中执行：

```bash
cd <仓库根目录>
./scripts/_legacy/final_freq_experiment.sh
```

### 2. 预期输出：

脚本会自动：
- 测试4个GPU频率：306、612、918、1300 MHz
- 每个频率下运行llama.cpp基准测试
- 实时显示进度和结果
- 生成详细的结果文件
- 自动恢复默认设置

### 3. 预期执行时间：约2-3分钟

## 📊 预期结果格式

### 实时输出示例：
```
🚀 Final Frequency Experiment
==========================================
📊 实验配置:
  模型: models/gguf/Phi-3-mini-4k-instruct-q4.gguf
  测试频率: 306 MHz 612 MHz 918 MHz 1300 MHz

==========================================
测试 1/4: 306 MHz
==========================================
🔄 设置GPU频率为 306 MHz...
   ✅ 频率设置成功
📊 实际频率: 306 MHz
🌡️  当前温度: 48.8°C
🚀 运行基准测试...
📊 基准测试结果:
   模型加载: 3.7s
   推理时间: 2.2s
   生成tokens: 20
   性能: 9.1 tokens/s
   输出: 2+2 is 4. Here...
```

### 结果文件示例：
```
频率    |  实际频率  |  tokens/s  |  温度  |  加载时间
---------|------------|------------|--------|----------
306 MHz  | 306 MHz    | 9.1 tokens/s | 48.8°C | 3.7s
612 MHz  | 612 MHz    | ?.? tokens/s | ?.?°C | ?.?s
918 MHz  | 918 MHz    | ?.? tokens/s | ?.?°C | ?.?s
1300 MHz | 1300 MHz   | ?.? tokens/s | ?.?°C | ?.?s
```

## 🎯 预期发现

基于之前的诊断测试，预期：
- **306 MHz**: 最低性能，约8-9 tokens/s
- **612 MHz**: 中低性能，约10-12 tokens/s
- **918 MHz**: 中等性能，约13-15 tokens/s
- **1300 MHz**: 最高性能，约16-20 tokens/s

## 📁 结果文件位置

实验完成后，结果将保存在：
- `data/frequency_experiment_results/final_results_YYYYMMDD_HHMMSS.txt` - 文本格式
- `data/frequency_experiment_results/final_results_YYYYMMDD_HHMMSS.json` - JSON格式（原始数据仅本地保留，由 `scripts/_legacy/final_freq_experiment.sh` 生成）

## ⚠️ 注意事项

1. **确保sudo权限激活**：在运行前执行 `sudo -v`
2. **模型文件存在**：确认Phi-3模型文件在正确位置
3. **系统稳定**：避免在实验期间运行其他重负载任务

## 🔍 故障排除

### 如果基准测试失败：
```bash
# 手动测试基准脚本
python3 src/_legacy/final_benchmark.py
```

### 如果频率设置失败：
```bash
# 检查sudo权限
sudo -v

# 检查频率文件权限
ls -la /sys/class/devfreq/17000000.gpu/
```

## 📈 后续分析

获得实验数据后，可进行以下分析：
1. 分析频率-性能关系
2. 计算能效比 (tokens/J)
3. 确定最优频率选择策略
4. 验证Phase-Aware DVFS可行性

---

**运行方式：**

```bash
cd <仓库根目录> && ./scripts/_legacy/final_freq_experiment.sh
```