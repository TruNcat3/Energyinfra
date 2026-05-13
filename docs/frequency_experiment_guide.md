# 🚀 Jetson Orin 频率控制实验指南

## 📋 实验目的
验证不同GPU频率对llama.cpp推理性能的影响，为Phase-Aware DVFS策略提供数据支持。

## ⏱️ 时间窗口
**重要**：sudo权限有效期约15分钟，请在此时间内完成以下步骤。

## 🔧 实验步骤

### 1. 频率控制测试（3-5分钟）

在**有sudo权限的终端**中执行：

```bash
cd /home/wt/work/Energyinfra
sudo ./scripts/test_frequency_control.sh
```

这个脚本会：
- ✅ 显示当前系统频率状态
- ✅ 列出所有可用的GPU频率
- ✅ 测试5个不同的GPU频率配置：
  - 306 MHz (最小)
  - 612 MHz (低中)
  - 918 MHz (中等)
  - 1122 MHz (高)
  - 1300 MHz (最大)
- ✅ 在每个频率下保持3秒并监控系统状态
- ✅ 最后恢复默认设置

### 2. 性能基准测试（手动执行）

由于频率设置需要sudo权限，但Python脚本不需要，我们采用以下策略：

**在sudo终端中设置特定频率，然后在普通终端运行基准测试：**

#### 测试1：低频率 (306 MHz)
```bash
# 在sudo终端中：
sudo sh -c 'echo "306000000" > /sys/class/devfreq/17000000.gpu/max_freq'
sudo sh -c 'echo "306000000" > /sys/class/devfreq/17000000.gpu/min_freq'
```

```bash
# 在普通终端中：
cd /home/wt/work/Energyinfra
python3 src/frequency_benchmark.py
```

#### 测试2：中等频率 (918 MHz)
```bash
# 在sudo终端中：
sudo sh -c 'echo "918000000" > /sys/class/devfreq/17000000.gpu/max_freq'
sudo sh -c 'echo "918000000" > /sys/class/devfreq/17000000.gpu/min_freq'
```

```bash
# 在普通终端中：
python3 src/frequency_benchmark.py
```

#### 测试3：高频率 (1300 MHz)
```bash
# 在sudo终端中：
sudo sh -c 'echo "1300500000" > /sys/class/devfreq/17000000.gpu/max_freq'
sudo sh -c 'echo "1300500000" > /sys/class/devfreq/17000000.gpu/min_freq'
```

```bash
# 在普通终端中：
python3 src/frequency_benchmark.py
```

### 3. 恢复默认设置

```bash
# 在sudo终端中：
sudo sh -c 'echo "1300500000" > /sys/class/devfreq/17000000.gpu/max_freq'
sudo sh -c 'echo "306000000" > /sys/class/devfreq/17000000.gpu/min_freq'
```

## 📊 预期结果

### 频率控制测试输出示例：
```
==========================================
Jetson Orin Frequency Control Test
==========================================
✅ Running with root privileges

📊 Current Frequency Status:
GPU:
  Current: 306 MHz
  Min: 306 MHz
  Max: 1300 MHz
  Governor: nvhost_podgov

...
🔄 Testing: GPU Min (306 MHz)
  ✅ Set to: 306 MHz
  📊 Actual: 306 MHz
  🌡️  Temperature: 48°C
```

### 性能基准测试输出示例：
```
============================================================
Frequency Benchmark for llama.cpp
============================================================

📊 Current System Frequencies:
  GPU: 306 MHz
  CPU: 947 MHz (avg)
  CPU Cores: 12

🚀 Running Benchmark...
  ✅ Load time: 3.6s
  ✅ Inference time: 1.2s
  ✅ Generated: 30 tokens
  ✅ Performance: 25.0 tokens/s
```

## 📈 数据分析

测试完成后，结果将保存在：
- `data/frequency_benchmarks/frequency_benchmark_YYYYMMDD_HHMMSS.json`

预期观察到的性能趋势：
- **306 MHz**: 最低性能，最低功耗
- **918 MHz**: 中等性能，中等功耗
- **1300 MHz**: 最高性能，最高功耗

## 🎯 实验目标

1. **验证频率控制有效性**：确认GPU频率可以正确设置
2. **测量性能差异**：量化不同频率下的tokens/s差异
3. **评估功耗影响**：结合tegrastats监控功耗变化
4. **确定最佳频率**：为不同工作负载找到最优频率点

## ⚠️ 注意事项

1. **时间限制**：sudo权限约15分钟，请快速操作
2. **系统稳定性**：频繁改变频率可能影响系统稳定性
3. **温度监控**：高频率下注意温度，避免过热
4. **数据保存**：每次测试后确认JSON文件正确保存

## 🔍 故障排除

### 权限错误
```bash
# 确认sudo权限仍然有效
sudo -v
```

### 频率设置失败
```bash
# 检查当前频率状态
cat /sys/class/devfreq/17000000.gpu/cur_freq
cat /sys/class/devfreq/17000000.gpu/governor
```

### 模型加载失败
```bash
# 确认模型文件存在
ls -lh /home/wt/work/Energyinfra/models/gguf/
```

## 📝 实验记录模板

完成实验后，请记录：

| GPU频率 | 平均tokens/s | 平均功耗 | 温度 | 备注 |
|---------|-------------|----------|------|------|
| 306 MHz | ? | ? | ? | |
| 612 MHz | ? | ? | ? | |
| 918 MHz | ? | ? | ? | |
| 1122 MHz | ? | ? | ? | |
| 1300 MHz | ? | ? | ? | |

## 🚀 下一步

基于这些数据，我们将：
1. 分析频率-性能关系曲线
2. 计算能效比 (tokens/J)
3. 确定最优频率选择策略
4. 验证Phase-Aware DVFS的可行性

---

**实验准备就绪！请在sudo权限有效期内开始执行。**