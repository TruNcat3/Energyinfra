# 实验文档索引

## 统一档位术语

本节定义所有实验文档中使用的统一术语。Jetson Orin 出厂提供 4 个官方电源档位 (nvpmodel mode)，我们的研究在此基础增加自定义配置进行对比。

### Jetson 官方电源档位

| 档位名称 | nvpmodel Mode | GPU 上限 | CPU 上限 | CPU 核数 | 功耗目标 |
|----------|:---:|---------:|---------:|:---:|:---:|
| **MAXN** | 0 | 1300 MHz | 2201 MHz | 12 | 无限制 |
| **50W** | 3 | 816 MHz | 1497 MHz | 12 | 50W |
| **30W** (出厂默认) | 2 | 612 MHz | 1728 MHz | 8 | 30W |
| **15W** | 1 | 408 MHz | 1113 MHz | 4 | 15W |

> 注：nvpmodel 设置的是频率上限 (MAX_FREQ)，实际运行频率由调度器动态调节。-1 表示不设上限。

### 我们的自定义配置

| 配置名称 | GPU 频率 | CPU 频率 | 说明 |
|----------|:---:|:---:|------|
| **GPU-Min/CPU-Max** | 306 MHz | 2201 MHz | GPU 最低频 + CPU 最高频 (我们的策略) |
| **Energy-Focus** | 306 MHz | 1497 MHz | GPU 最低频 + CPU 中等频 |

### 文档用词规范

| 含义 | 统一用词 | 避免使用 |
|------|---------|---------|
| MAXN 档位 | MAXN | max, 满性能, all_high, max_performance |
| 30W 档位 | 30W (出厂默认) | mid, MODE_30W, default, 平衡 |
| 15W 档位 | 15W | low, MODE_15W, 低功耗, energy_efficient |
| 50W 档位 | 50W | high-mid, MODE_50W |
| 我们的策略 | GPU-Min/CPU-Max | 我们的配置, GPU306_CPU2201 |
| 首 Token 延迟 | TTFT | 首词延迟, Time To First Token |
| 每 Token 延迟 | TPOT | 解码延迟, Time Per Output Token |
| 吞吐量 | Throughput (tok/s) | tokens/s, tps |
| 能效 | Energy Efficiency | 节能, 能量效率 |

---

## 实验报告列表

| 报告 | 阶段 | 日期 | 核心内容 |
|------|------|------|---------|
| [Phase 1 完成报告](PHASE1_COMPLETION_REPORT.md) | 基础建设 | 2026-05-06 | 项目结构搭建、核心模块原型 |
| [Phase 2 完成报告](PHASE2_COMPLETION_REPORT.md) | 核心模块 | 2026-05-07 | DVFS 控制器、指标采集、基准测试框架 |
| [Phase 2 分析报告](PHASE2_ANALYSIS_REPORT.md) | 数据分析 | 2026-05-08 | 频率扫描结果、Pareto 前沿分析 |
| [Phase 3 完成报告](PHASE3_COMPLETION_REPORT.md) | 前置实验 | 2026-05-10 | 7 个前置实验验证 |
| [方案总结](FINAL_SOLUTIONS_SUMMARY.md) | 运行时方案 | 2026-05-09 | TensorRT-LLM / llama.cpp 方案对比 |

## 核心实验发现

### 真实模型多配置实验 (Phi-3-mini Q4, 2026-05-13)

**实验矩阵**: 4 GPU x 3 CPU x 5 workloads x 3 repeats = 180 runs

| 关键发现 | 数据 |
|---------|------|
| GPU 频率对吞吐量无影响 | 306→1300 MHz 仅 1.01x (p=0.64) |
| CPU 频率是性能瓶颈 | 1036→2201 MHz 达 1.86x |
| LLM 推理为 memory-bound | GPU 计算 not bottleneck |
| 最优配置 | GPU 918 MHz + CPU 2201 MHz = 9.4 tok/s |

### 对比档位性能摘要

| 档位 | TTFT (ms) | TPOT (ms) | Throughput (tok/s) |
|------|:---------:|:---------:|:------------------:|
| MAXN | 498.7 | 107.0 | 9.2 |
| 30W (出厂默认) | 141.6 | 142.5 | 7.0 |
| 15W | 878.1 | 196.7 | 5.0 |
| GPU-Min/CPU-Max (我们) | 773.8 | 107.6 | 9.0 |

> 关键结论：GPU-Min/CPU-Max 策略以 GPU 最低频率达到接近 MAXN 的吞吐量 (9.0 vs 9.2 tok/s)，可大幅降低 GPU 功耗。

---

## 图表索引

| 图表 | 路径 | 说明 |
|------|------|------|
| 档位吞吐量对比 | `figures/power_mode_comparison/throughput_comparison.png` | 4 个核心档位吞吐量柱状图 |
| TTFT vs TPOT 散点图 | `figures/power_mode_comparison/ttft_tpot_scatter.png` | 全部配置的延迟分布 |
| CPU/GPU 频率效应 | `figures/power_mode_comparison/cpu_gpu_effect.png` | CPU 和 GPU 对吞吐量的独立影响 |
| 负载分档位对比 | `figures/power_mode_comparison/workload_by_mode.png` | 按 workload 分组的档位对比 |
| 能效代理图 | `figures/power_mode_comparison/energy_efficiency_proxy.png` | throughput/GPU_freq 效率对比 |
| 综合仪表盘 | `figures/power_mode_comparison/power_mode_dashboard.png` | 4 合 1 综合对比 |
| 真实模型分析 | `figures/real_model_experiment/` | 5 张真实模型分析图 |
| Phase 5 对比 | `figures/phase5_comparison/` | 6 张 Phase 5 对比图 |
