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
| **GPU-Min/CPU-Max** | 306 MHz | 2201 MHz | GPU 最低频 + CPU 最高频 |
| **Energy-Focus** | 306 MHz | 1497 MHz | GPU 最低频 + CPU 中等频 |

### 文档用词规范

| 含义 | 统一用词 | 避免使用 |
|------|---------|---------|
| MAXN 档位 | MAXN | max, 满性能, all_high, max_performance |
| 30W 档位 | 30W (出厂默认) | mid, MODE_30W, default, 平衡 |
| 15W 档位 | 15W | low, MODE_15W, 低功耗, energy_efficient |
| 50W 档位 | 50W | high-mid, MODE_50W |
| 首 Token 延迟 | TTFT | 首词延迟, Time To First Token |
| 每 Token 延迟 | TPOT | 解码延迟, Time Per Output Token |
| 吞吐量 | Throughput (tok/s) | tokens/s, tps |
| 能效 | Energy Efficiency | 节能, 能量效率 |

---

## 实验文档结构

### 总览文档

| 文档 | 内容 |
|------|------|
| [实验总览](实验总览.md) | 实验环境、频率配置空间、功耗采集方法、实验设计矩阵 |

### 分实验报告

| 文档 | 阶段 | 数据 | 核心结论 |
|------|------|------|---------|
| [实验1-混合阶段能耗分析](实验1-混合阶段能耗分析.md) | Mixed Phase | 170 runs | ⚠️ GPU governor bug — 结论已过时，见下方修正 |
| [实验2-分阶段能耗分析](实验2-分阶段能耗分析.md) | Prefill + Decode | 345 runs | ⚠️ GPU governor bug — 结论已过时，见下方修正 |
| [实验3-扩展负载汇率表](实验3-扩展负载汇率表.md) | Mixed + Decode | 672 runs | GPU 918MHz 最优 (10/14 workloads)；28 DVFS rules |

### 历史归档

| 报告 | 阶段 | 日期 | 核心内容 |
|------|------|------|---------|
| [Phase 1 完成报告](PHASE1_COMPLETION_REPORT.md) | 基础建设 | 2026-05-06 | 项目结构搭建、核心模块原型 |
| [Phase 2 完成报告](PHASE2_COMPLETION_REPORT.md) | 核心模块 | 2026-05-07 | DVFS 控制器、指标采集、基准测试框架 |
| [Phase 2 分析报告](PHASE2_ANALYSIS_REPORT.md) | 数据分析 | 2026-05-08 | 频率扫描结果、Pareto 前沿分析 |
| [Phase 3 完成报告](PHASE3_COMPLETION_REPORT.md) | 前置实验 | 2026-05-10 | 7 个前置实验验证 |
| [方案总结](FINAL_SOLUTIONS_SUMMARY.md) | 运行时方案 | 2026-05-09 | TensorRT-LLM / llama.cpp 方案对比 |

> **重要说明**：2026-05-13 的实验数据因 llama-cpp-python 未编译 CUDA 支持，所有推理实际运行在 CPU 上。
> 该问题已于 2026-05-14 修复。详见 [实验总览](实验总览.md) 中的历史说明。

---

## 核心实验结论 (修正后, 2026-05-15)

> **重要修正 (2026-05-15)**：实验 1 和实验 2 使用的 `userspace` GPU governor 无法在推理负载下锁定频率，
> GPU 会自发跳频至 1300MHz，导致 "GPU 频率无影响" 的错误结论。修正后使用 `performance` governor，
> 数据来自扩展实验 (672 runs, 14 workloads)。

### GPU 频率确实影响性能

| GPU 频率 | Decode TPS (CPU=1036) | GPU SoC 功耗 |
|:---:|:---:|:---:|
| 306 MHz | ~12 tok/s | ~8 W |
| 612 MHz | ~23 tok/s | ~13 W |
| 918 MHz | ~30 tok/s | ~19 W |
| 1300 MHz | ~41 tok/s | ~30 W |

### DVFS 最优配置 (修正后)

| 阶段 | 负载特征 | 最优 GPU | 最优 CPU | 理由 |
|------|---------|:---:|:---:|------|
| Mixed | output ≤ 64 tokens | 612 MHz | 1036 MHz | 低功耗即可满足短输出 |
| Mixed | output ≥ 128 tokens | 918 MHz | 1036 MHz | 高吞吐摊薄功耗 |
| Decode | output ≤ 64 tokens | 612 MHz | 1036 MHz | 能效最优 |
| Decode | output ≥ 128 tokens | 918 MHz | 1036 MHz | 吞吐优先 |

### 关键规律 (修正后)

1. **GPU 频率显著影响性能**：306→918 MHz 吞吐提升 2.5×，之前的"无影响"结论是 governor bug 所致
2. **GPU 612MHz 适合短输出**：低功耗下吞吐量已足够，能效最优
3. **GPU 918MHz 适合长输出**：在 14 个负载中 10 个的最优配置为 GPU 918MHz
4. **Decode 仍是功耗主体**：占推理总功耗 ~73%
5. **数据来源**：`data/energy_profiling/expanded_profiling_20260515_031015.csv`
