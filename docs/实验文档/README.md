# 实验文档索引

## 统一档位术语

本节定义所有实验文档中使用的统一术语。Jetson Orin 出厂提供 4 个官方电源档位 (nvpmodel mode)，本研究在此基础增加自定义配置进行对比。

### Jetson 官方电源档位

| 档位名称 | nvpmodel Mode | GPU 上限 | CPU 上限 | CPU 核数 | 功耗目标 |
|----------|:---:|---------:|---------:|:---:|:---:|
| **MAXN** | 0 | 1300 MHz | 2201 MHz | 12 | 无限制 |
| **50W** | 3 | 816 MHz | 1497 MHz | 12 | 50W |
| **30W** (出厂默认) | 2 | 612 MHz | 1728 MHz | 8 | 30W |
| **15W** | 1 | 408 MHz | 1113 MHz | 4 | 15W |

> 注：nvpmodel 设置的是频率上限 (MAX_FREQ)，实际运行频率由调度器动态调节。-1 表示不设上限。

### 本项目的自定义配置

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

### 专题分析与报告

| 报告 | 日期 | 核心内容 |
|------|------|---------|
| [实验 4：细粒度 GPU×EMC 能耗建模](实验4-细粒度GPUxEMC能耗建模.md) | 2026-05-16 | 11 GPU × 4 EMC 细粒度 profiling、三维汇率表 |
| [E2E 收益归因分析](E2E收益归因分析.md) | 2026-05-19 | E2E baseline 差异归因、token/J 收益来源分析 |
| [相关工作对比](related_work_comparison.md) | 2026-06-17 | DVFS / Serving / 边缘部署 / 多目标优化对比 |

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
5. **数据来源**：`data/energy_profiling/expanded_profiling_20260515_031015.csv`（原始数据仅本地保留，可由 `src/experiments/run_expanded_profiling.py` 重新生成）
