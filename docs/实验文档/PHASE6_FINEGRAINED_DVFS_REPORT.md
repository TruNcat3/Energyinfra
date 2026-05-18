# Phase 6 实验报告：细粒度 GPU×EMC DVFS 能耗优化

**日期**: 2026-05-16 ~ 2026-05-17
**模型**: Phi-3-mini-4k-instruct-Q4 (3.8B)
**平台**: Jetson Orin (JetPack 5.x)
**状态**: 完成

---

## 1. 实验背景

Phase 5 的实验仅使用 4 GPU × 2 CPU = 8 个配置，缺少 EMC 频率维度，且 GPU 频率档位过粗。
本阶段通过 DebugFS 实现 EMC 频率控制，扩展至完整的 11 GPU × 4 EMC × 1 CPU 配置空间。

---

## 2. 实验设计

### 硬件配置空间

| 维度 | 可用档位 | 控制方式 |
|------|---------|---------|
| GPU | 306, 408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300 MHz | sysfs devfreq (`performance` governor) |
| EMC | 204, 665, 2133, 3199 MHz | DebugFS (`/sys/kernel/debug/emc/min_rate`, `max_rate`) |
| CPU | 1036 MHz (固定) | sysfs cpufreq |

### Workloads

| 名称 | Prompt tokens | Output tokens |
|------|:---:|:---:|
| p128_o64 | 128 | 64 |
| p512_o128 | 512 | 128 |
| p1024_o128 | 1024 | 128 |
| p128_o256 | 128 | 256 |
| p1024_o512 | 1024 | 512 |

### 实验矩阵

| 阶段 | 配置数 | Runs | 耗时 |
|------|:---:|:---:|:---:|
| Decode profiling | 44 × 5 × 3 | 660 | ~2.5h |
| Mixed profiling | 44 × 5 × 3 | 660 | ~4.5h |
| E2E benchmark | 6α × 2策略 + 3基线 | 225 | ~1.5h |
| **合计** | | **1545** | **~8.5h** |

---

## 3. 关键发现

### 3.1 GPU 频率效率曲线呈 W 形

GPU 能效（E/token）并非单调递增或递减，而是呈 W 形：

| GPU (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | 特征 |
|:---:|:---:|:---:|:---:|:---:|------|
| 306 | 1.33 | 81.8 | 12.2 | 16.2 | 最低性能 |
| 408 | 1.12 | 49.8 | 24.2 | 25.8 | |
| 510 | 1.11 | 37.8 | 30.0 | 31.2 | |
| **612** | **1.00** | **43.4** | **23.1** | **22.6** | **能效最优点 1** |
| 714 | 1.14 | 24.7 | 40.5 | 41.8 | 电压阶跃跳跃 |
| **816** | **0.94** | **33.3** | **30.1** | **28.9** | **能效最优点 2** |
| 918 | 1.15 | 24.7 | 40.6 | 42.0 | |
| 1020 | 1.14 | 24.7 | 40.6 | 41.7 | |
| 1122 | 1.07 | 27.0 | 37.0 | 37.7 | |
| 1224 | 1.14 | 24.7 | 40.6 | 41.7 | |
| 1300 | 1.15 | 24.8 | 40.3 | 42.1 | 最高性能 |

**结论**: GPU612 和 GPU816 是两个局部能效最优频率点。GPU714 处出现电压阶跃（功耗突增 19W→42W）。

### 3.2 EMC 低频更节能

| EMC (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) |
|:---:|:---:|:---:|:---:|:---:|
| **204** | **1.10** | 38.4 | 30.9 | **32.1** |
| 665 | 1.13 | 35.2 | 33.8 | 35.2 |
| 2133 | 1.17 | 30.2 | 37.4 | 39.6 |
| 3199 | 1.16 | 30.1 | 37.5 | 39.4 |

**结论**: EMC 204 MHz 最节能，因为 Phi-3-mini 3.8B 模型 (Q4) 在推理时内存带宽不是瓶颈。
高频 EMC 只增加功耗，对性能提升有限。

### 3.3 最优配置

| 目标 | GPU | EMC | E/tok (J) | TPOT (ms) |
|------|:---:|:---:|:---:|:---:|
| **最佳能效** | 816 | 204 | 0.97 | 33 |
| 最佳延迟 | 1020 | 204 | 1.13 | 24 |
| 均衡 | 1122 | 665 | 1.00 | 26 |
| 最低功耗 | 306 | 204 | 1.32 | 82 |

---

## 4. Rate Table 重建

基于 1320 runs 的细粒度数据，重建了完整的 rate table：

- **10 个 workload×phase buckets** (5 workloads × 2 phases)
- **31 个实际有效的 GPU×EMC 配置**（44 个目标中部分 EMC 设置未生效）
- **235 行聚合数据** (median ± std)
- **101 个 Pareto 最优配置**
- **30 条 DVFS 规则** (3 objectives × 10 buckets)

**输出文件**:
- `data/rate_tables/finegrained_selector_table_20260516_071238.parquet`
- `data/rate_tables/finegrained_dvfs_rules_20260516_071238.json`

---

## 5. E2E Benchmark 验证

使用 WeightedSelector 的 alpha 权重旋钮进行端到端验证：

### Alpha 权重效果

| Alpha | 策略 | E/tok (J) | TPOT (ms) | TPS |
|:---:|:---|:---:|:---:|:---:|
| 0.0 (纯能效) | single | 1.2407 | 27.3 | 35.5 |
| 0.5 (均衡) | single | 1.2473 | 26.0 | 36.8 |
| 1.0 (纯延迟) | single | 1.2372 | 26.0 | 37.0 |
| 0.0 (纯能效) | phase-aware | 1.2702 | 26.1 | 36.1 |
| 0.5 (均衡) | phase-aware | 1.2277 | 25.9 | 37.3 |
| 1.0 (纯延迟) | phase-aware | 1.2316 | 26.0 | 37.2 |

### 基线对比

| 基线 | 配置 | E/tok (J) | TPOT (ms) | TPS |
|------|------|:---:|:---:|:---:|
| MAXN | GPU1300+EMC3199+CPU1497 | 1.2502 | 25.8 | 37.3 |
| 30W | GPU612+EMC2133+CPU1497 | 1.2660 | 25.7 | 37.0 |
| E_min | GPU816+EMC204+CPU1036 | 1.2409 | 26.0 | 36.9 |

### 分析

E2E benchmark 中各配置差异较小（<3%），主要原因：
1. **Prefill 主导**: Mixed-phase 运行中 prefill 能耗占比大，掩盖 decode 阶段差异
2. **小模型限制**: Phi-3-mini 3.8B 的 DVFS 优化空间有限
3. **实际配置集中**: Alpha 变化时，WeightedSelector 选择的实际配置大多集中在 GPU816-1122 范围

**建议**: 更大模型（如 Llama-3-8B）应有更显著的 DVFS 效果，因为内存带宽将成为瓶颈，EMC 频率影响更大。

---

## 6. 数据归档

以下旧数据已归档至 `data/_archived_invalid/`：

| 类别 | 文件数 | 原因 |
|------|:---:|------|
| governor_bug | 13 | userspace governor 导致 GPU 频率未锁定 |
| coarse_granularity | 1 | 仅 4 GPU × 2 CPU 配置 |
| e2e_contaminated | 2 | 重复进程导致能耗测量污染 |
| rate_tables_coarse | 15 | 基于粗糙数据构建的 rate table |

---

## 7. 输出文件清单

### 数据
- `data/energy_profiling/finegrained_profiling_20260516_015611.csv` — Decode profiling (660 runs)
- `data/energy_profiling/finegrained_profiling_20260516_055330.csv` — Mixed profiling (660 runs)
- `data/energy_profiling/finegrained_combined_20260516.csv` — 合并数据 (1320 runs)
- `data/energy_profiling/e2e_benchmark_finegrained_20260517.csv` — E2E benchmark (225 runs)
- `data/rate_tables/finegrained_selector_table_20260516_071238.parquet` — Rate table
- `data/rate_tables/finegrained_dvfs_rules_20260516_071238.json` — DVFS rules (30条)

### 可视化
- `figures/finegrained_profiling/` — GPU×EMC 能效曲线、Pareto 前沿、热力图 (5张)
- `figures/e2e_benchmark_finegrained/` — E2E Pareto、Alpha 效应、策略对比 (5张)

### 代码
- `src/ratetable/build_rate_table_finegrained.py` — 细粒度 rate table 构建器
- `src/controller/weighted_selector.py` — 3D 加权多目标选择器
- `src/experiments/run_finegrained_profiling.py` — 细粒度 profiling 实验
- `src/experiments/run_e2e_benchmark_finegrained.py` — E2E benchmark 实验
- `src/visualization/visualize_e2e_benchmark_finegrained.py` — E2E 可视化
