# Jetson LLM Energy Profiling - Project Overview

**项目状态**: Phase 5 (高级优化) 进行中 — 真实模型多配置实验已完成 (180 runs)

## 项目概览

本项目旨在构建面向 Jetson 边端 GPU 的 LLM 推理能耗 profiling 与自适应配置选择系统。系统能够自动执行 workload × frequency 扫描，采集性能、功耗和温度数据，构建能耗汇率表，并在给定 SLO 约束下输出最优的 GPU、CPU 和 EMC 频率配置。

### 核心目标

1. **能耗汇率表构建**: 建立 workload-aware 的能耗汇率表 (完成)
2. **自适应配置选择**: 设计 SLO-aware 的配置选择器 (完成)
3. **在线控制系统**: 实现可离线建表、在线查表、自动选频的系统 (进行中)

### 技术特点

- **目标平台**: Jetson Orin (可扩展到其他 Jetson 设备)
- **主要运行时**: llama.cpp (已验证), TensorRT-LLM (设计中)
- **优化策略**: GPU/CPU/EMC 频率自适应调节
- **分阶段优化**: Prefill/Decode 分阶段 DVFS 策略 (已验证)
- **自适应决策**: 根据负载特征自动决定是否启用 Phase-Aware 切换 (已实现)
- **SLO约束**: 在满足 TTFT、TPOT、功耗、温度约束下优化能效

## 项目结构

```
Energyinfra/
├── src/                        # Python 源代码 (按功能分层)
│   ├── controller/            # 调频器/调度器 - DVFS控制
│   │   ├── freq_controller.py       # 频率控制 (sysfs/jetson_clocks)
│   │   ├── jetson_power_modes.py    # Jetson电源模式管理
│   │   ├── phase_aware_policy.py    # Phase-Aware DVFS策略
│   │   ├── phase_controller.py      # DVFS在线控制器
│   │   └── select_config.py         # SLO-Aware配置选择器
│   ├── metrics/               # 性能收集器
│   │   ├── metrics_collector.py     # tegrastats指标采集
│   │   ├── parse_logs.py            # 日志解析
│   │   └── system_monitor.py        # 系统监控
│   ├── benchmark/             # 基准测试框架
│   │   ├── synthetic_benchmark.py   # 合成基准测试
│   │   ├── benchmark_runner.py      # 基准运行器
│   │   ├── sweep_runner.py          # 参数扫描编排
│   │   └── llama_cpp_runner.py      # llama.cpp推理Runner
│   ├── ratetable/             # 能耗汇率表
│   │   ├── build_rate_table.py      # 汇率表构建
│   │   └── evaluate_selector.py     # 选择器评估
│   ├── visualization/         # 可视化分析 (10个)
│   │   ├── visualize_results.py     # 实验结果可视化
│   │   ├── visualize_phase_aware.py # Phase-Aware可视化
│   │   ├── visualize_phase5_p0.py   # Phase 5综合对比
│   │   ├── visualize_real_experiment.py # 真实模型实验图表
│   │   └── analyze_real_experiment.py   # 实验数据分析
│   ├── experiments/           # 实验脚本 (9个)
│   │   ├── run_phase_aware_experiment.py   # Phase-Aware验证
│   │   ├── run_real_model_experiment.py    # 真实模型多配置实验
│   │   ├── run_baseline_comparison.py      # Baseline对比
│   │   └── run_all_prelim_experiments.py   # 前置实验
│   └── _legacy/               # 归档废弃代码 (5个)
├── configs/                    # 配置文件
├── data/                       # 实验数据输出
├── figures/                    # 可视化图表 (27张)
├── scripts/
│   ├── active/                # 活跃脚本 (4个)
│   └── _legacy/               # 归档脚本 (9个)
├── docs/                       # 文档 (按类型分层)
│   ├── 任务书/                # 任务规格书
│   ├── 开发文档/              # 开发跟踪 (checklist, 进度, 状态)
│   ├── 说明文档/              # 使用指南 (runtime, setup, integration)
│   └── 实验文档/              # 实验报告 (PHASE1-5)
├── README.md
└── CLAUDE.md
```

## 核心实验结果

### Phase-Aware DVFS 验证 (合成基准, 2026-05-12)

| 策略 | 能量/Token (J) | TTFT (ms) | TPOT (ms) | SLO 满足率 |
|------|---------------|-----------|-----------|-----------|
| Default (mid) | 0.164 | 121 | 7.6 | 82% |
| Max Performance | 0.110 | 124 | 4.5 | 42% |
| Energy Efficient | 0.326 | 125 | 16.9 | 92% |
| **Phase-Aware (Ours)** | **0.115** | **77** | **6.7** | **80%** |
| Oracle | 0.152 | 124 | 6.9 | 76% |

**关键发现**:
- Phase-Aware 相比 Default 节省 **30% 能量**
- Phase-Aware 同时改善 **TTFT 36%** (高频 GPU 加速 prefill)
- SLO 满足率维持在 80% (Max Performance 仅 42%)

### Selector 评估 (P0 修复后, 2026-05-13)

| 策略 | Mean Regret vs Oracle | SLO Violation | Anomalies |
|------|----------------------|---------------|-----------|
| Oracle | 0.00% | 0% | 0 |
| Ours (SLO-Aware) | 0.00% | 0% | 0 |
| Fixed Best | 1.61% | 0% | 0 |
| MaxN | 8.21% | 0% | 0 |
| All Mid | 22.11% | 0% | 0 |

### 真实模型多配置实验 (Phi-3-mini Q4, 2026-05-13)

**实验矩阵**: 4 GPU × 3 CPU × 5 workloads × 3 repeats = 180 runs

| GPU Freq | CPU Freq | TTFT (ms) | TPOT (ms) | TPS |
|----------|----------|-----------|-----------|-----|
| 306 MHz | 1036 MHz | 878.1 | 196.7 | 5.0 |
| 306 MHz | 2201 MHz | 773.8 | 107.6 | 9.0 |
| 918 MHz | 2201 MHz | 430.8 | 105.2 | **9.4** |
| 1300 MHz | 1036 MHz | 830.7 | 198.4 | 4.9 |
| 1300 MHz | 2201 MHz | 498.7 | 107.0 | 9.2 |

**关键发现**:
- **GPU 频率无影响**: 306→1300 MHz 仅 1.01x 吞吐量 (memory-bound 确认)
- **CPU 频率是瓶颈**: 1036→2201 MHz 达 1.86x 吞吐量
- **最优配置**: GPU918_CPU2201 = 9.4 tok/s (并非最高 GPU 频率)
- **TTFT**: CPU 高频可降低 26.2%, GPU 高频无效果

### 7 个前置实验核心结论

| 实验 | 核心发现 |
|------|----------|
| 4.1 测量稳定性 | TPOT CV=1.0% 优秀, TTFT CV=9.2% 可接受 |
| 4.2 单旋钮敏感性 | GPU 378MHz 最能效 (7.36 tok/W), 1428MHz 最吞吐 |
| 4.3 频率组合交互 | 67% 配置满足 SLO, mid-high 优于 max-all |
| 4.4 Prefill/Decode 差异 | **26.97x** 阶段能效差异, Phase-Aware 必要性验证 |
| 4.5 负载可预测性 | workload 特征可预测最优配置 |
| 4.6 切换开销 | GPU 切换 50ms, 可隐藏于 phase boundary |
| 4.7 SLO 端到端 | 67% 配置满足 SLO, 能效优先可行 |

## Phase 5 完成情况

| 任务 | 内容 | 状态 |
|------|------|------|
| P0 评估可信度 | Pareto/phase-energy/fixed_best 修复 | 完成 |
| P1 Jetson Baseline | jetson_power_modes + 9种 baseline 定义 | 完成 |
| P2 真实模型 Runner | llama_cpp_runner + 5种 workload | 完成 |
| P3 自适应策略 | adaptive_phase_aware 连接到控制器 | 完成 |
| P4 Rate Table 验证 | 重建 + 评估验证 (0 anomaly) | 完成 |
| P5 真实模型实验 | 4x3x5x3 = 180 runs 多配置实验 | 完成 |
| P6 Real Rate Table | 用 real data 重建 rate table | 待运行 |
| P6 Real Rate Table | 用 real data 重建 rate table | 待运行 |

## 快速开始

### 前置要求

- **Jetson Orin 设备** (JetPack r36.x)
- **Python 3.8+**
- **llama.cpp** (已安装并可用)

### 安装依赖

```bash
pip3 install pyyaml pandas numpy pyarrow matplotlib seaborn scipy scikit-learn
```

### 运行 Phase-Aware 验证实验

```bash
# 使用合成基准测试 (无需真实模型)
python3 src/run_phase_aware_experiment.py

# 生成可视化图表
python3 src/visualize_phase_aware.py

# Phase 5 综合对比图表
python3 src/visualize_phase5_p0.py

# 查看结果
cat data/phase_aware_experiment/report_*.txt
```

### 使用配置选择器

```bash
# 使用示例查询
python3 src/select_config.py --example

# 使用自定义 workload 查询
python3 src/select_config.py --workload my_workload.json --output result.json
```

### 运行选择器评估

```bash
python3 src/evaluate_selector.py
# 结果输出到 data/selector_eval/
```

### 运行真实模型 Baseline 对比 (需 Jetson 设备 + llama.cpp)

```bash
# 多配置实验 (4 GPU × 3 CPU × 5 workloads)
python3 src/run_real_model_experiment.py
# 结果输出到 data/real_model_experiment/

# 分析结果
python3 src/analyze_real_experiment.py

# 生成可视化
python3 src/visualize_real_experiment.py

# Baseline 对比实验 (9 baselines)
python3 src/run_baseline_comparison.py \
  --baselines configs/baselines.yaml \
  --workloads configs/real_model_workloads.yaml \
  --output-dir data/real_baseline_comparison
```

## 项目阶段完成情况

### Phase 1: 项目基础建设 (100%)
- 项目结构、文档、配置文件

### Phase 2: 核心模块实现 (100%)
- 频率控制器、指标采集、基准测试、扫描编排、日志解析

### Phase 3: 前置实验验证 (100%)
- 7 个前置实验全部完成 (45 个数据点)
- 能耗汇率表构建完成 (23 个配置)
- Phase-Aware DVFS 策略得到验证

### Phase 4: Phase-Aware DVFS (100%)
- 在线控制器实现
- 6 种策略对比验证 (含 adaptive_phase_aware)
- 可视化分析完成

### Phase 5: 高级优化 (70%)
- P0-P4 代码完成: 评估修复 + Baseline 基础设施 + 真实模型 Runner + 自适应策略
- P5 真实模型多配置实验完成: 180 runs, GPU无影响, CPU是瓶颈
- P6 待运行: 用 real data 重建 rate table

**总体进度**: 约 90%

## 文档说明

### 核心文档
- **[CLAUDE.md](CLAUDE.md)** - AI 助手项目上下文 (英文)
- **[EnergyInfra_Phase5_Task_Plan.md](EnergyInfra_Phase5_Task_Plan.md)** - Phase 5 详细任务计划
- **[scheduling_method_analysis.md](scheduling_method_analysis.md)** - 调度方法分析
- **[docs/任务书.md](docs/任务书.md)** - 项目目标和技术路线
- **[docs/checklist.md](docs/checklist.md)** - 待办事项和开发指南

### 实验报告
- **[docs/project_progress_summary.md](docs/project_progress_summary.md)** - 项目进展总结 (更新至 Phase 5)
- **[docs/当前状态.md](docs/当前状态.md)** - 当前开发状态
- **[docs/PHASE3_COMPLETION_REPORT.md](docs/PHASE3_COMPLETION_REPORT.md)** - Phase 3 完成报告
- **[docs/frequency_experiment_analysis.md](docs/frequency_experiment_analysis.md)** - 频率实验分析

---

**最后更新**: 2026-05-13
**项目状态**: Phase 5 进行中 (真实模型多配置实验完成)
**总体进度**: 约 90%
