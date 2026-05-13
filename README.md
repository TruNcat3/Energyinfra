# Jetson LLM Energy Profiling - Project Overview

**项目状态**: Phase 5 (高级优化) 进行中 — 真实模型多配置实验已完成 (180 runs)
**总体进度**: 约 90%

## 项目概览

本项目旨在构建面向 Jetson 边端 GPU 的 LLM 推理能耗 profiling 与自适应配置选择系统。系统能够自动执行 workload × frequency 扫描，采集性能、功耗和温度数据，构建能耗汇率表，并在给定 SLO 约束下输出最优的 GPU、CPU 和 EMC 频率配置。

### 核心目标

1. **能耗汇率表构建**: 建立 workload-aware 的能耗汇率表 (完成)
2. **自适应配置选择**: 设计 SLO-aware 的配置选择器 (完成)
3. **在线控制系统**: 实现可离线建表、在线查表、自动选频的系统 (进行中)

---

## 文档导航

### 任务规格书
| 文档 | 说明 |
|------|------|
| [docs/任务书/任务书.md](docs/任务书/任务书.md) | 项目目标、技术路线、成功标准 |
| [docs/任务书/jetson_llm_energy_rate_table_task_doc.md](docs/任务书/jetson_llm_energy_rate_table_task_doc.md) | 技术任务详述 |

### 开发文档
| 文档 | 说明 |
|------|------|
| [docs/开发文档/当前状态.md](docs/开发文档/当前状态.md) | 当前开发状态、模块进度、技术风险 |
| [docs/开发文档/checklist.md](docs/开发文档/checklist.md) | 待办事项清单、开发规范、检查清单 |
| [docs/开发文档/project_progress_summary.md](docs/开发文档/project_progress_summary.md) | 项目进展总结 (Phase 1-5) |
| [docs/开发文档/EnergyInfra_Phase5_Task_Plan.md](docs/开发文档/EnergyInfra_Phase5_Task_Plan.md) | Phase 5 详细任务计划 |
| [docs/开发文档/scheduling_method_analysis.md](docs/开发文档/scheduling_method_analysis.md) | 调度方法分析与优化方向 |
| [CLAUDE.md](CLAUDE.md) | AI 助手项目上下文 (英文) |

### 说明文档
| 文档 | 说明 |
|------|------|
| [docs/说明文档/LLAMACPP_INTEGRATION.md](docs/说明文档/LLAMACPP_INTEGRATION.md) | llama.cpp 集成指南 |
| [docs/说明文档/quick_experiment_guide.md](docs/说明文档/quick_experiment_guide.md) | 快速实验指南 |
| [docs/说明文档/runtime_setup_guide.md](docs/说明文档/runtime_setup_guide.md) | 运行时环境配置 |
| [docs/说明文档/DOCKER_MODEL_SETUP.md](docs/说明文档/DOCKER_MODEL_SETUP.md) | Docker 模型部署 |

### 实验文档
| 文档 | 说明 |
|------|------|
| [docs/实验文档/README.md](docs/实验文档/README.md) | **实验文档首页 — 统一档位术语、图表索引** |
| [docs/实验文档/PHASE1_COMPLETION_REPORT.md](docs/实验文档/PHASE1_COMPLETION_REPORT.md) | Phase 1 基础建设完成报告 |
| [docs/实验文档/PHASE2_COMPLETION_REPORT.md](docs/实验文档/PHASE2_COMPLETION_REPORT.md) | Phase 2 核心模块完成报告 |
| [docs/实验文档/PHASE3_COMPLETION_REPORT.md](docs/实验文档/PHASE3_COMPLETION_REPORT.md) | Phase 3 前置实验报告 |
| [docs/实验文档/PHASE2_ANALYSIS_REPORT.md](docs/实验文档/PHASE2_ANALYSIS_REPORT.md) | Phase 2 分析报告 |

### 实验数据与图表
| 路径 | 说明 |
|------|------|
| `data/real_model_experiment/` | 真实模型多配置实验数据 (180 runs) |
| `data/rate_tables/` | 能耗汇率表 (Parquet, 23 configs) |
| `data/selector_eval/` | 选择器评估结果 |
| `figures/real_model_experiment/` | 真实模型实验图表 (5张) |
| `figures/phase5_comparison/` | Phase 5 综合对比图表 (6张) |
| `figures/power_mode_comparison/` | Jetson 档位对比图表 (6张) |

---

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
│   ├── experiments/           # 实验脚本 (9个)
│   └── _legacy/               # 归档废弃代码 (5个)
├── configs/                    # 配置文件
├── data/                       # 实验数据输出
├── figures/                    # 可视化图表 (27张)
├── scripts/
│   ├── active/                # 活跃脚本 (4个)
│   └── _legacy/               # 归档脚本 (9个)
├── docs/                       # 文档 (按类型分层)
│   ├── 任务书/                # 任务规格书
│   ├── 开发文档/              # 开发跟踪
│   ├── 说明文档/              # 使用指南
│   └── 实验文档/              # 实验报告
├── README.md
└── CLAUDE.md
```

## 核心实验结果

> **档位术语说明**: 见 [docs/实验文档/README.md](docs/实验文档/README.md) — MAXN / 50W / 30W (出厂默认) / 15W / GPU-Min/CPU-Max (我们的策略)

### Jetson 档位对比 (真实模型, Phi-3-mini Q4)

| 档位 | GPU 频率 | CPU 频率 | TTFT (ms) | TPOT (ms) | Throughput (tok/s) |
|------|:---:|:---:|:---:|:---:|:---:|
| **MAXN** | 1300 MHz | 2201 MHz | 498.7 | 107.0 | 9.2 |
| **30W** (出厂默认) | 612 MHz | 1497 MHz | 141.6 | 142.5 | 7.0 |
| **15W** | 306 MHz | 1036 MHz | 878.1 | 196.7 | 5.0 |
| **GPU-Min/CPU-Max** (我们) | 306 MHz | 2201 MHz | 773.8 | 107.6 | **9.0** |

**关键发现**:
- **GPU 频率无影响**: 306→1300 MHz 仅 1.01x 吞吐量 (memory-bound 确认)
- **CPU 频率是瓶颈**: 1036→2201 MHz 达 1.86x 吞吐量
- **GPU-Min/CPU-Max 以 GPU 最低频率达到接近 MAXN 的吞吐量** (9.0 vs 9.2 tok/s)
- 最优配置: GPU 918 MHz + CPU 2201 MHz = 9.4 tok/s

### Phase-Aware DVFS 验证 (合成基准, 2026-05-12)

> **注意**: 此实验使用合成基准，结果中的 GPU 频率效应被高估。真实模型实验已确认 GPU 频率对 LLM 推理无显著影响。

| 策略 | 能量/Token (J) | TTFT (ms) | TPOT (ms) | SLO 满足率 |
|------|---------------|-----------|-----------|-----------|
| Default (mid) | 0.164 | 121 | 7.6 | 82% |
| Max Performance | 0.110 | 124 | 4.5 | 42% |
| Energy Efficient | 0.326 | 125 | 16.9 | 92% |
| **Phase-Aware (Ours)** | **0.115** | **77** | **6.7** | **80%** |
| Oracle | 0.152 | 124 | 6.9 | 76% |

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

## 快速开始

### 前置要求

- **Jetson Orin 设备** (JetPack r36.x)
- **Python 3.8+**
- **llama.cpp** (已安装并可用)

### 安装依赖

```bash
pip3 install pyyaml pandas numpy pyarrow matplotlib seaborn scipy scikit-learn
```

### 运行合成基准实验 (无需真实模型)

```bash
cd /home/wt/work/Energyinfra

# Phase-Aware 验证实验
python3 src/experiments/run_phase_aware_experiment.py

# 生成可视化图表
python3 src/visualization/visualize_phase_aware.py
python3 src/visualization/visualize_phase5_p0.py
```

### 运行真实模型实验 (需 Jetson + llama.cpp)

```bash
source jetson_llm_env/bin/activate

# 多配置实验 (4 GPU x 3 CPU x 5 workloads x 3 repeats = 180 runs)
python3 src/experiments/run_real_model_experiment.py

# 分析结果
python3 src/visualization/analyze_real_experiment.py

# 生成可视化
python3 src/visualization/visualize_real_experiment.py

# Jetson 档位对比图表 (6张)
python3 src/visualization/visualize_power_mode_comparison.py
```

### 使用配置选择器

```bash
python3 src/controller/select_config.py --example
```

---

## 项目阶段

| Phase | 描述 | 进度 |
|-------|------|------|
| Phase 1 | 项目基础建设 | 100% |
| Phase 2 | 核心模块实现 | 100% |
| Phase 3 | 前置实验验证 (7个) | 100% |
| Phase 4 | Phase-Aware DVFS | 100% |
| Phase 5 | 高级优化 | 70% |

**最后更新**: 2026-05-13
