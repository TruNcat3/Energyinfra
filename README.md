# Jetson-LLM Energy Rate Table & Adaptive DVFS Selector

**多目标 Workload-Aware DVFS 优化系统，面向 Jetson 边端 GPU 上的 LLM 推理**

**Platform**: Jetson Orin | **Runtime**: llama.cpp (GGUF) | **Models**: 7B / 8B / 14B (Q4_K_M)

---

## 核心思路

**离线 profiling → 能耗 rate table → 在线 workload-aware 频率 cap 选择**

1. **离线 profiling**: 对每个模型扫描 11 GPU 频率 × 12 workload，精确测量 E/tok、TPOT、功率
2. **Rate Table 构建**: 聚合统计 + Pareto 排序，生成 lock-rate-table 和 cap-rate-table
3. **多目标 Pareto 选择**: 对每个 workload 计算 E/tok × TPOT × Power 三维权衡前沿
4. **在线部署**: 根据输入长度选择最优 GPU cap，通过 performance governor + max_freq 控制

### 三模式频率控制

| 模式 | 方法 | 用途 |
|:---:|------|------|
| **Lock** | performance governor + min=max=freq | 离线精确 profiling |
| **Cap** | performance governor + max_freq | 在线部署（governor 在 cap 内动态调节） |
| **Dynamic** | simple_ondemand 默认 | Baseline 对比 |

> **关键发现**: Jetson Orin 的 `simple_ondemand` GPU governor 在推理负载下忽略 `max_freq` 设置，
> 因此 Cap 模式必须使用 `performance` governor + `max_freq` 实现。

---

## E2E 实测结果（378 runs, 3 模型, 9 策略）

### Pareto (我们的方法) vs Baselines

| 模型 | vs Dynamic E/tok | vs Dynamic Power | vs MAXN E/tok | vs MAXN Power |
|:---:|:---:|:---:|:---:|:---:|
| **7B (Qwen2.5)** | +0.8% | **+18.6%** | +4.5% | **+22.8%** |
| **8B (Llama-3.1)** | +1.4% | +4.1% | +5.0% | +8.8% |
| **14B (Qwen2.5)** | +1.8% | +3.3% | +5.5% | +7.5% |

### 跨策略对比（vs MAXN 基线的 E/tok 节省）

| 策略 | 7B | 8B | 14B | 特点 |
|:---|:---:|:---:|:---:|:---|
| ⭐ **Pareto (ours)** | +4.5% | +5.0% | +5.4% | 自动折中，稳定中等偏上 |
| slo_50ms | **+7.7%** | +4.4% | +4.5% | 7B 上最省能，依赖 SLO 阈值 |
| alpha_07 | +7.1% | +4.3% | **+6.3%** | 14B 上最优 |
| pwr_45w | +7.1% | **+6.8%** | — | 8B 上最优 |
| dynamic (默认) | +3.8% | +3.6% | +3.7% | 默认基线 |
| maxn (全频) | 0% | 0% | 0% | 最差基线 |

### 核心发现

1. **Pareto 优势在功率节省**: 7B 上功率从 47W 降至 38W（-18.6%），E/tok 仅损失 0.8%
2. **Workload-aware 选择有实际意义**: 7B 上 Pareto 为 5 个 workload 选择 612~1300MHz 不同 cap
3. **不同模型 DVFS 行为完全不同**: 7B compute-bound（中频甜点），8B memory-bound（408MHz 以上 TPS 扁平），14B compute-bound（高频最优）
4. **单目标策略在特定场景更优**: 有明确 SLO 时 slo_constrained 更好；Pareto 的价值在于无明确目标时的自动折中

### Oracle Gap 分析 (P13-P0): 单请求 DVFS 天花板

从 lock-mode 11 频率数据计算 oracle 理论最优，量化 DVFS 选择器的剩余优化空间：

| 模型 | Dynamic → Oracle | MAXN → Oracle | Pareto → Oracle |
|:---:|:---:|:---:|:---:|
| 8B (Llama-3.1) | +8.6% | +10.7% | **+0.7%** |
| 7B (Qwen2.5) | +12.3% | +15.6% | **+4.4%** |
| 14B (Qwen2.5) | +23.4% | +26.1% | **+9.4%** |

**结论**: Pareto 选择器距 oracle 上界仅 +4.9%，单请求 E/tok 优化空间有限。
DVFS 的真正价值在于长期运行的**功率节省**和**热管理**。

### 多目标 Pareto 评估 (EMO 指标)

采用 EMO 标准指标（MDR, JIR, HV）量化多维权衡优势：

| 指标 | 定义 |
|:---:|------|
| **MDR** | Multi-Objective Dominance Rate — 所有维度同时改进的比例 |
| **JIR** | Joint Improvement Ratio — 全部维度改进时的几何平均改进率 |
| **HV** | Hypervolume (Zitzler 1999) — 支配的目标空间体积 |

| 对比 (3D 离线) | 7B Waste | 8B Waste | 14B Waste |
|:---|:---:|:---:|:---:|
| Pareto vs MAXN | 14.8% | 10.2% | 11.6% |
| Pareto vs Dynamic | 4.5% | 0.2% | 0.3% |

**Hypervolume (Zitzler 1999 金标准)** — Pareto 支配更多目标空间：

| 模型 | Pareto HV | 第二名 (策略/HV) | Pareto 领先 | vs 中位数 |
|:---:|:---:|---|:---:|:---:|
| **7B** | 483.74 | pwr_45w 368.80 | +31.2% | 7.4× |
| **8B** | 73.68 | alpha_03 2.37 | **+3014%** | **33.0×** |
| **14B** | 96.02 | alpha_07 4.31 | **+2130%** | **30.6×** |

> 8B/14B 上 Pareto 支配的目标空间是所有其他策略的 **22-37 倍**。
> Dynamic/MAXN 的 HV < 3.4（点集高度聚集），Pareto 均匀覆盖整个 Pareto 前沿。

| 对比 (4D Serving) | MDR | Waste |
|:---|:---:|:---:|
| Pareto vs MAXN | 4.5% | 13.0% |
| ThermalSLO vs Dynamic | 5.4% | 10.0% |

### Phase 13: Thermal-SLO 反馈控制器

```
Offline Phase-aware Characterization (lock rate table, 已有)
        ↓
Workload-aware Initial Cap Selection (WorkloadCapSelector, 已有)
        ↓
Runtime SLO/Thermal Feedback Controller (ThermalSLOCapController) ← 新增
```

6 级优先级控制规则（10s 窗口）：
1. SLO violation → 立即升频
2. 温度 > 85°C → 升至 max（快速完成→空闲冷却）
3. 温度 > 75°C + SLO 裕度 + 连续 K window → 降频一步
4. 功率超限 → 降频一步
5. 距上次切频 < 20s → 跳过
6. 默认 → 保持

### 跨模型 DVFS 特征对比

| 特征 | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) |
|:---|:---:|:---:|:---:|
| Pareto 点数 (lock 11 freq) | 9-11/11 | 2-3/11 | 6-8/11 |
| DVFS 优化空间 | 36% (最大) | 17% (最小) | 23% |
| 最优 E/tok 频率 | 714-918 MHz (中频) | 408+ MHz (全平) | 1122-1300 MHz (高频) |
| Knee 点 | 714-918 MHz | 408-612 MHz | 408-918 MHz |

---

## 项目结构

```
Energyinfra/
├── src/
│   ├── controller/
│   │   ├── pareto_selector.py          # 多目标 Pareto DVFS 选择器
│   │   ├── workload_cap_selector.py     # Workload-aware Cap 选择器 (含 Pareto 策略)
│   │   ├── thermal_slo_controller.py   # Thermal-SLO 反馈控制器 (P13)
│   │   ├── cap_controller.py           # Lock/Cap/Dynamic 三模式频率控制
│   │   └── freq_controller.py          # sysfs/jetson_clocks 频率控制
│   ├── benchmark/
│   │   └── llama_cpp_runner.py         # llama.cpp 推理 runner (含 benchmark_mode)
│   ├── metrics/
│   │   └── metrics_collector.py        # tegrastats 功耗采集
│   ├── ratetable/
│   │   ├── build_workload_rate_table.py # Rate Table 构建 (含 Pareto rank)
│   │   ├── oracle_gap_analysis.py     # Oracle Gap 分析 (P13)
│   │   └── pareto_multi_objective_evaluation.py # 多目标 Pareto 评估 (MDR/JIR/HV)
│   ├── experiments/
│   │   ├── run_finegrained_profiling.py # Lock-mode 11 GPU freq profiling
│   │   ├── run_cap_profiling.py         # Cap-mode profiling (4 caps + baselines)
│   │   ├── run_cap_selector_benchmark.py # E2E 验证 benchmark
│   │   ├── run_finegrained_cap_profiling.py # Fine-grained 10-cap profiling (P13)
│   │   └── run_serving_benchmark.py      # Long-running serving benchmark (P13)
│   └── visualization/
│       ├── visualize_pareto.py          # Pareto 前沿可视化 (6 张图)
│       ├── visualize_cross_model_comparison.py # 跨模型对比 (4 张图)
│       ├── analyze_e2e_benchmark.py     # E2E benchmark 分析 (5 张图 + 报告)
│       └── visualize_phase13.py         # Phase 13 可视化 (9 张图, 含多目标指标)
├── data/                               # 实验报告 (原始数据仅本地保留)
│   ├── energy_profiling/               # Lock-mode profiling 报告
│   ├── rate_tables/                    # 汇率表构建报告 (parquet/json 原始数据仅本地)
│   ├── oracle_gap_analysis/            # Oracle gap 分析报告 (P13)
│   └── multi_obj_eval/                 # 多目标 Pareto 评估报告 (P13)
├── figures/
│   ├── pareto_frontier/                # Pareto 前沿图 (6 张)
│   ├── cross_model_comparison/         # 跨模型对比图 (4 张)
│   ├── e2e_benchmark/                  # E2E benchmark 图表 (5 张 + 报告)
│   └── phase13_analysis/               # Phase 13 分析图表 (P13)
├── scripts/active/
│   ├── run_remaining_models.sh         # 8B + 14B profiling (断点续跑)
│   └── run_e2e_benchmark_all.sh        # 3 模型 E2E benchmark (断点续跑)
└── docs/
```

---

## 快速开始

### 环境初始化

```bash
./setup.sh
```

一行完成目录结构创建、`jetson_llm_env` 虚拟环境构建与依赖安装（可重复执行）。
要求：Jetson Orin、JetPack r36.x、Python 3.10；`llama-cpp-python` 需 CUDA 编译，见
[docs/说明文档/LLAMACPP_INTEGRATION.md](docs/说明文档/LLAMACPP_INTEGRATION.md)。

> **数据与模型说明**：仓库不包含 GGUF 模型文件与原始实验数据（体积较大，均可由脚本重新生成）。
> 模型获取见 [docs/说明文档/DOCKER_MODEL_SETUP.md](docs/说明文档/DOCKER_MODEL_SETUP.md)；
> profiling 原始数据由下方步骤 1 生成，各阶段实验报告见 [docs/实验文档](docs/实验文档/README.md)。

### 1. 运行 Profiling（离线建表）

```bash
# 单模型 Lock-mode profiling (~5.5h, 792 runs)
sudo python3 src/experiments/run_finegrained_profiling.py \
  --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# 单模型 Cap-mode profiling (~1.5h, 216 runs)
sudo python3 src/experiments/run_cap_profiling.py \
  --model models/gguf/Qwen2.5-7B-Instruct-Q4_K_M.gguf

# 构建 Rate Table（含 Pareto rank）
python3 src/ratetable/build_workload_rate_table.py \
  --finegrained data/energy_profiling/finegrained_profiling_*.csv \
  --cap data/cap_profiling/cap_profiling_*.csv \
  --model Qwen2.5-7B-Instruct-Q4_K_M
```

### 2. 使用 Pareto 选择器

```python
from src.controller.pareto_selector import ParetoSelector

ps = ParetoSelector(
    lock_rate_table_path='data/rate_tables/lock_rate_table_*.parquet',
    cap_rate_table_path='data/rate_tables/cap_rate_table_with_savings_*.parquet',
    model='Qwen2.5-7B-Instruct-Q4_K_M',
)

# 计算某 workload 的 Pareto 前沿
frontier = ps.compute_frontier(512, 128, phase='decode', source='lock')
print(f'Pareto: {frontier.n_frontier}/{frontier.n_total} configs')

# 选择 knee point（自动折中）
knee = ps.select_from_frontier(frontier, strategy='knee_point')
print(f'Knee: GPU={knee["gpu_freq_mhz"]}MHz, E/tok={knee["energy_per_token_j"]}J')

# SLO 约束选择
slo = ps.select_from_frontier(frontier, strategy='slo_constrained', tpot_slo_ms=50.0)

# Alpha 连续调节 (0=能效优先, 1=延迟优先)
alpha = ps.alpha_to_frontier_point(frontier, alpha=0.7)
```

### 3. 在线 Cap 选择（部署）

```python
from src.controller.workload_cap_selector import WorkloadCapSelector

sel = WorkloadCapSelector(
    cap_rate_table_path='data/rate_tables/cap_rate_table_with_savings_*.parquet',
    lock_rate_table_path='data/rate_tables/lock_rate_table_*.parquet',
    model='Qwen2.5-7B-Instruct-Q4_K_M',
)

# Pareto 策略自动选择
cfg = sel.select(prompt_length=512, output_length=128, strategy='pareto')
print(f'GPU cap: {cfg["gpu_cap_mhz"]}MHz, pred E/tok: {cfg["pred_energy_per_token_j"]}J')
```

### 4. 生成可视化

```bash
python3 src/visualization/visualize_pareto.py          # Pareto 前沿 (6 图)
python3 src/visualization/visualize_cross_model_comparison.py  # 跨模型对比 (4 图)
python3 src/visualization/analyze_e2e_benchmark.py     # E2E 分析 (5 图 + 报告)
```

---

## 研究阶段与实验记录

各阶段的完整实验报告入口见 [docs/实验文档](docs/实验文档/README.md)。

| 阶段 | 内容 | 实验记录 |
|:---:|------|------|
| 1-3 | 基础设施、核心模块、前置验证实验 | [Phase 1-3 报告](docs/实验文档/README.md) |
| 4-5 | Phase-Aware DVFS、真实模型实验 | [实验总览](docs/实验文档/实验总览.md) |
| 6-7 | 细粒度 GPU×EMC DVFS；phase 切换开销验证（负结果） | [Phase 6 报告](docs/实验文档/PHASE6_FINEGRAINED_DVFS_REPORT.md) |
| 8-10 | 频率控制重构 (lock/cap/dynamic)、benchmark 模式、大模型 profiling | — |
| 11 | Workload-aware Rate Table + 多目标 Pareto | [Pareto 图表](figures/pareto_frontier/) |
| 12 | E2E Cap Selector Benchmark (378 runs) | [E2E 报告](figures/e2e_benchmark/e2e_benchmark_report.md) |
| 13 | Oracle Gap + Thermal-SLO 控制器 + Serving Benchmark | [Phase 13 图表](figures/phase13_analysis/) |

### 方法演进中的关键转折

- **Phase 7（负结果）**: 实测 sysfs phase-boundary DVFS 切换开销 ~715ms（占 TTFT 37-65%），据此放弃在线 phase 切换，转向**离线 profiling + 在线 workload-aware cap**
- **Phase 11**: 从返回单一最优解改为输出完整**多目标 Pareto 前沿**（权衡面而非点）
- **Phase 13**: 从单请求 E/tok 优化转向**长期 serving 的 SLO-stable thermal-aware energy management**

---

## 关键技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| GPU Governor (cap mode) | `performance` + `max_freq` | `simple_ondemand` 忽略 max_freq |
| GPU Governor (lock mode) | `performance` + min=max=freq | 精确锁定频率 |
| Profiling scope | 11 GPU freq × 12 workload | 覆盖 64→2048 tokens 范围 |
| Pareto objectives | E/tok, TPOT, Power (all minimize) | 三维权衡：能效 × 延迟 × 功耗 |
| Knee detection | 最大角度变化法 | 自动找到前沿折中点 |
| EOS 抑制 | logit_bias={eos: -100} + repeat_penalty=1.0 | 确保完整输出 |

---

## 文档导航

| 文档 | 说明 |
|------|------|
| [docs/实验文档/README.md](docs/实验文档/README.md) | 实验文档索引（术语规范 + 全部报告入口） |
| [docs/实验文档/实验总览.md](docs/实验文档/实验总览.md) | 实验环境、频率配置空间、功耗采集方法 |
| [docs/实验文档/related_work_comparison.md](docs/实验文档/related_work_comparison.md) | **相关工作对比分析** (DVFS/Serving/边缘部署/多目标) |
| [figures/e2e_benchmark/e2e_benchmark_report.md](figures/e2e_benchmark/e2e_benchmark_report.md) | E2E benchmark 详细报告 |
| [docs/说明文档/runtime_setup_guide.md](docs/说明文档/runtime_setup_guide.md) | 运行时环境搭建（llama.cpp / Jetson） |
| [docs/说明文档/DOCKER_MODEL_SETUP.md](docs/说明文档/DOCKER_MODEL_SETUP.md) | 模型获取与 GGUF 部署 |
| [docs/说明文档/quick_experiment_guide.md](docs/说明文档/quick_experiment_guide.md) | 快速实验指引 |
