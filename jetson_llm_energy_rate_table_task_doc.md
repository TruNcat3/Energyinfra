# 面向 Jetson 边端 GPU 的 LLM 推理能耗汇率表与自适应配置选择机制

## 1. 研究任务概述

本任务面向 Jetson Orin 等边端 GPU 平台上的大模型推理部署，研究在功耗、温度和时延约束下，如何根据负载特征自动选择 GPU、CPU 和 EMC 频率配置。与数据中心 GPU 不同，边端 GPU 的功耗预算更紧，散热条件更受限，默认的固定频率策略或全拉满策略并不总是能效最优。因此，本任务希望建立一个 workload-aware 的能耗汇率表，将不同负载在不同频率配置下的性能、功耗和能耗关系显式记录下来，并进一步设计一个 SLO-aware 的配置选择器。

该任务受到 NanoFlow 的启发。NanoFlow 的核心价值不在于具体的 nano-batching 形式，而在于它从 LLM 推理负载内部的资源使用差异出发，将 prefill、decode、dense operation 和 attention operation 等阶段进行区分，并通过自动参数搜索获得更优的执行配置。本文拟借鉴这种“负载特征建模 + 自动配置搜索”的思路，将目标从吞吐优化扩展到边端 GPU 上的能耗优化。

本任务的最终目标是构建一个可以离线建表、在线查表、自动选频的系统。给定模型、batch size、prompt length、output length、并发度、prefill/decode 比例和 SLO 约束，系统能够输出对应的 GPU、CPU 和 EMC 频率配置，使推理过程在满足 TTFT、TPOT、P99 latency、功耗或温度约束的前提下尽可能降低 energy/token。

---

## 2. 核心研究问题

本任务需要回答以下几个问题。

第一，Jetson 边端 GPU 上的 LLM 推理是否存在稳定的频率—性能—能耗关系。如果同一负载和同一频率配置下测量结果波动很大，那么能耗汇率表无法可靠使用。因此，首先需要验证测量稳定性。

第二，不同频率旋钮的作用是否不同。已有初步热力图表明，GPU 频率对吞吐和 per-token latency 的影响较强，CPU 频率更可能影响 TTFT、调度路径和尾延迟，EMC 频率则可能更多影响 decode 阶段的 KV cache 访问和内存带宽压力。该观察需要通过更系统的单旋钮实验确认。

第三，不同负载是否具有不同的最优配置。若所有 workload 的最优点都是最高 GPU、最高 CPU 和最高 EMC，则自适应选择的意义有限。相反，如果 batch、prompt length、output length、prefill/decode 比例不同会导致最优配置变化，那么就可以证明 workload-aware selector 的必要性。

第四，prefill 和 decode 阶段是否应该采用不同配置。Prefill 通常更偏向大矩阵计算，decode 阶段更容易受到 KV cache 访问、低 batch 利用率和访存效率影响。因此，phase-aware DVFS 是本任务中最值得重点验证的方向。

第五，能否用简单的 workload feature 预测最优配置。理想情况下，系统不需要在线重复搜索，而是根据 batch size、prompt length、output length、phase、concurrency 等特征直接查表或调用轻量模型得到推荐配置。

---

## 3. 总体技术路线

整体技术路线分为四个阶段。

### 3.1 离线频率扫描

首先在 Jetson 平台上构建可重复的 profiling 流程。系统枚举不同 workload 和频率配置，并记录每次运行的 latency、throughput、power、temperature 和 energy 指标。频率配置包括 GPU frequency、CPU frequency 和 EMC frequency。workload 维度包括模型、batch size、prompt length、output length、并发度和推理阶段。

### 3.2 能耗汇率表构建

基于原始 profiling 数据，计算派生指标，包括 energy/request、energy/token、tokens/J、latency-energy tradeoff、throughput-power tradeoff 和 Pareto dominance。对于每个 workload bucket，系统保留非支配配置，形成 Pareto frontier。

### 3.3 负载感知配置选择

给定一个新的推理请求或 batch window，系统提取 workload feature，并在能耗汇率表中查找对应 bucket。若存在满足 SLO 的配置，则选择 energy/token 最小的配置；若不存在完全匹配的 bucket，则使用最近邻、插值或轻量预测模型给出配置建议。

### 3.4 在线控制与稳定机制

在线控制器负责在推理过程中应用选定配置。为了避免频繁切换导致抖动，需要引入 hysteresis 机制。例如，只有当预测收益超过阈值时才切换频率，或者要求同一配置至少保持若干个 batch window。对于 prefill/decode 分阶段策略，应尽量在 phase boundary 处切换，从而降低切换开销对 token latency 的影响。

---

## 4. 前置验证实验设计

### 4.1 测量稳定性实验

目的在于确认同一 workload 和同一频率配置下，测量结果是否稳定。

实验设置如下。

- 固定模型、batch size、prompt length、output length 和频率配置。
- 每组配置重复运行 5 到 10 次。
- 固定 power mode、风扇策略、环境温度和后台进程状态。
- 记录 TTFT、TPOT、ITL、throughput、average power、maximum power、energy/token、temperature。

预期结论如下。

- 如果 energy/token 和 latency 的标准差较小，则说明后续建表可行。
- 如果波动较大，需要先处理热降频、governor 干扰、后台进程或测量窗口不一致的问题。

### 4.2 单旋钮敏感性实验

目的在于区分 GPU、CPU 和 EMC 频率对不同指标的影响。

实验设计如下。

| Sweep 对象 | 固定项 | 主要观察指标 | 预期验证内容 |
|---|---|---|---|
| GPU frequency | CPU 和 EMC 固定 | throughput、TPOT、ITL、energy/token | GPU 是否是主要性能旋钮 |
| CPU frequency | GPU 和 EMC 固定 | TTFT、P99 latency、host overhead | CPU 是否主要影响控制路径 |
| EMC frequency | GPU 和 CPU 固定 | TPOT、ITL、memory-sensitive latency、energy/token | EMC 是否主要影响 decode/KV cache 路径 |

该实验是后续建模的基础。若 GPU、CPU 和 EMC 的影响模式不同，则说明多旋钮配置选择有必要。

### 4.3 频率组合交互实验

目的在于验证最优配置是否来自多个频率旋钮的组合，而不是单一频率拉满。

建议先选取低、中、高三档频率。

| 频率维度 | 档位 |
|---|---|
| GPU | low / mid / high |
| CPU | low / mid / high |
| EMC | low / mid / high |

初始组合为 27 个配置。先通过稀疏搜索找到 Pareto frontier，再围绕 Pareto 点进行局部细化。该实验需要回答一个关键问题：中高 GPU + 中高 EMC 是否可能比全拉满配置具有更高 tokens/J。

### 4.4 Prefill/Decode 分阶段实验

目的在于验证 phase-aware DVFS 的必要性。

实验分为四类。

| 场景 | 目的 |
|---|---|
| only-prefill | 观察 prompt 处理阶段对 GPU 计算频率的敏感性 |
| only-decode | 观察逐 token 生成阶段对 EMC 和低 batch 利用率的敏感性 |
| prefill-heavy | 长 prompt、短输出，模拟上下文处理主导场景 |
| decode-heavy | 短 prompt、长输出，模拟持续生成主导场景 |

预期结论是，prefill 和 decode 的能效最优配置不同。如果该结论成立，则可以将 phase-aware DVFS 作为本文方法的核心设计点。

### 4.5 负载特征可预测性实验

目的在于验证能否根据 workload feature 预测最优配置。

输入特征包括：

- batch size
- prompt length
- output length
- concurrency
- phase
- model size
- quantization type
- current KV cache size
- GPU/CPU/EMC frequency

预测目标包括：

- TTFT
- TPOT
- P99 latency
- energy/token
- tokens/J
- feasible or infeasible under SLO

建议先比较三类模型。

| 模型 | 作用 |
|---|---|
| Linear Regression | 验证负载特征是否能解释主要趋势 |
| Polynomial / Interaction Model | 捕捉 batch × frequency、decode length × EMC 等交互 |
| XGBoost / RandomForest | 捕捉非线性 sweet spot 和配置边界 |

如果简单模型已经具有较好精度，则优先采用简单模型，因为解释性更强。如果非线性明显，则采用树模型作为 selector 的预测后端。

### 4.6 频率切换开销实验

目的在于确认在线调频是否会引入不可忽略的开销。

需要测量以下切换路径。

| 切换类型 | 指标 |
|---|---|
| GPU low → high | 切换时间、下一轮 kernel latency 抖动 |
| GPU high → low | 切换时间、性能恢复时间 |
| EMC low → high | decode latency 是否出现突刺 |
| CPU low → high | TTFT 和 host-side latency 抖动 |
| prefill → decode boundary 切换 | 切换是否能隐藏在阶段边界 |

如果切换开销较大，则在线策略不应按 token 级切换，而应按 request、batch window 或 phase boundary 切换。

### 4.7 SLO 约束下的端到端实验

目的在于证明本任务不是离线调参，而是一个实际可用的服务策略。

设置不同 SLO 约束。

| SLO 类型 | 约束示例 |
|---|---|
| latency-oriented | TTFT < X ms，TPOT < Y ms |
| throughput-oriented | throughput 不低于 MaxN 的 95% |
| energy-oriented | P99 latency 不退化，energy/token 最小 |
| thermal-oriented | temperature 不超过阈值，避免热降频 |

对比 baseline 如下。

| Baseline | 含义 |
|---|---|
| Default governor | 系统默认策略 |
| MaxN / jetson_clocks | 全部频率拉满 |
| Fixed best-efficiency | 离线找到的固定最高能效配置 |
| Oracle best | 离线已知每个 workload 最优点 |
| Ours | 基于负载特征和 SLO 自动选择配置 |

预期结果是，在满足 SLO 的前提下，本方法相比 MaxN 降低 energy/token；相比默认 governor 降低尾延迟或提升 tokens/J；相比固定 best-efficiency 策略，在 workload 变化时具有更稳定的性能表现。

---

## 5. 能耗汇率表设计

能耗汇率表分为两层。

### 5.1 原始 Profiling 表

原始表记录每次实验的完整结果。

| 字段 | 含义 |
|---|---|
| platform | Jetson 型号、JetPack 版本、power mode |
| model | 模型名称、参数量、量化方式 |
| runtime | TensorRT-LLM、llama.cpp、vLLM 或其他 runtime |
| phase | prefill、decode 或 mixed |
| batch_size | batch size |
| prompt_len | 输入 token 数 |
| output_len | 输出 token 数 |
| concurrency | 并发请求数 |
| gpu_freq | GPU 频率 |
| cpu_freq | CPU 频率 |
| emc_freq | EMC 频率 |
| ttft_ms | time to first token |
| tpot_ms | time per output token |
| itl_ms | inter-token latency |
| throughput | tokens/s |
| avg_power | 平均功耗 |
| max_power | 峰值功耗 |
| energy_request | 单请求能耗 |
| energy_token | 单 token 能耗 |
| tokens_per_joule | 能效 |
| temperature | 温度 |

### 5.2 派生汇率表

派生表用于配置选择。

| 指标 | 含义 |
|---|---|
| J/token | 单 token 能耗 |
| tokens/J | 单位能量产出 |
| ΔJ/Δms | 降低一单位延迟需要额外消耗的能量 |
| ΔW/Δtok/s | 提升一单位吞吐需要额外增加的功耗 |
| SLA margin | 当前配置距离 SLO 约束的余量 |
| dominance | 是否被其他配置支配 |
| pareto_rank | Pareto frontier 层级 |

配置选择器主要使用派生汇率表，而不是直接扫描全部原始数据。

---

## 6. 配置选择策略

配置选择问题可以形式化为：

\[
\min_{c \in C} E(c, w)
\]

其中，\(c\) 表示频率配置，\(w\) 表示负载特征，\(E\) 表示 energy/token 或 energy/request。

约束包括：

\[
TTFT(c,w) \leq S_{ttft}
\]

\[
TPOT(c,w) \leq S_{tpot}
\]

\[
P99(c,w) \leq S_{p99}
\]

\[
Power(c,w) \leq S_{power}
\]

\[
Temp(c,w) \leq S_{temp}
\]

如果多个配置都满足约束，则选择 energy/token 最小的配置。如果没有配置完全满足约束，则返回 closest feasible 配置，或者报告当前 SLO 不可满足。

在线策略应包含稳定机制。

- 避免每个 token 都切换频率。
- 优先在 request boundary、batch window boundary 或 prefill/decode boundary 切换。
- 只有当预测收益超过阈值时才切换。
- 对频率配置加入最小保持时间，减少抖动。
- 监控温度和实际 latency，若预测偏差增大，需要回退到安全配置。

---

## 7. Codex 开发任务书

### 7.1 项目名称

Jetson-LLM Energy Rate Table and Adaptive DVFS Selector

### 7.2 项目目标

实现一个面向 Jetson 边端 GPU 的 LLM 推理能耗 profiling 与自适应配置选择工具。系统能够自动执行 workload × frequency sweep，采集性能、功耗和温度数据，构建能耗汇率表，并在给定 SLO 约束下输出最优 GPU、CPU 和 EMC 频率配置。

### 7.3 输入

| 输入 | 说明 |
|---|---|
| workload config | 模型、batch size、prompt length、output length、并发度、phase |
| frequency config | GPU、CPU、EMC 可选频率 |
| SLO config | TTFT、TPOT、P99、power、temperature 约束 |
| benchmark command | 实际执行 LLM 推理的命令 |
| platform config | Jetson 型号、power mode、JetPack 版本、风扇策略 |

### 7.4 输出

| 输出 | 说明 |
|---|---|
| raw logs | 每次实验的原始日志 |
| profile.csv | 结构化 profiling 数据 |
| rate_table.parquet | 能耗汇率表 |
| pareto_frontier.csv | 每个 workload bucket 的 Pareto 配置 |
| selected_config.json | 给定 workload 和 SLO 后的推荐配置 |
| figures | heatmap、Pareto curve、energy-latency tradeoff、ablation 图 |

### 7.5 模块划分

| 模块 | 文件建议 | 功能 |
|---|---|---|
| 频率控制 | freq_controller.py | 设置和恢复 GPU、CPU、EMC 频率 |
| 负载运行 | benchmark_runner.py | 执行 LLM 推理 benchmark |
| 指标采集 | metrics_collector.py | 采集 tegrastats、latency、throughput、temperature |
| sweep 编排 | sweep_runner.py | 遍历 workload × frequency 配置 |
| 日志解析 | parse_logs.py | 从原始日志生成 CSV/Parquet |
| 汇率表生成 | build_rate_table.py | 计算 J/token、tokens/J、ΔJ/Δms 等 |
| Pareto 分析 | pareto.py | 删除被支配配置并生成 frontier |
| 配置选择器 | select_config.py | 根据 workload 和 SLO 选择配置 |
| 可视化 | plot_results.py | 生成 heatmap 和 tradeoff 图 |
| 在线控制器 | online_controller.py | 在推理服务中周期性选择和应用配置 |

### 7.6 配置文件示例

```yaml
platform:
  name: jetson_orin
  jetpack: "r36.x"
  power_mode: 0
  fan_mode: fixed

runtime:
  backend: tensorrt_llm
  benchmark_cmd_template: "python run_benchmark.py --model {model} --batch {batch_size} --prompt-len {prompt_len} --output-len {output_len}"

frequencies:
  gpu: [low, mid, high]
  cpu: [low, mid, high]
  emc: [low, mid, high]

workloads:
  models: ["qwen-7b-int4"]
  batch_size: [1, 2, 4, 8, 16]
  prompt_len: [128, 512, 1024, 2048]
  output_len: [32, 128, 512]
  phase: ["prefill", "decode", "mixed"]

slo:
  ttft_ms: 1000
  tpot_ms: 80
  p99_ms: 150
  max_power_w: 40
  max_temp_c: 80

sweep:
  repeats: 5
  warmup_runs: 1
  cooldown_sec: 30
```

### 7.7 Codex 实现步骤

第一步，实现 `freq_controller.py`。该模块需要支持读取当前 GPU、CPU 和 EMC 频率，设置指定频率，恢复默认频率，并记录设置是否成功。

第二步，实现 `metrics_collector.py`。该模块负责启动 tegrastats 或其他监控工具，解析功耗、温度、CPU/GPU/EMC 频率和内存占用，并按照统一时间戳写入日志。

第三步，实现 `benchmark_runner.py`。该模块根据配置文件生成 benchmark 命令，执行推理任务，并解析 benchmark 输出中的 TTFT、TPOT、ITL、throughput 和 latency 分布。

第四步，实现 `sweep_runner.py`。该模块负责遍历 workload 和 frequency 配置，执行 warmup、正式运行、冷却和日志保存。每个配置需要重复多次。

第五步，实现 `parse_logs.py`。该模块将 benchmark 日志和 tegrastats 日志对齐，生成结构化 CSV。

第六步，实现 `build_rate_table.py`。该模块计算 energy/request、energy/token、tokens/J、P95、P99 和能耗汇率指标。

第七步，实现 `pareto.py`。该模块对每个 workload bucket 删除被支配配置，输出 Pareto frontier。

第八步，实现 `select_config.py`。该模块根据输入 workload 和 SLO 约束，从 rate table 或 Pareto frontier 中选择最优配置。

第九步，实现 `plot_results.py`。该模块生成 GPU/CPU/EMC sweep heatmap、energy-latency tradeoff、tokens/J 对比图和 SLO ablation 图。

第十步，实现 `online_controller.py`。该模块作为后续扩展，用于在真实服务中周期性读取 workload 状态，并调用 selector 进行配置更新。

---

## 8. 推荐的数据目录结构

```text
jetson_llm_energy/
  configs/
    platform.yaml
    workloads.yaml
    sweep.yaml
    slo.yaml
  src/
    freq_controller.py
    metrics_collector.py
    benchmark_runner.py
    sweep_runner.py
    parse_logs.py
    build_rate_table.py
    pareto.py
    select_config.py
    plot_results.py
    online_controller.py
  data/
    raw_logs/
    parsed/
    rate_tables/
    pareto/
  figures/
    heatmaps/
    pareto_curves/
    ablations/
  scripts/
    run_sweep.sh
    build_table.sh
    select_config.sh
  README.md
```

---

## 9. 验收标准

本任务的阶段性验收标准如下。

第一，系统能够在固定 workload 和固定频率配置下重复运行，并输出稳定的 latency、power 和 energy/token 统计结果。

第二，系统能够完成 GPU、CPU 和 EMC 单旋钮 sweep，并生成对应 heatmap。

第三，系统能够完成低、中、高三档频率组合实验，并自动提取 Pareto frontier。

第四，系统能够区分 prefill-heavy 和 decode-heavy 场景，并验证二者的能效最优配置是否不同。

第五，系统能够根据 workload feature 和 SLO 约束输出推荐配置。

第六，相比 MaxN 或 jetson_clocks 全拉满策略，系统应在满足 SLO 的前提下降低 energy/token。

第七，相比固定低功耗策略，系统应在重负载或长 decode 场景下避免明显 P99 latency 退化。

---

## 10. 可预期论文贡献

本任务可以凝练为以下三点贡献。

第一，系统性刻画 Jetson 边端 GPU 上 LLM 推理在 GPU、CPU 和 EMC 频率变化下的性能与能耗行为，揭示 batch、prompt length、output length 和 prefill/decode 阶段导致的瓶颈迁移现象。

第二，提出 workload-aware energy rate table，将不同频率配置的收益和代价表示为 energy-latency、energy-throughput 和 tokens-per-joule 汇率，使系统能够快速定位 Pareto-optimal 配置。

第三，设计 SLO-aware adaptive DVFS selector，在满足 TTFT、TPOT 和 P99 latency 约束的同时，根据负载特征自动选择边端 GPU 的能效最优运行点。

---

## 11. 当前阶段建议

当前阶段不建议直接实现完整在线系统，而应先完成以下最小闭环。

1. 固定一个模型和一个 runtime。
2. 选择 3 个 batch size、3 个 prompt length 和 3 个 output length。
3. 选择 GPU、CPU、EMC 的低、中、高三档频率。
4. 完成离线 sweep 和日志解析。
5. 生成 heatmap 和 Pareto frontier。
6. 验证 workload-aware selector 是否优于固定 MaxN 和固定低功耗策略。

只要这个最小闭环成立，后续再扩展到更多模型、更多频率档位和在线控制。