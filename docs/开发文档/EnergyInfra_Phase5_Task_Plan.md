# EnergyInfra 下一阶段任务书

## 1. 阶段定位

EnergyInfra 当前已经完成从前置实验数据到 **rate table、SLO-aware selector、phase-aware DVFS controller** 的工程闭环。SyntheticBenchmark 上的验证结果显示，Phase-Aware 策略相比 default baseline 具备明显能耗和 TTFT 改善，但这些结果还不能直接作为真实 Jetson + LLM 的最终结论。

下一阶段建议命名为：

```text
Phase 5: Real-Model Validation with Official Jetson Baselines
```

本阶段核心目标是：

> 修正现有评估可信度问题，补齐 Jetson 官方 baseline，并在真实 llama.cpp/Phi-3-mini 等模型上验证 SLO-aware selector 与 adaptive phase-aware DVFS 是否仍然有效。

---

## 2. 当前状态总结

### 2.1 已完成模块

当前工程已经实现以下模块：

```text
src/analyze_existing_experiments.py
src/build_rate_table.py
src/select_config.py
src/evaluate_selector.py
src/phase_aware_policy.py
src/phase_controller.py
src/run_phase_aware_experiment.py
src/visualize_phase_aware.py
```

已具备以下能力：

1. 读取 4.1 到 4.7 前置实验数据；
2. 统一数据字段并构建 bucket-level rate table；
3. 在 workload bucket 内计算 Pareto frontier；
4. 基于 SLO 过滤选择最低 energy/token 配置；
5. 对比 MaxN、all_mid、fixed best、oracle 和 ours；
6. 实现 phase-aware policy 与 phase-aware controller；
7. 运行 5 workload × 5 strategy × 10 repeats 的 synthetic 验证实验。

### 2.2 当前 synthetic 结果

最新 synthetic Phase-Aware 验证实验中：

| 策略 | Energy/token | TTFT | TPOT | SLO 满足率 |
|---|---:|---:|---:|---:|
| default | 0.1643 J | 121.4 ms | 7.55 ms | 82% |
| max_perf | 0.1095 J | 123.7 ms | 4.47 ms | 42% |
| energy_efficient | 0.3259 J | 125.2 ms | 16.88 ms | 92% |
| phase_aware | 0.1151 J | 77.2 ms | 6.68 ms | 80% |
| oracle | 0.1518 J | 123.6 ms | 6.91 ms | 76% |

当前可以支持的结论：

```text
1. rate table → selector → phase-aware controller 的工程闭环已经打通；
2. synthetic benchmark 上 phase-aware 策略有明显收益；
3. phase-aware 策略对中长 prompt 和长 output workload 更有效；
4. selector、Pareto、regret 分析流程已经可以复用到真实数据。
```

当前不能支持的结论：

```text
1. 不能证明真实 Jetson Orin 上也能获得 30% energy/token 降低；
2. 不能证明 decode 阶段真实瓶颈一定来自 EMC/KV cache；
3. 不能证明真实频率切换开销可被 phase boundary 完全隐藏；
4. 不能证明当前 SLO 满足率可以代表真实服务行为；
5. 不能作为最终论文级真实性能提升结论。
```

---

## 3. 当前需要优先修正的问题

### 3.1 fixed best 与 oracle 比较异常

当前 selector evaluation 中，`fixed_best_efficiency` 可能出现负 regret，并且平均 energy/token 远低于 oracle。该结果在概念上不合理。原因很可能是 prefill-only、decode-only 和 mixed 的 energy/token 归一化混用，导致 fixed best 从 prefill-only 中选到极低 energy/token，再用于其他 bucket 对比。

需要修正：

```text
fixed best 不能跨 phase 全局选择；
oracle 必须是每个 bucket 内的理论最优；
regret 不允许系统性小于 0；
不同 phase 的 energy/token 需要使用对应归一化定义。
```

### 3.2 Pareto dominance 判断需要 strict-better 条件

Pareto 支配关系应为：

```python
j_dominates_i = (
    all(obj_j <= obj_i for obj in objectives)
    and any(obj_j < obj_i for obj in objectives)
)
```

如果缺少 `any(obj_j < obj_i)`，等价点可能被错误地认为互相支配，影响 Pareto 点数量和 Pareto rank。

### 3.3 Phase-Aware 不应无条件启用

workload breakdown 显示，Phase-Aware 在长 prompt 或长 output 场景收益明显，但在 short-short 场景下能耗收益不稳定。因此后续策略应从：

```text
always phase-aware
```

调整为：

```text
adaptive phase-aware
```

即只有当预期节能大于切换开销、decode length 足够长、SLO margin 足够时才启用 phase-aware。

---

## 4. 下一阶段总体目标

下一阶段目标包括：

1. 修正 evaluation 可信度问题；
2. 增加 Jetson 官方 baseline；
3. 用真实 llama.cpp 模型进行 baseline 和 phase-aware 对比；
4. 将 phase-aware 从固定规则升级为 adaptive policy；
5. 在真实数据上重新构建 rate table、selector evaluation 和 phase-aware report。

本阶段不优先做强化学习、复杂预测模型或生产级 online service。

---

# 5. 任务优先级

## P0：修正评估逻辑与可信度问题

### P0.1 修正 fixed best / oracle / regret 计算

修改文件：

```text
src/evaluate_selector.py
```

修正要求：

1. `oracle_best_per_bucket` 必须定义为同一 workload bucket 内满足 SLO 的最低 energy/token 配置；
2. `fixed_best_efficiency` 不能跨 phase 直接选择全局最低 energy/token；
3. fixed best 至少拆成以下三类：

```text
fixed_best_prefill
fixed_best_decode
fixed_best_mixed
```

或者只在同类 phase 内比较。

4. regret 定义保持：

```text
energy_regret = (energy_selected - energy_oracle) / energy_oracle
```

如果出现负 regret，需要输出 warning，并检查是否存在跨 phase 比较或归一化错误。

验收标准：

```text
1. oracle_best_per_bucket 的 mean regret 必须为 0；
2. fixed_best 的 regret 不应系统性为负；
3. selector_eval_report.md 中不再出现 fixed best 明显优于 oracle 的异常结论。
```

---

### P0.2 修正 phase-aware energy/token 归一化

修改文件：

```text
src/build_rate_table.py
src/evaluate_selector.py
src/phase_aware_policy.py
```

统一能耗定义：

```text
prefill-only:
  energy_per_input_token_j = energy_j / prompt_length

decode-only:
  energy_per_output_token_j = energy_j / output_length

mixed:
  energy_per_output_token_j = total_energy_j / output_length
  energy_per_total_token_j = total_energy_j / (prompt_length + output_length)
```

selector 默认使用：

```text
prefill: energy_per_input_token_j
decode: energy_per_output_token_j
mixed: energy_per_output_token_j
```

报告中同时保留 `energy_per_total_token_j`，用于辅助分析。

验收标准：

```text
1. profile_raw.parquet 中包含三类 energy 字段；
2. selector_table.parquet 中明确记录 selector 使用的 energy objective；
3. evaluate_selector.py 按 phase 选择正确的 energy objective。
```

---

### P0.3 修正 Pareto dominance

修改文件：

```text
src/build_rate_table.py
```

修正 Pareto 判断逻辑：

```python
j_dominates_i = (
    all(obj_j <= obj_i for obj in objectives)
    and any(obj_j < obj_i for obj in objectives)
)
```

验收标准：

```text
1. Pareto frontier 仍然按 workload bucket 内部计算；
2. 等价点不会被错误地互相支配；
3. 重新生成 pareto_by_bucket.parquet 和 selector_table.parquet。
```

---

## P1：增加 Jetson 官方 baseline

### P1.1 增加 baseline 配置文件

新增配置文件：

```text
configs/baselines.yaml
```

建议包含以下 baseline：

```yaml
baselines:
  - name: jetson_default
    type: system_default
    description: "No manual frequency control, no jetson_clocks"

  - name: nvpmodel_low
    type: nvpmodel
    description: "Official Jetson low-power mode"

  - name: nvpmodel_mid
    type: nvpmodel
    description: "Official Jetson balanced mode"

  - name: nvpmodel_high
    type: nvpmodel
    description: "Official Jetson max-power mode"

  - name: jetson_clocks
    type: jetson_clocks
    description: "Lock clocks to max frequency"

  - name: manual_min
    type: manual_frequency
    gpu_freq_mhz: 378
    cpu_freq_mhz: 1020
    emc_freq_mhz: 133

  - name: manual_mid
    type: manual_frequency
    gpu_freq_mhz: 846
    cpu_freq_mhz: 1479
    emc_freq_mhz: 1600

  - name: manual_max
    type: manual_frequency
    gpu_freq_mhz: 1428
    cpu_freq_mhz: 2015
    emc_freq_mhz: 2133

  - name: fixed_best_static
    type: selector_static
    description: "Best fixed config from profiling data"

  - name: adaptive_phase_aware
    type: policy
    description: "Our adaptive phase-aware policy"
```

注意：

```text
nvpmodel 的真实 mode ID 不应硬编码。
需要通过 sudo nvpmodel -q --verbose 或用户配置自动识别。
```

---

### P1.2 实现 baseline runner

新增文件：

```text
src/run_baseline_comparison.py
```

功能：

1. 读取 `configs/baselines.yaml`；
2. 对每个 baseline 应用对应系统状态；
3. 运行同一组 workloads；
4. 采集 latency、tokens/s、power、temperature、frequency；
5. 输出统一 CSV 和报告。

输出目录：

```text
data/real_baseline_comparison/
```

输出文件：

```text
data/real_baseline_comparison/detailed_results_<timestamp>.csv
data/real_baseline_comparison/comparison_summary_<timestamp>.csv
data/real_baseline_comparison/workload_breakdown_<timestamp>.csv
data/real_baseline_comparison/baseline_report_<timestamp>.md
```

每条记录至少包括：

```text
timestamp
baseline_name
baseline_type
nvpmodel_mode
jetson_clocks_enabled
workload_name
model_name
runtime
prompt_len
output_len
batch_size
repeat
ttft_ms
tpot_ms
total_time_ms
tokens_per_second
total_energy_j
energy_per_output_token_j
energy_per_total_token_j
avg_power_w
p95_power_w
max_power_w
avg_temp_c
p95_temp_c
max_temp_c
gpu_freq_mhz_observed
cpu_freq_mhz_observed
emc_freq_mhz_observed
slo_met
slo_violations
```

---

### P1.3 baseline 环境恢复机制

baseline runner 必须支持实验前后恢复系统状态。

要求：

```text
1. 记录实验前 nvpmodel mode；
2. 记录实验前 jetson_clocks 状态；
3. 每次 baseline 运行结束后恢复或切换到下一个 baseline；
4. 实验结束后恢复初始状态；
5. 出现异常时也尽量恢复。
```

建议新增工具文件：

```text
src/jetson_power_modes.py
```

包含：

```python
get_current_nvpmodel()
list_nvpmodel_modes()
set_nvpmodel_mode(mode_id)
enable_jetson_clocks()
disable_or_restore_jetson_clocks()
record_current_power_state()
restore_power_state()
```

---

## P2：真实 llama.cpp 模型验证

### P2.1 真实模型 workload 配置

新增配置文件：

```text
configs/real_model_workloads.yaml
```

建议 workload 沿用 synthetic 验证中的五类：

```yaml
workloads:
  - name: short_short
    prompt_len: 128
    output_len: 64
    batch_size: 1

  - name: short_long
    prompt_len: 128
    output_len: 256
    batch_size: 1

  - name: medium_medium
    prompt_len: 512
    output_len: 128
    batch_size: 1

  - name: long_medium
    prompt_len: 1024
    output_len: 128
    batch_size: 1

  - name: long_long
    prompt_len: 1024
    output_len: 512
    batch_size: 1
```

初始实验矩阵：

```text
5 workloads × 8 baselines × 5 repeats = 200 runs
```

第一轮建议 baseline：

```text
jetson_default
nvpmodel_low
nvpmodel_mid
nvpmodel_high
jetson_clocks
manual_mid
fixed_best_static
adaptive_phase_aware
```

oracle 不需要真实执行，可在后处理中由 profiling 数据计算。

---

### P2.2 接入 llama.cpp benchmark

新增或扩展：

```text
src/llama_cpp_runner.py
```

功能：

1. 调用 llama.cpp 可执行文件；
2. 控制 prompt length 和 output length；
3. 解析 TTFT、TPOT、tokens/s、total time；
4. 与 metrics collector 对齐时间窗口；
5. 输出统一格式结果。

需要支持：

```text
model_path
prompt_len
output_len
batch_size
threads
gpu_layers
repeat
```

输出字段应与 synthetic benchmark 对齐，使后续 rate table 和 selector 不需要大改。

---

### P2.3 真实功耗与温度采集

真实实验必须使用 tegrastats 或已有 metrics collector 采集：

```text
avg_power_w
p95_power_w
max_power_w
avg_temp_c
p95_temp_c
max_temp_c
gpu_freq_mhz_observed
cpu_freq_mhz_observed
emc_freq_mhz_observed
```

注意事项：

```text
1. 不允许用 avg_power_w * constant 估计 max power；
2. 需要保存原始 tegrastats 日志；
3. 每次实验前应有 warmup；
4. 每个 baseline 之间应设置 cooldown；
5. 需要记录 fan mode 和环境温度。
```

---

## P3：Adaptive Phase-Aware 策略

### P3.1 从固定 phase-aware 改为条件启用

修改文件：

```text
src/phase_controller.py
src/phase_aware_policy.py
```

当前固定策略：

```text
prefill: high GPU, high CPU, mid EMC
decode: mid GPU, mid CPU, high EMC
```

应升级为：

```text
adaptive_phase_aware:
  if expected_energy_saving > switching_energy_overhead
  and switching_latency_overhead < slo_margin
  and output_len > break_even_tokens
  then enable phase switching
  else use single best config
```

建议增加策略输出字段：

```text
policy
switch_enabled
selection_reason
prefill_config
decode_config
single_config
switching_overhead_ms
switching_energy_j
expected_energy_saving_j
net_energy_saving_j
break_even_tokens
slo_margin_ms
```

---

### P3.2 分 workload 统计 phase-aware 收益

新增分析输出：

```text
data/real_baseline_comparison/phase_aware_break_even_analysis.csv
figures/real_baseline_comparison/phase_aware_by_workload.png
```

按 workload 输出：

```text
workload_name
prompt_len
output_len
default_energy_j
adaptive_phase_energy_j
energy_saving_pct
ttft_reduction_pct
tpot_reduction_pct
switch_enabled_rate
slo_met_rate
```

目标是回答：

```text
1. 哪些 workload 适合 phase-aware？
2. short output 是否不适合 phase switching？
3. break-even output length 大约是多少？
4. 长 prompt 和长 output 的收益是否更明显？
```

---

## P4：真实数据上的 rate table 和 selector evaluation

### P4.1 构建 real rate table

扩展命令：

```bash
python src/build_rate_table.py \
  --input data/real_baseline_comparison/detailed_results_*.csv \
  --output data/real_rate_tables/
```

输出：

```text
data/real_rate_tables/profile_raw.parquet
data/real_rate_tables/profile_agg_by_config.parquet
data/real_rate_tables/pareto_by_bucket.parquet
data/real_rate_tables/selector_table.parquet
data/real_rate_tables/rate_table_quality_report.json
data/real_rate_tables/rate_table_summary.md
```

要求：

```text
1. 和 synthetic rate table 使用相同 schema；
2. 增加 runtime=llama.cpp；
3. 增加 model_name=Phi-3-mini-Q4；
4. 标记 data_source=real_jetson；
5. 记录 baseline_name 和 baseline_type。
```

---

### P4.2 真实数据上的 selector evaluation

扩展命令：

```bash
python src/evaluate_selector.py \
  --selector-table data/real_rate_tables/selector_table.parquet \
  --output-dir data/real_selector_eval/
```

对比策略：

```text
jetson_default
nvpmodel_low
nvpmodel_mid
nvpmodel_high
jetson_clocks
manual_mid
fixed_best_static
slo_aware_selector
adaptive_phase_aware
oracle_best_per_bucket
```

输出：

```text
data/real_selector_eval/selector_comparison.csv
data/real_selector_eval/selector_regret.csv
data/real_selector_eval/selector_eval_report.md
figures/real_selector_eval/selector_energy_comparison.png
figures/real_selector_eval/selector_slo_violation_rate.png
figures/real_selector_eval/selector_regret_vs_oracle.png
```

---

## 6. 推荐实验矩阵

### 6.1 第一轮真实实验矩阵

| 维度 | 配置 |
|---|---|
| Runtime | llama.cpp |
| Model | Phi-3-mini Q4 |
| Workload | short_short, short_long, medium_medium, long_medium, long_long |
| Baseline | jetson_default, nvpmodel_low, nvpmodel_mid, nvpmodel_high, jetson_clocks, manual_mid, fixed_best_static, adaptive_phase_aware |
| Repeats | 5 |
| Warmup | 1–2 runs |
| Metrics | TTFT, TPOT, tokens/s, energy/token, avg/p95/max power, avg/p95/max temp, observed freq |

实验规模：

```text
5 workloads × 8 baselines × 5 repeats = 200 runs
```

### 6.2 第二轮扩展实验矩阵

如果第一轮趋势稳定，再增加：

```text
manual_min
manual_max
slo_aware_selector
more prompt/output combinations
batch_size = 2/4
different quantization or model sizes
```

不要一开始全量扩展，避免数据量过大但结论不清晰。

---

## 7. 核心评价指标

### 7.1 性能指标

```text
TTFT mean / P95
TPOT mean / P95
total latency
tokens/s
```

### 7.2 能耗指标

```text
total_energy_j
energy_per_output_token_j
energy_per_total_token_j
tokens_per_joule
avg_power_w
p95_power_w
max_power_w
```

### 7.3 稳定性指标

```text
SLO satisfaction rate
SLO violation breakdown
temperature P95 / max
frequency switching overhead
measurement CV
```

### 7.4 策略指标

```text
energy reduction vs jetson_default
energy reduction vs nvpmodel_mid
energy reduction vs jetson_clocks
regret vs oracle
SLO violation rate
break-even output length
```

---

## 8. 需要回答的核心问题

下一阶段实验需要回答：

1. 相比 Jetson default，adaptive phase-aware 是否仍有能耗收益？
2. 相比 nvpmodel 官方挡位，adaptive phase-aware 是否仍有额外收益？
3. 相比 jetson_clocks/MaxN，adaptive phase-aware 是否能在较低能耗下保持接近性能？
4. 短输出场景下 phase switching 是否不值得启用？
5. 长 prompt 或长 output 场景是否更适合 phase-aware？
6. 真实 llama.cpp 下 GPU、CPU、EMC 的敏感度是否与 synthetic 一致？
7. 真实频率切换开销是否会抵消收益？
8. SLO-aware selector 相比 fixed static config 是否更稳定？
9. 官方功耗模式是否过于粗粒度，无法适配 prefill/decode 差异？

---

## 9. 报告与图表要求

### 9.1 真实 baseline 总表

```text
figures/real_baseline_comparison/baseline_summary_table.png
```

包含：

```text
baseline
energy/token
TTFT
TPOT
tokens/s
SLO rate
avg power
p95 temp
```

### 9.2 能耗-延迟 Pareto 图

```text
figures/real_baseline_comparison/energy_latency_pareto.png
```

横轴：

```text
TPOT or total latency
```

纵轴：

```text
energy_per_output_token_j
```

标注：

```text
jetson_default
nvpmodel modes
jetson_clocks
manual_mid
adaptive_phase_aware
oracle
```

### 9.3 分 workload 收益图

```text
figures/real_baseline_comparison/phase_aware_by_workload.png
```

展示不同 workload 下 adaptive phase-aware 相比 default/nvpmodel_mid 的能耗和 TTFT 变化。

### 9.4 SLO 满足率图

```text
figures/real_baseline_comparison/slo_satisfaction_rate.png
```

展示每个 baseline 的 SLO 满足率。

### 9.5 频率切换开销图

```text
figures/real_baseline_comparison/switching_overhead.png
```

展示不同频率切换的实际耗时和切换后 latency spike。

---

## 10. 验收标准

### 10.1 P0 验收

```text
1. fixed best 不再出现系统性负 regret；
2. oracle regret 为 0；
3. Pareto dominance 修正为 non-worse + strict-better；
4. 重新生成 selector_eval_report.md；
5. 报告中说明不同 phase 的 energy/token 归一化规则。
```

### 10.2 P1 验收

```text
1. configs/baselines.yaml 已实现；
2. 能自动识别或记录 nvpmodel mode；
3. 能运行 jetson_default、nvpmodel、jetson_clocks、manual frequency baseline；
4. 实验结束后能恢复系统功耗/频率状态；
5. baseline 结果统一输出为 CSV。
```

### 10.3 P2 验收

```text
1. llama.cpp benchmark 能被脚本调用；
2. 能生成真实模型 detailed_results 和 comparison_summary；
3. 每条结果包含 latency、throughput、power、temperature、frequency；
4. 至少完成 5 workloads × 8 baselines × 5 repeats；
5. 生成 real_baseline_report.md。
```

### 10.4 P3 验收

```text
1. adaptive phase-aware 能根据收益和切换开销决定是否启用；
2. short output 场景可以自动回退 single config；
3. 输出 switch_enabled、selection_reason、net_energy_saving；
4. 生成 phase_aware_break_even_analysis.csv。
```

### 10.5 P4 验收

```text
1. 真实数据生成 real_rate_tables；
2. 真实数据生成 real_selector_eval；
3. 能比较官方 baseline、manual baseline、selector 和 adaptive phase-aware；
4. 报告中明确 synthetic 与 real 的趋势差异。
```

---

## 11. 给 Codex 的简化任务说明

```text
请在 EnergyInfra 工程中推进 Phase 5: Real-Model Validation with Official Jetson Baselines。

第一步，修正当前 evaluate_selector.py 和 build_rate_table.py 中的可信度问题：
1. fixed_best_efficiency 不允许跨 prefill/decode/mixed 直接选择全局最低 energy/token；
2. oracle 必须是每个 workload bucket 内满足 SLO 的最低 energy/token 配置；
3. 如果 regret 出现系统性负值，需要检查并报告；
4. Pareto dominance 必须使用 non-worse + strict-better 条件；
5. 重新生成 selector_eval_report.md。

第二步，新增 configs/baselines.yaml 和 src/run_baseline_comparison.py，支持 Jetson 官方 baseline：
jetson_default、nvpmodel_low、nvpmodel_mid、nvpmodel_high、jetson_clocks、manual_min、manual_mid、manual_max、fixed_best_static、adaptive_phase_aware。
nvpmodel 的 mode ID 不要硬编码，需要自动读取或由配置指定。实验前后必须记录并恢复系统状态。

第三步，新增或扩展 llama.cpp runner，使其能够运行 Phi-3-mini Q4 模型，并输出与 synthetic benchmark 对齐的指标：
TTFT、TPOT、total_time、tokens/s、energy/token、avg/p95/max power、avg/p95/max temperature、observed GPU/CPU/EMC frequency。

第四步，运行第一轮真实实验：
5 workloads × 8 baselines × 5 repeats。
workloads 使用 short_short、short_long、medium_medium、long_medium、long_long。
baselines 使用 jetson_default、nvpmodel_low、nvpmodel_mid、nvpmodel_high、jetson_clocks、manual_mid、fixed_best_static、adaptive_phase_aware。

第五步，将 phase-aware 从 always enable 改为 adaptive enable：
只有当 expected_energy_saving > switching_energy_overhead、switching_latency_overhead < SLO margin、output_len > break_even_tokens 时才启用 phase switching，否则回退 single config。

第六步，基于真实实验结果生成：
data/real_baseline_comparison/detailed_results_<timestamp>.csv
data/real_baseline_comparison/comparison_summary_<timestamp>.csv
data/real_baseline_comparison/workload_breakdown_<timestamp>.csv
data/real_baseline_comparison/baseline_report_<timestamp>.md
data/real_rate_tables/selector_table.parquet
data/real_selector_eval/selector_eval_report.md

最后，在报告中明确比较：
adaptive_phase_aware vs jetson_default
adaptive_phase_aware vs nvpmodel_mid
adaptive_phase_aware vs jetson_clocks
adaptive_phase_aware vs fixed_best_static
adaptive_phase_aware vs oracle

请不要优先实现强化学习或复杂 ML 预测模型。当前阶段重点是修正评估可信度、补齐官方 baseline、完成真实 llama.cpp 验证。
```

---

## 12. 当前不优先做的事情

本阶段不优先做：

```text
1. 强化学习控制器；
2. 复杂 ML 预测模型；
3. 生产级在线服务接口；
4. TensorRT-LLM 深度集成；
5. 大规模全组合频率扫描；
6. 多模型全量扩展。
```

这些内容应在真实 llama.cpp 实验确认策略有效后再推进。

---

## 13. 最终目标

本阶段完成后，EnergyInfra 应能形成如下结论链条：

```text
1. Jetson 官方功耗模式是粗粒度 baseline；
2. 固定频率策略无法同时适配 prefill/decode 和不同 workload；
3. rate table 可以描述 workload × frequency 的能耗汇率；
4. SLO-aware selector 可以在可行域内选择低能耗配置；
5. adaptive phase-aware 可以在长 prompt/长 output 场景获得额外收益；
6. 真实 llama.cpp 数据验证 synthetic 结论中哪些成立、哪些需要修正。
```

如果真实实验仍能显示 adaptive phase-aware 相比 Jetson default / nvpmodel_mid / jetson_clocks 有稳定收益，那么下一阶段再进入 Workload-Aware + Pareto 多目标优化和更复杂预测模型。
