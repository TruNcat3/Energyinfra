请在 EnergyInfra 工程中实现 Phase 3 的数据分析、能耗汇率表构建与配置选择闭环。当前已有实验数据位于 data/experiments_4_1_to_4_7，其中包含 experiment_4_1 到 experiment_4_7 的 CSV 文件。请不要修改已有实验数据，新增分析脚本、rate table 构建模块、selector 模块和评估模块即可。

当前数据主要来自 SyntheticBenchmark，适合验证分析流程、可视化流程、rate table 构建逻辑和 selector 逻辑，但不能直接作为真实 Jetson Orin + TensorRT-LLM 的最终策略结论。所有报告中需要明确标注 synthetic 数据的限制。后续真实实验数据接入后，应能复用本阶段实现的 rate table、selector 和 evaluation pipeline。

本阶段暂不实现完整 online_controller，也不优先实现强化学习或复杂预测模型。当前目标是先完成离线数据到配置决策的最小闭环。

============================================================
一、总体目标
============================================================

实现以下核心能力：

1. 读取 data/experiments_4_1_to_4_7 下的已有实验数据；
2. 修正并统一数据字段，包括阶段化 energy/token 归一化、max power 统计、频率配置计数等；
3. 构建 bucket-level energy rate table；
4. 在每个 workload bucket 内计算 Pareto frontier；
5. 实现 SLO-aware select_config.py，在满足 SLO 的配置中选择 energy/token 最小的频率组合；
6. 实现 evaluate_selector.py，对比 MaxN、fixed best、oracle 和 ours；
7. 在数据上进一步挖掘 SLO 可行域、边际收益递减、prefill/decode 瓶颈迁移和 selector regret；
8. 生成 Phase 3 preliminary findings 报告，为后续真实 Jetson + TensorRT-LLM 实验提供依据。

============================================================
二、新增目录
============================================================

请新增以下目录，如果已存在则复用。

data/analysis/
data/rate_tables/
data/selector_eval/
figures/phase3_analysis/

============================================================
三、新增文件
============================================================

请新增以下代码文件：

src/analyze_existing_experiments.py
src/build_rate_table.py
src/select_config.py
src/evaluate_selector.py
src/phase_aware_policy.py

请新增以下配置文件：

configs/selector.yaml
configs/rate_table.yaml
configs/phase3_experiments.yaml

============================================================
四、任务 1：实现 analyze_existing_experiments.py
============================================================

文件路径：

src/analyze_existing_experiments.py

功能：

读取 data/experiments_4_1_to_4_7 下的 7 个 CSV 文件，生成每个实验的统计摘要和初步发现报告。

输入文件：

data/experiments_4_1_to_4_7/experiment_4_1_*.csv
data/experiments_4_1_to_4_7/experiment_4_2_*.csv
data/experiments_4_1_to_4_7/experiment_4_3_*.csv
data/experiments_4_1_to_4_7/experiment_4_4_*.csv
data/experiments_4_1_to_4_7/experiment_4_5_*.csv
data/experiments_4_1_to_4_7/experiment_4_6_*.csv
data/experiments_4_1_to_4_7/experiment_4_7_*.csv

输出文件：

data/analysis/experiment_4_1_stability_summary.csv
data/analysis/experiment_4_2_knob_sensitivity.csv
data/analysis/experiment_4_3_config_comparison.csv
data/analysis/experiment_4_4_phase_summary.csv
data/analysis/experiment_4_5_workload_summary.csv
data/analysis/experiment_4_6_switching_summary.csv
data/analysis/experiment_4_7_slo_summary.csv
data/analysis/phase3_preliminary_findings.md

具体分析要求：

1. 实验 4.1 测量稳定性

计算以下指标的 mean、std、cv、median、min、max：

ttft_ms
tpot_ms
energy_per_token_j
tokens_per_second
avg_power_w
temperature_c

其中：

cv = std / mean * 100

报告中需要判断：

- TPOT 是否稳定；
- throughput 是否稳定；
- energy/token 是否需要多次重复；
- 是否可以用单次测量建表。

如果 energy/token CV 大于 5%，报告中应明确建议使用 median 或 trimmed mean。

2. 实验 4.2 单旋钮敏感性

按 sweep 字段分组，分别分析 GPU、CPU、EMC 频率变化对以下指标的影响：

ttft_ms
tpot_ms
energy_per_token_j
tokens_per_second
avg_power_w

计算：

delta_metric = max(metric) - min(metric)
relative_delta = delta_metric / mean(metric)
sensitivity = delta_metric / (max(freq) - min(freq))

输出每个旋钮的敏感度。

需要注意，当前 synthetic 数据可能主要体现 GPU frequency 的影响。如果 CPU 或 EMC 敏感度较低，请在报告中标记为 synthetic benchmark limitation，而不是直接得出 CPU/EMC 不重要的结论。

3. 实验 4.3 频率组合交互

比较以下配置：

all_low
all_mid
all_high
mid_gpu_mid_cpu_high_emc

输出：

best_energy_config
best_latency_config
best_throughput_config
best_tokens_per_joule_config

并判断 MaxN/all_high 是否是 energy/token 最优点。

4. 实验 4.4 Prefill/Decode 阶段差异

按 scenario 分组：

prefill
decode
prefill_heavy
decode_heavy

分别找出：

lowest_energy_config
lowest_tpot_config
lowest_ttft_config
highest_throughput_config

并比较不同 scenario 的最优 GPU/CPU/EMC 配置是否一致。

如果 prefill 与 decode 的最优配置不同，报告中标记 phase-aware DVFS 具备进一步验证价值。

如果 synthetic 数据没有体现 decode 对 EMC 更敏感，也要明确说明这是 synthetic benchmark 建模不足，真实 Jetson 实验需要重点验证。

5. 实验 4.5 workload feature 可预测性

当前数据量较小，不要训练模型，只做 descriptive summary。

输出不同 batch、prompt、output 配置下的：

ttft_ms
tpot_ms
energy_per_token_j
tokens_per_second

并在报告中指出当前 4.5 数据不足以支撑预测模型，需要后续扩展为正交实验。

6. 实验 4.6 switching overhead

按 switch_name 分组，比较 before 和 after。

输出：

switch_name
before_gpu_freq
after_gpu_freq
switching_overhead_ms
delta_ttft_ms
delta_tpot_ms
delta_energy_per_token_j
delta_avg_power_w

如果只有 GPU 切换数据，则报告中说明当前缺少 CPU、EMC 和 phase-boundary 切换数据。

7. 实验 4.7 SLO 验证

重新解释 SLO。

硬约束只包括：

TTFT
TPOT
Power
Temperature

energy/token 不是 SLO 硬约束，而是优化目标。

请重新生成 SLO summary，不要因为 energy/token 超过阈值就判定 SLO violation。energy/token 只能用于在满足 SLO 的配置中选择最优点。

============================================================
五、任务 2：实现基础数据修正与字段统一
============================================================

这部分可以在 build_rate_table.py 中实现，也可以抽取为工具函数。

需要统一处理以下问题。

1. 阶段化 energy/token 归一化

新增字段：

energy_per_input_token_j
energy_per_output_token_j
energy_per_total_token_j

归一化规则：

prefill-only:
  energy_per_input_token_j = energy_j / prompt_length

decode-only:
  energy_per_output_token_j = energy_j / output_length

mixed:
  energy_per_output_token_j = total_energy_j / output_length
  energy_per_total_token_j = total_energy_j / (prompt_length + output_length)

如果某些字段缺失，需要尽量从已有字段推导。如果无法推导，则填 NaN，并在 quality report 中记录。

selector 默认使用：

- prefill-only 使用 energy_per_input_token_j；
- decode-only 使用 energy_per_output_token_j；
- mixed 使用 energy_per_output_token_j，同时保留 energy_per_total_token_j 作为辅助分析指标。

2. max power 与温度统计

真实 tegrastats 数据应直接统计：

avg_power_w
p95_power_w
max_power_w
avg_temp_c
p95_temp_c
max_temp_c

当前 synthetic 数据如果只有 avg_power_w、max_power_w、temperature_c，则直接保留，并新增标记字段：

power_source = synthetic
power_statistics_limited = true

不能使用 avg_power_w * constant 来伪造 max_power_w。

3. bucket-level Pareto

Pareto frontier 必须在同一个 workload bucket 内计算，不能跨 workload 比较。

bucket key 至少包括：

model
runtime
batch_size
prompt_length
output_length
phase or scenario
concurrency

当前 synthetic 数据如果缺少 model、runtime、concurrency，请使用默认值：

model = synthetic_qwen_7b_int4
runtime = synthetic
concurrency = 1

4. 频率档位计数

不要依赖 YAML 注释中的 total_configs。

实现一个配置统计函数，自动输出：

num_gpu_freqs
num_cpu_freqs
num_emc_freqs
num_frequency_configs
num_workload_configs
num_total_runs
estimated_runtime_min

输出到：

data/analysis/experiment_plan_summary.json
data/analysis/experiment_plan_summary.md

============================================================
六、任务 3：实现 build_rate_table.py
============================================================

文件路径：

src/build_rate_table.py

功能：

将已有实验数据转换为 bucket-level energy rate table，并生成 Pareto table 和 selector table。

输入：

data/experiments_4_1_to_4_7/*.csv
data/parsed/*.csv
data/parsed/*.parquet

如果多个来源同时存在，优先使用 data/experiments_4_1_to_4_7 作为当前 Phase 3 测试输入。

输出：

data/rate_tables/profile_raw.parquet
data/rate_tables/profile_agg_by_config.parquet
data/rate_tables/pareto_by_bucket.parquet
data/rate_tables/selector_table.parquet
data/rate_tables/rate_table_quality_report.json
data/rate_tables/rate_table_summary.md

profile_raw.parquet：

保存统一字段后的原始记录。

profile_agg_by_config.parquet：

按 workload bucket + frequency config 聚合，重复实验取 median，同时保留 mean、std、cv。

聚合字段包括：

ttft_ms
tpot_ms
total_time_ms
tokens_per_second
avg_power_w
max_power_w
temperature_c
energy_per_input_token_j
energy_per_output_token_j
energy_per_total_token_j

pareto_by_bucket.parquet：

在每个 bucket 内计算 Pareto frontier。

默认 Pareto 目标：

minimize energy_per_token_j
minimize ttft_ms
minimize tpot_ms
minimize avg_power_w

注意不同场景 energy_per_token_j 的来源：

prefill-only 使用 energy_per_input_token_j
decode-only 使用 energy_per_output_token_j
mixed 使用 energy_per_output_token_j

selector_table.parquet：

每个 bucket 下保留所有 SLO-feasible 的 Pareto 配置，以及至少一个 fallback safe config。

字段至少包括：

model
runtime
batch_size
prompt_length
output_length
phase
scenario
concurrency
gpu_freq_mhz
cpu_freq_mhz
emc_freq_mhz
ttft_ms_median
tpot_ms_median
energy_per_token_j_median
tokens_per_joule_median
avg_power_w_median
max_power_w
temperature_c
slo_ttft_met
slo_tpot_met
slo_power_met
slo_temp_met
slo_all_met
ttft_margin
tpot_margin
power_margin
temp_margin
is_pareto
pareto_rank
measurement_cv
pareto_confidence

如果某个 bucket 只有一个配置，则该配置 is_pareto=true，但 pareto_confidence=low。

============================================================
七、任务 4：实现 select_config.py
============================================================

文件路径：

src/select_config.py

功能：

给定 workload + SLO，从 selector_table.parquet 中选择满足 SLO 的最低能耗配置。

输入示例：

{
  "model": "synthetic_qwen_7b_int4",
  "runtime": "synthetic",
  "batch_size": 1,
  "prompt_length": 512,
  "output_length": 128,
  "phase": "mixed",
  "concurrency": 1,
  "slo": {
    "ttft_ms": 1000,
    "tpot_ms": 80,
    "max_power_w": 40,
    "max_temp_c": 80
  }
}

输出示例：

{
  "selected_config": {
    "gpu_freq_mhz": 846,
    "cpu_freq_mhz": 1479,
    "emc_freq_mhz": 1600
  },
  "predicted_metrics": {
    "ttft_ms": 10.7,
    "tpot_ms": 7.55,
    "energy_per_token_j": 0.109,
    "avg_power_w": 20.8,
    "temperature_c": 51.0
  },
  "selection_reason": "min_energy_under_slo",
  "fallback_used": false,
  "matched_bucket": "exact"
}

选择逻辑：

1. 先查找 exact bucket；
2. 在 exact bucket 内过滤满足 SLO 的配置；
3. SLO 只包含 TTFT、TPOT、power、temperature，不包含 energy/token；
4. 在满足 SLO 的配置中选择 energy/token 最小的点；
5. 如果没有满足 SLO 的配置，选择 SLO violation 最小的配置，fallback_used=true；
6. 如果没有 exact bucket，则使用 nearest bucket；
7. 如果 nearest bucket 距离过大，则回退到 known-safe config；
8. 输出 selection_reason。

nearest bucket 的距离可以先用简单公式：

distance =
  alpha * abs(log(batch_query / batch_bucket))
+ beta  * abs(log(prompt_query / prompt_bucket))
+ gamma * abs(log(output_query / output_bucket))

如果 phase 不一致，需要增加较大惩罚项。

selection_reason 可取：

exact_min_energy_under_slo
nearest_min_energy_under_slo
closest_slo_feasible
fallback_known_safe
no_available_config

============================================================
八、任务 5：实现 evaluate_selector.py
============================================================

文件路径：

src/evaluate_selector.py

功能：

对比不同配置策略在已有数据上的表现，重点输出 selector regret 和 SLO violation rate。

对比策略：

1. MaxN / all_high
2. all_mid
3. energy-efficient / all_low
4. fixed_best_efficiency
5. oracle_best_per_bucket
6. ours_slo_aware_selector
7. ours_phase_aware_selector

定义：

MaxN / all_high:
  使用最高 GPU、最高 CPU、最高 EMC 配置。

all_mid:
  使用中档 GPU、CPU、EMC 配置。

energy-efficient / all_low:
  使用低频或已有数据中标记为 energy_efficient 的配置。

fixed_best_efficiency:
  在全局 profiling 数据或训练集中找到一个单一最低 energy/token 配置，并将它应用到所有 workload bucket。

oracle_best_per_bucket:
  对每个 workload bucket，选择该 bucket 内满足 SLO 的最低 energy/token 配置。

ours_slo_aware_selector:
  调用 select_config.py 输出配置。

ours_phase_aware_selector:
  调用 phase_aware_policy.py，分别选择 prefill_config 和 decode_config，并考虑切换开销。

核心指标：

energy_per_token_j
tokens_per_joule
ttft_ms
tpot_ms
avg_power_w
max_power_w
temperature_c
slo_violation_rate
infeasible_bucket_ratio
energy_regret_vs_oracle
mean_regret
p95_regret
max_regret

regret 定义：

energy_regret = (energy_selected - energy_oracle) / energy_oracle

如果某个 bucket 内没有任何配置满足 SLO，则标记 infeasible，不参与 energy regret 统计，但要计入 infeasible_bucket_ratio。

输出：

data/selector_eval/selector_comparison.csv
data/selector_eval/selector_regret.csv
data/selector_eval/selector_eval_summary.json
data/selector_eval/selector_eval_report.md

图表：

figures/phase3_analysis/selector_regret_vs_oracle.png
figures/phase3_analysis/selector_energy_comparison.png
figures/phase3_analysis/selector_slo_violation_rate.png

============================================================
九、任务 6：实现数据挖掘分析
============================================================

请在 analyze_existing_experiments.py 或独立工具函数中实现以下分析。

1. SLO 可行域分析

目标：

找出每个 workload bucket 下哪些频率配置满足 SLO。

输出：

data/analysis/slo_feasible_region.csv
figures/phase3_analysis/slo_feasible_region_heatmap.png

统计字段：

num_total_configs
num_slo_feasible_configs
slo_feasible_ratio
best_energy_feasible_config
best_latency_feasible_config
max_slo_margin_config

图表要求：

- 不满足 SLO 的点灰掉；
- 满足 SLO 的点按 energy/token 着色；
- 标注 MaxN、all_mid、oracle 和 ours。

2. 边际收益递减分析

目标：

识别频率继续升高后，延迟收益变小但功耗继续增加的 knee point。

对每个旋钮分别计算：

delta_ttft_per_freq
delta_tpot_per_freq
delta_energy_per_token_per_freq
delta_power_per_freq
delta_tokens_per_joule_per_freq

输出：

data/analysis/marginal_gain_by_knob.csv
figures/phase3_analysis/marginal_gain_gpu.png
figures/phase3_analysis/marginal_gain_cpu.png
figures/phase3_analysis/marginal_gain_emc.png

每个 workload bucket 需要输出：

latency_knee_freq
energy_knee_freq
recommended_freq_range

3. Prefill/Decode 瓶颈迁移分析

目标：

判断 prefill 和 decode 是否对不同频率旋钮敏感。

分组：

prefill
decode
prefill_heavy
decode_heavy
mixed

输出：

data/analysis/phase_bottleneck_shift.csv
figures/phase3_analysis/prefill_decode_sensitivity.png

关键指标：

gpu_sensitivity_tpot
cpu_sensitivity_ttft
emc_sensitivity_tpot
gpu_sensitivity_energy
emc_sensitivity_energy

报告中需要说明：

- prefill 是否更 GPU-sensitive；
- decode 是否更 EMC-sensitive；
- CPU 是否主要影响 TTFT 或 host-side latency；
- 如果 synthetic 数据不支持这些结论，需要明确说明真实 Jetson 实验仍需验证。

4. Selector regret 分析

由 evaluate_selector.py 输出。

重点比较：

MaxN / all_high
all_mid
fixed_best_efficiency
oracle_best_per_bucket
ours_slo_aware_selector
ours_phase_aware_selector

输出：

data/selector_eval/selector_regret.csv
figures/phase3_analysis/selector_regret_vs_oracle.png

============================================================
十、任务 7：实现 phase_aware_policy.py
============================================================

文件路径：

src/phase_aware_policy.py

功能：

实现离线 phase-aware 策略模拟，不做真实在线控制。

策略逻辑：

1. 对同一个 request/workload，分别查询 prefill 配置和 decode 配置；
2. 估计从 prefill_config 切换到 decode_config 的开销；
3. 如果切换收益大于切换成本，则启用 phase-aware；
4. 如果 decode 长度较短或 SLO margin 不足，则使用 single-config。

启用条件：

expected_energy_saving > switching_energy_overhead
switching_latency_overhead < slo_margin
decode_length > break_even_tokens

输出示例：

{
  "policy": "phase_aware",
  "prefill_config": {
    "gpu_freq_mhz": 1428,
    "cpu_freq_mhz": 1479,
    "emc_freq_mhz": 1600
  },
  "decode_config": {
    "gpu_freq_mhz": 846,
    "cpu_freq_mhz": 1479,
    "emc_freq_mhz": 2133
  },
  "switch_enabled": true,
  "break_even_tokens": 64,
  "selection_reason": "phase_aware_energy_saving"
}

如果当前数据不足以估计切换成本，则使用 configs/selector.yaml 中的默认切换开销，并在输出中标记：

switching_overhead_source = default_config

============================================================
十一、任务 8：更新 synthetic benchmark 说明，但不强制重写
============================================================

当前 synthetic benchmark 主要体现 GPU frequency 对性能的影响，因此可能天然支持“只调 GPU”的策略。

如果时间允许，请改进 SyntheticBenchmark，使其更接近真实 LLM 推理：

Prefill:
  更敏感于 GPU frequency。

Decode:
  更敏感于 EMC frequency 和 KV cache size。

TTFT:
  部分敏感于 CPU frequency。

Power:
  GPU、CPU、EMC 频率升高应产生不同功耗影响。

如果暂时不修改 synthetic benchmark，也必须在 phase3_preliminary_findings.md 中说明该限制。

============================================================
十二、任务 9：生成 Phase 3 preliminary findings 报告
============================================================

输出文件：

data/analysis/phase3_preliminary_findings.md

报告需要包括：

1. 数据来源说明；
2. synthetic benchmark 限制；
3. 4.1 到 4.7 的主要发现；
4. SLO 可行域分析；
5. 边际收益递减分析；
6. prefill/decode 瓶颈迁移分析；
7. selector regret 分析；
8. 当前数据能支持的结论；
9. 当前数据不能支持的结论；
10. 后续真实 Jetson + TensorRT-LLM 实验建议。

报告中必须明确：

当前数据能够支持：

- 验证分析流程；
- 验证可视化流程；
- 验证 SLO 过滤逻辑；
- 验证 rate table 构建；
- 验证 selector 输入输出格式；
- 验证 Pareto 与 regret 分析方法。

当前数据暂时不能支持：

- Jetson Orin 真实能效最优配置；
- CPU/EMC 对真实 LLM 推理的影响强度；
- Decode 阶段真实 KV cache 访存瓶颈；
- 真实频率切换开销；
- 最终论文级能效提升比例。

============================================================
十三、验收标准
============================================================

完成后应满足以下条件。

1. 能够运行：

python src/analyze_existing_experiments.py

并生成：

data/analysis/experiment_4_1_stability_summary.csv
data/analysis/experiment_4_2_knob_sensitivity.csv
data/analysis/experiment_4_3_config_comparison.csv
data/analysis/experiment_4_4_phase_summary.csv
data/analysis/experiment_4_5_workload_summary.csv
data/analysis/experiment_4_6_switching_summary.csv
data/analysis/experiment_4_7_slo_summary.csv
data/analysis/phase3_preliminary_findings.md

2. 能够运行：

python src/build_rate_table.py

并生成：

data/rate_tables/profile_raw.parquet
data/rate_tables/profile_agg_by_config.parquet
data/rate_tables/pareto_by_bucket.parquet
data/rate_tables/selector_table.parquet

3. 能够运行：

python src/select_config.py --workload examples/workload_query.json

并输出推荐配置 JSON。

如果没有 examples/workload_query.json，请自动生成一个示例 query。

4. 能够运行：

python src/evaluate_selector.py

并生成：

data/selector_eval/selector_comparison.csv
data/selector_eval/selector_regret.csv
data/selector_eval/selector_eval_report.md

5. Pareto frontier 必须按 workload bucket 计算。

6. energy/token 不能作为 SLO 硬约束，只能作为优化目标。

7. selector 必须支持：

exact bucket
nearest bucket
fallback known-safe config
no feasible config

8. evaluate_selector.py 必须包含：

MaxN / all_high
all_mid
fixed_best_efficiency
oracle_best_per_bucket
ours_slo_aware_selector

如果实现 phase_aware_policy.py，则还需要包含：

ours_phase_aware_selector

9. 所有生成报告都需要明确标注当前数据来自 synthetic benchmark，不代表真实 Jetson Orin + TensorRT-LLM 最终结论。

============================================================
十四、当前不做的事情
============================================================

本阶段不要优先实现以下内容：

online_controller.py 的完整在线控制逻辑
reinforcement learning based controller
复杂 ML 预测模型
生产服务接口
真实 TensorRT-LLM runtime 深度集成

这些内容应在以下条件满足后再推进：

rate table 已稳定生成；
selector 能够在离线数据上稳定输出配置；
selector regret 相比 fixed best 明显降低；
SLO violation rate 可控；
真实 Jetson + TensorRT-LLM 数据已接入；
frequency switching overhead 已真实测量。