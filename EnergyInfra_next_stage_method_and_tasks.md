# EnergyInfra 下一阶段方法与任务设计汇总

> 面向 Jetson Orin 边端 LLM 推理的 Workload-/SLO-/Thermal-aware Dynamic Frequency Capping 设计建议

## 0. 核心结论

当前工程已经完成了较完整的单请求 profiling、rate table、多目标 Pareto 选择和 E2E 验证。现有结果说明：  
如果继续主打 **single-request average E/token**，论文数值会偏弱，因为 default dynamic governor 本身已经较强，且当前 cap 搜索空间较小。

因此，下一阶段不建议继续单纯堆全量单请求 profiling，而应将工作主线调整为：

> **从单请求 E/token 优化，转向长期端侧 LLM serving 下的 SLO-stable thermal-aware energy management。**

也就是说，论文主指标应从：

```text
平均 E/token 提升
```

转为：

```text
在满足 p95/p99 TPOT SLO 的前提下，
降低长期平均功耗、峰值功耗、温度、thermal throttling 时间和 SLO violation。
```

下一阶段方法可以定位为：

> **EnergyInfra-Cap: Workload-, SLO-, and Thermal-aware Dynamic Frequency Capping for Edge LLM Inference**

---

## 1. 当前结果的判断

### 1.1 当前实验已经比较完整

当前工程已经完成：

```text
3 个模型：
- Qwen2.5-7B
- Llama-3.1-8B
- Qwen2.5-14B

9 种策略：
- pareto
- min_energy
- slo_50ms
- slo_45ms
- alpha_03
- alpha_07
- pwr_45w
- dynamic
- maxn

5 个 workload
3 次重复
共 378 runs
```

因此，现在的问题不是“数据太少”，而是：

```text
当前优化目标没有充分体现端侧 DVFS 的系统价值。
```

### 1.2 当前数值不够漂亮的原因

#### 原因 1：Dynamic baseline 本身很强

Jetson 默认 governor 已经可以根据底层硬件利用率动态调整频率。  
如果只看单请求平均 E/token，workload-aware 策略很难显著超过 dynamic。

因此，EnergyInfra 不应该定位成“替代 governor”，而应定位成：

> 为 governor 提供 LLM-aware、SLO-aware、thermal-aware 的高层 cap 约束。

#### 原因 2：当前 Pareto 是折中策略，不是收益最大化策略

当前 Pareto 默认选择 knee point，本质上是一个无明确约束的折中点。  
它不一定在 E/token 上最优，也不一定在 power 或 SLO 场景下最优。

已有结果里可以看到：

```text
7B:
- pareto vs MAXN: 约 +4.5% E/token savings
- slo_50ms vs MAXN: 约 +7.7%
- pwr_45w vs MAXN: 约 +7.1%

8B:
- pareto vs MAXN: 约 +5.0%
- pwr_45w vs MAXN: 约 +6.8%

14B:
- pareto vs MAXN: 约 +5.4%
- alpha_07 vs MAXN: 约 +6.3%
```

这说明不是没有更好的点，而是当前默认 Pareto 选择策略不是最利于论文主结果的目标函数。

#### 原因 3：当前 cap 搜索空间较小

当前部署态 GPU cap 只有：

```text
612 / 816 / 1020 / 1300 MHz
```

只有 4 个候选点时，cap-mode Pareto frontier 很容易塌缩，方法收益也容易被压缩。  
后续应补充更细粒度 cap profiling。

#### 原因 4：单请求 E/token 不足以体现端侧价值

端侧 LLM 推理真正关心的是长期运行中的：

```text
- 热稳定性
- 功耗峰值
- thermal throttling
- p95/p99 TPOT
- SLO violation
- 频率震荡
- 长期 tokens/J
```

这些指标在单次请求平均 E/token 里体现不出来。

---

## 2. 下一阶段方法定位

建议将方法组织成三层：

```text
Offline phase-aware characterization
        ↓
Workload-aware initial cap selection
        ↓
Runtime SLO/thermal feedback DVFS controller
```

### 2.1 第一层：Offline Phase-aware Characterization

目的不是在线切频，而是建立对 workload 的理解。

需要回答：

```text
- 不同模型的 DVFS 行为是否不同？
- 不同 prompt/output length 下最优 cap 是否不同？
- prefill 和 decode 对频率的敏感性是否不同？
- 哪些模型是 compute-bound，哪些更接近 memory-bound？
- 单请求 DVFS 的 oracle upper bound 有多大？
```

### 2.2 第二层：Workload-aware Initial Cap Selection

请求到来时，根据：

```text
- model
- prompt length
- expected output length
- TTFT SLO
- TPOT SLO
- power budget
```

从 rate table 中选择初始 cap。

这个初始选择可以来自：

```text
- SLO-constrained selection
- power-budget selection
- min-energy selection
- Pareto frontier pruning
- closest-to-ideal
```

但论文主方法不建议只写 “Pareto knee point”，而应写成更明确的 SLO/thermal-aware constrained optimization。

### 2.3 第三层：Runtime SLO/Thermal Feedback Controller

动态 DVFS 应作为在线闭环控制器，而不是请求内部 phase switching。

输入状态包括：

```text
- current cap
- current GPU temperature
- temperature slope
- recent p50/p95/p99 TPOT
- SLO slack
- SLO violation rate
- current/average/max power
- recent tokens/s
- queue or request mix state
```

输出动作包括：

```text
- keep cap
- raise cap by one step
- lower cap by one step
- fallback to max cap
- enter thermal-protection mode
```

---

## 3. 动态 DVFS 的正确角色

### 3.1 不建议的方式

不建议继续采用：

```text
prefill 开始 → 切高频
decode 开始 → 切低频
```

原因是已有实验说明，用户态 sysfs/debugfs 的 phase-boundary hard switching 开销过大，约数百毫秒级，可能占 TTFT 的很大比例。  
这会导致在短请求和中等请求中，切换开销抵消甚至超过节能收益。

### 3.2 建议的方式

动态 DVFS 应该改为：

```text
request-level 初始 cap
+
window-level 动态修正 cap
```

即：

```text
每个请求开始前：
  根据 workload 选择初始 cap。

每个 serving window：
  根据温度、SLO slack、p95/p99 TPOT 和功耗修正 cap。
```

窗口可以定义为：

```text
- 每 5 秒
- 每 10 秒
- 每 N 个请求
```

### 3.3 控制规则示例

```text
if p95_TPOT > 0.95 * TPOT_SLO:
    raise cap by one step

elif SLO_violation_rate > threshold:
    raise cap quickly or fallback to max cap

elif temperature > thermal_threshold and SLO_slack is sufficient:
    lower cap by one step

elif temperature_slope > slope_threshold and SLO_slack is sufficient:
    lower cap by one step

elif power > power_budget:
    lower cap by one step

else:
    keep cap
```

为了避免频率震荡，需要加入 hysteresis：

```text
- 连续 K 个 window 满足降频条件才降频
- 一旦出现 SLO violation 可以快速升频
- 升频和降频使用不同阈值
- 两次 cap 调整之间设置最小间隔
```

---

## 4. Prefill / Decode 是否还需要区分

结论：

> **需要区分，但只作为 workload characterization 和 rate-table 建模维度，不作为在线 DVFS 切换粒度。**

### 4.1 为什么还需要区分

prefill 和 decode 对指标的影响不同：

```text
prefill:
  主要影响 TTFT

decode:
  主要影响 TPOT、tokens/s、长期功耗和温度
```

prompt/output 结构也会影响最优 cap：

```text
长 prompt + 短 output:
  prefill 占比高，TTFT 更重要

短 prompt + 长 output:
  decode 占比高，TPOT 和长期功耗更重要

长 prompt + 长 output:
  TTFT、TPOT、功耗和温度都重要
```

因此，论文中仍然应该保留：

```text
Phase-aware characterization
```

用来解释 workload-aware cap selection 的必要性。

### 4.2 为什么不作为在线切换粒度

在线阶段不建议按 prefill/decode 边界切换 exact frequency，原因是：

```text
- 用户态切频开销过高
- 短请求中收益被开销抵消
- 中等请求中收益容易被静态功耗和切换延迟稀释
- 实现复杂且稳定性不足
```

更合理的表述是：

> Prefill/decode 是分析粒度，不是在线控制粒度。

### 4.3 推荐论文叙事

可以写成：

```text
Although prefill and decode exhibit different sensitivity to frequency scaling,
request-internal hard DVFS switching is not suitable on Jetson due to high userspace
switching overhead. Therefore, EnergyInfra uses phase-aware profiling to build
workload representations, but performs runtime control at request/window granularity
through frequency capping.
```

中文表述：

> 虽然 prefill 和 decode 对频率调节表现出不同敏感性，但 Jetson 用户态路径下请求内部 hard DVFS 切换开销较高。因此，EnergyInfra 将 phase 作为离线表征和建模粒度，而将在线控制粒度上移到 request/window 级别，通过 frequency cap 实现低开销动态调节。

---

## 5. 后续数据采集设计

建议将数据分成三类，不要全部做成大规模 phase-level profiling。

---

### 5.1 A 类：Phase-aware Characterization 数据

目的：

```text
支撑论文 observation 和 workload 建模。
```

采集内容：

```text
不同模型、不同 workload、不同 GPU freq/cap 下：
- TTFT
- prefill latency
- decode TPOT
- tokens/s
- energy/token
- avg/max power
- temperature
```

规模：

```text
不需要非常大，只需要覆盖典型 workload 和典型频率点。
```

用途：

```text
Observation 1:
  不同模型的 DVFS 行为不同。

Observation 2:
  prefill 和 decode 对频率敏感性不同。

Observation 3:
  prompt/output ratio 会影响最优 cap。

Observation 4:
  phase boundary 可分析，但不适合作为在线 hard switching 控制点。
```

---

### 5.2 B 类：Request-level 主 Rate Table

这是 selector 真正使用的表。

每条记录建议包含：

```text
model
prompt_length
output_length
target_gpu_cap
actual_gpu_freq_mean
actual_gpu_freq_p50
actual_gpu_freq_p95
TTFT
TPOT_p50
TPOT_p95
TPOT_p99
total_latency
energy_per_token
tokens_per_joule
avg_power
max_power
temperature_start
temperature_end
temperature_max
is_complete_output
SLO_met
```

用途：

```text
- workload-aware initial cap selection
- oracle gap analysis
- runtime controller 的初始预测
- per-workload cap map 可视化
```

---

### 5.3 C 类：Long-running Serving Trace

这是下一阶段最重要的数据。

每个 window 记录：

```text
window_id
time_start
time_end
request_mix
selected_cap
actual_gpu_freq_mean
actual_gpu_freq_p95
avg_power
max_power
temperature_mean
temperature_max
temperature_slope
tokens/s
p50_TPOT
p95_TPOT
p99_TPOT
SLO_violation_rate
throttling_event
cap_switch_count
```

用途：

```text
- 评估长期 tokens/J
- 评估温度稳定性
- 评估 p95/p99 TPOT
- 评估 SLO violation
- 评估 thermal throttling 预防效果
- 评估频率震荡
```

---

## 6. 具体任务设计

## Task 1：Oracle Gap Analysis

### 目标

先判断当前数值不高的根本原因：

```text
是 selector 没选好？
还是平台/运行时的单请求 DVFS 空间本身有限？
```

### 需要比较的策略

```text
Dynamic governor
MAXN
Best static cap
Current Pareto
Oracle-Energy
Oracle-SLO
Oracle-Power
Oracle-Thermal
```

### Oracle 定义

```text
Oracle-Energy:
  每个 workload 选择实测 E/token 最低的 cap。

Oracle-SLO:
  每个 workload 选择满足 TPOT SLO 的最低能耗 cap。

Oracle-Power:
  每个 workload 选择满足 power budget 的最低 TPOT cap。

Oracle-Thermal:
  每个 serving window 选择不会导致温度持续上升的最高效 cap。
```

### 判断标准

```text
如果 Oracle 相比 Dynamic 也只有 3%–5%：
  单请求 DVFS 空间确实有限，论文不应主打 E/token。

如果 Oracle 能到 10%–20%，但当前 Pareto 只有 1%–5%：
  selector 策略还有明显优化空间，应优先优化策略。
```

### 输出

```text
- dynamic-vs-oracle gap
- per-model oracle upper bound
- per-workload oracle best cap
- Pareto 与 oracle 的差距
```

---

## Task 2：Fine-grained Cap Profiling

### 目标

补充部署态 cap 搜索空间，避免只有 4 个 cap 导致 Pareto 前沿过小。

### 建议 GPU cap

```text
408 / 510 / 612 / 714 / 816 / 918 / 1020 / 1122 / 1224 / 1300 MHz
```

### 建议探索方式

不建议一开始全量暴力扫，而是采用 adaptive refinement：

```text
第一轮：
  使用已有 4 档粗扫结果。

第二轮：
  围绕每个模型 sweet spot 补点：
    7B: 612–918 MHz
    8B: 408–816 MHz
    14B: 816–1300 MHz

第三轮：
  只将有潜力的点纳入完整 E2E benchmark。
```

### 输出

```text
- 更细粒度 cap rate table
- per-model sweet spot
- per-workload cap map
- refined Pareto frontier
```

---

## Task 3：Request-level Rate Table 构建

### 目标

建立真正用于在线 selector 的部署态 rate table。

### 数据字段

```text
model
prompt_length
output_length
target_gpu_cap
actual_gpu_freq_mean/p50/p95
TTFT
TPOT_p50/p95/p99
total_latency
energy_per_token
tokens_per_joule
avg_power
max_power
temperature_start/end/max
is_complete_output
SLO_met
```

### 聚合统计

对每个 `(model, prompt_length, output_length, cap)` 统计：

```text
mean
median
std
p50
p95
p99
CV
```

### 作用

```text
- 支撑初始 cap selection
- 支撑 oracle gap analysis
- 支撑 thermal controller 的预测
- 支撑论文图表
```

---

## Task 4：Thermal-SLO Feedback Controller

### 目标

实现下一阶段核心在线机制。

### 控制粒度

```text
request-level:
  请求开始前选择初始 cap。

window-level:
  每 5s / 10s / N requests 根据状态修正 cap。
```

### 输入

```text
model
prompt_length
expected_output_length
current_cap
current_temperature
temperature_slope
recent_p50_TPOT
recent_p95_TPOT
recent_p99_TPOT
SLO_violation_rate
avg_power
max_power
recent_tokens_per_second
```

### 输出

```text
keep cap
raise cap by one step
lower cap by one step
fallback to max cap
enter thermal-protection mode
```

### 规则示例

```python
def update_cap(state):
    if state.slo_violation_rate > violation_threshold:
        return raise_cap_fast()

    if state.p95_tpot > 0.95 * state.tpot_slo:
        return raise_cap_one_step()

    if state.temperature > thermal_threshold and state.slo_slack > slack_threshold:
        return lower_cap_one_step()

    if state.temperature_slope > slope_threshold and state.slo_slack > slack_threshold:
        return lower_cap_one_step()

    if state.avg_power > power_budget and state.slo_slack > slack_threshold:
        return lower_cap_one_step()

    return keep_cap()
```

### Hysteresis 机制

```text
- 连续 K 个 window 满足降频条件才降频
- SLO violation 可立即升频
- 升频阈值和降频阈值分离
- 两次 cap 调整之间设置 cooldown interval
- 记录 cap_switch_count，避免过度震荡
```

---

## Task 5：Long-running Serving Benchmark

### 目标

将论文主结果从单请求 E/token 转到长期服务稳定性。

### Trace 设计

#### Trace A：Short-chat dominated

```text
70% p64_o64
20% p128_o128
10% p512_o128
```

#### Trace B：Long-generation dominated

```text
20% p64_o64
30% p128_o512
30% p512_o512
20% p1024_o512
```

#### Trace C：Bursty mixed workload

```text
短请求和长请求交错
加入连续 5–10 个高负载 burst
模拟真实端侧服务中的突发负载
```

### 实验时长

```text
Exploratory:
  30 min

Main evaluation:
  60 min
```

### Baselines

```text
MAXN
Dynamic governor
Best static cap
pwr_45w
Current Pareto
Oracle
Thermal-SLO-Aware controller
```

### 主指标

```text
average tokens/J
average power
peak power
max temperature
time above thermal threshold
thermal throttling duration
p50/p95/p99 TPOT
SLO violation rate
cap switch count
frequency oscillation count
```

### 预期主结果形式

不要只汇报：

```text
E/token improved by X%
```

而要汇报：

```text
在满足 p95/p99 TPOT SLO 的前提下：
- 平均功耗降低 X%
- 峰值功耗降低 X%
- 最高温度降低 X°C
- thermal throttling 时间降低 X%
- SLO violation rate 降低 X%
- tokens/J 提升 X%
```

---

## 7. 推荐实验矩阵

### 7.1 单请求 request-level profiling

```text
Models:
  7B / 8B / 14B

GPU caps:
  408 / 510 / 612 / 714 / 816 / 918 / 1020 / 1122 / 1224 / 1300

Workloads:
  p64_o64
  p128_o128
  p128_o512
  p512_o128
  p512_o512
  p1024_o512
  p2048_o512

Repeats:
  exploratory: 2
  main: 3
```

### 7.2 Long-running serving

```text
Models:
  7B first
  then 8B / 14B

Traces:
  short-chat
  long-generation
  bursty-mixed

Duration:
  30 min exploratory
  60 min main

Strategies:
  MAXN
  Dynamic
  Best static cap
  Current Pareto
  Thermal-SLO controller
  Oracle
```

---

## 8. 论文叙事建议

### 8.1 原叙事的问题

原来的叙事容易变成：

```text
我们做了 Pareto DVFS，
相比 dynamic E/token 提升 1%–2%。
```

这个主结果偏弱。

### 8.2 推荐叙事

建议改成：

```text
1. Characterization
   Jetson LLM 推理的最优频率不是单调的；
   不同模型和 workload 的 DVFS 行为不同；
   prefill/decode 敏感性不同；
   但请求内部 phase switching 开销过高。

2. Motivation
   Default governor 不理解 LLM 请求语义、SLO 和温度趋势；
   静态 cap 无法适应 workload 变化；
   因此需要 workload-aware + SLO-aware + thermal-aware dynamic cap。

3. Method
   Offline rate table
   Workload-aware initial cap
   Thermal-SLO feedback controller
   Hysteresis-based stable adjustment

4. Evaluation
   Single-request oracle gap
   Per-workload cap selection
   Long-running serving
   p95/p99 TPOT
   SLO violation
   temperature
   peak power
   tokens/J
```

### 8.3 推荐主结论表达

推荐写成：

> 在长期端侧 LLM serving 中，EnergyInfra 在满足 p95/p99 TPOT SLO 的前提下，显著降低平均功耗、峰值功耗和热限频风险，并保持接近 MAXN/Dynamic 的响应性能。

而不是：

> Pareto 比 dynamic E/token 提升 1%–2%。

---

## 9. 可视化设计建议

### Figure 1：Motivation Figure

内容：

```text
不同模型 / workload 下 GPU cap 与 E/token、TPOT、power 的关系
```

目的：

```text
证明固定 cap 或 default governor 不能覆盖所有场景。
```

### Figure 2：Prefill vs Decode Sensitivity

内容：

```text
prefill latency vs GPU freq
decode TPOT vs GPU freq
```

目的：

```text
证明 phase-aware characterization 有必要。
```

### Figure 3：Why Not Phase Switching

内容：

```text
phase switching overhead 与 TTFT 占比
```

目的：

```text
说明为什么不做 request-internal hard DVFS。
```

### Figure 4：System Overview

内容：

```text
Offline profiling → Rate table → Initial cap selection → Runtime feedback controller
```

### Figure 5：Long-running Temperature Curve

内容：

```text
不同策略下 60min temperature over time
```

### Figure 6：Power and TPOT Tradeoff

内容：

```text
average power / peak power / p95 TPOT / SLO violation
```

### Figure 7：Cap Adaptation Timeline

内容：

```text
selected cap over time
temperature over time
p95 TPOT over time
```

目的：

```text
展示 dynamic controller 如何根据系统状态调整 cap。
```

---

## 10. 执行优先级

### P0：Oracle Gap Analysis

优先级最高。

目标：

```text
判断单请求 DVFS 是否还有理论优化空间。
```

输出：

```text
dynamic-vs-oracle gap
per-model oracle upper bound
per-workload best cap
```

---

### P1：Fine-grained Cap Profiling

目标：

```text
补充部署态 cap 点，避免 4 档 cap 搜索空间过小。
```

输出：

```text
refined cap rate table
per-model sweet spot
per-workload cap map
```

---

### P2：Thermal-SLO Feedback Controller

目标：

```text
实现下一阶段核心方法。
```

输出：

```text
runtime controller
window-level cap adjustment log
hysteresis policy
```

---

### P3：Long-running Serving Benchmark

目标：

```text
产生论文主结果。
```

输出：

```text
temperature curve
power curve
p95/p99 TPOT
SLO violation rate
thermal throttling duration
tokens/J
```

---

### P4：Paper Story and Figures

目标：

```text
将论文从 Pareto E/token 优化，改写成 SLO-stable energy management。
```

输出：

```text
motivation figures
system overview
long-running evaluation
oracle gap analysis
```

---

## 11. 最终确认的技术路线

最终建议路线为：

```text
保留 prefill/decode 作为离线分析维度；
放弃请求内部 phase hard switching；
将动态 DVFS 设计成 request/window 粒度的 Thermal-SLO-aware cap controller；
通过 long-running serving 实验证明其在温度、p95/p99 TPOT、SLO violation 和功耗峰值上的优势。
```

一句话概括：

> **EnergyInfra 的下一阶段不应再追求单次请求 E/token 的小幅提升，而应转向长期端侧 LLM 服务中的动态功耗—热稳定—SLO 协同控制。**

---

## 12. 给 Codex / Claude 的执行提示

可以直接给工程助手如下任务：

```text
请基于当前 EnergyInfra 工程实现 Phase 13：

1. 增加 oracle gap analysis 脚本：
   - 输入现有 cap selector benchmark 和 cap rate table
   - 输出 Dynamic / MAXN / Static / Pareto / Oracle 的对比
   - 支持 Oracle-Energy、Oracle-SLO、Oracle-Power

2. 扩展 fine-grained cap profiling：
   - GPU cap 增加到 408,510,612,714,816,918,1020,1122,1224,1300 MHz
   - 保留现有 workload，并增加长输出 workload
   - 输出 refined cap rate table

3. 实现 ThermalSLOCapController：
   - request-level initial cap selection
   - window-level feedback adjustment
   - 输入 temperature, temp_slope, p95/p99 TPOT, SLO violation, avg/max power
   - 支持 hysteresis 和 cooldown

4. 实现 long-running serving benchmark：
   - 支持 short-chat、long-generation、bursty-mixed 三类 trace
   - 支持 MAXN、Dynamic、BestStatic、Pareto、ThermalSLO、Oracle baseline
   - 每个 window 记录 cap、temperature、power、TPOT、SLO violation

5. 增加 visualization：
   - temperature over time
   - p95/p99 TPOT over time
   - cap adaptation timeline
   - power/temperature/SLO summary
   - oracle gap summary
```

