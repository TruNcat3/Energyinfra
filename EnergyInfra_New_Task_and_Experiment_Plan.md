# EnergyInfra 新任务修改计划、实验计划与设计说明

**文档版本**：v2.0  
**日期**：2026-05-20  
**项目方向**：Jetson 端侧 LLM 推理能效优化  
**核心调整**：从 request 内部 phase-boundary hard DVFS 转向 workload/thermal-aware frequency cap  
**目标读者**：Codex、工程实现人员、实验设计人员、论文/报告撰写人员  

---

## 0. 文档目的

本文件用于重新定义 EnergyInfra 后续工程任务和实验路线。此前 Phase 6/7 已完成细粒度 GPU×EMC profiling、GPU×EMC×CPU 三维 rate table、alpha-weighted selector、true phase-switching E2E benchmark 和 strong/weak baseline 对比。最新实验说明，基于用户态 sysfs/debugfs 的 prefill→decode 边界强制调频开销过大，难以在单次请求内获得稳定收益。

因此，本文件将后续任务从 request 内部 true phase-switching hard DVFS 调整为：**离线 lock-frequency profiling + 在线 workload/thermal-aware frequency cap**。

新的核心思想是：离线阶段仍通过锁频 profiling 学习能耗—性能边界；在线阶段不再强制锁频，而是在 request/window 粒度设置 GPU/EMC/CPU 频率上限，让 Jetson governor 在 cap 范围内快速动态调节。

---

## 1. 当前实验结论回顾

### 1.1 已验证的有效结论

1. Jetson Orin 上 LLM 推理的 GPU 频率—能耗关系不是单调的。最低频不一定最省能，最高频不一定最高效，中间频率可能形成能效甜点。
2. EMC 频率对 Phi-3-mini-Q4 当前负载不是一阶瓶颈。提高 EMC 频率可降低部分 TPOT，但功耗增长可能抵消性能收益。
3. 弱 baseline 到合理 DVFS 配置之间存在空间，说明 DVFS 不是完全没有价值。
4. default dynamic governor 表现不弱，说明后续策略需要利用 governor 不知道的高层 LLM workload 语义，而不是简单替代 governor。

### 1.2 已验证的不适合路径

1. 用户态 request 内部调频开销过高。当前 `set_gpu()`、`set_emc()`、`set_cpu()` 都通过 sysfs/debugfs 同步写入，GPU/EMC/CPU 三个频率域切换合计约 700 ms。
2. phase-boundary hard switching 不适合作为 Jetson 在线控制主路径。对短输出请求，切换开销远大于可节省能耗；对中等输出请求，收益被切换时间和静态功耗稀释。
3. 单次 E/token 不足以体现端侧 DVFS 价值。端侧场景更关注长期温度、功耗峰值、throttling、p95/p99 TPOT 和 SLO violation。

---

## 2. 新设计动机

### 2.1 放弃 phase-boundary 后，动机从哪里出发

放弃 phase-boundary 并不意味着工作动机减弱，而是说明端侧 LLM 推理的合适控制粒度需要重新选择。新的动机应从以下矛盾出发：

> 端侧 LLM 推理受功耗、温度、SLO 和请求长度共同约束。不同请求的 prompt length、expected output length、decode 占比和延迟目标不同，导致最优 GPU/EMC/CPU 运行点随 workload 和状态变化。默认 governor 虽能根据底层负载反应，但不感知 LLM 请求语义；手工固定功耗档位又无法适应请求多样性。因此，需要一个 workload-aware 和 thermal-aware 的 frequency cap 系统，在不引入细粒度调频开销的前提下，为默认 governor 提供高层约束。

### 2.2 新动机的四个观察

**Observation 1：端侧 LLM 的最优配置随 workload 变化。**  
短 prompt/短 output、长 prompt/短 output、短 prompt/长 output、长 prompt/长 output 对 TTFT、TPOT、功耗和温度的敏感性不同。固定 MAXN、固定 30W、固定低频或固定 E_min 都无法覆盖所有请求。

**Observation 2：默认 governor 缺少 LLM 语义。**  
Jetson governor 可以根据硬件利用率调节频率，但它不知道请求预计输出长度、SLO 侧重、持续请求状态、温度趋势和用户可接受的延迟 slack。因此，EnergyInfra 不应替代 governor，而应为 governor 提供 LLM-aware 的频率上限。

**Observation 3：request 内部强制切频不是合适控制粒度。**  
Prefill/decode 边界是自然的分析粒度，但不是 Jetson 用户态路径下的合适在线控制粒度。后续应把控制粒度上移到 request-level、batch-level、time-window-level、thermal-trigger-level 和 long-running serving-level。

**Observation 4：端侧优化目标应包含长期热稳定和 SLO 稳定性。**  
对于端侧平台，平均 E/token 不是唯一目标。长期 tokens/J、温度曲线、功耗峰值、thermal throttling、p95/p99 TPOT、SLO violation 和频率震荡同样重要。

---

## 3. 新方法总览

### 3.1 方法名称建议

建议论文/报告中使用：

> **EnergyInfra-Cap: Workload- and Thermal-aware Frequency Capping for Edge LLM Inference**

### 3.2 方法主线

1. **Offline lock-frequency profiling**  
   精确测量 GPU/EMC/CPU 不同运行点下的性能、功耗和温度。

2. **Energy rate table construction**  
   将 workload、配置、性能、能耗、温度聚合为能耗汇率表。

3. **Cap rate table conversion**  
   将 exact frequency 表扩展或重建为 dynamic governor + cap 条件下的 cap table。

4. **Workload-aware cap selection**  
   根据 prompt length、expected output length、SLO 和 temperature 选择 GPU/EMC/CPU cap。

5. **Native governor execution**  
   在线阶段不锁频，保留 governor，让实际频率在 cap 范围内动态变化。

6. **Long-running serving optimization**  
   在持续服务中通过 cap policy 降低功耗峰值、温度和 SLO violation。

### 3.3 与旧方法的差异

| 项目 | 旧方法：Phase Switching | 新方法：Frequency Cap |
|---|---|---|
| 控制粒度 | prefill→decode 边界 | request/window/thermal interval |
| 在线动作 | 强制锁定 exact frequency | 设置 frequency upper bound |
| governor | 被绕开或强制 performance | 保留 governor 自适应 |
| 调频路径 | 多次 sysfs/debugfs 同步写 | 少量 cap 更新 |
| 主要风险 | 切换开销大 | cap 效果需实测 |
| 主要指标 | 单次 E/token | 长期 tokens/J、温度、p95 TPOT、SLO |
| 适用场景 | 理想快速 DVFS 硬件 | Jetson 用户态可部署系统 |

---

## 4. 工程修改计划

## Phase 8：Frequency Control 重构

### 4.1 目标

将频率控制从单一 lock-frequency 模式扩展为三种模式：

- `lock mode`：用于离线 profiling；
- `cap mode`：用于在线控制；
- `dynamic mode`：用于恢复默认 governor 和 baseline。

### 4.2 建议文件

```text
src/controller/freq_controller.py
src/controller/cap_controller.py
src/controller/governor_controller.py
```

如果希望减少改动，可先在 `freq_controller.py` 中实现所有模式，再逐步拆分。

### 4.3 API 设计

```python
class FrequencyController:
    def lock_config(self, gpu_mhz: int, emc_mhz: int, cpu_mhz: int) -> dict:
        """Profiling-only. Force GPU/EMC/CPU to exact frequencies."""

    def set_cap(self, gpu_cap_mhz: int, emc_cap_mhz: int, cpu_cap_mhz: int) -> dict:
        """Online mode. Set max frequency caps while keeping dynamic governors."""

    def restore_dynamic(self) -> dict:
        """Restore default dynamic governors and full frequency range."""

    def read_actual_state(self) -> dict:
        """Read current frequency, governor, temperature, and cap state."""
```

### 4.4 GPU lock mode

保留原逻辑：

```text
governor = performance
min_freq = target
max_freq = target
```

仅用于 profiling，不作为在线策略。

### 4.5 GPU cap mode

新增逻辑：

```text
governor = simple_ondemand or available dynamic governor
min_freq = low/default
max_freq = selected_cap
```

示意代码：

```python
def set_gpu_cap(mhz: int, min_mhz: int = 306):
    cap_hz = GPU_FREQS_HZ[mhz]
    min_hz = GPU_FREQS_HZ[min_mhz]

    try:
        write(f"{GPU_PATH}/governor", "simple_ondemand")
    except OSError:
        pass

    write(f"{GPU_PATH}/min_freq", str(min_hz))
    write(f"{GPU_PATH}/max_freq", str(cap_hz))
    return read_current_gpu_freq()
```

在线路径不应默认 `sleep(0.3)`。应记录 apply latency，但不阻塞等待稳定。不能假设 target cap 等于 actual frequency。

### 4.6 EMC cap mode

新增逻辑：

```text
EMC_MIN = low/default
EMC_MAX = selected_cap
```

示意代码：

```python
def set_emc_cap(mhz: int, min_mhz: int = 204):
    write(EMC_MIN, str(EMC_HZ[min_mhz]))
    write(EMC_MAX, str(EMC_HZ[mhz]))
    return read_current_emc_freq()
```

EMC 可能受 BPMP、thermal、memory demand 限制，必须记录 `target_emc_cap_mhz` 和 `actual_emc_mhz`。

### 4.7 CPU cap mode

```python
def set_cpu_cap(mhz: int, min_mhz: int = 1036):
    for cpu in cpus:
        write(cpu/scaling_governor, "schedutil")
        write(cpu/scaling_min_freq, min_hz)
        write(cpu/scaling_max_freq, cap_hz)
```

CPU cap 作为辅助维度，第一阶段可先固定或只测两档。

### 4.8 验收标准

- `lock_config()`、`set_cap()`、`restore_dynamic()` 均可运行；
- cap mode 不再强制 `min=max`；
- cap mode 单次 apply latency 显著小于原 lock path；
- 每条实验记录包含 target cap、actual frequency 和 governor；
- lock mode 和 cap mode 可在同一脚本中清晰区分。

---

## Phase 9：Benchmark Runner 修改与 EOS 控制

### 5.1 目标

解决长输出 workload 中 output_tokens 不稳定的问题，使长任务实验可比较。

### 5.2 修改点

在 `llama_cpp_runner.py` 中增加 benchmark mode：

```python
def run_single_inference(..., benchmark_mode: bool = False):
    if benchmark_mode:
        temperature = 0.0
        top_p = 1.0
        repeat_penalty = 1.0
        stop = []
    else:
        temperature = 0.7
```

如果 llama-cpp-python 支持 ignore_eos 或类似参数，应加入：

```python
ignore_eos = True
```

如果不支持，则构造不容易触发 EOS 的 synthetic prompt，并在实验分析中只统计 complete-output runs。

### 5.3 新增字段

```text
target_output_tokens
output_tokens
is_complete_output
completion_ratio
benchmark_mode
temperature
top_p
stop_policy
```

### 5.4 验收标准

- p128_o512、p128_o1024、p512_o1024 的 complete-output ratio 显著提高；
- 所有主实验可过滤 complete-output-only；
- 不再将 output_tokens 不足的 run 混入主结论。

---

## Phase 10：Cap Profiling 与 Cap Rate Table

### 6.1 目标

建立 dynamic governor + cap 条件下的 cap rate table。旧 rate table 是 exact-frequency oracle，新表应反映实际在线部署条件。

### 6.2 实验矩阵

第一版建议：

```text
GPU cap: 612 / 816 / 1020 / 1300
EMC cap: 204 / 665 / 2133 / 3199
CPU cap: 1036 or 1497
Workloads: 6
Repeats: 3
```

Workloads：

```text
p128_o128
p128_o256
p128_o512
p512_o512
p128_o1024
p512_o1024
```

如果成本较高，可先固定 CPU cap，只测 GPU×EMC cap。

### 6.3 数据字段

```text
control_mode = cap
target_gpu_cap_mhz
target_emc_cap_mhz
target_cpu_cap_mhz
actual_gpu_mhz_mean
actual_gpu_mhz_p50
actual_gpu_mhz_p95
actual_emc_mhz_mean
actual_emc_mhz_p50
actual_emc_mhz_p95
actual_cpu_mhz_mean
actual_cpu_mhz_p50
actual_cpu_mhz_p95
gpu_governor
cpu_governor
ttft_ms
tpot_ms
tokens_per_second
energy_per_token_j
avg_power_w
peak_power_w
temperature_start_c
temperature_end_c
temperature_max_c
is_complete_output
```

### 6.4 聚合逻辑

新表 key：

```text
model × runtime × prompt_length × output_length × cap_config
```

其中 cap_config 为：

```text
gpu_cap_mhz × emc_cap_mhz × cpu_cap_mhz
```

### 6.5 输出文件

```text
data/rate_tables/cap_selector_table_*.parquet
data/rate_tables/cap_dvfs_rules_*.json
figures/cap_profiling/
```

### 6.6 验收标准

- 生成 cap rate table；
- 能画出 cap Pareto frontier；
- 能比较 cap mode 与 lock oracle；
- 能判断 default dynamic 是否已经接近 cap Pareto；
- complete-output-only 分析可用。

---

## Phase 11：Workload-aware Cap Selector

### 7.1 目标

实现输出 frequency cap 的 selector。

### 7.2 输入特征

基础版：

```text
prompt_length
expected_output_length
target_tpot_ms
target_ttft_ms
alpha
temperature_c
```

扩展版：

```text
recent_tpot_ms
recent_tps
recent_power_w
recent_gpu_freq
recent_emc_freq
queue_length
batch_size
arrival_rate
concurrency
```

### 7.3 策略类型

#### Static Rule Selector

用于快速验证：

```text
short output:
    GPU cap 816, EMC cap 204/665
medium output:
    GPU cap 816/1020, EMC cap 665
long output:
    GPU cap 1020/1300, EMC cap 665/2133
high temperature:
    reduce GPU/EMC cap one level
strict SLO:
    increase GPU cap one level
```

#### Alpha-weighted Cap Selector

沿用 alpha：

```text
alpha = 0.0 energy-oriented
alpha = 0.5 balanced
alpha = 1.0 latency-oriented
```

评分对象从 exact config 改成 cap config。

#### Thermal-aware Cap Selector

```text
temp < 60°C:
    use normal cap
60°C <= temp < 70°C:
    reduce cap by one level
temp >= 70°C:
    enforce thermal-safe cap
```

阈值需根据 Jetson 实测修正。

### 7.4 输出格式

```python
{
    "gpu_cap_mhz": 816,
    "emc_cap_mhz": 665,
    "cpu_cap_mhz": 1497,
    "policy": "workload_aware_cap",
    "reason": "medium output, normal temperature, balanced alpha",
    "pred_tpot_ms": ...,
    "pred_energy_per_token_j": ...,
    "pred_power_w": ...
}
```

### 7.5 验收标准

- selector 可读取 cap table；
- 支持 alpha sweep；
- 支持 thermal rule；
- 输出选择原因；
- 可与旧 WeightedSelector 并存；
- 不再输出 exact frequency 作为在线动作。

---

## Phase 12：Request-level Cap E2E Benchmark

### 8.1 目标

验证 request-level cap policy 相比 default dynamic governor 是否能获得更好的能效、温度或 SLO 稳定性。

### 8.2 对比策略

```text
default_dynamic
lock_oracle
static_cap_low
static_cap_balanced
static_cap_high
workload_aware_cap
thermal_aware_cap
MAXN
30W_mode
```

说明：

- `default_dynamic` 是核心 baseline；
- `lock_oracle` 只作为上界；
- `min_freq` 只作为 weak baseline，不作为主对比；
- `MAXN` 和 `30W_mode` 作为传统固定配置 baseline。

### 8.3 实验模式

短请求：

```text
p128_o64
p512_o128
```

中等请求：

```text
p128_o256
p128_o512
p512_o512
```

长 decode：

```text
p128_o1024
p512_o1024
```

### 8.4 指标

```text
avg E/token
tokens/J
TTFT p50/p95
TPOT p50/p95/p99
SLO violation
avg power
peak power
temperature max
frequency oscillation count
complete output ratio
```

### 8.5 验收标准

- workload_aware_cap 不显著恶化 SLO；
- 相比 default_dynamic 至少在某些 workload 上降低 power peak 或 temperature；
- complete-output-only 结论与 all-runs 结论分开报告；
- 如果 token/J 提升小，也要报告 thermal 和 tail-latency 指标。

---

## Phase 13：Long-running Serving Evaluation

### 9.1 目标

通过拉长任务时长，验证 cap policy 在持续服务场景下的价值。

这是后续最关键的实验，因为短时间单请求 E/token 很容易被静态功耗、EOS、温度初始状态和采样误差影响。

### 9.2 实验设置

建议运行 30–60 分钟连续请求流：

```text
arrival pattern:
    fixed interval
    or Poisson arrival

request mix:
    40% short
    40% medium
    20% long

model:
    Phi-3-mini-Q4 first
    then Qwen2.5-7B-Q4 / Llama-3-8B-Q4
```

### 9.3 对比策略

```text
default_dynamic
static_cap_balanced
workload_aware_cap
thermal_aware_cap
MAXN
30W_mode
```

### 9.4 控制周期

测试不同控制周期：

```text
request-level cap
5-second window cap
10-second window cap
thermal-triggered cap
```

### 9.5 关键指标

```text
long-term tokens/J
rolling E/token
rolling TPOT p95
temperature curve
time-to-throttle
throttling duration
power peak
SLO violation over time
frequency trace
```

### 9.6 预期结论

即使平均 E/token 只有 3%–8% 改善，只要能够证明：

- 温度降低；
- throttling 延后或减少；
- p95 TPOT 更稳定；
- 功耗峰值下降；
- SLO violation 减少；

该方法仍具有系统价值。

---

## Phase 14：可选增强：Frequency Daemon

### 10.1 目标

降低 Python 每次打开/写入 sysfs/debugfs 的软件开销，并为后续服务化控制做准备。

### 10.2 设计

```text
energyinfra-daemon
    - root 权限启动
    - 预打开 sysfs/debugfs 文件描述符
    - 接收 Unix domain socket 命令
    - 支持 set_cap / lock_config / restore_dynamic
    - 异步返回 apply result
```

### 10.3 协议示例

请求：

```json
{
  "cmd": "set_cap",
  "gpu_cap_mhz": 816,
  "emc_cap_mhz": 665,
  "cpu_cap_mhz": 1497,
  "mode": "cap"
}
```

返回：

```json
{
  "ok": true,
  "apply_latency_ms": 12.3,
  "actual_gpu_mhz": 612,
  "actual_emc_mhz": 204,
  "actual_cpu_mhz": 1036
}
```

### 10.4 注意事项

daemon 只能降低用户态文件操作和权限切换开销。如果硬件/driver 生效本身很慢，daemon 无法把 700 ms 变成 1 ms。因此 daemon 仍应服务于 request/window-level cap，而不是恢复 request 内部 phase hard switching。

---

## 11. 建议目录结构

```text
src/
  controller/
    freq_controller.py
    cap_controller.py
    cap_selector.py
    thermal_policy.py
  ratetable/
    build_cap_rate_table.py
    evaluate_cap_selector.py
  experiments/
    run_cap_profiling.py
    run_request_cap_e2e.py
    run_long_serving_cap.py
  visualization/
    visualize_cap_profiling.py
    visualize_request_cap_e2e.py
    visualize_long_serving.py
scripts/
  active/
    run_cap_profiling.sh
    run_request_cap_e2e.sh
    run_long_serving_cap.sh
docs/
  开发文档/
    EnergyInfra_New_Task_and_Experiment_Plan.md
  实验文档/
    Phase8_Cap_Controller_Report.md
    Phase9_Cap_Profiling_Report.md
    Phase10_Request_Cap_E2E_Report.md
    Phase11_Long_Serving_Report.md
```

---

## 12. 实施优先级

### P0：立即执行

1. 新增 `set_cap()` API；
2. 保留 `lock_config()` 作为 profiling-only；
3. 实现 `restore_dynamic()`；
4. 修改 runner，增加 benchmark mode，降低 EOS 干扰；
5. CSV 中记录 target cap、actual freq、governor、complete output；
6. 设计 request-level cap benchmark。

### P1：短期完成

1. 运行 cap profiling；
2. 构建 cap selector table；
3. 实现 cap selector；
4. 与 default_dynamic 比较；
5. 输出 request-level cap E2E 图表。

### P2：中期完成

1. 30–60 分钟 long-running serving；
2. thermal-aware cap；
3. frequency/power/temperature trace 可视化；
4. 更大模型验证；
5. multi-request / batch 场景。

### P3：可选

1. root daemon；
2. socket-based control；
3. online calibration；
4. MPC/RL-style controller；
5. kernel/governor-level policy。

---

## 13. 成功标准

### 最低成功标准

- cap mode 工程可运行；
- default_dynamic 与 workload_aware_cap 可公平对比；
- complete-output runs 可稳定生成；
- request-level cap 不显著恶化 SLO；
- 提供 E/token、TPOT、Power、Temperature 图表。

### 理想成功标准

- 相比 default_dynamic，workload_aware_cap 在 complete-output runs 上提升 3%–8% tokens/J；
- power peak 降低 5%–15%；
- 长时间运行最高温度降低 3–8°C；
- p95 TPOT 更稳定；
- thermal throttling 延后或减少。

### 强成功标准

- 在更大模型或长时间服务场景中，thermal_aware_cap 相比 default_dynamic 获得 10%+ long-term tokens/J 改善；
- SLO violation 不增加；
- 同时降低功耗峰值和温度。

---

## 14. 实验报告模板

后续每个实验报告建议包含：

```text
1. 实验目的
2. 实验平台
3. 模型与 runtime
4. 控制策略
5. workload 分布
6. 数据过滤规则
7. complete-output 统计
8. E/token 与 tokens/J
9. TTFT/TPOT p50/p95/p99
10. Power 与 temperature
11. Frequency trace
12. SLO violation
13. 与 default_dynamic 的差异
14. 结论与下一步
```

---

## 15. 新论文/报告叙事

建议采用以下主线：

> EnergyInfra 的实验表明，Jetson 端侧 LLM 推理中存在非单调的频率—能耗关系，但 request 内部 phase-boundary hard switching 在用户态路径下会引入显著调频开销。因此，我们将锁频 profiling 与在线控制解耦：离线阶段使用 lock mode 建立能耗汇率表，在线阶段通过 workload-aware 和 thermal-aware frequency cap 为系统 governor 提供高层约束。该方法避免了 prefill→decode 边界同步切频开销，并将优化目标从单次平均 E/token 扩展到长期 tokens/J、功耗峰值、温度和 SLO 稳定性。

---

## 16. 给 Codex 的执行提示

实现时请遵守：

1. 不要删除旧的 lock-frequency profiling 代码；
2. 新增 cap mode，不要替代 profiling mode；
3. phase switching 保留为 negative result，不作为主路径；
4. 所有实验必须记录 actual frequency；
5. E2E 分析必须区分 complete 和 incomplete output；
6. default_dynamic 是核心 baseline；
7. 长时间实验比单次请求实验更重要；
8. 若 token/J 收益较小，必须同时报告 power peak、temperature、p95 TPOT 和 SLO violation；
9. cap policy 不追求精确控制实际频率，而是限制频率上界；
10. 将 governor 视为底层快速响应机制，EnergyInfra 提供 workload-aware hint。

---

## 17. 参考方向

- GreenLLM：SLO-aware dynamic frequency scaling for energy-efficient LLM serving。
- BiScale：disaggregated LLM serving 中 coarse/fine 两级 phase-aware placement 与 DVFS。
- VoltanaLLM：frequency control 与 request routing 联合优化。
- Embedded DVFS governor：deadline slack、runtime counter、thermal-aware control。
- ZeroDVFS：利用高层语义和模型化方法减少 workload-specific profiling 依赖。
- SysScale：多频率域移动处理器能效控制。

---

## 18. 一句话总结

EnergyInfra 后续应从精确 phase-boundary hard DVFS 转向能耗汇率表驱动的 workload/thermal-aware frequency cap。也就是：**离线精确测，在线不硬切；让 governor 快速响应，让 EnergyInfra 提供 LLM 语义和热约束下的频率边界。**
