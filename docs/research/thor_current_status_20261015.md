# Thor 当前研究状态

最近更新：2026-10-22

## 总体判断

项目已经完成 Thor 上的工程适配和主要实验闭环，当前处于“按场景收敛多目标证据”的阶段，而不是继续盲目增加频率档位。主要风险从“能否运行”转移到了“trace、SLO 和重复统计是否能支持可比结论”。

## 已完成工作

| 方向 | 状态 | 说明 |
|---|---|---|
| 硬件控制 | 完成 | CPU/GPU/EMC、cap/lock/dynamic、恢复和 root 边界 |
| 采集层 | 完成 | tegrastats、功率、温度、telemetry gap 检查 |
| 运行时 | 完成 | llama.cpp Thor 构建、随机模型、fixed-work integrity gate |
| 资源筛选 | 完成 | batch、ubatch、parallel 等 Stage 1A 初筛 |
| serving | 完成 | 固定 trace、重复、日志、结果摘要和恢复 |
| 多目标分析 | 完成 v1 | 12 个来源、281 行统一证据，保留场景边界和双基线差值 |
| P8 容量 | 完成 | 高并发共享容量证据 |
| bursty 低并发 | 完成 | 90 个 full trials |
| serial think-time | 完成 | 30 秒间隔，已生成 full 结果 |
| queue-aware serving | 完成 | 1,041,216-policy search 后的硬件验证和 paired confirmation |
| targeted edge confirmation | 完成 | serial 15/15、bursty 45/45，五个 P1 候选同协议确认 |
| phase-switch gate | 完成但未晋级 | 180/180 blocks，所有 switch 候选未通过能量 gate |
| MoE fixed-horizon serving | 完成 | 同一固定窗口下完成 native dynamic、静态 CAP、MoE placement/DVFS 对比 |
| MoE phase-aware serving v1 | 完成 v1 | Long/Short 无并发 trace 完成 3 repeats；收益有限但可重复，尚不进入并发在线控制 |
| MoE phase-aware serving v2 | 完成 v2 | 提前 Prefill 切换、短输出 fallback、128/256/512 token 摊销测试完成；待 hybrid selector 验证 |
| MoE workload-aware hybrid v3 | 完成但未晋级 | 混合输出长度三重复完成；hybrid 能耗高于 native 和静态 peak guard，不进入在线策略 |
| Q4/Q5/Q8 precision pilot | 完成首轮 | 同源 gate 与 20-block pilot 已有，仍限定于 llama.cpp |
| 7B/14B scale pilot | 完成首轮 | 两个随机权重模型均有 5-block validation |

## 已有数据给出的 Thor 证据

### Queue-aware serving

在 600 秒 frozen arrival trace 的硬件验证中，Dynamic 完成率为 81.43%，未满足正式 joint SLO；MaxClock、MaxClockIdleDynamic、BalancedWeighted、QueueEnergyMin 和 QueuePowerMin 均为 100% 观测达标。`queue_power_min` 的验证结果为约 52.047 W、3.219 J/token，相对 MaxClock 约降低 5.47% J/token 和约 5.53% 功率；随后 165/165 请求的 paired confirmation 保留了 QueuePowerMin 的可行性和排序。该结论属于 queue-aware long-generation serving，不迁移为单用户串行结论。

### Targeted low-concurrency confirmation

五个 P1 候选完成了串行 15/15 和 bursty 45/45 trials。串行 `rate_008` 下，QueuePowerMin 为约 37.47 W、4.684 J/token，在 interactive 和 relaxed SLO 下通过全重复门槛；相对 native dynamic 的功率与 J/token 均降低约 4.77%，相对 MaxClock 均降低约 9.88%。严格 SLO 下没有全重复可行点。

bursty 的 4/8/12 tok/s 下，QueuePowerMin 相对 native dynamic 分别降低约 4.07%/2.62%/2.83% 功率和 5.01%/4.47%/6.52% J/token，但所有策略、所有速率均未通过 95% 全重复门槛。因此它是近可行的多目标证据，不是正式 SLO 约束收益。两组结果说明控制点必须按 arrival mode 选择，串行可行点不能直接迁移到突发队列。

### Long-duration thermal serving

长时 thermal full 完成 9/9 trials，每个 trial 均为 180/180 fixed-work 请求。QueuePowerMin 约为 37.87 W、4.734 J/token，相对 native dynamic 的功率和 J/token 均降低约 2.84%，相对 MaxClock 均降低约 10.02%。三种策略的峰值温度分别约为 58.5 C、64.0 C 和 61.2 C，未触发 85 C 安全阈值；功率和 TPOT 的早晚 workload-block 对齐漂移也保持在较小范围。

严格 thermal gate 仍未全重复通过：Native dynamic 为 1/3，MaxClock 和 QueuePowerMin 均为 2/3。失败主要是一次温度斜率门槛和一次 SLO 重复，不应被解释为 thermal runaway。该结果支持“长时收益方向存在”，但尚不足以写成无条件的 thermal-stable 结论。

### Model-space and MoE assessment

当前 Thor 收益偏小的首要原因不是控制器一定无效，而是实验对象仍偏向小型 dense 随机权重模型：7B Q4 模型可以完整驻留设备，单用户低并发下没有明显的专家放置、主机到 GPU 搬运或显存容量压力，频率变化只能优化已经较窄的 active bottleneck。MoE 评估和下一阶段 placement-aware 方案见 [Thor MoE 模型空间评估](thor_moe_model_space_assessment_20261018.md)。

### Phase-switch

phase-switch gate 已完成 180/180 blocks。GPU 1575 MHz/EMC 4266 MHz 的切换候选在三个 workload 上相对 no-switch 的能量变化约为 `+0.608%`、`-0.149%`、`-0.073%`，置信区间均不能支持严格的负能量结论；其余降频切换候选能量明显上升。因此 phase-aware 控制目前是完整的负结果：保留诊断价值，但不能晋级为在线能效策略。

### MoE fixed-horizon 与 phase-aware v1

为避免 request-active energy 的变量时长偏差，MoE serving 使用固定的
`periodic_short`（6 个 `128/128` 请求、40 秒）和 `periodic_long`（6 个
`1024/128` 请求、60 秒）窗口，并使用同一随机权重 Qwen3 MoE Q4_K_M、同一
llama.cpp Thor runtime 和三次重复。

- Long：`phase_high_to_memory_aware` 相对 `static_native_dynamic` 的固定窗口能耗和平均功率均下降约 3.6%，峰值功率下降约 14.9%；`phase_power_guard` 的峰值功率下降约 20.3%，但 TPOT p95 接近 31 ms。
- Short：最低固定窗口能耗来自 `static_peak_guard_1359_3200_972`，能耗和平均功率下降约 3.7%；phase-aware 最佳能耗点约下降 2.2%，但没有稳定超过静态点。
- `strict_20` 和 `balanced_30` 在当前 TTFT 约束下没有可行策略；`edge_50` 和 `relaxed_100` 才形成可解释的 Pareto 集合。
- Phase switch p95 约为 87~111 ms，说明切换本身是主要开销之一；短请求无法充分摊平该开销，Long decode 更适合继续探索 phase-aware。

因此当前结论是“phase-aware 在 Thor 上存在有限、场景相关的收益”，而不是已经得到通用在线策略。详细数据、SLO 和边界见
[Thor phase-aware 结果分析](thor_phase_aware_findings_20261022.md)。

### MoE phase-aware v2：切换摊销

v2 将 Prefill 切换提前约 250 ms，并对小于 256 output tokens 的请求回退到静态
peak guard。full 结果显示：

- `long_128`：phase-aware 请求回退后，静态 peak guard 相对 native dynamic 约降低 4.9% 固定窗口能耗；这不是 phase switch 的增量收益。
- `long_256`：`phase_long_high_to_memory_aware` 相对 native dynamic 约降低 0.4% 能耗，TTFT p95 约 952 ms。
- `long_512`：`phase_long_max_to_balanced` 相对 native dynamic 约降低 1.3% 能耗，TTFT p95 约 323 ms。
- `short_128`：native dynamic 仍是最低能耗点，phase-aware fallback 增加约 2.6% 能耗。
- 实际 phase switch p95 在 `45~143 ms`，提前 Prefill 切换改善了长输出 TTFT，但没有消除 Decode 切换的能耗成本。

因此当前更合理的控制形式是 workload-aware hybrid，而不是所有请求统一启用
phase switch：短请求使用 native，128 token 长请求使用静态 peak guard，256/512
token 长请求再分别选择 phase-aware 候选。

### Workload-aware hybrid v3：未形成稳健收益

v3 将上述规则放入同一个 `mixed_output_lengths` 固定 trace，在相同 120 秒
窗口中执行 `short_128`、`long_128`、`long_256` 和 `long_512`，并与 native
dynamic、静态 peak guard 做三次重复比较。full 结果为：

- native dynamic：`3623.24 J`、`30.19 W`、峰值 `79.29 W`；
- static peak guard：`3586.20 J`、`29.88 W`、峰值 `72.77 W`，相对 native 能耗下降约 1.0%；
- workload hybrid：`3642.12 J`、`30.35 W`、峰值 `77.70 W`，相对 native 能耗增加约 0.5%，相对静态 peak guard 增加约 1.6%；
- hybrid 的 TTFT/TPOT p95 为 `950.49/26.20 ms`，均差于 native dynamic 的 `722.76/23.01 ms`。

`edge_50` 下混合 trace 没有策略通过，`relaxed_100` 下三种策略均可行。quick
单次结果曾显示约 1.2% hybrid 能耗下降，但 full 三重复没有复现，因此不能作为
收益证据。当前应停止继续扩展 phase-aware 频点，保留该结果作为“全局硬件切换
开销抵消细粒度策略收益”的负结果。

### Precision 和 model scale

Q4/Q5/Q8 的结果共同显示，低频/低 EMC 点可以显著降低功率，但常以更高 J/token 和更差尾延迟为代价；高性能锁频点相对 MaxClock 的 J/token 改善通常只有约 0.5% 到 1.2% 的量级，不能包装成普遍收益。7B/14B pilot 也呈现同样边界：14B 的低功率点并不意味着更低 J/token，模型规模改变了工作点和可行 SLO，但没有推出单调的“模型越大越值得调频”规律。

上述数值来自实验机的 `data/protocol_v2/` 结果摘要，原始 telemetry 和模型不进入仓库；每项结论仍以对应协议的 hash、SLO 和重复规则为准。

## 研究认识

1. Thor 的收益不是在所有场景下同方向扩大；
2. 平均功率、每 token 能量和 TPOT 存在实际冲突；
3. P8 容量场景和低并发边端场景必须独立报告；
4. 严格 SLO 的失败可能来自突发 trace 和统计门槛，不应直接归因于硬件频率；
5. Pareto 前沿比单一“最优频率”更符合部署者的不同偏好。

## 当前方法结论

### 已经成立

- Thor 的优化对象应是 GPU、EMC、CPU、batch/parallel、arrival mode 和 phase 的联合配置，而不是单一 GPU 频率；
- Power 和 Energy/token 必须同时保留，降低平均功率不能自动等价为降低续航消耗；
- P1/P2 低并发、bursty low-concurrency 和 P8 shared capacity 是三个不同问题，必须分别建立 Pareto 前沿；
- 统一的流程应为 `profiling -> frozen trace -> repeated serving -> SLO filter -> Pareto -> online validation`；
- 随机权重模型能固定模型形状和系统变量，但不能支持质量、准确率或真实模型泛化结论。

### 尚未成立

- Thor 相对 native dynamic 的全面功率或 J/token 优势；
- phase-aware 在并发和长时间 serving 中稳定降低总能量；当前只在无并发固定窗口上观察到有限收益；
- Q4/Q5/Q8 或 7B/14B 的差异能稳定改变 Pareto 排序；
- 随机权重结果可以代表预训练模型或真实电池续航。

### 证据解释规则

`quick` 只能作为执行 gate；单次 repeat 只能作为探索性结果；总体达标率不能替代逐 repeat、逐 prompt bucket 的达标率；P8 容量点不能迁移为单用户边端结论。任何“收益”必须同时给出 baseline、SLO、workload、重复次数、Power 和 Energy/token。

## Thor 收口任务

| 优先级 | 任务 | 结束条件 | 当前状态 |
|---|---|---|---|
| P0 | 串行 think-time 统一汇总 | 3 repeats、prompt-aware SLO、Power/J-token/P95 | 已完成并确认可行 Pareto 点 |
| P0 | bursty 结果归档 | 完整 trials、失败/缺样本说明、近可行候选 | 已完成，无正式可行点 |
| P0 | 结果契约统一 | 绝对值、相对 dynamic/MaxClock、trace/hash | 已实现统一离线证据表 |
| P1 | Q4/Q5/Q8 同源 gate | 同一 F16 source、5 validation repeats | 首轮完成，llama.cpp 范围内 |
| P1 | 7B/14B matched pilot | 同一 SLO 下分别选择 Pareto | 首轮完成，仍为随机模型 |
| P1 | phase-switch gate | 配对能量 CI 上界低于 0 且 SLO 达标 | 完成但未通过 |
| P1 | phase-aware v1 fixed-horizon | Long/Short 三重复、固定窗口、SLO/Pareto 报告 | 已完成；收益有限，待 v2 优化切换开销 |
| P1 | queue-aware serving | 每个 block SLO 达标、paired confirmation | 已完成 |
| P2 | 长时间 thermal serving | 温度、漂移、恢复和 telemetry 完整 | full 已完成；收益方向稳定，但 thermal gate 为 1/3、2/3、2/3，尚非全重复认证 |
| P2 | MoE model-space study | 专家放置、路由统计、DVFS 和 SLO 联合证据 | runtime gate 已实现，等待外部 GGUF |
| P2 | 真实权重 sanity check | 模型 provenance 和最小质量集完整 | planned |

详细入口和出口条件见 [Thor 实验收口计划](thor_completion_plan_20261015.md)。

## 证据等级

| 等级 | 含义 | Thor 当前内容 |
|---|---|---|
| L0 | 工程可运行 | 已达到 |
| L1 | 单点/quick 可重复 | 已达到，但只能作探索性证据 |
| L2 | 同 trace、多重复、可恢复 | bursty full 已达到 |
| L3 | 跨场景、跨 SLO、Pareto 可解释 | serial 与 queue-aware 子场景达到；bursty 保留负结果 |
| L4 | 跨模型/精度/真实模型质量约束 | 尚未完成 |

## 下一阶段出口条件

- A/B/C 三类 workload 均有 native dynamic、max clock 和至少两个 EnergyInfra 候选；
- 每个场景至少 3 次重复，并报告原始达标率和置信区间；
- Power 与 Energy/token 同时报告；
- SLO 解释包含 TTFT、TPOT、prompt bucket 和重复规则；
- 所有结果能由配置文件、trace hash、runtime manifest 和命令复现。
