# 面向边端 GPU 的多目标能耗调优框架

**项目**：EnergyInfra
**平台实例**：Jetson Orin、Jetson AGX Thor 及后续 Jetson 边端 GPU
**统一运行时**：以 llama.cpp/GGUF 为主，平台可扩展其他 runtime
**当前主模型**：Orin 使用多模型历史基线；Thor 以随机权重 Qwen2.5 7B Q4_K_M 做系统控制变量
**文档日期**：2026-10-15

## 1. 框架定位与研究问题

EnergyInfra 研究的是边端 GPU LLM 推理的系统能效问题，而不是某个单一平台的频率脚本。Orin 和 Thor 通过统一的 profiling、SLO、Pareto 和在线控制方法接入同一研究框架；平台差异只保留在硬件 adapter、runtime manifest 和实验配置中。

边端 LLM 推理不是单一的“最快”或“最低功率”优化问题。降低 GPU/EMC/CPU 频率可能降低平均功率，却也可能增加排队和 decode 时间，使每 token 能量上升；提高频率可能缩短 busy time，也不一定降低平台总能耗。

EnergyInfra 将目标定义为：在固定工作量和指定 SLO 下，测量并搜索 `Power`、`Energy/token`、`TTFT`、`TPOT`、`Throughput` 和 `Temperature` 之间的非支配解集。最终策略由部署者在 Pareto 前沿上按偏好选择，而不是由框架宣称一个适用于所有负载的全局最优点。

## 2. 系统架构

```text
Hardware profiles → Resource screening → Frozen serving traces
        → Measurement → SLO-aware analysis → Pareto decision
        → static cap / phase switch / thermal feedback
```

`configs/` 描述实验身份和硬件策略，`scripts/active/` 负责编排，`src/` 保留通用控制器和指标模块，`data/protocol_v2/` 保存不可变 trace、摘要和结果，`docs/` 保存协议和解释。

## 3. 优化变量和指标

控制变量包括 GPU cap/lock、EMC cap、CPU frequency/floor、governor、batch、microbatch、parallel slots、prefill/decode 策略以及请求到达过程。

对每个固定 workload 和硬件策略记录：

```text
f(x) = (average_power_w,
        energy_per_output_token_j,
        TTFT_P95_ms,
        TPOT_P95_ms,
        temperature_max_c)
```

吞吐和 SLO 达标率是约束或分层筛选条件，不与能效指标混成一个未经说明的加权分数。

报告必须明确 TTFT 是否按 prompt length 分层、TPOT 阈值、重复规则、prompt bucket 规则、最低样本数、telemetry gap 和 stream coverage 门槛。

## 4. 实验场景分层

### A. 单用户串行边端

请求之间存在明确 think-time，主要观察 idle energy、单请求尾延迟和低并发功率。当前协议为每 30 秒提交一个请求。

### B. 低平均负载但有突发

平均 token rate 较低，但短时间内可能有多个请求接近到达，更接近事件驱动或间歇式 agent 请求，不能称为纯单用户场景。

### C. P8 共享容量

用于网关或小型共享服务，重点是 capacity、吞吐和功率/能效，而不是单用户交互延迟。

三类场景不能合并成一张“总排行榜”。同一个频率点在 A 中可能因 idle energy 占主导，在 C 中可能因 queueing 和 batch efficiency 获益。

## 5. 平台实例与当前证据

### 5.1 Jetson Orin

Orin 是当前已有数据最完整的平台，包含 GPU/EMC profiling、7B/8B/14B 模型、lock/cap rate table、workload-aware selector、378-run E2E benchmark、Oracle gap、MDR/JIR/HV、phase-aware 和 thermal-SLO 分析。已有结果显示 Pareto 方法相对 MAXN 在不同模型上有约 4.5% 到 5.5% 的 Energy/token 优势，功率收益最高约 22.8%。这些数字只对对应的模型、runtime、workload 和 SLO 有效。

### 5.2 Jetson AGX Thor

已确认：

- CPU/GPU/EMC 控制、硬件恢复、tegrastats 采集和固定工作完整性检查已经接入；
- Stage 1A 资源筛选和 Stage 1B serving 已形成可重复流程；
- 静态候选点至少包含 native dynamic、max clock、低功耗 EMC、c1/c3；
- 低并发 bursty full campaign 共 90 个 trial，已完成并恢复硬件；
- 30 秒 think-time 串行协议已完成运行，用于补充真正单用户边端证据。

`edge_c3_p2 @ rate_012` 在 bursty 低并发 full 数据中，TPOT 100 ms 下总体达标率约 90%，三次重复为 `[100%, 80%, 80%]`，平均功率约 40.5 W，TPOT P95 约 30.9 ms。由于未达到 95% 保守门槛，它应标记为“接近可行候选”，不能写成“满足 SLO”。

另外，1,041,216-policy queue-aware search 已完成硬件 validation 和 paired confirmation；QueuePowerMin 在 long-generation arrival trace 下保持 100% 观测达标，并相对 MaxClock 取得约 5.47% J/token 改善。phase-switch gate 已完成 180/180 blocks，但没有候选同时通过能量和 SLO 晋级门槛。上述结果不能合并成一个 Thor 总收益：queue-aware 改善来自 arrival/queue/drain 场景，serial/bursty 的排序不同；phase-switch 的负结果也不能否定静态或 queue-aware Pareto。

Thor 当前最重要的结论是方法边界已经比单点 benchmark 清楚：低并发场景中 idle horizon、到达过程和重复达标率会改变候选排序；P8 场景中 batch/并发效率又会改变排序。因此 Thor 的后续工作应优先完成场景内的统一结果表和置信区间，而不是把所有场景压成一个平均收益。

## 6. 已验证的方法认识

### 6.1 功率和能效必须分开优化

降低频率通常有机会降低瞬时功率，但会延长 busy time。固定输出 token 数下，`Power`、`Energy/token` 和 `TPOT` 可能朝不同方向变化。功率优先策略可以是合理 Pareto 点，但只有在 J/token 不恶化且 SLO 达标时，才应称为能效改进。

### 6.2 decode 不是只看 GPU 频率

decode 的访存和 KV-cache 行为使 EMC、GPU、CPU governor、batch/parallel 和 runtime 调度共同影响结果。不能从单次频率相关性推断“只调 EMC”或“只调 GPU”就是最优；必须在同一 workload 上保留联合候选并记录实际频率。

### 6.3 SLO 是实验身份而不是事后筛选参数

TPOT 50 ms、75 ms、100 ms 会产生不同可行集；TTFT 还必须按 prompt bucket 分层。严格门槛下没有可行点时，应报告 infeasible 和近可行候选，不能自动放宽阈值后继续使用“满足 SLO”的表述。

### 6.4 在线控制必须先过离线 gate

phase switch、thermal feedback 和 queue-aware policy 都可能在低功率时损失能量或尾延迟。只有通过固定工作、配对能量和 SLO gate 的候选，才允许进入长时间 serving；否则只能作为诊断实验。

## 7. 场景和结果边界

- 随机权重模型适合系统测量，不代表模型质量或准确率；
- Orin 旧实验数字不能与 Thor 直接合并统计；
- quick 结果只能作为 gate 或探索性证据；
- 功率下降必须和 Energy/token、TTFT、TPOT 一起解释；
- 没有统一 workload、runtime、trace 和 SLO 时，不能声称 Thor 全面超越原生动态策略。

单用户串行、低平均负载带突发和 P8 共享容量不能合并成一个总排行榜。同一个硬件点在不同场景中的功率、能效和延迟排名可能不同。缺少某个平台或某类负载的数据时，应在平台矩阵中标为 planned，而不是用另一平台的数据替代。

## 8. 研究定位

量化和模型压缩是部署前置条件，调度和 continuous batching 是可组合的上层优化，GPU DVFS 是直接优化对象，SLO/Thermal feedback 是运行时约束，Pareto 是结果表达方式。EnergyInfra 提供位于运行时和硬件控制之间的能耗调优层，不替代量化、调度或 kernel 优化。

## 9. 后续路线

1. 按 [Thor 收口计划](thor_completion_plan_20261015.md) 完成 think-time 和 bursty 的统一结果表；
2. 对 A/B/C 三类场景使用同一指标表输出绝对值、相对 native dynamic 的变化和置信区间；
3. 完成 Q4/Q5/Q8 同源 gate，再决定是否扩大精度和 serving 搜索；
4. 完成 7B/14B matched pilot，明确模型规模是否改变频率敏感性；
5. 仅将通过 gate 的静态 Pareto 点转化为 phase-aware 和 workload-aware 在线选择器；
6. 用长时间 serving 验证温度、功率波动、频率切换开销和恢复可靠性；
7. 在真实权重 sanity check 后，再讨论随机模型结果的外推边界；
8. 同一模型、同一 runtime、同一 trace 和同一 SLO 对齐后，再汇报“超越”结论。
