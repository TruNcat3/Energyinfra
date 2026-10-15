# 实验数据目录与证据等级

仓库提交可审计的摘要、报告、配置和图表；原始 telemetry、模型、build 和长时间日志保留在实验机或外部存储。每个结果必须能追溯到平台、runtime、模型、trace、硬件配置和 SLO。

## Orin 数据

| 数据目录 | 内容 | 状态 |
|---|---|---|
| data/energy_profiling/ | GPU/EMC profiling 和模型对比 | available |
| data/rate_tables/ | lock/cap rate table 和 DVFS 规则 | available |
| data/e2e_benchmark/ | E2E cap selector/phase serving | available |
| data/oracle_gap_analysis/ | Oracle gap | available |
| data/multi_obj_eval/ | MDR/JIR/HV/Pareto 评估 | available |
| figures/ | Orin 结果图表和报告 | available |

## Thor 数据

Thor 原始协议结果位于实验机的 data/protocol_v2/。仓库主要提交配置、脚本、协议文档和结果解释，避免提交大量 telemetry 和模型文件。

| 数据类别 | 结果位置 | 状态 |
|---|---|---|
| Resource screen | data/protocol_v2/thor_joint_resource_power_v1* | available |
| P8 capacity | data/protocol_v2/thor_slo_capacity_frontier_v1* | available |
| Bursty low-concurrency | data/protocol_v2/thor_edge_low_concurrency_v1* | available |
| Serial think-time | data/protocol_v2/thor_edge_interactive_think_v1* | available |
| Queue-aware Pareto hardware validation | data/protocol_v2/thor_queue_aware_{pareto,hardware_validation,confirmation}_v1* | available |
| Precision/model scale | data/protocol_v2/precision_v1, random_scale_v1 | partial |
| Phase-switch gate/validation | data/protocol_v2/thor_phase_{switch_gate,validation}_full_v1* | partial |
| Long thermal serving | Thor protocol output directories | planned |

缺少某个平台或某类实验的数据时，在平台与实验矩阵中标记为 planned，并保留配置、入口脚本和目标指标，禁止用其他平台的数据替代。`partial` 表示已有完整或独立 pilot 数据，但仍缺少跨 runtime、质量或长期运行边界；它不能等同于全面完成。

## Thor 收口数据要求

Thor 每个场景的正式摘要应至少能回答以下问题：

1. 在相同 frozen trace 和 SLO 下，候选相对 native dynamic 的 Power 和 Energy/token 分别变化多少？
2. 这个变化是否在每个 repeat、prompt bucket 和 telemetry gate 下成立？
3. 候选属于 power、energy、latency 还是综合 Pareto 前沿，是否牺牲了另一个目标？
4. 该结果是单用户、bursty edge 还是 P8 shared capacity，能否迁移到其他场景？

因此，`RESULTS.md` 只有平均功率而没有 J/token、SLO 规则、重复达标率或 trace/hash 时，只能作为工程日志，不能作为最终研究证据。Thor 当前优先补齐 serial/bursty 的统一摘要，再补精度、模型规模和长期热稳定性。
