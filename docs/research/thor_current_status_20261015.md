# Thor 当前研究状态

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
| 多目标分析 | 完成初版 | SLO 过滤、prompt bucket、Pareto 候选 |
| P8 容量 | 完成 | 高并发共享容量证据 |
| bursty 低并发 | 完成 | 90 个 full trials |
| serial think-time | 已运行 | 30 秒间隔，等待整理为最终报告 |

## 研究认识

1. Thor 的收益不是在所有场景下同方向扩大；
2. 平均功率、每 token 能量和 TPOT 存在实际冲突；
3. P8 容量场景和低并发边端场景必须独立报告；
4. 严格 SLO 的失败可能来自突发 trace 和统计门槛，不应直接归因于硬件频率；
5. Pareto 前沿比单一“最优频率”更符合部署者的不同偏好。

## 证据等级

| 等级 | 含义 | Thor 当前内容 |
|---|---|---|
| L0 | 工程可运行 | 已达到 |
| L1 | 单点/quick 可重复 | 已达到，但只能作探索性证据 |
| L2 | 同 trace、多重复、可恢复 | bursty full 已达到 |
| L3 | 跨场景、跨 SLO、Pareto 可解释 | 正在完成 |
| L4 | 跨模型/精度/真实模型质量约束 | 尚未完成 |

## 下一阶段出口条件

- A/B/C 三类 workload 均有 native dynamic、max clock 和至少两个 EnergyInfra 候选；
- 每个场景至少 3 次重复，并报告原始达标率和置信区间；
- Power 与 Energy/token 同时报告；
- SLO 解释包含 TTFT、TPOT、prompt bucket 和重复规则；
- 所有结果能由配置文件、trace hash、runtime manifest 和命令复现。
