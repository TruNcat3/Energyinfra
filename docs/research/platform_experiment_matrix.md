# 平台与实验矩阵

EnergyInfra 使用统一的实验轨道覆盖不同边端 GPU。状态定义：available 表示已有可追溯数据，partial 表示已有部分平台/模型/负载结果，planned 表示协议已定义但数据尚未完成，invalidated 表示历史数据不能作为有效证据。

| 实验轨道 | 统一问题 | Orin | Thor | 备注 |
|---|---|---:|---:|---|
| 平台与环境识别 | OS、JetPack、频率接口、温度和功率路径 | available | available | 使用各自 platform profile |
| 单 knob profiling | GPU/CPU/EMC 对性能和能效的影响 | available | partial | Thor EMC exact lock 受接口限制 |
| GPU×EMC 静态表 | 构建 rate table 和 Pareto front | available | partial | Thor 已有候选 setpoint，需继续扩展 |
| Workload-aware selector | 按输入/输出/phase 选择配置 | available | partial | Thor static Pareto 已跑，在线闭环待补 |
| P1/P2 edge serving | 低并发、低 batch serving | partial | available | Thor 已有 serial 和 bursty 方案 |
| P8 shared capacity | 并发 capacity 和吞吐能效 | available | available | 横向比较前需统一 trace/SLO |
| Prompt-aware SLO | TTFT 按 prompt bucket 分层 | partial | available | 统一阈值和置信区间仍需整理 |
| Phase-aware DVFS | prefill/decode 分阶段控制 | available | partial | Thor 已有 gate/diagnostic，长期 serving planned |
| Thermal feedback | 温度、功率和 SLO 闭环 | available | partial | Thor 真实长期验证 planned |
| 长时间 serving | 热稳定性、switch overhead、恢复 | available | partial | Thor 需要同定义 trace |
| 模型规模 | 7B/8B/14B scaling | available | partial | Thor 14B 已准备，完整对比 planned |
| 量化精度 | Q4/Q5/Q8 能效空间 | partial | partial | 需统一模型和上下文设置 |
| 真实权重 sanity check | 排除随机模型特定行为 | available | planned | Thor 当前随机权重只作系统控制变量 |
| 跨平台 Pareto | 同协议下比较平台特征 | partial | partial | 先完成相同 workload/SLO 对齐 |

没有数据的单元保留为 planned，不能使用另一平台数据填充。
