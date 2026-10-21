# EnergyInfra 文档入口

EnergyInfra 的统一主题是面向边端 GPU 的多目标能效调优框架。Orin 和 Thor 共享方法、指标和报告模板，平台特有内容通过 adapter 和 profile 隔离。

## 当前主线

- [面向边端 GPU 的多目标能耗调优框架](research/edge_gpu_multi_objective_framework.md)
- [Thor 实验收口计划](research/thor_completion_plan_20261015.md)
- [平台与实验矩阵](research/platform_experiment_matrix.md)
- [Thor 当前研究状态](research/thor_current_status_20261015.md)
- [Thor phase-aware 结果分析](research/thor_phase_aware_findings_20261022.md)
- [数据目录与证据等级](../data/README.md)
- [代码与实验组织方式](architecture/code_and_experiment_structure.md)
- [Thor 复现实验指南](operations/thor_reproduction.md)

## 当前实验协议

- [串行边端 think-time 协议](experiments/thor_edge_interactive_think_v1.md)
- [Stage 1B 持久 serving](experiments/thor_joint_serving_stage1b_v2.md)
- [Prompt-aware SLO](experiments/thor_prompt_aware_slo_v1.md)
- [低并发 bursty trace](experiments/thor_q4_bursty_fixed_trace_v1.md)
- [Queue-aware Pareto 硬件验证](experiments/thor_queue_aware_hardware_validation_v1.md)
- [Queue-aware paired confirmation](experiments/thor_queue_aware_confirmation_v1.md)
- [Phase validation 与静态候选诊断](experiments/thor_phase_validation_v1.md)
- [Thor unified evidence aggregation](experiments/thor_unified_evidence_v1.md)
- [Thor targeted edge confirmation](experiments/thor_edge_confirmation_v1.md)
- [MoE phase-aware serving](experiments/thor_moe_phase_aware_serving_v1.md)

## 文档分层

| 目录 | 用途 |
|---|---|
| `docs/research/` | 研究主线、结果综合和技术报告 |
| `docs/architecture/` | 代码、配置、数据和实验组织规范 |
| `docs/operations/` | 环境部署、运行、权限和故障恢复 |
| `docs/experiments/` | 单个协议的设计、命令和结果解释 |
| `docs/实验文档/` | 原有 Orin/Phase 历史报告和中文实验记录 |
| `docs/开发文档/` | 开发过程记录；当前状态以 `docs/research/` 为准 |

历史文档不删除，但引用时必须标注平台、运行时、模型、数据有效性和是否可与 Thor 结果比较。
