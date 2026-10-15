# EnergyInfra

## 面向边端 GPU 的多目标能耗调优框架

EnergyInfra 面向 Jetson Orin、Jetson AGX Thor 等边端 GPU 上的 LLM 推理，围绕以下目标构建可复现实验和在线调优链路：

- 在满足 TTFT、TPOT、吞吐和温度约束的前提下降低平均功率；
- 降低固定工作量的每 token 能量；
- 用 Pareto 前沿表达功率、能效和延迟之间的取舍，而不是输出一个脱离场景的“全局最优点”；
- 将离线 profiling 结果转化为可部署的 GPU/EMC/CPU cap、phase-aware 和 SLO-aware 控制策略。

## 当前研究主线

```text
固定模型/运行时/硬件配置
          ↓
资源筛选与静态 profiling
          ↓
固定 trace + 重复 serving 测量
          ↓
TTFT / TPOT / Power / J per token / Thermal 数据
          ↓
SLO 过滤 + 多目标 Pareto 分析
          ↓
按用户偏好选择功率优先、能效优先或延迟优先策略
          ↓
在线 cap / phase / thermal feedback 控制
```

当前主平台是 Thor，运行时为隔离构建的 llama.cpp，主模型为随机权重 Qwen2.5 7B Q4_K_M。随机权重只用于控制变量和系统性能实验，不用于模型质量结论。

## 当前 Thor 证据

| 场景 | 当前状态 | 解释 |
|---|---|---|
| P8 共享容量 | 已完成 | c1/c3 在 TPOT 100 ms 约束下可支持到 64 tok/s，存在明确功率/能效收益 |
| 低并发突发负载 | 已完成 | 90 个测试点完成；严格的全重复、全 prompt bucket 门槛下暂无最终可行点 |
| 单用户串行 think-time | 已完成运行 | 请求间隔固定为 30 秒；用于判断真正边端交互场景，结果需与突发场景分开解释 |
| 静态硬件候选 | 已完成初版 | 已覆盖动态、最高频、低 EMC、c1/c3 等候选配置 |
| 运行时和采集 | 已完成 | 硬件恢复、tegrastats、固定工作完整性和 telemetry gap 防护已接入 |

低并发突发场景中，目前最接近严格可行的是 `edge_c3_p2 @ rate_012`：TPOT 100 ms 下总体达标率约 90%，三次重复为 `[100%, 80%, 80%]`，平均功率约 40.5 W，TPOT P95 约 30.9 ms。它是候选点，不应标记为已经满足 95% SLO。

## 文档入口

- [项目研究总报告](docs/research/edge_gpu_multi_objective_framework.md)
- [Thor 当前研究状态](docs/research/thor_current_status_20261015.md)
- [代码与实验组织方式](docs/architecture/code_and_experiment_structure.md)
- [Thor 复现实验指南](docs/operations/thor_reproduction.md)
- [串行边端协议说明](docs/experiments/thor_edge_interactive_think_v1.md)
- [配置文件组织](configs/README.md)
- [脚本组织](scripts/README.md)
- [历史实验文档索引](docs/实验文档/README.md)

## 快速开始

```bash
cd /home/wt/energyinfra/Energyinfra-main
source .venv-thor/bin/activate

# 普通用户：生成固定 trace 和协议文件
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full prepare

# root：运行硬件实验
sudo -E bash scripts/active/run_thor_edge_interactive_think_nohup.sh full run

# 普通用户：生成报告
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full report
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full analyze
```

硬件控制和 `/sys`、DebugFS 写入需要 root。准备、分析和报告阶段尽量使用普通用户；只有实际 serving run 使用 `sudo -E`。

## 结果解释边界

EnergyInfra 当前报告严格区分平均功率、每 token 能量、TTFT/TPOT、SLO 达标率和 Pareto 前沿。Orin 早期结果和旧 Phase 文档仍保留，但属于历史基线，不能与 Thor 的随机权重、llama.cpp 版本和 serving trace 结果直接合并统计。

大型模型、原始 telemetry 日志和中间构建产物不应提交到代码仓库；发布结果时应提交配置、摘要、哈希和可复现命令。
