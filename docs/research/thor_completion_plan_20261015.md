# Thor 实验收口计划

## 1. 目标

Thor 适配已经能够完成硬件控制、固定工作 serving、功率/温度采集和离线 Pareto 分析。下一阶段的目标不是继续无约束增加测试点，而是把已有证据收敛为可审计的多目标结论，并补齐会改变结论解释的关键维度：

- 单用户低并发、低平均负载突发和 P8 共享容量分别报告；
- Power、Energy/token、TTFT、TPOT、达标率和温度使用同一结果契约；
- 同一 workload、runtime、模型、trace 和 SLO 下比较 baseline 与 EnergyInfra；
- 没有实测数据的组合明确标记为 `planned`，不从 Orin 或其他场景借用数值；
- 先完成可比性和统计门槛，再扩大模型、精度和在线控制实验。

## 2. 已经可以写入研究结论的内容

| 结论 | 证据边界 | 状态 |
|---|---|---|
| Pareto 比单一最优频率更适合部署决策 | 功率、J/token、TTFT、TPOT 存在冲突 | established |
| Thor 必须联合观察 GPU、EMC 和 CPU | 控制接口、候选 profile 和 serving 已打通 | established |
| 低并发和 P8 不能共用一个排行榜 | arrival、排队、idle horizon 和 batch 行为不同 | established |
| 严格 SLO 可能因 trace/重复门槛导致无可行点 | bursty full 的全重复过滤结果 | established |
| 随机权重适合系统控制变量，不支持质量结论 | 当前 Thor 主模型 provenance | established |
| Thor 已经全面优于 native dynamic | 尚没有跨场景、同 SLO、同模型的完整证据 | not established |

## 3. 收口轨道

### 3.1 轨道 A：低并发边端最终表

输入为串行 think-time 和 bursty low-concurrency 两个 frozen-trace campaign。先离线生成统一表，再判断是否需要补跑硬件。

必须输出：

- 每个 workload/profile、strategy、repeat 的 Power、总能量、J/token、TTFT P95、TPOT P95、温度和达标率；
- `TPOT <= 50/75/100 ms` 与 prompt-aware TTFT schedule 的交叉结果；
- 每个 repeat 的达标率，而不是只给总体平均值；
- 相对 native dynamic 和 max clock 的绝对差值及百分比；
- power-minimum、energy-minimum、latency-minimum、knee 和非支配候选。

出口条件：至少 3 次重复、每个场景保留相同 trace、所有 trial 通过 fixed-work/telemetry gate；如果没有候选达到 95% 约束，报告“无可行点”并保留近可行候选。

### 3.2 轨道 B：量化和模型规模敏感性

在同一 runtime、上下文、硬件候选和 trace 下比较 Q4_K_M、Q5_K_M、Q8_0；随后比较 7B/14B。该轨道只回答频率敏感性、功率、J/token 和延迟是否随模型表示变化，不回答模型质量。

出口条件：同源模型 provenance、至少 5 次独立 validation repeat、每种格式/规模使用相同 SLO；如果结果差异小于重复噪声，停止扩展 serving 组合，不把微小差异解释为优化收益。

### 3.3 轨道 C：phase-aware 和 thermal-aware 在线控制

先完成 phase-switch gate，再将通过 gate 的候选接入 frozen-trace serving；最后进行长时间热稳定性验证。phase switch 只有同时满足 SLO 和配对能量置信区间门槛才能晋级。

出口条件：记录切换延迟、切换前后频率实际值、Power/J/token 变化、温度轨迹、恢复状态和异常 trial；没有通过 gate 的候选只作为诊断结果。

### 3.4 轨道 D：真实权重 sanity check

选择一个可合法使用的预训练 GGUF，在不改变 runtime、trace、硬件候选和指标的前提下复核 Q4 主结论。该实验只用于排除随机权重特有行为，不与随机模型合并成一个统计样本。

出口条件：记录模型许可证、哈希、量化来源和质量评测最小集；如果真实模型与随机模型的频率排序不同，应分别维护结论。

## 4. 推荐执行顺序

```text
A: serial/bursty offline summary
  -> A: same-SLO Pareto and baseline comparison
  -> B: Q4/Q5/Q8 same-source gate
  -> B: 7B/14B matched pilot
  -> C: phase gate and long thermal serving
  -> D: pretrained-model sanity check
```

不能跳过 A 直接用 P8 结果代表边端，也不能在 B 的 quick 结果之后直接声称量化或模型规模带来收益。

## 5. 当前运行入口

准备、运行和报告阶段必须使用与协议页一致的目录；root 只用于实际硬件 run：

```bash
cd /home/wt/energyinfra/Energyinfra-main

# A: single-user think-time
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full prepare
sudo -E bash scripts/active/run_thor_edge_interactive_think_nohup.sh full run
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full report
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full analyze

# A: bursty low-concurrency
bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full prepare
sudo -E bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full run
bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full report
bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full analyze

# B: same-source precision gate
sudo -E bash scripts/active/run_thor_precision_nohup.sh prepare
sudo -E bash scripts/active/run_thor_precision_nohup.sh gate

# C: phase gate
sudo -E bash scripts/active/run_thor_phase_switch_gate_nohup.sh run
```

如果历史目录由 root 创建，使用新的 run directory 或先只修复目标目录 owner；不要对整个仓库执行无差别权限修改。每个 campaign 完成后先检查 `RESULTS.md`、重复数、失败数、hash 和 `hardware_restored`，再进入下一阶段。

## 6. 不应提前宣称的结果

- 不把平均功率下降直接称为能效提升；必须同时检查 J/token；
- 不把单次 quick、单个 repeat 或单个 prompt bucket 称为满足 SLO；
- 不把 P8 shared capacity 的吞吐结论迁移到单用户边端；
- 不把 Thor 的部分候选与 Orin 的历史 Pareto 数值直接合并；
- 不把随机模型的系统结果描述成准确率、质量或真实电池续航结论。
