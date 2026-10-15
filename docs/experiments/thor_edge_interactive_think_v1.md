# Thor serial interactive think-time v1

## 目的

把低平均负载和短时突发从真正的单用户边端场景中分离出来，测量请求之间存在明显 think-time 时的 idle energy、尾延迟和硬件策略差异。

## 协议

- 配置：`configs/experiments/thor_edge_interactive_think_v1.json`；
- SLO：`configs/experiments/thor_edge_interactive_think_full_slo_v1.json`；
- wrapper：`scripts/active/run_thor_edge_interactive_think_nohup.sh`；
- 请求数：10；每个请求间隔：30 秒；full repeats：3；
- workload mix：128/128、512/128、128/512、1024/512；
- 策略：native dynamic、max clock、EnergyInfra c1、低 EMC、c3，分别测试 P1/P2；
- 能耗范围：包含请求之间的固定 horizon idle energy。

## 解释边界

该协议不是吞吐容量测试，也不是 bursty workload。它回答的问题是：在单用户请求之间有较长空闲期时，降低 GPU/EMC/CPU 频率是否能降低平台功率或每 token 能量，以及是否造成可接受的 TTFT/TPOT 损失。

## 运行

```bash
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full prepare
sudo -E bash scripts/active/run_thor_edge_interactive_think_nohup.sh full run
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full report
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full analyze
```

最终报告必须与低并发 bursty 协议和 P8 capacity 报告分开。

## 收口判据

该协议的正式结果不是“30 秒间隔下平均功率最低的点”，而是同一固定 horizon 下的多目标集合。报告必须按 workload 和 repeat 输出：

- 请求级 TTFT/TPOT P95、完成数和 SLO 达标率；
- 包含 idle horizon 的总能量、平均功率和 J/token；
- 相对 native dynamic、max clock 的绝对值和百分比；
- power-minimum、energy-minimum、latency-minimum 和 knee 候选。

如果 idle energy 让低频点功率下降但 J/token 上升，应分别标记为 power-oriented Pareto 点和 energy regression，不能合并为“续航提升”。
