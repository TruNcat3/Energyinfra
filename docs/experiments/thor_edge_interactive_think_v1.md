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
