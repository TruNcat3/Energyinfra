# 配置文件组织

| 路径 | 作用 |
|---|---|
| `platform*.yaml` | 平台和频率接口描述 |
| `power_modes.yaml` | 电源模式和硬件策略 |
| `models/` | 模型路径、provenance 和精度 |
| `experiments/` | 可复现实验身份、trace、策略和输出目录 |
| `slo*.yaml/json` | SLO、重复和达标规则 |

实验配置使用 `<platform>_<scenario>_<purpose>_v<N>.json`。配置一旦用于生成 frozen trace，不应原地修改；协议变化应创建新版本和新 output directory。

Thor serving 配置至少需要明确 hardware config、runtime manifest、model path、workload mix、arrival mode、batch、ubatch、parallel、repeats、telemetry 规则和 SLO 规则。

配置是实验身份的一部分。不要只在 shell 命令行中覆盖关键频率或 SLO，否则结果无法审计。
