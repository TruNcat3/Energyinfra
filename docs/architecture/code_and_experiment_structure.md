# 代码与实验组织方式

## 1. 组织原则

当前仓库包含 Orin 基线、Thor 适配、运行时构建和大量实验产物。两者属于同一框架的不同平台实例。为了不破坏历史可复现性，第一阶段不直接移动现有脚本，而采用“公共算法 + 平台 adapter + 稳定路径 + 清晰命名”的渐进式整理方式。

- `configs/` 定义实验身份，不在 shell 脚本中硬编码关键实验参数；
- `scripts/active/` 只放当前仍可复现的编排、准备、运行和分析脚本；
- `scripts/_legacy/` 放不再作为主线入口的旧脚本；
- `src/` 放平台无关的控制器、指标和选择器；
- `data/protocol_v2/` 的 trace、manifest 和结果属于实验产物，不直接手工修改；
- `docs/experiments/` 记录协议，`docs/research/` 记录跨实验结论。

## 2. 平台适配

- Orin：保留既有 platform.yaml、frequencies.yaml、src/ 和 rate-table 流程；
- Thor：使用 platform_thor.yaml、platform_profiles.yaml、Thor protocol 和 run_thor_* 入口；
- 公共分析：使用相同的 Power、Energy/token、TTFT、TPOT、Temperature 和 SLO 字段；
- 平台特殊能力：只在 adapter/config 层记录，例如 Thor EMC setpoint 限制或 Orin power mode。

## 3. 目录职责

| 目录 | 责任 | 不应放入 |
|---|---|---|
| `src/controller/` | 频率、cap、Pareto、thermal 控制 | 单次实验专用解析 |
| `src/benchmark/` | 推理 runner 和基准接口 | 硬件实验 shell 编排 |
| `src/metrics/` | telemetry 和指标解析 | 具体 SLO 结论 |
| `src/ratetable/` | profiling 聚合和 rate table | 原始日志 |
| `scripts/active/` | prepare/run/report/analyze 入口 | 大型模型文件 |
| `configs/experiments/` | 可复现实验配置 | 临时命令行覆盖 |
| `configs/models/` | 模型 provenance | 下载缓存 |
| `data/protocol_v2/` | frozen trace、manifest、结果 | 手工编辑后的结果 |
| `docs/research/` | 主线研究叙事 | 未核验的外部数字 |
| `docs/experiments/` | 单个实验协议 | 跨实验排名 |

## 3. 实验生命周期

```text
prepare → quick gate → full run → report → analyze → archive
```

- `prepare`：检查 runtime/model/config，生成 frozen trace 和 manifest；
- `quick gate`：验证 server、SSE、功率采集和硬件恢复；
- `full run`：root 下运行，固定 trace，不改变实验身份；
- `report`：输出逐 trial 和聚合摘要；
- `analyze`：离线计算 SLO、Pareto 和对 baseline 的差异；
- `archive`：保存 config hash、trace hash、runtime manifest 和最终报告。

## 3.1 统一结果契约

每个可发布的 serving 结果至少包含以下四组字段。字段缺失时只能标记为 partial，不能通过脚本默认值补齐：

| 组 | 必填内容 |
|---|---|
| 身份 | platform、OS/JetPack、runtime manifest、model hash、config hash、trace hash |
| 工作量 | prompt/output tokens、arrival mode、batch/ubatch、parallel、repeat、固定 horizon |
| 性能 | completed、TTFT P50/P95、TPOT P50/P95、throughput、SLO attainment、失败原因 |
| 能源 | average/max power、total energy、J/token、temperature、telemetry sample count/gap |

聚合报告还必须同时提供：

- 相对 `native_dynamic` 和 `max_clock` 的绝对差值与百分比；
- 每个 repeat 和 prompt bucket 的达标率；
- power-minimum、energy-minimum、latency-minimum、normalized-knee 和完整非支配集合；
- 95% confidence interval 或明确说明样本不足；
- hardware restore、fixed-work integrity 和 telemetry gate 的结果。

`Power`、`Energy/token`、`TTFT`、`TPOT` 和 `Temperature` 不应在报告生成阶段隐式加权成单一分数。用户偏好不同，就应从同一 Pareto 集合选择不同点。

## 4. 命名约定

```text
<platform>_<scenario>_<purpose>_v<N>.json
run_<platform>_<scenario>_<purpose>_nohup.sh
analyze_<platform>_<metric>_v<N>.py
data/protocol_v2/<protocol>/{quick,full}/
```

`quick` 只能用于 gate 和探索，不能直接写成最终收益；`full` 必须注明 repeats、SLO 和 workload 类型。

## 5. PR 变更边界

一个实验 PR 应优先包含配置、prepare/run/analyze 入口、协议文档和小型摘要或 manifest。不应把模型、完整 telemetry 日志、编译目录和 `.venv` 放进 PR；大型数据应通过 provenance、哈希和外部存储引用。
