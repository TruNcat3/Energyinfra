# 代码与实验组织方式

## 1. 组织原则

当前仓库包含早期 Orin 原型、Thor 适配、运行时构建和大量实验产物。为了不破坏历史可复现性，第一阶段不直接移动现有脚本，而采用“稳定路径 + 清晰命名 + 文档分层”的渐进式整理方式。

- `configs/` 定义实验身份，不在 shell 脚本中硬编码关键实验参数；
- `scripts/active/` 只放当前仍可复现的编排、准备、运行和分析脚本；
- `scripts/_legacy/` 放不再作为主线入口的旧脚本；
- `src/` 放平台无关的控制器、指标和选择器；
- `data/protocol_v2/` 的 trace、manifest 和结果属于实验产物，不直接手工修改；
- `docs/experiments/` 记录协议，`docs/research/` 记录跨实验结论。

## 2. 目录职责

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
