# 脚本组织

`scripts/active/` 是当前可复现实验入口，按职责分为：

- `prepare_*`：模型、runtime、trace 和协议准备；
- `run_*_nohup.sh`：长时间实验 wrapper，负责权限、日志和 graceful stop；
- `run_*_v1.py`：硬件/serving 实验主程序；
- `analyze_*_v1.py`：离线分析，默认不触碰硬件；
- `compare_*`、`explore_*`：结果比较和探索性分析；
- `setup_*`、`verify_*`：环境和运行时检查。

每个新实验至少提供一个配置、prepare/protocol freeze 入口、带日志和 PID 的 run wrapper、report/analyze 入口以及一份 `docs/experiments/` 协议说明。硬件实验脚本必须具备恢复路径；分析脚本必须默认不修改硬件频率。

`scripts/_legacy/` 仅保留历史入口。新实验不要从 legacy 脚本复制未审计的硬件控制逻辑，应复用当前 `run_energyinfra_v2.py`、Thor protocol 和 runtime manifest 机制。
