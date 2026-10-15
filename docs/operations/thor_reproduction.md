# Thor 复现实验指南

## 1. 权限边界

准备、报告和离线分析使用普通用户；真实硬件 run 使用 root。这样可以避免把整个仓库和结果目录写成 root-owned。

```bash
cd /home/wt/energyinfra/Energyinfra-main
source .venv-thor/bin/activate
```

如果历史 sudo 实验导致新目录没有写权限，应先处理目标目录 owner，不要对整个仓库执行无差别 `chmod -R 777`。

## 2. 串行边端协议

```bash
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full prepare
sudo -E bash scripts/active/run_thor_edge_interactive_think_nohup.sh full run
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full report
bash scripts/active/run_thor_edge_interactive_think_nohup.sh full analyze
```

协议固定 10 个 workload 请求，每个请求间隔 30 秒，使用同一组 P1/P2 策略，full 为 3 次重复。它用于单用户低并发，不代表 bursty 或 P8 capacity。

## 3. bursty 低并发协议

```bash
bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full prepare
sudo -E bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full run
bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full report
bash scripts/active/run_thor_edge_low_concurrency_nohup.sh full analyze
```

该协议包含低平均输出速率和短时突发，结果应标记为 bursty edge，而不是 pure single-user。

## 4. 日志和恢复

启动命令会输出 PID、日志和 graceful stop 命令：

```bash
tail -f logs/<run-log>.log
sudo kill -TERM <pid>
```

优先使用脚本输出的 supervisor PID。中断后必须确认硬件已经恢复 dynamic 状态，再开始下一组测试。

## 5. 常见故障

| 现象 | 含义 | 处理 |
|---|---|---|
| `PermissionError` | 结果目录由 root 创建 | 修复目标目录 owner，或使用新的用户可写 run-dir |
| `fixed-work integrity gate failed` | stream 事件或 stop frame 不完整 | 不纳入结果，检查 runtime/server 日志 |
| `Telemetry sample gap exceeds protocol limit` | tegrastats 采样间断 | 检查后台进程、采样间隔和负载；不要静默删除该 trial |
| `KeyError` in report | config 与 runner schema 不一致 | 创建新 protocol 目录并补齐配置字段 |
| `lock failed` on EMC | 当前 Thor EMC 接口不接受目标值 | 使用已验证的 EMC 候选，记录 actual/configured 值 |

## 6. 发布前检查

报告发布前应同时保存 config、runtime manifest、trace hash、trial 数量、重复次数、失败数、Power、Energy/token、TTFT、TPOT、SLO 规则和 hardware restored 记录。
