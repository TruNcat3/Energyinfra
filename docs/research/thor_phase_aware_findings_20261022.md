# Thor phase-aware 与 MoE fixed-horizon 结果分析

**日期**：2026-10-22  
**平台**：Jetson AGX Thor  
**模型**：随机权重 Qwen3 MoE Q4_K_M，13B 总参数、约 1.8B active 参数  
**运行时**：隔离构建的 llama.cpp Thor runtime  
**质量边界**：仅为系统能效实验，不包含模型质量结论

## 1. 实验目的

前序 Thor 实验发现，request-active energy 容易把“更高频、更快结束”误认为
更好的 serving 能效。此次实验沿用 Orin 的 phase-aware 思路，同时使用固定
arrival horizon，分别测量 Prefill 和 Decode 的资源策略：

1. 请求 dispatch 前使用 Prefill 配置；
2. 首个 streamed output token 作为 Prefill/Decode 边界；
3. 首 token 后切换 Decode 配置；
4. 在相同的 40 秒/60 秒窗口内积分功率；
5. 将硬件切换开销纳入能耗、TTFT 和 TPOT 结果。

v1 只使用无并发周期 trace。Thor 的 GPU/EMC/CPU 控制是全局的，在并发请求
中无法把一个全局频率切换无歧义地归属于单个请求，因此并发 phase-aware 暂不
纳入本结论。

## 2. 实验协议

| 项目 | 配置 |
|---|---|
| `periodic_short` | 6 个 `128 prompt / 128 output` 请求，40 秒 |
| `periodic_long` | 6 个 `1024 prompt / 128 output` 请求，60 秒 |
| repeats | full 为 3 次 |
| SLO | `strict_20`、`balanced_30`、`edge_50`、`relaxed_100` |
| 固定窗口指标 | 总能耗、平均功率、峰值功率、能量/token |
| 延迟指标 | TTFT p95、TPOT p95、phase-switch latency p95 |
| 控制策略 | native dynamic、静态 CAP、Prefill/Decode 分阶段 CAP |

配置和复现实验入口：

- [实验配置](../../configs/experiments/thor_moe_phase_aware_serving_v1.json)
- [实验脚本](../../scripts/active/run_thor_moe_phase_aware_serving_v1.py)
- [nohup 入口](../../scripts/active/run_thor_moe_phase_aware_serving_nohup.sh)
- [实验协议](../experiments/thor_moe_phase_aware_serving_v1.md)

## 3. Full 结果

相对 `static_native_dynamic` 的变化如下，负值表示下降：

| Trace | 策略 | 固定窗口能耗 | 平均功率 | 峰值功率 | TTFT p95 | TPOT p95 | Switch p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| Long | native dynamic | 1954.57 J | 32.58 W | 80.55 W | 884.82 ms | 18.47 ms | 83.62 ms |
| Long | phase high → memory-aware | **1884.52 J (-3.6%)** | **31.41 W (-3.6%)** | **68.52 W (-14.9%)** | 1060.03 ms | 25.98 ms | 92.61 ms |
| Long | phase power guard | 1901.87 J (-2.7%) | 31.70 W (-2.7%) | **64.20 W (-20.3%)** | 1170.72 ms | 30.99 ms | 96.43 ms |
| Short | native dynamic | 1359.14 J | 33.98 W | 78.24 W | 300.49 ms | 18.12 ms | 87.66 ms |
| Short | static peak guard | **1308.52 J (-3.7%)** | **32.71 W (-3.7%)** | 71.38 W (-8.8%) | 391.71 ms | 22.67 ms | 98.35 ms |
| Short | phase max → balanced | 1329.58 J (-2.2%) | 33.24 W (-2.2%) | 72.43 W (-7.4%) | **386.00 ms** | **21.74 ms** | 87.55 ms |
| Short | phase high → memory-aware | 1337.09 J (-1.6%) | 33.43 W (-1.6%) | **66.98 W (-14.4%)** | 441.15 ms | 27.17 ms | 100.18 ms |

## 4. SLO 与 Pareto 解释

- `strict_20`：Long 和 Short 均无可行策略。
- `balanced_30`：Long 和 Short 均无可行策略，主要受 TTFT 约束影响。
- `edge_50`：Long 有 6 个可行策略，Short 有 5 个可行策略。
- `relaxed_100`：Long 和 Short 的策略均进入可行集合。

当前 Thor 结果不能简单表述为“phase-aware 全面领先”：

- Long workload 的 phase-aware 策略取得了固定窗口能耗和峰值功率的同步下降；
- Short workload 的最佳能耗点是静态 peak guard，phase-aware 没有稳定超过它；
- `phase_power_guard` 可以进一步压低峰值功率，但在 Short 的 `edge_50` TTFT
  门槛附近失去可行性；
- Pareto 集合保留多个策略，部署者可以在续航、瞬时热约束和交互延迟之间选择。

## 5. 方法结论

这次实验支持 Orin 思路在 Thor 上“方向成立、收益有限、强依赖负载”的判断。
收益受三个因素限制：

1. 硬件切换 p95 约 `87~111 ms`，频繁切换对短请求不划算；
2. Thor 频率控制是全局的，并发请求下阶段边界会互相影响；
3. 当前随机 MoE 模型和 128-token decode 尚未形成足够长的 Decode 区间来充分摊平切换成本。

因此 v1 还不能晋级为通用在线控制策略，也不能替代已有 queue-aware 或静态
Pareto 结论。

## 6. Phase-aware v2 结果：切换摊销

v2 将 Prefill 切换提前约 250 ms，并对小于 256 output tokens 的请求回退到静态
peak guard。full 结果如下：

| Trace | 最佳策略 | 相对 native dynamic 的能耗变化 | TTFT p95 | TPOT p95 | Switch p95 |
|---|---|---:|---:|---:|---:|
| `long_128` | static peak guard | -4.9% | 993.83 ms | 21.41 ms | - |
| `long_256` | phase high → memory-aware | -0.4% | 952.40 ms | 27.78 ms | 131.57 ms |
| `long_512` | phase max → balanced | -1.3% | 322.58 ms | 24.34 ms | 45.29 ms |
| `short_128` | native dynamic | 0.0% | 312.39 ms | 17.48 ms | - |

v2 的结论是：提前 Prefill 切换确实改善了长输出的 TTFT，但 Decode 阶段的
切换仍有 `45~143 ms` 开销，能耗增益只有 `0.3%~1.3%`。短请求使用 phase-aware
fallback 反而比 native dynamic 增加约 2.6% 能耗。因此当前应采用 workload-aware
hybrid，而不是对所有请求统一切换。

## 7. Workload-aware hybrid v3 结果

v3 将 v2 的输出长度策略放入同一个 `mixed_output_lengths` 固定 trace：
`short_128`、`long_128`、`long_256`、`long_512`、`short_128`、`long_256`，
总窗口 120 秒，full 三次重复。

| 策略 | 固定窗口能耗 | 平均功率 | 峰值功率 | TTFT p95 | TPOT p95 |
|---|---:|---:|---:|---:|---:|
| native dynamic | 3623.24 J | 30.19 W | 79.29 W | 722.76 ms | 23.01 ms |
| static peak guard | **3586.20 J (-1.0%)** | **29.88 W (-1.0%)** | **72.77 W (-8.2%)** | 987.56 ms | 24.04 ms |
| workload hybrid | 3642.12 J (+0.5%) | 30.35 W (+0.5%) | 77.70 W (-2.0%) | 950.49 ms | 26.20 ms |

在 `edge_50` 下混合 trace 没有策略通过；在 `relaxed_100` 下三种策略均可行。
quick 单次运行的 hybrid 能耗约低 1.2%，但 full 三重复变为比 native 高约 0.5%，
因此 quick 结果不能作为收益证据。这个结果说明 workload-aware 选择本身还不够，
全局硬件切换开销、策略切换后的尾延迟和固定窗口基线差异抵消了 phase-aware 的
局部收益。

当前 phase-aware 结论降级为：在 Thor 的全局 DVFS 控制下，长 decode 存在局部
峰值功率收益，但尚未形成稳健的总能耗或 SLO 优势，不进入在线默认策略。

## 8. 下一步 workload-aware hybrid

下一阶段不再继续增加 phase-aware 频点，而转向运行时和 arrival-aware 控制：

1. 将静态 peak guard 作为当前低并发默认候选；
2. 把到达间隔、队列深度和 batch/ubatch 状态纳入选择器，而不是只按 output length 切换；
3. 在 shared serving 中验证 queue-aware 静态策略，优先于 per-request 全局频率切换；
4. 只有当 runtime 能降低切换延迟或提供 phase-local 控制后，再重新评估 phase-aware；
5. 不把当前串行负结果外推为所有 GPU 或所有 runtime 的结论。
