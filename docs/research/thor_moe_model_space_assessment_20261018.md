# Thor MoE 模型空间评估与实验计划

## 结论先行

MoE 值得作为下一条研究分支，但不能把“参数总量更大”直接等同于“更容易获得能效收益”。对当前 Thor 实验，最可能扩大优化空间的不是单纯把 Qwen2.5-7B dense 换成更大的模型，而是同时引入：

- 专家权重的 GPU、CPU 和分层放置；
- active experts/token、专家负载不均衡和热点专家缓存；
- prefill 与 decode 不同的资源敏感性；
- batch、parallel、arrival trace 与专家路由之间的交互。

建议采用“dense 结果作为控制基线，MoE 作为 placement-aware 扩展”的论文路线，而不是用 MoE 替换现有结论。

## 为什么当前 Thor 收益较小

### 1. 模型规模没有制造足够的系统压力

当前主实验使用 7B 级随机 Q4 dense 模型。它可以完整驻留 Thor，实验也以单用户、低并发为主。此时 GPU/EMC/CPU 调节不会显著改变权重搬运路径，调频收益自然受到限制。

Jetson AGX Thor 具有 128 GB LPDDR5X 和 273 GB/s 内存带宽，GPU 最大频率约 1.57 GHz，平台功耗范围为 40--130 W。较大的统一内存容量降低了“小模型被迫分层放置”的压力；如果模型仍然全部驻留 GPU，MoE 的总参数量未必转化为可观的控制空间。[NVIDIA Thor platform overview](https://developer.nvidia.com/blog/introducing-nvidia-jetson-thor-the-ultimate-platform-for-physical-ai)

### 2. dense 模型没有路由和专家局部性

当前模型每个 token 都执行相同 dense FFN 路径。它没有专家选择、路由不均衡、专家热集合或 CPU/GPU 专家搬运，因此无法暴露 MoE 最有价值的系统变量。

Mixtral 8x7B 是一个较清晰的实验候选：论文报告其总参数约 47B，但每个 token 激活约 13B 参数。这个“总容量大、active compute 较小”的结构比单纯扩大 dense 模型更适合研究权重驻留与带宽权衡。[Mixtral of Experts](https://arxiv.org/abs/2401.04088)

DeepSeekMoE 进一步强调专家 specialization 和 shared experts，这意味着专家访问分布可能具有更强的结构，但也要求记录路由统计，不能只看最终 tok/s。[DeepSeekMoE](https://arxiv.org/abs/2401.06066)

### 3. 当前控制维度仍主要是频率点

Thor 当前已经探索了 GPU、EMC、CPU、batch、ubatch 和 parallel，但 MoE 需要新增“专家放置”这个一级决策变量。否则实验仍然是在相同的执行图上搜索频率，无法回答专家搬运和缓存是否带来新的 Pareto 前沿。

### 4. 当前 SLO 和负载强调单用户 serving

这符合边端低 batch 使用方式，但也意味着队列和并发不足以放大路由不均衡。MoE 至少需要同时保留 serial think-time、bursty low-concurrency 和 small parallel 三种负载；P8 高并发仍应独立报告，不能替代边端结论。

## KTransformers、Strata 与我们的机会

KTransformers 已经把 CPU/GPU 专家放置、GPU expert mask、动态 expert update 和异步 CPU-GPU 调度作为核心能力，目标是让大型 MoE 在受限本地设备上运行。[KTransformers expert scheduling tutorial](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md)

Strata 的定位更窄也更激进：它面向 RAM 受限设备，把 GPU、RAM 和 NVMe 都纳入 MoE 稳态路径，并使用专家预测和 expert-centric prefill。[Strata research engine](https://github.com/Cintu07/strata)

这些工作说明专家放置、预取、缓存和调度确实会产生复杂妥协，但它们不等同于我们的研究问题：

| 方向 | KTransformers/Strata 重点 | EnergyInfra Thor 重点 |
|---|---|---|
| 主要目标 | 可运行性、吞吐、延迟、模型容量 | Power、Energy/token、TTFT、TPOT、温度和 SLO 的 Pareto |
| 控制变量 | expert placement、prefetch、cache、CPU/GPU overlap | placement 与 GPU/EMC/CPU DVFS、batch、arrival mode 联合选择 |
| 硬件假设 | 工作站 GPU、CPU、RAM，部分方案含 NVMe | 集成 Blackwell GPU、共享 LPDDR5X、边端功耗和热约束 |
| 证据形态 | runtime/kernel benchmark | 固定 trace、多重复、硬件恢复、SLO gate、能耗测量 |

因此不建议复刻它们的推理引擎。更有价值的方向是把它们的 placement 和 scheduling trade-off 作为输入变量，研究在 Thor 上何时值得牺牲 TTFT、CPU 占用或功率来换 Energy/token。这个定位也避免与它们单纯竞争 tok/s。

## 推荐实验路线

### M0：模型和 runtime gate

- 首选 Mixtral 8x7B Q4_K_M，记录总参数、active experts、量化来源、模型哈希和许可证。
- 备用 Qwen3-MoE，仅在当前锁定的 llama.cpp commit 能稳定加载、输出固定 token 数且 provenance 可复现时使用。当前 llama.cpp 源码已包含 Qwen3 MoE 架构分支，但仍需在 Thor 的 pinned build 上单独验证。[llama.cpp model architecture](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)
- 不使用随机权重作为 MoE 主结论。随机权重可以用于 runtime gate，但随机 router 不能代表真实专家使用分布。
- 使用一个 dense matched reference 作为对照，至少按 active parameter scale、量化格式、上下文和 workload 保持可解释。

### M1：专家放置矩阵

固定同一模型和同一 trace，仅改变放置：

| Placement | 目的 |
|---|---|
| all-GPU | MoE 计算基线，隔离路由但不引入搬运 |
| all CPU-MoE | 放大系统内存、EMC 和 CPU 影响 |
| partial CPU-MoE | 探索 GPU/CPU 专家放置 Pareto |
| hot-set/cache | 后续扩展，不作为第一版正式 baseline |

每种 placement 至少比较 native dynamic、MaxClock、QueuePowerMin 和一个低 EMC 点。频率搜索不需要恢复完整 4320 点，而应先用 M1 的 placement sensitivity 证明是否存在值得扩大搜索的维度。

llama.cpp server 当前提供 `--cpu-moe` 和 `--n-cpu-moe`，可以作为第一版 placement baseline；缓存和专家预测功能不应在第一版直接引入，因为它们会同时改变 runtime 和控制策略。[llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)

### M2：路由和负载观测

运行时需要新增或导出：

- 每层 active expert 数量和专家 ID histogram；
- expert load imbalance、entropy 和 top-k 热点比例；
- CPU/GPU expert placement、实际 GPU/EMC/CPU 频率；
- TTFT prefill、TPOT decode、Power、Energy/token 和温度时间序列。

没有路由统计时，只能说“MoE 模型在 Thor 上运行”，不能解释为什么某个频率或 placement 更好。

### M3：正式 SLO 与论文 gate

每个 workload 至少 3--5 次重复，采用已有 fixed-work integrity gate 和 all-repeat SLO 规则。一个 MoE placement 只有在 SLO 通过后才允许进入 Pareto；如果相对 dense baseline 的 Energy/token 改善小于重复噪声，就停止扩展该 placement。

建议把“有论文价值”的最低证据定义为：同一 SLO、同一 trace、同一质量/输出约束下，至少一个 MoE placement 相对 native dynamic 的 Energy/token 置信区间支持负差值，并同时解释 Power、TTFT 和 TPOT 的代价。

## 论文价值判断

MoE 会增加论文价值，但前提是研究问题从“Thor 上找一个更好的频率点”升级为：

> 在统一内存边端 GPU 上，如何联合选择专家放置、DVFS 和 serving 资源，使 Power、Energy/token、TTFT、TPOT 与 thermal risk 形成 workload-aware Pareto 前沿？

如果只使用 all-GPU MoE、只比较 3 个频率点，收益很可能仍然小，且贡献会退化成模型替换实验。如果引入 CPU-MoE/partial placement、路由统计和 workload-aware SLO，才有机会形成区别于当前 dense 结果的机制性贡献。

## 当前判断

- 不建议删除或替换现有 dense Thor 结果；它是可复现的控制基线。
- 建议先做一个 Mixtral Q4 runtime gate 和 `all-GPU/CPU-MoE` 小矩阵，不要直接启动大规模 full search。
- 若 CPU-MoE 的 Energy/token 与 TPOT 都没有明显可优化空间，再停止 MoE 扩展；这说明 Thor 的瓶颈更可能在 runtime/kernel 或功率遥测，而不是模型结构。
- 若 placement 造成明显的 Power--Energy/token--TTFT 冲突，再进入 partial-MoE 和路由感知 Pareto 搜索。
