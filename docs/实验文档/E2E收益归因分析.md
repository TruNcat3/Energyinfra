# E2E 收益归因分析

> 本文档为研究过程阶段记录，保留撰写时点的原始结论；最新汇总见 [实验总览](实验总览.md) 与 [文档索引](README.md)。

**更新日期**：2026-05-19  
**对应实验**：实验 4（细粒度 GPU×EMC 能耗建模）完成后的 E2E 验证  
**更新原因**：分析 E2E baseline 能耗差异较小的原因与其他工作 token/J 收益差异的来源，并给出后续实验设计建议。

---

## 1. 当前阶段判断

> 撰写时点状态，仅供参考

此时已完成从基础 profiling 到细粒度 GPU×EMC DVFS 建模的主要工程闭环，具体包括：

- 通过 DebugFS 增加 EMC 频率控制；
- 完成 11 GPU × 4 EMC × 5 workloads × 2 phases × 3 repeats 的细粒度 profiling；
- 构建 GPU×EMC×CPU 三维能耗汇率表；
- 升级 WeightedSelector，使其支持 alpha 权重旋钮；
- 完成 225-run 的 E2E benchmark，用于验证 alpha-weighted DVFS 策略。

总体上，工程已经从“能耗测量脚本”推进到“能耗汇率表 + 配置选择器 + E2E 实验框架”。但是，当前实验结论需要区分 **decode-only profiling** 和 **mixed-phase E2E benchmark** 两个层次。前者结论较强，后者收益较弱。

---

## 2. 细粒度 GPU×EMC profiling 的主要结论

### 2.1 GPU 频率存在明确的能效甜点

Decode 阶段下，GPU 频率对性能和能耗均有显著影响。低频虽然降低功耗，但 TPOT 明显增大，导致 E/token 反而变差；最高频虽然提高吞吐，但功耗上升明显，也不是最优能效点。

| GPU 频率 | E/token | TPOT | TPS | Power |
|---:|---:|---:|---:|---:|
| 306 MHz | 1.33 J | 81.8 ms | 12.2 | 16.2 W |
| 612 MHz | 1.00 J | 43.4 ms | 23.1 | 22.6 W |
| 816 MHz | 0.94 J | 33.3 ms | 30.1 | 28.9 W |
| 1300 MHz | 1.15 J | 24.8 ms | 40.3 | 42.1 W |

因此，当前最稳妥的结论是：

> 在 Phi-3-mini-Q4 的 Jetson Orin 推理场景中，GPU 频率与能效之间呈非单调关系。中间频率，尤其 GPU816 附近，是比最低频和最高频更优的能效运行点。

这说明 DVFS 仍然有价值，尤其适合通过离线汇率表寻找 workload-specific 的运行点。

### 2.2 EMC 频率不是当前小模型的一阶瓶颈

EMC 频率提升能够改善 TPOT 和 TPS，但功耗也同步升高，最终 E/token 没有改善。

| EMC 频率 | E/token | TPOT | TPS | Power |
|---:|---:|---:|---:|---:|
| 204 MHz | 1.10 J | 38.4 ms | 30.9 | 32.1 W |
| 665 MHz | 1.13 J | 35.2 ms | 33.8 | 35.2 W |
| 2133 MHz | 1.17 J | 30.2 ms | 37.4 | 39.6 W |
| 3199 MHz | 1.16 J | 30.1 ms | 37.5 | 39.4 W |

更准确的表述不是“EMC 无影响”，而是：

> 对 Phi-3-mini-Q4 当前配置而言，EMC 不是主要性能瓶颈。提高 EMC 能带来一定延迟收益，但其功耗代价更明显，因此低 EMC 频率更适合作为能效优先配置。

---

## 3. E2E benchmark 的主要结论需要更谨慎

E2E benchmark 中，alpha-selected 策略和固定 baseline 之间的差异较小，基本在 3% 以内。

| 策略 | Alpha | E/token | TPOT | TPS |
|---|---:|---:|---:|---:|
| single_config | 0.0 | 1.2407 | 27.3 ms | 35.5 |
| single_config | 0.5 | 1.2473 | 26.0 ms | 36.8 |
| single_config | 1.0 | 1.2372 | 26.0 ms | 37.0 |
| phase_aware | 0.0 | 1.2702 | 26.1 ms | 36.1 |
| phase_aware | 0.5 | 1.2277 | 25.9 ms | 37.3 |
| phase_aware | 1.0 | 1.2316 | 26.0 ms | 37.2 |

Baseline 之间也很接近：

| Baseline | E/token | TPOT | TPS |
|---|---:|---:|---:|
| MAXN | 1.2502 | 25.8 ms | 37.3 |
| 30W_mode | 1.2660 | 25.7 ms | 37.0 |
| E_min | 1.2409 | 26.0 ms | 36.9 |

从这些结果看，当前 E2E 层面不宜写成“DVFS 带来显著端到端节能”。更准确的结论应是：

> 当前 E2E benchmark 证明了细粒度 DVFS selector 和测量框架可以运行，但在 Phi-3-mini-Q4 的 mixed-phase 推理场景下，不同频率配置的端到端 E/token 差异较小。当前收益不足以支撑“显著提升 token/J”的强主张。

---

## 4. 为什么多个 baseline 的 E2E 能耗差异不大

### 4.1 当前 baseline 本质上不是不同系统

当前比较对象主要是同一设备、同一模型、同一 runtime、同一算子实现下的不同频率配置：

- MAXN：GPU1300 + EMC3199 + CPU1497
- 30W_mode：GPU612 + EMC2133 + CPU1497
- E_min：GPU816 + EMC204 + CPU1036

这些 baseline 没有改变 kernel 实现、调度机制、KV cache 管理、memory layout、量化方式、稀疏格式或 batching 策略。因此它们之间的差异只是运行点差异，而不是系统能力差异。

### 4.2 几个 baseline 都接近 Pareto 区域

当前 E_min 本身就是从 profiling 中得到的能效优选点，不是弱 baseline。30W_mode 在当前小模型上也能达到接近 MAXN 的吞吐。MAXN 虽然功耗更高，但运行时间略短，会抵消一部分能耗劣势。因此这几个 baseline 实际上都不弱，彼此差异自然小。

### 4.3 Mixed-phase E2E 稀释了 decode 差异

Decode-only profiling 中，GPU306、GPU816、GPU1300 的 E/token 差异明显。但 E2E benchmark 包含 prefill + decode。若 prefill 占据较多总能耗，且当前策略没有真正实现 prefill/decode 在线切频，则 decode 阶段的能效差异会被总能耗平均掉。

### 4.4 静态功耗和不可控功耗占比不低

Jetson 平台存在 SoC 基线功耗、板级功耗、DRAM/PMIC/风扇损耗、CPU runtime overhead、采样与同步开销等固定或半固定开销。DVFS 主要影响动态功耗和部分性能。如果可调功耗只占总功耗的一部分，则即使 GPU/EMC 动态功耗改善明显，反映到系统级 E/token 上也可能只剩几个百分点。

### 4.5 当前 phase-aware E2E 还不是真正在线 phase switching

当前 E2E 脚本中，phase-aware 会选择 prefill config 和 decode config，但实际运行时仍用 decode config 执行完整 mixed inference。也就是说，当前还没有真正实现：

```text
prefill config → phase boundary → decode config
```

因此，当前 E2E 的 phase-aware 结果更像是“phase-aware selector 选出的 decode 配置在完整推理中表现如何”，不能等同于真实在线 phase-aware DVFS 的收益。

### 4.6 输出 token 数不稳定会影响 E/token 对比

E2E CSV 中存在部分 run 的 output_tokens 低于目标输出长度的情况，尤其在长输出 workload 中较明显。这可能来自模型提前 EOS，但它会影响 E/token、TPOT 和 TPS 的横向比较。后续应将完整输出 runs 与提前结束 runs 分开统计。

---

## 5. 为什么其他工作常有 40%–50% token/J 差距

EnergyInfra 当前的收益尺度不应直接和那些动辄 40%–50% token/J 提升的工作对比，因为优化杠杆不同。

其他工作的大幅收益通常来自以下几类因素：

1. **baseline 更弱**  
   许多论文 baseline 是默认系统、naive runtime、固定高频、未优化调度或通用 kernel。若 baseline 离 Pareto frontier 很远，收益自然更大。

2. **优化对象不是单纯 DVFS**  
   很多系统工作改变的是主导开销，例如 kernel fusion、attention 优化、KV cache 管理、batching、offload、稀疏格式、量化布局、memory scheduling 等。这些会直接减少单位 token 的实际计算或访存量。

3. **模型和负载更大**  
   大模型、长输出、高并发服务中，decode 占比更高，KV cache 和访存压力更明显。此时频率、内存带宽和调度策略的影响更容易被放大。

4. **对比的是跨后端或跨系统**  
   例如 PyTorch vs TensorRT-LLM、naive attention vs FlashAttention、CPU fallback vs GPU kernel、unpaged KV vs paged KV。这类比较的收益尺度天然大于同一 runtime 内部的频率配置选择。

因此，应将 EnergyInfra 当前定位为：

> 面向边端 LLM 推理的 workload/phase-aware 能耗汇率表建模与配置选择方法。

而不是：

> 已经实现 40%–50% token/J 提升的端到端推理系统优化方法。

---

## 6. 当前可以支撑的论文/报告结论

### 6.1 可以强支撑的结论

1. **Jetson LLM 推理存在非单调 DVFS 能效曲线**  
   最低频和最高频都不一定是最优点，中间频率可能达到更优 E/token。

2. **GPU 频率是当前小模型 decode 阶段的重要控制维度**  
   GPU 频率显著影响 TPOT、TPS、Power 和 E/token。

3. **EMC 对当前 Phi-3-mini-Q4 不是一阶瓶颈**  
   提升 EMC 可以降低 TPOT，但能耗代价更高，低 EMC 更适合能效优先策略。

4. **离线 profiling + rate table + weighted selector 的工程链路已经跑通**  
   当前已经形成从真实测量到 Pareto 建表，再到 alpha-weighted 配置选择和 E2E 验证的实验框架。

### 6.2 不宜过度声明的结论

1. 不宜声明当前 E2E 已实现显著 token/J 提升。  
2. 不宜把当前 phase-aware E2E 结果解释为真实在线 phase switching 收益。  
3. 不宜声称 EMC 对推理完全无影响。  
4. 不宜直接和改变 kernel/runtime 的系统工作比较 token/J 提升幅度。

---

## 7. 建议调整后的研究叙事

原叙事如果强调“端到端大幅节能”，目前证据不足。更建议调整为：

> 边端 LLM 推理中，硬件频率配置与能效之间存在 workload- and phase-dependent 的非单调关系。EnergyInfra 通过真实 profiling 构建能耗汇率表，并利用 alpha-weighted selector 在能耗和延迟之间选择合适配置。当前细粒度实验表明，中间 GPU 频率和低 EMC 频率可形成更优能效点；但在 Phi-3-mini-Q4 的 mixed-phase E2E 推理中，不同配置的系统级差异较小，说明小模型和未真正在线切频会掩盖 decode 阶段的 DVFS 收益。后续需要在更大模型、长 decode、真实在线 phase switching 和长期热稳定条件下验证收益上限。

这个叙事更严谨，也更接近目前数据能支撑的范围。

---

## 8. 后续实验建议

### 8.1 更大模型验证

优先选择：

- Llama-3-8B / Llama-3.1-8B
- Qwen2.5-7B
- Mistral-7B
- 同一模型的不同量化版本，例如 Q4、Q5、Q8

目的：放大计算和访存压力，观察 GPU/EMC DVFS 是否在大模型上产生更明显 E2E 差异。

### 8.2 更长 decode workload

增加 output length：

- 512 tokens
- 1024 tokens
- 2048 tokens

并尽量控制提前 EOS，保证不同配置之间 output_tokens 可比。

### 8.3 增加真正弱 baseline

建议加入：

- Jetson 默认动态调频；
- simple_ondemand 默认策略；
- MAXN 固定高频；
- 最低频固定；
- 只调 GPU、不调 EMC；
- 只调 EMC、不调 GPU；
- workload-unaware 固定配置；
- phase-unaware 最优配置；
- 随机配置。

这样可以更清楚地区分“建表选择”的收益和“频率空间本身”的收益。

### 8.4 实现真实在线 phase switching

需要将 E2E 实验升级为：

```text
设置 prefill config
运行 prefill
记录 prefill energy / latency
在 phase boundary 切换到 decode config
运行 decode
记录 decode energy / latency
记录 switch overhead
计算 total E/token
```

这一步是验证 phase-aware DVFS 是否真正有效的关键。

### 8.5 加入长期稳定性和热效应实验

建议做 30 分钟到 1 小时连续推理，记录：

- 温度曲线；
- 是否 thermal throttling；
- 实际频率是否偏离目标；
- E/token 是否漂移；
- selector 推荐配置是否随温度状态变化。

---

## 9. 对当前文档和代码的修改建议

### 9.1 文档修改

建议在内部开发文档（未随仓库发布）中记录以下结论：

> 当前 E2E benchmark 中不同配置的 E/token 差异较小，主要因为比较对象均为同一 runtime 下接近 Pareto 的频率配置，且 Phi-3-mini-Q4 的 mixed-phase 推理中 prefill、系统静态功耗和 runtime overhead 稀释了 decode 阶段的 DVFS 差异。当前结果证明了细粒度汇率表和 selector 的工程可行性，但尚不足以支撑显著端到端节能结论。后续需在更大模型、长 decode 和真实在线 phase switching 下继续验证。

### 9.2 E2E CSV 字段修改

建议保留以下字段：

- `target_gpu_mhz`
- `target_emc_mhz`
- `target_cpu_mhz`
- `actual_gpu_mhz`
- `actual_emc_mhz`
- `actual_cpu_mhz`
- `prefill_gpu`
- `prefill_emc`
- `decode_gpu`
- `decode_emc`
- `phase_switch`
- `output_tokens`
- `target_output_tokens`
- `is_complete_output`

这样才能复核 selector 是否真的选中了预期配置，以及频率控制是否生效。

### 9.3 数据分析修改

建议所有 E2E 结果都分成三组报告：

1. 所有有效 runs；
2. output_tokens 达到目标的 complete runs；
3. 提前 EOS 或 output_tokens 不足的 incomplete runs。

对于 token/J 对比，优先使用 complete runs。

---

## 10. 总结

当前 EnergyInfra 的核心价值已经比较清晰：

1. 证明 Jetson 上 LLM 推理存在非单调频率—性能—能耗关系；
2. 发现 GPU816 + EMC204 一类中间频率低内存频率组合具有较好能效；
3. 建立了可复用的 GPU×EMC×CPU 细粒度能耗汇率表；
4. 实现了 alpha-weighted selector 和 E2E benchmark 流程。

但当前 E2E baseline 差异较小，不能简单包装成显著 token/J 提升。更准确地说：

> 当前工作已经证明了能耗汇率表建模和配置选择框架的可行性，并在 decode profiling 中观察到清晰的 DVFS 能效空间；但在 Phi-3-mini-Q4 的 mixed E2E 推理中，收益被小模型、prefill 占比、系统静态功耗、强 baseline 和未真实在线切频机制显著稀释。后续应以更大模型、长 decode 和真实 phase switching 为重点，验证该方法在更高压力场景下的收益上限。

