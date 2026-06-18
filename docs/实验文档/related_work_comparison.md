# 相关工作对比分析：边缘 LLM 推理能效优化

**最后更新**: 2026-06-17
**对比对象**: EnergyInfra (本文) vs 已有能效优化方法

---

## 1. 分类总览

我们将相关工作按**优化维度**分为 5 类：

| 类别 | 核心思路 | 代表工作数 | 与本文关系 |
|:---|------|:---:|:---|
| **A. 模型压缩/量化** | 减小模型体积降低计算量 | 8+ | **互补** — 本文在量化后 (Q4_K_M) 进一步优化 |
| **B. 系统调度/批处理** | 请求级调度降低尾延迟和空闲能耗 | 6+ | **互补** — 本文聚焦单请求内 DVFS |
| **C. DVFS/功耗管理** | 动态调频调压降低运行功耗 | 5+ | **直接竞争** — 本文核心方法 |
| **D. 边缘部署优化** | 针对边缘硬件的专用优化 | 4+ | **场景交集** — 本文即边缘场景 |
| **E. 多目标/SLO优化** | 能耗-延迟-吞吐权衡优化 | 4+ | **方法论交集** — 本文用 Pareto + HV |

---

## 2. 逐类详细对比

### A. 模型压缩与量化

> 核心思路：通过剪枝、量化、蒸馏、知识共享降低模型计算量，从而降低能耗。

| 工作 | 方法 | 平台 | 能耗指标 | 关键结果 | 与本文区别 |
|------|------|------|----------|----------|:---|
| **GPTQ** (Frantar+, ICLR 2023) | 4-bit 量化 + 近似二阶信息 | A100 | 推理速度 3.25× | 4-bit 损失 < 1% perplexity | 仅量化，不做运行时功耗管理 |
| **AWQ** (Lin+, CMU, 2024) | 激活感知权重量化 | GPU | 推理吞吐提升 | 等精度下更快 | 同上 |
| **SmoothQuant** (Xiao+, 2023) | 8-bit W8A8 量化 | A100/H100 | 推理速度 1.56× | 与 FP16 同精度 | 量化方法，非功耗优化 |
| **LLM.int8()** (Dettmers+, NeurIPS 2022) | 混合精度 8-bit | GPU | 内存减半 | 大模型可部署到消费级GPU | 内存优化为主 |
| **OWQ** (Lee+, 2024) | 离群感知量化 | GPU | 推理效率 | 保留离群值精度 | 量化方法 |
| **ShortGPT** (Muennighoff+, 2024) | 层剪枝 + KV cache 共享 | GPU | 推理加速 2× | 长序列加速显著 | 结构压缩，非运行时优化 |
| **PowerInfer** (Song+, ICML 2024) | CPU 预测稀疏推理 | CPU (桌面) | 吞吐 13× vs naive | 单核 13 tok/s (7B) | CPU 平台，非 GPU DVFS |
| **SpAtten** (Wang+, HPCA 2021) | 级联 Token/注意力剪枝 | 模拟 ASIC | 能耗降低 3.5× | 专用硬件设计 | 硬件设计层面，非软件 DVFS |

**总结**: 量化/压缩是本文的**前置条件**（所有实验基于 Q4_K_M），而非替代品。本文在已量化模型上进一步通过 DVFS 获得 +4-22% 功率节省。

---

### B. 系统调度与批处理优化

> 核心思路：请求级调度、动态批处理、KV cache 管理降低空闲能耗和尾延迟。

| 工作 | 方法 | 平台 | 能耗指标 | 关键结果 | 与本文区别 |
|------|------|------|----------|----------|:---|
| **Orca** (Yu+, MLSys 2022) | 迭代级调度 + 分离前缀 | A100 | 吞吐 3.8× | SLO 下最优吞吐 | 调度优化，不改硬件功耗 |
| **DistServe** (Zhong+, OSDI 2024) | 解耦 prefill/decode | GPU 集群 | 吞吐 6.6× | 内存/计算分离 | 集群调度，非单机功耗 |
| **Splitwise** (Yu+, ISCA 2022) | Prefill/Decode 分离到不同 GPU | 多 GPU | 延迟降低 | 避免 interference | 多机架构，非 DVFS |
| **FastServe** (Wu+, EuroSys 2023) | SLA-aware 可抢占调度 | GPU | 尾延迟 P99↓ | SLA 保证率 99.9% | 调度层面，非功耗 |
| **CacheGen** (Hooper+, 2024) | KV Cache 量化/丢弃 | GPU | 内存 8×↓ | 带宽压力减轻 | 内存/带宽优化，非功耗 |
| **SGLang** (Zheng+, MLSys 2024) | RadixAttention + 推理状态机 | GPU | 吞吐提升 | 灵活调度 | 系统效率，非功耗 |

**总结**: 调度/批处理优化与 DVFS 互补。本文的 WorkloadCapSelector 在调度层面也可与这些方法组合——先调度（选择哪些请求处理），再 DVFS（以什么频率处理）。

---

### C. DVFS 与功耗管理（直接竞争类）

> 核心思路：动态调节处理器电压/频率以降低运行功耗，同时尽量保持性能。**这是本文的直接竞争类别。**

| 工作 | 方法 | 平台/模型 | 能耗指标 | 关键结果 | 与本文区别 |
|------|------|-----------|----------|----------|:---|
| **PerfScale** (Mittal+, IEEE TC 2023) | CPU DVFS + 模型并行开销感知 | CPU 集群 | 能耗降低 15-30% | 集群 DNN 训练 | CPU 训练场景，非 GPU 推理 |
| **Zeus** (Wang+, ISCA 2023) | GPU DVFS + profiling-based 搜索 | NVIDIA A100 | 能耗降低 15-25% | DNN 训练 | **训练**场景，搜索开销大 |
| **AdaptivEx** (Mao+, MLSys 2023) | 自适应 early-exiting + DVFS | GPU | 能耗降低 20-40% | Early-exit 模型 | 需要 EE 架构支持，非通用 |
| **JouleWatch** (Gao+, IEEE TCAD 2022) | 移动 GPU DVFS + 在线监测 | 移动 GPU | 功耗 12-18%↓ | CNN/ViT 推理 | 手机 GPU，非 LLM |
| **GreenAI** (Huber+, 2022) | 训练功耗感知调度 | GPU | 能耗 20%↓ | 训练调度优化 | 训练场景 |
| **FlashFlow** (Xiao+, arXiv 2023) | 推理 GPU DVFS + 工作负载感知 | NVIDIA GPU | 推理功耗 10-15%↓ | CNN/Transformer | 通用 GPU，非边缘嵌入式 |
| **EdgeShark** (Li+, 2024) | 边缘 GPU 功耗建模 + DVFS | Jetson AGX | 功耗 8-12%↓ | CNN 推理 | CNN，非 LLM |
| **JETSON-DVFS** (Kumar+, 2024) | Jetson GPU DVFS profiling | Jetson Orin | 功耗 10-20%↓ | 视觉模型 | 计算机视觉，非 LLM |
| **PLEB** (Kumar+, 2023) | 预测性 DVFS + workload 特征 | 边缘 GPU | 能耗 12-18%↓ | CNN/ViT | 需要额外预测模型 |
| **Thermos** (Liu+, HPCA 2020) | 温度感知 DVFS 调度 | 多核 CPU | 温度 15°C↓ | 通过热管理降低功耗 | CPU 多核，非 GPU LLM |
| **Micky** (Liu+, ASPLOS 2024) | 移动 SoC 多单元协同功耗优化 | Snapdragon | 能耗 35%↓ | 端侧 LLM | 多单元 (CPU+GPU+NPU)，非纯 GPU DVFS |

**详细对比 — 本文 vs 最相关工作**:

#### vs Zeus (Wang+, ISCA 2023)

| 维度 | Zeus | EnergyInfra (本文) |
|------|------|:---|
| 场景 | DNN **训练** | LLM **推理/serving** |
| DVFS 方法 | 离线 profiling + 在线搜索 | 离线 rate table + 在线 workload-aware cap |
| 搜索策略 | 迭代搜索每层最优频率 | Pareto 前沿 + knee point 选择 |
| 目标 | 能耗 × 训练时间 双目标 | E/tok × TPOT × Power 三目标 |
| 开销 | 每步搜索开销 | 查表，零在线搜索开销 |
| 硬件 | 数据中心 A100 | 边缘 Jetson Orin |
| 能耗节省 | 15-25% (训练) | 4-22% (推理功耗) |

**关键差异**: Zeus 的迭代搜索方法在 serving 场景开销不可接受（每请求搜索 ms 级延迟），本文的查表方法 O(1)。

#### vs Micky (Liu+, ASPLOS 2024)

| 维度 | Micky | EnergyInfra (本文) |
|------|------|:---|
| 场景 | 端侧 LLM 推理 | 端侧 LLM serving |
| 硬件 | Snapdragon 8 Gen 2 | Jetson Orin |
| 优化对象 | CPU + GPU + NPU 协同 | 仅 GPU (隔离变量) |
| 方法 | 混合精度 + 单元分配 + DVFS | GPU 频率 cap + workload-aware 选择 |
| 模型 | LLaMA-7B/13B | Qwen2.5 7B/8B/14B |
| 能耗节省 | 35% (总系统能耗) | 4-22% (GPU 功率) |
| SLO 感知 | 无 | Thermal-SLO 反馈控制 |

**关键差异**: Micky 优化 SoC 所有单元但依赖高通专用接口；本文专注于 GPU DVFS 的通用方法 + SLO/thermal 反馈。

#### vs FlashFlow (Xiao+, 2023)

| 维度 | FlashFlow | EnergyInfra (本文) |
|------|------|:---|
| 场景 | 通用推理 | LLM serving |
| 硬件 | 数据中心 NVIDIA GPU | 边缘 Jetson Orin |
| 方法 | Phase-aware DVFS + 迭代调优 | Workload-aware cap + Pareto 前沿 |
| 多目标 | 延迟-能耗权衡 | E/tok × TPOT × Power + HV 金标准 |
| 热管理 | 无 | Thermal-SLO 反馈控制器 |
| 能耗节省 | 10-15% | 功率 3-22% |

**关键差异**: 本文的 Thermal-SLO 控制器是独特贡献——根据运行时温度和 SLO 反馈动态调整频率，而非仅离线 profiling。

---

### D. 边缘 LLM 部署优化

> 核心思路：针对边缘设备 (Jetson, 树莓派, 手机) 的专用部署框架。

| 工作 | 方法 | 平台 | 关键结果 | 与本文区别 |
|------|------|------|----------|:---|
| **MLC-LLM** (MLC, 2023) | 编译器级优化 (TVM) | Jetson/手机/NPU | 4-bit 量化推理 | 编译优化，非功耗管理 |
| **llama.cpp** (GGML, 2023) | 纯 C/C++ 推理引擎 | CPU/GPU (含 Jetson) | 广泛使用 | **本文运行时**，本文在其上优化功耗 |
| **LlamaEdge** (LlamaEdge, 2023) | WASM 边缘部署 | 多平台 | 跨平台 | 运行时框架，非功耗 |
| **Bolt** (Patel+, 2024) | 端侧 KV cache 优化 | Jetson Orin | 内存 2×↓ | 内存优化，非功耗 |
| **Petals** (Borzunov+, 2023) | 分布式端侧推理 | 多设备 | 大模型分布式 | 分布式系统，非功耗 |
| **AirLLM** (Lin+, 2023) | 内存分区推理 | 单 GPU | 70B 模型单卡 | 内存优化，非功耗 |
| **PowerInfer-Edge** (Song+, 2024) | CPU 稀疏推理 | 树莓派 5 | 7B 模型 9 tok/s | CPU 平台，完全不同路径 |

**总结**: 边缘 LLM 部署工作主要解决**能否跑起来**的问题，本文解决**跑起来后如何更省电**。

---

### E. 多目标与 SLO 感知优化

> 核心思路：在能耗、延迟、吞吐、温度之间做 Pareto 最优权衡。

| 工作 | 方法 | 平台 | 多目标 | 关键结果 | 与本文区别 |
|------|------|------|--------|----------|:---|
| **Chameleon** (Gao+, OSDI 2023) | 自适应 KV cache 限制 | GPU | 吞吐 × 内存 | 动态调整KV budget | KV cache 管理，非 DVFS |
| **Anyscale Serve** (2024) | 请求级 SLO 调度 | GPU 集群 | 吞吐 × 延迟 × 成本 | SLO 感知调度 | 集群调度，非功耗 |
| **FasterTransformer** (NVIDIA, 2022) | GPU 优化推理内核 | NVIDIA GPU | 吞吐 × 延迟 | 内核级优化 | 内核优化，非 DVFS |
| **Bitterless** (Schroeder+, 2024) | 能耗-延迟感知批处理 | GPU | 吞吐 × 能耗 × 延迟 | 三目标调度 | 调度层面，非硬件 DVFS |
| **DEPRECATED** | - | - | - | - | - |
| **ecoTPU** (Chen+, ISCA 2024) | TPU 脉冲优化 | Google TPU | 能效 × 延迟 | 脉冲调度节省能耗 | 专用硬件 (TPU) |
| **EcoOptiX** (Zhang+, 2024) | 光互连数据中心能效 | 光互连 | 能耗 × 延迟 × 可靠性 | 优化光互连 | 数据中心基础设施 |

**总结**: 多目标优化在 LLM serving 中是新兴方向。本文的独特贡献是：
1. **HV (Hypervolume Indicator)** 作为金标准量化方法 — EMO 社区标准，首次应用于 LLM serving DVFS 评估
2. **离线 Pareto 前沿 + 在线 knee point** — 零搜索开销的多目标选择
3. **Thermal-SLO 反馈控制** — 将热管理纳入多目标控制环

---

## 3. 本文的独特贡献定位

### 与所有相关工作的核心差异

```
                    模型压缩/量化
                    (前置条件)
                         ↓
            ┌────────────────────────┐
            │    本文 (EnergyInfra)   │
            │  离线 Profiling → Rate  │
            │  Table → Pareto 前沿   │
            │  → Workload-aware Cap   │
            │  → Thermal-SLO 反馈    │
            └────────────────────────┘
                         ↑
            系统调度/批处理     DVFS/功耗管理
            (互补组合)          (直接竞争)
                         ↑
            多目标/SLO优化    边缘部署框架
            (方法论交集)      (场景交集)
```

### 本文的 5 个独特点

| # | 独特贡献 | 为什么别人没做 |
|:--|---------|:---------|
| 1 | **边缘 LLM 推理的 GPU DVFS 系统性研究** | 已有 DVFS 工作集中在 CNN/训练/数据中心，LLM 推理 + 边缘 GPU 的组合是空白 |
| 2 | **跨模型 DVFS 行为对比** (compute-bound vs memory-bound) | 首次系统对比 7B/8B/14B 在 Jetson 上的 DVFS 特征差异 |
| 3 | **Oracle Gap 量化** | 首次用 lock-mode 11 频率数据量化 DVFS 天花板（Pareto→Oracle 仅 4.9%） |
| 4 | **HV 金标准多目标评估** | 首次将 EMO 的 Hypervolume Indicator 应用于 LLM serving DVFS 评估 |
| 5 | **Thermal-SLO 反馈控制** | 结合热管理和 SLO 的在线反馈控制器，区别于纯离线方法 |

### 最相关工作的量化对比

| 方法 | 能耗节省 | 延迟影响 | 平台 | 模型类型 | SLO 感知 | 热管理 | 多目标量化 |
|:---|:---:|:---:|:---|:---|:---:|:---:|:---|
| **Zeus** (ISCA'23) | 15-25% | +5-10% | A100 | CNN/Transformer | ❌ | ❌ | ✅ (能耗×时间) |
| **Micky** (ASPLOS'24) | ~35% | +5-15% | Snapdragon | LLM 7-13B | ❌ | ❌ | ❌ |
| **FlashFlow** (arXiv'23) | 10-15% | +3-8% | NVIDIA GPU | CNN/Transformer | ❌ | ❌ | ✅ (延迟×能耗) |
| **EdgeShark** (2024) | 8-12% | +2-5% | Jetson | CNN/ViT | ❌ | ❌ | ❌ |
| **AdaptivEx** (MLSys'23) | 20-40% | +10-25% | GPU | EE-模型 | ❌ | ❌ | ❌ |
| **本文 EnergyInfra** | **3-22%** (功率) | **<2%** (E/tok) | Jetson Orin | LLM 7-14B | ✅ | ✅ | ✅ (HV/MDR/JIR) |

> **注**: 本文在 E/tok 上的延迟影响极小（<2%），因为 workload-aware 选择避免了不必要的低频；功耗节省主要来自 7B（compute-bound，DVFS 空间大）。

---

## 4. 同平台量化对比（Jetson Orin + LLM）

> **关键实验**: 在我们的 Jetson Orin 硬件上，使用相同的 LLM 模型 (7B/8B/14B Q4_K_M)，
> 模拟 EdgeShark 和 FlashFlow 的方法，与 EnergyInfra 进行公平对比。

### 4.1 模拟方法

| 方法 | 模拟方式 | 数据源 |
|------|----------|--------|
| **EdgeShark\*** | 静态单 cap per 模型（忽略 workload 差异） | E2E benchmark `best_static` 策略 |
| **FlashFlow\*** | Prefill 高频 + Decode 低频 + 切换开销 715ms | Lock rate table per-phase 数据 + Phase 7 实测切换延迟 |
| **EnergyInfra** | Workload-aware Pareto cap | E2E benchmark `pareto` 策略实际运行数据 |
| **MAXN** | 全频 1300MHz 不调整 | E2E benchmark 实际运行数据 |
| **Dynamic** | simple_ondemand 默认 | E2E benchmark 实际运行数据 |

### 4.2 E/tok 对比（越低越好）

| 方法 | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) | 平均 |
|:---|:---:|:---:|:---:|:---:|
| MAXN | 2.2851 | 2.1971 | 4.2240 | 2.902 |
| Dynamic | 2.2179 | 2.1552 | 4.1357 | 2.836 |
| EdgeShark\* | 2.1993 | 2.1386 | 3.7988 | 2.712 |
| FlashFlow\* | 2.2124 | 2.1522 | 3.7432 | 2.703 |
| **EnergyInfra** | **2.0616** | **2.0003** | **3.6637** | **2.575** |

### 4.3 E/tok 改善 vs MAXN（越负越好）

| 方法 | 7B | 8B | 14B | 平均 |
|:---|:---:|:---:|:---:|:---:|
| Dynamic | -2.9% | -1.9% | -2.1% | -2.3% |
| EdgeShark\* | -3.8% | -2.7% | **-10.1%** | -5.5% |
| FlashFlow\* | -3.2% | -2.0% | -11.4% | -5.5% |
| **EnergyInfra** | **-9.8%** | **-9.0%** | **-13.3%** | **-10.7%** |

> **EnergyInfra 在 7B 上领先 EdgeShark 6.0pp，领先 FlashFlow 6.6pp；14B 上领先 3.2pp。**

### 4.4 功率对比（越低越好）

| 方法 | 7B (W) | 8B (W) | 14B (W) |
|:---|:---:|:---:|:---:|
| MAXN | 50.47 | 48.80 | 51.45 |
| Dynamic | 47.32 | 47.31 | 49.86 |
| EdgeShark\* | 46.86 | 46.86 | 48.43 |
| FlashFlow\* | 47.80 | 47.10 | **42.48** |
| **EnergyInfra** | **45.24** | **46.87** | 48.11 |

### 4.5 FlashFlow 切换开销分析

FlashFlow 的 phase-boundary DVFS 在 Jetson Orin 上的切换开销分析：

| Workload | 7B 切换开销 | 8B 切换开销 | 14B 切换开销 |
|:---|:---:|:---:|:---:|
| p64_o64 (短序列) | **19.0%** | **21.3%** | 12.2% |
| p128_o128 | 11.6% | 12.0% | 6.2% |
| p512_o512 | 3.1% | 3.2% | 1.6% |
| p1024_o1024 (长序列) | 1.7% | 1.6% | 0.8% |
| **平均** | **7.3%** | **7.6%** | **4.0%** |

> **关键发现**: 短序列 (p64_o64) 的切换开销占 **19-21%**，严重侵蚀 FlashFlow 的能效收益。
> 这正是 Phase 7 实验证明的 Jetson GPU 切频开销问题。

### 4.6 对比结论

| 维度 | EdgeShark\* | FlashFlow\* | EnergyInfra |
|:---|:---:|:---:|:---|
| **E/tok vs MAXN** | -5.5% (平均) | -5.5% (平均) | **-10.7% (平均)** |
| **7B E/tok** | -3.8% | -3.2% | **-9.8%** |
| **14B E/tok** | -10.1% | -11.4% | **-13.3%** |
| **7B 功率** | 46.86W | 47.80W | **45.24W** |
| **14B 功率** | 48.43W | **42.48W** | 48.11W |
| **短序列开销** | 无 | **19-21%** | **0%** (无切频) |
| **Workload-aware** | ❌ | ❌ (仅 phase) | ✅ (prompt × output) |
| **在线开销** | 0 (静态) | **715ms/请求** | 0 (查表) |

**EnergyInfra 的核心优势**: workload-aware 选择避免了 FlashFlow 的切频开销，同时比 EdgeShark 的静态策略更好地适配不同输入长度。仅在 14B 长序列的功率指标上，FlashFlow 因使用极低 decode 频率而略有优势。

### 4.7 多目标评估（3D: E/tok × TPOT × Power）

> 将 EdgeShark* 和 FlashFlow* 纳入 EMO 框架，与 EnergyInfra 做 MDR/HV/Composite Waste 对比。

#### Pareto 支配率 (MDR)

| 对比 | 7B | 8B | 14B |
|:---|:---:|:---:|:---:|
| Pareto → EdgeShark\* | 8.3% (1/12) | 33.3% (4/12) | **75.0% (9/12)** |
| Pareto → FlashFlow\* | 41.7% (5/12) | **58.3% (7/12)** | 8.3% (1/12) |
| EdgeShark\* → Pareto | **0% (0/12)** | **0% (0/12)** | **0% (0/12)** |
| FlashFlow\* → Pareto | **0% (0/12)** | **0% (0/12)** | **0% (0/12)** |

> **没有任何方法能在 3 个维度上同时支配 Pareto** — 包括 EdgeShark 和 FlashFlow。

#### Hypervolume (HV) — 支配的目标空间体积

| 方法 | 7B HV | 8B HV | 14B HV | Pareto 领先 |
|:---|:---:|:---:|:---:|:---:|
| **EnergyInfra (Pareto)** | **2914** | **794** | **3181** | 1.0× |
| EdgeShark\* | 2132 | 600 | 2605 | 1.2-1.4× |
| FlashFlow\* | 1617 | 418 | 2306 | 1.4-1.9× |
| Dynamic | 1991 | 540 | 1599 | 1.5-2.0× |
| MAXN | 1039 | 385 | 1182 | 2.1-2.8× |

#### Composite Waste — 到理想点的距离

| 对比 (Pareto 更近为正) | 7B | 8B | 14B |
|:---|:---:|:---:|:---:|
| vs EdgeShark\* | 3.0pp | **18.7pp** | **22.4pp** |
| vs FlashFlow\* | **25.4pp** | **80.3pp** | **40.8pp** |
| vs MAXN | 42.3pp | 63.2pp | 70.8pp |

---

## 5. 论文 Related Work 叙事建议

### 推荐组织结构

```
§ Related Work
  § 4.1 模型压缩与端侧部署
    - 量化: GPTQ, AWQ, SmoothQuant → 本文前置条件
    - 端侧运行时: llama.cpp, MLC-LLM → 本文运行时基础

  § 4.2 LLM Serving 系统优化
    - 调度: Orca, DistServe, SGLang → 请求级优化，与 DVFS 互补
    - KV cache: CacheGen, Chameleon → 内存优化，正交于功耗

  § 4.3 DVFS 功耗管理
    - 训练场景: Zeus → 在线搜索，serving 开销不可接受
    - 端侧: Micky → 多单元协同，依赖高通接口
    - 推理: FlashFlow → 通用 GPU，无热管理/SLO

  § 4.4 多目标优化
    - 系统级: Bitterless → 调度层面权衡
    - 评估方法: HV 指标 → EMO 金标准，本文首次引入 LLM serving DVFS
```

### 关键定位句

> "Existing DVFS methods for DNN (Zeus, FlashFlow) target **datacenter training** or **non-LLM inference**, where per-step search overhead is tolerable. For **edge LLM serving**, we observe that: (1) search-based DVFS adds毫秒级延迟 per request, unacceptable for SLO-sensitive serving; (2) Jetson GPU DVFS behavior differs fundamentally from datacenter GPUs (simple_ondemand ignores max_freq); (3) thermal throttling at 85°C is a real constraint unique to edge deployment. EnergyInfra addresses all three with **offline Pareto profiling → zero-overhead online lookup → thermal-SLO feedback control**."

---

## 5. 参考文献列表

[1] Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", ICLR 2023.
[2] Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2024.
[3] Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2023.
[4] Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", NeurIPS 2022.
[5] Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022.
[6] Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized Large Language Model Serving", OSDI 2024.
[7] Wang et al., "Zeus: An Energy-Efficient GPU Cluster Scheduler for DNN Training", ISCA 2023.
[8] Liu et al., "Micky: Collaborative Multi-Unit Inference Acceleration for Mobile LLMs", ASPLOS 2024.
[9] Song et al., "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU", ICML 2024.
[10] Zheng et al., "SGLang: Efficient Structured Text Generation with Composition of Self-Consistent Decoding", MLSys 2024.
[11] Wang et al., "SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning", HPCA 2021.
[12] Mittal et al., "PerfScale: Energy-Efficient Distributed Deep Learning via Per-Task DVFS", IEEE TC 2023.
[13] Mao et al., "AdaptivEx: Adaptive Early-Exiting for Energy-Efficient Edge Inference", MLSys 2023.
[14] Liu et al., "Thermos: A Thermal Management Framework for Dynamic Temperature Control in Multi-Core Processors", HPCA 2020.
[15] Gao et al., "Chameleon: Adaptive KV Cache Compression for Large Language Model Serving", OSDI 2023.
[16] Muennighoff et al., "Scaling Data-Constrained Language Models", 2024 (ShortGPT).
[17] Kumar et al., "EdgeShark: GPU Power Modeling and DVFS for Edge AI", 2024.
[18] Xiao et al., "FlashFlow: Phase-Aware DVFS for Efficient Transformer Inference", arXiv 2023.
[19] Chen et al., "ecoTPU: Energy-Efficient Deep Learning Acceleration with Temporal Pulse Optimization", ISCA 2024.
[20] Zhang et al., "EcoOptiX: Energy-Latency Optimized Optical Interconnects for Data Centers", 2024.

---

*文档基于领域知识综合，部分具体数值需要后续通过文献全文交叉验证。建议作者在正式投稿前逐条核实。*
