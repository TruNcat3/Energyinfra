# Jetson LLM Energy Rate Table 项目进展总结

**更新时间**: 2026-05-13
**项目阶段**: Phase 5 - 高级优化 (进行中)

---

## 已完成的主要工作

### 1. 环境搭建与验证
- **llama.cpp环境**: 成功安装并验证 (版本 0.3.22)
- **模型准备**: 下载并配置Phi-3-mini Q4模型 (2.3GB)
- **系统监控**: 实现无需sudo的系统监控脚本
- **基准测试**: 完成llama.cpp基准性能测试

### 2. 频率控制实验
- **频率控制验证**: 成功实现GPU频率动态控制
- **多频率测试**: 完成4个频率配置的完整测试
- **重要发现**: GPU频率不是主要瓶颈，4.25x频率提升仅带来8.3%性能提升
- **最佳能效**: 最低频率(306 MHz)具有最佳能效比

### 3. 7个前置实验 (全部完成)

| 实验 | 核心发现 |
|------|----------|
| 4.1 测量稳定性 | TPOT CV=1.0% 优秀, TTFT CV=9.2% 可接受 |
| 4.2 单旋钮敏感性 | GPU 378MHz 最能效 (7.36 tok/W), 1428MHz 最吞吐 |
| 4.3 频率组合交互 | 67% 配置满足 SLO |
| 4.4 Prefill/Decode 差异 | **26.97x** 阶段能效差异 |
| 4.5 负载可预测性 | workload 特征可预测最优配置 |
| 4.6 切换开销 | GPU 50ms, 可隐藏于 phase boundary |
| 4.7 SLO 端到端 | 能效优先策略可行 |

### 4. 能耗汇率表构建
- **配置数量**: 23个配置 (3 GPU x 3 phases + 变体)
- **数据格式**: Parquet 格式，支持高效查询
- **包含指标**: TTFT, TPOT, energy/token, tokens/J, power, temperature
- **Phase-specific 能量列**: `energy_per_input_token_j`, `energy_per_output_token_j`, `energy_per_total_token_j`
- **SLO标记**: 每个配置标记是否满足SLO约束
- **Pareto分析**: 修正后的严格Pareto最优标记

### 5. SLO-Aware 配置选择器
- **选择策略**: 精确匹配 -> 最近匹配 -> 回退安全配置
- **SLO约束**: TTFT < 1000ms, TPOT < 80ms, Power < 40W, Temp < 80C
- **距离计算**: 对数距离 + phase mismatch 惩罚
- **Phase-aware 能量选择**: 根据phase自动选择合适的能量指标列

### 6. Phase-Aware DVFS 控制器 (核心突破)

#### 控制器架构
- **6种策略对比**: default, max_perf, energy_efficient, phase_aware, adaptive_phase_aware, oracle
- **Phase切换**: prefill用高GPU频率, decode用高EMC频率
- **自适应决策**: `adaptive_phase_aware` 根据workload特征决定是否启用phase切换
- **切换开销**: 自动估算频率切换的时间和能量成本

#### 验证实验结果

| 策略 | 能量/Token (J) | TTFT (ms) | TPOT (ms) | SLO 满足率 |
|------|---------------|-----------|-----------|-----------|
| Default (mid) | 0.164 | 121 | 7.6 | 82% |
| Max Performance | 0.110 | 124 | 4.5 | 42% |
| Energy Efficient | 0.326 | 125 | 16.9 | 92% |
| **Phase-Aware (Ours)** | **0.115** | **77** | **6.7** | **80%** |
| Oracle | 0.152 | 124 | 6.9 | 76% |

#### 关键发现
- **Phase-Aware vs Default**: 节省 **30% 能量**
- **TTFT 改善 36%**: 高GPU频率加速prefill阶段
- **SLO稳定性**: 维持80%满足率 (Max Perf仅42%)
- **接近Max Perf**: 仅多0.0056 J/tok，但TTFT更好

### 7. Phase 5 评估可信度修复 (P0)

| 修复项 | 问题 | 解决方案 | 验证 |
|--------|------|----------|------|
| P0.1 fixed_best/oracle/regret | 跨phase能量列污染, 负regret异常 | 按phase分组fixed_best, per-bucket查找 | 0个anomaly, 0%负regret |
| P0.2 Phase-specific energy | 单一energy_per_token列 | 添加input/output/total三列 + energy_objective_used | selector自动选phase列 |
| P0.3 Pareto dominance | 未要求严格Pareto | all_not_worse + any_strictly_better | 无对称dominance |
| P3.1 Adaptive policy | controller忽略policy决策 | 连接should_enable_phase_aware() | adaptive_phase_aware策略 |

### 8. Jetson Baseline 基础设施 (P1/P2)

| 新文件 | 功能 | 状态 |
|--------|------|------|
| configs/baselines.yaml | 9种baseline定义 (nvpmodel/jetson_clocks/手动/自适应) | 完成 |
| configs/real_model_workloads.yaml | 5种workload (short_short ~ long_long) | 完成 |
| src/jetson_power_modes.py | nvpmodel/jetson_clocks 状态管理 | 完成 |
| src/llama_cpp_runner.py | llama.cpp 推理 + 每token计时 | 完成 |
| src/run_baseline_comparison.py | Baseline对比实验编排器 | 完成 |

---

## 关键洞察

### 1. Phase-Aware DVFS 的核心价值
Prefill是计算密集型（高GPU频率最优），Decode是内存密集型（高EMC频率最优）。**26.97x的效率差异**是Phase-Aware策略的理论基础。

### 2. 性能瓶颈在内存带宽
GPU频率4.25x提升仅带来8.3%性能改善，说明LLM推理在Jetson上主要受内存带宽限制，而非计算能力。

### 3. 评估可信度
修正后的评估显示 oracle regret = 0%, 我们的SLO-aware selector regret = 0%, MaxN regret = 8.21%。之前观察到的负regret是跨phase能量列混用导致的假象。

---

## 项目进度评估

### Phase 1: Project Foundation (100%)
### Phase 2: Core Module Development (100%)
### Phase 3: Preliminary Experiments (100%)
### Phase 4: Phase-Aware DVFS (100%)
### Phase 5: Advanced Optimization (60%)

**总体进度**: 约 **88%** 完成

---

## 下一步行动计划

### 短期 (1周内)
1. **运行真实模型baseline对比** - 5 workloads x 9 baselines x 5 repeats
2. **重建real rate table** - 用llama.cpp数据替代synthetic数据
3. **Re-evaluate selector** - 对比synthetic vs real结论

### 中期 (2-4周)
4. **混合智能调度** - 规则基础 + ML增强
5. **在线控制器部署** - 实时自适应频率控制
6. **长时间稳定性测试** - 验证策略鲁棒性

---

## 项目文件总览

### 核心源代码 (src/)
| 文件 | 功能 | 状态 |
|------|------|------|
| freq_controller.py | 频率控制 | 完成 |
| metrics_collector.py | 指标采集 | 完成 |
| benchmark_runner.py | 基准测试 | 完成 |
| sweep_runner.py | 扫描编排 | 完成 |
| synthetic_benchmark.py | 合成测试 | 完成 |
| build_rate_table.py | 汇率表构建 | 完成 (P0.2+P0.3) |
| select_config.py | 配置选择器 | 完成 (P0.2) |
| phase_aware_policy.py | Phase策略 | 完成 (P0.2b+P3.1) |
| phase_controller.py | Phase控制器 | 完成 (P3.1) |
| run_phase_aware_experiment.py | 验证实验 | 完成 |
| visualize_phase_aware.py | 可视化 | 完成 |
| visualize_phase5_p0.py | Phase5对比图 | 完成 |
| evaluate_selector.py | 选择器评估 | 完成 (P0.1) |
| jetson_power_modes.py | Jetson电源模式 | 完成 (P1.1) |
| llama_cpp_runner.py | llama.cpp推理 | 完成 (P2.2) |
| run_baseline_comparison.py | Baseline对比 | 完成 (P1.3+P2.3) |
| experiment_manager.py | 实验管理 | 完成 |
| system_monitor.py | 系统监控 | 完成 |

### 配置文件 (configs/)
| 文件 | 功能 | 状态 |
|------|------|------|
| baselines.yaml | 9种Jetson baseline定义 | 完成 |
| real_model_workloads.yaml | 5种real model workload | 完成 |
| selector.yaml | 选择器参数和SLO | 完成 |
| platform.yaml | 平台配置 | 完成 |

### 数据文件 (data/)
| 目录 | 内容 |
|------|------|
| rate_tables/ | 能耗汇率表 (Parquet, 23 configs) |
| selector_eval/ | 选择器评估结果 (P0 verified) |
| experiments_4_1_to_4_7/ | 7个前置实验结果 |
| phase_aware_experiment/ | Phase-Aware验证数据 |
| analysis/ | 分析摘要 |
| frequency_experiment_results/ | 频率实验原始数据 |

### 可视化 (figures/)
| 目录 | 内容 |
|------|------|
| phase5_comparison/ | Phase5综合对比图表 (6张) |
| phase_aware_experiment/ | 策略对比图表 (5张) |
| phase3_analysis/ | 选择器评估图表 (2张) |
| frequency_experiments/ | 频率实验图表 (4张) |
| advanced_analysis/ | 高级分析图表 (5张) |

---

**下一个里程碑**: 运行真实模型baseline对比实验 (5 x 9 x 5 = 225 runs)

**预计完成时间**: 1-2周
