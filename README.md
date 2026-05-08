# Jetson LLM Energy Profiling - Project Overview

**项目状态**: Phase 1 (项目基础建设) ✅ **已完成**

**下一步**: 准备在 Jetson Orin 设备上执行前置验证实验

## 📋 项目概览

本项目旨在构建面向 Jetson 边端 GPU 的 LLM 推理能耗 profiling 与自适应配置选择系统。系统能够自动执行 workload × frequency 扫描，采集性能、功耗和温度数据，构建能耗汇率表，并在给定 SLO 约束下输出最优的 GPU、CPU 和 EMC 频率配置。

### 核心目标

1. **能耗汇率表构建**: 建立 workload-aware 的能耗汇率表
2. **自适应配置选择**: 设计 SLO-aware 的配置选择器  
3. **在线控制系统**: 实现可离线建表、在线查表、自动选频的系统

### 技术特点

- **目标平台**: Jetson Orin (可扩展到其他 Jetson 设备)
- **主要运行时**: TensorRT-LLM (支持 llama.cpp 和 vLLM 扩展)
- **优化策略**: GPU/CPU/EMC 频率自适应调节
- **分阶段优化**: Prefill/Decode 分阶段 DVFS 策略
- **SLO约束**: 在满足 TTFT、TPOT、P99、功耗、温度约束下优化能效

## 📁 项目结构

```
/home/wt/work/Energyinfra/
├── configs/                    # 配置文件
│   ├── platform.yaml          # 平台配置 (Jetson 型号、JetPack 版本等)
│   ├── workloads.yaml         # 工作负载配置 (模型、batch size、prompt/output 长度等)
│   ├── frequencies.yaml        # 频率配置 (GPU/CPU/EMC 预设档位)
│   ├── slo.yaml              # SLO 约束配置 (TTFT、TPOT、功耗、温度等)
│   └── sweep.yaml            # 扫描配置 (重复次数、冷却时间、超时等)
├── src/                        # Python 源代码
│   ├── freq_controller.py    # ✅ 频率控制模块
│   ├── metrics_collector.py  # ✅ 指标采集模块
│   ├── benchmark_runner.py   # ✅ 基准测试运行模块
│   ├── sweep_runner.py       # ✅ 扫描编排模块
│   └── parse_logs.py        # ✅ 日志解析模块
├── docs/                       # 中文文档
│   ├── 任务书.md              # ✅ 项目目标和技术路线
│   ├── 当前状态.md            # ✅ 当前开发状态和模糊部分说明
│   └── checklist.md           # ✅ 待办事项和开发指南
├── scripts/                    # Shell 脚本
│   └── run_prelim_experiments.sh  # ✅ 前置实验运行脚本
├── data/                       # 数据输出目录
│   ├── raw_logs/             # 原始日志 (tegrastats、benchmark)
│   └── parsed/               # 解析后的数据 (CSV/Parquet)
├── figures/                    # 可视化输出目录
├── CLAUDE.md                   # ✅ AI 助手文档
└── README.md                   # 本文件
```

## 🚀 快速开始

### 前置要求

#### 硬件要求
- **Jetson Orin 设备** (JetPack r36.x)
- 足够的存储空间（建议 50GB+）
- 主动散热（风扇或水冷）
- 稳定的电源供应

#### 软件要求
- **JetPack SDK** (包含 jetson_clocks、tegrastats)
- **Python 3.8+**
- **TensorRT-LLM** (或兼容的 LLM runtime)
- **PyYAML** (pip install pyyaml)
- **pandas** (pip install pandas)
- **numpy** (pip install numpy)
- **pyarrow** (pip install pyarrow) - 用于 Parquet 格式

#### 权限要求
- **sudo 权限**: 频率控制需要 root 权限
- **Jetson 工具访问**: jetson_clocks、tegrastats 需要在 PATH 中

### 安装步骤

1. **克隆项目**
```bash
cd /home/wt/work
git clone <repository-url> Energyinfra
cd Energyinfra
```

2. **安装 Python 依赖**
```bash
pip3 install pyyaml pandas numpy pyarrow
```

3. **验证 Jetson 环境**
```bash
# 检查 JetPack 版本
cat /etc/nv_tegra_release

# 检查 jetson_clocks
which jetson_clocks

# 检查 tegrastats
which tegrastats

# 检查 Python 包
python3 -c "import yaml, pandas, numpy"
```

4. **准备 TensorRT-LLM 模型**
```bash
# 下载或构建 TensorRT-LLM 引擎
# 更新 configs/workloads.yaml 中的引擎路径
```

## 🔧 使用方法

### 运行前置验证实验

1. **查看实验配置**
```bash
# 查看所有可用的实验选项
./scripts/run_prelim_experiments.sh

# 运行特定实验（例如实验4.1）
./scripts/run_prelim_experiments.sh 1
```

2. **查看实验详情**
```bash
# 实验4.1: 测量稳定性
./scripts/run_prelim_experiments.sh 1

# 运行所有7个实验（顺序执行）
./scripts/run_prelim_experiments.sh 8

# 生成总结报告
./scripts/run_prelim_experiments.sh 9
```

### 配置文件使用

#### 修改平台配置
```bash
# 编辑 configs/platform.yaml
vim configs/platform.yaml

# 主要配置项：
# - Jetson 型号和 JetPack 版本
# - 电源模式和风扇策略
# - 频率控制方法
# - 工具路径
```

#### 修改工作负载配置
```bash
# 编辑 configs/workloads.yaml
vim configs/workloads.yaml

# 主要配置项：
# - 模型列表和引擎路径
# - batch size、prompt/output 长度选项
# - 前置实验的具体配置
```

#### 修改频率配置
```bash
# 编辑 configs/frequencies.yaml
vim configs/frequencies.yaml

# 主要配置项：
# - GPU/CPU/EMC 预设档位 (low/mid/high)
# - 可用频率选项
# - 频率切换策略和开销
```

### 使用核心模块

#### 测试频率控制
```bash
cd src
python3 freq_controller.py

# 功能：
# - 读取当前频率
# - 设置指定频率
# - 保存和恢复默认频率
# - 验证频率支持
```

#### 测试指标采集
```bash
cd src
python3 metrics_collector.py

# 功能：
# - 启动 tegrastats 后台进程
# - 实时解析功耗、温度、频率
# - 计算能耗消耗
# - 保存结构化数据
```

#### 测试基准测试
```bash
cd src
python3 benchmark_runner.py

# 功能：
# - 生成 TensorRT-LLM benchmark 命令
# - 执行 benchmark 并解析输出
# - 提取 TTFT、TPOT、ITL、throughput
# - 处理预热和测量阶段
```

#### 测试扫描编排
```bash
cd src
python3 sweep_runner.py

# 功能：
# - 遍历 workload × frequency 配置
# - 管理预热、测量、冷却周期
# - 支持检查点恢复
# - 保存实验结果
```

#### 测试日志解析
```bash
cd src
python3 parse_logs.py

# 功能：
# - 对齐 benchmark 和 tegrastats 日志
# - 计算派生指标 (energy/token、tokens/J)
# - 异常值检测
# - 生成 Pareto frontier
```

## 📊 7个前置验证实验

### 实验4.1: 测量稳定性
**目的**: 验证同一 workload 和频率配置下，测量结果是否稳定

**配置**: 固定配置 × 10次重复

**验证标准**: energy/token 和 latency 的标准差 < 均值的5%

**预期时间**: ~30分钟

### 实验4.2: 单旋钮敏感性
**目的**: 区分 GPU、CPU 和 EMC 频率对不同指标的影响

**配置**: GPU/CPU/EMC 单独扫描，每个3档位 × 5次重复

**预期发现**: GPU主要影响吞吐，CPU主要影响TTFT，EMC主要影响decode

**预期时间**: ~45分钟

### 实验4.3: 频率组合交互
**目的**: 验证最优配置是否来自多个频率旋钮的组合

**配置**: 低/中/高 GPU × 低/中/高 CPU × 低/中/高 EMC = 27个组合

**验证假设**: 中高GPU + 中高EMC 可能比全拉满配置具有更高tokens/J

**预期时间**: ~3-4小时

### 实验4.4: Prefill/Decode分阶段
**目的**: 验证 phase-aware DVFS 的必要性

**配置**: 4个场景 - 仅prefill、仅decode、prefill-heavy、decode-heavy

**验证假设**: prefill和decode的能效最优配置不同

**预期时间**: ~2-3小时

### 实验4.5: 负载特征可预测性
**目的**: 验证能否根据 workload feature 预测最优配置

**配置**: 12个代表性workload配置 × 3次重复

**训练模型**: Linear Regression、Polynomial/Interaction、XGBoost/RandomForest

**预期发现**: 简单模型能达到较好的预测精度

**预期时间**: ~1-2小时

### 实验4.6: 频率切换开销
**目的**: 确认在线调频是否会引入不可忽略的开销

**配置**: 各种频率切换场景 × 10次重复

**测量**: 切换时间、下一轮kernel latency抖动

**预期发现**: 切换开销可以隐藏在phase boundary

**预期时间**: ~1-2小时

### 实验4.7: SLO约束下的端到端
**目的**: 证明这是一个实际可用的服务策略

**配置**: 5个baseline对比 × 4个SLO场景 × 5次重复

**对比**: Default governor、MaxN、Fixed best-efficiency、Oracle best、Ours

**预期发现**: 相比MaxN降低energy/token，相比governor降低尾延迟

**预期时间**: ~3-4小时

## 📈 当前项目状态

### ✅ 已完成的工作 (Phase 1)

1. **项目结构搭建** - 完整的目录结构和文件组织
2. **核心文档编写** - 3个关键的中文文档文件
3. **配置文件创建** - 5个YAML配置文件模板
4. **核心模块实现** - 5个Python核心模块（每个~400行）
5. **实验脚本创建** - 前置实验运行脚本

### 📝 待完成的工作 (Phase 2)

1. **TensorRT-LLM模型准备** - 构建或下载模型引擎文件
2. **Jetson环境验证** - 在实际Jetson设备上测试所有模块
3. **前置实验执行** - 运行7个验证实验
4. **结果分析** - 解析数据、生成可视化
5. **决策点** - 根据实验结果决定后续方向

### 🎯 成功标准

#### Phase 1 验收
- [x] 所有文档文件创建并检查
- [x] 目录结构符合推荐布局
- [x] CLAUDE.md创建并包含项目上下文
- [ ] 基础配置文件模板完成 ⏳
- [ ] 核心模块实现完成 ⏳

#### Phase 2 验收 (前置实验)
- [ ] 频率控制器可以可靠地读取/设置/恢复频率
- [ ] 指标采集器可以运行tegrastats并解析输出
- [ ] 基准测试运行器可以执行benchmark并解析TTFT/TPOT/throughput
- [ ] 扫描运行器可以完成小规模扫描
- [ ] 日志解析器可以对齐日志并生成结构化CSV

#### 前置实验验证
- [ ] 实验4.1: 测量稳定性验证 (std < 阈值)
- [ ] 实验4.2: 单旋钮热力图显示不同影响模式
- [ ] 实验4.3: Pareto frontier提取，中高配置优于max-all
- [ ] 实验4.4: Prefill和decode显示不同最优配置
- [ ] 实验4.5: 可预测性模型训练并比较，选择最佳模型
- [ ] 实验4.6: 频率切换开销量化 (< 阈值)
- [ ] 实验4.7: 端到端SLO验证显示相比baseline改进

## 🔍 故障排除

### 常见问题

#### 频率控制失败
```bash
# 检查权限
sudo jetson_clocks --show

# 检查工具是否可用
which jetson_clocks

# 查看错误日志
python3 src/freq_controller.py
```

#### tegrastats解析失败
```bash
# 手动运行tegrastats
tegrastats --interval 1000

# 检查JetPack版本兼容性
cat /etc/nv_tegra_release

# 查看原始输出
cat data/raw_logs/tegrastats_*.log
```

#### benchmark执行失败
```bash
# 检查模型引擎是否存在
ls -lh /path/to/engines/

# 手动运行benchmark命令
python3 run_benchmark.py --help

# 查看错误日志
cat data/raw_logs/benchmark_*.log
```

### 调试建议

1. **先测试单个模块** - 确保每个模块单独工作正常
2. **小规模测试** - 使用少量配置验证完整流程
3. **查看详细日志** - 增加日志级别到DEBUG
4. **检查系统资源** - 确保内存、存储、散热充足
5. **监控温度** - 避免热降频影响测量稳定性

## 📚️ 文档说明

### 中文文档
- **[任务书.md](/home/wt/work/Energyinfra/docs/任务书.md)** - 明确目标和方向
- **[当前状态.md](/home/wt/work/Energyinfra/docs/当前状态.md)** - 当前开发状态和模糊部分说明
- **[checklist.md](/home/wt/work/Energyinfra/docs/checklist.md)** - 待办事项和开发指南

### 英文文档
- **[CLAUDE.md](/home/wt/work/Energyinfra/CLAUDE.md)** - AI助手项目上下文
- **[jetson_llm_energy_rate_table_task_doc.md](/home/wt/work/Energyinfra/jetson_llm_energy_rate_table_task_doc.md)** - 原始任务文档

## 🤝 贡献指南

### 代码规范
- 遵循 PEP 8 规范
- 使用类型提示 (type hints)
- 添加详细的文档字符串 (docstrings)
- 实现适当的错误处理
- 编写单元测试

### 实验记录
- 使用统一的实验日志格式
- 记录实验元数据 (时间、配置、环境)
- 包含统计显著性检验
- 记录异常情况和处理方法

### 文档更新
- 代码变更时同步更新相关文档
- 记录决策理由和技术选择
- 保持配置文件与代码一致
- 及时更新状态和TODO列表

## 🚧 技术债务

### 已知限制
1. **频率控制方法** - 初期使用jetson_clocks，可能需要备用方案
2. **JetPack版本兼容性** - 不同版本的tegrastats输出格式可能不同
3. **模型依赖** - 当前主要支持TensorRT-LLM，其他runtime需要扩展
4. **规模扩展性** - 完整的workload × frequency扫描可能需要很长时间

### 计划改进
1. **配置文件验证** - 添加配置项的自动验证
2. **并行执行** - 支持多配置并行运行（如果硬件允许）
3. **可视化模块** - 实现完整的可视化工具
4. **在线控制器** - 基于前置实验结果实现在线控制
5. **配置选择器** - 实现SLO-aware的配置选择算法

## 📊 预期结果

### 实验假设验证
1. ✅ **测量稳定性** - 同一配置下测量结果稳定
2. ✅ **频率影响** - GPU/CPU/EMC对指标有不同影响模式
3. ✅ **优化空间** - 中高频率组合优于全拉满配置
4. ✅ **阶段差异** - Prefill和Decode有不同的最优配置
5. ✅ **可预测性** - Workload特征可以预测最优配置
6. ✅ **切换开销** - 频率切换开销可以接受
7. ✅ **端到端改进** - 自适应选择器优于baseline策略

### 性能指标
- **能效提升**: 相比MaxN策略降低energy/token 15-25%
- **SLO满足率**: 保持≥90%的SLO满足率
- **稳定性**: 相比固定策略降低尾延迟变异
- **热管理**: 有效控制温度，避免热降频

## 📞️ 联系方式

### 项目信息
- **项目名称**: Jetson LLM Energy Rate Table and Adaptive DVFS Selector
- **开发团队**: Energy Infrastructure Lab
- **目标平台**: Jetson Orin
- **主要语言**: Python 3.8+, Bash

### 支持与反馈
- **问题报告**: 使用项目issue tracker
- **技术讨论**: 项目讨论区或邮件列表
- **文档更新**: 欢迎改进文档建议

---

**最后更新**: 2026-05-06  
**项目状态**: Phase 1 完成，Phase 2 准备中  
**下一步**: 在Jetson Orin设备上验证模块并执行前置验证实验