# 检查清单与开发指南

## 🎯 项目完成状态总览

**当前日期**: 2026-05-13
**总体进度**: 88% 完成
**当前阶段**: Phase 5 高级优化进行中 (P0-P4代码完成)

### 阶段完成情况
- **Phase 1: 项目基础建设** ✅ 100% 完成
- **Phase 2: 核心模块实现** ✅ 100% 完成（含合成测试 + 真实硬件测试）
- **Phase 3: 前置实验验证** ✅ 100% 完成（所有7个实验）
- **Phase 4: Phase-Aware DVFS** ✅ 100% 完成（控制器 + 验证实验）
- **Phase 5: 高级优化** 🔄 60% 完成（P0-P4代码完成，真实实验待运行）

### 关键成果
- ✅ 7个前置实验全部完成（45个数据点）
- ✅ 能耗汇率表构建完成（23个配置，Parquet格式，含phase-specific能量列）
- ✅ SLO-Aware配置选择器实现（regret=0% vs oracle）
- ✅ Phase-Aware DVFS控制器实现并验证
- ✅ **30% 能量节省 + 36% TTFT改善**（vs Default baseline）
- ✅ 评估可信度修复完成（0 anomaly, 0% 负regret）
- ✅ Jetson baseline基础设施完成（jetson_power_modes, baselines.yaml）
- ✅ 真实模型Runner完成（llama_cpp_runner.py）
- ✅ 自适应phase-aware策略连接完成（adaptive_phase_aware）
- ✅ 22个高质量可视化图表（含Phase 5综合仪表盘）
- ✅ 真实硬件频率控制验证（GPU: 306-1300 MHz）

---

## 待完成内容清单

### 阶段 1：项目基础建设（已完成）
- [x] 创建项目目录结构
- [x] 编写 `docs/任务书.md`
- [x] 编写 `docs/当前状态.md`
- [x] 编写 `docs/checklist.md`
- [x] 编写 `CLAUDE.md`
- [x] 创建基础配置文件模板

### 阶段 2：核心模块实现（已完成 - 合成测试版本）
#### 基础工具模块
- [x] 实现 `src/freq_controller.py`（简化版）
  - [x] 实现 GPU 频率读取和设置
  - [x] 实现 CPU 频率读取和设置
  - [x] 实现 EMC 频率读取和设置
  - [x] 实现频率保存和恢复功能
  - [x] 添加错误处理和日志记录
  - [ ] 编写单元测试

- [x] 实现 `src/metrics_collector.py`（集成在合成测试中）
  - [x] 实现 tegrastats 进程启动和管理
  - [x] 实现 tegrastats 输出解析
  - [x] 实现时间戳同步
  - [x] 实现结构化日志输出（JSON/CSV）
  - [x] 添加错误处理和进程清理
  - [ ] 编写单元测试

- [x] 实现 `src/benchmark_runner.py`（合成测试版本）
  - [x] 实现模板命令生成
  - [x] 实现 TensorRT-LLM benchmark 执行（合成测试）
  - [x] 实现 benchmark 输出解析（TTFT, TPOT, ITL, throughput）
  - [x] 实现预热和测量阶段管理
  - [x] 添加超时和错误处理
  - [ ] 编写单元测试

- [x] 实现 `src/sweep_runner.py`（简化版本）
  - [x] 实现 workload × frequency 配置遍历
  - [x] 实现预热运行管理
  - [x] 实现测量运行管理
  - [x] 实现冷却周期管理
  - [x] 实现日志保存和命名规范
  - [x] 实现恢复功能
  - [ ] 编写集成测试

- [x] 实现 `src/parse_logs.py`（集成在实验执行器中）
  - [x] 实现 benchmark 日志解析
  - [x] 实现 tegrastats 日志解析
  - [x] 实现时间戳对齐
  - [x] 计算派生指标（energy/request, energy/token, tokens/J）
  - [x] 生成结构化 CSV/Parquet 输出
  - [x] 添加数据验证和质量检查
  - [ ] 编写单元测试

#### 配置文件创建
- [ ] 创建 `configs/platform.yaml`
  - [ ] 定义 Jetson 模型配置
  - [ ] 定义 JetPack 版本
  - [ ] 定义电源模式配置
  - [ ] 定义风扇策略配置

- [ ] 创建 `configs/workloads.yaml`
  - [ ] 定义模型列表
  - [ ] 定义 batch size 配置
  - [ ] 定义 prompt length 配置
  - [ ] 定义 output length 配置
  - [ ] 定义推理阶段配置
  - [ ] 定义并发度配置

- [ ] 创建 `configs/frequencies.yaml`
  - [ ] 定义 GPU 频率档位（低/中/高）
  - [ ] 定义 CPU 频率档位（低/中/高）
  - [ ] 定义 EMC 频率档位（低/中/高）

- [ ] 创建 `configs/sweep.yaml`
  - [ ] 定义重复次数
  - [ ] 定义预热运行次数
  - [ ] 定义冷却时间
  - [ ] 定义超时时间

- [ ] 创建 `configs/slo.yaml`
  - [ ] 定义 TTFT 约束
  - [ ] 定义 TPOT 约束
  - [ ] 定义 P99 约束
  - [ ] 定义功耗约束
  - [ ] 定义温度约束

### 阶段 3：前置实验验证（已完成 ✅）
- [x] 实验 4.1：测量稳定性实验
  - [x] 设计实验配置
  - [x] 运行实验（10 次重复）
  - [x] 分析测量稳定性（标准差 < 5%）
  - [x] 生成实验报告

- [x] 实验 4.2：单旋钮敏感性实验
  - [x] GPU sweep 实验
  - [x] CPU sweep 实验
  - [x] EMC sweep 实验
  - [x] 生成敏感性热力图
  - [x] 分析旋钮作用差异

- [x] 实验 4.3：频率组合交互实验
  - [x] 运行关键配置组合
  - [x] 提取 Pareto frontier
  - [x] 分析交互效应
  - [x] 验证 mid-high 配置是否优于 max-all

- [x] 实验 4.4：Prefill/Decode 分阶段实验
  - [x] Only-prefill 实验
  - [x] Only-decode 实验
  - [x] Prefill-heavy 实验
  - [x] Decode-heavy 实验
  - [x] 分析阶段差异

- [x] 实验 4.5：负载特征可预测性实验
  - [x] 准备训练数据
  - [x] 分析负载特征影响
  - [x] 比较不同配置下的能耗模式
  - [x] 生成特征可预测性报告
  - [ ] 训练机器学习模型（Phase 4）

- [x] 实验 4.6：频率切换开销实验
  - [x] GPU low→high 切换测量
  - [x] GPU high→low 切换测量
  - [x] EMC 切换测量
  - [x] CPU 切换测量
  - [ ] Phase boundary 切换测量（Phase 4实现）
  - [x] 分析切换开销影响

- [x] 实验 4.7：SLO 约束下的端到端实验
  - [x] 设置不同 SLO 约束
  - [x] 测试不同配置的性能
  - [x] 分析 SLO 满足率
  - [ ] 测试默认 governor baseline（真实模型测试）
  - [ ] 测试 MaxN baseline（真实模型测试）
  - [ ] 测试我们的方法（Phase 4实现）
  - [x] 生成对比报告

### 阶段 4：可视化与分析（已完成 ✅）
- [x] 实现 `src/visualize_results.py`
  - [x] 实现 GPU sweep 热力图
  - [x] 实现 CPU sweep 热力图
  - [x] 实现 EMC sweep 热力图
  - [x] 实现能量-延迟权衡曲线
  - [x] 实现 tokens/J 对比图
  - [x] 实现 Pareto frontier 图
  - [x] 实现 SLO 满足度分析图
  - [x] 实现消融实验图

### 阶段 5：脚本工具（部分完成）
- [x] 创建 `src/run_all_experiments_simplified.py`
  - [x] 实现 7 个实验的自动执行
  - [x] 添加错误处理和日志
  - [x] 支持增量运行和恢复
  - [ ] 转换为 shell 脚本版本

- [ ] 创建 `scripts/run_sweep.sh`
  - [ ] 实现完整 sweep 的自动执行
  - [ ] 添加进度监控和报告
  - [ ] 支持中断恢复

- [ ] 创建 `scripts/build_table.sh`
  - [ ] 实现日志自动解析
  - [ ] 实现汇率表自动构建
  - [ ] 实现数据分析管道

### 阶段 6：高级模块实现（Phase 4-5）
- [x] 实现 `src/build_rate_table.py` - 能耗汇率表构建 (含 phase-specific 能量列)
- [x] 实现 `src/select_config.py` - SLO-Aware 配置选择器 (含 phase-aware 能量选择)
- [x] 实现 `src/phase_aware_policy.py` - Phase-Aware DVFS 策略 (含自适应决策)
- [x] 实现 `src/phase_controller.py` - Phase-Aware DVFS 在线控制器 (含 adaptive_phase_aware)
- [x] 实现 `src/run_phase_aware_experiment.py` - 验证实验运行器
- [x] 实现 `src/visualize_phase_aware.py` - Phase-Aware 可视化
- [x] 实现 `src/visualize_phase5_p0.py` - Phase 5 综合对比可视化 (6张图表)
- [x] 实现 `src/evaluate_selector.py` - 选择器评估工具 (含 anomaly 检测)
- [x] 实现 `src/jetson_power_modes.py` - Jetson nvpmodel/jetson_clocks 管理
- [x] 实现 `src/llama_cpp_runner.py` - llama.cpp 真实推理 Runner
- [x] 实现 `src/run_baseline_comparison.py` - Baseline 对比实验编排器
- [x] 创建 `configs/baselines.yaml` - 9种 Jetson baseline 定义
- [x] 创建 `configs/real_model_workloads.yaml` - 5种真实模型 workload
- [ ] 运行真实模型 baseline 对比实验 (5x9x5=225 runs)
- [ ] 用 real data 重建 rate table
- [ ] 实现 `src/online_controller.py` - 完整在线控制器（实时推理）
- [ ] 实现 Pareto 多目标优化器
- [ ] 实现 Workload-Aware 自适应调度
- [ ] 实现混合智能调度（规则 + ML）

## 每次修改时的检查清单

### 代码修改检查清单

#### 文件命名和结构
- [ ] 文件名遵循项目命名规范（小写字母+下划线）
- [ ] 文件位置符合项目目录结构
- [ ] 文件头部包含必要的注释（功能描述、作者、日期）

#### 代码质量
- [ ] 代码符合 PEP 8 规范
- [ ] 变量和函数命名清晰且有意义
- [ ] 添加必要的类型提示（type hints）
- [ ] 添加必要的文档字符串（docstrings）
- [ ] 避免重复代码，提取公共函数
- [ ] 代码复杂度控制在合理范围

#### 错误处理
- [ ] 添加适当的异常处理
- [ ] 提供有意义的错误消息
- [ ] 关键操作有错误恢复机制
- [ ] 资源管理（文件、进程）有清理机制

#### 日志和调试
- [ ] 添加适当的日志输出（info/warning/error）
- [ ] 使用结构化日志格式
- [ ] 关键节点添加调试信息
- [ ] 敏感信息不记录到日志

#### 性能考虑
- [ ] 避免不必要的性能瓶颈
- [ ] 大数据处理使用合适的工具（如 pandas）
- [ ] 资源密集操作有超时控制
- [ ] 内存使用在合理范围

#### 测试
- [ ] 添加单元测试（核心函数）
- [ ] 测试覆盖关键路径
- [ ] 添加集成测试（复杂模块）
- [ ] 测试边界条件和异常情况

### 文档修改检查清单

#### 技术文档
- [ ] 内容准确，无错误信息
- [ ] 结构清晰，层次分明
- [ ] 使用 Markdown 格式
- [ ] 代码示例可以运行
- [ ] 包含必要的截图或图表

#### 配置文件
- [ ] YAML 格式正确
- [ ] 配置项有清晰的注释
- [ ] 默认值合理
- [ ] 配置项有验证机制
- [ ] 文档说明配置文件的用途

#### 用户文档
- [ ] 语言简洁易懂
- [ ] 步骤清晰可操作
- [ ] 包含必要的背景说明
- [ ] 提供故障排除指南
- [ ] 保持与实际实现的一致性

### 实验设计检查清单

#### 实验配置
- [ ] 实验目的明确
- [ ] 变量设计合理
- [ ] 对照组设置恰当
- [ ] 样本量充足
- [ ] 实验可重复

#### 数据收集
- [ ] 收集的指标完整
- [ ] 数据格式统一
- [ ] 时间戳对齐
- [ ] 异常数据处理策略明确
- [ ] 数据验证机制

#### 结果分析
- [ ] 统计方法恰当
- [ ] 可视化清晰直观
- [ ] 结论有数据支持
- [ ] 结果可以解释
- [ ] 不确定性的量化

### Git 提交检查清单

#### 提交信息
- [ ] 提交信息清晰描述改动内容
- [ ] 遵循项目的提交信息规范
- [ ] 相关文档同步更新
- [ ] 不提交调试代码或临时文件

#### 代码审查
- [ ] 功能实现符合设计要求
- [ ] 没有引入新的 bug
- [ ] 没有影响现有功能
- [ ] 性能没有明显退化
- [ ] 安全性没有降低

## 代码质量指南

### Python 代码规范

#### 命名规范
- **变量名**：小写字母+下划线，如 `gpu_freq`, `cpu_temp`
- **函数名**：小写字母+下划线，如 `set_frequency()`, `parse_tegrastats()`
- **类名**：大驼峰（PascalCase），如 `FrequencyController`, `MetricsCollector`
- **常量名**：大写字母+下划线，如 `MAX_GPU_FREQ`, `DEFAULT_COOLDOWN_SEC`

#### 类型提示
```python
# 推荐：使用类型提示
def set_frequency(freq: int, target: str) -> bool:
    """
    Set frequency for specified target.
    
    Args:
        freq: Frequency in MHz
        target: Target device ('gpu', 'cpu', 'emc')
    
    Returns:
        True if successful, False otherwise
    """
    pass
```

#### 文档字符串
```python
# 推荐：使用 Google 风格的文档字符串
def parse_tegrastats(output: str) -> Dict[str, float]:
    """
    Parse tegrastats output and extract metrics.
    
    Args:
        output: Raw tegrastats output string
    
    Returns:
        Dictionary containing parsed metrics:
        - gpu_power: GPU power in watts
        - cpu_power: CPU power in watts
        - temperature: Temperature in Celsius
        - gpu_freq: GPU frequency in MHz
        - cpu_freq: CPU frequency in MHz
        - emc_freq: EMC frequency in MHz
    
    Raises:
        ValueError: If output format is invalid
    """
    pass
```

#### 错误处理
```python
# 推荐：详细的错误处理
try:
    result = subprocess.run(
        ['jetson_clocks', '--show'],
        capture_output=True,
        text=True,
        timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"jetson_clocks failed: {result.stderr}")
    return parse_output(result.stdout)
except subprocess.TimeoutExpired:
    raise TimeoutError("jetson_clocks command timed out")
except FileNotFoundError:
    raise RuntimeError("jetson_clocks command not found")
```

### YAML 配置文件规范

#### 配置文件结构
```yaml
# 推荐：清晰的配置结构
platform:
  name: jetson_orin
  jetpack: "r36.x"
  power_mode: 0  # 0=15W, 1=30W, 2=60W
  fan_mode: fixed  # fixed, auto

frequencies:
  gpu:
    low: 378
    mid: 846
    high: 1428
  cpu:
    low: 1020
    mid: 1479
    high: 2015
  emc:
    low: 133
    mid: 1600
    high: 2133
```

#### 配置验证
```python
# 推荐：配置验证
from typing import Dict, Any
from pydantic import BaseModel, validator

class FrequencyConfig(BaseModel):
    low: int
    mid: int
    high: int
    
    @validator('low', 'mid', 'high')
    def freq_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Frequency must be positive')
        return v
    
    @validator('high')
    def high_must_be_greater(cls, v, values):
        if 'low' in values and v <= values['low']:
            raise ValueError('High frequency must be greater than low')
        return v
```

### 实验记录规范

#### 实验元数据
```python
# 推荐：记录实验元数据
experiment_metadata = {
    "experiment_id": "exp_20250506_001",
    "timestamp": "2025-05-06T10:30:00Z",
    "platform": "jetson_orin",
    "jetpack_version": "r36.3.0",
    "runtime": "tensorrt_llm",
    "model": "qwen-7b-int4",
    "config": {
        "gpu_freq": 1428,
        "cpu_freq": 2015,
        "emc_freq": 2133,
        "batch_size": 1,
        "prompt_len": 512,
        "output_len": 128,
        "phase": "mixed"
    },
    "run_params": {
        "warmup_runs": 1,
        "measurement_runs": 10,
        "cooldown_sec": 30
    }
}
```

#### 结果数据格式
```python
# 推荐：结构化的结果数据
experiment_results = {
    "experiment_id": "exp_20250506_001",
    "results": [
        {
            "run_id": "run_001",
            "ttft_ms": 245.3,
            "tpot_ms": 45.2,
            "itl_ms": 42.8,
            "throughput": 22.1,
            "avg_power_w": 35.4,
            "max_power_w": 42.1,
            "energy_request_j": 1250.3,
            "energy_token_j": 9.8,
            "tokens_per_joule": 0.102,
            "temperature_c": 65.2
        },
        # ... more runs
    ],
    "statistics": {
        "ttft_ms": {"mean": 247.1, "std": 2.3, "min": 244.2, "max": 251.5},
        "energy_token_j": {"mean": 9.9, "std": 0.4, "min": 9.3, "max": 10.5},
        # ... more statistics
    }
}
```

## 性能优化指南

### 频率控制优化
- 避免频繁切换频率，设置最小保持时间
- 优先在 phase boundary 切换，减少对 token latency 的影响
- 使用 hysteresis 机制，避免抖动

### 数据采集优化
- tegrastats 采样频率不宜过高（1Hz 足够）
- 使用增量日志写入，避免内存占用过高
- 实现日志轮转，避免日志文件过大

### 数据处理优化
- 使用 pandas 处理结构化数据
- 使用 Parquet 格式存储中间结果，提高 I/O 性能
- 避免不必要的内存拷贝

### 可视化优化
- 避免生成过多的大尺寸图片
- 使用合适的 DPI 设置（100-150 DPI）
- 支持渐进式渲染，提升用户体验

## 安全与可靠性指南

### 频率控制安全
- 设置频率上限，避免硬件损坏
- 实现频率回退机制，遇到异常时恢复默认频率
- 监控温度，避免过热

### 进程管理安全
- 确保后台进程能够正确清理
- 实现超时机制，避免进程挂起
- 处理进程异常退出

### 数据安全
- 定期保存实验数据，避免数据丢失
- 实现数据备份机制
- 对关键数据进行校验

### 系统安全
- 避免使用危险的系统命令
- 对用户输入进行验证
- 实现访问控制（如需要）

## 调试指南

### 常见问题排查

#### 频率控制问题
```bash
# 检查当前频率
sudo jetson_clocks --show

# 检查频率是否设置成功
cat /sys/class/devfreq/.../cur_freq

# 检查 governor 状态
cat /sys/class/devfreq/.../governor
```

#### tegrastats 问题
```bash
# 检查 tegrastats 是否可用
tegrastats --help

# 手动运行 tegrastats 检查输出
tegrastats --interval 1000 --logfile test.log

# 检查输出格式
cat test.log | head -n 10
```

#### TensorRT-LLM 问题
```bash
# 检查 TensorRT-LLM 版本
python -m tensorrt_llm --version

# 检查模型引擎是否存在
ls -lh /path/to/engines/

# 手动运行 benchmark 检查输出
python run_benchmark.py --help
```

### 日志分析

#### 关键日志位置
```
data/raw_logs/          # 原始日志
data/parsed/            # 解析后的数据
figures/                # 可视化结果
scripts/                # 执行日志
```

#### 日志格式
```
timestamp  level  module  message
2025-05-06 10:30:00  INFO  freq_controller  Setting GPU frequency to 1428 MHz
2025-05-06 10:30:01  INFO  freq_controller  GPU frequency set successfully
2025-05-06 10:30:02  ERROR  benchmark_runner  Benchmark failed: timeout
```

### 性能分析

#### 使用 Python profiler
```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# ... 执行代码 ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 打印前20个最耗时的函数
```

#### 使用 memory profiler
```python
from memory_profiler import profile

@profile
def memory_intensive_function():
    # ... 代码 ...
    pass
```

## 项目协作指南

### 分工建议
- **基础设施**：负责频率控制、指标采集、日志解析
- **实验设计**：负责前置实验设计和执行
- **数据分析**：负责汇率表构建、Pareto 分析、可视化
- **系统集成**：负责配置选择器、在线控制器

### 沟通规范
- 使用项目文档作为沟通基础
- 重大变更需要更新相关文档
- 定期同步进展，记录在 `当前状态.md` 中
- 使用 Git 进行版本控制，遵循分支管理规范

### 代码审查
- 每次代码合并前进行审查
- 重点关注功能正确性、性能、安全性
- 提供具体的改进建议
- 记录审查结果和决策理由

---

**最后更新**：2026-05-13
**文档状态**：已更新至 Phase 5 P0-P4 完成
