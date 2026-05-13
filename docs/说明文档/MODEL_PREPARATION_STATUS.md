# 模型准备状态文档
# Model Preparation Status

**最后更新**: 2026-05-06  
**项目阶段**: 环境配置和模型准备

## 🔍 当前状态

### ✅ 环境配置完成情况
- **Python 虚拟环境**: ✅ 已创建 `/home/wt/work/Energyinfra/jetson_llm_env`
- **依赖包安装**: ✅ 已安装 pandas, matplotlib, seaborn, scipy, scikit-learn, pyyaml, numpy
- **频率控制器**: ✅ 已修复，支持混合模式（sysfs读取 + sudo设置）
- **Jetson 设备**: ✅ Jetson Orin 设备检测成功

### ❌ 模型准备情况
- **TensorRT-LLM**: ❌ 未安装
- **模型引擎文件**: ❌ 路径为占位符，需要转换或下载
- **Docker 容器**: ❌ 需要sudo权限访问

## 📋 需要准备的模型

### 主要测试模型
根据配置文件，主要测试模型为：

1. **qwen-7b-int4** 
   - 参数量: 7B
   - 量化: INT4
   - 引擎路径: `/path/to/qwen-7b-int4.engine`
   - 上下文长度: 4096
   - 最大输出长度: 512

2. **llama-3-8b-fp16** (可选)
   - 参数量: 8B
   - 量化: FP16
   - 引擎路径: `/path/to/llama-3-8b-fp16.engine`
   - 上下文长度: 8192
   - 最大输出长度: 1024

## 🚀 TensorRT-LLM 安装和模型准备方案

### 方案1: 使用官方 TensorRT-LLM Docker 容器（推荐）

#### 优势
- 官方支持，配置完善
- 包含所有必要的依赖和工具
- 已优化的性能

#### 步骤
1. **拉取官方容器**:
   ```bash
   sudo docker pull nvcr.io/nvidia/tensorrt-llm:v0.12.0-nightly-trtllm-python-py3
   ```

2. **启动容器**:
   ```bash
   sudo docker run --gpus all --privileged -it --rm \
     -v /home/wt/work/Energyinfra:/workspace \
     nvcr.io/nvidia/tensorrt-llm:v0.12.0-nightly-trtllm-python-py3
   ```

3. **在容器内转换模型**:
   ```bash
   # 转换 Qwen-7B-INT4 模型
   python3 examples/llama/convert_checkpoint.py \
     --model_dir /path/to/qwen-7b-hf \
     --output_dir /workspace/models/qwen-7b-int4 \
     --dtype float16 \
     --calibrate_checkpoint
   ```

4. **构建 TensorRT 引擎**:
   ```bash
   trtllm-build --checkpoint_dir /workspace/models/qwen-7b-int4 \
     --output_dir /workspace/engines/qwen-7b-int4.engine \
     --gemm_plugin float16
   ```

### 方案2: 本地安装 TensorRT-LLM

#### 优势
- 直接在 Jetson 上运行
- 不需要 Docker
- 可以自定义配置

#### 步骤
1. **安装依赖**:
   ```bash
   # 安装 TensorRT
   sudo apt-get update
   sudo apt-get install -y tensorrt
   
   # 安装 Python 绑定
   pip3 install tensorrt==10.0.1
   ```

2. **克隆 TensorRT-LLM 仓库**:
   ```bash
   git clone https://github.com/NVIDIA/TensorRT-LLM.git
   cd TensorRT-LLM
   ```

3. **构建和安装**:
   ```bash
   # 构建项目
   make -C tools preproc
   pip install --extra-index-url https://pypi.nvidia.com nvidia-tensorrt
   pip install -r requirements.txt
   ```

4. **转换模型**: 同方案1

### 方案3: 使用预构建的引擎（快速测试）

#### 优势
- 无需转换模型
- 快速开始测试
- 适合初步验证

#### 可用资源
1. **NVIDIA NGC**: 
   - 访问: https://catalog.ngc.nvidia.com/
   - 搜索: "TensorRT-LLM Qwen"
   - 下载预构建的引擎

2. **社区贡献**:
   - 检查 GitHub 上的开源项目
   - Hugging Face 模型库
   - Jetson 论坛和社区

## 📊 项目兼容性检查

### 当前环境与需求对比

| 组件 | 需求 | 当前状态 | 兼容性 |
|------|------|----------|--------|
| Python | 3.8+ | 3.10.12 | ✅ 兼容 |
| CUDA | 11.8+ | 待确认 | ⏳ 需要检查 |
| TensorRT | 8.6+ | 待安装 | ❌ 缺失 |
| TensorRT-LLM | 0.6+ | 未安装 | ❌ 缺失 |
| 模型引擎 | Qwen-7B-INT4 | 占位符 | ❌ 缺失 |

### Jetson Orin 硬件规格
- **GPU**: NVIDIA Orin (Ampere 架构)
- **CPU**: ARM Cortex-A78AE
- **内存**: 32GB (根据配置)
- **架构**: ARM64

## 🔧 暂时替代方案

由于模型准备需要时间，可以采用以下替代方案进行初步测试：

### 方案A: 使用简化的基准测试脚本
创建不依赖实际 LLM 推理的基准测试，用于验证：
- 频率控制器功能
- 指标收集器功能
- 实验执行流程

### 方案B: 使用较小的模型
使用更小的模型进行测试，减少准备时间：
- **TinyBERT**: 小型 BERT 模型
- **DistilBERT**: 轻量级模型
- **合成测试**: 使用数学计算替代实际推理

### 方案C: 模拟器模式
开发模拟器模式，使用：
- 计算密集型内核模拟 prefill 阶段
- 内存访问模式模拟 decode 阶段
- 预定义的延迟和功耗数据

## 📅 建议的实施顺序

### 短期（本周）
1. **验证核心模块**: 使用替代方案验证频率控制和指标收集
2. **Docker 设置**: 配置 TensorRT-LLM Docker 环境
3. **模型选择**: 确定要使用的模型和量化方法

### 中期（下周）
1. **模型转换**: 转换选定的模型到 TensorRT 引擎
2. **小规模测试**: 运行单次基准测试
3. **系统验证**: 验证端到端流程

### 长期（下下周）
1. **完整实验**: 执行所有前置验证实验
2. **数据分析**: 分析实验结果
3. **优化迭代**: 根据结果调整配置

## 🎯 决策点

### 决策1: TensorRT-LLM 安装方式
**选项**: 
- Docker 容器（推荐）
- 本地安装
- 使用预构建引擎

**影响**: 
- Docker: 容易设置，但需要sudo权限
- 本地安装: 配置灵活，但需要更多时间
- 预构建引擎: 最快开始，但选项有限

### 决策2: 模型选择
**选项**:
- Qwen-7B-INT4（主要推荐）
- Llama-3-8B-FP16
- 更小的模型用于快速测试

**影响**:
- 模型大小影响实验时间和内存使用
- 量化方法影响精度和性能

### 决策3: 替代方案使用
**选项**:
- 立即使用替代方案验证系统
- 等待 TensorRT-LLM 准备完成
- 并行进行两者

**影响**:
- 替代方案可以快速开始，但不能替代真实测试
- 等待会延迟项目，但确保准确性

## 📝 风险和缓解措施

### 主要风险
1. **模型准备时间**: 转换和优化模型可能需要数小时到数天
   - **缓解**: 使用预构建引擎或较小模型

2. **内存限制**: Jetson Orin 内存可能不够容纳大型模型
   - **缓解**: 使用量化模型，调整批处理大小

3. **性能兼容性**: 模型可能在 Jetson 上性能不如预期
   - **缓解**: 选择针对 Jetson 优化的模型

4. **技术复杂性**: TensorRT-LLM 配置复杂
   - **缓解**: 使用官方文档和示例，逐步配置

### 备选计划
如果 TensorRT-LLM 无法工作，考虑：
- **vLLM**: 另一个高性能推理引擎
- **llama.cpp**: 轻量级推理引擎
- **TensorFlow Lite**: TensorFlow 的推理引擎

## ✅ 下一步行动

1. **立即行动**:
   - 选择 TensorRT-LLM 安装方式
   - 决定是否使用替代方案进行初步测试

2. **今日任务**:
   - 配置 Docker 或本地环境
   - 开始模型转换（如果决定进行）
   - 准备替代测试脚本（如果需要）

3. **本周目标**:
   - 完成环境配置
   - 成功运行第一次基准测试
   - 验证核心模块功能

---

**文档状态**: 初始版本，需要根据实际进展更新  
**责任者**: 系统架构师 + 开发团队  
**审查频率**: 每日更新