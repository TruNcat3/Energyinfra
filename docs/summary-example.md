# Energyinfra 项目总结与示例

[项目首页](../README.md) | [实验文档索引](实验文档/README.md)

## 项目总结

Energyinfra 面向 Jetson Orin 上的 LLM 推理，通过离线 profiling 建立能耗
rate table，再根据输入长度、输出长度和优化目标选择 GPU 频率 cap。
主要关注每 token 能耗（E/tok）、每 token 延迟（TPOT）和功率之间的权衡。

基本流程是：

1. 在目标设备上采集 Lock/Cap 模式的 profiling 数据。
2. 构建 rate table，保留模型、workload 和频率配置对应的测量结果。
3. 使用 Pareto 或 SLO 等策略选择 cap，再在实际推理中验证效果。

环境准备见 [运行时搭建指南](说明文档/runtime_setup_guide.md)，完整实验流程
见 [实验总览](实验文档/实验总览.md)。

## 示例：为一个 workload 选择 GPU cap

下面以 Qwen2.5-7B、输入 512 tokens、输出 128 tokens 为例。先按照项目首页
的 profiling 和建表步骤生成该模型的 Cap/Lock rate table，然后将代码中的
两个路径替换成实际生成的 Parquet 文件路径。仓库不包含这些原始数据文件。

在已安装项目依赖的环境中，从仓库根目录运行：

```python
from src.controller.workload_cap_selector import WorkloadCapSelector

selector = WorkloadCapSelector(
    cap_rate_table_path="data/rate_tables/cap_rate_table_with_savings_<timestamp>.parquet",
    lock_rate_table_path="data/rate_tables/lock_rate_table_<timestamp>.parquet",
    model="Qwen2.5-7B-Instruct-Q4_K_M",
)

config = selector.select(
    prompt_length=512,
    output_length=128,
    strategy="pareto",
)
if config is None:
    raise SystemExit("没有可用的 cap 数据，请检查模型名称和 rate table。")

print(f"GPU cap: {config['gpu_cap_mhz']} MHz")
print(f"预计 E/token: {config['pred_energy_per_token_j']:.3f} J")
```

这个示例读取已有 rate table 并输出选择结果。输出的能耗是表中数据给出的
预测值；实际部署和对比实验应在目标 Jetson 上测量，记录模型、workload、
governor 和频率配置。详细口径见 [实验结果详述](实验文档/实验结果详述.md)。
