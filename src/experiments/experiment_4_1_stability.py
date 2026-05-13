#!/usr/bin/env python3
"""
Experiment 4.1: Measurement Stability Test
验证测量的稳定性和可重复性
"""

import subprocess
import time
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List
import os
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, 'src')

from synthetic_benchmark import SyntheticBenchmark
from freq_controller import FrequencyController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Experiment4_1_Stability:
    """
    Experiment 4.1: Measurement Stability Test

    目的：验证在不同重复运行下，测量指标是否稳定
    验证标准：
    - std(energy/token) < threshold (目标: <5%)
    - std(latency) < threshold (目标: <5%)
    - 系统稳定性 > 99%
    """

    def __init__(self, config: Dict):
        """初始化实验配置"""
        self.config = config
        self.benchmark = SyntheticBenchmark(config)
        self.freq_controller = FrequencyController(config, control_method='hybrid')

        # 实验参数
        self.workload = {
            'batch_size': 1,
            'prompt_len': 512,
            'output_len': 128,
            'phase': 'mixed'
        }

        self.frequency = {
            'gpu_freq': 846,   # 中档频率
            'cpu_freq': 1479,
            'emc_freq': 1600
        }

        self.repeats = 10     # 重复次数
        self.warmup_runs = 1  # 预热次数

        # 稳定性阈值
        self.stability_thresholds = {
            'energy_token_cv': 0.05,    # 能耗变异系数 < 5%
            'ttft_cv': 0.05,           # TTFT变异系数 < 5%
            'tpot_cv': 0.05,           # TPOT变异系数 < 5%
            'throughput_cv': 0.05        # 吞吐量变异系数 < 5%
        }

    def run_single_iteration(self, iteration: int) -> Dict:
        """
        运行单次实验迭代

        Args:
            iteration: 迭代编号

        Returns:
            包含测量结果的字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{self.repeats}")
        logger.info(f"{'='*60}")

        try:
            # 设置频率
            logger.info("Setting frequencies...")
            # 注意：在实际环境中，这里需要sudo权限
            # 合成测试中我们只记录频率设置

            # 短暂等待，模拟频率切换
            time.sleep(0.1)

            # 运行基准测试
            logger.info("Running benchmark...")
            result = self.benchmark.run_benchmark(self.workload, self.frequency)

            # 添加元数据
            result['iteration'] = iteration
            result['timestamp'] = datetime.now().isoformat()

            # 添加频率信息（合成测试模拟）
            result['gpu_freq_set_mhz'] = self.frequency['gpu_freq']
            result['cpu_freq_set_mhz'] = self.frequency['cpu_freq']
            result['emc_freq_set_mhz'] = self.frequency['emc_freq']

            logger.info(f"Iteration {iteration + 1} completed:")
            logger.info(f"  TTFT: {result['ttft_ms']:.2f} ms")
            logger.info(f"  TPOT: {result['tpot_ms']:.2f} ms")
            logger.info(f"  Throughput: {result['tokens_per_second']:.2f} tokens/s")
            logger.info(f"  Energy/token: {result['energy_per_token_j']:.4f} J/token")
            logger.info(f"  Total Energy: {result['total_energy_j']:.2f} J")

            return result

        except Exception as e:
            logger.error(f"Iteration {iteration + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_stability_metrics(self, results: List[Dict]) -> Dict:
        """
        计算稳定性指标

        Args:
            results: 所有实验结果列表

        Returns:
            稳定性指标字典
        """
        if not results:
            return {}

        # 提取关键指标
        ttft_values = [r['ttft_ms'] for r in results]
        tpot_values = [r['tpot_ms'] for r in results]
        energy_token_values = [r['energy_per_token_j'] for r in results]
        throughput_values = [r['tokens_per_second'] for r in results]

        # 计算统计量
        metrics = {
            'ttft': {
                'mean': np.mean(ttft_values),
                'std': np.std(ttft_values),
                'min': np.min(ttft_values),
                'max': np.max(ttft_values),
                'cv': np.std(ttft_values) / np.mean(ttft_values) if np.mean(ttft_values) > 0 else 0,
                'count': len(ttft_values)
            },
            'tpot': {
                'mean': np.mean(tpot_values),
                'std': np.std(tpot_values),
                'min': np.min(tpot_values),
                'max': np.max(tpot_values),
                'cv': np.std(tpot_values) / np.mean(tpot_values) if np.mean(tpot_values) > 0 else 0,
                'count': len(tpot_values)
            },
            'energy_token': {
                'mean': np.mean(energy_token_values),
                'std': np.std(energy_token_values),
                'min': np.min(energy_token_values),
                'max': np.max(energy_token_values),
                'cv': np.std(energy_token_values) / np.mean(energy_token_values) if np.mean(energy_token_values) > 0 else 0,
                'count': len(energy_token_values)
            },
            'throughput': {
                'mean': np.mean(throughput_values),
                'std': np.std(throughput_values),
                'min': np.min(throughput_values),
                'max': np.max(throughput_values),
                'cv': np.std(throughput_values) / np.mean(throughput_values) if np.mean(throughput_values) > 0 else 0,
                'count': len(throughput_values)
            }
        }

        return metrics

    def evaluate_stability(self, stability_metrics: Dict) -> Dict:
        """
        评估稳定性是否满足阈值

        Args:
            stability_metrics: 稳定性指标字典

        Returns:
            评估结果字典
        """
        evaluation = {
            'ttft_stable': stability_metrics['ttft']['cv'] <= self.stability_thresholds['ttft_cv'],
            'tpot_stable': stability_metrics['tpot']['cv'] <= self.stability_thresholds['tpot_cv'],
            'energy_token_stable': stability_metrics['energy_token']['cv'] <= self.stability_thresholds['energy_token_cv'],
            'throughput_stable': stability_metrics['throughput']['cv'] <= self.stability_thresholds['throughput_cv'],
            'overall_stable': False
        }

        # 总体稳定性：所有指标都稳定
        evaluation['overall_stable'] = all([
            evaluation['ttft_stable'],
            evaluation['tpot_stable'],
            evaluation['energy_token_stable'],
            evaluation['throughput_stable']
        ])

        return evaluation

    def generate_report(self, results: List[Dict], stability_metrics: Dict, evaluation: Dict) -> str:
        """
        生成实验报告

        Args:
            results: 所有实验结果
            stability_metrics: 稳定性指标
            evaluation: 评估结果

        Returns:
            报告字符串
        """
        report = []
        report.append("="*80)
        report.append("Experiment 4.1: Measurement Stability Test - Report")
        report.append("="*80)
        report.append("")

        # 实验配置
        report.append("📋 Experiment Configuration:")
        report.append(f"  Workload: Batch={self.workload['batch_size']}, Prompt={self.workload['prompt_len']}, Output={self.workload['output_len']}")
        report.append(f"  Frequency: GPU={self.frequency['gpu_freq']}MHz, CPU={self.frequency['cpu_freq']}MHz, EMC={self.frequency['emc_freq']}MHz")
        report.append(f"  Repeats: {self.repeats}")
        report.append("")

        # 稳定性评估
        report.append("📊 Stability Evaluation:")

        def format_status(status) -> str:
            return "✅ PASS" if bool(status) else "❌ FAIL"

        report.append(f"  TTFT Stability: {format_status(evaluation['ttft_stable'])}")
        report.append(f"    - Mean: {stability_metrics['ttft']['mean']:.2f} ms")
        report.append(f"    - Std:  {stability_metrics['ttft']['std']:.2f} ms")
        report.append(f"    - CV:   {stability_metrics['ttft']['cv']*100:.2f}% (threshold: {self.stability_thresholds['ttft_cv']*100:.1f}%)")

        report.append(f"  TPOT Stability: {format_status(evaluation['tpot_stable'])}")
        report.append(f"    - Mean: {stability_metrics['tpot']['mean']:.2f} ms")
        report.append(f"    - Std:  {stability_metrics['tpot']['std']:.2f} ms")
        report.append(f"    - CV:   {stability_metrics['tpot']['cv']*100:.2f}% (threshold: {self.stability_thresholds['tpot_cv']*100:.1f}%)")

        report.append(f"  Energy/token Stability: {format_status(evaluation['energy_token_stable'])}")
        report.append(f"    - Mean: {stability_metrics['energy_token']['mean']:.4f} J/token")
        report.append(f"    - Std:  {stability_metrics['energy_token']['std']:.4f} J/token")
        report.append(f"    - CV:   {stability_metrics['energy_token']['cv']*100:.2f}% (threshold: {self.stability_thresholds['energy_token_cv']*100:.1f}%)")

        report.append(f"  Throughput Stability: {format_status(evaluation['throughput_stable'])}")
        report.append(f"    - Mean: {stability_metrics['throughput']['mean']:.2f} tokens/s")
        report.append(f"    - Std:  {stability_metrics['throughput']['std']:.2f} tokens/s")
        report.append(f"    - CV:   {stability_metrics['throughput']['cv']*100:.2f}% (threshold: {self.stability_thresholds['throughput_cv']*100:.1f}%)")

        report.append("")
        report.append(f"  Overall Stability: {format_status(evaluation['overall_stable'])}")

        # 详细结果
        report.append("")
        report.append("📈 Detailed Results:")
        report.append("")
        report.append("Iter | TTFT(ms) | TPOT(ms) | Throughput(tok/s) | Energy(J) | Energy/Token(J)")
        report.append("-"*80)

        for result in results:
            iter_str = f"{result['iteration']+1:4d}"
            ttft_str = f"{result['ttft_ms']:8.2f}"
            tpot_str = f"{result['tpot_ms']:8.2f}"
            thr_str = f"{result['tokens_per_second']:15.2f}"
            eng_str = f"{result['total_energy_j']:8.2f}"
            etok_str = f"{result['energy_per_token_j']:13.4f}"

            report.append(f"{iter_str} | {ttft_str} | {tpot_str} | {thr_str} | {eng_str} | {etok_str}")

        report.append("="*80)

        return "\n".join(report)

    def save_results(self, results: List[Dict], stability_metrics: Dict, evaluation: Dict):
        """
        保存实验结果到文件

        Args:
            results: 所有实验结果
            stability_metrics: 稳定性指标
            evaluation: 评估结果
        """
        # 创建输出目录
        output_dir = "data/experiment_4_1"
        os.makedirs(output_dir, exist_ok=True)

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存CSV文件
        df = pd.DataFrame(results)
        csv_path = f"{output_dir}/stability_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to: {csv_path}")

        # 保存稳定性指标
        stability_path = f"{output_dir}/stability_metrics_{timestamp}.json"

        # 转换类型以便JSON序列化
        def make_serializable(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, bool):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return make_serializable(obj.tolist())
            else:
                return obj

        serializable_stability = make_serializable(stability_metrics)
        serializable_evaluation = make_serializable(evaluation)

        with open(stability_path, 'w') as f:
            json.dump({
                'stability_metrics': serializable_stability,
                'evaluation': serializable_evaluation,
                'configuration': {
                    'workload': self.workload,
                    'frequency': self.frequency,
                    'repeats': self.repeats,
                    'thresholds': self.stability_thresholds
                }
            }, f, indent=2)
        logger.info(f"Stability metrics saved to: {stability_path}")

        # 保存报告
        report = self.generate_report(results, stability_metrics, evaluation)
        report_path = f"{output_dir}/stability_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

        return {
            'results_file': csv_path,
            'stability_file': stability_path,
            'report_file': report_path
        }

    def run_experiment(self):
        """
        运行完整的实验4.1
        """
        logger.info("Starting Experiment 4.1: Measurement Stability Test")
        logger.info(f"Configuration: {self.workload}, {self.frequency}")
        logger.info(f"Repeats: {self.repeats}")

        # 运行预热
        if self.warmup_runs > 0:
            logger.info(f"\nRunning {self.warmup_runs} warmup iteration(s)...")
            for i in range(self.warmup_runs):
                self.run_single_iteration(-1 - i)  # 使用负数标记预热

        # 运行实际实验
        logger.info(f"\nStarting {self.repeats} measurement iterations...")
        results = []

        for i in range(self.repeats):
            result = self.run_single_iteration(i)
            if result:
                results.append(result)

            # 短暂冷却时间
            if i < self.repeats - 1:
                time.sleep(0.5)

        # 计算稳定性指标
        logger.info("\nCalculating stability metrics...")
        stability_metrics = self.calculate_stability_metrics(results)

        # 评估稳定性
        logger.info("Evaluating stability against thresholds...")
        evaluation = self.evaluate_stability(stability_metrics)

        # 生成和保存报告
        logger.info("Generating and saving results...")
        saved_files = self.save_results(results, stability_metrics, evaluation)

        # 打印报告
        report = self.generate_report(results, stability_metrics, evaluation)
        print(report)

        # 返回结果
        return {
            'results': results,
            'stability_metrics': stability_metrics,
            'evaluation': evaluation,
            'saved_files': saved_files
        }


def main():
    """主函数"""
    # 配置
    config = {
        'synthetic_benchmark': {
            'base_time_per_token_ms': 0.2,
            'base_tpot_ms': 8.0,
            'variability_factor': 0.1
        },
        'frequencies': {
            'gpu': {
                'presets': {
                    'low': 378,
                    'mid': 846,
                    'high': 1428
                }
            }
        }
    }

    # 创建并运行实验
    experiment = Experiment4_1_Stability(config)
    results = experiment.run_experiment()

    # 根据评估结果返回状态
    if results['evaluation']['overall_stable']:
        print("\n✅ Experiment 4.1 PASSED: Measurements are stable within thresholds")
        return 0
    else:
        print("\n❌ Experiment 4.1 FAILED: Measurements exceed stability thresholds")
        return 1


if __name__ == "__main__":
    sys.exit(main())