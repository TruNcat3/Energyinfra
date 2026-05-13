#!/usr/bin/env python3
"""
Plot Frequency Experiment Results
绘制频率实验结果的可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_frequency_results(json_file):
    """Plot frequency experiment results"""

    # Load results
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = data['results']

    # Extract data
    frequencies = [r['frequency_mhz'] for r in results]
    actual_freqs = [r['actual_mhz'] for r in results]
    performances = [r['tokens_per_second'] for r in results]
    temperatures = [r['temperature'] for r in results]
    inference_times = [r['inference_time'] for r in results]
    load_times = [r['load_time'] for r in results]

    # Create output directory
    output_dir = Path("figures/frequency_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance vs Frequency
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, performances, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('GPU Frequency (MHz)', fontsize=12)
    plt.ylabel('Performance (tokens/s)', fontsize=12)
    plt.title('GPU Frequency vs Performance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, (freq, perf) in enumerate(zip(frequencies, performances)):
        plt.annotate(f'{perf:.2f}',
                    (freq, perf),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_vs_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Temperature vs Frequency
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, temperatures, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('GPU Frequency (MHz)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('GPU Frequency vs Temperature', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, (freq, temp) in enumerate(zip(frequencies, temperatures)):
        plt.annotate(f'{temp:.1f}°C',
                    (freq, temp),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'temperature_vs_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Inference Time vs Frequency
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, inference_times, 'go-', linewidth=2, markersize=8)
    plt.xlabel('GPU Frequency (MHz)', fontsize=12)
    plt.ylabel('Inference Time (s)', fontsize=12)
    plt.title('GPU Frequency vs Inference Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, (freq, time) in enumerate(zip(frequencies, inference_times)):
        plt.annotate(f'{time:.3f}s',
                    (freq, time),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'inference_time_vs_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Energy Efficiency Analysis
    # Assuming power is proportional to frequency
    power_ratios = [f / frequencies[0] for f in frequencies]
    energy_efficiency = [perf / power for perf, power in zip(performances, power_ratios)]

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, energy_efficiency, 'mo-', linewidth=2, markersize=8)
    plt.xlabel('GPU Frequency (MHz)', fontsize=12)
    plt.ylabel('Energy Efficiency (tokens/J relative)', fontsize=12)
    plt.title('GPU Frequency vs Energy Efficiency', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add value labels
    for i, (freq, eff) in enumerate(zip(frequencies, energy_efficiency)):
        plt.annotate(f'{eff:.2f}',
                    (freq, eff),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'energy_efficiency_vs_frequency.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Comprehensive Summary
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Performance
    ax1.plot(frequencies, performances, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('GPU Frequency (MHz)')
    ax1.set_ylabel('Performance (tokens/s)')
    ax1.set_title('Performance vs Frequency')
    ax1.grid(True, alpha=0.3)

    # Temperature
    ax2.plot(frequencies, temperatures, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('GPU Frequency (MHz)')
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_title('Temperature vs Frequency')
    ax2.grid(True, alpha=0.3)

    # Inference Time
    ax3.plot(frequencies, inference_times, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('GPU Frequency (MHz)')
    ax3.set_ylabel('Inference Time (s)')
    ax3.set_title('Inference Time vs Frequency')
    ax3.grid(True, alpha=0.3)

    # Energy Efficiency
    ax4.plot(frequencies, energy_efficiency, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('GPU Frequency (MHz)')
    ax4.set_ylabel('Energy Efficiency (relative)')
    ax4.set_title('Energy Efficiency vs Frequency')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Frequency Experiment Results Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_experiment_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ All plots saved to: {output_dir}/")
    print("   📊 performance_vs_frequency.png")
    print("   📊 temperature_vs_frequency.png")
    print("   📊 inference_time_vs_frequency.png")
    print("   📊 energy_efficiency_vs_frequency.png")
    print("   📊 frequency_experiment_summary.png")

    # Calculate and print key insights
    print("\n📈 Key Insights:")
    print(f"  Performance range: {min(performances):.2f} - {max(performances):.2f} tokens/s")
    print(f"  Performance improvement: {((max(performances) - min(performances)) / min(performances) * 100):.1f}%")
    print(f"  Temperature range: {min(temperatures):.1f}°C - {max(temperatures):.1f}°C")
    print(f"  Best energy efficiency: {frequencies[energy_efficiency.index(max(energy_efficiency))]} MHz")
    print(f"  Worst energy efficiency: {frequencies[energy_efficiency.index(min(energy_efficiency))]} MHz")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "data/frequency_experiment_results/final_results_20260510_194153.json"

    if Path(json_file).exists():
        plot_frequency_results(json_file)
    else:
        print(f"❌ File not found: {json_file}")
        print("Usage: python3 plot_frequency_results.py <json_file>")