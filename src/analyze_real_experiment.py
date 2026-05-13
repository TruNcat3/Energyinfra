#!/usr/bin/env python3
"""
Real Model Experiment Analysis
Quick analysis summary of the real model experiment results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob


def load_latest_data():
    files = sorted(glob.glob('data/real_model_experiment/real_experiment_*.csv'))
    if not files:
        raise FileNotFoundError("No experiment data found")
    latest = files[-1]
    print(f"Loading: {latest}")
    return pd.read_csv(latest), latest


def analyze(df: pd.DataFrame):
    total = len(df)
    zero_mask = df['output_tokens'] == 0
    n_zero = zero_mask.sum()

    print("=" * 60)
    print("REAL MODEL EXPERIMENT ANALYSIS")
    print("=" * 60)
    print(f"\nTotal runs: {total}")
    print(f"Zero-token anomalies: {n_zero} ({n_zero/total*100:.1f}%)")

    df_valid = df[~zero_mask].copy()
    print(f"Valid runs: {len(df_valid)}")

    print(f"\nConfigs tested: {df_valid['config_name'].nunique()}")
    print(f"Workloads tested: {df_valid['workload'].nunique()}")

    # GPU frequency effect
    print("\n" + "-" * 60)
    print("GPU FREQUENCY EFFECT (averaged over CPU freqs & workloads)")
    print("-" * 60)
    gpu_stats = df_valid.groupby('gpu_freq_mhz').agg({
        'ttft_ms': ['mean', 'std'],
        'tpot_ms': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std'],
    }).round(2)
    for gpu in gpu_stats.index:
        ttft_m = gpu_stats.loc[gpu, ('ttft_ms', 'mean')]
        tpot_m = gpu_stats.loc[gpu, ('tpot_ms', 'mean')]
        tps_m = gpu_stats.loc[gpu, ('tokens_per_second', 'mean')]
        print(f"  GPU {int(gpu):4d} MHz: TTFT={ttft_m:7.1f}ms  TPOT={tpot_m:7.1f}ms  TPS={tps_m:5.1f}")

    # CPU frequency effect
    print("\n" + "-" * 60)
    print("CPU FREQUENCY EFFECT (averaged over GPU freqs & workloads)")
    print("-" * 60)
    cpu_stats = df_valid.groupby('cpu_freq_mhz').agg({
        'ttft_ms': ['mean', 'std'],
        'tpot_ms': ['mean', 'std'],
        'tokens_per_second': ['mean', 'std'],
    }).round(2)
    for cpu in cpu_stats.index:
        ttft_m = cpu_stats.loc[cpu, ('ttft_ms', 'mean')]
        tpot_m = cpu_stats.loc[cpu, ('tpot_ms', 'mean')]
        tps_m = cpu_stats.loc[cpu, ('tokens_per_second', 'mean')]
        print(f"  CPU {int(cpu):4d} MHz: TTFT={ttft_m:7.1f}ms  TPOT={tpot_m:7.1f}ms  TPS={tps_m:5.1f}")

    # Per-config summary
    print("\n" + "-" * 60)
    print("PER-CONFIG SUMMARY (sorted by throughput)")
    print("-" * 60)
    config_stats = df_valid.groupby('config_name').agg({
        'gpu_freq_mhz': 'first',
        'cpu_freq_mhz': 'first',
        'ttft_ms': 'mean',
        'tpot_ms': 'mean',
        'tokens_per_second': 'mean',
    }).sort_values('tokens_per_second', ascending=False)

    print(f"  {'Config':<20} {'GPU':>4} {'CPU':>4} {'TTFT':>8} {'TPOT':>8} {'TPS':>6}")
    for cfg, row in config_stats.iterrows():
        print(f"  {cfg:<20} {int(row['gpu_freq_mhz']):4d} {int(row['cpu_freq_mhz']):4d} "
              f"{row['ttft_ms']:8.1f} {row['tpot_ms']:8.1f} {row['tokens_per_second']:6.1f}")

    # Per-workload summary
    print("\n" + "-" * 60)
    print("PER-WORKLOAD SUMMARY")
    print("-" * 60)
    for wl in sorted(df_valid['workload'].unique()):
        wdata = df_valid[df_valid['workload'] == wl]
        pl = int(wdata['prompt_length'].iloc[0])
        ol = int(wdata['output_length'].iloc[0])
        print(f"\n  {wl} (prompt={pl}, output={ol}):")
        best = wdata.loc[wdata['tokens_per_second'].idxmax()]
        worst = wdata.loc[wdata['tokens_per_second'].idxmin()]
        print(f"    Best TPS:  {best['config_name']} ({best['tokens_per_second']:.1f} tok/s)")
        print(f"    Worst TPS: {worst['config_name']} ({worst['tokens_per_second']:.1f} tok/s)")
        print(f"    TTFT range: {wdata['ttft_ms'].min():.1f} - {wdata['ttft_ms'].max():.1f} ms")
        print(f"    Mean TPS: {wdata['tokens_per_second'].mean():.1f} ± {wdata['tokens_per_second'].std():.1f}")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # GPU scaling factor
    gpu_min = df_valid[df_valid['gpu_freq_mhz'] == df_valid['gpu_freq_mhz'].min()]['tokens_per_second'].mean()
    gpu_max = df_valid[df_valid['gpu_freq_mhz'] == df_valid['gpu_freq_mhz'].max()]['tokens_per_second'].mean()
    gpu_speedup = gpu_max / gpu_min if gpu_min > 0 else 0

    # CPU scaling factor
    cpu_min = df_valid[df_valid['cpu_freq_mhz'] == df_valid['cpu_freq_mhz'].min()]['tokens_per_second'].mean()
    cpu_max = df_valid[df_valid['cpu_freq_mhz'] == df_valid['cpu_freq_mhz'].max()]['tokens_per_second'].mean()
    cpu_speedup = cpu_max / cpu_min if cpu_min > 0 else 0

    print(f"\n  1. GPU freq scaling: {df_valid['gpu_freq_mhz'].min():.0f}→{df_valid['gpu_freq_mhz'].max():.0f} MHz "
          f"= {gpu_speedup:.2f}x throughput")
    print(f"  2. CPU freq scaling: {df_valid['cpu_freq_mhz'].min():.0f}→{df_valid['cpu_freq_mhz'].max():.0f} MHz "
          f"= {cpu_speedup:.2f}x throughput")
    print(f"  3. Bottleneck: {'CPU-bound' if cpu_speedup > gpu_speedup else 'GPU-bound'} "
          f"(CPU scaling {cpu_speedup:.2f}x vs GPU scaling {gpu_speedup:.2f}x)")

    # TTFT scaling
    ttft_gpu_min = df_valid[df_valid['gpu_freq_mhz'] == df_valid['gpu_freq_mhz'].min()]['ttft_ms'].mean()
    ttft_gpu_max = df_valid[df_valid['gpu_freq_mhz'] == df_valid['gpu_freq_mhz'].max()]['ttft_ms'].mean()
    ttft_cpu_min = df_valid[df_valid['cpu_freq_mhz'] == df_valid['cpu_freq_mhz'].min()]['ttft_ms'].mean()
    ttft_cpu_max = df_valid[df_valid['cpu_freq_mhz'] == df_valid['cpu_freq_mhz'].max()]['ttft_ms'].mean()

    print(f"  4. TTFT GPU effect: {ttft_gpu_min:.1f}→{ttft_gpu_max:.1f} ms "
          f"({(ttft_gpu_min-ttft_gpu_max)/ttft_gpu_min*100:.1f}% reduction)")
    print(f"  5. TTFT CPU effect: {ttft_cpu_min:.1f}→{ttft_cpu_max:.1f} ms "
          f"({(ttft_cpu_min-ttft_cpu_max)/ttft_cpu_min*100:.1f}% reduction)")

    print()


if __name__ == '__main__':
    df, source = load_latest_data()
    analyze(df)
