#!/home/wt/work/Energyinfra/jetson_llm_env/bin/python3
"""
Visualization for Qwen2.5-14B GPU-only Profiling Results
Generates comparison charts: finegrained profiling + cap profiling
"""
import sys, os
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

FG_PATH = 'data/energy_profiling/finegrained_profiling_20260522_074223.csv'
CAP_PATH = 'data/cap_profiling/cap_profiling_20260523_031353.csv'
OUT_DIR = Path('figures/14b_profiling')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
fg = pd.read_csv(FG_PATH)
cap = pd.read_csv(CAP_PATH)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Qwen2.5-14B-Instruct GPU DVFS Profiling', fontsize=16, fontweight='bold')

# Color setup
gpus = sorted(fg['gpu_freq_mhz'].unique())
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(gpus)))
phase_colors = {'decode': '#2196F3', 'mixed': '#FF9800'}

# ── 1. GPU Freq vs TPOT (decode & mixed) ──
ax = axes[0, 0]
for phase, color in phase_colors.items():
    sub = fg[fg['phase'] == phase].groupby('gpu_freq_mhz')['tpot_ms'].mean()
    ax.plot(sub.index, sub.values, 'o-', color=color, label=phase, linewidth=2, markersize=6)
ax.set_xlabel('GPU Frequency (MHz)')
ax.set_ylabel('TPOT (ms)')
ax.set_title('GPU Freq vs TPOT')
ax.legend()
ax.grid(True, alpha=0.3)

# ── 2. GPU Freq vs E/tok (decode & mixed) ──
ax = axes[0, 1]
for phase, color in phase_colors.items():
    sub = fg[fg['phase'] == phase].groupby('gpu_freq_mhz')['energy_per_token_j'].mean()
    ax.plot(sub.index, sub.values, 'o-', color=color, label=phase, linewidth=2, markersize=6)
ax.set_xlabel('GPU Frequency (MHz)')
ax.set_ylabel('Energy/Token (J)')
ax.set_title('GPU Freq vs Energy Efficiency')
ax.legend()
ax.grid(True, alpha=0.3)

# ── 3. GPU Freq vs Power ──
ax = axes[0, 2]
for phase, color in phase_colors.items():
    sub = fg[fg['phase'] == phase].groupby('gpu_freq_mhz')['avg_power_w'].mean()
    ax.plot(sub.index, sub.values, 'o-', color=color, label=phase, linewidth=2, markersize=6)
ax.set_xlabel('GPU Frequency (MHz)')
ax.set_ylabel('Avg Power (W)')
ax.set_title('GPU Freq vs Power')
ax.legend()
ax.grid(True, alpha=0.3)

# ── 4. Pareto: E/tok vs TPOT ──
ax = axes[1, 0]
for phase, color in phase_colors.items():
    sub = fg[fg['phase'] == phase].groupby('gpu_freq_mhz').agg({
        'energy_per_token_j': 'mean', 'tpot_ms': 'mean'
    })
    ax.scatter(sub['tpot_ms'], sub['energy_per_token_j'], c=color, s=80, label=phase, zorder=5)
    for gpu in sub.index:
        ax.annotate(f'{int(gpu)}', (sub.loc[gpu, 'tpot_ms'], sub.loc[gpu, 'energy_per_token_j']),
                   fontsize=7, ha='center', va='bottom')
ax.set_xlabel('TPOT (ms)')
ax.set_ylabel('Energy/Token (J)')
ax.set_title('Pareto: E/tok vs TPOT')
ax.legend()
ax.grid(True, alpha=0.3)
ax.invert_xaxis()

# ── 5. TTFT vs GPU freq ──
ax = axes[1, 1]
mixed = fg[fg['phase'] == 'mixed'].groupby('gpu_freq_mhz')['ttft_ms'].mean()
ax.bar(range(len(mixed)), mixed.values, color=phase_colors['mixed'], alpha=0.8)
ax.set_xticks(range(len(mixed)))
ax.set_xticklabels([str(int(g)) for g in mixed.index], rotation=45)
ax.set_xlabel('GPU Frequency (MHz)')
ax.set_ylabel('TTFT (ms)')
ax.set_title('TTFT vs GPU Freq (Mixed Phase)')
ax.grid(True, alpha=0.3, axis='y')

# ── 6. Cap Profiling: cap vs actual ──
ax = axes[1, 2]
cap_data = cap.groupby(['control_mode', 'gpu_cap_mhz']).agg({
    'tpot_ms': 'mean', 'energy_per_token_j': 'mean',
    'avg_power_w': 'mean', 'actual_gpu_mhz': 'mean',
}).reset_index()

modes = cap_data['control_mode'].unique()
x_labels = [f"Cap≤{int(r['gpu_cap_mhz'])}" if r['control_mode'] == 'cap'
            else 'Dynamic' for _, r in cap_data.iterrows()]
x = range(len(cap_data))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], cap_data['tpot_ms'], width, label='TPOT (ms)', color='#2196F3')
ax2 = ax.twinx()
bars2 = ax2.bar([i + width/2 for i in x], cap_data['energy_per_token_j'], width, label='E/tok (J)', color='#FF9800')
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=30, ha='right')
ax.set_ylabel('TPOT (ms)', color='#2196F3')
ax2.set_ylabel('Energy/Token (J)', color='#FF9800')
ax.set_title('Cap Profiling: GPU Cap Effect')
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
out_path = OUT_DIR / '14b_gpu_profiling_overview.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')

# ── Second figure: 3.8B vs 14B comparison ──
# Load old 3.8B data if available
old_paths = list(Path('data/energy_profiling').glob('finegrained_combined_*.csv'))
if not old_paths:
    old_paths = list(Path('data/energy_profiling').glob('finegrained_profiling_2026051*.csv'))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Size Comparison: 3.8B vs 14B DVFS Space', fontsize=14, fontweight='bold')

if old_paths:
    old = pd.read_csv(sorted(old_paths)[-1])
    old_dec = old[old['phase'] == 'decode'] if 'phase' in old.columns else old

    # Filter to common GPU frequencies
    common_gpus = sorted(set(fg['gpu_freq_mhz'].unique()) & set(old_dec['gpu_freq_mhz'].unique()))

    for i, (metric, ylabel) in enumerate([
        ('tpot_ms', 'TPOT (ms)'),
        ('energy_per_token_j', 'Energy/Token (J)'),
        ('avg_power_w', 'Power (W)'),
    ]):
        ax = axes[i]
        # 14B
        sub14 = fg[fg['phase'] == 'decode'].groupby('gpu_freq_mhz')[metric].mean()
        sub14 = sub14.reindex(common_gpus)
        ax.plot(common_gpus, sub14.values, 'o-', label='14B (Qwen2.5-14B)', color='#E53935', linewidth=2)

        # 3.8B
        sub38 = old_dec.groupby('gpu_freq_mhz')[metric].mean()
        sub38 = sub38.reindex(common_gpus)
        ax.plot(common_gpus, sub38.values, 's-', label='3.8B (Phi-3-mini)', color='#1E88E5', linewidth=2)

        ax.set_xlabel('GPU Frequency (MHz)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Add DVFS space annotations
    sub14_306 = fg[(fg['phase']=='decode') & (fg['gpu_freq_mhz']==306)][metric].mean()
    sub14_1300 = fg[(fg['phase']=='decode') & (fg['gpu_freq_mhz']==1300)][metric].mean()
else:
    axes[0].text(0.5, 0.5, 'No 3.8B data found', ha='center', va='center')

plt.tight_layout()
out_path2 = OUT_DIR / '14b_vs_38b_comparison.png'
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path2}')

print('Done!')
