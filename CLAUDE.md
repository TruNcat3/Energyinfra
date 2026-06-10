# Claude AI Assistant Documentation

## Project Overview

**Project Name**: Jetson-LLM Energy Rate Table and Adaptive DVFS Selector

**Goal**: Build an energy profiling and adaptive configuration selection system for LLM inference on Jetson edge GPUs. The system automatically profiles different workload × frequency configurations, builds an energy rate table, and provides SLO-aware frequency selection for GPU/CPU/EMC to optimize energy efficiency under latency, power, and thermal constraints.

**Platform**: Jetson Orin with llama.cpp runtime (TensorRT-LLM planned)

**Current Phase**: Phase 13 Code Complete — Oracle Gap + Thermal-SLO Controller + Serving Benchmark (awaiting hardware validation)

## Key Concepts

### Energy Rate Table
A workload-aware lookup table that maps workload characteristics (batch size, prompt length, output length, phase, etc.) and frequency configurations (GPU/CPU/EMC frequencies) to performance metrics (TTFT, TPOT, throughput), power metrics (avg/max power), and energy metrics (energy/token, tokens/J).

### Adaptive DVFS (Dynamic Voltage and Frequency Scaling)
Automatic adjustment of GPU, CPU, and EMC frequencies based on workload characteristics and SLO constraints to optimize energy efficiency while meeting performance requirements.

### Phase-Aware DVFS
Different frequency configurations for prefill and decode phases, as these phases have different computational and memory access patterns.

### SLO-Aware Selection
Choosing configurations that satisfy Service Level Objectives (SLOs) such as TTFT < X ms, TPOT < Y ms, while minimizing energy consumption.

## Project Structure

```
/home/wt/work/Energyinfra/
├── src/                        # Python source (organized by function)
│   ├── controller/            # Frequency/DVFS Control
│   │   ├── freq_controller.py       # sysfs/jetson_clocks frequency control
│   │   ├── cap_controller.py        # Lock/Cap/Dynamic 三模式频率控制 (P8)
│   │   ├── pareto_selector.py       # 多目标 Pareto DVFS 选择器 (P11)
│   │   ├── workload_cap_selector.py  # Workload-aware Cap 选择器 (P11)
│   │   └── thermal_slo_controller.py # Thermal-SLO 反馈控制器 (P13)
│   ├── metrics/               # Metrics Collection
│   │   ├── metrics_collector.py     # tegrastats integration
│   │   └── parse_logs.py            # Log parsing
│   ├── benchmark/             # Benchmark Framework
│   │   └── llama_cpp_runner.py      # llama.cpp inference runner (含 benchmark_mode)
│   ├── ratetable/             # Rate Table
│   │   ├── build_workload_rate_table.py # Rate Table 构建 (含 Pareto rank)
│   │   └── oracle_gap_analysis.py   # Oracle Gap 分析 (P13)
│   ├── experiments/           # Experiment Scripts
│   │   ├── run_finegrained_profiling.py  # Lock-mode 11 GPU freq profiling
│   │   ├── run_cap_profiling.py          # Cap-mode profiling (4 caps + baselines)
│   │   ├── run_cap_selector_benchmark.py # E2E 验证 benchmark (P12)
│   │   ├── run_finegrained_cap_profiling.py # Fine-grained 10-cap profiling (P13)
│   │   └── run_serving_benchmark.py      # Long-running serving benchmark (P13)
│   ├── visualization/         # Visualization
│   │   ├── visualize_pareto.py          # Pareto 前沿可视化 (6 图)
│   │   ├── visualize_cross_model_comparison.py # 跨模型对比 (4 图)
│   │   ├── analyze_e2e_benchmark.py     # E2E benchmark 分析 (5 图 + 报告)
│   │   └── visualize_phase13.py         # Phase 13 可视化 (6 图)
│   └── _legacy/               # Archived code
├── data/
│   ├── energy_profiling/               # Lock-mode profiling CSV (3 models × 792 rows)
│   ├── cap_profiling/                  # Cap-mode profiling CSV
│   ├── rate_tables/                    # Lock/cap rate tables + DVFS rules
│   ├── cap_selector_benchmark/         # E2E benchmark results (378 runs)
│   ├── oracle_gap_analysis/            # Oracle gap analysis results
│   └── serving_benchmark/              # Serving benchmark results (P13)
├── figures/
│   ├── pareto_frontier/                # Pareto 前沿图
│   ├── cross_model_comparison/         # 跨模型对比图
│   ├── e2e_benchmark/                  # E2E benchmark 图表
│   └── phase13_analysis/               # Phase 13 分析图表
├── docs/
│   ├── 任务书/                # Task specifications
│   ├── 开发文档/              # Development docs (当前状态.md)
│   └── 说明文档/              # User/setup guides
├── EnergyInfra_next_stage_method_and_tasks.md  # Phase 13 任务书
├── README.md
└── CLAUDE.md
```
│   ├── visualization/         # Visualization (10 files)
│   ├── experiments/           # Experiment Scripts (9 files)
│   └── _legacy/               # Archived code (5 files)
├── configs/                    # Configuration files
├── data/                       # Data output
├── figures/                    # Visualization output
├── scripts/
│   ├── active/                # Active shell scripts
│   └── _legacy/               # Archived shell scripts
├── docs/                       # Documentation (categorized)
│   ├── 任务书/                # Task specifications
│   ├── 开发文档/              # Development docs
│   ├── 说明文档/              # User/setup guides
│   └── 实验文档/              # Experiment reports
├── README.md
└── CLAUDE.md
```

## Critical Documentation Files

### Task Specifications (docs/任务书/)
| File | Purpose |
|------|---------|
| `docs/任务书/任务书.md` | Project goals, technical roadmap, success criteria |
| `docs/任务书/jetson_llm_energy_rate_table_task_doc.md` | Detailed technical task specification |

### Development Docs (docs/开发文档/)
| File | Purpose |
|------|---------|
| `docs/开发文档/当前状态.md` | Current status, module progress, risks, next steps |
| `docs/开发文档/checklist.md` | TODO checklist, code quality guidelines |
| `docs/开发文档/project_progress_summary.md` | Full progress summary (Phase 1-5) |
| `docs/开发文档/EnergyInfra_Phase5_Task_Plan.md` | Phase 5 detailed task plan |
| `docs/开发文档/scheduling_method_analysis.md` | Scheduling method analysis + real model validation |

### User Guides (docs/说明文档/)
| File | Purpose |
|------|---------|
| `docs/说明文档/LLAMACPP_INTEGRATION.md` | llama.cpp integration guide |
| `docs/说明文档/runtime_setup_guide.md` | Runtime environment setup |
| `docs/说明文档/quick_experiment_guide.md` | Quick experiment guide |

### Experiment Reports (docs/实验文档/)
| File | Purpose |
|------|---------|
| `docs/实验文档/PHASE1_COMPLETION_REPORT.md` | Phase 1 foundation completion |
| `docs/实验文档/PHASE2_COMPLETION_REPORT.md` | Phase 2 core modules completion |
| `docs/实验文档/PHASE3_COMPLETION_REPORT.md` | Phase 3 preliminary experiments |

### Key Data & Figures
| Path | Content |
|------|---------|
| `data/energy_profiling/finegrained_combined_20260516.csv` | 1320-run fine-grained GPU×EMC profiling |
| `data/energy_profiling/e2e_benchmark_finegrained_20260517.csv` | 225-run E2E benchmark with alpha sweep |
| `data/rate_tables/finegrained_selector_table_20260516_071238.parquet` | Rate table (31 configs, 10 buckets) |
| `data/rate_tables/finegrained_dvfs_rules_20260516_071238.json` | 30 DVFS rules |
| `figures/e2e_benchmark_finegrained/` | 5 E2E benchmark charts |
| `figures/finegrained_profiling/` | 5 fine-grained profiling charts |
| `data/_archived_invalid/` | Archived incorrect data (4 categories) |

---

### 1. docs/任务书/任务书.md (Project Goals and Direction)
**Purpose**: Defines project objectives, technical roadmap, and success criteria

**Key Sections**:
- Project goals (energy rate table construction, adaptive configuration selection, online control)
- Technical targets (platform: Jetson Orin, runtime: TensorRT-LLM)
- Research questions (measurement stability, frequency knob effects, workload differences)
- Technical roadmap (4 phases: offline profiling → rate table → configuration selection → online control)
- Success criteria and risk assessment

**When to reference**:
- Understanding project scope and objectives
- Designing new features or modules
- Making architectural decisions
- Evaluating project progress

### 2. docs/开发文档/当前状态.md (Current Development Status)
**Purpose**: Tracks current development status, unclear parts, and compromises

**Key Sections**:
- Completed work and current development phase
- Module implementation status table
- Configuration file status table
- Ambiguities in technical details (frequency control, tegrastats parsing, benchmark compatibility)
- Experimental design uncertainties (stability thresholds, frequency presets, time costs)
- Current risks and limitations
- Key decision points

**When to reference**:
- Understanding what has been implemented and what remains
- Identifying unresolved technical questions
- Making implementation decisions based on current constraints
- Planning next steps based on current status

### 3. docs/开发文档/checklist.md (Checklist and Guidelines)
**Purpose**: Comprehensive checklist of TODO items and modification guidelines

**Key Sections**:
- Phased TODO checklist (6 major phases with detailed sub-tasks)
- Code modification checklist (naming, quality, error handling, logging, performance, testing)
- Documentation modification checklist (technical docs, config files, user docs)
- Experiment design checklist (configuration, data collection, result analysis)
- Git commit checklist
- Code quality guidelines (Python, YAML, experimental records)
- Performance optimization guidelines
- Security and reliability guidelines
- Debugging and troubleshooting guides

**When to reference**:
- Before making any code changes
- Before committing code
- When debugging issues
- When designing experiments
- When performing code reviews

## Implementation Status

### Completed (Phase 1) ✅
- ✅ Project directory structure created
- ✅ Core documentation written (任务书.md, 当前状态.md, checklist.md)
- ✅ CLAUDE.md created with project context
- ✅ Basic configuration file templates

### Completed (Phase 2) ✅
- ✅ Frequency controller (freq_controller.py) - sysfs + jetson_clocks hybrid
- ✅ Metrics collector (metrics_collector.py) - tegrastats integration
- ✅ Benchmark runner (benchmark_runner.py) - TensorRT-LLM + synthetic
- ✅ Sweep runner (sweep_runner.py) - workload × frequency orchestration
- ✅ Log parser (parse_logs.py) - structured output

### Completed (Phase 3) ✅
- ✅ Experiment 4.1: Measurement stability verified (TPOT CV=1.0%)
- ✅ Experiment 4.2: Single-knob sensitivity (GPU/CPU/EMC sweeps)
- ✅ Experiment 4.3: Frequency combination interactions
- ✅ Experiment 4.4: Prefill/Decode phase differences (26.97x difference!)
- ✅ Experiment 4.5: Workload feature predictability
- ✅ Experiment 4.6: Frequency switching overhead quantified
- ✅ Experiment 4.7: End-to-end SLO validation (67% configs meet SLO)

### Completed (Phase 4) ✅
- ✅ Energy rate table built from experimental data (23 configs)
- ✅ SLO-Aware config selector (select_config.py)
- ✅ Phase-Aware DVFS policy (phase_aware_policy.py)
- ✅ Phase-Aware online controller (phase_controller.py)
- ✅ Validation experiment: 5 strategies × 5 workloads × 10 repeats
- ✅ Comparison visualizations (5 charts)
- ✅ Key result: **30% energy reduction + 36% TTFT improvement** vs default

### Completed (Phase 5)
- ✅ P0-P4: Evaluation, Jetson baseline, real model runner, phase-aware policy, rate table
- ✅ Real model multi-config experiment (180 runs, initial)
- ✅ GPU frequency control bug found and fixed (userspace→performance governor)
- ✅ Expanded profiling experiment (672 runs, 14 workloads, corrected governor)
- ✅ Rate table rebuilt from corrected GPU data (28 buckets, DVFS rules mined)

### Completed (Phase 6) - Fine-Grained GPU×EMC DVFS
- ✅ EMC frequency control via DebugFS (204/665/2133/3199 MHz)
- ✅ Fine-grained profiling: 11 GPU × 4 EMC × 5 workloads × 2 phases × 3 repeats = **1320 runs**
- ✅ Rate table rebuilt: 10 buckets, 31 GPU×EMC configs, 101 Pareto-optimal, 30 DVFS rules
- ✅ WeightedSelector upgraded to 3D (GPU×EMC×CPU) with alpha knob
- ✅ E2E benchmark: 225 runs (6 alpha × 2 strategies + 3 baselines), 5 visualizations

### Key Experimental Results (Fine-Grained, 2026-05-16)

> **Fine-grained profiling** expands from 4 GPU configs to 11 GPU × 4 EMC = 44 configs.
> EMC controlled via `/sys/kernel/debug/emc/min_rate` and `max_rate` (requires root).

| GPU (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) |
|:---:|:---:|:---:|:---:|:---:|
| 306 | 1.33 | 81.8 | 12.2 | 16.2 |
| 612 | 1.00 | 43.4 | 23.1 | 22.6 |
| 816 | 0.94 | 33.3 | 30.1 | 28.9 |
| 1300 | 1.15 | 24.8 | 40.3 | 42.1 |

**Key findings**:
- GPU612 and GPU816 are energy efficiency sweet spots (W-shaped curve)
- EMC 204 MHz is most energy-efficient (memory bandwidth not bottleneck for 3.8B model)
- Best config: **GPU816 + EMC204** (E/tok ≈ 0.97J, TPOT ≈ 33ms)

**Data files**: `data/energy_profiling/finegrained_combined_20260516.csv`
**Rate table**: `data/rate_tables/finegrained_selector_table_20260516_071238.parquet`
**E2E benchmark**: `data/energy_profiling/e2e_benchmark_finegrained_20260517.csv`

### Completed (Phase 11) ✅ — Workload-Aware Rate Tables + Multi-Objective Pareto

- ✅ Multi-model profiling: 7B/8B/14B, each 792 rows lock + 216 rows cap
- ✅ Rate tables built for all 3 models (lock + cap + DVFS rules)
- ✅ Cross-model DVFS comparison: 7B compute-bound (36% DVFS space), 8B memory-flat (17%), 14B compute-bound (23%)
- ✅ Multi-objective Pareto selector (`src/controller/pareto_selector.py`): non-dominated sorting, knee detection, 5 strategies
- ✅ Pareto integrated into `WorkloadCapSelector` as `strategy='pareto'`
- ✅ Pareto rank pre-computation in rate table builder
- ✅ 6 Pareto visualization charts + 4 cross-model comparison charts

### Completed (Phase 12) ✅ — E2E Cap Selector Benchmark

- ✅ 378 runs total: 3 models × 9 strategies × 5 workloads × 3 repeats, 0 errors
- ✅ Strategies: pareto, min_energy, slo_50ms, slo_45ms, alpha_03, alpha_07, pwr_45w, dynamic, maxn
- ✅ Pareto achieves +0.8%~+1.8% E/tok savings vs dynamic, +4.5%~+5.5% vs MAXN
- ✅ Power savings up to **22.8%** (7B Pareto vs MAXN: 47W → 38W)
- ✅ 5 analysis charts + detailed report in `figures/e2e_benchmark/`

### Completed (Phase 13) 🔧 — Long-running Serving + Thermal (Code Complete, Awaiting Hardware)

**P0: Oracle Gap Analysis** ✅
- Oracle definitions: Oracle-Energy (min E/tok), Oracle-SLO (TPOT-constrained), Oracle-Power (power-constrained)
- Key result: Pareto → Oracle gap = **+4.9%** average (8B: +0.7%, 7B: +4.4%, 14B: +9.4%)
- Confirms single-request DVFS optimization space is fundamentally limited
- File: `src/ratetable/oracle_gap_analysis.py` (502 lines)
- Data: `data/oracle_gap_analysis/oracle_gap_*.csv`

**P1: Fine-grained Cap Profiling** 🔧 (needs hardware)
- 10 GPU caps [408..1300] instead of 4, richer Pareto frontier
- Checkpoint/resume support for 3h experiments
- File: `src/experiments/run_finegrained_cap_profiling.py` (500 lines)

**P2: Thermal-SLO Feedback Controller** ✅ (synthetic trace validated)
- Window-based (10s) feedback with 6-priority control rules
- Hysteresis (K=3 consecutive windows) prevents oscillation
- Thermal protection raises (not lowers) frequency: high temp → fast completion → idle cool
- File: `src/controller/thermal_slo_controller.py` (693 lines)

**P3: Long-running Serving Benchmark** 🔧 (needs hardware)
- 5 baselines (MAXN/Dynamic/BestStatic/Pareto/ThermalSLO) × 3 traces
- Per-window metrics: TPOT, power, temperature, SLO violation, tokens/J
- File: `src/experiments/run_serving_benchmark.py` (732 lines)

**P4: Phase 13 Visualization** ✅ (oracle gap chart generated, others await serving data)
- 6 charts: Oracle Gap, Temperature, Power/TPOT, Cap Timeline, Dashboard, Radar
- File: `src/visualization/visualize_phase13.py` (520 lines)

## Technical Decisions

### Platform Selection
**Chosen**: Jetson Orin
**Reasoning**: Latest generation edge GPU with best performance for this research

### Runtime Selection
**Chosen**: llama.cpp (validated), TensorRT-LLM (planned)
**Reasoning**: llama.cpp works on Jetson out-of-box with GGUF models; TensorRT-LLM provides best throughput

### Experiment Scope
**Chosen**: Comprehensive (all 7 preliminary experiments)
**Reasoning**: Full validation of assumptions before committing to complete system implementation

### Frequency Control Strategy
**Chosen**: Hybrid (read via sysfs, set via jetson_clocks)
**Reasoning**: sysfs reads don't require sudo; jetson_clocks is the official NVIDIA tool

### Phase-Aware DVFS Strategy
**Chosen**: High GPU freq for prefill, high EMC freq for decode
**Reasoning**: Experiment 4.4 showed 26.97x phase efficiency difference; prefill is compute-intensive, decode is memory-intensive

## Key Technical Challenges

### 1. Frequency Control
**Challenge**: Setting GPU/CPU/EMC frequencies reliably
**Status**: ✅ Implemented (hybrid sysfs + jetson_clocks)
**Key finding**: GPU devfreq `userspace` governor fails under load (GPU jumps to max freq). Must use `performance` governor for reliable locking. Corrected data shows GPU 306→918 MHz yields 2.5× throughput improvement.

### 2. Metrics Collection
**Challenge**: Parsing tegrastats output reliably
**Status**: ✅ Implemented (system_monitor.py + metrics_collector.py)

### 3. Benchmark Integration
**Challenge**: Integrating with LLM runtime benchmark
**Status**: ✅ Synthetic benchmark working, llama.cpp integration tested

### 4. Measurement Stability
**Challenge**: Ensuring consistent measurements across repeated runs
**Status**: ✅ Validated (TPOT CV=1.0%, TTFT CV=9.2%)

## Development Guidelines

### Before Making Changes
1. Read `docs/当前状态.md` to understand current implementation status
2. Consult `docs/checklist.md` for modification guidelines
3. Review `docs/任务书.md` to ensure alignment with project goals
4. Check if the change affects critical decisions listed in `当前状态.md`

### Code Implementation
1. Follow PEP 8 standards
2. Use type hints for function signatures
3. Add comprehensive docstrings (Google style)
4. Implement proper error handling and logging
5. Write unit tests for core functionality

### Experiment Design
1. Define clear objectives and success criteria
2. Use appropriate statistical methods
3. Ensure reproducibility (same conditions, multiple runs)
4. Document experimental setup and parameters
5. Analyze results and draw data-driven conclusions

### Configuration Management
1. Use YAML format for configuration files
2. Add clear comments explaining each parameter
3. Validate configuration values
4. Provide sensible defaults
5. Document configuration purpose and usage

## Common Patterns and Utilities

### Frequency Control Pattern
```python
class FrequencyController:
    def __init__(self, platform_config):
        # Initialize based on platform configuration
        pass
    
    def get_frequency(self, target: str) -> int:
        """Get current frequency for target (gpu/cpu/emc)"""
        pass
    
    def set_frequency(self, freq: int, target: str) -> bool:
        """Set frequency for target, return success status"""
        pass
    
    def save_defaults(self):
        """Save current frequencies as defaults"""
        pass
    
    def restore_defaults(self):
        """Restore saved default frequencies"""
        pass
```

### Metrics Collection Pattern
```python
class MetricsCollector:
    def __init__(self, interval_ms: int = 1000):
        """Initialize tegrastats with specified interval"""
        pass
    
    def start_collection(self) -> str:
        """Start background collection, return log file path"""
        pass
    
    def stop_collection(self):
        """Stop background collection and cleanup"""
        pass
    
    def parse_logs(self, log_file: str) -> pd.DataFrame:
        """Parse tegrastats logs into structured data"""
        pass
```

### Benchmark Running Pattern
```python
class BenchmarkRunner:
    def __init__(self, config):
        """Initialize with benchmark configuration"""
        pass
    
    def run_benchmark(self, workload: dict, frequency: dict) -> dict:
        """
        Run benchmark with specified workload and frequency
        Returns parsed results (TTFT, TPOT, throughput, etc.)
        """
        pass
    
    def parse_output(self, output: str) -> dict:
        """Parse benchmark output into structured metrics"""
        pass
```

### Sweep Orchestration Pattern
```python
class SweepRunner:
    def __init__(self, config):
        """Initialize with sweep configuration"""
        pass
    
    def run_sweep(self, workloads: list, frequencies: list) -> str:
        """
        Run complete sweep of workload × frequency combinations
        Returns directory containing all results
        """
        pass
    
    def run_single_config(self, workload: dict, frequency: dict) -> dict:
        """Run single configuration with warmup, measurement, and cooldown"""
        pass
```

## Debugging and Troubleshooting

### Common Issues

#### Frequency Control Issues
- **Problem**: Cannot set frequencies
- **Check**: sudo permissions, jetson_clocks availability
- **Debug**: `sudo jetson_clocks --show` to check current status

#### tegrastats Issues
- **Problem**: Cannot parse tegrastats output
- **Check**: JetPack version compatibility
- **Debug**: Run `tegrastats --interval 1000 --logfile test.log` manually

#### Benchmark Issues
- **Problem**: Benchmark fails or times out
- **Check**: Model engine availability, TensorRT-LLM installation
- **Debug**: Run benchmark command manually

### Debug Tools
```bash
# Check system status
sudo jetson_clocks --show
tegrastats --interval 1000 --logfile debug.log

# Check process status
ps aux | grep tegrastats
ps aux | grep benchmark

# Check system logs
journalctl -xe
dmesg | tail -n 50
```

## Performance Considerations

### Frequency Switching Overhead
- Switching between frequencies takes time and may cause temporary performance degradation
- Minimize frequency changes (hysteresis mechanism)
- Prefer switching at phase boundaries (prefill/decode) or request boundaries

### Data Collection Overhead
- tegrastats sampling frequency: 1Hz is sufficient for most experiments
- Avoid overly frequent sampling to reduce CPU overhead
- Use efficient data structures for log storage

### Experiment Duration
- Single configuration: ~30 seconds (including warmup, measurement, cooldown)
- Full sweep: 27 configs × 3 workloads × 10 repeats = ~6.75 hours
- Implement incremental execution and resume capability

## Success Criteria

### Foundation Phase (Current)
- ✅ All documentation files created and reviewed
- ✅ Directory structure matches planned layout
- ⏳ CLAUDE.md created with project context

### Core Module Phase
- ❌ Frequency controller can read/set/restore frequencies reliably
- ❌ Metrics collector runs tegrastats and parses output correctly
- ❌ Benchmark runner executes benchmarks and parses TTFT/TPOT/throughput
- ❌ Sweep runner can complete a small sweep
- ❌ Log parser aligns logs and generates structured CSV

### Preliminary Experiments Phase
- ❌ Experiment 4.1: Measurement stability verified
- ❌ Experiment 4.2: Single-knob heatmaps show different impact patterns
- ❌ Experiment 4.3: Pareto frontier extracted, mid-high config outperforms max-all
- ❌ Experiment 4.4: Prefill and decode show different optimal configs
- ❌ Experiment 4.5: Predictability models trained, best model selected
- ❌ Experiment 4.6: Switching overhead quantified (< threshold)
- ❌ Experiment 4.7: End-to-end SLO validation shows improvement over baselines

## Key Decision Points

### Decision 1: Frequency Control Method
**Status**: ✅ Decided - Hybrid (sysfs read + jetson_clocks set)
**Result**: Works reliably on Jetson Orin

### Decision 2: Runtime Selection
**Status**: ✅ Decided - llama.cpp for prototyping, TensorRT-LLM for production
**Result**: llama.cpp verified with Phi-3-mini Q4 model

### Decision 3: Phase-Aware DVFS Validation
**Status**: ✅ Validated - 30% energy saving + 36% TTFT improvement
**Result**: Phase-Aware strategy significantly outperforms fixed strategies

### Decision 4: Next Optimization Direction
**Status**: Decided - Real model baseline comparison
**Result**: Infrastructure complete (llama_cpp_runner, jetson_power_modes, baselines.yaml), ready for real hardware experiment

## Project Dependencies

### System Requirements
- **Hardware**: Jetson Orin device
- **Software**: JetPack SDK (with jetson_clocks, tegrastats)
- **Runtime**: TensorRT-LLM
- **Language**: Python 3.8+

### Python Dependencies
```python
pyyaml           # Configuration parsing
pandas           # Data manipulation
numpy            # Numerical operations
matplotlib       # Plotting
seaborn          # Statistical visualization
scipy            # Statistical analysis
scikit-learn     # ML models
xgboost          # Gradient boosting
```

### System Dependencies
- `jetson_clocks` - Frequency control
- `tegrastats` - System metrics monitoring
- TensorRT-LLM benchmark tools

## File Naming Conventions

### Source Files
- Python modules: `module_name.py` (lowercase with underscores)
- Classes: `ClassName` (PascalCase)
- Functions: `function_name()` (lowercase with underscores)
- Constants: `CONSTANT_NAME` (uppercase with underscores)

### Data Files
- Raw logs: `experiment_id_timestamp_raw.log`
- Parsed data: `experiment_id_timestamp_parsed.parquet`
- Results: `experiment_id_timestamp_results.csv`
- Plots: `experiment_id_timestamp_plot_type.png`

### Configuration Files
- Platform: `configs/platform.yaml`
- Workloads: `configs/workloads.yaml`
- Frequencies: `configs/frequencies.yaml`
- Sweep: `configs/sweep.yaml`
- SLO: `configs/slo.yaml`

## Logging and Debugging

### Log Levels
- **INFO**: Normal operation progress
- **WARNING**: Non-critical issues that don't affect functionality
- **ERROR**: Critical errors that prevent operation
- **DEBUG**: Detailed diagnostic information

### Log Format
```
timestamp  level  module  message
2025-05-06 10:30:00  INFO  freq_controller  Setting GPU frequency to 1428 MHz
2025-05-06 10:30:01  ERROR  benchmark_runner  Benchmark failed: timeout after 300s
```

### Key Logging Locations
- `data/raw_logs/` - Raw tegrastats and benchmark logs
- Script execution logs in `scripts/`
- Python module logs to stdout/stderr

## Testing Strategy

### Unit Tests
- Test individual functions and methods
- Mock external dependencies (system calls, processes)
- Focus on core functionality and edge cases

### Integration Tests
- Test module interactions
- Test with real system components (when available)
- Test complete workflows (e.g., single configuration run)

### Validation Tests
- Verify measurement stability (repeat runs, analyze variance)
- Validate against known baselines
- Statistical significance testing

## Project Collaboration

### Role Suggestions
- **Infrastructure Developer**: freq_controller, metrics_collector, parse_logs
- **Experiment Designer**: benchmark_runner, sweep_runner, experiment planning
- **Data Analyst**: build_rate_table, pareto, select_config
- **System Integrator**: online_controller, end-to-end testing

### Communication
- Use project documentation as single source of truth
- Update `当前状态.md` with progress and blockers
- Document decisions and rationale
- Use Git for version control with clear commit messages

## Next Immediate Steps

1. **Phase 13**: Long-running serving evaluation with temperature-aware frequency adjustment
2. **Research paper**: Write paper from Phase 10-12 results (3 models, Pareto DVFS, E2E validation)
3. **Jetson runtime integration**: Deploy cap selector as a llama.cpp wrapper or middleware

## Important Notes

- This is a research project with experimental validation requirements
- Success depends on validation of key assumptions (measurement stability, optimization space)
- Plan includes decision points to pivot based on experimental results
- Emphasis on reproducibility and detailed documentation
- Modular design allows for incremental development and testing
- **Key finding**: Model size fundamentally changes DVFS behavior (3.8B: W-shape sweet spot, 14B: monotonic improvement)
- EMC debugfs control is currently non-functional (clk_rate ignores min/max_rate)

---

**Last Updated**: 2026-06-04
**AI Assistant Notes**: Phase 1-12 complete. Multi-objective Pareto DVFS selector implemented and validated across 3 models (7B/8B/14B, 378 E2E runs). Pareto achieves up to +5.5% E/tok savings vs MAXN and up to +22.8% power reduction vs dynamic on 7B. Available models: Phi-3-mini (3.8B), Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B.