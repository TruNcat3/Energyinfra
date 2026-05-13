# Claude AI Assistant Documentation

## Project Overview

**Project Name**: Jetson-LLM Energy Rate Table and Adaptive DVFS Selector

**Goal**: Build an energy profiling and adaptive configuration selection system for LLM inference on Jetson edge GPUs. The system automatically profiles different workload × frequency configurations, builds an energy rate table, and provides SLO-aware frequency selection for GPU/CPU/EMC to optimize energy efficiency under latency, power, and thermal constraints.

**Platform**: Jetson Orin with llama.cpp runtime (TensorRT-LLM planned)

**Current Phase**: Phase 5 In Progress - Real model experiment completed (180 runs), P6 rate table rebuild pending

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
│   │   ├── jetson_power_modes.py    # nvpmodel/jetson_clocks management
│   │   ├── phase_aware_policy.py    # Phase-Aware DVFS policy
│   │   ├── phase_controller.py      # DVFS online controller
│   │   └── select_config.py         # SLO-Aware config selector
│   ├── metrics/               # Metrics Collection
│   │   ├── metrics_collector.py     # tegrastats integration
│   │   ├── parse_logs.py            # Log parsing
│   │   └── system_monitor.py        # System monitoring
│   ├── benchmark/             # Benchmark Framework
│   │   ├── synthetic_benchmark.py   # Synthetic (no real model needed)
│   │   ├── benchmark_runner.py      # Benchmark execution
│   │   ├── sweep_runner.py          # Sweep orchestration
│   │   └── llama_cpp_runner.py      # llama.cpp inference runner
│   ├── ratetable/             # Rate Table
│   │   ├── build_rate_table.py      # Energy rate table builder
│   │   └── evaluate_selector.py     # Selector evaluation tool
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

### In Progress (Phase 5)
- ✅ P0: Evaluation credibility fixes (Pareto, phase-specific energy, fixed_best/oracle/regret)
- ✅ P1: Jetson baseline infrastructure (jetson_power_modes.py, baselines.yaml)
- ✅ P2: Real model runner (llama_cpp_runner.py, real_model_workloads.yaml)
- ✅ P3: Adaptive phase-aware policy connected to controller
- ✅ P4: Rate table rebuilt, evaluation verified (0 anomalies, 0% negative regret)
- ✅ Phase 5 comparison visualizations (6 charts)
- ✅ Run real model multi-config experiment (180 runs, 4 GPU × 3 CPU × 5 workloads × 3 repeats)
- ✅ Real model experiment analysis and visualizations (5 charts)
- ⏳ Run real model baseline comparison experiment (9 baselines)
- ⏳ Rebuild rate table with real data
- ⏳ Hybrid intelligent scheduling (rule-based + ML)
- ⏳ Long-term stability validation

### Key Experimental Results

| Strategy | Energy/Token | TTFT | TPOT | SLO Rate |
|----------|-------------|------|------|----------|
| Default (mid) | 0.164 J | 121 ms | 7.6 ms | 82% |
| Max Performance | 0.110 J | 124 ms | 4.5 ms | 42% |
| **Phase-Aware** | **0.115 J** | **77 ms** | **6.7 ms** | **80%** |
| Oracle | 0.152 J | 124 ms | 6.9 ms | 76% |

### Real Model Experiment Results (2026-05-13)

| Metric | Finding |
|--------|---------|
| GPU freq scaling | 306→1300 MHz = **1.01x** throughput (no effect) |
| CPU freq scaling | 1036→2201 MHz = **1.86x** throughput (dominant) |
| Best config | GPU918_CPU2201 = **9.4 tok/s** |
| Bottleneck | **CPU-bound** (memory-bound workload confirmed) |
| Zero-token anomalies | 35/180 (19.4%), filtered in analysis |
| TTFT CPU effect | 26.2% reduction at high CPU freq |
| TTFT GPU effect | 0.8% reduction (negligible) |

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
**Key finding**: GPU frequency 4.25x increase → only 8.3% performance gain (memory-bound)

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

1. **Run real model baseline comparison** - 5 workloads x 9 baselines x 5 repeats
2. **Rebuild rate table with real data** - Compare synthetic vs real conclusions
3. **Online controller deployment** - Real-time adaptive frequency control
4. **Long-term stability testing** - Validate over extended runs
5. **Production-grade optimization** - Thermal effects, batching, concurrency

## Important Notes

- This is a research project with experimental validation requirements
- Success depends on validation of key assumptions (measurement stability, optimization space)
- Plan includes decision points to pivot based on experimental results
- Emphasis on reproducibility and detailed documentation
- Modular design allows for incremental development and testing

---

**Last Updated**: 2026-05-13
**AI Assistant Notes**: Phase 1-4 complete. Phase 5 P0-P5 complete. Real model experiment (180 runs) confirmed: GPU frequency has NO effect on throughput (memory-bound), CPU frequency is the dominant bottleneck (1.86x scaling). Next: rebuild rate table with real data (P6), baseline comparison experiment.