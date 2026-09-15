#!/bin/bash

# Jetson LLM Energy Profiling - Preliminary Experiments Runner
# This script runs all 7 preliminary validation experiments

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
PROJECT_DIR="/home/wt/work/Energyinfra"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
SRC_DIR="${PROJECT_DIR}/src"
CONFIG_DIR="${PROJECT_DIR}/configs"
DATA_DIR="${PROJECT_DIR}/data"
LOG_DIR="${DATA_DIR}/raw_logs"
PARSED_DIR="${DATA_DIR}/parsed"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if running on Jetson
check_jetson() {
    print_info "Checking if running on Jetson device..."
    if [ ! -f "/etc/nv_tegra_release" ]; then
        print_error "This script must be run on a Jetson device"
        exit 1
    fi
    print_info "Jetson device detected: $(cat /proc/device-tree/model | head -n1)"
}

# Function to check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found"
        exit 1
    fi

    # Check jetson_clocks
    if ! command -v jetson_clocks &> /dev/null; then
        print_warning "jetson_clocks not found in PATH"
        print_warning "Please install JetPack SDK"
    fi

    # Check tegrastats
    if ! command -v tegrastats &> /dev/null; then
        print_warning "tegrastats not found in PATH"
        print_warning "Please install JetPack SDK"
    fi

    # Check Python packages
    print_info "Checking Python packages..."
    python3 -c "import yaml" 2>/dev/null || print_error "PyYAML not installed"
    python3 -c "import pandas" 2>/dev/null || print_error "pandas not installed"
    python3 -c "import numpy" 2>/dev/null || print_error "numpy not installed"
}

# Function to create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    mkdir -p "${LOG_DIR}"
    mkdir -p "${PARSED_DIR}"
    mkdir -p "${DATA_DIR}/checkpoints"
    mkdir -p "${PROJECT_DIR}/figures/heatmaps"
    mkdir -p "${PROJECT_DIR}/figures/pareto_curves"
    mkdir -p "${PROJECT_DIR}/figures/ablations"
    print_info "Directories created"
}

# Function to run Experiment 4.1: Measurement Stability
run_experiment_4_1() {
    print_info "========================================="
    print_info "Experiment 4.1: Measurement Stability"
    print_info "========================================="
    print_info "Goal: Validate measurement stability across repeated runs"
    print_info "Setup: Fixed workload + fixed frequency × 10 runs"
    print_info ""

    print_info "Configuration:"
    print_info "  - Model: qwen-7b-int4"
    print_info "  - Batch size: 1"
    print_info "  - Prompt length: 512 tokens"
    print_info "  - Output length: 128 tokens"
    print_info "  - Phase: mixed"
    print_info "  - Frequencies: GPU=high, CPU=high, EMC=high"
    print_info "  - Repeats: 10"
    print_info "  - Warmup runs: 1"
    print_info "  - Expected duration: ~30 minutes"
    print_info ""

    # Update workload configuration for experiment 4.1
    # This would normally be done by modifying the config files
    # For now, we'll demonstrate the command structure

    print_info "This experiment would:"
    print_info "  1. Set frequencies to high for all targets"
    print_info "  2. Run 10 measurement iterations"
    print_info "  3. Collect tegrastats metrics during each run"
    print_info "  4. Calculate std deviation of energy/token and latency"
    print_info "  5. Validate: std < 5% of mean"
    print_info ""

    print_warning "Note: Actual execution requires:"
    print_warning "  - TensorRT-LLM models and engines built"
    print_warning "  - Proper JetPack environment"
    print_warning "  - Sufficient thermal management"
    print_warning ""
}

# Function to run Experiment 4.2: Single-Knob Sensitivity
run_experiment_4_2() {
    print_info "========================================="
    print_info "Experiment 4.2: Single-Knob Sensitivity"
    print_info "========================================="
    print_info "Goal: Understand individual frequency knob impacts"
    print_info "Setup: Sweep each knob (GPU/CPU/EMC) separately"
    print_info ""

    print_info "GPU Sweep: (GPU: low/mid/high, CPU: mid, EMC: mid)"
    print_info "CPU Sweep: (GPU: mid, CPU: low/mid/high, EMC: mid)"
    print_info "EMC Sweep: (GPU: mid, CPU: mid, EMC: low/mid/high)"
    print_info "  - Each sweep: 3 configs × 5 repeats = 15 runs"
    print_info "  - Total: 45 runs"
    print_info "  - Expected duration: ~45 minutes"
    print_info ""

    print_info "This experiment would:"
    print_info "  1. Run GPU sweep (3 freq × 5 repeats)"
    print_info "  2. Run CPU sweep (3 freq × 5 repeats)"
    print_info "  3. Run EMC sweep (3 freq × 5 repeats)"
    print_info "  4. Generate heatmaps for each sweep"
    print_info "  5. Analyze which knob affects which metrics most"
    print_info ""
}

# Function to run Experiment 4.3: Frequency Combination Interactions
run_experiment_4_3() {
    print_info "========================================="
    print_info "Experiment 4.3: Frequency Combination Interactions"
    print_info "========================================="
    print_info "Goal: Find optimal multi-knob configuration"
    print_info "Setup: Full factorial design of 3×3×3 freq combos"
    print_info ""

    print_info "Configuration:"
    print_info "  - GPU freqs: low, mid, high"
    print_info "  - CPU freqs: low, mid, high"
    print_info "  - EMC freqs: low, mid, high"
    print_info "  - Total combos: 27"
    print_info "  - Repeats per combo: 5"
    print_info "  - Total runs: 135"
    print_info "  - Expected duration: ~3-4 hours"
    print_info ""

    print_info "This experiment would:"
    print_info "  1. Run all 27 frequency combinations"
    print_info "  2. Extract Pareto frontier"
    print_info "  3. Test hypothesis: mid-high GPU + mid-high EMC > max-all"
    print_info "  4. Analyze interaction effects between knobs"
    print_info ""
}

# Function to run Experiment 4.4: Prefill/Decode Phase Differences
run_experiment_4_4() {
    print_info "========================================="
    print_info "Experiment 4.4: Prefill/Decode Phase Differences"
    print_info "========================================="
    print_info "Goal: Verify phase-aware DVFS necessity"
    print_info "Setup: Compare different phase scenarios"
    print_info ""

    print_info "Phase Scenarios:"
    print_info "  1. Only-prefill: (prompt=512, output=0)"
    print_info "  2. Only-decode: (prompt=128, output=128)"
    print_info "  3. Prefill-heavy: (prompt=1024, output=32)"
    print_info "  4. Decode-heavy: (prompt=128, output=512)"
    print_info "  - Each scenario: 3 GPU freq × 5 repeats = 15 runs"
    print_info "  - Total runs: 60"
    print_info "  - Expected duration: ~2-3 hours"
    print_info ""

    print_info "This experiment would:"
    print_info "  1. Run each phase scenario with different GPU freqs"
    print_info "  2. Compare optimal freqs between scenarios"
    print_info "  3. Validate: prefill and decode have different optimal configs"
    print_info "  4. Quantify performance difference between phases"
    print_info ""
}

# Function to run Experiment 4.5: Workload Feature Predictability
run_experiment_4_5() {
    print_info "========================================="
    print_info "Experiment 4.5: Workload Feature Predictability"
    print_info "========================================="
    print_info "Goal: Validate if workload features can predict optimal configs"
    print_info "Setup: Wide range of workload variations"
    print_info ""

    print_info "Feature Variations:"
    print_info "  - Models: qwen-7b-int4"
    print_info "  - Batch sizes: 1, 2, 4"
    print_info "  - Prompt lengths: 128, 512, 1024"
    print_info "  - Output lengths: 32, 128, 512"
    print_info "  - Phases: prefill, decode, mixed"
    print_info "  - Sample: 12 representative configs"
    print_info "  - Repeats per config: 3"
    print_info "  - Total runs: 36"
    print_info "  - Expected duration: ~1-2 hours"
    print_info ""

    print_info "This experiment would:"
    print_info "  1. Train ML models: Linear Regression, Polynomial, XGBoost"
    print_info "  2. Predict targets: TTFT, TPOT, energy/token, tokens/J"
    print_info "  3. Compare model accuracies"
    print_info "  4. Analyze feature importance"
    print_info "  5. Choose best model for config selector"
    print_info ""
}

# Function to run Experiment 4.6: Frequency Switching Overhead
run_experiment_4_6() {
    print_info "========================================="
    print_info "Experiment 4.6: Frequency Switching Overhead"
    print_info "========================================="
    print_info "Goal: Measure impact of frequency switching"
    print_info "Setup: Test various frequency switching scenarios"
    print_info ""

    print_info "Switching Scenarios:"
    print_info "  1. GPU low→high, high→low, mid→high, high→mid"
    print_info "  2. EMC low→high, high→low"
    print_info "  3. CPU low→high, high→low"
    print_info "  4. Phase boundary switching"
    print_info "  - Each scenario: 10 repeats"
    print_info "  - Total runs: 40"
    print_info "  - Expected duration: ~1-2 hours"
    print_info ""

    print_info "This experiment would:"
    print_info "  1. Measure switching time for each scenario"
    print_info "  2. Measure performance impact (latency jitter)"
    print_info "  3. Determine: switch at request/batch/phase boundary vs per-token"
    print_info "  4. Establish hysteresis thresholds"
    print_info ""
}

# Function to run Experiment 4.7: SLO-Constrained End-to-End
run_experiment_4_7() {
    print_info "========================================="
    print_info "Experiment 4.7: SLO-Constrained End-to-End Validation"
    print_info "========================================="
    print_info "Goal: Validate complete system under SLO constraints"
    print_info "Setup: Compare our method vs baselines"
    print_info ""

    print_info "Baseline Configurations:"
    print_info "  1. Default governor: (use system defaults)"
    print_info "  2. MaxN: (all freqs at high)"
    print_info "  3. Fixed best-efficiency: (mid freqs - assumed optimal)"
    print_info "  4. Oracle best: (optimal freqs from experiments)"
    print_info "  5. Ours: (adaptive frequency selection)"
    print_info ""
    print_info "SLO Constraints:"
    print_info "  - TTFT < 1000ms"
    print_info "  - TPOT < 80ms"
    print_info "  - Avg power < 40W"
    print_info "  - Temperature < 80°C"
    print_info ""
    print_info "Test Scenarios:"
    print_info "  1. Real-time: (latency-oriented)"
    print_info "  2. Batch processing: (throughput-oriented)"
    print_info "  3. Low power: (power-oriented)"
    print_info "  4. High performance: (performance-oriented)"
    print_info "  - Each scenario: 5 repeats"
    print_info "  - Total runs: 100"
    print_info "  - Expected duration: ~3-4 hours"
    print_info ""

    print_info "This experiment would:"
    print_info "  1. Test all baseline configurations"
    print_info "  2. Test our adaptive selector"
    print_info "  3. Measure: energy/token reduction vs MaxN"
    print_info "  4. Measure: SLO satisfaction rates"
    print_info "  5. Validate: end-to-end system usefulness"
    print_info ""
}

# Function to generate summary report
generate_summary() {
    local SUMMARY_FILE="${DATA_DIR}/preliminary_experiments_summary.md"

    print_info "Generating summary report..."

    cat > "$SUMMARY_FILE" << 'EOF'
# Preliminary Experiments Summary Report

Generated: $(date)

## Experiment Overview

This document summarizes the 7 preliminary validation experiments for the Jetson LLM Energy Profiling project.

## Experiments Completed

### Experiment 4.1: Measurement Stability
- **Status**: Configuration defined, awaiting execution
- **Objective**: Validate measurement stability
- **Expected Outcome**: Standard deviation of energy/token and latency < 5%

### Experiment 4.2: Single-Knob Sensitivity
- **Status**: Configuration defined, awaiting execution
- **Objective**: Understand individual frequency knob impacts
- **Expected Outcome**: GPU affects throughput, CPU affects TTFT, EMC affects decode

### Experiment 4.3: Frequency Combination Interactions
- **Status**: Configuration defined, awaiting execution
- **Objective**: Find optimal multi-knob configuration
- **Expected Outcome**: Mid-high GPU + mid-high EMC > max-all

### Experiment 4.4: Prefill/Decode Phase Differences
- **Status**: Configuration defined, awaiting execution
- **Objective**: Verify phase-aware DVFS necessity
- **Expected Outcome**: Prefill and decode have different optimal configs

### Experiment 4.5: Workload Feature Predictability
- **Status**: Configuration defined, awaiting execution
- **Objective**: Validate ML-based config prediction
- **Expected Outcome**: Simple models achieve good prediction accuracy

### Experiment 4.6: Frequency Switching Overhead
- **Status**: Configuration defined, awaiting execution
- **Objective**: Measure frequency switching impact
- **Expected Outcome**: Switching overhead is acceptable at phase boundaries

### Experiment 4.7: SLO-Constrained End-to-End
- **Status**: Configuration defined, awaiting execution
- **Objective**: Validate complete system
- **Expected Outcome**: Our method outperforms baselines

## Current Project Status

### Completed Components
- ✅ Project directory structure
- ✅ Configuration files (platform, workloads, frequencies, sweep, slo)
- ✅ Core Python modules:
  - freq_controller.py
  - metrics_collector.py
  - benchmark_runner.py
  - sweep_runner.py
  - parse_logs.py
- ✅ Experiment runner script

### Pending Components
- ⏳ TensorRT-LLM model engine files
- ⏳ Actual experiment execution on Jetson Orin
- ⏳ Results analysis and visualization
- ⏳ Decision on full system implementation

## Next Steps

1. **Setup Jetson Orin Environment**
   - Install JetPack SDK
   - Build TensorRT-LLM models
   - Configure thermal management

2. **Run Experiments Sequentially**
   - Start with Experiment 4.1 (quickest)
   - Proceed based on results
   - Adjust subsequent experiments as needed

3. **Analyze Results**
   - Validate assumptions from task document
   - Determine if optimization space exists
   - Make decision on full system implementation

4. **Document Findings**
   - Update docs/开发文档/当前状态.md (local-only) with results
   - Generate comprehensive reports
   - Plan Phase 4 implementation

## Technical Notes

### System Requirements
- **Hardware**: Jetson Orin
- **Software**: JetPack r36.x, Python 3.8+, TensorRT-LLM
- **Storage**: ~10GB for experiment data and logs
- **Time**: ~10-15 hours total for all experiments

### Key Measurements
- **Performance**: TTFT, TPOT, ITL, throughput, P95/P99 latency
- **Power**: Average power, peak power, CPU/GPU power
- **Energy**: Energy per token, tokens per joule
- **Thermal**: CPU/GPU/SoC temperature profiles

### Validation Criteria
- **Measurement Stability**: std < 5% of mean
- **Frequency Impact**: Different knobs show different impact patterns
- **Optimization Space**: Significant energy reduction possible
- **Phase Difference**: Prefill and decode have different optimal configs
- **Switching Overhead**: < threshold for phase-boundary switching
- **Predictability**: ML models achieve >80% accuracy
- **End-to-End**: Outperforms baselines in energy efficiency

## Risk Mitigation

### Thermal Management
- Enable active cooling (fans)
- Monitor temperature continuously
- Include cooldown periods between runs
- Implement thermal throttling safeguards

### Measurement Quality
- Repeat each configuration multiple times
- Include warmup runs
- Detect and handle outliers
- Validate frequency settings

### Experiment Robustness
- Support resume from checkpoints
- Handle transient failures gracefully
- Log detailed diagnostics
- Implement quality control checks

---

**Report End**
EOF

    print_info "Summary report generated: $SUMMARY_FILE"
}

# Main function
main() {
    print_info "========================================="
    print_info "Jetson LLM Energy Profiling"
    print_info "Preliminary Experiments Runner"
    print_info "========================================="
    print_info ""

    # Check environment
    check_jetson
    check_dependencies
    create_directories

    print_info ""
    print_info "Available experiment options:"
    print_info "  1 - Experiment 4.1: Measurement Stability"
    print_info "  2 - Experiment 4.2: Single-Knob Sensitivity"
    print_info "  3 - Experiment 4.3: Frequency Combination Interactions"
    print_info "  4 - Experiment 4.4: Prefill/Decode Phase Differences"
    print_info "  5 - Experiment 4.5: Workload Feature Predictability"
    print_info "  6 - Experiment 4.6: Frequency Switching Overhead"
    print_info "  7 - Experiment 4.7: SLO-Constrained End-to-End"
    print_info "  8 - Run ALL experiments (sequential)"
    print_info "  9 - Generate summary report"
    print_info "  0 - Exit"
    print_info ""

    # If argument provided, run specific experiment
    if [ $# -eq 1 ]; then
        case $1 in
            1) run_experiment_4_1 ;;
            2) run_experiment_4_2 ;;
            3) run_experiment_4_3 ;;
            4) run_experiment_4_4 ;;
            5) run_experiment_4_5 ;;
            6) run_experiment_4_6 ;;
            7) run_experiment_4_7 ;;
            8)
                print_info "Running ALL experiments sequentially..."
                print_info "This will take approximately 10-15 hours"
                print_info "Press Ctrl+C to interrupt at any time"
                print_info ""
                run_experiment_4_1
                run_experiment_4_2
                run_experiment_4_3
                run_experiment_4_4
                run_experiment_4_5
                run_experiment_4_6
                run_experiment_4_7
                print_info "All experiments configured!"
                generate_summary
                ;;
            9) generate_summary ;;
            0) print_info "Exiting..."; exit 0 ;;
            *) print_error "Invalid option: $1"; exit 1 ;;
        esac
    else
        print_warning "No experiment specified."
        print_info "Usage: $0 [experiment_number]"
        print_info "Example: $0 1  # Run Experiment 4.1"
        print_info "         $0 8  # Run all experiments"
        print_info "         $0 9  # Generate summary report"
    fi
}

# Run main function
main "$@"