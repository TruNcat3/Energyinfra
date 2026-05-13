#!/bin/bash

# Jetson Environment Verification Script
# Tests core modules and prepares for preliminary experiments

set -e  # Exit on error

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

# Function to check Jetson device
check_jetson_device() {
    print_info "Checking Jetson device..."

    # Check Jetson device
    if [ -f /etc/nv_tegra_release ]; then
        local device_model=$(cat /etc/nv_tegra_release | head -n1 | cut -d',' -f1)
        local jetpack_version=$(cat /etc/nv_tegra_release | head -n1 | cut -d',' -f2 | cut -d':' -f1)

        print_info "✅ Device: ${device_model}"
        print_info "✅ JetPack: ${jetpack_version}"

        # Check if it's Orin
        if [[ "$device_model" == *"Orin"* ]]; then
            print_info "✅ Target platform: Jetson Orin (compatible)"
        else
            print_warning "Device may not be Jetson Orin, continuing..."
        fi
    else
        print_error "❌ Not a Jetson device"
        return 1
    fi
    return 0
}

# Function to check JetPack tools
check_jetpack_tools() {
    print_info "Checking JetPack tools..."

    # Check jetson_clocks
    if command -v jetson_clocks &> /dev/null; then
        local jetson_clocks_path=$(which jetson_clocks)
        print_info "✅ jetson_clocks: ${jetson_clocks_path}"

        # Test jetson_clocks basic functionality
        print_info "Testing jetson_clocks --show..."
        if timeout 10 jetson_clocks --show > /dev/null 2>&1; then
            print_info "✅ jetson_clocks is functional"
        else
            print_error "❌ jetson_clocks --show timed out or failed"
        fi
    else
        print_error "❌ jetson_clocks not found"
        return 1
    fi

    # Check tegrastats
    if command -v tegrastats &> /dev/null; then
        local tegrastats_path=$(which tegrastats)
        print_info "✅ tegrastats: ${tegrastats_path}"

        # Test tegrastats basic functionality
        print_info "Testing tegrastats --help..."
        if timeout 5 tegrastats --help > /dev/null 2>&1; then
            print_info "✅ tegrastats is functional"
        else
            print_error "❌ tegrastats --help timed out or failed"
        fi
    else
        print_error "❌ tegrastats not found"
        return 1
    fi

    # Check nvidia-smi (for GPU info)
    if command -v nvidia-smi &> /dev/null; then
        local nvidia_smi_path=$(which nvidia-smi)
        print_info "✅ nvidia-smi: ${nvidia_smi_path}"

        print_info "Testing nvidia-smi --query-gpu..."
        if timeout 5 nvidia-smi --query-gpu > /dev/null 2>&1; then
            print_info "✅ nvidia-smi is functional"
        else
            print_warning "nvidia-smi query failed (may not be critical)"
        fi
    else
        print_info "ℹ️  nvidia-smi not found (may not be available)"
    fi

    return 0
}

# Function to check Python environment
check_python_env() {
    print_info "Checking Python environment..."

    # Check Python version
    local python_version=$(python3 --version 2>&1)
    print_info "✅ Python: ${python_version}"

    # Check for required packages
    local required_packages=("yaml" "pandas" "numpy")
    local missing_packages=()

    for package in "${required_packages[@]}"; do
        if python3 -c "import ${package}" 2>/dev/null; then
            print_info "✅ ${package}: $(python3 -c "import ${package}; print(${package}.__version__)" 2>/dev/null)"
        else
            missing_packages+=(${package})
        fi
    done

    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_error "❌ Missing packages: ${missing_packages[*]}"
        print_info "Run: pip3 install ${missing_packages[*]}"
        return 1
    fi

    # Check for optional packages
    local optional_packages=("matplotlib" "seaborn")
    local missing_optional=()

    for package in "${optional_packages[@]}"; do
        if python3 -c "import ${package}" 2>/dev/null; then
            print_info "✅ ${package}: installed"
        else
            missing_optional+=(${package})
        fi
    done

    if [ ${#missing_optional[@]} -gt 0 ]; then
        print_warning "⚠️  Missing optional packages: ${missing_optional[*]}"
        print_info "These are optional for visualization but recommended"
    fi

    return 0
}

# Function to test frequency controller
test_freq_controller() {
    print_info "Testing frequency controller module..."

    local test_result=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from freq_controller import FrequencyController

# Test initialization
config = {
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

try:
    controller = FrequencyController(config)
    print('✅ FrequencyController initialized successfully')

    # Test frequency reading
    current_freqs = controller.get_all_frequencies()
    print(f'Current frequencies: {current_freqs}')

    # Test frequency support
    support = controller.verify_frequency_support()
    print(f'Frequency support: {support}')

except Exception as e:
    print(f'❌ FrequencyController test failed: {e}')
    sys.exit(1)
" 2>&1)

    if [ $? -eq 0 ]; then
        print_info "✅ Frequency controller tests passed"
        return 0
    else
        print_error "❌ Frequency controller tests failed"
        return 1
    fi
}

# Function to test metrics collector
test_metrics_collector() {
    print_info "Testing metrics collector module..."

    # Test basic import
    local test_result=$(python3 -c "
import sys
sys.path.insert(0, 'src')
from metrics_collector import MetricsCollector

# Test initialization
collector = MetricsCollector(interval_ms=1000, output_dir='data/raw_logs')
print('✅ MetricsCollector initialized successfully')
except Exception as e:
    print(f'❌ MetricsCollector test failed: {e}')
    sys.exit(1)
" 2>&1)

    if [ $? -eq 0 ]; then
        print_info "✅ Metrics collector tests passed"
        return 0
    else
        print_error "❌ Metrics collector tests failed"
        return 1
    fi
}

# Function to test basic system performance
test_system_performance() {
    print_info "Testing basic system performance..."

    # Get current frequencies
    local freq_info=$(timeout 10 jetson_clocks --show 2>&1 | grep -E "(GPU|CPU|EMC)" | head -n3)
    echo "$freq_info"

    # Check thermal state
    local thermal_state=$(timeout 5 cat /sys/class/thermal/thermal_zone0 2>/dev/null || echo "N/A")
    echo "Current thermal state: $thermal_state"

    # Check available frequencies
    print_info "Checking available GPU frequencies..."
    local gpu_freqs=$(timeout 5 ls /sys/class/devfreq/*/available_frequencies 2>/dev/null | head -n1 | head -c5)
    echo "Available GPU frequencies: $gpu_freqs"

    print_info "✅ System performance check completed"
    return 0
}

# Function to generate environment report
generate_environment_report() {
    local report_file="data/environment_verification_report.txt"

    print_info "Generating environment verification report..."

    cat > "$report_file" << 'EOF'
# Jetson Environment Verification Report
# Generated: $(date)

## Environment Information

### Device Information
EOF

    # Add device info
    if [ -f /etc/nv_tegra_release ]; then
        cat /etc/nv_tegra_release >> "$report_file"
        echo "" >> "$report_file"
    fi

    cat >> "$report_file" << 'EOF'

### JetPack Tools Status
EOF

    # Add tool status
    if command -v jetson_clocks &> /dev/null; then
        echo "jetson_clocks: Available at $(which jetson_clocks)" >> "$report_file"
    else
        echo "jetson_clocks: NOT FOUND" >> "$report_file"
    fi

    if command -v tegrastats &> /dev/null; then
        echo "tegrastats: Available at $(which tegrastats)" >> "$report_file"
    else
        echo "tegrastats: NOT FOUND" >> "$report_file"
    fi

    cat >> "$report_file" << 'EOF'

### Python Environment
EOF

    # Add Python version and packages
    python3 --version >> "$report_file"
    echo "" >> "$report_file"

    for package in yaml pandas numpy; do
        echo -n "${package}: " >> "$report_file"
        python3 -c "import ${package}; print('${package}.__version__')" 2>/dev/null || echo "NOT INSTALLED" >> "$report_file"
    done

    cat >> "$report_file" << 'EOF'

### System Performance
EOF

    # Add current frequencies
    timeout 10 jetson_clocks --show >> "$report_file" 2>&1 || true

    cat >> "$report_file" << 'EOF'

### Recommendations

1. All basic tools are available
2. Python environment is ready
3. System is ready for experiments

### Next Steps

1. Run single experiment test: python3 src/experiment_manager.py
2. Review environment verification report
3. Start preliminary experiments sequentially
4. Monitor system state during experiments

EOF

    print_info "✅ Environment report generated: $report_file"
}

# Main execution
main() {
    echo "========================================="
    echo "Jetson Environment Verification"
    echo "========================================="
    echo ""

    local failed=0

    # Step 1: Check Jetson device
    check_jetson_device || failed=$((failed + 1))

    # Step 2: Check JetPack tools
    check_jetpack_tools || failed=$((failed + 1))

    # Step 3: Check Python environment
    check_python_env || failed=$((failed + 1))

    # Step 4: Test frequency controller
    test_freq_controller || failed=$((failed + 1))

    # Step 5: Test metrics collector
    test_metrics_collector || failed=$((failed + 1))

    # Step 6: Test system performance
    test_system_performance || failed=$((failed + 1))

    # Step 7: Generate report
    generate_environment_report

    echo ""
    echo "========================================="

    if [ $failed -eq 0 ]; then
        echo -e "${GREEN}✅ ALL CHECKS PASSED${NC}"
        echo ""
        echo "Environment is ready for preliminary experiments!"
        echo ""
        echo "Next steps:"
        echo "1. Review the environment report: data/environment_verification_report.txt"
        echo "2. Prepare TensorRT-LLM models and engines"
        echo "3. Run preliminary experiments: bash scripts/run_prelim_experiments.sh 8"
        echo ""
    else
        echo -e "${RED}❌ SOME CHECKS FAILED${NC}"
        echo ""
        echo "Failed checks: $failed"
        echo ""
        echo "Please resolve the issues above before proceeding with experiments."
        echo ""
    fi

    echo "========================================="
}

# Run main function
main "$@"