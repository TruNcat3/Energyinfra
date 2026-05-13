# Selector Performance Evaluation Report

**Generated**: 2026-05-13 02:44:56
**Data Source**: Synthetic Benchmark (Phase 3)

## Strategy Comparison Summary

| Strategy | SLO Violation Rate | Mean Energy Regret | Median Energy Regret | P95 Energy Regret | Mean Energy (J/token) | Mean TTFT (ms) | SLO Feasible Buckets |
|----------|-------------------|-------------------|---------------------|-------------------|----------------------|----------------|----------------------|
| Maxn All High | 0.00% | 8.21% | 0.00% | 34.15% | 0.1465 | 9.08 | 7/7 |
| All Mid | 0.00% | 22.11% | 0.00% | 71.44% | 0.1471 | 9.41 | 7/7 |
| Energy Efficient All Low | 0.00% | 0.00% | 0.00% | 0.00% | 0.1371 | 9.48 | 7/7 |
| Fixed Best Efficiency | 0.00% | 1.61% | 0.00% | 6.69% | 0.1396 | 9.87 | 7/7 |
| Oracle Best Per Bucket | 0.00% | 0.00% | 0.00% | 0.00% | 0.1371 | 9.48 | 7/7 |
| Ours Slo Aware Selector | 0.00% | 0.00% | 0.00% | 0.00% | 0.1371 | 9.48 | 7/7 |

## Key Findings

**Best Performing Strategy**: Energy Efficient All Low
- Mean Energy Regret: 0.00%
- SLO Violation Rate: 0.00%

## Performance vs Baselines

**MaxN vs Oracle**:
- MaxN Regret: 8.21%
- Oracle Regret: 0.00% (baseline)
- Performance Gap: 8.21%

**Our Selector vs MaxN**:
- Our Selector Regret: 0.00%
- MaxN Regret: 8.21%
- Improvement: 8.21%

## ✅ No Anomalies

No negative regret detected — oracle regret is correctly 0% for all buckets.

## ⚠️ Data Limitations

**Important**: This evaluation uses synthetic benchmark data.

**Limitations:
- Results may not reflect real Jetson Orin + llama.cpp performance
- CPU/EMC frequency effects may be misrepresented
- Power and thermal modeling is simplified
- Switching overhead not included in evaluation

**What This Evaluation Validates:
- ✅ Selector logic and SLO filtering correctness
- ✅ Regret calculation methodology
- ✅ Strategy comparison framework
- ✅ Analysis pipeline and reporting

## Recommendations

1. **Real Hardware Validation**: Re-evaluate with actual Jetson Orin measurements
2. **Expand Workload Coverage**: Test with diverse workload patterns
3. **Include Switching Overhead**: Account for frequency transition costs
4. **Thermal Effects**: Model temperature impact on frequency stability
5. **Production Testing**: Validate under real serving conditions