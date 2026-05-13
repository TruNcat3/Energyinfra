# Energy Rate Table Build Summary

**Generated**: 2026-05-13T02:34:10.843513
**Data Source**: data/experiments_4_1_to_4_7

## Data Overview

**Raw Data**:
- Total Rows: 45
- Total Columns: 43

**Final Selector Table**:
- Total Rows: 23
- Unique Buckets: 7
- Total Configurations: 23
- SLO-Feasible Configs: 23
- Pareto Configs: 20
- Avg Measurement CV: 4.30%

## Data Completeness
- Power Statistics: ✅
- Temperature Statistics: ✅
- Phase Information: ✅
- Frequency Information: ✅

## ⚠️ Data Limitations
- Data from synthetic benchmark, not real Jetson Orin measurements
- Power statistics may be limited (synthetic modeling)
- CPU/EMC frequency effects may be underestimated
- Switching overhead not included in current data
- Thermal effects simplified in synthetic data

## Recommendations
- Validate with real Jetson Orin + llama.cpp experiments
- Expand workload matrix for better coverage
- Add comprehensive power and thermal measurements
- Measure actual frequency switching overhead
- Implement real-time SLO monitoring

## Output Files
- `profile_raw.parquet`: Raw experimental data with normalized fields
- `profile_agg_by_config.parquet`: Aggregated data by bucket + config
- `pareto_by_bucket.parquet`: Pareto frontier analysis
- `selector_table.parquet`: Final selector table for config selection
- `rate_table_quality_report.json`: Data quality assessment
- `rate_table_summary.md`: This summary report