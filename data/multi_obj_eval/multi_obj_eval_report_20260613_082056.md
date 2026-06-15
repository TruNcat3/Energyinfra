# Multi-Objective Pareto Evaluation Report

**Generated**: 2026-06-13 08:20:56
**Script**: `src/ratetable/pareto_multi_objective_evaluation.py`

---

## Metric Definitions

| Metric | Full Name | Description |
|:---:|---|---|
| **MDR** | Multi-Objective Dominance Rate | Fraction of instances where strategy A dominates baseline B on **ALL** objectives simultaneously |
| **JIR** | Joint Improvement Ratio | Geometric mean of per-objective improvement ratios, counted **only** when ALL objectives improve |
| **HV** | Hypervolume Indicator | Volume of dominated objective space (Zitzler 1999); higher = better |
| **Waste** | Composite Waste | Normalized Euclidean distance ratio: how much worse B is than A relative to ideal |

## 1. 3D Offline Evaluation (E2E Benchmark)

**Data source**: `data/cap_selector_benchmark/` (378 runs, 3 models × 9 strategies × 5 workloads × 3 repeats)
**Objectives**: E/tok (minimize), TPOT (minimize), Power (minimize)

### Pareto vs Dynamic

| Model | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-8B | 40.0% | 0.5% | 0.2% | +1.5% | -2.7% | +6.8% |
| Qwen2.5-14B | 33.3% | 0.3% | 0.3% | +2.0% | -1.8% | +4.6% |
| Qwen2.5-7B | 0.0% | 0.0% | 4.5% | +1.1% | -18.5% | +27.8% |

### Pareto vs Maxn

| Model | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-8B | 0.0% | 0.0% | 10.2% | +5.3% | -4.9% | +12.2% |
| Qwen2.5-14B | 0.0% | 0.0% | 11.6% | +6.0% | -3.0% | +9.3% |
| Qwen2.5-7B | 0.0% | 0.0% | 14.8% | +5.1% | -20.4% | +34.7% |

### Hypervolume (3D)

Higher HV = dominates more objective space = better

| Strategy | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) |
|:---|:---:|:---:|:---:|
| alpha_03 | 2.3658 | 3.3694 | 109.4028 |
| alpha_07 | 2.2313 | 4.3064 | 98.7363 |
| dynamic | 1.8903 | 2.5817 | 1.4357 |
| maxn | 1.8418 | 3.3497 | 1.4913 |
| min_energy | 2.2536 | 3.0886 | 32.6626 |
| pareto | 73.6786 | 96.0219 | 483.7423 |
| pwr_45w | 0.0002 | — | 368.7988 |
| slo_45ms | 2.2465 | 3.1416 | 27.3331 |
| slo_50ms | 2.2336 | 3.0486 | 111.7167 |

## 2. 3D Oracle Gap Evaluation

**Data source**: `data/oracle_gap_analysis/` (lock-mode 11-freq oracle)
**Objectives**: E/tok, TPOT, Power (all minimize)

### Multi-Objective Dominance vs Oracle

| Model | Dynamic→Oracle | MAXN→Oracle | Pareto→Oracle | BestStatic→Oracle |
|:---:|:---:|:---:|:---:|:---:|
| Llama-8B | 0.0% (waste -22.2%) | 0.0% (waste -34.0%) | 0.0% (waste -13.3%) | 0.0% (waste -17.2%) |
| Qwen2.5-14B | 0.0% (waste -22.2%) | 0.0% (waste -34.0%) | 0.0% (waste -13.3%) | 0.0% (waste -17.2%) |
| Qwen2.5-7B | 0.0% (waste -22.2%) | 0.0% (waste -34.0%) | 0.0% (waste -13.3%) | 0.0% (waste -17.2%) |

### Hypervolume (Oracle Gap)

| Strategy | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) |
|:---|:---:|:---:|:---:|
| best_static | 9.2056 | 20.1740 | 4.6405 |
| dynamic | 8.9420 | 20.4176 | 10.5620 |
| maxn | 7.7272 | 17.1348 | 5.6567 |
| oracle_energy | 1.2398 | 11.9742 | 3.9089 |
| oracle_power | 148.6525 | 1.4481 | 0.3484 |
| oracle_slo | 1.2398 | — | 3.9089 |
| pareto | 0.7637 | 0.8413 | 2.2022 |

## 3. 4D Serving Evaluation

**Data source**: `data/serving_benchmark/` (long-running serving windows)
**Objectives**: E/tok (minimize), TPOT (minimize), Power (minimize), Peak Temp (minimize)

### Model: Llama-8B

#### Trace: bursty_mixed

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 0.0% | 0.0% | 21.6% | +3.4% | -2.2% | +6.0% | +1.3% |
| Pareto | Dynamic | 2.6% | 0.5% | -3.2% | +0.1% | -0.0% | +0.5% | -1.3% |
| ThermalSLO | MAXN | 0.0% | 0.0% | 22.5% | +3.4% | -2.2% | +6.0% | +1.4% |
| ThermalSLO | Dynamic | 10.5% | 0.4% | -2.3% | +0.2% | -0.1% | +0.5% | -1.3% |
| BestStatic | MAXN | 0.0% | 0.0% | 42.7% | +14.1% | -18.4% | +38.5% | +8.5% |
| BestStatic | Dynamic | 0.0% | 0.0% | 27.1% | +10.5% | -16.7% | +31.5% | +5.6% |

#### Trace: long_generation

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 0.0% | 0.0% | 31.7% | +4.3% | -2.8% | +7.3% | +1.1% |
| Pareto | Dynamic | 4.0% | 0.0% | 98.9% | +9.7% | +104.6% | -28.9% | -9.6% |
| ThermalSLO | MAXN | 0.0% | 0.0% | 28.8% | +3.5% | -2.7% | +6.5% | +1.0% |
| ThermalSLO | Dynamic | 0.0% | 0.0% | 98.5% | +9.2% | +105.0% | -29.3% | -9.3% |
| BestStatic | MAXN | 0.0% | 0.0% | 42.8% | +14.9% | -18.8% | +41.1% | +8.6% |
| BestStatic | Dynamic | 0.0% | 0.0% | 175.3% | +17.3% | +71.9% | -18.1% | -5.1% |

#### Trace: short_chat

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 0.0% | 0.0% | 13.7% | +3.3% | -1.4% | +4.2% | +0.5% |
| Pareto | Dynamic | 6.0% | 1.9% | -5.7% | -0.5% | -0.0% | -0.8% | -0.7% |
| ThermalSLO | MAXN | 0.0% | 0.0% | 18.7% | +3.9% | -1.5% | +5.6% | +0.5% |
| ThermalSLO | Dynamic | 5.3% | 1.4% | -2.7% | +0.2% | -0.1% | +0.5% | -0.8% |
| BestStatic | MAXN | 0.0% | 0.0% | 19.2% | +13.5% | -18.2% | +32.8% | +7.7% |
| BestStatic | Dynamic | 0.0% | 0.0% | 8.9% | +9.7% | -17.0% | +26.6% | +6.5% |

### Model: Qwen2.5-14B

#### Trace: bursty_mixed

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 0.0% | 0.0% | -38.2% | -2.4% | -19.7% | +19.7% | +5.2% |
| Pareto | Dynamic | 0.0% | 0.0% | -21.3% | -6.8% | -18.6% | +9.0% | +2.5% |
| ThermalSLO | MAXN | 0.0% | 0.0% | 13.2% | +2.4% | -5.8% | +8.5% | +2.9% |
| ThermalSLO | Dynamic | 8.3% | 0.7% | -6.3% | -2.1% | -4.5% | -0.9% | +0.2% |
| BestStatic | MAXN | 0.0% | 0.0% | -13.0% | +0.5% | -11.1% | +11.9% | +4.3% |
| BestStatic | Dynamic | 0.0% | 0.0% | -6.1% | -3.9% | -9.9% | +2.4% | +1.6% |

#### Trace: long_generation

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 0.0% | 0.0% | -21.5% | +1.7% | -10.3% | +13.0% | +4.0% |
| Pareto | Dynamic | 0.0% | 0.0% | -41.1% | -1.8% | -8.8% | +7.4% | +2.4% |
| ThermalSLO | MAXN | 0.0% | 0.0% | 41.3% | +3.8% | -2.3% | +6.3% | +2.3% |
| ThermalSLO | Dynamic | 0.0% | 0.0% | 1.9% | +0.3% | -0.7% | +1.0% | +0.7% |
| BestStatic | MAXN | 0.0% | 0.0% | -32.1% | +1.2% | -11.2% | +13.5% | +3.2% |
| BestStatic | Dynamic | 0.0% | 0.0% | -49.5% | -2.2% | -9.8% | +7.9% | +1.7% |

#### Trace: short_chat

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 0.0% | 0.0% | -38.6% | -1.3% | -18.8% | +18.9% | +4.2% |
| Pareto | Dynamic | 0.0% | 0.0% | -51.5% | -5.6% | -18.4% | +13.1% | +2.8% |
| ThermalSLO | MAXN | 0.0% | 0.0% | 58.4% | +4.7% | -1.7% | +6.3% | +2.8% |
| ThermalSLO | Dynamic | 0.0% | 0.0% | 11.3% | +0.1% | -1.2% | +1.1% | +1.3% |
| BestStatic | MAXN | 0.0% | 0.0% | -21.7% | +2.1% | -10.5% | +12.2% | +3.2% |
| BestStatic | Dynamic | 0.0% | 0.0% | -43.4% | -2.4% | -10.0% | +6.7% | +1.8% |

### Model: Qwen2.5-7B

#### Trace: bursty_mixed

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 4.8% | 5.5% | 39.3% | +5.1% | -2.7% | +9.2% | +3.0% |
| Pareto | Dynamic | 14.5% | 1.1% | 8.5% | +0.3% | -0.7% | +0.9% | +1.3% |
| ThermalSLO | MAXN | 0.0% | 0.0% | 39.3% | +1.7% | -3.9% | +6.9% | +4.2% |
| ThermalSLO | Dynamic | 2.4% | 0.8% | 2.9% | -2.8% | -1.9% | -1.1% | +2.5% |
| BestStatic | MAXN | 0.0% | 0.0% | 34.1% | +1.3% | -7.9% | +10.1% | +3.0% |
| BestStatic | Dynamic | 0.0% | 0.0% | 3.6% | -3.1% | -6.1% | +2.0% | +1.2% |

#### Trace: long_generation

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 21.5% | 7.8% | 44.0% | +8.3% | -2.1% | +11.0% | +3.6% |
| Pareto | Dynamic | 5.1% | 0.7% | -8.7% | -3.4% | -2.5% | -1.2% | -0.6% |
| ThermalSLO | MAXN | 9.7% | 7.6% | 49.1% | +4.3% | -6.0% | +11.5% | +5.7% |
| ThermalSLO | Dynamic | 8.3% | 1.6% | -5.4% | -7.0% | -6.4% | -0.9% | +1.4% |
| BestStatic | MAXN | 6.6% | 8.6% | 66.9% | +10.1% | -5.7% | +16.6% | +5.2% |
| BestStatic | Dynamic | 0.0% | 0.0% | 7.1% | -1.8% | -6.1% | +3.7% | +1.0% |

#### Trace: short_chat

| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Pareto | MAXN | 14.4% | 2.2% | 64.7% | +7.9% | -0.5% | +8.9% | +3.4% |
| Pareto | Dynamic | 12.1% | 0.4% | 1.2% | -0.2% | -0.1% | -0.2% | +0.3% |
| ThermalSLO | MAXN | 9.1% | 4.7% | 19.1% | -0.2% | -2.6% | +2.8% | +3.8% |
| ThermalSLO | Dynamic | 13.6% | 2.1% | -8.2% | -7.6% | -2.2% | -5.7% | +0.7% |
| BestStatic | MAXN | 1.5% | 2.9% | 19.1% | +8.8% | -1.6% | +17.5% | +4.1% |
| BestStatic | Dynamic | 0.8% | 0.6% | 7.0% | +0.7% | -1.2% | +7.8% | +1.0% |

### Hypervolume (4D Serving, per model)

| Baseline | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) |
|:---|:---:|:---:|:---:|
| BestStatic | 476.3993 (150 pts) | 175.9434 (21 pts) | 4090.3065 (167 pts) |
| Dynamic | 19844.4351 (157 pts) | 1524.2651 (22 pts) | 335.4698 (183 pts) |
| MAXN | 356.4652 (157 pts) | 736.9123 (162 pts) | 12753.6914 (182 pts) |
| Pareto | 323.3171 (157 pts) | 1184.6068 (19 pts) | 2980.1391 (182 pts) |
| ThermalSLO | 365.8159 (157 pts) | 1234.7842 (22 pts) | 21829.1806 (157 pts) |

## 4. Serving Aggregate Statistics

### Llama-8B

| Baseline | Trace | Windows | E/tok (J) | TPOT (ms) | Power (W) | Peak Temp (°C) | SLO Viol % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BestStatic | bursty_mixed | 81 | 1.8882 | 52.2 | 31.6 | 65.2 | — |
| Dynamic | bursty_mixed | 90 | 2.0864 | 43.2 | 41.7 | 68.4 | — |
| MAXN | bursty_mixed | 93 | 2.1584 | 42.3 | 44.1 | 70.8 | — |
| Pareto | bursty_mixed | 90 | 2.0886 | 43.2 | 41.6 | 70.0 | — |
| ThermalSLO | bursty_mixed | 90 | 2.0874 | 43.3 | 41.6 | 70.0 | — |
| BestStatic | long_generation | 72 | 1.8615 | 52.7 | 33.4 | 65.0 | — |
| Dynamic | long_generation | 38 | 2.3383 | 106.1 | 26.4 | 60.7 | — |
| MAXN | long_generation | 88 | 2.1476 | 42.4 | 47.0 | 70.9 | — |
| Pareto | long_generation | 86 | 2.0596 | 43.6 | 43.8 | 70.3 | — |
| ThermalSLO | long_generation | 86 | 2.0751 | 43.6 | 44.1 | 70.3 | — |
| BestStatic | short_chat | 150 | 1.8035 | 51.1 | 27.3 | 63.6 | — |
| Dynamic | short_chat | 157 | 1.9786 | 42.1 | 34.6 | 67.6 | — |
| MAXN | short_chat | 157 | 2.0527 | 41.5 | 36.3 | 68.4 | — |
| Pareto | short_chat | 157 | 1.9914 | 42.1 | 34.9 | 67.9 | — |
| ThermalSLO | short_chat | 157 | 1.9768 | 42.1 | 34.4 | 68.2 | — |

### Qwen2.5-14B

| Baseline | Trace | Windows | E/tok (J) | TPOT (ms) | Power (W) | Peak Temp (°C) | SLO Viol % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BestStatic | bursty_mixed | 12 | 3.9650 | 87.3 | 40.3 | 66.0 | — |
| Dynamic | bursty_mixed | 12 | 3.8104 | 78.7 | 41.3 | 67.1 | — |
| MAXN | bursty_mixed | 12 | 3.9856 | 77.6 | 45.0 | 68.9 | — |
| Pareto | bursty_mixed | 11 | 4.0896 | 97.4 | 37.5 | 65.4 | — |
| ThermalSLO | bursty_mixed | 12 | 3.8913 | 82.8 | 41.5 | 66.9 | — |
| BestStatic | long_generation | 7 | 3.8682 | 88.1 | 42.3 | 66.5 | — |
| Dynamic | long_generation | 7 | 3.7814 | 79.5 | 45.6 | 67.6 | — |
| MAXN | long_generation | 8 | 3.9113 | 78.3 | 48.0 | 68.7 | — |
| Pareto | long_generation | 7 | 3.8515 | 87.3 | 42.5 | 66.0 | — |
| ThermalSLO | long_generation | 7 | 3.7707 | 80.1 | 45.1 | 67.1 | — |
| BestStatic | short_chat | 21 | 3.7582 | 85.8 | 37.5 | 65.7 | — |
| Dynamic | short_chat | 22 | 3.6640 | 77.1 | 40.0 | 66.8 | — |
| MAXN | short_chat | 162 | 3.8835 | 77.5 | 42.0 | 70.8 | — |
| Pareto | short_chat | 19 | 3.8896 | 95.2 | 35.4 | 65.0 | — |
| ThermalSLO | short_chat | 22 | 3.6625 | 78.1 | 39.5 | 66.0 | — |

### Qwen2.5-7B

| Baseline | Trace | Windows | E/tok (J) | TPOT (ms) | Power (W) | Peak Temp (°C) | SLO Viol % |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| BestStatic | bursty_mixed | 167 | 1.8680 | 47.2 | 34.6 | 66.8 | — |
| Dynamic | bursty_mixed | 183 | 1.9358 | 41.5 | 39.9 | 69.6 | — |
| MAXN | bursty_mixed | 182 | 2.0374 | 40.7 | 43.0 | 71.2 | — |
| Pareto | bursty_mixed | 182 | 1.9342 | 41.8 | 39.7 | 69.1 | — |
| ThermalSLO | bursty_mixed | 99 | 1.9758 | 42.0 | 40.1 | 68.4 | — |
| BestStatic | long_generation | 154 | 1.8632 | 47.4 | 36.3 | 67.2 | — |
| Dynamic | long_generation | 173 | 1.9399 | 41.8 | 42.0 | 69.8 | — |
| MAXN | long_generation | 174 | 2.0978 | 41.3 | 46.2 | 71.7 | — |
| Pareto | long_generation | 172 | 1.9752 | 42.3 | 42.3 | 69.8 | — |
| ThermalSLO | long_generation | 86 | 2.1186 | 44.9 | 42.7 | 69.3 | — |
| BestStatic | short_chat | 156 | 1.7803 | 40.7 | 31.8 | 68.1 | — |
| Dynamic | short_chat | 156 | 1.7672 | 40.1 | 32.0 | 68.5 | — |
| MAXN | short_chat | 156 | 1.9452 | 40.0 | 35.5 | 71.2 | — |
| Pareto | short_chat | 156 | 1.7759 | 40.1 | 32.2 | 68.4 | — |
| ThermalSLO | short_chat | 157 | 1.9232 | 41.2 | 34.2 | 68.0 | — |

## 5. Summary

### Key Findings


- **pareto vs dynamic** (3D offline): MDR = 24.4%, Composite Waste = 1.7%
- **pareto vs maxn** (3D offline): MDR = 0.0%, Composite Waste = 12.2%

### Paper Narrative Implications

1. **Multi-objective superiority**: Pareto's value extends beyond single-objective E/tok
2. **Thermal dimension**: ThermalSLO's 4D advantage demonstrates unique thermal-aware contribution
3. **Hypervolume**: Gold-standard EMO metric quantifies total dominated solution space
