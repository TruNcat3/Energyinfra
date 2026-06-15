# Oracle Gap Analysis Report

**Generated**: 2026-06-10 08:13:08
**TPOT SLO**: 50.0 ms | **Power Budget**: 45.0 W
**Models**: 3 | **Workloads**: 36

## 8B

| Strategy | Avg E/tok (J) | Avg TPOT (ms) | Avg Power (W) | Gap vs Oracle-Energy (%) |
|:---|:---:|:---:|:---:|:---:|
| Dynamic | 2.1552 | 42.8 | 47.3 | +8.6% |
| MAXN | 2.1971 | 41.8 | 48.8 | +10.7% |
| Best Static | 2.1386 | 42.9 | 46.9 | +7.7% |
| Pareto (knee) | 2.0003 | 41.9 | 46.9 | +0.7% |
| Oracle-Energy | 1.9855 | 41.9 | 46.5 | — |
| Oracle-SLO | 1.9855 | 41.9 | 46.5 | — |
| Oracle-Power | 2.3033 | 128.4 | 19.5 | — |

**Oracle-SLO feasibility**: 12/12 workloads feasible
**Oracle-Power feasibility**: 12/12 workloads feasible
## 14B

| Strategy | Avg E/tok (J) | Avg TPOT (ms) | Avg Power (W) | Gap vs Oracle-Energy (%) |
|:---|:---:|:---:|:---:|:---:|
| Dynamic | 4.1357 | 79.3 | 49.9 | +23.4% |
| MAXN | 4.2240 | 78.0 | 51.4 | +26.1% |
| Best Static | 3.7988 | 79.1 | 48.4 | +13.5% |
| Pareto (knee) | 3.6637 | 77.4 | 48.1 | +9.4% |
| Oracle-Energy | 3.3765 | 84.1 | 41.9 | — |
| Oracle-Power | 3.4590 | 84.7 | 41.2 | — |

**Oracle-SLO feasibility**: 0/12 workloads feasible
  Infeasible: p1024_o1024, p1024_o128, p1024_o512, p128_o128, p128_o512, p2048_o128, p2048_o512, p256_o256, p512_o1024, p512_o128, p512_o512, p64_o64
  Their fallback Oracle-Tpot: avg TPOT = 77.2ms
**Oracle-Power feasibility**: 12/12 workloads feasible
## 7B

| Strategy | Avg E/tok (J) | Avg TPOT (ms) | Avg Power (W) | Gap vs Oracle-Energy (%) |
|:---|:---:|:---:|:---:|:---:|
| Dynamic | 2.2179 | 43.1 | 47.3 | +12.3% |
| MAXN | 2.2851 | 41.0 | 50.5 | +15.6% |
| Best Static | 2.1993 | 43.3 | 46.9 | +11.3% |
| Pareto (knee) | 2.0616 | 45.3 | 45.2 | +4.4% |
| Oracle-Energy | 1.9802 | 42.6 | 47.2 | — |
| Oracle-SLO | 1.9802 | 42.6 | 47.2 | — |
| Oracle-Power | 2.0991 | 46.3 | 44.2 | — |

**Oracle-SLO feasibility**: 12/12 workloads feasible
**Oracle-Power feasibility**: 12/12 workloads feasible

## Cross-Model Oracle Gap Summary

| Model | Dynamic vs Oracle | MAXN vs Oracle | Best Static vs Oracle | Pareto vs Oracle |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 8B | +8.6% | +10.7% | +7.7% | +0.7% |
| 14B | +23.4% | +26.1% | +13.5% | +9.4% |
| 7B | +12.3% | +15.6% | +11.3% | +4.4% |

## Key Findings

**Average Pareto gap to Oracle-Energy**: +4.9%
**Average Dynamic gap to Oracle-Energy**: +14.8%
**Average MAXN gap to Oracle-Energy**: +17.5%

**Conclusion**: Single-request DVFS optimization space is limited (Pareto → Oracle gap < 5%). Paper narrative should focus on long-running serving metrics (power, temperature, SLO violation) rather than single-request E/token improvement.


## Lock vs Cap Frontier Richness

- 8B: Oracle uses 11 lock freqs (richer search space), current cap has 4 caps
- 14B: Oracle uses 11 lock freqs (richer search space), current cap has 4 caps
- 7B: Oracle uses 11 lock freqs (richer search space), current cap has 4 caps