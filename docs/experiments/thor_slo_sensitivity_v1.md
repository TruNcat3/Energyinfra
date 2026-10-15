# Thor SLO sensitivity v1

This is an offline re-evaluation of the frozen `strict_confirm` and `confirm`
serving runs. It performs no inference and makes no hardware changes.

The analysis crosses TTFT limits of 500, 750, 1000, and 1500 ms with TPOT
limits of 30, 50, and 100 ms. A strategy is feasible only when every one of
the three frozen-trace repetitions achieves at least 95 percent per-request
attainment. Power, energy per token, TTFT, and TPOT are all minimized when
constructing each feasible Pareto set.

Run:

```bash
.venv-thor/bin/python scripts/active/analyze_thor_slo_sensitivity_v1.py
```

Outputs are written under
`data/protocol_v2/thor_slo_capacity_frontier_v1/slo_sensitivity_v1`.
