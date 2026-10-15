# Thor multi-load operating modes v1

This analysis answers whether the Thor operating-mode tradeoffs persist beyond the three
`output_tokens=512` phase-switch workloads. It reuses the completed full static sweep and prospective
validation instead of spending additional hardware time.

## Coverage

- 12 prompt/output workload shapes.
- Five operating modes: Energy/Battery, Latency, GPU power-cap, EMC power-cap, and Extreme power-cap.
- Four request-level SLO profiles from strict `500/30 ms` through relaxed `2000/100 ms`.
- Separate energy-latency and power-latency Pareto classifications.
- Validated 20-block data is preferred; the original 3-block static table is used only when a selected
  point was not part of the prospective frontier validation.

This is a fixed-request analysis without queueing. Existing frozen-trace serving results remain the
authority for online goodput and queue TTFT.

## Generate report

```bash
cd /home/wt/energyinfra/Energyinfra-main
.venv-thor/bin/python scripts/active/analyze_thor_multiload_modes_v1.py
```

No sudo, clock changes, model loading, or GPU execution is required.

Outputs:

- `reports/thor_multiload_modes_v1/summary.json`
- `reports/thor_multiload_modes_v1/RESULTS.md`
