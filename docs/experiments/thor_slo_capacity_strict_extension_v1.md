# Thor strict SLO capacity extension v1

This stage closes the unresolved capacity interval below 24 output-token/s for
the strict 500 ms TTFT / 30 ms TPOT SLO. It reuses the frozen workload mix,
runtime, hardware candidates, fixed-work integrity gates, and conservative
per-repeat feasibility rule from the main capacity-frontier experiment.

## Design

- Offered output rates: 8, 12, 16, 20, and 22 token/s.
- Duration: 120 seconds per trace.
- Repetitions: three independently frozen traces per rate.
- Strategies: the same eight retained baselines and joint Pareto candidates
  used by the capacity confirmation stage.
- A rate is reported feasible only when a strategy satisfies the required SLO
  attainment in every repetition.
- Results are isolated under `strict_confirm`; existing pilot, refine, and
  confirm results are not changed.

## Commands

Prepare without root:

```bash
bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh prepare strict_confirm
```

Run with hardware-control permission:

```bash
sudo -E bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh run strict_confirm
```

The final report is written to
`data/protocol_v2/thor_slo_capacity_frontier_v1/strict_confirm/CAPACITY.md`.
