# Thor SLO capacity frontier v1

## Objective

This experiment locates the maximum feasible offered output-token rate separately for the 500/30,
1000/50, and 2000/100 ms joint TTFT/TPOT SLOs. It preserves EnergyInfra's offline-profile,
SLO-feasibility, and Pareto-selection flow while adding persistent-server resource dimensions.

## Candidate provenance

Hardware candidates are distilled from the measured Q4 full static table for the same weighted workload
mix. The catalog assigns power-minimum, energy-minimum, normalized-knee, latency-minimum, and validated
queue-reference roles, deduplicates them, and retains at most three unique hardware states. Selected
states are crossed with P2/P4/P8 resource plans. Native dynamic, MaxClock, EnergyInfra DVFS-only, and
continuous-batching baselines remain in every screen.

The immutable catalog records source CSV row indices, source hashes, aggregate metrics, hardware JSON,
and role assignments. No frequency point is invented by this stage.

## Search stages

`pilot` screens 8, 12, 16, 20, 24, 28, 32, 44, 56, and 70 output-token/s for 60 seconds.

`refine` reads the pilot's measured feasibility and inserts a quantized midpoint into each unresolved
SLO capacity bracket.

`confirm` retains the measured boundary pair for each SLO, prunes weak screen candidates while preserving
all baselines, and runs three 120-second frozen repeats.

Every request must pass Stage 1B fixed-work and SSE-coverage gates. A strategy is feasible only when every
repeat reaches at least 95% joint SLO attainment. Power, total energy, J/token, TTFT, and TPOT remain
separate Pareto objectives.

## Pilot

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh prepare pilot
sudo -E bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh run pilot
```

After pilot completion:

```bash
bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh prepare refine
sudo -E bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh run refine
```

After refinement:

```bash
bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh prepare confirm
sudo -E bash scripts/active/run_thor_slo_capacity_frontier_nohup.sh run confirm
```

Each run writes a Stage 1B report plus `CAPACITY.md` and `capacity_frontier.json` under its phase directory.
