# Thor Q4 full Pareto-frontier protocol

This campaign uses the same-source Q4_K_M 7B synthetic model selected by the
precision pilot. It treats a Pareto frontier as a set of non-dominated
tradeoffs, never as a unique optimum.

## Declared design space

- 4 GPU caps: 738, 1116, 1485 and 1575 MHz.
- 3 EMC caps: 2750, 3200 and 4266 MHz.
- 3 CPU floors: minimum supported, 486 and 972 MHz.
- CPU ceiling fixed at 1458 MHz.
- Two locked references plus Dynamic and MaxClock: 40 configurations.
- The 12 original protocol workloads, 3 profile repeats: 1440 static blocks.
- TPOT constraints 50/75/100 ms and TTFT scenarios unconstrained/1 s/2 s.

The four lower-is-better Pareto dimensions are average input power, J/token,
TTFT and TPOT. All profile-frontier members across workload/SLO combinations
are deduplicated and independently validated for 20 repeats, together with
Dynamic and MaxClock. The exact number of validation blocks is known only
after the static frontiers are frozen.

## Preferences

Four normalized weighted profiles are declared before measurement:
energy-weighted, power-weighted, latency-weighted and balanced. Normalization
uses the min/max values within each profile-derived frontier. These profiles
are examples of user sensitivity, not universal rankings. Minimum-energy and
minimum-power are retained only as frontier anchors for continuity.

`selection.json` freezes profile-derived frontiers and preference choices.
`frontier_validation.csv` reports every validated frontier member without
reselection. `comparison.csv` reports anchors and declared preference choices.

## Run

Inspect the plan without touching hardware:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_q4_frontier_nohup.sh plan
```

Start profile, selection, frontier-union validation and reporting:

```bash
sudo -E bash scripts/active/run_thor_q4_frontier_nohup.sh run
```

The launcher prints a log path, supervisor PID and graceful TERM command.
Serving is not started automatically. It remains blocked unless every trace
workload has an independently validated feasible point for the requested
preference and SLO. Random weights do not evaluate quantization quality.
