# Thor thermal stability v1

## Purpose

This protocol tests whether the serial P1 QueuePowerMin result remains useful
after the platform reaches a long-duration thermal state. It compares only
native dynamic, MaxClock and QueuePowerMin on the same frozen single-user trace.
It does not reopen the frequency search and does not mix bursty or P8 evidence.

The `quick` preset is an execution gate of approximately five minutes per
strategy and is not a thermal-stability result. The `full` preset contains six
workload blocks, 60 requests and about 30.4 minutes per trial. Three strategies
and three repeats produce nine full trials, or roughly 4.6 hours plus startup
and cooldown.

## Formal gate

Every full repeat must satisfy all of the following:

- exact fixed work and complete telemetry;
- full-preset workload-block-aligned early/late power windows;
- interactive SLO attainment of at least 95%;
- peak measured platform temperature below 85 C;
- final-window positive temperature slope no greater than 0.25 C/min;
- absolute early-to-late power drift no greater than 5%;
- positive early-to-late median TPOT drift no greater than 10%.

Power and Energy/token are still reported separately against native dynamic and
MaxClock. Quick runs intentionally do not evaluate the power-drift gate because
their early and late windows contain different workload compositions. Passing
the full gate establishes stability only for this model, runtime,
fan/ambient condition and trace. It is not a battery-life or model-quality claim.

## Commands

Run preparation as the normal user:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_thermal_stability_nohup.sh quick prepare
```

Run the quick hardware gate with root:

```bash
sudo -E bash scripts/active/run_thor_thermal_stability_nohup.sh quick run
```

Generate the thermal report as the normal user:

```bash
bash scripts/active/run_thor_thermal_stability_nohup.sh quick analyze
```

Only after inspecting `quick/THERMAL_RESULTS.md`, prepare and run full:

```bash
bash scripts/active/run_thor_thermal_stability_nohup.sh full prepare
sudo -E bash scripts/active/run_thor_thermal_stability_nohup.sh full run
bash scripts/active/run_thor_thermal_stability_nohup.sh full analyze
```

The runner restores the entry hardware state on normal completion, handled
failure and SIGTERM. If restoration remains pending, use the existing Stage 1B
`restore` command with the same config, preset and run directory before resume.

## Outputs

- `RESULTS.md`: standard Stage 1B fixed-trace/SLO report;
- `THERMAL_RESULTS.md`: thermal stability and dual-baseline summary;
- `trial_thermal.csv`: one row per trial;
- `thermal_summary.csv`: one row per strategy;
- `thermal_trials.json`: individual gate outcomes;
- `thermal_analysis_manifest.json`: analysis provenance;
- `telemetry/*.jsonl`: receive-timestamped raw tegrastats and parsed sensors.

## Completed full result

The full campaign completed 9/9 trials, with 180/180 exact requests in every
trial and hardware restoration completed. Peak temperatures remained below the
85 C safety limit:

| Strategy | Power | Energy/token | Peak temperature | Interactive attainment | Thermal gate |
|---|---:|---:|---:|---:|---:|
| Native dynamic P1 | 38.98 W | 4.872 J/token | 58.5 C | 95.0% | 1/3 |
| MaxClock P1 | 42.09 W | 5.261 J/token | 64.0 C | 100.0% | 2/3 |
| QueuePowerMin P1 | 37.87 W | 4.734 J/token | 61.2 C | 98.3% | 2/3 |

QueuePowerMin is approximately 2.84% lower in power and Energy/token than native
dynamic, and approximately 10.02% lower than MaxClock. The long-run direction
is consistent with the shorter serial confirmation, but strict thermal
stability is not yet an all-repeat result: the remaining failures are one
temperature-slope gate and one SLO repeat, not an over-temperature event.
