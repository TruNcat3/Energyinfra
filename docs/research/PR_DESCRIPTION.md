# PR: Reorganize EnergyInfra as an edge-GPU multi-objective energy framework

## Summary

- Reframe the project around multi-objective energy tuning for edge GPU LLM inference.
- Make Thor the current experimental platform and label Orin/early Phase material as historical baseline.
- Add a unified research report, current status report, architecture guide, operations guide, and protocol documentation.
- Document the separation between serial single-user, bursty low-average-load, and P8 shared-capacity workloads.
- Document reproducibility requirements for config, runtime manifest, trace hash, SLO, telemetry, and hardware restoration.
- Add repository organization guides for configs, scripts, experiments, and reports without moving existing active entry points.

## Research position

The project evaluates the Pareto frontier over platform power, energy per output token, TTFT, TPOT, and temperature under explicit SLO rules. It does not claim a universal optimum. Results are workload-specific and must distinguish power reduction from energy-per-token improvement.

## Current evidence

- Thor hardware control, llama.cpp runtime, telemetry, fixed traces, serving, recovery, and SLO analysis are operational.
- P8 capacity experiments show a useful c1/c3 feasible region at TPOT 100 ms up to approximately 64 tok/s.
- The low-concurrency bursty full campaign completed 90 trials; its strict all-repeat/all-prompt-bucket gate has no final feasible point yet.
- The serial 30-second think-time campaign has completed runtime execution and is the next scenario for consolidated Pareto reporting.

## Validation note

This change is documentation and organization focused. Existing experiment output, model files, telemetry logs, build trees, and virtual environments should remain outside the PR.
