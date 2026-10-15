# Supplementary headroom screening and independent confirmation

This extension does not modify parent profiling, selections or comparison files.
It repairs only minimum-power cases marked `fail` in comparison_full.json, across
all existing SLO/scenario combinations. `infeasible` cases are retained in the
parent report and not silently relaxed. The original EnergyInfra selector and
ThermalSLO adapters remain separate unfinished work.

Candidates must meet 90% of each applicable latency limit in EVERY stored
profiling request. The exact SLO itself remains unchanged. Rank all eligible CAP
configurations by profile energy-integrated average power, then TPOT and ID; take
at most three. Exclude the original failed configuration. No qualifying candidate
means no repair, not a fallback marked feasible.

Screen all preregistered candidates for ten measurement blocks. Select the FIRST
in preregistered order whose observed attainment is >=95%, not the lowest-power
one observed during screening. Confirm that one on another ten independent blocks.
A confirmation failure is final for this supplement; no retesting until it passes.
Dynamic and max_clock are rerun in both stages. Identical workload/configuration
pairs are shared between SLO cases and not counted as independent new evidence.
Wilson intervals are included; 10/10 does not establish a 95% lower confidence
bound. Do not treat many requests within a short block as independent repeats.

## Run

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_energyinfra_v2_repair_nohup.sh \
  --parent /home/wt/energyinfra/Energyinfra-main/data/protocol_v2/thor_20260915 \
  --run-dir /home/wt/energyinfra/Energyinfra-main/data/protocol_v2/thor_20260920_repair
```

The launcher runs screening and, only on success, confirmation in one background
job. Do not run other hardware experiments at the same time. Each stage uses the
v2 hardware mutex and restores its entry state. Source/model/runtime changes
reject resume. Reissue the same command to resume committed blocks. A CPU/GPU
process killed with SIGKILL cannot restore hardware; run the supplementary Python
entrypoint with `restore` and the same arguments before resuming.

To stop gracefully, find the active Python process whose command includes
`run_energyinfra_v2_repair.py` and this run directory, send SIGTERM to that process.
The supervisor exits on the failed stage, without launching confirmation. The
printed supervisor PID alone is NOT the inference PID.

Outputs: plan.json, screen_decisions.json, stage manifests, raw per-block JSON and
telemetry, comparison.json, RESULTS.md. Confirmed candidates are not automatically
injected into serving or the old selection table. Review independent confirmation
before integrating them into the serving experiment plan.

No tests or hardware experiments were executed while adding this supplement.
