# Thor measurement protocol v2

## Scope and provenance

This is a separate, corrected measurement and constrained-search pipeline. Old
scripts and data are untouched. It must NOT be reported as a verbatim reproduction
of the original selector, phase-switch controller, or ThermalSLO serving policy.
Those policies remain in the original entrypoints and are not plugged into v2 yet.

`fine` matches the later GPU-only CAP matrix shape: 10 GPU caps, maximum EMC,
CPU 1458 MHz, seven workloads, three blocks: 210 CAP + 42 baseline blocks.
`full` expands to 10 GPU x 4 EMC x 3 CPU x 12 workloads x 3 blocks:
4320 CAP + 72 baseline blocks. A short block may contain multiple fixed-length
requests to collect enough power samples; these are not independent block repeats.
Both cohorts are CAP experiments, not true phase-specific lock tables.

The baseline called `max_clock` locks GPU and CPU to their available maxima and
sets maximum EMC cap. It does not change nvpmodel, and is NOT a claim to have
selected a hardware MAXN power mode. Dynamic restores full frequency ranges and
configured native governors. Record nvpmodel, fan settings and ambient conditions
with the campaign; do not change them between strategies.

## Measurement contract

- Exact input token IDs are repeated/truncated from a fixed tokenized seed, without
  implicit BOS. Context overflow is rejected. The input hash is recorded.
- The low-level generator produces exactly the requested number of model token
  steps. Special/EOG tokens are included and do not terminate this synthetic
  fixed-compute workload. This is not a normal chat completion or quality test.
- `model.reset()` precedes every request, preventing prefix-cache hits. Process
  warmup remains enabled. All output IDs and monotonic token timestamps are saved.
- TTFT includes prefill and first sampling. TPOT is the mean of token intervals,
  not text chunks. No energy claim is made for sub-sample prefill/decode phases.
- Continuous tegrastats logs retain raw lines, including temperature and clocks.
  Timestamps are host line-receipt timestamps, not device sensor timestamps.
  Linear interpolation and trapezoidal integration use bracketing samples.
  Missing rails, large gaps, and too few in-window samples abort, without writing
  a successful task record. Sensor calibration and timestamp jitter remain limits.
- Block J/token includes small Python inter-request overhead, excludes warmup and
  settling. Serving energy includes idle, queueing, clock switching and drain.
- Lowest-power and lowest-energy selection searches ALL valid CAP points for each
  exact workload. The three-dimensional nondominated frontier is also exported.
  Median-based offline feasibility is explicitly only a prediction. Validation
  reports actual request attainment and Wilson intervals; a 95% point estimate is
  not a 95% lower confidence bound. Requests within a block are correlated.
- No fallback candidate is labelled feasible. Incomplete grids block selection.
  Unsupported controls fail fast rather than silently shrinking the search space.

## Commands

Run from `/home/wt/energyinfra/Energyinfra-main`. Choose a new absolute campaign
path and keep the same path through all stages. Do not launch old hardware jobs
concurrently: the v2 hardware lock only coordinates v2 jobs.

```bash
RUN=/home/wt/energyinfra/Energyinfra-main/data/protocol_v2/thor_20260915
PY=.venv-thor/bin/python
SCRIPT=scripts/active/run_energyinfra_v2.py
LAUNCH=scripts/active/run_energyinfra_v2_nohup.sh

# Print planned counts without loading a model or touching hardware.
$PY "$SCRIPT" plan --run-dir "$RUN"

# Run this first, then wait for calibration.json with passed=true.
sudo -E bash "$LAUNCH" calibrate --run-dir "$RUN"

# Later GPU-only CAP shape, using corrected metrics. Wait for each stage.
sudo -E bash "$LAUNCH" profile --cohort fine --run-dir "$RUN"
sudo -E $PY "$SCRIPT" select --cohort fine --run-dir "$RUN"
sudo -E bash "$LAUNCH" validate --cohort fine --run-dir "$RUN"

# Expanded Thor grid. Do not start before calibration has passed.
sudo -E bash "$LAUNCH" profile --cohort full --run-dir "$RUN"
sudo -E $PY "$SCRIPT" select --cohort full --run-dir "$RUN"
sudo -E bash "$LAUNCH" validate --cohort full --run-dir "$RUN"

# One strategy/trace/scenario per invocation; each runs three repeats.
sudo -E bash "$LAUNCH" serve --cohort full --run-dir "$RUN" \
  --strategy dynamic --trace short_chat --slo 50 --scenario tpot_only
# Repeat with max_clock, min_power, min_energy in randomized order.
# Other traces: long_generation, bursty_mixed.
# Other thresholds: 100, 250; joint TTFT scenario: interactive_1s.
```

Do not paste the whole sequence unattended: background launch completion is not
experiment completion. The log, progress.json and stage output directories show
progress. Reissue the identical command to resume committed blocks; an interrupted
block is rerun in full. A serving repeat is committed only after full drain.
Protocol, source, model and environment fingerprints reject mixed resumes.

SIGINT/SIGTERM restore the entry state. SIGKILL/power loss cannot run cleanup;
`restore_pending.json` then blocks new hardware jobs. With no other hardware job
running, explicitly restore using:

```bash
sudo -E .venv-thor/bin/python scripts/active/run_energyinfra_v2.py restore \
  --run-dir /home/wt/energyinfra/Energyinfra-main/data/protocol_v2/thor_20260915
```

## Outputs and interpretation

`selection_full.json` contains the complete rate table, frontier, candidates and
infeasible cases. `comparison_full.csv` and JSON contain validation results with
separate deltas against Dynamic and max_clock. Negative deltas mean lower values;
energy/power improvements are only deployment wins when the SLO also holds.
Total energy comparisons require the same completed work, not just equal repeats.

Serving imports the existing RequestGenerator and freezes each generated trace.
It replays fixed arrivals through a single worker and bounded FIFO. Drops count
as SLO failures. Per-workload optimized strategies require a validated feasible
configuration for EVERY workload present, otherwise the run refuses to start.
This is a corrected open-loop replay extension, not the legacy serving scheduler.

## Remaining alignment work and acceptance

The original selectors, BestStatic, ThermalSLO and true prefill/decode switching
still need a versioned adapter and independent comparison before claiming the
ENTIRE original EnergyInfra protocol is reproduced with corrected measurements.
The new pipeline intentionally does not substitute a newly invented knee for the
original algorithm. Original scripts remain usable only in a separately labelled
legacy-metric cohort; do not merge their tables into v2.

Before expensive runs, explicitly validate on hardware: exact length and repeat
consistency, no prefix reuse, VIN sampling continuity, under-load caps, graceful
restore, forced interruption/resume, all-infeasible SLOs, and deterministic traces.
No tests or hardware campaigns were run as part of adding these files.
