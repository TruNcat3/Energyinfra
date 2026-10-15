# Prospective validation and paired CPU-governor diagnosis

This stage validates frozen candidates selected from `thor_phase_diag_full_v1`.
It never imports selection measurements into the new statistics, never changes
candidates based on validation results, and never overwrites the diagnostic
campaign. It is a hardware-control extension, not an unchanged original
EnergyInfra CAP/thermal controller and not yet phase-switch serving.

## Candidate design and independent work

Settings are loaded from `configs/experiments/thor_phase_validation_v1.json`.
The referenced diagnostic configuration supplies the model, platform/version,
supported points, and corrected tegrastats telemetry adapter. Existing shared
code/configurations remain unchanged. Another platform needs its own supported
profile and candidates; no nearest-frequency substitution is performed.

There are five roles per workload: Dynamic, MaxClock, frozen minimum-energy
candidate, a lower-power compromise, and a locked-GPU/dynamic-CPU control.
The energy candidate uses GPU/EMC/CPU 1575/4266/1458 for 64/64 and
1485/4266/1458 for 512/512 and 2048/128. The power compromise uses
1485/2750/1458. GPU and CPU are locked for the candidates; EMC is cap-only.
Dynamic and MaxClock preserve their existing v2 definitions and nvpmodel is
not changed. The active nvpmodel mode is recorded when its query is available.

Full validation uses 20 repeats x 3 workloads x 5 roles = 300 blocks. Each
block has two warmup requests, excluded from measurement, followed by an
identical fixed number of measured requests for every role of that workload:
six for 64/64, two for 512/512, three for 2048/128. These provide 120, 40 and
60 measured requests per role respectively, across 20 independent repeat
blocks. Fresh prefix state and exact token-step generation are unchanged.
The workload order is randomized per round; role order uses randomized Latin
rotations balanced over five rounds. This mitigates ordering drift but cannot
eliminate environmental interference or thermal history.

The CPU control is causal only when the compared GPU/EMC points are identical.
In particular, do not attribute the 64/64 candidate/control difference solely
to CPU: that candidate uses GPU 1575 and the control uses GPU 1485. The dedicated
Nsight pair holds GPU 1485 and EMC cap 4266 constant for both CPU settings.

## Launch validation

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_phase_validation_nohup.sh validate \
  --preset full \
  --run-dir data/protocol_v2/thor_phase_validation_full_v1
```

Optional quick path: use `--preset quick` and a different directory such as
`data/protocol_v2/thor_phase_validation_quick_v1`. Quick executes five blocks
on 512/512 with one warmup and one measured request each. It cannot establish
confidence intervals or validated gains. Do not reuse the quick directory for
full validation.

Both stages run under nohup after foreground sudo authentication. The launcher
prints the supervisor/worker PID, log and graceful stop command. Resume with
the identical command; completed atomic blocks are skipped. Configuration,
source, selection-artifact, model, runtime and platform fingerprints are frozen.
The completed model checksum must match the selection model. Unsupported
frequencies, temperature >=85 C, insufficient power samples and inconsistent
GPU/CPU lock observations are errors, not silently discarded measurements.
Median clock checks are not proof of every-cycle lock fidelity; raw telemetry
and clock extrema remain available.

Use SIGTERM, not SIGKILL. After a crash with pending restoration:

```bash
sudo -E .venv-thor/bin/python scripts/active/run_phase_validation_v1.py restore \
  --run-dir data/protocol_v2/thor_phase_validation_full_v1
```

## Reports and evidence

`RESULTS.md`, `summary.json/csv`, `paired_comparisons.json/csv` and
`slo_attainment.csv` are produced after measurements and hardware restoration.
Bootstrap reporting may take additional time; a `measurements_complete` line
is followed by restoration and then `validation_complete`.

Energy and power differences pair the same workload and repeat against each
baseline. Requested token counts and prompt hashes must match. Whole-block VIN
energy is integrated on its own measurement interval. It is not battery energy,
and the results do not include serving idle periods, arrival queues or network
latency. Token-count-normalized energy includes both prefill and decode.

For uncertainty, resample complete paired repeat blocks 10,000 times. The three
predeclared primary energy comparisons are energy-candidate versus MaxClock,
one per workload. Each uses alpha=0.05/3, a 98.33% percentile interval, as a
Bonferroni adjustment across that family. Bootstrap intervals are approximate,
especially for small effects and serial environmental drift. Secondary energy
comparisons and power intervals are nominal 95% without multiple-comparison
correction. A negative mean with an interval crossing zero is inconclusive.
All workloads and candidates must be reported, including failures.

SLO scenarios remain separate: TPOT-only, joint TTFT<=1 second, joint
TTFT<=2 seconds, each crossed with TPOT<=50/75/100 ms. No long-input threshold
is silently relaxed. Observed 95% request attainment is not a statistical
guarantee. The request-level Wilson interval assumes independent requests;
within-block correlation can make it optimistic. Energy evidence and SLO
evidence are separate, so a lower-energy point failing the selected TTFT
scenario is not an SLO-constrained winner.

## Paired Nsight diagnosis

Run this only after the validation run has released the hardware. The global
hardware lock prevents overlapping captures and benchmark jobs. This command
automates eight captures sequentially: two CPU settings x two phases x two
counterbalanced repeats.

```bash
sudo -E bash scripts/active/run_phase_validation_nohup.sh profile-pair \
  --profiler nsys \
  --run-dir data/protocol_v2/thor_cpu_pair_nsys_v1
```

The pair holds GPU at 1485 MHz and EMC cap at 4266 MHz. CPU is either fixed at
1458 MHz or dynamic with the same 1458 MHz upper cap. Prompt length is 512;
the existing capture implementation warms up and captures prefill plus first
sampling or eight subsequent decode steps. Nsight uses CUDA profiler API
ranges and CUDA graph node tracing. It collects CUDA/OS runtime events, not CPU
instruction samples. Compare GPU kernel durations, host CUDA API waits, copies
and launch gaps; exact attribution to CPU sampling functions may require a
later CPU-stack profile. Tool overhead means these captures cannot quantify
the benchmark energy or speedup.

Reports, individual logs, worker PID files and completion markers are under
`cpu_pair/`. The parent log announces each child log and the final report.
The supervisor's SIGTERM forwards to the Python capture worker first and waits
for hardware restoration. SIGKILL can leave a pending snapshot. In that case
recover using the original phase diagnostic restore entry and the child run:

```bash
sudo -E .venv-thor/bin/python scripts/active/run_phase_diagnostic_v1.py restore \
  --run-dir data/protocol_v2/thor_cpu_pair_nsys_v1/cpu_pair/hardware_run
```

Profilers are optional for validation and are not installed by this change.
They must be ARM64 versions compatible with Thor/CUDA and available on root's
PATH. Missing tools fail explicitly. After inspecting nsys, optional
`--profiler ncu` with a new output directory runs the same pair with bounded
16-launch memory/SM/occupancy sections and no profiler clock override. This is
a small kernel sample, not exhaustive kernel coverage; targeted kernels may
need follow-up captures. Profiler identity is frozen for resumption.

References:
- https://docs.nvidia.com/nsight-systems/UserGuide/
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/

After independent validation, add only supported CPU-floor/governor variants
to the platform-specific characterization space. Then return to per-workload
static characterization, constrained multi-objective selection and independent
serving validation. Do not substitute these few candidates for that full flow,
and do not attribute a benefit to a phase-switch policy before measuring its
switching latency and energy.

Creating these files does not launch tests, benchmarks or profilers.

## Archived outcome

The full validation completed 300/300 blocks. The locked energy candidate was
lower in J/token than MaxClock by about 1.23% for 64/64, 0.70% for 512/512
and 0.56% for 2048/128, with paired intervals below zero under the declared
family correction. These are static candidate-validation results, not evidence
that a phase switch is beneficial. The separate 180-block phase-switch gate
did not promote any switched candidate, so this validation remains a
platform-specific static-control result.
