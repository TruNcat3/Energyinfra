# Thor phase/frequency diagnostic

This is an attribution experiment, not a replacement for EnergyInfra's static
table, workload-dependent Pareto search, independent validation or original
online controller evaluation. It does not yet change frequency at the
prefill/decode boundary. Historical code, models and results are untouched.

## Configuration and scope

Load `configs/experiments/thor_phase_diagnostic_v1.json`. It references the base
hardware profile, uses the paired full-vocabulary random Qwen model already
built by the attribution preparation, and requires L4T R38.2.2 / the base
profile's Ubuntu 24.04 and 6.8.12-tegra kernel. Another platform/version needs
its own configuration and supported controller adapter, not an automatic
frequency substitution. Random synthetic tokens do not establish text quality.

The full campaign has 18 configurations, three workloads and three repeats:
162 blocks. Twelve points cross four locked GPU frequencies with three EMC
caps, keeping CPU frequency locked at 1458 MHz. Additional points isolate CPU
972/2601 MHz, CPU dynamic scaling with GPU still locked, historical-style CAP,
Dynamic and MaxClock. EMC remains cap-only; MaxClock does not select nvpmodel.
Runtime clocks can still be limited by thermal/power management.

Each block warms up once and measures consecutive complete requests for at
least five seconds. Long requests can exceed this duration. Workloads are
64/64, 512/512 and 2048/128 input/output tokens, with fresh prefix state every
request. Three repeats are diagnostic sampling, not a claim of 95% SLO
attainment. The quick campaign has five configurations and one 512/128 block
per configuration. Use separate directories for quick and full.

## Run, resume and stop

Authenticate sudo before nohup. The launcher refuses non-root execution, so
an unattended password prompt cannot stall this run. It prints the actual
Python PID and log path. Start with quick:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_phase_diagnostic_nohup.sh \
  --preset quick \
  --run-dir data/protocol_v2/thor_phase_diag_quick_v1
```

After inspecting the quick results, start the full campaign:

```bash
sudo -E bash scripts/active/run_phase_diagnostic_nohup.sh \
  --preset full \
  --run-dir data/protocol_v2/thor_phase_diag_full_v1
```

Resume using the identical command. Completed atomic block records are skipped.
Configuration, model, source and platform fingerprints must match. Use the
printed `sudo kill -TERM PID` to stop; do not use SIGKILL. Existing v2 hardware
experiments and this runner share a global hardware lock.

If a crash or SIGKILL left `restore_pending.json` true, first recover the entry
GPU/CPU min/max/governors and EMC cap, then resume:

```bash
sudo -E .venv-thor/bin/python scripts/active/run_phase_diagnostic_v1.py restore \
  --run-dir data/protocol_v2/thor_phase_diag_full_v1
```

Do not manually change hardware settings while an experiment owns the lock.
The temperature guard aborts at 85 C; it does not silently select another point.

## Measurement definitions and telemetry correction

- TTFT: prompt evaluation plus first-token sampling, with no serving queue or
  network. Tokenization/prompt construction and reset follow the unchanged v2
  measurement convention.
- Decode: interval from the first returned token to the last, divided by
  output_tokens minus one for TPOT. EOG/special IDs count as real token steps.
- Whole-block energy: continuous VIN integration, including both phases and
  small between-request/reset overheads. It is not calibrated battery energy.
- Phase energy: interpolate the same receive-timestamped power stream at phase
  boundaries. Require at least three internal samples and at least three sample
  intervals of duration. This is a coarse estimate, not kernel-level energy.
  Short/undersampled phases have null energy, never invented values from one
  sample. Group phase power is omitted unless every request phase is measurable.
- Actual EMC/GPU and all CPU core frequencies come from the same tegrastats
  records, with original lines retained. The old debugfs EMC clk_rate reading
  that remained at 665 MHz is not used as an actual frequency measurement.
  EMC utilization is retained as a vendor-reported percentage, not converted
  into achieved GB/s. Kernel memory bandwidth requires profiler counters.

`RESULTS.md`, `summary.json` and `summary.csv` contain TTFT/TPOT, whole-block
power, J/output-token and phase power coverage. CSV includes deltas against
both Dynamic and MaxClock. `slo_sensitivity.csv` reports all combinations of
TPOT 50/75/100 ms and TTFT 1/2 seconds, request counts and Wilson intervals.
These are observed diagnostic pass rates, not independent validation or newly
selected Pareto optima. Raw blocks include clock histograms, all-core medians,
temperatures and EMC cap excursions. A requested lock is not proof that actual
clocks stayed fixed; inspect these observations before attributing a difference.

## Optional profiling, separate from energy measurements

Profilers are not needed for quick/full sweeps. If installed, run Nsight Systems
first to see CUDA kernels, host API waits, copies and launch gaps. Capture only
prefill or eight decode steps, after warmup. For example, after the full run:

```bash
sudo -E bash scripts/active/run_phase_profiler_nohup.sh nsys decode \
  --run-dir data/protocol_v2/thor_phase_diag_full_v1 \
  --point g1485_e4266_c1458_lock --prompt-tokens 512

sudo -E bash scripts/active/run_phase_profiler_nohup.sh nsys prefill \
  --run-dir data/protocol_v2/thor_phase_diag_full_v1 \
  --point g1485_e4266_c1458_lock --prompt-tokens 512
```

Do not run these commands concurrently. Compare against
`g738_e4266_c1458_lock`, `g1485_e2750_c1458_lock` and the CPU control points
after identifying the dominant execution components. Profile captures are
stored separately and never enter the benchmark report.

For selected kernels, use Nsight Compute:

```bash
sudo -E bash scripts/active/run_phase_profiler_nohup.sh ncu decode \
  --run-dir data/protocol_v2/thor_phase_diag_full_v1 \
  --point g1485_e4266_c1458_lock --prompt-tokens 512
```

This requests SM/memory workload and occupancy sections, disables ncu's own
clock control and limits collection to 16 launches by default. It does not
cover every kernel or layer. Set `PHASE_NCU_KERNEL_REGEX` to a kernel identified
by nsys and `PHASE_NCU_LAUNCH_COUNT` to a small desired sample count. The Nsight
version must support this Thor/CUDA stack; missing tools or unsupported counters
are errors, not simulated results. Add installed profiler binaries to root's
PATH if necessary. No profiler packages or privileged permission rules are
installed by these scripts.

Profiler replay/tracing changes execution. Do not use its timings, clocks or
power to replace the non-profiled benchmark numbers. nsys's node-level CUDA graph
trace is intentionally diagnostic. Stop the Python worker via the worker PID
file printed by the profiler launcher, allowing normal hardware restoration.

Tool references:
- https://docs.nvidia.com/nsight-systems/UserGuide/
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/

No tests, hardware runs or profiler captures are performed merely by adding
these scripts. Once the diagnosis supports a candidate, separately implement
and validate phase switching, counting switch latency/energy and preserving
identical requests and SLO criteria against both baselines.
