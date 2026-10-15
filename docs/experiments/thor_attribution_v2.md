# Small attribution probes

This stage separates model dimensions and controller logic before another long
serving campaign. Historical models, controllers, frequency tables and results
are not changed. No promise of better energy efficiency is assumed.

## Paired models

Both are newly constructed random Qwen2.5-shaped models, with 32768 versus 152064
vocabulary entries. Layer/head/hidden/FFN shapes match the reference configuration;
Q/K/V biases are included and initialized to zero. Every random tensor uses its
own seed derived from its name: growing the embedding no longer shifts the RNG
stream for all transformer layers. Models retain separate input/output embeddings.
Both use synthetic vocabularies and fixed token-step generation, not production
text semantics or trained-weight model quality. Do not compare the new compact
model to the historical compact one as if weights and biases were unchanged.

Build requires >=30 GiB free disk at each construction step. It writes only inside
the new run directory, uses temporary F16 files and removes its own source only
after quantization succeeds. Existing unrecognized outputs are never overwritten.
An interrupted build with temporary files requires a new directory or explicit
manual cleanup of those temporary files. Full tensor construction can use tens of
GiB RAM; do not run concurrently with other model/hardware jobs.

## Model probe

Two models x five controls x three workloads x three blocks = 90 blocks, rather
than another complete frequency sweep. Controls: Dynamic, MaxClock, GPU738/EMC4266,
GPU1485/EMC4266 and GPU1485/EMC2750; all CAP CPU bounds are 1458 MHz. Workloads:
p64/o64, p512/o512, p2048/o128. Model order is shuffled per repeat block, hardware
order is identical between paired models. Calibration warmup/energy-bracketing and
exact-token machinery are inherited from v2. Raw tegrastats and sampled observed
clocks accompany results. This is not a CUDA-kernel profiler or a power-rail
calibration. Short blocks can contain multiple requests, not independent repeats.

## Controller replay

The original controller and a separately named power-hysteresis-fixed subclass
receive identical synthetic window inputs, a deterministic virtual clock and NO
hardware controller. Scenarios cover sustained power excess, cool idle, SLO
pressure, thermal warning, critical temperature and a broken power streak.
Only power-streak preservation differs in the variant's decision logic. Original
controller code remains untouched. Replay results test behavior, NOT measured
energy/performance and NOT counterfactual feedback on a real workload. The new
variant has not been hardware-validated or added to original-policy runs.

## Execution order

First run `run_thor_attribution_prepare_nohup.sh --parent ... --run-dir ...` without
sudo. Wait for both model builds and controller replay to finish. Then run
`sudo -E bash run_thor_attribution_probe_nohup.sh --parent ... --run-dir ...` using
the same directory. Logs/PIDs are printed. The probe has a hardware mutex,
SIGTERM cleanup and resumable committed blocks. The prepare supervisor is not the
model-builder PID; interrupting preparation may leave temporary build files.

Outputs: model metadata, controller_replay.json, MODEL_RESULTS.md,
model_comparison.json and per-block telemetry. No tests or probes were run when
adding these scripts. Scheduler ablation and live original/fixed-controller A/B
remain subsequent steps, selected based on these small probes, not silently
included in this stage.
