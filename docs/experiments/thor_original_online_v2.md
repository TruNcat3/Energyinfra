# Original online policy integration on Thor

## What is imported unchanged

- `WorkloadCapSelector.select(..., strategy='pareto')` and its original three-objective frontier and knee-point selection, with no additional SLO constraint in the serving Pareto baseline.
- BestStatic selects `min_energy` at p512/o512, as the original serving entrypoint does. It is not a new trace-weighted global optimum.
- ThermalSLO uses the original initial Pareto selection and original `update_cap` rules, thresholds, hysteresis, cooldown and priority order. It controls only GPU caps. No queue-sensitive rule or policy bug fix is inserted.
- Dynamic and MaxClock run again alongside all three policies; historical results are not reused as contemporary baselines.

## Explicit adaptations and limits

Configuration lives in `configs/experiments/thor_original_online_v2.json`. Default CPU cap is 1458 MHz and EMC cap 4266 MHz. The full static table already includes all 10 GPU caps for this slice and all 12 workloads, so no extra profiling is necessary. Another measured CPU/EMC slice requires a separate config/output directory. Do not collapse different CPU/EMC points into duplicate GPU rows: the original controller cannot choose those dimensions.

The adapter exports corrected v2 block medians with the original selector's column names. No fake lock/decode table is generated. CAP whole-request energy and latency retain their real measurement scope. Full Pareto frontiers and original selection dictionaries are saved before running hardware. A legacy `slo_met: true` is never used as evidence of measured feasibility.

Real token steps, cold request prefix state, continuous VIN sampling and boundary-aligned integration come from v2. The original trace generator is replayed open-loop with a bounded FIFO and full drain. This is intentionally NOT the original scheduler, which did not honor generated arrival timestamps. Window updates occur after request boundaries, approximately every 10 seconds as in the original; an in-flight long request can delay feedback. A final partial window is measured where resolvable, but does not trigger a pointless post-work control change.

Thermal feedback receives TPOT violations only, as in the original. Reports separately evaluate TPOT and joint client TTFT (including queue/control overhead). Failing joint SLO is a result, not grounds to change the controller secretly. Temperature uses the discovered platform junction/CPU/SoC sensor and endpoint maxima, not a claim of continuous GPU peak temperature. Power windows include idle and integration/wait overhead consistently across strategies.

The original controller's power-lowering hysteresis counter can be reset in the thermal-opportunity branch. This behavior is retained; policy bug fixes must be a separately named variant. Its critical-temperature action raises GPU cap by design; hardware thermal protection remains enabled.

MaxClock locks available GPU/CPU maxima and caps EMC; it does not change nvpmodel or claim a MAXN power mode. All policies use v2 native minimum bounds. Record nvpmodel, cooling and ambient conditions separately and keep them unchanged. Strategy order is randomized within each of three repeat blocks; repeats are interleaved between strategies, but the same process remains warm. This does not establish statistically significant small savings on its own.

## Run and outputs

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_energyinfra_original_online_v2_nohup.sh \
  --parent /home/wt/energyinfra/Energyinfra-main/data/protocol_v2/thor_20260915 \
  --run-dir /home/wt/energyinfra/Energyinfra-main/data/protocol_v2/thor_original_online_v2
```

Default: short_chat, five strategies, three repeats, 600 seconds of arrivals per run plus complete drain. The log prints per-window controller decisions. Approximate minimum arrival-window time is 150 minutes; actual runtime includes load, warmup and drain. Long/bursty traces are selectable via a copied config and new directory, not by altering a running protocol.

Outputs: `original_cap_slice.parquet`, `original_policy_plan.json`, `schedule.json`, frozen traces, raw event and telemetry logs, per-trial requests/windows, `comparison.json`, `RESULTS.md`. Drops count as SLO failures. Energy deltas are shown only after matching repeats, traces and completed token work. Joint/TPOT attainment are separate columns; only >=95% observed attainment passes the configured target, not a confidence-bound guarantee.

Reissue the identical command to skip completed trials. SIGTERM restores entry hardware state. After SIGKILL/power loss, run the Python entrypoint with `restore --parent ... --run-dir ...` before resuming. Do not run other hardware jobs concurrently. Original source, adapter, settings, selected profile evidence and model/runtime provenance are checked on resume.

No tests, dry runs or hardware runs were performed while adding this adapter. True prefill/decode phase switching remains outside this online-policy integration; these are whole-request CAP experiments.
