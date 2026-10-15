# Thor Q4 long-generation profile-specific SLO sweep v1

This sweep complements the completed 2000/50 ms campaign. It reruns the exact
same three frozen arrival-scale-5 traces with policy maps independently selected
and validated for:

| Campaign | TTFT | TPOT |
|---|---:|---:|
| long_context_2s / 75 | 2000 ms | 75 ms |
| long_context_2s / 100 | 2000 ms | 100 ms |

The 1000/50 ms audit is not launched with an unvalidated fallback: at least one
long workload has no independently validated Pareto-policy candidate under that
profile, and the measured serving audit already leaves only MaxClock feasible.

Dynamic, MaxClock and MaxClock-idle-dynamic are rerun in each campaign so every
energy delta remains paired with the same run order, thermal history and traces.
Each campaign randomizes strategy order per repeat and is independently resumable.

Run both profiles sequentially under one nohup supervisor:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_long_generation_slo_sweep_nohup.sh run
```

The minimum fixed-horizon time is about seven hours. Full drain may extend it.
After completion, the 2000/100 ms result can also be conservatively audited under
3000/100 ms; a policy that passes 2000 ms necessarily passes the looser TTFT gate.

If an ungraceful stop leaves restoration pending:

```bash
sudo -E bash scripts/active/run_thor_long_generation_slo_sweep_nohup.sh restore
```
