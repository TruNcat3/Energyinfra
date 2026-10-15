# Thor Q4 long-generation TTFT-extension serving run

The static reclassification produced identical workload-specific hardware maps
for `3000/75`, `3000/100` and `5000/150`. All measured static long-generation
points already pass the three marginal p95 thresholds; only online queueing can
separate these profiles.

To avoid three physically identical campaigns, this protocol executes the
`5000/150` map once on the existing frozen `arrival_x5` traces. The generic
runner uses its known `long_context_2s/100ms` label only as an execution adapter;
it does not select the hardware map. The custom comparison supplies that map.
The resulting per-request records must then be reclassified offline under all
three target SLOs:

- TTFT 3000 ms, TPOT 75 ms
- TTFT 3000 ms, TPOT 100 ms
- TTFT 5000 ms, TPOT 150 ms

Completion and derived end-to-end latency remain mandatory. A strategy is
feasible only if every repeat reaches at least 95% joint goodput.

Run:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_long_generation_ttft_extension_nohup.sh run
```

The launcher prints the exact log and PID file. It also links missing auxiliary
static artifacts into the custom-selection directory without modifying the
original full-frontier campaign.
