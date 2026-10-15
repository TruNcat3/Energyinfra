# Thor Q4 original bursty fixed-trace serving v1

This experiment freezes the unmodified `bursty_mixed` traces produced by the
repository's original `RequestGenerator`. Background requests and burst requests
are neither thinned nor shifted.

The burst generator submits groups of `p1024_o1024` requests two seconds apart.
The current serving runner is serial FIFO without continuous batching, so a
two-second end-to-end TTFT constraint is structurally infeasible for most requests
inside each burst. Consistent with the original EnergyInfra serving benchmark,
the declared constraint is therefore TPOT <= 50 ms. End-to-end TTFT and queue wait
remain reported as diagnostic capacity metrics.

Freeze and inspect the deterministic traces. Use sudo because the existing
`data/protocol_v2` campaign directory is root-owned:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_bursty_fixed_trace_nohup.sh prepare
```

Then run all seven strategies on the exact same traces:

```bash
sudo -E bash scripts/active/run_thor_bursty_fixed_trace_nohup.sh run
```

The minimum fixed-horizon duration is 3.5 hours. Complete drain after overloaded
bursts can make the wall-clock run longer. Idle energy, drain energy, queueing and
frequency-switch overhead are included.

Results are written to:

```text
data/protocol_v2/thor_q4_bursty_fixed_trace_v1/RESULTS.md
```

If an ungraceful stop leaves restoration pending, run:

```bash
sudo -E bash scripts/active/run_thor_bursty_fixed_trace_nohup.sh restore
```
