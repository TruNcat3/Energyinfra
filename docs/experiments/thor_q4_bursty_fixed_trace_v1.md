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

## 当前结论边界

该协议的 burst 会产生排队和 drain，因此不能代表纯单用户 edge。当前 full campaign 中，`edge_c3_p2 @ rate_012` 在 TPOT 100 ms 下约有 90% 总体达标率，三次 repeat 为 `[100%, 80%, 80%]`，平均功率约 40.5 W、TPOT P95 约 30.9 ms。它是接近可行候选，不满足 95% 的保守正式 SLO。

正式归档还必须保留每个 repeat 的达标率、失败请求、queue/drain 时间、总能量和 J/token。quick 或只保留平均值的结果不能替代 full campaign。
