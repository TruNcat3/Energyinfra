# Thor targeted edge confirmation v1

## Purpose

This protocol is a targeted confirmation after the completed broad edge
campaign. It does not repeat the full static table. It uses five already
measured P1 candidates and keeps parallelism at one to represent the common
low-batch edge case:

- native dynamic;
- MaxClock;
- QueuePowerMin;
- low-EMC candidate;
- high-performance c3 candidate.

The serial and bursty campaigns use separate configs, ports, trace modes and
output directories. Their results must not be merged into one ranking.

## Campaigns

| Mode | Trace | Rates | Repeats | Output |
|---|---|---|---:|---|
| serial | 30-second think-time | 8 tok/s | 3 | `data/protocol_v2/thor_edge_confirmation_v1/serial/full` |
| bursty | coupled stratified | 4/8/12 tok/s | 3 | `data/protocol_v2/thor_edge_confirmation_v1/bursty/full` |

Both modes use the same 128/512/1024 prompt buckets, Qwen2.5-7B random Q4,
Stage 1B patched runtime, P1, batch 512, ubatch 256 and the three declared
SLO profiles. A candidate is formally feasible only when the protocol reports
all-repeat attainment at least 95%; a power-only or near-feasible point remains
diagnostic.

## Run

Prepare as a normal user:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_edge_confirmation_nohup.sh serial full prepare
bash scripts/active/run_thor_edge_confirmation_nohup.sh bursty full prepare
```

Run hardware stages separately with root:

```bash
sudo -E bash scripts/active/run_thor_edge_confirmation_nohup.sh serial full run
sudo -E bash scripts/active/run_thor_edge_confirmation_nohup.sh bursty full run
```

Generate each protocol report without root:

```bash
bash scripts/active/run_thor_edge_confirmation_nohup.sh serial full report
bash scripts/active/run_thor_edge_confirmation_nohup.sh bursty full report
```

After both reports complete, regenerate the cross-protocol evidence index:

```bash
python3 scripts/active/analyze_thor_unified_evidence_v1.py \
  --data-root data/protocol_v2 \
  --out-dir reports/thor_unified_evidence_v1
```

The unified analyzer scans these directories when they exist and labels them
as `edge_confirmation_serial` and `edge_confirmation_bursty`. Missing
confirmation directories remain visible in `evidence_sources.json`; rows are
never copied into or used to overwrite the historical campaigns.

## Decision rules

Report Power and Energy/token separately. A candidate can be power-Pareto but
energy-dominated. Compare each candidate with native dynamic and MaxClock only
within the same mode, rate, trace and SLO profile. If serial and bursty choose
different points, that is a workload-aware result rather than an inconsistency.

## Completed results

Both full campaigns completed with fixed work and hardware restoration:

| Mode | Completed trials | Formal result |
|---|---:|---|
| serial | 15/15 | QueuePowerMin is feasible for interactive and relaxed SLOs; no candidate is all-repeat feasible for strict SLO |
| bursty | 45/45 | No candidate is all-repeat feasible at 95% under strict, interactive or relaxed SLO |

For the serial `rate_008` profile, QueuePowerMin measured 37.47 W and
4.684 J/token. Relative to native dynamic this is approximately 4.77% lower
power and 4.77% lower energy/token, while TTFT P95 improved from 895.1 ms to
803.2 ms and TPOT P95 improved from 38.4 ms to 26.4 ms. Relative to MaxClock,
QueuePowerMin reduces power and energy/token by approximately 9.88%, trading
about 118.7 ms TTFT P95 and 2.0 ms TPOT P95 while remaining within the
interactive SLO. The c3 point is a separate lower-latency Pareto tradeoff; the
low-EMC point is dominated by QueuePowerMin in this campaign.

For bursty rates 4/8/12 tok/s, QueuePowerMin lowers power by approximately
4.07%/2.62%/2.83% and energy/token by approximately 5.01%/4.47%/6.52% versus
native dynamic. These are diagnostic savings, not constrained wins, because at
least one repeat misses the 95% attainment gate at every rate. Queueing delay,
rather than token generation alone, is the dominant feasibility limit.

The completed evidence therefore supports a scenario-scoped claim: Thor has a
useful serial low-concurrency Pareto point, but the same point is not formally
validated for bursty low-concurrency traffic. It does not support a universal
Thor advantage.
