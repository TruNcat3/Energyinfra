# Thor Q4 TTFT-extension protocol

This stage tests whether the remaining optimization space is constrained by
TTFT rather than TPOT. It keeps the measured Q4 static frontier and the frozen
long-generation traces unchanged.

Profiles:

| Profile | TTFT p95 prefilter | TPOT p95 prefilter |
|---|---:|---:|
| `long_context3s_75ms` | 3000 ms | 75 ms |
| `long_context3s_100ms` | 3000 ms | 100 ms |
| `background5s_150ms` | 5000 ms | 150 ms |

For every workload and profile, the offline preparation does the following:

1. Loads measured `frontier_validation.json` rows from the complete static campaign.
2. Reclassifies feasibility using measured TTFT p95 and TPOT p95.
3. Recomputes the feasible non-dominated set over power, energy/token, TTFT and TPOT.
4. Keeps the complete Pareto set and emits four preference-dependent representatives.

The weighted representatives are not universal optima. They are reproducible
operating-point choices for energy-, power-, latency- and balanced preferences.

Static p95 reclassification is only a candidate prefilter. A point is considered
SLO-feasible only after the fixed frozen traces pass completion, TTFT, TPOT and
derived end-to-end constraints at request level in every repeat.

Prepare the maps without changing hardware state:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_ttft_extension_prepare.sh
```

If the existing `data/protocol_v2` tree is still root-owned, repair ownership once:

```bash
sudo chown -R wt:wt /home/wt/energyinfra/Energyinfra-main/data/protocol_v2
```

Outputs are written to:

```text
data/protocol_v2/thor_q4_ttft_extension_v1/plans.json
data/protocol_v2/thor_q4_ttft_extension_v1/comparison.json
data/protocol_v2/thor_q4_ttft_extension_v1/protocol.json
```
