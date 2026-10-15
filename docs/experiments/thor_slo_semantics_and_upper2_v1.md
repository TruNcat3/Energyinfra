# Thor SLO semantics and capacity upper extension v1

The semantics analysis compares two common interpretations of a multi-metric
SLO using the completed 20-request TPOT30 stability traces.

- Marginal: every repetition independently reaches 95 percent TTFT attainment
  and 95 percent TPOT attainment.
- Joint request: 95 percent of requests in every repetition satisfy TTFT and
  TPOT simultaneously.

Run the offline semantics analysis with:

```bash
.venv-thor/bin/python scripts/active/analyze_thor_slo_semantics_v1.py
```

The second capacity extension uses the same workload order, prompt seeds, and
normalized arrivals as the 30/32/36 tps experiment. It tests 40, 44, and 48
output-token/s with six strategies and three repetitions, for 54 trials.

Prepare without root:

```bash
bash scripts/active/run_thor_prompt_aware_targeted_nohup.sh capacity_upper2 prepare
```

Run with hardware-control permission:

```bash
sudo -E bash scripts/active/run_thor_prompt_aware_targeted_nohup.sh capacity_upper2 run
```
