# Thor edge serving SLO tiers v2

The strict 500/30 ms profile is retained only as a reference. Edge deployments
have heterogeneous application requirements and should not be judged by one
datacenter-oriented interactive threshold.

This sensitivity audit evaluates seven tiers from strict interactive through
delay-tolerant operation. Every request is checked against TTFT, TPOT and an
output-length-derived E2E deadline. A strategy is eligible only when completion
is 100 percent and joint attainment is at least 95 percent in every repeat.
P99 compliance remains a separate, stricter diagnostic.

| Tier | TTFT | TPOT |
|---|---:|---:|
| strict reference | 500 ms | 30 ms |
| edge fast | 1000 ms | 40 ms |
| edge interactive | 1000 ms | 50 ms |
| edge balanced | 2000 ms | 50 ms |
| edge conversational | 2000 ms | 75 ms |
| edge relaxed | 3000 ms | 100 ms |
| edge background | 5000 ms | 150 ms |

These are declared sensitivity tiers, not universal standards. The strict and
conversational endpoints are informed by current MLCommons ranges; intermediate
tiers expose the tradeoff surface on Thor.

Run the offline audit with:

```bash
cd /home/wt/energyinfra/Energyinfra-main
.venv-thor/bin/python scripts/active/analyze_thor_slo_profiles_v1.py \
  --config configs/experiments/thor_slo_profiles_v2.json
```

Results are written to `reports/thor_slo_audit_v2/`.
