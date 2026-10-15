# Thor prompt-aware SLO analysis v1

This offline analysis replaces a single global TTFT threshold with a limit
conditioned on prompt length. It reads the existing fixed-work serving results
and performs no inference or hardware control.

Two schedules are evaluated:

- `tiered_strict`: 500 ms for 128 tokens, 750 ms for 512 tokens, and 1000 ms
  for 1024 tokens.
- `tiered_balanced`: 750 ms for 128 tokens, 1000 ms for 512 tokens, and 1500 ms
  for 1024 tokens.

Each schedule is crossed with 30, 50, and 100 ms TPOT limits. Feasibility
requires at least 95 percent attainment in every repetition and in every prompt
bucket present in that repetition. This prevents a frequent short-prompt class
from hiding violations in a less frequent long-prompt class.

Run:

```bash
.venv-thor/bin/python scripts/active/analyze_thor_prompt_aware_slo_v1.py
```

Outputs are written under
`data/protocol_v2/thor_slo_capacity_frontier_v1/prompt_aware_slo_v1`.

## Interpretation

Prompt-aware SLO is a feasibility analysis, not a license to select the loosest schedule after seeing
the measurements. The schedule, TPOT threshold, 95% attainment rule, repeat count and prompt buckets
are part of the experiment identity. If `tiered_strict` has no feasible point while `tiered_balanced`
does, report the change in service contract explicitly and keep both frontiers.
