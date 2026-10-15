# Thor prompt-aware targeted experiments v1

This protocol contains two coupled-trace follow-ups.

`capacity_upper` tests 30, 32, and 36 output-token/s with six baseline and
Pareto strategies. It uses one exact 4:3:2:1 workload block per repetition and
locates the upper boundary for the prompt-aware 100 ms TPOT SLO.

`tpot30_stability` tests 18, 20, and 22 output-token/s with the C3 P2 and C3 P8
strategies. It uses two workload blocks, or 20 requests and 4864 output tokens,
per repetition. Its primary analysis applies the conventional 95 percent gate
to each complete repetition. A separate prompt-bucket diagnostic retains the
more conservative per-bucket gate.

Prepare both scenarios without root:

```bash
.venv-thor/bin/python scripts/active/prepare_thor_prompt_aware_targeted_v1.py --scenario all
```

Run each scenario separately with hardware-control permission:

```bash
sudo -E bash scripts/active/run_thor_prompt_aware_targeted_nohup.sh capacity_upper run
sudo -E bash scripts/active/run_thor_prompt_aware_targeted_nohup.sh tpot30_stability run
```

Do not start both hardware runs concurrently. Analyze a completed scenario with
the same wrapper and the `analyze` stage.
