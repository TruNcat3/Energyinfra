# Thor queue-aware paired confirmation v1

This campaign adds seven paired blocks for the three strategies that matter:
QueuePowerMin, BalancedWeighted and MaxClockIdleDynamic.

The new traces retain the measured arrival timestamps and workload multiset of
the calibrated traces, but deterministically reorder workload blocks. This
tests hardware-run variability and sensitivity to request order without
claiming to model every production arrival distribution.

Prepare the seven immutable traces without changing hardware:

```bash
bash scripts/active/run_thor_queue_aware_confirmation_nohup.sh prepare
```

Run 21 trials:

```bash
sudo -E bash scripts/active/run_thor_queue_aware_confirmation_nohup.sh run
```

After completion, combine the original three blocks and seven confirmation
blocks, then calculate six-profile SLO results and paired 95% intervals:

```bash
sudo -E bash scripts/active/run_thor_queue_aware_confirmation_nohup.sh analyze
```

The primary decision is whether QueuePowerMin's paired energy interval lies
fully below zero against BalancedWeighted and MaxClockIdleDynamic while every
block continues to satisfy the joint SLO.
