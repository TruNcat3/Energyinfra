# Thor queue-aware Pareto hardware validation v1

This campaign validates the two non-dominated policies produced by the full
1,041,216-policy queue-aware search.

Strategies:

- `queue_energy_min`: GPU 1575 MHz only for `512x512`; GPU 1485 MHz otherwise.
- `queue_power_min`: GPU 1485 MHz for all four workloads.
- Both queue policies use EMC 4266 MHz, CPU cap 1458 MHz, CPU floor 486 MHz and cap mode.
- `balanced_weighted`, Dynamic, MaxClock and MaxClockIdleDynamic are rerun as paired baselines.

The generic runner uses `long_context_2s/100ms` as an execution adapter. After
the hardware run, raw request records are independently reclassified under all
six target TTFT/TPOT profiles.

Prepare without hardware changes:

```bash
bash scripts/active/run_thor_queue_aware_hardware_validation_nohup.sh prepare
```

Run the hardware campaign:

```bash
sudo -E bash scripts/active/run_thor_queue_aware_hardware_validation_nohup.sh run
```

After completion, produce the six-profile report:

```bash
sudo -E bash scripts/active/run_thor_queue_aware_hardware_validation_nohup.sh analyze
```

The run is not considered successful merely because power falls. Fixed-work
total energy, completion, TTFT, TPOT, E2E and every-repeat attainment must all
pass. Queue candidates remain predictions until this campaign completes.

## Archived outcome

The hardware validation completed with 70/70 requests for every retained
strategy. Dynamic reached 81.43% joint SLO attainment and was not feasible;
MaxClock, MaxClockIdleDynamic, BalancedWeighted, QueueEnergyMin and
QueuePowerMin reached 100% observed attainment. QueuePowerMin measured about
52.047 W and 3.219 J/token, approximately -5.53% power and -5.47% J/token
against MaxClock. This is evidence for the frozen long-generation arrival
protocol only, not for single-user or P8 transfer.
