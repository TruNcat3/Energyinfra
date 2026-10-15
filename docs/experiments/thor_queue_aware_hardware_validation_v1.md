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
