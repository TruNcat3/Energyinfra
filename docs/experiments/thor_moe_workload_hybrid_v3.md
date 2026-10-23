# Thor MoE workload-aware hybrid v3

v2 showed that a single phase-aware policy is not uniformly useful: native
dynamic is best for short requests, static peak guard is competitive for
128-token outputs, and longer outputs have different phase candidates. v3
tests a workload-aware hybrid selector inside one mixed fixed trace.

## Policy under test

- `short_128`: native dynamic throughout.
- `long_128`: static `g1359_e3200_c972` peak guard.
- `long_256`: high-resource Prefill followed by lower-EMC Decode.
- `long_512`: max-clock Prefill followed by balanced Decode.

The hybrid is compared with native dynamic and static peak guard under the same
mixed arrival trace. All requests are sequential and non-overlapping because
Thor hardware frequency controls are global.

## Trace and commands

The trace contains `short_128`, `long_128`, `long_256`, `long_512`, another
`short_128`, and another `long_256` request over a 120-second fixed horizon.
The full preset repeats the trace three times per strategy.

```bash
cd /home/wt/energyinfra/Energyinfra-main

sudo -E bash \
  scripts/active/run_thor_moe_phase_aware_serving_v3_nohup.sh \
  quick run
```

After the quick gate:

```bash
sudo -E bash \
  scripts/active/run_thor_moe_phase_aware_serving_v3_nohup.sh \
  full run
```

Results are written below
`data/protocol_v2/thor_moe_workload_hybrid_v3/<preset>/`.
