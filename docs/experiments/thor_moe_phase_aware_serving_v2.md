# Thor MoE phase-aware serving v2

v1 showed that phase-aware serving can reduce fixed-window energy on long
decode traces, but hardware transitions around `87~111 ms` make frequent
switches unattractive for short requests. v2 tests the two mechanisms that can
change that result:

- Prefill transitions are scheduled `250 ms` before dispatch instead of at the
  request boundary, so the transition does not become dispatch/TTFT delay.
- Requests with fewer than `256` output tokens use a static peak-power fallback;
  only longer outputs switch from Prefill to Decode hardware.
- Long traces use `128`, `256`, and `512` output tokens to estimate the
  amortization threshold of phase switching.

## Strategies

- `static_native_dynamic`: native dynamic throughout.
- `static_peak_guard_1359_3200_972`: static lower-power baseline.
- `phase_long_max_to_balanced`: long output only; max-clock Prefill and balanced Decode.
- `phase_long_high_to_memory_aware`: long output only; high-resource Prefill and lower EMC Decode.

## Protocol

The experiment uses sequential, non-overlapping fixed arrival traces. It keeps
the global Thor hardware-control limitation explicit: this is not yet a
per-request concurrent controller. The full preset repeats each strategy and
trace three times. The model remains the random-weight Qwen3 MoE GGUF, so the
experiment makes no quality claim.

## Commands

```bash
cd /home/wt/energyinfra/Energyinfra-main

sudo -E bash \
  scripts/active/run_thor_moe_phase_aware_serving_v2_nohup.sh \
  quick run
```

After the quick gate:

```bash
sudo -E bash \
  scripts/active/run_thor_moe_phase_aware_serving_v2_nohup.sh \
  full run
```

Results are written below
`data/protocol_v2/thor_moe_phase_aware_serving_v2/<preset>/`.
