# Thor MoE phase-aware serving v1

This experiment ports the phase-aware idea used in the Orin workflow to Thor.
It separates the resource policy for prefill and decode instead of assigning a
single frequency point to the complete request.

## Method

- Prefill hardware is applied before dispatching each request.
- The first streamed output token is the phase boundary.
- The controller applies the decode hardware immediately after that boundary.
- Energy is integrated over the same frozen request horizon for every strategy.
- The phase-switch latency is retained in the fixed-window measurement.
- Only sequential, non-overlapping traces are used in v1 because Thor frequency
  controls are global; independent phase changes for concurrent requests would
  otherwise be ambiguous.

The experiment uses the random-weight Qwen3 MoE GGUF already used by the Thor
MoE experiments. It makes no model-quality claim.

## Strategies

- `static_native_dynamic`: native dynamic governor throughout.
- `static_balanced_1575_3200_972`: previously validated fixed CAP baseline.
- `static_peak_guard_1359_3200_972`: lower-power static baseline.
- `phase_max_to_balanced`: maximum-clock prefill followed by balanced decode.
- `phase_high_to_memory_aware`: high-resource prefill followed by lower GPU/EMC decode.
- `phase_power_guard`: moderate prefill followed by a low-power decode point.

## Commands

Run the one-repeat gate first:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_moe_phase_aware_serving_nohup.sh quick run
```

After the quick run is complete, run the three-repeat experiment:

```bash
sudo -E bash scripts/active/run_thor_moe_phase_aware_serving_nohup.sh full run
```

The report is written to:

`data/protocol_v2/thor_moe_phase_aware_serving_v1/<preset>/RESULTS.md`

The report must be interpreted together with the fixed-horizon MoE report. A
phase-aware point is useful only if its fixed-window energy and power gains
remain after including phase-switch overhead and it remains feasible under the
selected SLO.
