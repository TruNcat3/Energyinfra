# Thor phase-switch gate v1

## Purpose

This experiment decides whether a first-token prefill/decode frequency switch has enough fixed-work
energy benefit to justify a queue-aware serving experiment. It does not assume that lower power means
lower energy.

The random-weight Qwen2.5-7B Q4 model is retained to keep model shape and kernels consistent. Results
are systems-only and make no output-quality claim.

## Design

- Workloads: `128/512`, `512/512`, and `1024/512` prompt/output tokens.
- Prefill and no-switch baseline: locked GPU 1485 MHz, EMC 4266 MHz, CPU 1458 MHz.
- The switch occurs immediately after the first sampled token.
- Decode candidates independently reduce GPU or EMC, plus a 1575 MHz control.
- `same_point_switch` isolates pure control overhead.
- Candidate order is deterministic and randomized across 10 paired repeats.
- VIN input energy covers generation and restoration to the prefill point.
- Generated token count is fixed; early EOS does not shorten work.
- Raw JSONL is fsynced after every block, so an interrupted full run resumes safely.

The primary gate requires both:

1. At least 95% attainment for TTFT <= 1000 ms and TPOT <= 50 ms.
2. The upper endpoint of the paired 95% confidence interval for energy change is below zero.

Power, energy/token, TTFT, TPOT, first-ITL, and transition latency are all retained. Pareto membership
does not imply a universal optimum.

## Run

Quick plumbing run:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_phase_switch_gate_nohup.sh quick
```

Full paired experiment:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_phase_switch_gate_nohup.sh run
```

The launcher prints the log path, PID, `tail -f` command, and graceful-stop command. Re-running the
same command skips completed blocks. Do not mix quick and full presets in one output directory; use a
separate directory when a quick run already exists:

```bash
sudo -E bash scripts/active/run_thor_phase_switch_gate_nohup.sh quick \
  --run-dir data/protocol_v2/thor_phase_switch_gate_quick_v1
```

Regenerate the report without touching clocks:

```bash
bash scripts/active/run_thor_phase_switch_gate_nohup.sh report
```

Emergency dynamic restore:

```bash
sudo -E bash scripts/active/run_thor_phase_switch_gate_nohup.sh restore
```

## Outputs

- `data/protocol_v2/thor_phase_switch_gate_v1/protocol.json`
- `data/protocol_v2/thor_phase_switch_gate_v1/raw_blocks.jsonl`
- `data/protocol_v2/thor_phase_switch_gate_v1/summary.json`
- `data/protocol_v2/thor_phase_switch_gate_v1/RESULTS.md`

## Promotion status

The gate is intentionally stricter than a power comparison. A candidate must clear both the joint SLO
attainment rule and the paired energy confidence bound before it can be used in queue-aware or thermal
serving. Until that report is archived, phase-aware DVFS remains `partial`; a diagnostic first-token
switch must not be described as an online optimization result.

Only a candidate that clears the energy/SLO gate should be promoted into the frozen-trace serving
protocol. A lower-power candidate that consumes more joules for the same 512 generated tokens remains
a valid power Pareto point, but is not an energy-efficiency improvement.
