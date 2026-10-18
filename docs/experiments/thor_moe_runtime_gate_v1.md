# Thor MoE runtime gate v1

## Purpose

This is a non-root model/runtime gate before any MoE energy experiment. It does
not change hardware frequencies, does not claim Power or Energy/token results,
and does not copy a model into the repository. It verifies that an external
GGUF can load under the pinned Thor llama.cpp build with three expert placements:

- all experts on GPU;
- all MoE experts on CPU;
- first 24 MoE layers on CPU.

The first candidate is Qwen3-30B-A3B: 30.5B total parameters, 3.3B activated
parameters, 128 experts and 8 activated experts per token. The official model
card is at [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B). The model
must remain outside the repository because the Thor root filesystem currently
has limited free space.

## Gate conditions

Every placement must satisfy:

- server health becomes ready;
- the 16-token warmup completes exactly;
- all 128/128 and 1024/128 fixed-output requests complete exactly for two repeats;
- the placement arguments are accepted by the pinned runtime;
- model and binary hashes are written to `manifest.json`.

This gate is only a prerequisite. It does not establish that CPU-MoE is faster,
more efficient or better for the final paper.

## Run

Place a compatible GGUF on a filesystem with enough free space, then run as a
normal user:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_moe_runtime_gate_nohup.sh \
  /external/models/Qwen3-30B-A3B-Q4_K_M.gguf
```

Follow the log printed by the wrapper. No `sudo` is required and no hardware
frequency is changed. Outputs are written to:

```text
data/protocol_v2/thor_moe_runtime_gate_v1/
```

The next stage is not automatic: only after this gate succeeds should the model
be connected to a fixed-trace energy protocol with native dynamic, MaxClock and
QueuePowerMin baselines.
