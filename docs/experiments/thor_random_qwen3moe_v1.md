# Thor Random Qwen3-MoE Systems Probe v1

This probe removes the external model-download dependency while preserving the Qwen3-MoE execution path needed for expert-placement research.

## Architecture

| Property | Probe | Official Qwen3-30B-A3B |
|---|---:|---:|
| GGUF architecture | `qwen3moe` | `qwen3moe` |
| Hidden size | 2048 | 2048 |
| Attention heads / KV heads | 32 / 4 | 32 / 4 |
| Head dimension | 128 | 128 |
| Expert width | 768 | 768 |
| Experts / active experts | 128 / 8 | 128 / 8 |
| Transformer blocks | 20 | 48 |
| Total parameters | 13,084,744,704 | about 30.5B |
| Active-parameter estimate | 1,760,123,904 | about 3.3B |

The model is deliberately random-weight and unsuitable for quality evaluation. It is valid only for runtime, placement, latency, power, energy, memory, and thermal experiments.

## Prepare

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/prepare_thor_qwen3moe_random_nohup.sh
```

The builder writes one tensor at a time, retains a committed F16 intermediate after an interrupted quantization, produces Q4_K_M, verifies provenance, and deletes F16 after successful completion. It does not need root and does not change hardware frequencies.

## Runtime gate

After preparation completes:

```bash
bash scripts/active/run_thor_moe_runtime_gate_nohup.sh \
  models/moe_random_v1/Qwen3MoE-13B-A1.8B-random-v151936-Q4_K_M.gguf
```

The gate compares all-GPU, all-CPU-MoE, and 10-layer CPU-MoE placement before any DVFS or full serving experiment is allowed.

