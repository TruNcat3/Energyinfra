# Thor MoE Model Preparation v1

This stage downloads the official Qwen `Qwen3-30B-A3B-Q4_K_M.gguf` artifact before the no-root MoE runtime gate.

## Pinned provenance

- Repository: `Qwen/Qwen3-30B-A3B-GGUF`
- Revision: `e4d4bafdfb96a411a163846265362aceb0b9c63a`
- File size: `18556685824` bytes
- SHA256: `0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5`
- Quantization: `Q4_K_M`

The downloader reserves 8 GiB beyond the remaining download, resumes a `.part` file after interruption, verifies size and SHA256, then atomically renames the artifact and writes a provenance manifest. It does not change hardware frequencies and does not require root.

## Prepare

Create the isolated Hugging Face Xet download environment once:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/setup_thor_moe_download_env.sh
```

The worker prefers the official Xet client and uses adaptive concurrent chunk downloads. If Xet is unavailable, the direct-download fallback requests a fresh signed URL after every failure so an expired CDN redirect cannot terminate a resumable transfer.

Use the repository model directory:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/prepare_thor_moe_model_nohup.sh
```

Or use an external disk:

```bash
ENERGYINFRA_MOE_MODEL_DIR=/mnt/models/moe_v1 \
  bash scripts/active/prepare_thor_moe_model_nohup.sh
```

Follow the log path printed by the launcher. After the manifest reports completion, run:

```bash
bash scripts/active/run_thor_moe_runtime_gate_nohup.sh \
  models/moe_v1/Qwen3-30B-A3B-Q4_K_M.gguf
```

The runtime gate compares all-GPU, all-CPU-MoE, and partial-CPU-MoE placement without changing DVFS state.
