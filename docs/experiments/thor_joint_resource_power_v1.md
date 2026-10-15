# Thor joint resource-power optimization v1

## Motivation

The completed Thor experiments show that a DVFS-only policy is close to its local optimum. This
experiment expands the decision space to real continuous batching, parallel slots, prompt-evaluation
batching, microbatching, and later hardware power states.

The installed `llama-cpp-python 0.3.35` server cannot provide true continuous batching for one model:
its application layer serializes model access with global locks. `n_batch` and `n_ubatch` only control
logical and physical prompt-evaluation batches. Using concurrent HTTP clients against that server
would therefore measure queueing, not batching.

The new runtime is the official CUDA `llama-server`, pinned to llama.cpp commit
`4df29be4f4c3673f428170fda944a5b19f743bb8`, the `vendor/llama.cpp` submodule
recorded by the official `llama-cpp-python v0.3.35` tag. It is built in separate source and build
directories and does not replace the working Python environment.

The previously prepared `ce8784bdb153` runtime is incompatible: on Thor it crashes in the
single-sequence CUDA decode path, and its ABI does not match the installed Python package. Its Stage 1A
results must not be combined with this experiment.

## Stage 0: prepare the runtime

Run without sudo:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_llama_server_prepare_nohup.sh
```

The launcher prints a persistent log and PID. The build uses four jobs by default. To reduce or
increase build parallelism:

```bash
LLAMA_BUILD_JOBS=2 bash scripts/active/run_thor_llama_server_prepare_nohup.sh
```

Preparation performs no model inference and no hardware-frequency changes. It produces:

- `build/llama.cpp-4df29be4f4c3-thor-make/bin/llama-server`
- `build/llama.cpp-4df29be4f4c3-thor-make/bin/llama-batched-bench`
- `data/protocol_v2/thor_joint_resource_power_v1_4df29be/runtime_manifest.json`

The manifest records source commit, binary hashes, CLI-help hash, CUDA compiler, CMake, architecture,
kernel, and configuration hash.

## Planned Stage 1: utilization surface

The first hardware experiment keeps the QueuePowerMin hardware state fixed and varies only:

- true parallel slots and closed-loop concurrency: `1, 2, 4, 8`;
- prompt logical batch and physical microbatch;
- four prompt/output shapes;
- fixed completed output tokens and identical request sets.

It will report total VIN energy, joules per completed output token, requests/s, tokens/s, TTFT, TPOT,
joint SLO attainment, and per-request queue delay. Sequential requests will not be labeled batching.

Stage 1 is split into two gates. Stage 1A uses `llama-batched-bench` internal timings to screen real
parallel decode and prompt microbatch configurations without reporting energy. Stage 1B will use a
persistent `llama-server`, so model-loading energy is excluded and HTTP queue/SLO timing is observable.

Quick Stage 1A screen:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_resource_screen_nohup.sh quick
```

Full Stage 1A screen:

```bash
bash scripts/active/run_thor_resource_screen_nohup.sh full
```

These commands run without sudo and do not change hardware frequencies. Results are resumable and are
written to:

- `data/protocol_v2/thor_joint_resource_power_v1_4df29be/resource_screen_quick/`
- `data/protocol_v2/thor_joint_resource_power_v1_4df29be/resource_screen/`

## Planned Stage 2: joint Pareto search

Only resource configurations on the Stage 1 frontier will be crossed with four representative hardware
states: QueuePowerMin, lock1485, race-to-idle 1575, and EMC3200 power-cap. This factorized design avoids
an unnecessarily large Cartesian sweep while preserving the important interactions.

Random model weights remain suitable for systems measurements but not quality, speculative acceptance,
or application-level utility. Speculative decoding is intentionally excluded from v1.
