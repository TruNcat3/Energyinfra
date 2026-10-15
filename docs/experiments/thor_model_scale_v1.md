# Thor 7B/14B model-scale pilot

This is a model-scale extension to the CPU-floor experiment. It is not an
unchanged reproduction of the original Orin controller, an exhaustive search,
or evidence that a larger model necessarily gives better energy savings.

## Model and controls

- Keep the existing full-vocabulary random Qwen2.5-7B GGUF unchanged.
- Build Qwen2.5-14B with 48 layers, hidden size 5120, FFN size 13824,
  40 attention heads, 8 KV heads, and 152064 vocabulary entries.
- Use independent input/output embeddings, Q/K/V biases, RMSNorm and RoPE
  matching the dense architecture. Both models use Q4_K_M.
- Copy tokenizer metadata from the existing 7B GGUF, not a smaller tokenizer.
- Generate Gaussian random matrices, unit normalization weights and zero
  biases. Per-tensor seeds are deterministic; random weights are not pretrained.
- Keep the same inference engine, fixed token-count policy, context settings,
  measurement rail, and SLO scenarios through the existing CPU-floor runner.
- The architectures differ in depth, width and attention geometry. This is a
  same-family scale comparison, not a claim that parameter count alone causes
  any observed difference. It does not measure answer quality or battery life.

Official architecture source (snapshot recorded on 2026-09-21):
https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/raw/main/config.json

## Preparation

Preparation needs the existing `.venv-thor` packages `numpy`, `gguf` and
`llama_cpp`. It calls the installed llama.cpp quantization C API, so a separate
`llama-quantize` executable and network downloads are not needed.

Budget at least approximately 45 GiB of free space on the output filesystem.
The script checks a conservative F16-plus-quantized-output requirement before
generation. It keeps at most one full tensor in memory, with small generation
chunks; it does not materialize the entire F16 model in RAM.

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_model_scale_nohup.sh prepare
```

This only prepares the model. Wait for its log to report the completed model
before launching quick. No hardware frequency controls are written and no
inference benchmark is run by preparation.

Output:

```text
models/random_scale_v1/Qwen2.5-14B-random-v152064-Q4_K_M.gguf
models/random_scale_v1/Qwen2.5-14B-random-v152064-Q4_K_M.json
models/random_scale_v1/prepare_progress.json
```

The manifest records architecture, parameter count, reference-model hash,
tokenizer metadata hash, builder hash, quantizer hash/version and output hash.
After successful quantization, only the F16 intermediate created by this
builder is removed. To retain it, run the preparation Python entry point with
`--keep-f16` instead. Existing 7B artifacts are never modified.

Preparation logs every tensor and the quantizer output. On restart, a committed
F16 intermediate can be reused; incomplete `.partial` outputs are regenerated.
Conflicting provenance and uncommitted final outputs are rejected rather than
silently overwritten. Generation itself resumes at the F16 stage, not midway
through a tensor. Successful preparation does not imply inference is tested.

## 14B quick

Only after preparation has finished:

```bash
sudo -E bash scripts/active/run_thor_model_scale_nohup.sh quick
```

Six configurations at input/output 512/128, one profiling repeat, followed by
one repeat of independent validation for selected candidates and baselines.
This checks execution and catches gross SLO/model-loading problems. A single
repeat is not a stable performance or SLO-attainment claim.

## Matched 7B and 14B pilot

After inspecting quick results:

```bash
sudo -E bash scripts/active/run_thor_model_scale_nohup.sh pilot
```

The launcher runs 7B and then 14B, never concurrently. Each model uses:

- 12 CAP combinations: GPU 1116/1485 MHz, EMC 2750/4266 MHz, CPU floor
  minimum-supported/486/972 MHz, with CPU ceiling fixed at 1458 MHz.
- One locked reference: GPU 1485 MHz, CPU 1458 MHz, EMC cap 4266 MHz.
- Native Dynamic and MaxClock references.
- Three workloads: input/output 64/64, 512/512 and 2048/128.
- Three profiling repeats: 15 x 3 x 3 = 135 static blocks per model,
  270 across both models, excluding warmups and independent validation.
- Five independent validation repeats for unique selected workload/point
  pairs and their baselines. Validation is exploratory, not the earlier
  20-repeat confirmatory campaign.

Within each model, all declared pilot points are evaluated. Selection remains
per workload and per SLO: full feasible Pareto sets are retained, with minimum
power and minimum J/token anchors frozen before independent validation.

TPOT limits are 50/75/100 ms. TTFT scenarios are unconstrained, 1 s and 2 s.
Infeasibility remains visible; there is no implicit fallback or SLO relaxation.
Compare savings against that model's own Dynamic and MaxClock baselines at
the same SLO. Do not compare 7B at 50 ms against 14B at 100 ms and call the
difference a same-service optimization gain.

Reports are separate:

```text
data/protocol_v2/thor_model_scale_v1/7b_pilot/RESULTS.md
data/protocol_v2/thor_model_scale_v1/14b_pilot/RESULTS.md
```

Use `static_table.csv` for matched-frequency sensitivity and `comparison.csv`
for SLO-constrained, independently measured candidates. Record TTFT, TPOT,
power, J/token and observed SLO attainment, including losing or infeasible
points. Existing within-model intervals are not corrected across both models;
do not present a cross-model statistical significance claim from this pilot.
Model order is fixed 7B then 14B, so confirm any model-scale conclusion with
counterbalanced runs before treating it as causal.

All stages print a log path, supervisor PID and `sudo kill -TERM PID` command.
During pilot, TERM is forwarded to the active runner and the launcher waits
for its hardware restoration. Do not use SIGKILL for ordinary stopping.
Rerun the same stage to resume committed work; use `--run-root PATH` for a new
campaign without mixing previous results. This option changes result paths,
not the prepared model's configured output path.

No full 12-workload sweep, serving experiment or automatic change of SLO is
started here. First determine whether model scale changes the frequency
sensitivity and feasible frontier enough to justify a larger campaign.

## Current status and interpretation

The 14B artifact being prepared is useful for system-scale sensitivity, but both 7B and 14B use
synthetic random weights. The pilot can establish whether model shape changes power, J/token or
frequency sensitivity under a matched protocol; it cannot establish quality, battery life or a
universal scale law. A larger model result must be reported against its own Dynamic and MaxClock
baselines and its own SLO feasibility, never as a direct win over a smaller model.

## Archived pilot interpretation

The 5-block 7B/14B pilot completed for the declared workloads. For 64/64,
the 7B dynamic point was about 75.5 W and 2.76 J/token, while the 14B
dynamic point was about 81.8 W and 5.13 J/token. Low-power caps reduced
instantaneous power for both models but increased J/token relative to their
own high-performance references. This supports scale-sensitive profiling,
not a universal claim that larger models create more DVFS savings.
