# Thor precision pilot

This experiment isolates weight representation within the existing llama.cpp
runtime. It does not claim equivalence to TensorRT INT4 AWQ, FP8, or Thor
NVFP4 engines, which use different calibration, kernels, activation precision
and often KV-cache formats.

## Why these three formats

`Q4_K_M` is the current baseline and a common llama.cpp edge deployment
choice. `Q5_K_M` is an intermediate weight-only format. `Q8_0` is a higher
precision weight-only reference that helps identify how much of the observed
frequency sensitivity comes from memory traffic and quantized kernels.

NVIDIA edge material commonly uses INT4 AWQ for Jetson-class deployment, while
newer Thor TensorRT Edge-LLM material discusses INT4/NVFP4 weights and FP8 KV
cache. Those are relevant follow-up paths, but mixing their results with this
llama.cpp table would conflate precision with runtime and kernel changes.

References:

- NVIDIA Jetson INT4 AWQ example: https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1/
- TensorRT-LLM INT4 PTQ/AWQ workflow: https://developer.nvidia.com/blog/accelerating-hebrew-llm-performance-with-nvidia-tensorrt-llm/
- Thor TensorRT Edge-LLM reduced precision overview: https://developer.nvidia.com/blog/accelerate-ai-inference-for-edge-and-robotics-with-nvidia-jetson-t4000-and-jetpack-7-1/
- llama.cpp tensor encodings: https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes

## Model caveat

The 7B model is full-shape Qwen2.5 with 152064 vocabulary entries, but its
weights are synthetic random weights. Quantization quality, perplexity,
instruction accuracy and task quality are therefore not measured. This pilot
measures model loading, latency, power, energy/token and frequency sensitivity
only. A real model quality study requires a pretrained checkpoint and a fixed
evaluation set.

The F16 source is generated once with the same deterministic per-tensor random
recipe and then quantized to Q5_K_M and Q8_0. The existing Q4_K_M paired model
is kept as the Q4 baseline. The F16 source is retained for reproducibility and
future formats; it is not loaded for inference in this pilot.

## Preparation

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_precision_nohup.sh prepare
```

The preparation stage creates approximately one 7B F16 source plus Q5_K_M and
Q8_0 outputs. It checks disk space, records hashes and never writes hardware
frequency controls. It does not delete the existing Q4 model.

## Quick comparison

After preparation finishes:

```bash
sudo -E bash scripts/active/run_thor_precision_nohup.sh quick
```

The launcher runs Q4_K_M, Q5_K_M and Q8_0 sequentially. Each uses the same
six hardware points, one 512/128 workload, one profiling repeat and one
independent validation repeat. All runs use the same CPU-floor, SLO and
tegrastats measurement logic. This is a feasibility probe, not a final
statistical comparison.

Reports:

```text
data/protocol_v2/thor_precision_v1/q4_k_m_quick/RESULTS.md
data/protocol_v2/thor_precision_v1/q5_k_m_quick/RESULTS.md
data/protocol_v2/thor_precision_v1/q8_0_quick/RESULTS.md
```

## Same-source gate

The first quick comparison keeps the historical Q4 model as a baseline. It is
useful for a smoke check but cannot establish a pure precision effect unless
all formats are derived from the same F16 source. The gate creates a same-source
Q4 model and repeats all three formats five times:

```bash
sudo -E bash scripts/active/run_thor_precision_nohup.sh prepare
sudo -E bash scripts/active/run_thor_precision_nohup.sh gate
```

Gate reports are written under
`data/protocol_v2/thor_precision_v1/{q4_k_m,q5_k_m,q8_0}_gate/`. They use the
same six hardware points and 512/128 workload, with five profile and five
validation repeats. Only after this gate is stable should the three-format
Pareto table be started.

Interpretation should use the same workload and SLO across formats. Report
TTFT, TPOT, average power, J/token and observed SLO attainment. A lower power
number is not an energy win if the lower precision or lower clock increases
runtime enough to raise J/token.

## Gate for a larger experiment

Only if Q4/Q5/Q8 show a material and repeatable difference should we run a
three-format pilot over 64/64, 512/512 and 2048/128 with five validation
repeats. The full frequency/SLO/serving expansion comes after that gate.

After the same-source gate passes, run the Pareto pilot:

```bash
sudo -E bash scripts/active/run_thor_precision_nohup.sh pareto
```

Each format uses 15 points, three workloads and three profiling repeats, for
135 static blocks per format and 405 total. Candidate selection is performed
independently per format, workload and SLO. Frozen selected candidates and
Dynamic/MaxClock baselines then receive 20 independent validation repeats.
Results are written under `data/protocol_v2/thor_precision_pareto_v1/`.

The next separate track, if required, is a pretrained-model quality plus
TensorRT Edge-LLM experiment using INT4 AWQ or NVFP4/FP8. It needs a different
runtime and calibration data, so it must receive separate configuration,
provenance and baseline tables.

## Current status and interpretation

The precision track is a sensitivity study inside llama.cpp, not a comparison of all edge deployment
formats. Q4/Q5/Q8 results can be compared only after the same-source gate, with the same workload,
hardware candidates, SLO and independent repeats. Lower precision is not assumed to be better: lower
memory traffic may reduce power while kernel or runtime behavior increases execution time and J/token.
Until the gate and validation table are complete, precision remains `partial` in the platform matrix.

## Archived pilot interpretation

The same-source gate and 20-block pilot completed for Q4_K_M, Q5_K_M and
Q8_0. At 64/64, the low-power Q4 candidate was about 63.8 W and 3.10
J/token, while MaxClock was about 93.6 W and 2.36 J/token; Q5 and Q8 show
the same direction. The low-power point therefore demonstrates a power
frontier, not an energy win. High-performance locked points reduce J/token
slightly relative to MaxClock, but do not provide a broad low-power solution.
These figures are limited to llama.cpp weight-only formats and synthetic
weights; they are not INT4 AWQ, NVFP4, FP8 or a quality comparison.
