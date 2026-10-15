# Thor serving SLO profiles v1

## Why TPOT alone is insufficient

Current serving tools report client-observed TTFT, TPOT, ITL, end-to-end request
latency, throughput and goodput. Goodput counts requests that meet all configured
SLOs rather than treating decode speed as the only gate. Tail percentiles and
completion are also required to expose queueing failures.

The audit uses four declared threshold profiles:

| Profile | TTFT | TPOT | Intended interpretation |
|---|---:|---:|---|
| strict_interactive_8b | 500 ms | 30 ms | MLPerf Llama3.1-8B interactive-inspired |
| thor_edge_balanced | 2000 ms | 50 ms | Thor-specific middle point |
| conversational_8b | 2000 ms | 100 ms | MLPerf Llama3.1-8B conversational-inspired |
| capacity_reference | 3000 ms | 100 ms | vLLM adaptive capacity-search defaults |

For each request, the E2E deadline is `TTFT_limit + TPOT_limit * (requested_output_tokens - 1)`.
The audit reports joint attainment, per-repeat 95% qualification, p95/p99 tails,
Wilson intervals, completion, power, energy per token and fixed-work total energy.

These profiles are threshold-inspired analyses, not MLPerf-compliant runs. MLPerf
uses its own LoadGen, workload, quality and run-validity rules. The random-weight
model in this campaign cannot satisfy a quality benchmark.

## Primary sources

- MLCommons inference rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- vLLM benchmark metrics and goodput: https://github.com/vllm-project/vllm-bench/blob/main/README.md
- NVIDIA GenAI-Perf metrics: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html
- ElastiLM on-device SLO study: https://xumengwei.github.io/files/MobiCom25-ElastiLM.pdf
- CLONE edge LLM study: https://www.usenix.org/system/files/atc25-tian.pdf

## Run

This is an offline audit and makes no hardware changes:

```bash
cd /home/wt/energyinfra/Energyinfra-main
.venv-thor/bin/python scripts/active/analyze_thor_slo_profiles_v1.py
```

The report is written to `reports/thor_slo_audit_v1/RESULTS.md`.
