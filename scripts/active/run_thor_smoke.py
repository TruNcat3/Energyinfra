#!/usr/bin/env python3
"""Run a small end-to-end llama.cpp and energy-collection smoke test on Thor."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.controller.cap_controller import CapController
from src.metrics.metrics_collector import MetricsCollector


def run_once(
    runner: LlamaCppRunner,
    prompt_length: int,
    output_length: int,
    log_dir: Path,
    interval_ms: int,
) -> dict:
    collector = MetricsCollector(output_dir=str(log_dir), interval_ms=interval_ms)
    collector.start_collection(tag="thor_smoke")
    started = time.time()
    try:
        result = runner.run_single_inference(
            prompt_length=prompt_length,
            output_length=output_length,
            benchmark_mode=True,
        )
    finally:
        elapsed = time.time() - started
        time.sleep(max(interval_ms / 1000.0, 0.1))
        collector.stop_collection()

    energy = collector.compute_energy()
    result = dict(result or {})
    result.update(
        {
            "wall_time_s": elapsed,
            "total_energy_j": energy.get("total_energy_j", energy.get("energy_j", 0.0)),
            "avg_power_w": energy.get("avg_power_w", 0.0),
            "max_power_w": energy.get("max_power_w", 0.0),
            "temperature_c": energy.get("temperature_c", energy.get("avg_temp_tj_c")),
            "power_samples": len(collector.get_samples_df()),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/gguf/EnergyInfra-random-Qwen2-Q4_K_M.gguf",
    )
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--n-ctx", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--interval-ms", type=int, default=500)
    parser.add_argument("--apply-cap", action="store_true")
    parser.add_argument("--gpu-cap", type=int, default=1020)
    parser.add_argument("--emc-cap", type=int, default=2750)
    parser.add_argument("--cpu-cap", type=int, default=1497)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"model not found: {args.model}; run scripts/create_random_gguf.py first"
        )

    cap = CapController()
    print(json.dumps(cap.read_actual_state(), indent=2, sort_keys=True))
    if args.apply_cap:
        applied = cap.set_cap(
            gpu_cap_mhz=args.gpu_cap,
            emc_cap_mhz=args.emc_cap,
            cpu_cap_mhz=args.cpu_cap,
        )
        print(json.dumps({"cap_result": applied}, indent=2, sort_keys=True))
        if not applied.get("ok", False):
            print(json.dumps({"restore_result": cap.restore_dynamic()}, indent=2, sort_keys=True))
            raise SystemExit("cap application failed; run this command with sudo")

    runner = LlamaCppRunner(
        model_path=str(args.model),
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )
    try:
        runner.run_single_inference(
            prompt_length=min(args.prompt_tokens, 16),
            output_length=4,
            benchmark_mode=True,
        )
        for index in range(args.repeats):
            result = run_once(
                runner,
                args.prompt_tokens,
                args.max_tokens,
                ROOT / "data/raw_logs",
                args.interval_ms,
            )
            print(json.dumps({"iteration": index + 1, **result}, default=str, sort_keys=True))
    finally:
        if args.apply_cap:
            print(json.dumps({"restore_result": cap.restore_dynamic()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
