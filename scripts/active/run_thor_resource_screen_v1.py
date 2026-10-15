#!/usr/bin/env python3
"""Screen real llama.cpp parallel decode and prompt microbatch resource surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_joint_resource_power_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def emit(**values: Any) -> None:
    print(json.dumps(values, sort_keys=True), flush=True)


def job_id(job: dict[str, Any]) -> str:
    encoded = json.dumps(job, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_inputs(config_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    config = load_json(config_path)
    runtime = config["runtime"]
    bench = resolve(runtime["batched_bench_binary"])
    model = resolve(config["model"]["path"])
    manifest = resolve(config["output_dir"]) / "runtime_manifest.json"
    for path in (bench, model, manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    recorded = load_json(manifest)
    if recorded["actual_commit"] != runtime["commit"] or recorded["batched_bench"]["sha256"] != digest(bench):
        raise RuntimeError("Runtime manifest does not match the configured batched benchmark")
    return config, bench, model, manifest


def make_jobs(config: dict[str, Any], preset: str) -> list[dict[str, Any]]:
    pilot = config["pilot"]
    utilization = pilot["utilization_surface"]
    workloads = pilot["workloads"]
    repeats = int(pilot["repeats"])
    npl = [int(value) for value in utilization["parallel_slots"]]
    if npl != [int(value) for value in utilization["closed_loop_concurrency"]]:
        raise ValueError("Stage 1A requires parallel slots to equal closed-loop concurrency")
    if preset == "quick":
        workloads = [[128, 128]]
        repeats = 1
        npl = [1, 2, 4]
    jobs: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for workload in workloads:
            for parallel in npl:
                jobs.append({
                    "surface": "utilization",
                    "workload": [int(workload[0]), int(workload[1])],
                    "n_batch": int(utilization["n_batch"]),
                    "n_ubatch": int(utilization["n_ubatch"]),
                    "npl": [parallel],
                    "repeat": repeat,
                })
    micro = pilot["prefill_microbatch_surface"]
    micro_workloads = micro["workloads"]
    batches = [int(value) for value in micro["n_batch"]]
    ubatches = [int(value) for value in micro["n_ubatch"]]
    if preset == "quick":
        micro_workloads = [[128, 128]]
        batches = [256, 512]
        ubatches = [128, 256]
    for repeat in range(repeats):
        for workload in micro_workloads:
            for batch in batches:
                for ubatch in ubatches:
                    if ubatch > batch:
                        continue
                    jobs.append({
                        "surface": "microbatch",
                        "workload": [int(workload[0]), int(workload[1])],
                        "n_batch": batch,
                        "n_ubatch": ubatch,
                        "npl": [int(micro["parallel_slots"])],
                        "repeat": repeat,
                    })
    random.Random(20261004).shuffle(jobs)
    return jobs


def protocol(config_path: Path, config: dict[str, Any], bench: Path, model: Path, manifest: Path, preset: str) -> dict[str, Any]:
    model_stat = model.stat()
    return {
        "schema": "energyinfra-thor-resource-screen-protocol-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preset": preset,
        "config": str(config_path),
        "config_sha256": digest(config_path),
        "runtime_manifest": str(manifest),
        "runtime_manifest_sha256": digest(manifest),
        "batched_bench": str(bench),
        "batched_bench_sha256": digest(bench),
        "model": {"path": str(model), "size_bytes": model_stat.st_size, "mtime_ns": model_stat.st_mtime_ns},
        "scope": "llama_decode resource microbenchmark; model load excluded from reported internal timings",
        "energy_scope": "none in Stage 1A; process energy would include model loading and is intentionally not reported",
        "slo_scope": "prefill/decode timing proxies only; not request-level TTFT/TPOT",
    }


def validate_protocol(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        atomic(path, expected)
        return
    existing = load_json(path)
    immutable = ["preset", "config_sha256", "runtime_manifest_sha256", "batched_bench_sha256", "model"]
    if any(existing.get(key) != expected.get(key) for key in immutable):
        raise RuntimeError("Existing screen protocol differs; use a new --run-dir")


def existing_ids(raw_dir: Path) -> set[str]:
    return {path.stem for path in raw_dir.glob("*.json")}


def execute_job(
    bench: Path,
    model: Path,
    context: int,
    timeout_s: int,
    job: dict[str, Any],
    raw_dir: Path,
    log_dir: Path,
) -> int:
    identifier = job_id(job)
    prompt, generation = job["workload"]
    command = [
        str(bench), "-m", str(model), "-c", str(context),
        "-b", str(job["n_batch"]), "-ub", str(job["n_ubatch"]),
        "-t", "4", "-tb", "4", "-ngl", "999",
        "-npp", str(prompt), "-ntg", str(generation),
        "-npl", ",".join(str(value) for value in job["npl"]),
        "--output-format", "jsonl",
    ]
    started = time.monotonic()
    environment = dict(os.environ)
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{identifier}.log").write_text(output)
        raise RuntimeError(
            f"batched-bench timed out after {timeout_s}s for {identifier}; "
            f"see {log_dir / (identifier + '.log')}"
        ) from error
    elapsed = time.monotonic() - started
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{identifier}.log").write_text(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"batched-bench failed for {identifier}; see {log_dir / (identifier + '.log')}")
    parsed = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if all(key in value for key in ("pp", "tg", "pl", "t_pp", "t_tg")):
            parsed.append(value)
    expected = len(job["npl"])
    if len(parsed) != expected:
        raise RuntimeError(f"Expected {expected} benchmark rows, parsed {len(parsed)} for {identifier}")
    rows = []
    for value in parsed:
        row = {
            "schema": "energyinfra-thor-resource-screen-row-v1",
            "job_id": identifier,
            "surface": job["surface"],
            "repeat": job["repeat"],
            "workload": job["workload"],
            "n_batch": int(value["n_batch"]),
            "n_ubatch": int(value["n_ubatch"]),
            "parallel_sequences": int(value["pl"]),
            "prompt_tokens_per_sequence": int(value["pp"]),
            "generated_tokens_per_sequence": int(value["tg"]),
            "prefill_time_s": float(value["t_pp"]),
            "aggregate_prefill_tps": float(value["speed_pp"]),
            "decode_time_s": float(value["t_tg"]),
            "aggregate_decode_tps": float(value["speed_tg"]),
            "per_sequence_decode_tps": float(value["tg"]) / float(value["t_tg"]),
            "decode_iteration_ms": float(value["t_tg"]) * 1000.0 / float(value["tg"]),
            "total_benchmark_time_s": float(value["t"]),
            "combined_token_tps": float(value["speed"]),
            "flash_attention_mode": int(value["flash_attn"]),
            "process_wall_time_s_including_model_load": elapsed,
        }
        rows.append(row)
    atomic(raw_dir / f"{identifier}.json", {"job": job, "command": command, "rows": rows})
    return len(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def aggregate(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(raw_dir.glob("*.json")):
        rows.extend(load_json(path)["rows"])
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["surface"], tuple(row["workload"]), row["n_batch"],
            row["n_ubatch"], row["parallel_sequences"],
        )
        grouped[key].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        surface, workload, batch, ubatch, parallel = key
        decode_tps = [value["aggregate_decode_tps"] for value in values]
        prefill_tps = [value["aggregate_prefill_tps"] for value in values]
        iteration = [value["decode_iteration_ms"] for value in values]
        output.append({
            "surface": surface,
            "workload": list(workload),
            "n_batch": batch,
            "n_ubatch": ubatch,
            "parallel_sequences": parallel,
            "repeats": len(values),
            "aggregate_prefill_tps_mean": mean(prefill_tps),
            "aggregate_prefill_tps_stdev": statistics.stdev(prefill_tps) if len(prefill_tps) > 1 else None,
            "aggregate_decode_tps_mean": mean(decode_tps),
            "aggregate_decode_tps_stdev": statistics.stdev(decode_tps) if len(decode_tps) > 1 else None,
            "per_sequence_decode_tps_mean": mean([value["per_sequence_decode_tps"] for value in values]),
            "decode_iteration_ms_mean": mean(iteration),
            "decode_iteration_ms_pessimistic": max(iteration),
            "prefill_time_ms_mean": mean([value["prefill_time_s"] for value in values]) * 1000.0,
        })
    baselines = {
        tuple(row["workload"]): row
        for row in output
        if row["surface"] == "utilization" and row["parallel_sequences"] == 1
    }
    for row in output:
        baseline = baselines.get(tuple(row["workload"]))
        if baseline is None or row["surface"] != "utilization":
            row["decode_throughput_scaling_vs_parallel1"] = None
            continue
        row["decode_throughput_scaling_vs_parallel1"] = (
            row["aggregate_decode_tps_mean"] / baseline["aggregate_decode_tps_mean"]
        )
    return output


def render(rows: list[dict[str, Any]], completed: int, expected: int) -> str:
    lines = [
        "# Thor joint resource-power Stage 1A resource screen",
        "",
        f"Status: **{completed}/{expected} process jobs**.",
        "",
        "This is a `llama_decode` resource microbenchmark. Internal PP/TG timings exclude model loading. "
        "No energy or request-level SLO claim is made at this stage.",
        "",
        "## Parallel utilization surface",
        "",
        "| Workload | Parallel | Batch/ubatch | PP tok/s | Decode tok/s | Scaling | Per-seq tok/s | Decode iteration ms | Repeats |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["surface"] != "utilization":
            continue
        scaling = row["decode_throughput_scaling_vs_parallel1"]
        lines.append(
            f"| {row['workload'][0]}/{row['workload'][1]} | {row['parallel_sequences']} | "
            f"{row['n_batch']}/{row['n_ubatch']} | {row['aggregate_prefill_tps_mean']:.1f} | "
            f"{row['aggregate_decode_tps_mean']:.2f} | "
            f"{'n/a' if scaling is None else f'{scaling:.2f}x'} | "
            f"{row['per_sequence_decode_tps_mean']:.2f} | {row['decode_iteration_ms_mean']:.2f} | {row['repeats']} |"
        )
    lines += [
        "",
        "## Prompt batch/microbatch surface at parallel=4",
        "",
        "| Workload | Batch | Ubatch | PP tok/s | Decode tok/s | Per-seq tok/s | Decode iteration ms | Repeats |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["surface"] != "microbatch":
            continue
        lines.append(
            f"| {row['workload'][0]}/{row['workload'][1]} | {row['n_batch']} | {row['n_ubatch']} | "
            f"{row['aggregate_prefill_tps_mean']:.1f} | {row['aggregate_decode_tps_mean']:.2f} | "
            f"{row['per_sequence_decode_tps_mean']:.2f} | {row['decode_iteration_ms_mean']:.2f} | {row['repeats']} |"
        )
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Aggregate decode tok/s measures batching/resource efficiency.",
        "- Per-sequence tok/s and decode-iteration latency expose the latency cost of batching.",
        "- PP timing is a prefill kernel proxy, not end-to-end TTFT.",
        "- Model load, HTTP queueing, sampling, and VIN energy are intentionally deferred to persistent-server validation.",
        "",
    ]
    return "\n".join(lines)


def report(run_dir: Path, expected: int) -> None:
    raw_dir = run_dir / "raw"
    rows = aggregate(raw_dir)
    completed = len(existing_ids(raw_dir))
    atomic(run_dir / "summary.json", {"completed_jobs": completed, "expected_jobs": expected, "rows": rows})
    (run_dir / "RESULTS.md").write_text(render(rows, completed, expected))
    emit(event="report", completed_jobs=completed, expected_jobs=expected, rows=len(rows), output=str(run_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "report"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir")
    parser.add_argument("--preset", choices=["quick", "full"], default="full")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config, bench, model, manifest = load_inputs(config_path)
    suffix = "resource_screen_quick" if args.preset == "quick" else "resource_screen"
    run_dir = resolve(args.run_dir or (Path(config["output_dir"]) / suffix))
    jobs = make_jobs(config, args.preset)
    if args.stage == "report":
        report(run_dir, len(jobs))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    validate_protocol(run_dir / "protocol.json", protocol(config_path, config, bench, model, manifest, args.preset))
    raw_dir = run_dir / "raw"
    log_dir = run_dir / "process_logs"
    raw_dir.mkdir(exist_ok=True)
    done = existing_ids(raw_dir)
    pending = [job for job in jobs if job_id(job) not in done]
    per_slot_context = int(config["pilot"]["per_slot_context_tokens"])
    timeout_s = int(config["pilot"]["job_timeout_s"])
    contexts = sorted({per_slot_context * max(job["npl"]) for job in jobs})
    emit(
        event="start",
        preset=args.preset,
        completed=len(done),
        pending=len(pending),
        context_tokens=contexts,
        job_timeout_s=timeout_s,
    )
    produced = 0
    for index, job in enumerate(pending, 1):
        context = per_slot_context * max(job["npl"])
        count = execute_job(bench, model, context, timeout_s, job, raw_dir, log_dir)
        produced += count
        atomic(run_dir / "progress.json", {
            "completed_jobs": len(done) + index,
            "expected_jobs": len(jobs),
            "last_job": job,
        })
        emit(
            event="job",
            progress=f"{index}/{len(pending)}",
            surface=job["surface"],
            workload=job["workload"],
            batch=job["n_batch"],
            ubatch=job["n_ubatch"],
            parallel=job["npl"],
            rows=count,
        )
    report(run_dir, len(jobs))
    emit(event="complete", new_rows=produced)


if __name__ == "__main__":
    main()
