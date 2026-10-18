#!/usr/bin/env python3
"""Run a non-root Thor MoE model/placement runtime gate.

This gate intentionally does not change hardware frequencies or claim energy
results. It verifies that a supplied GGUF model loads under the pinned Thor
llama.cpp build, that all declared placement arguments are accepted, and that
fixed output requests complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def post(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def prompt_tokens(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [100 + rng.randrange(100000) for _ in range(length)]


def wait_health(url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if post(url + "/health", None, 1).get("status") == "ok":
                return
        except (OSError, ValueError, urllib.error.URLError):
            pass
        time.sleep(0.2)
    raise RuntimeError(f"server health timeout: {url}")


def start_server(
    binary: Path,
    model: Path,
    runtime: dict[str, Any],
    placement: dict[str, Any],
    port: int,
    log_path: Path,
) -> tuple[subprocess.Popen[str], str, list[str], Any]:
    command = [
        str(binary), "-m", str(model), "-c", str(runtime["context_tokens"]),
        "-b", str(runtime["batch"]), "-ub", str(runtime["ubatch"]),
        "-t", str(runtime["threads"]), "-tb", str(runtime["threads"]),
        "-ngl", "999", "-np", "1", "--cont-batching", "--no-cache-prompt",
        "--host", str(runtime["host"]), "--port", str(port),
        "--reasoning-format", "none", "--skip-chat-parsing",
        *[str(value) for value in placement["extra_args"]],
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w")
    process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    url = f"http://{runtime['host']}:{port}"
    try:
        wait_health(url, float(runtime["startup_timeout_s"]))
    except BaseException:
        process.terminate()
        process.wait(timeout=20)
        handle.close()
        raise
    return process, url, command, handle


def stop_server(process: subprocess.Popen[str], handle: Any) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    handle.close()


def run_request(url: str, workload: dict[str, Any], seed: int, timeout: float) -> dict[str, Any]:
    started = time.monotonic()
    payload = {
        "prompt": prompt_tokens(int(workload["prompt_tokens"]), seed),
        "n_predict": int(workload["output_tokens"]),
        "temperature": 0,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": False,
    }
    try:
        response = post(url + "/completion", payload, timeout)
        timings = response.get("timings") or {}
        predicted = int(response.get("tokens_predicted", -1))
        exact = predicted == int(workload["output_tokens"])
        return {
            "prompt": workload["name"],
            "prompt_tokens": workload["prompt_tokens"],
            "requested_output_tokens": workload["output_tokens"],
            "output_tokens": predicted,
            "completion_exact": exact,
            "ttft_ms": timings.get("prompt_ms"),
            "tpot_ms": timings.get("predicted_ms", 0) / max(1, predicted - 1),
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "error": None,
        }
    except BaseException as exc:
        return {
            "prompt": workload["name"],
            "prompt_tokens": workload["prompt_tokens"],
            "requested_output_tokens": workload["output_tokens"],
            "output_tokens": 0,
            "completion_exact": False,
            "ttft_ms": None,
            "tpot_ms": None,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "error": repr(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="External GGUF path; it is not copied into the repository")
    parser.add_argument("--config", default="configs/experiments/thor_moe_runtime_gate_v1.json")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    config_path = resolve(args.config).resolve()
    config = load(config_path)
    model = Path(args.model).expanduser().resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    runtime = config["runtime"]
    binary = resolve(runtime["server_binary"]).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    manifest = resolve(runtime["manifest"])
    if manifest.is_file():
        persisted = load(manifest)
        if persisted.get("base_commit") != runtime["base_commit"]:
            raise RuntimeError("Pinned runtime base commit mismatch")
    run_dir = resolve(args.run_dir or config["output_dir"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for placement_index, placement in enumerate(config["placements"]):
        port = int(runtime["port_base"]) + placement_index
        log_path = run_dir / "server_logs" / f"server_{placement['name']}.log"
        process, url, command, handle = start_server(binary, model, runtime, placement, port, log_path)
        try:
            warmup = run_request(url, {"name": "warmup", "prompt_tokens": 128, "output_tokens": 16}, 17, float(runtime["request_timeout_s"]))
            if not warmup["completion_exact"]:
                raise RuntimeError(f"warmup failed for {placement['name']}: {warmup}")
            for repeat in range(int(config["repeats"])):
                for workload_index, workload in enumerate(config["workloads"]):
                    result = run_request(
                        url, workload,
                        int(config["seed"]) + repeat * 1009 + workload_index,
                        float(runtime["request_timeout_s"]),
                    )
                    row = {
                        "placement": placement["name"],
                        "placement_label": placement["label"],
                        "repeat": repeat,
                        "model": str(model),
                        "model_sha256": file_hash(model),
                        "server_command": command,
                        **result,
                    }
                    rows.append(row)
                    print(json.dumps({"event": "case", **row}, sort_keys=True), flush=True)
        finally:
            stop_server(process, handle)
    fields = [key for key in rows[0] if key != "server_command"]
    with (run_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    (run_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest = {
        "schema": "energyinfra-thor-moe-runtime-gate-result-v1",
        "config": str(config_path),
        "model": str(model),
        "model_sha256": file_hash(model),
        "binary": str(binary),
        "binary_sha256": file_hash(binary),
        "placements": [item["name"] for item in config["placements"]],
        "rows": len(rows),
        "hardware_frequency_changes": False,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": len(rows), "run_dir": str(run_dir)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
