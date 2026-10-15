#!/usr/bin/env python3
"""Freeze the original EnergyInfra bursty traces without load transformation."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from src.experiments.run_serving_benchmark import RequestGenerator


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_q4_bursty_fixed_trace_v1.json"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def digest(value):
    payload = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = resolve(args.config)
    experiment = json.loads(config_path.read_text())
    static_path = resolve(experiment["static_config"])
    settings = json.loads(static_path.read_text())
    diagnostic_path = resolve(settings["diagnostic_config"])
    diagnostic = json.loads(diagnostic_path.read_text())
    base_path = resolve(diagnostic["base_config"])
    base = json.loads(base_path.read_text())
    output = resolve(experiment["calibration_trace_dir"])
    output.mkdir(parents=True, exist_ok=True)
    traces = []
    statistics = []
    for repeat in range(experiment["trace_repeats"]):
        seed = base["seed"] + repeat
        trace = RequestGenerator(experiment["trace"], experiment["duration_s"], seed).generate_requests()
        if not trace or trace[-1]["scheduled_time"] > experiment["duration_s"]:
            raise RuntimeError("Original RequestGenerator produced an invalid trace")
        traces.append(trace)
        burst = [request for request in trace if request["is_burst"]]
        statistics.append({
            "repeat": repeat,
            "seed": seed,
            "requests": len(trace),
            "normal_requests": len(trace) - len(burst),
            "burst_requests": len(burst),
            "first_burst_s": burst[0]["scheduled_time"] if burst else None,
            "last_request_s": trace[-1]["scheduled_time"],
        })
    protocol = {
        "schema": "energyinfra-original-request-generator-trace-v1",
        "trace": experiment["trace"],
        "duration_s": experiment["duration_s"],
        "repeats": experiment["trace_repeats"],
        "trace_hashes": [digest(trace) for trace in traces],
        "statistics": statistics,
        "sources": {
            str(Path(__file__).resolve().relative_to(ROOT)): file_hash(Path(__file__).resolve()),
            str(config_path.relative_to(ROOT)): file_hash(config_path),
            "src/experiments/run_serving_benchmark.py": file_hash(ROOT / "src/experiments/run_serving_benchmark.py"),
            str(static_path.relative_to(ROOT)): file_hash(static_path),
            str(base_path.relative_to(ROOT)): file_hash(base_path),
        },
        "transformation": "none",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        existing = json.loads(protocol_path.read_text())
        same_identity = all(existing.get(key) == protocol[key]
                            for key in ("schema", "trace", "duration_s", "repeats", "statistics"))
        same_traces = all((output / f"trace_{repeat}.json").exists()
                          and json.loads((output / f"trace_{repeat}.json").read_text()) == trace
                          for repeat, trace in enumerate(traces))
        if not same_identity or not same_traces:
            raise RuntimeError("Frozen bursty trace provenance changed; use a new output directory")
    atomic(protocol_path, protocol)
    for repeat, trace in enumerate(traces):
        path = output / f"trace_{repeat}.json"
        if path.exists() and json.loads(path.read_text()) != trace:
            raise RuntimeError(f"Frozen trace changed: {path}")
        atomic(path, trace)
    lines = [
        "# Original bursty_mixed frozen traces",
        "",
        "No request was removed, shifted or rescaled.",
        "",
        "| Repeat | Seed | Requests | Normal | Burst | First burst s | Last request s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in statistics:
        first = "--" if row["first_burst_s"] is None else f"{row['first_burst_s']:.3f}"
        lines.append(f"| {row['repeat']} | {row['seed']} | {row['requests']} | "
                     f"{row['normal_requests']} | {row['burst_requests']} | {first} | "
                     f"{row['last_request_s']:.3f} |")
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"trace_protocol": str(protocol_path), "trace_hashes": protocol["trace_hashes"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
