#!/usr/bin/env python3
"""Freeze seven deterministic request-order variants for paired confirmation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_queue_aware_confirmation_v1.json"

import sys
sys.path.insert(0, str(ROOT))
from scripts.active import run_cpu_floor_v1 as floor


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def reordered_trace(source: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    arrivals = [float(item["scheduled_time"]) for item in source]
    requests = [copy.deepcopy(item) for item in source]
    block_size = 4
    blocks = [requests[index:index + block_size] for index in range(0, len(requests), block_size)]
    rng.shuffle(blocks)
    ordered = [item for block in blocks for item in block]
    if len(ordered) > 1:
        shift = rng.randrange(len(ordered))
        ordered = ordered[shift:] + ordered[:shift]
    result = []
    for request_id, (arrival, request) in enumerate(zip(arrivals, ordered)):
        request["scheduled_time"] = arrival
        request["request_id"] = request_id
        result.append(request)
    return result


def prepare(config_path: Path) -> Path:
    config = load(config_path)
    source_dir = resolve(config["source_trace_dir"])
    output_dir = resolve(config["confirmation_trace_dir"])
    source_protocol_path = source_dir / "protocol.json"
    source_protocol = load(source_protocol_path)
    sources = [load(source_dir / f"trace_{index}.json") for index in range(len(source_protocol["trace_hashes"]))]
    repeat_count = int(config["confirmation_repeats"])
    seed = int(config["trace_seed"])
    traces = [reordered_trace(sources[index % len(sources)], seed + index) for index in range(repeat_count)]
    trace_hashes = [floor.v2.digest(trace) for trace in traces]
    protocol = {
        "schema": "energyinfra-confirmation-traces-v1",
        "method": "deterministic block-shuffle of workload order with frozen source arrival times",
        "trace_seed": seed,
        "trace_hashes": trace_hashes,
        "source_protocol": str(source_protocol_path),
        "source_protocol_sha256": file_hash(source_protocol_path),
        "source_trace_sha256": {
            f"trace_{index}.json": file_hash(source_dir / f"trace_{index}.json")
            for index in range(len(sources))
        },
        "statistical_scope": "paired run noise and request-order sensitivity; not arbitrary production traffic",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists() and load(protocol_path) != protocol:
        raise RuntimeError("confirmation trace provenance changed; use a new output directory")
    for index, trace in enumerate(traces):
        atomic(output_dir / f"trace_{index}.json", trace)
    atomic(protocol_path, protocol)
    print(json.dumps({"protocol": str(protocol_path), "traces": len(traces), "trace_hashes": trace_hashes}))
    return protocol_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    prepare(args.config.resolve())


if __name__ == "__main__":
    main()
