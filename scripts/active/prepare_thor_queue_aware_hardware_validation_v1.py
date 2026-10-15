#!/usr/bin/env python3
"""Build a frozen comparison adapter for queue-aware hardware validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_queue_aware_hardware_validation_v1.json"
WORKLOADS = ((64, 64), (128, 512), (512, 512), (1024, 512))


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
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


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_workload(label: str) -> tuple[int, int]:
    left, right = label.split("x", 1)
    return int(left), int(right)


def queue_maps(shortlist: list[dict[str, Any]]) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in shortlist:
        unique.setdefault(tuple(int(value) for value in row["policy_indices"]), row)
    energy = next(
        (row for row in unique.values() if "energy_min" in row["role"]),
        None,
    )
    power = next(
        (row for row in unique.values() if "power_min_without_energy_regression" in row["role"]),
        None,
    )
    if energy is None or power is None:
        raise RuntimeError("queue-aware shortlist lacks energy-min or power-min")
    return {
        "queue_energy_min": {
            parse_workload(label): hardware for label, hardware in energy["hardware_by_workload"].items()
        },
        "queue_power_min": {
            parse_workload(label): hardware for label, hardware in power["hardware_by_workload"].items()
        },
    }


def balanced_map(directory: Path) -> dict[tuple[int, int], dict[str, Any]]:
    choices: dict[tuple[int, int], dict[str, dict[str, Any]]] = {key: {} for key in WORKLOADS}
    for path in sorted(directory.glob("result_*_balanced_weighted.json")):
        result = load(path)
        for response in result["responses"]:
            request = response["request"]
            key = (int(request["prompt_length"]), int(request["output_length"]))
            if key in choices:
                choices[key][canonical(response["hardware"])] = response["hardware"]
    invalid = {key: len(values) for key, values in choices.items() if len(values) != 1}
    if invalid:
        raise RuntimeError(f"balanced hardware map is missing or inconsistent: {invalid}")
    return {key: next(iter(values.values())) for key, values in choices.items()}


def comparison_rows(maps: dict[str, dict[tuple[int, int], dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for strategy, mapping in maps.items():
        for key in WORKLOADS:
            if key not in mapping:
                raise RuntimeError(f"{strategy} lacks workload {key}")
            rows.append({
                "workload": list(key),
                "strategy": strategy,
                "scenario": "long_context_2s",
                "ttft_limit_ms": 2000,
                "tpot_limit_ms": 100,
                "hardware": mapping[key],
                "passed": True,
                "status": "pass_observed",
                "selection_role": "queue_aware_hardware_validation_adapter",
                "hardware_validated": strategy == "balanced_weighted",
            })
    return rows


def prepare(config_path: Path) -> Path:
    config = load(config_path)
    shortlist_path = resolve(config["queue_shortlist"])
    search_path = resolve(config["queue_search_result"])
    balanced_dir = resolve(config["balanced_source_dir"])
    adapter = resolve(config["adapter_dir"])
    maps = queue_maps(load(shortlist_path))
    maps["balanced_weighted"] = balanced_map(balanced_dir)
    rows = comparison_rows(maps)
    selection = {
        "schema": "energyinfra-queue-aware-validation-adapter-v1",
        "warning": "Queue candidates are predicted and require this hardware validation.",
        "strategies": {
            strategy: {"x".join(map(str, key)): value for key, value in mapping.items()}
            for strategy, mapping in maps.items()
        },
        "source": {
            "shortlist": str(shortlist_path),
            "shortlist_sha256": sha256(shortlist_path),
            "search_result": str(search_path),
            "search_result_sha256": sha256(search_path),
        },
    }
    adapter.mkdir(parents=True, exist_ok=True)
    comparison_path = adapter / "comparison.json"
    selection_path = adapter / "selection.json"
    atomic(comparison_path, rows)
    atomic(selection_path, selection)
    protocol = {
        "schema": selection["schema"],
        "comparison": str(comparison_path),
        "comparison_sha256": sha256(comparison_path),
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "comparison_rows": len(rows),
        "hardware_frequency_changes": False,
    }
    protocol_path = adapter / "protocol.json"
    atomic(protocol_path, protocol)
    print(json.dumps({"adapter_protocol": str(protocol_path), "comparison_rows": len(rows)}))
    return protocol_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    prepare(args.config.resolve())


if __name__ == "__main__":
    main()
