#!/usr/bin/env python3
"""Reclassify the measured Thor frontier for looser TTFT SLO profiles.

This is an offline selector. It never changes clocks and does not claim online
joint-SLO feasibility; the frozen-trace serving campaign remains authoritative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_q4_ttft_extension_v1.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
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


def number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite metric: {value!r}")
    return result


def workload_key(row: dict[str, Any]) -> tuple[int, ...]:
    value = row.get("workload")
    if not isinstance(value, list) or not value:
        raise ValueError(f"invalid workload: {value!r}")
    return tuple(int(item) for item in value)


def dominates(left: dict[str, Any], right: dict[str, Any], objectives: list[str]) -> bool:
    lhs = [number(left[name]) for name in objectives]
    rhs = [number(right[name]) for name in objectives]
    return all(a <= b for a, b in zip(lhs, rhs)) and any(a < b for a, b in zip(lhs, rhs))


def pareto(rows: list[dict[str, Any]], objectives: list[str]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in rows
        if not any(
            other is not candidate and dominates(other, candidate, objectives)
            for other in rows
        )
    ]


def normalized_scores(
    rows: list[dict[str, Any]], objectives: list[str], weights: dict[str, float]
) -> list[tuple[float, dict[str, Any]]]:
    bounds = {
        objective: (
            min(number(row[objective]) for row in rows),
            max(number(row[objective]) for row in rows),
        )
        for objective in objectives
    }
    scored = []
    for row in rows:
        score = 0.0
        for objective in objectives:
            low, high = bounds[objective]
            normalized = 0.0 if high == low else (number(row[objective]) - low) / (high - low)
            score += weights[objective] * normalized
        scored.append((score, row))
    return sorted(
        scored,
        key=lambda item: (
            item[0],
            number(item[1]["energy_per_token_j"]),
            number(item[1]["avg_power_w"]),
            number(item[1]["ttft_p95_ms"]),
            number(item[1]["tpot_p95_ms"]),
            canonical(item[1].get("hardware")),
        ),
    )


def source_rows(rows: list[dict[str, Any]], source_filter: dict[str, Any]) -> list[dict[str, Any]]:
    scenario = source_filter["scenario"]
    tpot_limit = number(source_filter["tpot_limit_ms"])
    selected = [
        row
        for row in rows
        if row.get("strategy") == "frontier_point"
        and row.get("scenario") == scenario
        and abs(number(row.get("tpot_limit_ms")) - tpot_limit) < 1e-9
    ]
    if selected:
        return selected
    available = sorted(
        {
            (str(row.get("scenario")), row.get("tpot_limit_ms"))
            for row in rows
            if row.get("strategy") == "frontier_point"
        }
    )
    raise RuntimeError(f"source filter matched no frontier rows; available={available}")


def deduplicate(rows: list[dict[str, Any]]) -> dict[tuple[int, ...], list[dict[str, Any]]]:
    unique: dict[tuple[tuple[int, ...], str], dict[str, Any]] = {}
    for row in rows:
        key = (workload_key(row), canonical(row.get("hardware")))
        previous = unique.get(key)
        rank = (
            int(row.get("requests") or 0),
            int(row.get("blocks") or 0),
            -number(row["energy_per_token_j"]),
        )
        previous_rank = None
        if previous is not None:
            previous_rank = (
                int(previous.get("requests") or 0),
                int(previous.get("blocks") or 0),
                -number(previous["energy_per_token_j"]),
            )
        if previous is None or rank > previous_rank:
            unique[key] = row
    grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for (workload, _), row in unique.items():
        grouped.setdefault(workload, []).append(row)
    return grouped


def compact(row: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    result = {
        "point": row.get("point"),
        "hardware": row.get("hardware"),
        "avg_power_w": number(row["avg_power_w"]),
        "energy_per_token_j": number(row["energy_per_token_j"]),
        "ttft_p95_ms": number(row["ttft_p95_ms"]),
        "tpot_p95_ms": number(row["tpot_p95_ms"]),
        "requests": int(row.get("requests") or 0),
        "blocks": int(row.get("blocks") or 0),
    }
    if score is not None:
        result["normalized_utility"] = score
    return result


def comparison_row(
    source: dict[str, Any], profile: dict[str, Any], strategy: str, score: float
) -> dict[str, Any]:
    row = dict(source)
    row.update(
        {
            "scenario": profile["scenario"],
            "ttft_limit_ms": number(profile["ttft_limit_ms"]),
            "tpot_limit_ms": number(profile["tpot_limit_ms"]),
            "strategy": strategy,
            "selection_role": "custom_slo_pareto_representative",
            "passed": True,
            "status": "pass_observed",
            "source_status": "static_p95_reclassified",
            "normalized_utility": score,
            "online_joint_slo_validated": False,
        }
    )
    return row


def build(config_path: Path) -> Path:
    config = load_json(config_path)
    frontier_path = Path(config["frontier_validation"])
    selection_path = Path(config["selection_metadata"])
    output_dir = Path(config["output_dir"])
    objectives = list(config["objectives"])
    weights = config["preference_weights"]

    for name, vector in weights.items():
        if set(vector) != set(objectives):
            raise ValueError(f"{name}: weight/objective mismatch")
        total = sum(number(value) for value in vector.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{name}: weights sum to {total}, expected 1")

    rows = load_json(frontier_path)
    if not isinstance(rows, list):
        raise TypeError("frontier_validation must contain a JSON list")
    grouped = deduplicate(source_rows(rows, config["source_filter"]))

    result: dict[str, Any] = {
        "protocol": config["protocol"],
        "method": "per-workload feasible Pareto plus normalized weighted representatives",
        "warning": "Static p95 is a prefilter; frozen-trace request-level joint SLO is authoritative.",
        "objectives": objectives,
        "preference_weights": weights,
        "source": {
            "frontier_validation": str(frontier_path),
            "frontier_validation_sha256": sha256(frontier_path),
            "selection_metadata": str(selection_path),
            "selection_metadata_sha256": sha256(selection_path),
            "source_filter": config["source_filter"],
        },
        "profiles": {},
    }
    flat_comparison: list[dict[str, Any]] = []

    for profile in config["slo_profiles"]:
        profile_result: dict[str, Any] = {
            "scenario": profile["scenario"],
            "ttft_limit_ms": number(profile["ttft_limit_ms"]),
            "tpot_limit_ms": number(profile["tpot_limit_ms"]),
            "workloads": {},
        }
        for workload in sorted(grouped):
            feasible = [
                row
                for row in grouped[workload]
                if number(row["ttft_p95_ms"]) <= number(profile["ttft_limit_ms"])
                and number(row["tpot_p95_ms"]) <= number(profile["tpot_limit_ms"])
            ]
            frontier = pareto(feasible, objectives) if feasible else []
            selections: dict[str, Any] = {}
            for strategy, vector in weights.items():
                if not frontier:
                    continue
                score, selected = normalized_scores(frontier, objectives, vector)[0]
                selections[strategy] = compact(selected, score)
                flat_comparison.append(comparison_row(selected, profile, strategy, score))
            profile_result["workloads"]["x".join(map(str, workload))] = {
                "workload": list(workload),
                "measured_points": len(grouped[workload]),
                "feasible_points": len(feasible),
                "pareto_points": [compact(row) for row in frontier],
                "selections": selections,
            }
        result["profiles"][profile["name"]] = profile_result

    output_dir.mkdir(parents=True, exist_ok=True)
    plans_path = output_dir / "plans.json"
    atomic_json(plans_path, result)
    atomic_json(output_dir / "comparison.json", flat_comparison)
    atomic_json(
        output_dir / "protocol.json",
        {
            "protocol": config["protocol"],
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "plans": str(plans_path),
            "plans_sha256": sha256(plans_path),
            "comparison": str(output_dir / "comparison.json"),
            "comparison_rows": len(flat_comparison),
            "hardware_frequency_changes": False,
        },
    )
    return plans_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    path = build(args.config.resolve())
    print(json.dumps({"plans": str(path), "hardware_frequency_changes": False}))


if __name__ == "__main__":
    main()
