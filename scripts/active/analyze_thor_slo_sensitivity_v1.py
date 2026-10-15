#!/usr/bin/env python3
"""Re-evaluate frozen Thor serving results over an TTFT/TPOT SLO grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_slo_sensitivity_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def rate_for(profile: str) -> int:
    if not profile.startswith("rate_"):
        raise RuntimeError(f"Unsupported profile: {profile}")
    return int(profile.removeprefix("rate_"))


def read_results(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    results: list[dict[str, Any]] = []
    manifest: list[dict[str, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for value in config["inputs"]:
        directory = resolve(value)
        files = sorted(directory.glob("result_*.json"))
        if not files:
            raise RuntimeError(f"No result files in {directory}")
        for path in files:
            result = load(path)
            if result.get("schema") != "energyinfra-thor-joint-serving-result-v2":
                raise RuntimeError(f"Unsupported result schema: {path}")
            key = (result["profile"], int(result["repeat"]), result["strategy"]["name"])
            if key in seen:
                raise RuntimeError(f"Duplicate result key {key}: {path}")
            seen.add(key)
            if not result.get("comparable_fixed_work"):
                raise RuntimeError(f"Non-comparable fixed work: {path}")
            if result["offered"] != result["completed_exact"]:
                raise RuntimeError(f"Incomplete requests: {path}")
            if result["offered_output_tokens"] != result["completed_output_tokens"]:
                raise RuntimeError(f"Incomplete output tokens: {path}")
            for response in result["responses"]:
                if not response.get("completion_exact") or response.get("error") is not None:
                    raise RuntimeError(f"Invalid response in {path}: {response.get('request_id')}")
            results.append(result)
            manifest.append({
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            })
    return results, manifest


def grouped(results: list[dict[str, Any]], expected_repeats: int) -> dict[tuple[int, str], list[dict[str, Any]]]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(rate_for(result["profile"]), result["strategy"]["name"])].append(result)
    for key, trials in groups.items():
        repeats = sorted(int(trial["repeat"]) for trial in trials)
        if repeats != list(range(expected_repeats)):
            raise RuntimeError(f"Unexpected repeats for {key}: {repeats}")
    return groups


def aggregate(
    rate: int,
    trials: list[dict[str, Any]],
    ttft_limit: float,
    tpot_limit: float,
    required: float,
    require_all: bool,
) -> dict[str, Any]:
    responses = [response for trial in trials for response in trial["responses"]]
    repeat_attainment = []
    for trial in trials:
        passed = sum(
            response["ttft_end_to_end_ms"] <= ttft_limit
            and response["tpot_ms"] <= tpot_limit
            for response in trial["responses"]
        )
        repeat_attainment.append(passed / len(trial["responses"]))
    passed = sum(
        response["ttft_end_to_end_ms"] <= ttft_limit
        and response["tpot_ms"] <= tpot_limit
        for response in responses
    )
    feasible = all(value >= required for value in repeat_attainment)
    if not require_all:
        feasible = passed / len(responses) >= required
    fixed_energy = sum(trial["fixed_horizon"]["total_energy_j"] for trial in trials)
    fixed_duration = sum(float(trial["duration_s"]) for trial in trials)
    total_energy = sum(float(trial["total_energy_j"]) for trial in trials)
    tokens = sum(int(trial["completed_output_tokens"]) for trial in trials)
    strategy = trials[0]["strategy"]
    return {
        "offered_output_tps": rate,
        "strategy": strategy["name"],
        "strategy_label": strategy["label"],
        "method_class": strategy["method_class"],
        "parallel": int(strategy["parallel"]),
        "attainment": passed / len(responses),
        "repeat_attainment": repeat_attainment,
        "feasible": feasible,
        "requests": len(responses),
        "avg_power_w": fixed_energy / fixed_duration,
        "energy_per_token_j": total_energy / tokens,
        "ttft_p95_ms": percentile([response["ttft_end_to_end_ms"] for response in responses], 0.95),
        "tpot_p95_ms": percentile([response["tpot_ms"] for response in responses], 0.95),
    }


def pareto(rows: list[dict[str, Any]], objectives: list[str]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            no_worse = all(other[name] <= row[name] for name in objectives)
            better = any(other[name] < row[name] for name in objectives)
            if no_worse and better:
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier


def analyze(config_path: Path) -> None:
    config = load(config_path)
    if config.get("schema") != "energyinfra-thor-slo-sensitivity-v1":
        raise RuntimeError("Unsupported SLO-sensitivity schema")
    output = resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results, manifest = read_results(config)
    expected_repeats = int(config["expected_repeats"])
    groups = grouped(results, expected_repeats)
    required = float(config["required_attainment"])
    require_all = bool(config["require_all_repeats"])
    objectives = list(config["pareto_objectives"])
    detail_rows: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    rates = sorted({key[0] for key in groups})
    for ttft in config["ttft_thresholds_ms"]:
        for tpot in config["tpot_thresholds_ms"]:
            cells = []
            for rate in rates:
                rows = [
                    aggregate(rate, trials, float(ttft), float(tpot), required, require_all)
                    for (group_rate, _), trials in groups.items()
                    if group_rate == rate
                ]
                rows.sort(key=lambda row: row["strategy"])
                feasible = [row for row in rows if row["feasible"]]
                frontier = pareto(feasible, objectives)
                frontier_names = {row["strategy"] for row in frontier}
                for row in rows:
                    row.update({
                        "ttft_limit_ms": int(ttft),
                        "tpot_limit_ms": int(tpot),
                        "pareto": row["strategy"] in frontier_names,
                    })
                    detail_rows.append(row)
                cells.append({
                    "offered_output_tps": rate,
                    "feasible": bool(feasible),
                    "feasible_strategies": [row["strategy"] for row in feasible],
                    "pareto_strategies": sorted(frontier_names),
                })
            feasible_rates = [cell["offered_output_tps"] for cell in cells if cell["feasible"]]
            capacity = max(feasible_rates) if feasible_rates else None
            capacity_rows = [
                row for row in detail_rows
                if row["ttft_limit_ms"] == ttft
                and row["tpot_limit_ms"] == tpot
                and row["offered_output_tps"] == capacity
                and row["feasible"]
            ] if capacity is not None else []
            matrix.append({
                "ttft_limit_ms": int(ttft),
                "tpot_limit_ms": int(tpot),
                "observed_capacity_tps": capacity,
                "feasible_rates_tps": feasible_rates,
                "capacity_pareto_strategies": sorted(
                    row["strategy"] for row in capacity_rows if row["pareto"]
                ),
                "power_min_at_capacity": min(capacity_rows, key=lambda row: row["avg_power_w"])["strategy"] if capacity_rows else None,
                "energy_min_at_capacity": min(capacity_rows, key=lambda row: row["energy_per_token_j"])["strategy"] if capacity_rows else None,
                "rates": cells,
            })

    report = {
        "schema": "energyinfra-thor-slo-sensitivity-analysis-v1",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "analysis_source_sha256": sha256(Path(__file__).resolve()),
        "required_attainment": required,
        "require_all_repeats": require_all,
        "input_result_files": len(results),
        "rates_tps": rates,
        "matrix": matrix,
    }
    atomic_json(output / "sensitivity_matrix.json", report)
    atomic_json(output / "input_manifest.json", {
        "schema": "energyinfra-thor-slo-sensitivity-input-manifest-v1",
        "files": manifest,
    })

    columns = [
        "ttft_limit_ms", "tpot_limit_ms", "offered_output_tps", "strategy",
        "method_class", "parallel", "feasible", "pareto", "attainment",
        "repeat_attainment", "requests", "avg_power_w", "energy_per_token_j",
        "ttft_p95_ms", "tpot_p95_ms",
    ]
    with (output / "sensitivity_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in detail_rows:
            item = {name: row[name] for name in columns}
            item["repeat_attainment"] = json.dumps(item["repeat_attainment"])
            writer.writerow(item)

    lines = [
        "# Thor TTFT/TPOT SLO sensitivity v1",
        "",
        f"Inputs: {len(results)} fixed-work result files across {len(rates)} offered rates.",
        f"Feasibility: every one of {expected_repeats} repetitions must reach at least {required:.0%} request attainment.",
        "Capacity is the highest observed feasible offered rate; it is not an interpolated saturation estimate.",
        "",
        "| TTFT limit ms | TPOT limit ms | Capacity tps | Feasible rates | Pareto at capacity | Power minimum | Energy minimum |",
        "|---:|---:|---:|---|---|---|---|",
    ]
    for item in matrix:
        capacity = "--" if item["observed_capacity_tps"] is None else str(item["observed_capacity_tps"])
        rates_text = ", ".join(str(rate) for rate in item["feasible_rates_tps"]) or "--"
        pareto_text = ", ".join(item["capacity_pareto_strategies"]) or "--"
        lines.append(
            f"| {item['ttft_limit_ms']} | {item['tpot_limit_ms']} | {capacity} | "
            f"{rates_text} | {pareto_text} | {item['power_min_at_capacity'] or '--'} | "
            f"{item['energy_min_at_capacity'] or '--'} |"
        )
    (output / "SENSITIVITY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "event": "slo_sensitivity_complete",
        "input_results": len(results),
        "matrix_points": len(matrix),
        "detail_rows": len(detail_rows),
        "output": str(output / "SENSITIVITY.md"),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    analyze(resolve(args.config))


if __name__ == "__main__":
    main()
