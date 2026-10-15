#!/usr/bin/env python3
"""Reclassify one frozen-trace serving run under several joint SLO profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_q4_long_generation_ttft_reclassification_v1.json"
BASELINES = ("dynamic", "max_clock", "max_clock_idle_dynamic")
OBJECTIVES = ("avg_power_w", "energy_per_token_j", "ttft_p95_ms", "tpot_p95_ms")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic(path: Path, payload: Any) -> None:
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


def percentile(values: list[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def wilson(passed: int, offered: int) -> list[float]:
    if not offered:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = passed / offered
    denominator = 1 + z * z / offered
    center = (p + z * z / (2 * offered)) / denominator
    radius = z * math.sqrt(p * (1 - p) / offered + z * z / (4 * offered * offered)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def response_metrics(response: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = response["request"]
    metrics = response.get("metrics") or {}
    return request, metrics


def request_pass(response: dict[str, Any], ttft_limit: float, tpot_limit: float) -> bool:
    if response.get("dropped"):
        return False
    request, metrics = response_metrics(response)
    requested_output = int(request["output_length"])
    completed = int(metrics.get("output_tokens") or 0) >= requested_output
    ttft = metrics.get("ttft_end_to_end_ms")
    tpot = metrics.get("tpot_ms")
    e2e = metrics.get("arrival_to_completion_ms")
    e2e_limit = ttft_limit + tpot_limit * max(0, requested_output - 1)
    return bool(
        completed
        and ttft is not None
        and tpot is not None
        and e2e is not None
        and float(ttft) <= ttft_limit
        and float(tpot) <= tpot_limit
        and float(e2e) <= e2e_limit
    )


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    lhs = [float(left[name]) for name in OBJECTIVES]
    rhs = [float(right[name]) for name in OBJECTIVES]
    return all(a <= b for a, b in zip(lhs, rhs)) and any(a < b for a, b in zip(lhs, rhs))


def paired_interval(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return {"mean_pct": mean, "ci95_low_pct": None, "ci95_high_pct": None, "values_pct": values}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    critical = {2: 12.706204736, 3: 4.30265273}.get(len(values), 1.959963984540054)
    margin = critical * math.sqrt(variance / len(values))
    return {
        "mean_pct": mean,
        "ci95_low_pct": mean - margin,
        "ci95_high_pct": mean + margin,
        "values_pct": values,
    }


def aggregate(
    strategy: str,
    runs: list[dict[str, Any]],
    profile: dict[str, Any],
    required: float,
) -> dict[str, Any]:
    ttft_limit = float(profile["ttft_limit_ms"])
    tpot_limit = float(profile["tpot_limit_ms"])
    responses = [response for run in runs for response in run["responses"]]
    repeat_attainment = []
    passed = 0
    ttft_values: list[float] = []
    tpot_values: list[float] = []
    e2e_values: list[float] = []
    workload_passed: dict[str, int] = defaultdict(int)
    workload_offered: dict[str, int] = defaultdict(int)
    for run in runs:
        repeat_passed = sum(request_pass(response, ttft_limit, tpot_limit) for response in run["responses"])
        repeat_attainment.append(repeat_passed / len(run["responses"]) if run["responses"] else 0.0)
    for response in responses:
        request, metrics = response_metrics(response)
        workload = f"p{request['prompt_length']}_o{request['output_length']}"
        workload_offered[workload] += 1
        accepted = request_pass(response, ttft_limit, tpot_limit)
        passed += int(accepted)
        workload_passed[workload] += int(accepted)
        if not response.get("dropped") and metrics:
            ttft_values.append(float(metrics["ttft_end_to_end_ms"]))
            tpot_values.append(float(metrics["tpot_ms"]))
            e2e_values.append(float(metrics["arrival_to_completion_ms"]))
    offered = len(responses)
    energy = sum(float(run["total_energy_j"]) for run in runs)
    tokens = sum(int(run["output_tokens"]) for run in runs)
    duration = sum(float(run["duration_with_drain_s"]) for run in runs)
    return {
        "strategy": strategy,
        "repeats": len(runs),
        "offered": offered,
        "completed": sum(int(run["completed"]) for run in runs),
        "passed": passed,
        "joint_goodput": passed / offered if offered else 0.0,
        "wilson_ci95": wilson(passed, offered),
        "repeat_attainment": repeat_attainment,
        "all_repeats_pass": all(value >= required for value in repeat_attainment),
        "ttft_p95_ms": percentile(ttft_values, 0.95),
        "ttft_p99_ms": percentile(ttft_values, 0.99),
        "tpot_p95_ms": percentile(tpot_values, 0.95),
        "tpot_p99_ms": percentile(tpot_values, 0.99),
        "e2e_p95_ms": percentile(e2e_values, 0.95),
        "e2e_p99_ms": percentile(e2e_values, 0.99),
        "total_energy_j": energy,
        "avg_power_w": energy / duration,
        "energy_per_token_j": energy / tokens,
        "per_repeat_energy_j": [float(run["total_energy_j"]) for run in runs],
        "workload_attainment": {
            name: {
                "passed": workload_passed[name],
                "offered": workload_offered[name],
                "attainment": workload_passed[name] / workload_offered[name],
            }
            for name in sorted(workload_offered)
        },
    }


def analyze(config_path: Path) -> Path:
    config = load(config_path)
    input_dir = resolve(config["input_dir"])
    output_dir = resolve(config["output_dir"])
    files = sorted(input_dir.glob("result_*_*.json"))
    if not files:
        raise RuntimeError(f"no result files in {input_dir}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance = {}
    for path in files:
        run = load(path)
        grouped[run["strategy"]].append(run)
        provenance[str(path.relative_to(ROOT))] = digest(path)
    for strategy, runs in grouped.items():
        runs.sort(key=lambda item: int(item["repeat"]))
        repeats = [int(item["repeat"]) for item in runs]
        if repeats != list(range(len(runs))):
            raise RuntimeError(f"{strategy}: incomplete repeats {repeats}")

    profiles = []
    required = float(config["required_observed_attainment"])
    for profile in config["profiles"]:
        rows = [aggregate(strategy, runs, profile, required) for strategy, runs in sorted(grouped.items())]
        by_strategy = {row["strategy"]: row for row in rows}
        for row in rows:
            for baseline in BASELINES:
                base = by_strategy[baseline]
                row[f"energy_delta_vs_{baseline}_pct"] = 100.0 * (
                    row["total_energy_j"] / base["total_energy_j"] - 1.0
                )
                paired = [
                    100.0 * (value / reference - 1.0)
                    for value, reference in zip(row["per_repeat_energy_j"], base["per_repeat_energy_j"])
                ]
                row[f"paired_energy_vs_{baseline}"] = paired_interval(paired)
        feasible = [row for row in rows if row["all_repeats_pass"]]
        for row in rows:
            row["serving_pareto"] = row in feasible and not any(
                other is not row and dominates(other, row) for other in feasible
            )
        profiles.append({"profile": profile, "rows": rows})

    payload = {
        "schema": config["schema"],
        "input_dir": str(input_dir),
        "required_observed_attainment": required,
        "provenance": provenance,
        "profiles": profiles,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "reclassification.json"
    atomic(result_path, payload)

    csv_fields = [
        "profile", "ttft_limit_ms", "tpot_limit_ms", "strategy", "passed", "offered",
        "joint_goodput", "all_repeats_pass", "ttft_p95_ms", "ttft_p99_ms", "tpot_p95_ms",
        "tpot_p99_ms", "e2e_p95_ms", "avg_power_w", "energy_per_token_j", "total_energy_j",
        "energy_delta_vs_dynamic_pct", "energy_delta_vs_max_clock_pct",
        "energy_delta_vs_max_clock_idle_dynamic_pct", "serving_pareto",
    ]
    with (output_dir / "reclassification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for item in profiles:
            for row in item["rows"]:
                writer.writerow({
                    **{key: row.get(key) for key in csv_fields},
                    "profile": item["profile"]["name"],
                    "ttft_limit_ms": item["profile"]["ttft_limit_ms"],
                    "tpot_limit_ms": item["profile"]["tpot_limit_ms"],
                })

    lines = [
        "# Thor Q4 long-generation TTFT-extension reclassification",
        "",
        "A request passes only when completion, TTFT, TPOT and derived end-to-end latency all pass.",
        f"A strategy is feasible only when every repeat reaches at least {required * 100:.0f}% joint goodput.",
        "Energy and power use identical frozen work and include the full arrival horizon plus queue drain.",
        "",
    ]
    for item in profiles:
        profile = item["profile"]
        lines.extend([
            f"## {profile['name']} ({profile['ttft_limit_ms']:.0f}/{profile['tpot_limit_ms']:.0f} ms)",
            "",
            "| Strategy | Joint | Every repeat | TTFT p95/p99 ms | TPOT p95/p99 ms | W | J/token | Energy vs Dynamic | Energy vs Max | Energy vs MaxIdle | Pareto |",
            "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ])
        for row in item["rows"]:
            lines.append(
                f"| {row['strategy']} | {100 * row['joint_goodput']:.2f}% | {str(row['all_repeats_pass'])} | "
                f"{row['ttft_p95_ms']:.1f}/{row['ttft_p99_ms']:.1f} | "
                f"{row['tpot_p95_ms']:.1f}/{row['tpot_p99_ms']:.1f} | "
                f"{row['avg_power_w']:.3f} | {row['energy_per_token_j']:.3f} | "
                f"{row['energy_delta_vs_dynamic_pct']:+.3f}% | "
                f"{row['energy_delta_vs_max_clock_pct']:+.3f}% | "
                f"{row['energy_delta_vs_max_clock_idle_dynamic_pct']:+.3f}% | "
                f"{str(row['serving_pareto'])} |"
            )
        lines.append("")
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = analyze(args.config.resolve())
    print(json.dumps({"result": str(result), "report": str(result.parent / "RESULTS.md")}))


if __name__ == "__main__":
    main()
