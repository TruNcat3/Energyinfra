#!/usr/bin/env python3
"""Evaluate frozen Thor results with prompt-length-aware TTFT SLOs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.active import analyze_thor_slo_sensitivity_v1 as common


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_prompt_aware_slo_v1.json"


def response_passes(response: dict[str, Any], limits: dict[int, float], tpot_limit: float) -> bool:
    prompt_tokens = int(response["prompt_tokens"])
    if prompt_tokens not in limits:
        raise RuntimeError(f"No TTFT limit for prompt length {prompt_tokens}")
    return (
        float(response["ttft_end_to_end_ms"]) <= limits[prompt_tokens]
        and float(response["tpot_ms"]) <= tpot_limit
    )


def attainment(responses: list[dict[str, Any]], limits: dict[int, float], tpot_limit: float) -> float:
    return sum(response_passes(response, limits, tpot_limit) for response in responses) / len(responses)


def bucket_attainment(
    responses: list[dict[str, Any]], limits: dict[int, float], tpot_limit: float
) -> dict[str, float]:
    output = {}
    for prompt_tokens in sorted({int(response["prompt_tokens"]) for response in responses}):
        bucket = [response for response in responses if int(response["prompt_tokens"]) == prompt_tokens]
        output[str(prompt_tokens)] = attainment(bucket, limits, tpot_limit)
    return output


def aggregate(
    rate: int,
    trials: list[dict[str, Any]],
    schedule_name: str,
    limits: dict[int, float],
    tpot_limit: float,
    required: float,
    require_all: bool,
    require_buckets: bool,
) -> dict[str, Any]:
    responses = [response for trial in trials for response in trial["responses"]]
    repeat_attainment = [attainment(trial["responses"], limits, tpot_limit) for trial in trials]
    repeat_buckets = [bucket_attainment(trial["responses"], limits, tpot_limit) for trial in trials]
    pooled_attainment = attainment(responses, limits, tpot_limit)
    feasible = pooled_attainment >= required
    if require_all:
        feasible = feasible and all(value >= required for value in repeat_attainment)
    if require_buckets:
        feasible = feasible and all(
            value >= required
            for buckets in repeat_buckets
            for value in buckets.values()
        )
    fixed_energy = sum(float(trial["fixed_horizon"]["total_energy_j"]) for trial in trials)
    fixed_duration = sum(float(trial["duration_s"]) for trial in trials)
    total_energy = sum(float(trial["total_energy_j"]) for trial in trials)
    tokens = sum(int(trial["completed_output_tokens"]) for trial in trials)
    strategy = trials[0]["strategy"]
    return {
        "schedule": schedule_name,
        "tpot_limit_ms": int(tpot_limit),
        "offered_output_tps": rate,
        "strategy": strategy["name"],
        "strategy_label": strategy["label"],
        "method_class": strategy["method_class"],
        "parallel": int(strategy["parallel"]),
        "attainment": pooled_attainment,
        "bucket_attainment": bucket_attainment(responses, limits, tpot_limit),
        "repeat_attainment": repeat_attainment,
        "repeat_bucket_attainment": repeat_buckets,
        "feasible": feasible,
        "requests": len(responses),
        "avg_power_w": fixed_energy / fixed_duration,
        "energy_per_token_j": total_energy / tokens,
        "ttft_p95_ms": common.percentile(
            [float(response["ttft_end_to_end_ms"]) for response in responses], 0.95
        ),
        "tpot_p95_ms": common.percentile(
            [float(response["tpot_ms"]) for response in responses], 0.95
        ),
    }


def prefix_capacity(rates: list[int], feasible_rates: list[int]) -> int | None:
    feasible = set(feasible_rates)
    last = None
    for rate in rates:
        if rate not in feasible:
            break
        last = rate
    return last


def analyze(config_path: Path) -> None:
    config = common.load(config_path)
    if config.get("schema") != "energyinfra-thor-prompt-aware-slo-v1":
        raise RuntimeError("Unsupported prompt-aware SLO schema")
    output = common.resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results, manifest = common.read_results(config)
    expected_repeats = int(config["expected_repeats"])
    groups = common.grouped(results, expected_repeats)
    rates = sorted({rate for rate, _ in groups})
    required = float(config["required_attainment"])
    require_all = bool(config["require_all_repeats"])
    require_buckets = bool(config["require_each_prompt_bucket"])
    objectives = list(config["pareto_objectives"])
    detail_rows: list[dict[str, Any]] = []
    matrix = []

    for schedule in config["ttft_schedules"]:
        schedule_name = schedule["name"]
        limits = {int(key): float(value) for key, value in schedule["limits_ms"].items()}
        observed_prompts = {
            int(response["prompt_tokens"])
            for result in results
            for response in result["responses"]
        }
        if observed_prompts != set(limits):
            raise RuntimeError(
                f"TTFT schedule {schedule_name} covers {sorted(limits)}, observed {sorted(observed_prompts)}"
            )
        for tpot in config["tpot_thresholds_ms"]:
            cells = []
            combination_rows = []
            for rate in rates:
                rows = [
                    aggregate(
                        rate, trials, schedule_name, limits, float(tpot), required,
                        require_all, require_buckets,
                    )
                    for (group_rate, _), trials in groups.items()
                    if group_rate == rate
                ]
                rows.sort(key=lambda row: row["strategy"])
                feasible = [row for row in rows if row["feasible"]]
                frontier = common.pareto(feasible, objectives)
                frontier_names = {row["strategy"] for row in frontier}
                for row in rows:
                    row["pareto"] = row["strategy"] in frontier_names
                    detail_rows.append(row)
                    combination_rows.append(row)
                cells.append({
                    "offered_output_tps": rate,
                    "feasible": bool(feasible),
                    "feasible_strategies": [row["strategy"] for row in feasible],
                    "pareto_strategies": sorted(frontier_names),
                })
            feasible_rates = [cell["offered_output_tps"] for cell in cells if cell["feasible"]]
            highest = max(feasible_rates) if feasible_rates else None
            prefix = prefix_capacity(rates, feasible_rates)
            at_highest = [
                row for row in combination_rows
                if row["offered_output_tps"] == highest and row["feasible"]
            ] if highest is not None else []
            matrix.append({
                "schedule": schedule_name,
                "ttft_limits_ms": {str(key): int(value) for key, value in limits.items()},
                "tpot_limit_ms": int(tpot),
                "highest_observed_feasible_tps": highest,
                "sampled_prefix_capacity_tps": prefix,
                "monotonic_over_sampled_rates": highest == prefix,
                "feasible_rates_tps": feasible_rates,
                "pareto_at_highest": sorted(row["strategy"] for row in at_highest if row["pareto"]),
                "power_min_at_highest": min(at_highest, key=lambda row: row["avg_power_w"])["strategy"] if at_highest else None,
                "energy_min_at_highest": min(at_highest, key=lambda row: row["energy_per_token_j"])["strategy"] if at_highest else None,
                "rates": cells,
            })

    report = {
        "schema": "energyinfra-thor-prompt-aware-slo-analysis-v1",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": common.sha256(config_path),
        "analysis_source_sha256": common.sha256(Path(__file__).resolve()),
        "required_attainment": required,
        "require_all_repeats": require_all,
        "require_each_prompt_bucket": require_buckets,
        "input_result_files": len(results),
        "rates_tps": rates,
        "matrix": matrix,
    }
    common.atomic_json(output / "prompt_aware_matrix.json", report)
    common.atomic_json(output / "input_manifest.json", {
        "schema": "energyinfra-thor-prompt-aware-input-manifest-v1",
        "files": manifest,
    })

    columns = [
        "schedule", "tpot_limit_ms", "offered_output_tps", "strategy", "method_class",
        "parallel", "feasible", "pareto", "attainment", "bucket_attainment",
        "repeat_attainment", "repeat_bucket_attainment", "requests", "avg_power_w",
        "energy_per_token_j", "ttft_p95_ms", "tpot_p95_ms",
    ]
    with (output / "prompt_aware_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in detail_rows:
            item = {name: row[name] for name in columns}
            for name in ("bucket_attainment", "repeat_attainment", "repeat_bucket_attainment"):
                item[name] = json.dumps(item[name], sort_keys=True)
            writer.writerow(item)

    gate_description = (
        f"Gate: every repetition and every observed prompt bucket must reach {required:.0%} attainment."
        if require_buckets
        else f"Gate: every complete repetition must reach {required:.0%} overall request attainment."
    )
    lines = [
        "# Thor prompt-aware SLO analysis v1",
        "",
        f"Inputs: {len(results)} fixed-work results over {len(rates)} offered rates.",
        gate_description,
        "The sampled-prefix capacity stops at the first infeasible sampled rate.",
        "",
        "| TTFT schedule | TPOT ms | Highest feasible tps | Sampled-prefix tps | Monotonic | Feasible rates | Pareto at highest |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in matrix:
        highest = "--" if item["highest_observed_feasible_tps"] is None else str(item["highest_observed_feasible_tps"])
        prefix = "--" if item["sampled_prefix_capacity_tps"] is None else str(item["sampled_prefix_capacity_tps"])
        feasible = ", ".join(str(rate) for rate in item["feasible_rates_tps"]) or "--"
        pareto_names = ", ".join(item["pareto_at_highest"]) or "--"
        lines.append(
            f"| {item['schedule']} | {item['tpot_limit_ms']} | {highest} | {prefix} | "
            f"{'yes' if item['monotonic_over_sampled_rates'] else 'no'} | {feasible} | {pareto_names} |"
        )
    (output / "PROMPT_AWARE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "event": "prompt_aware_slo_complete",
        "input_results": len(results),
        "matrix_points": len(matrix),
        "detail_rows": len(detail_rows),
        "output": str(output / "PROMPT_AWARE.md"),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    analyze(common.resolve(args.config))


if __name__ == "__main__":
    main()
