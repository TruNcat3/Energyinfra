#!/usr/bin/env python3
"""Compare marginal and per-request joint TTFT/TPOT SLO semantics."""

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


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_slo_semantics_tpot30_v1.json"


def aggregate(
    rate: int,
    trials: list[dict[str, Any]],
    schedule: str,
    limits: dict[int, float],
    tpot_limit: float,
    required: float,
) -> dict[str, Any]:
    responses = [response for trial in trials for response in trial["responses"]]
    repeat_ttft = []
    repeat_tpot = []
    repeat_joint = []
    for trial in trials:
        count = len(trial["responses"])
        ttft = [
            float(response["ttft_end_to_end_ms"]) <= limits[int(response["prompt_tokens"])]
            for response in trial["responses"]
        ]
        tpot = [float(response["tpot_ms"]) <= tpot_limit for response in trial["responses"]]
        repeat_ttft.append(sum(ttft) / count)
        repeat_tpot.append(sum(tpot) / count)
        repeat_joint.append(sum(a and b for a, b in zip(ttft, tpot)) / count)
    strategy = trials[0]["strategy"]
    fixed_energy = sum(float(trial["fixed_horizon"]["total_energy_j"]) for trial in trials)
    fixed_duration = sum(float(trial["duration_s"]) for trial in trials)
    total_energy = sum(float(trial["total_energy_j"]) for trial in trials)
    tokens = sum(int(trial["completed_output_tokens"]) for trial in trials)
    return {
        "schedule": schedule,
        "offered_output_tps": rate,
        "strategy": strategy["name"],
        "strategy_label": strategy["label"],
        "parallel": int(strategy["parallel"]),
        "marginal_feasible": all(value >= required for value in repeat_ttft + repeat_tpot),
        "joint_feasible": all(value >= required for value in repeat_joint),
        "repeat_ttft_attainment": repeat_ttft,
        "repeat_tpot_attainment": repeat_tpot,
        "repeat_joint_attainment": repeat_joint,
        "avg_power_w": fixed_energy / fixed_duration,
        "energy_per_token_j": total_energy / tokens,
        "ttft_p95_ms": common.percentile(
            [float(response["ttft_end_to_end_ms"]) for response in responses], 0.95
        ),
        "tpot_p95_ms": common.percentile(
            [float(response["tpot_ms"]) for response in responses], 0.95
        ),
    }


def analyze(config_path: Path) -> None:
    config = common.load(config_path)
    if config.get("schema") != "energyinfra-thor-slo-semantics-v1":
        raise RuntimeError("Unsupported SLO-semantics schema")
    output = common.resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    results, manifest = common.read_results(config)
    groups = common.grouped(results, int(config["expected_repeats"]))
    rates = sorted({rate for rate, _ in groups})
    required = float(config["required_attainment"])
    tpot_limit = float(config["tpot_limit_ms"])
    objectives = list(config["pareto_objectives"])
    rows = []
    matrix = []
    for schedule in config["ttft_schedules"]:
        limits = {int(key): float(value) for key, value in schedule["limits_ms"].items()}
        schedule_rows = []
        for (rate, _), trials in groups.items():
            row = aggregate(rate, trials, schedule["name"], limits, tpot_limit, required)
            rows.append(row)
            schedule_rows.append(row)
        for semantics, field in (("marginal", "marginal_feasible"), ("joint_request", "joint_feasible")):
            feasible_rates = []
            cells = []
            for rate in rates:
                candidates = [row for row in schedule_rows if row["offered_output_tps"] == rate and row[field]]
                frontier = common.pareto(candidates, objectives)
                if candidates:
                    feasible_rates.append(rate)
                cells.append({
                    "offered_output_tps": rate,
                    "feasible_strategies": [row["strategy"] for row in candidates],
                    "pareto_strategies": [row["strategy"] for row in frontier],
                })
            matrix.append({
                "schedule": schedule["name"],
                "semantics": semantics,
                "feasible_rates_tps": feasible_rates,
                "highest_observed_feasible_tps": max(feasible_rates) if feasible_rates else None,
                "rates": cells,
            })
    report = {
        "schema": "energyinfra-thor-slo-semantics-analysis-v1",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": common.sha256(config_path),
        "analysis_source_sha256": common.sha256(Path(__file__).resolve()),
        "input_result_files": len(results),
        "required_attainment": required,
        "tpot_limit_ms": tpot_limit,
        "matrix": matrix,
    }
    common.atomic_json(output / "semantics.json", report)
    common.atomic_json(output / "input_manifest.json", {
        "schema": "energyinfra-thor-slo-semantics-input-manifest-v1",
        "files": manifest,
    })
    columns = [
        "schedule", "offered_output_tps", "strategy", "parallel",
        "marginal_feasible", "joint_feasible", "repeat_ttft_attainment",
        "repeat_tpot_attainment", "repeat_joint_attainment", "avg_power_w",
        "energy_per_token_j", "ttft_p95_ms", "tpot_p95_ms",
    ]
    with (output / "semantics_rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["schedule"], item["offered_output_tps"], item["strategy"])):
            item = {key: row[key] for key in columns}
            for key in ("repeat_ttft_attainment", "repeat_tpot_attainment", "repeat_joint_attainment"):
                item[key] = json.dumps(item[key])
            writer.writerow(item)
    lines = [
        "# Thor marginal versus joint-request SLO semantics v1",
        "",
        f"Each repetition contains 20 requests. Required attainment is {required:.0%}.",
        "Marginal semantics gates TTFT and TPOT attainment separately.",
        "Joint-request semantics requires the same request to satisfy both limits.",
        "",
        "| TTFT schedule | Semantics | Feasible rates tps | Highest observed tps | Pareto at highest |",
        "|---|---|---|---:|---|",
    ]
    for item in matrix:
        rates_text = ", ".join(str(rate) for rate in item["feasible_rates_tps"]) or "--"
        highest = item["highest_observed_feasible_tps"]
        highest_text = "--" if highest is None else str(highest)
        cell = next((cell for cell in item["rates"] if cell["offered_output_tps"] == highest), None)
        pareto_text = ", ".join(cell["pareto_strategies"]) if cell else "--"
        lines.append(
            f"| {item['schedule']} | {item['semantics']} | {rates_text} | {highest_text} | {pareto_text or '--'} |"
        )
    (output / "SEMANTICS.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({
        "event": "slo_semantics_complete",
        "input_results": len(results),
        "rows": len(rows),
        "output": str(output / "SEMANTICS.md"),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    analyze(common.resolve(args.config))


if __name__ == "__main__":
    main()
