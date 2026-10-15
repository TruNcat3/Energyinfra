#!/usr/bin/env python3
"""Re-evaluate frozen serving results under multi-metric SLO profiles."""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_slo_profiles_v1.json"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


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


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return values[index]


def wilson(passed, offered):
    if not offered:
        return [0.0, 1.0]
    z = 1.959963984540054
    probability = passed / offered
    denominator = 1 + z * z / offered
    center = (probability + z * z / (2 * offered)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / offered +
                           z * z / (4 * offered * offered)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_campaign(path):
    groups = defaultdict(dict)
    for result_path in sorted(path.glob("result_*.json")):
        result = json.loads(result_path.read_text())
        strategy = result["strategy"]
        repeat = result["repeat"]
        if repeat in groups[strategy]:
            raise RuntimeError(f"Duplicate result for {strategy} repeat {repeat}: {path}")
        groups[strategy][repeat] = result
    if not groups:
        raise RuntimeError(f"No serving results found: {path}")
    return groups


def request_metrics(response, profile):
    if response.get("dropped"):
        return {"complete": False, "pass": False}
    metrics = response.get("metrics", {})
    request = response.get("request", {})
    ttft = metrics.get("ttft_end_to_end_ms")
    tpot = metrics.get("tpot_ms")
    e2e = metrics.get("arrival_to_completion_ms")
    requested_output = request.get("output_length")
    if None in (ttft, tpot, e2e, requested_output):
        return {"complete": False, "pass": False}
    e2e_limit = profile["ttft_ms"] + profile["tpot_ms"] * max(0, requested_output - 1)
    return {
        "complete": True,
        "pass": ttft <= profile["ttft_ms"] and tpot <= profile["tpot_ms"] and e2e <= e2e_limit,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "e2e_ms": e2e,
        "e2e_limit_ms": e2e_limit,
        "e2e_budget_ratio": e2e / e2e_limit if e2e_limit else math.inf,
    }


def fixed_work_comparable(candidate, baseline, repeats):
    return set(candidate) == repeats and set(baseline) == repeats and all(
        candidate[index].get("comparable_fixed_work")
        and baseline[index].get("comparable_fixed_work")
        and candidate[index]["trace_sha256"] == baseline[index]["trace_sha256"]
        and candidate[index]["output_tokens"] == baseline[index]["output_tokens"]
        for index in repeats
    )


def evaluate(campaign, groups, profile_name, profile):
    repeat_sets = [set(results) for results in groups.values()]
    expected = set.intersection(*repeat_sets)
    if not expected:
        raise RuntimeError(f"No common repeats in {campaign}")
    rows = []
    for strategy, by_repeat in groups.items():
        results = [by_repeat[index] for index in sorted(expected)]
        per_repeat = []
        all_requests = []
        for result in results:
            requests = [request_metrics(response, profile) for response in result["responses"]]
            offered = result["offered"]
            passed = sum(item["pass"] for item in requests)
            per_repeat.append(passed / offered if offered else 0.0)
            all_requests.extend(requests)
        offered = sum(result["offered"] for result in results)
        passed = sum(item["pass"] for item in all_requests)
        complete = sum(item["complete"] for item in all_requests)
        energy = sum(result["total_energy_j"] for result in results)
        duration = sum(result["duration_with_drain_s"] for result in results)
        tokens = sum(result["output_tokens"] for result in results)
        valid = [item for item in all_requests if item["complete"]]
        interval = wilson(passed, offered)
        ttft_p99 = percentile([item["ttft_ms"] for item in valid], 0.99)
        tpot_p99 = percentile([item["tpot_ms"] for item in valid], 0.99)
        e2e_ratio_p99 = percentile([item["e2e_budget_ratio"] for item in valid], 0.99)
        row = {
            "campaign": campaign,
            "profile": profile_name,
            "strategy": strategy,
            "repeats": len(results),
            "offered": offered,
            "completed": complete,
            "joint_passed": passed,
            "joint_attainment": passed / offered if offered else 0.0,
            "wilson_ci95_low": interval[0],
            "wilson_ci95_high": interval[1],
            "every_repeat_joint_95_pass": complete == offered and all(
                value >= profile["required_joint_attainment"] for value in per_repeat
            ),
            "p99_all_metrics_pass": bool(ttft_p99 is not None and tpot_p99 is not None and
                                         e2e_ratio_p99 is not None and
                                         ttft_p99 <= profile["ttft_ms"] and
                                         tpot_p99 <= profile["tpot_ms"] and
                                         e2e_ratio_p99 <= 1.0),
            "ttft_p95_ms": percentile([item["ttft_ms"] for item in valid], 0.95),
            "ttft_p99_ms": ttft_p99,
            "tpot_p95_ms": percentile([item["tpot_ms"] for item in valid], 0.95),
            "tpot_p99_ms": tpot_p99,
            "e2e_p95_ms": percentile([item["e2e_ms"] for item in valid], 0.95),
            "e2e_p99_ms": percentile([item["e2e_ms"] for item in valid], 0.99),
            "e2e_budget_ratio_p99": e2e_ratio_p99,
            "total_energy_j": energy,
            "avg_power_w": energy / duration,
            "energy_per_token_j": energy / tokens if tokens else None,
            "duration_with_drain_s": duration,
            "switch_count": sum(len(result["switches"]) for result in results),
        }
        rows.append(row)
    repeats = expected
    for row in rows:
        candidate = groups[row["strategy"]]
        for baseline in ("dynamic", "max_clock", "max_clock_idle_dynamic"):
            base = groups.get(baseline, {})
            comparable = fixed_work_comparable(candidate, base, repeats)
            row[f"energy_delta_vs_{baseline}_pct"] = (
                100 * (row["total_energy_j"] /
                       sum(base[index]["total_energy_j"] for index in repeats) - 1)
                if comparable else None
            )
    eligible = [row for row in rows if row["every_repeat_joint_95_pass"]]
    objectives = ("avg_power_w", "energy_per_token_j", "ttft_p95_ms", "tpot_p95_ms")
    for row in rows:
        row["joint_slo_pareto"] = False
    for candidate in eligible:
        dominated = any(
            all(other[key] <= candidate[key] for key in objectives)
            and any(other[key] < candidate[key] for key in objectives)
            for other in eligible if other is not candidate
        )
        candidate["joint_slo_pareto"] = not dominated
    return sorted(rows, key=lambda item: item["strategy"])


def render(output, config, rows):
    number = lambda value: "--" if value is None else f"{value:.3f}"
    lines = [
        "# Thor serving multi-metric SLO audit v1",
        "",
        "This is an offline reclassification of frozen request-level measurements; no hardware result was regenerated.",
        "A request passes only when TTFT, TPOT and its output-length-derived E2E deadline all pass.",
        "The 95% joint-goodput gate is evaluated independently in every repeat. P99 compliance is reported separately.",
        "These threshold-inspired profiles are not MLPerf submissions, and random-weight model quality is not evaluated.",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["campaign"], row["profile"])].append(row)
    for (campaign, profile_name), items in grouped.items():
        profile = config["profiles"][profile_name]
        lines.extend([
            "",
            f"## {campaign}: {profile_name}",
            "",
            f"TTFT <= {profile['ttft_ms']} ms, TPOT <= {profile['tpot_ms']} ms, "
            f"joint attainment >= {100 * profile['required_joint_attainment']:.0f}% per repeat.",
            "",
            "| Strategy | Joint % | Every repeat >=95% | P99 pass | TTFT p95/p99 ms | TPOT p95/p99 ms | E2E budget p99 | W | J/token | Energy vs Max % | Pareto |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in items:
            lines.append(
                f"| {row['strategy']} | {100 * row['joint_attainment']:.2f} | "
                f"{row['every_repeat_joint_95_pass']} | {row['p99_all_metrics_pass']} | "
                f"{number(row['ttft_p95_ms'])}/{number(row['ttft_p99_ms'])} | "
                f"{number(row['tpot_p95_ms'])}/{number(row['tpot_p99_ms'])} | "
                f"{number(row['e2e_budget_ratio_p99'])}x | {number(row['avg_power_w'])} | "
                f"{number(row['energy_per_token_j'])} | {number(row['energy_delta_vs_max_clock_pct'])} | "
                f"{row['joint_slo_pareto']} |"
            )
        eligible = [row for row in items if row["every_repeat_joint_95_pass"]]
        if eligible:
            lowest = min(eligible, key=lambda item: item["total_energy_j"])
            lines.extend(["", f"Lowest observed fixed-work energy among eligible strategies: {lowest['strategy']} "
                          f"({number(lowest['energy_delta_vs_max_clock_pct'])}% vs MaxClock)."])
        else:
            lines.extend(["", "No strategy passes the declared joint-goodput gate in every repeat."])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = json.loads(config_path.read_text())
    if config["schema"] != "energyinfra-serving-slo-audit-v1":
        raise RuntimeError("Unsupported SLO audit schema")
    output = resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    provenance = {str(config_path.relative_to(ROOT)): file_hash(config_path),
                  str(Path(__file__).resolve().relative_to(ROOT)): file_hash(Path(__file__).resolve())}
    for campaign, relative in config["campaigns"].items():
        path = resolve(relative)
        groups = load_campaign(path)
        provenance[str(path.relative_to(ROOT) / "protocol.json")] = file_hash(path / "protocol.json")
        for profile_name, profile in config["profiles"].items():
            rows.extend(evaluate(campaign, groups, profile_name, profile))
    atomic(output / "protocol.json", {"schema": config["schema"], "config": config,
                                      "provenance": provenance})
    atomic(output / "summary.json", rows)
    if rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    render(output, config, rows)
    print(json.dumps({"report": str(output / "RESULTS.md"), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
