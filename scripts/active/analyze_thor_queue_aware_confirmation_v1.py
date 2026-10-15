#!/usr/bin/env python3
"""Combine initial and confirmation blocks for paired queue-policy analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_queue_aware_confirmation_analysis_v1.json"


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


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def t_critical_95(n: int) -> float:
    table = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
             8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201}
    return table.get(n, 1.960)


def interval(values: list[float]) -> dict[str, Any]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = t_critical_95(len(values)) * math.sqrt(variance / len(values))
    return {"n": len(values), "mean_pct": mean, "ci95_low_pct": mean - margin,
            "ci95_high_pct": mean + margin, "values_pct": values}


def request_pass(response: dict[str, Any], profile: dict[str, Any]) -> bool:
    if response.get("dropped") or not response.get("metrics"):
        return False
    request = response["request"]
    metrics = response["metrics"]
    output = int(request["output_length"])
    e2e_limit = float(profile["ttft_limit_ms"]) + float(profile["tpot_limit_ms"]) * max(0, output - 1)
    return bool(
        int(metrics.get("output_tokens") or 0) >= output
        and float(metrics.get("ttft_end_to_end_ms") or math.inf) <= float(profile["ttft_limit_ms"])
        and float(metrics.get("tpot_ms") or math.inf) <= float(profile["tpot_limit_ms"])
        and float(metrics.get("arrival_to_completion_ms") or math.inf) <= e2e_limit
    )


def collect(config: dict[str, Any]) -> tuple[dict[tuple[str, int], dict[str, dict[str, Any]]], dict[str, str]]:
    wanted = set(config["strategies"])
    blocks: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    provenance = {}
    for campaign in config["campaigns"]:
        directory = resolve(campaign["path"])
        for path in sorted(directory.glob("result_*_*.json")):
            result = load(path)
            if result["strategy"] not in wanted:
                continue
            key = (campaign["name"], int(result["repeat"]))
            blocks[key][result["strategy"]] = result
            provenance[str(path.relative_to(ROOT))] = file_hash(path)
    expected = set(config["strategies"])
    incomplete = {str(key): sorted(expected - set(value)) for key, value in blocks.items() if set(value) != expected}
    if incomplete:
        raise RuntimeError(f"incomplete paired blocks: {incomplete}")
    for key, values in blocks.items():
        hashes = {value["trace_sha256"] for value in values.values()}
        if len(hashes) != 1:
            raise RuntimeError(f"trace mismatch in block {key}")
    return dict(sorted(blocks.items())), provenance


def analyze(config_path: Path) -> Path:
    config = load(config_path)
    blocks, provenance = collect(config)
    required = float(config["required_observed_attainment"])
    strategy_runs = {
        strategy: [values[strategy] for values in blocks.values()]
        for strategy in config["strategies"]
    }
    profile_results = []
    for profile in config["profiles"]:
        rows = []
        for strategy, runs in strategy_runs.items():
            responses = [response for run in runs for response in run["responses"]]
            repeat_attainment = [
                sum(request_pass(response, profile) for response in run["responses"]) / len(run["responses"])
                for run in runs
            ]
            metrics = [response["metrics"] for response in responses if not response.get("dropped")]
            energy = sum(float(run["total_energy_j"]) for run in runs)
            duration = sum(float(run["duration_with_drain_s"]) for run in runs)
            tokens = sum(int(run["output_tokens"]) for run in runs)
            rows.append({
                "strategy": strategy,
                "blocks": len(runs),
                "offered": len(responses),
                "completed": sum(int(run["completed"]) for run in runs),
                "joint_goodput": sum(request_pass(response, profile) for response in responses) / len(responses),
                "repeat_attainment": repeat_attainment,
                "all_repeats_pass": all(value >= required for value in repeat_attainment),
                "ttft_p95_ms": percentile([float(item["ttft_end_to_end_ms"]) for item in metrics], 0.95),
                "ttft_p99_ms": percentile([float(item["ttft_end_to_end_ms"]) for item in metrics], 0.99),
                "tpot_p95_ms": percentile([float(item["tpot_ms"]) for item in metrics], 0.95),
                "tpot_p99_ms": percentile([float(item["tpot_ms"]) for item in metrics], 0.99),
                "total_energy_j": energy,
                "avg_power_w": energy / duration,
                "energy_per_token_j": energy / tokens,
            })
        profile_results.append({"profile": profile, "rows": rows})

    candidate = config["candidate"]
    paired = {}
    for baseline in config["baselines"]:
        values = [
            100.0 * (runs[candidate]["total_energy_j"] / runs[baseline]["total_energy_j"] - 1.0)
            for runs in blocks.values()
        ]
        paired[f"{candidate}_vs_{baseline}"] = interval(values)
    payload = {
        "schema": config["schema"],
        "statistical_scope": "10 paired blocks covering run noise and frozen request-order variation",
        "blocks": [{"campaign": key[0], "repeat": key[1]} for key in blocks],
        "paired_energy": paired,
        "profiles": profile_results,
        "provenance": provenance,
    }
    output = resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "confirmation.json"
    atomic(result_path, payload)

    lines = [
        "# Thor queue-aware confirmation",
        "",
        f"Paired blocks: {len(blocks)}.",
        "The interval covers repeated hardware runs and deterministic request-order variants; it is not a population guarantee.",
        "",
        "## Paired fixed-work energy",
        "",
        "| Comparison | Mean | 95% CI |",
        "|---|---:|---:|",
    ]
    for name, value in paired.items():
        lines.append(f"| {name} | {value['mean_pct']:+.3f}% | [{value['ci95_low_pct']:+.3f}%, {value['ci95_high_pct']:+.3f}%] |")
    for item in profile_results:
        profile = item["profile"]
        lines.extend([
            "",
            f"## {profile['name']} ({profile['ttft_limit_ms']:.0f}/{profile['tpot_limit_ms']:.0f} ms)",
            "",
            "| Strategy | Joint | Every block | TTFT p95/p99 | TPOT p95/p99 | W | J/token |",
            "|---|---:|:---:|---:|---:|---:|---:|",
        ])
        for row in item["rows"]:
            lines.append(
                f"| {row['strategy']} | {100 * row['joint_goodput']:.2f}% | {row['all_repeats_pass']} | "
                f"{row['ttft_p95_ms']:.1f}/{row['ttft_p99_ms']:.1f} | "
                f"{row['tpot_p95_ms']:.1f}/{row['tpot_p99_ms']:.1f} | "
                f"{row['avg_power_w']:.3f} | {row['energy_per_token_j']:.3f} |"
            )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "report": str(output / "RESULTS.md")}))
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    analyze(args.config.resolve())


if __name__ == "__main__":
    main()
