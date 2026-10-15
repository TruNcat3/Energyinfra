#!/usr/bin/env python3
"""Build a workload-by-workload operating-mode report from validated Thor data."""

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
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_multiload_modes_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_workload(value: Any) -> tuple[int, int]:
    if isinstance(value, list):
        return int(value[0]), int(value[1])
    if isinstance(value, str) and value.startswith("["):
        parsed = json.loads(value)
        return int(parsed[0]), int(parsed[1])
    prompt, output = str(value).split("/")
    return int(prompt), int(output)


def number(value: Any) -> float:
    return float(value)


def normalized(row: dict[str, Any], provenance: str) -> dict[str, Any]:
    return {
        "workload": list(parse_workload(row["workload"])),
        "point": row["point"],
        "provenance": provenance,
        "blocks": int(row["blocks"]),
        "requests": int(row["requests"]),
        "avg_power_w": number(row["avg_power_w"]),
        "energy_per_token_j": number(row["energy_per_token_j"]),
        "ttft_ms": number(row["ttft_ms"]),
        "tpot_ms": number(row["tpot_ms"]),
        "ttft_p95_ms": number(row["ttft_p95_ms"]),
        "tpot_p95_ms": number(row["tpot_p95_ms"]),
    }


def load_sources(validated_path: Path, fallback_path: Path) -> dict[tuple[tuple[int, int], str], dict[str, Any]]:
    table: dict[tuple[tuple[int, int], str], dict[str, Any]] = {}
    with fallback_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            item = normalized(row, "static_3_block")
            table[(tuple(item["workload"]), item["point"])] = item
    for row in load_json(validated_path):
        if not row.get("complete", False):
            continue
        item = normalized(row, "validated_20_block")
        table[(tuple(item["workload"]), item["point"])] = item
    return table


def delta(value: float, baseline: float) -> float:
    return (value / baseline - 1.0) * 100.0


def geometric_delta(values: list[float]) -> float:
    return (math.exp(sum(math.log(1.0 + value / 100.0) for value in values) / len(values)) - 1.0) * 100.0


def dominates(left: dict[str, Any], right: dict[str, Any], objectives: tuple[str, ...]) -> bool:
    return all(left[key] <= right[key] for key in objectives) and any(
        left[key] < right[key] for key in objectives
    )


def analyze(config: dict[str, Any], table: dict[tuple[tuple[int, int], str], dict[str, Any]]) -> dict[str, Any]:
    modes = {mode["name"]: mode for mode in config["modes"]}
    baseline_name = config["baseline"]
    rows = []
    missing = []
    for raw_workload in config["workloads"]:
        workload = tuple(int(value) for value in raw_workload)
        selected: list[dict[str, Any]] = []
        for mode in config["modes"]:
            source = table.get((workload, mode["point"]))
            if source is None:
                missing.append({"workload": list(workload), "mode": mode["name"], "point": mode["point"]})
                continue
            item = dict(source)
            item.update({"mode": mode["name"], "label": mode["label"], "intent": mode["intent"]})
            item["slo"] = {
                profile["name"]: (
                    item["ttft_p95_ms"] <= profile["ttft_ms"]
                    and item["tpot_p95_ms"] <= profile["tpot_ms"]
                )
                for profile in config["slo_profiles"]
            }
            selected.append(item)
        baseline = next((item for item in selected if item["mode"] == baseline_name), None)
        if baseline is None:
            continue
        for item in selected:
            item["delta_vs_energy_battery_pct"] = {
                "energy_per_token_j": delta(item["energy_per_token_j"], baseline["energy_per_token_j"]),
                "avg_power_w": delta(item["avg_power_w"], baseline["avg_power_w"]),
                "ttft_p95_ms": delta(item["ttft_p95_ms"], baseline["ttft_p95_ms"]),
                "tpot_p95_ms": delta(item["tpot_p95_ms"], baseline["tpot_p95_ms"]),
            }
            item["pareto_energy_latency"] = not any(
                other is not item
                and dominates(other, item, ("energy_per_token_j", "ttft_p95_ms", "tpot_p95_ms"))
                for other in selected
            )
            item["pareto_power_latency"] = not any(
                other is not item
                and dominates(other, item, ("avg_power_w", "ttft_p95_ms", "tpot_p95_ms"))
                for other in selected
            )
            item["pareto_all_objectives"] = not any(
                other is not item
                and dominates(
                    other,
                    item,
                    ("energy_per_token_j", "avg_power_w", "ttft_p95_ms", "tpot_p95_ms"),
                )
                for other in selected
            )
            rows.append(item)

    aggregate = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)
    for name, mode_rows in grouped.items():
        changes = [row["delta_vs_energy_battery_pct"] for row in mode_rows]
        aggregate.append({
            "mode": name,
            "label": modes[name]["label"],
            "point": modes[name]["point"],
            "workloads": len(mode_rows),
            "validated_workloads": sum(row["provenance"] == "validated_20_block" for row in mode_rows),
            "geomean_delta_vs_energy_battery_pct": {
                metric: geometric_delta([change[metric] for change in changes])
                for metric in ("energy_per_token_j", "avg_power_w", "ttft_p95_ms", "tpot_p95_ms")
            },
            "slo_passed_workloads": {
                profile["name"]: sum(row["slo"][profile["name"]] for row in mode_rows)
                for profile in config["slo_profiles"]
            },
            "energy_latency_pareto_workloads": sum(row["pareto_energy_latency"] for row in mode_rows),
            "power_latency_pareto_workloads": sum(row["pareto_power_latency"] for row in mode_rows),
            "all_objective_pareto_workloads": sum(row["pareto_all_objectives"] for row in mode_rows),
        })
    aggregate.sort(key=lambda row: [mode["name"] for mode in config["modes"]].index(row["mode"]))
    return {"rows": rows, "aggregate": aggregate, "missing": missing}


def fmt_delta(value: float) -> str:
    return f"{value:+.2f}%"


def render(result: dict[str, Any], config: dict[str, Any], provenance: dict[str, Any]) -> str:
    lines = [
        "# Thor multi-load operating modes v1",
        "",
        "This report reuses the completed 12-workload static sweep and prospective validation. "
        "It does not rerun hardware or infer model quality.",
        "",
        "## Aggregate comparison",
        "",
        "Geometric-mean changes are relative to `Energy/Battery` for the same workload.",
        "",
        "| Mode | Hardware point | Validated loads | Energy/token | Power | TTFT p95 | TPOT p95 | Strict 500/30 | 1000/50 | Energy-latency Pareto | Power-latency Pareto |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    total_workloads = len(config["workloads"])
    for row in result["aggregate"]:
        changes = row["geomean_delta_vs_energy_battery_pct"]
        passes = row["slo_passed_workloads"]
        lines.append(
            f"| {row['label']} | `{row['point']}` | {row['validated_workloads']}/{total_workloads} | "
            f"{fmt_delta(changes['energy_per_token_j'])} | {fmt_delta(changes['avg_power_w'])} | "
            f"{fmt_delta(changes['ttft_p95_ms'])} | {fmt_delta(changes['tpot_p95_ms'])} | "
            f"{passes['interactive_strict']}/{total_workloads} | {passes['interactive']}/{total_workloads} | "
            f"{row['energy_latency_pareto_workloads']}/{total_workloads} | "
            f"{row['power_latency_pareto_workloads']}/{total_workloads} |"
        )

    lines += ["", "## Per-workload comparison", ""]
    by_workload: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in result["rows"]:
        by_workload[tuple(row["workload"])].append(row)
    order = [mode["name"] for mode in config["modes"]]
    for workload in config["workloads"]:
        key = tuple(workload)
        lines += [
            f"### {key[0]}/{key[1]}",
            "",
            "| Mode | W | J/token | Energy delta | Power delta | TTFT p95 ms | TPOT p95 ms | 500/30 | 1000/50 | E-L Pareto | P-L Pareto | Source |",
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---|",
        ]
        for row in sorted(by_workload.get(key, []), key=lambda item: order.index(item["mode"])):
            changes = row["delta_vs_energy_battery_pct"]
            lines.append(
                f"| {row['label']} | {row['avg_power_w']:.3f} | {row['energy_per_token_j']:.4f} | "
                f"{fmt_delta(changes['energy_per_token_j'])} | {fmt_delta(changes['avg_power_w'])} | "
                f"{row['ttft_p95_ms']:.2f} | {row['tpot_p95_ms']:.2f} | "
                f"{'pass' if row['slo']['interactive_strict'] else 'fail'} | "
                f"{'pass' if row['slo']['interactive'] else 'fail'} | "
                f"{'yes' if row['pareto_energy_latency'] else 'no'} | "
                f"{'yes' if row['pareto_power_latency'] else 'no'} | {row['provenance']} |"
            )
        lines.append("")

    lines += [
        "## Interpretation rules",
        "",
        "- Lower `J/token` is the fixed-work battery/energy objective.",
        "- Lower `W` is a thermal, current, or platform power-budget objective; it does not by itself extend battery life.",
        "- Energy-latency and power-latency Pareto sets are reported separately because their user preferences differ.",
        "- Static request SLO feasibility does not include queueing; frozen-trace serving remains the authority for online goodput.",
        "- Random weights preserve model shape and kernels but do not support output-quality conclusions.",
        "",
        "## Provenance",
        "",
        f"- Validated source SHA256: `{provenance['validated_sha256']}`",
        f"- Static fallback SHA256: `{provenance['fallback_sha256']}`",
        f"- Missing requested points: `{len(result['missing'])}`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config_path = resolve(args.config)
    config = load_json(config_path)
    validated_path = resolve(config["validated_source"])
    fallback_path = resolve(config["static_fallback"])
    output_dir = resolve(args.output_dir or config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    table = load_sources(validated_path, fallback_path)
    result = analyze(config, table)
    provenance = {
        "schema": "energyinfra-thor-multiload-modes-report-v1",
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "validated_source": str(validated_path),
        "validated_sha256": sha256(validated_path),
        "fallback_source": str(fallback_path),
        "fallback_sha256": sha256(fallback_path),
    }
    atomic_json(output_dir / "summary.json", {**provenance, **result})
    (output_dir / "RESULTS.md").write_text(render(result, config, provenance))
    print(json.dumps({
        "output": str(output_dir),
        "workload_rows": len(result["rows"]),
        "missing": len(result["missing"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
