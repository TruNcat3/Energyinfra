#!/usr/bin/env python3
"""Analyze long Thor serving trials for thermal, power, latency and SLO drift."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def pct(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline in {None, 0.0}:
        return None
    return (value / baseline - 1.0) * 100.0


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def slope_per_minute(samples: list[tuple[float, float]]) -> float | None:
    if len(samples) < 2:
        return None
    origin = samples[0][0]
    xs = [item[0] - origin for item in samples]
    ys = [item[1] for item in samples]
    center_x, center_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((value - center_x) ** 2 for value in xs)
    if denominator == 0:
        return None
    return 60.0 * sum((x - center_x) * (y - center_y) for x, y in zip(xs, ys)) / denominator


def telemetry(path: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    powers: list[tuple[float, float]] = []
    temperatures: list[tuple[float, float]] = []
    with path.open() as handle:
        for line in handle:
            item = json.loads(line)
            timestamp = float(item["t"])
            if item.get("power_w") is not None:
                powers.append((timestamp, float(item["power_w"])))
            sensors = item.get("temperatures_c") or {
                name: float(value)
                for name, value in re.findall(
                    r"(?<!\S)([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C",
                    item.get("raw", ""),
                )
            }
            if sensors:
                temperatures.append((timestamp, max(float(value) for value in sensors.values())))
    return powers, temperatures


def window(samples: list[tuple[float, float]], start: float, end: float) -> list[tuple[float, float]]:
    return [item for item in samples if start <= item[0] <= end]


def average_power(samples: list[tuple[float, float]], start: float, end: float) -> float | None:
    if end <= start or len(samples) < 2:
        return None
    energy = 0.0
    covered = 0.0
    for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
        left, right = max(ta, start), min(tb, end)
        if right <= left or tb <= ta:
            continue
        va = pa + (pb - pa) * (left - ta) / (tb - ta)
        vb = pa + (pb - pa) * (right - ta) / (tb - ta)
        energy += (va + vb) * 0.5 * (right - left)
        covered += right - left
    return energy / covered if covered >= (end - start) * 0.99 else None


def latency_window(responses: list[dict[str, Any]], start: float, end: float, key: str) -> list[float]:
    return [
        float(response[key])
        for response in responses
        if start <= float(response["scheduled_time_s"]) < end and response.get(key) is not None
    ]


def trial_row(result: dict[str, Any], config: dict[str, Any], preset: str, run_dir: Path) -> dict[str, Any]:
    analysis = config["thermal_analysis"]
    settings = analysis["windows_by_preset"][preset]
    start = float(result["measurement_start_monotonic"])
    duration = float(result["duration_s"])
    end = start + duration
    warmup = float(settings["warmup_s"])
    width = float(settings["comparison_window_s"])
    early_start, early_end = start + warmup, start + warmup + width
    late_start, late_end = end - width, end
    telemetry_path = Path(result["telemetry_file"])
    if not telemetry_path.is_absolute():
        telemetry_path = run_dir / telemetry_path
    if not telemetry_path.is_file():
        raise FileNotFoundError(telemetry_path)
    powers, temperatures = telemetry(telemetry_path)
    measured_temperatures = window(temperatures, start, end)
    start_temperatures = window(temperatures, start, min(end, start + float(analysis["start_window_s"])))
    early_temperatures = window(temperatures, early_start, early_end)
    late_temperatures = window(temperatures, late_start, late_end)
    if not measured_temperatures or not early_temperatures or not late_temperatures:
        raise RuntimeError(f"Temperature windows are incomplete: {telemetry_path}")
    responses = result["responses"]
    early_tpot = latency_window(responses, warmup, warmup + width, "tpot_ms")
    late_tpot = latency_window(responses, duration - width, duration + 0.001, "tpot_ms")
    early_ttft = latency_window(responses, warmup, warmup + width, "ttft_end_to_end_ms")
    late_ttft = latency_window(responses, duration - width, duration + 0.001, "ttft_end_to_end_ms")
    early_power = average_power(powers, early_start, early_end)
    late_power = average_power(powers, late_start, late_end)
    early_tpot_median, late_tpot_median = median(early_tpot), median(late_tpot)
    row: dict[str, Any] = {
        "profile": result["profile"],
        "repeat": result["repeat"],
        "strategy": result["strategy"]["name"],
        "label": result["strategy"]["label"],
        "hardware": result["strategy"]["hardware"],
        "method_class": result["strategy"].get("method_class", ""),
        "duration_s": duration,
        "offered": result["offered"],
        "completed_exact": result["completed_exact"],
        "fixed_work_ok": result["comparable_fixed_work"],
        "avg_power_w": result["fixed_horizon"]["avg_power_w"],
        "total_energy_j": result["fixed_horizon"]["total_energy_j"],
        "energy_per_token_j": result["fixed_horizon"]["total_energy_j"] / result["completed_output_tokens"],
        "temperature_start_c": mean([value for _, value in start_temperatures]),
        "temperature_early_c": mean([value for _, value in early_temperatures]),
        "temperature_late_c": mean([value for _, value in late_temperatures]),
        "temperature_peak_c": max(value for _, value in measured_temperatures),
        "temperature_rise_c": mean([value for _, value in late_temperatures]) - mean([value for _, value in start_temperatures]),
        "temperature_late_slope_c_per_min": slope_per_minute(late_temperatures),
        "power_early_w": early_power,
        "power_late_w": late_power,
        "power_drift_pct": pct(late_power, early_power),
        "tpot_early_median_ms": early_tpot_median,
        "tpot_late_median_ms": late_tpot_median,
        "tpot_drift_pct": pct(late_tpot_median, early_tpot_median),
        "ttft_early_p95_ms": percentile(early_ttft, 0.95),
        "ttft_late_p95_ms": percentile(late_ttft, 0.95),
        "telemetry_samples": len(window(powers, start, end)),
        "telemetry_file": str(telemetry_path.relative_to(run_dir)),
    }
    for slo in config["slo_profiles"]:
        passed = sum(
            bool(response.get("completion_exact"))
            and response.get("ttft_end_to_end_ms") is not None
            and response.get("tpot_ms") is not None
            and float(response["ttft_end_to_end_ms"]) <= float(slo["ttft_ms"])
            and float(response["tpot_ms"]) <= float(slo["tpot_ms"])
            for response in responses
        )
        row[f"{slo['name']}_attainment"] = passed / len(responses) if responses else 0.0
    required = analysis["required_slo"]
    checks = {
        "fixed_work": bool(row["fixed_work_ok"]),
        "temperature_guard": row["temperature_peak_c"] < float(analysis["abort_temperature_c"]),
        "temperature_slope": row["temperature_late_slope_c_per_min"] is not None and row["temperature_late_slope_c_per_min"] <= float(analysis["max_positive_temperature_slope_c_per_min"]),
        "power_drift": row["power_drift_pct"] is not None and abs(row["power_drift_pct"]) <= float(analysis["max_absolute_power_drift_pct"]),
        "tpot_drift": row["tpot_drift_pct"] is not None and row["tpot_drift_pct"] <= float(analysis["max_positive_tpot_drift_pct"]),
        "slo": row[f"{required}_attainment"] >= float(config["required_attainment"]),
    }
    row["gate_checks"] = checks
    row["thermal_stability_pass"] = all(checks.values())
    return row


def aggregate(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["strategy"]].append(row)
    summary = []
    for strategy, group in groups.items():
        first = group[0]
        item = {
            "strategy": strategy,
            "label": first["label"],
            "hardware": first["hardware"],
            "method_class": first["method_class"],
            "repeats": len(group),
            "stable_repeats": sum(bool(row["thermal_stability_pass"]) for row in group),
            "all_repeats_stable": all(bool(row["thermal_stability_pass"]) for row in group),
            "avg_power_w": mean([float(row["avg_power_w"]) for row in group]),
            "energy_per_token_j": mean([float(row["energy_per_token_j"]) for row in group]),
            "temperature_peak_c": max(float(row["temperature_peak_c"]) for row in group),
            "temperature_rise_median_c": median([float(row["temperature_rise_c"]) for row in group]),
            "temperature_slope_median_c_per_min": median([float(row["temperature_late_slope_c_per_min"]) for row in group]),
            "power_drift_median_pct": median([float(row["power_drift_pct"]) for row in group]),
            "tpot_drift_median_pct": median([float(row["tpot_drift_pct"]) for row in group]),
        }
        for slo in config["slo_profiles"]:
            key = f"{slo['name']}_attainment"
            item[key] = mean([float(row[key]) for row in group])
            item[f"{slo['name']}_all_repeats_feasible"] = all(
                float(row[key]) >= float(config["required_attainment"]) for row in group
            )
        summary.append(item)
    native = next((row for row in summary if row["method_class"] == "native"), None)
    maximum = next((row for row in summary if row["method_class"] == "race_to_idle"), None)
    for row in summary:
        row["power_delta_vs_native_pct"] = pct(row["avg_power_w"], native["avg_power_w"] if native else None)
        row["energy_delta_vs_native_pct"] = pct(row["energy_per_token_j"], native["energy_per_token_j"] if native else None)
        row["power_delta_vs_max_clock_pct"] = pct(row["avg_power_w"], maximum["avg_power_w"] if maximum else None)
        row["energy_delta_vs_max_clock_pct"] = pct(row["energy_per_token_j"], maximum["energy_per_token_j"] if maximum else None)
    return summary


def fmt(value: Any, digits: int = 2) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [key for key in rows[0] if key != "gate_checks"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def report(summary: list[dict[str, Any]], rows: list[dict[str, Any]], config: dict[str, Any], preset: str) -> str:
    lines = [
        "# Thor thermal stability v1",
        "",
        f"- Preset: `{preset}`",
        f"- Trials: `{len(rows)}`",
        f"- Required SLO: `{config['thermal_analysis']['required_slo']}`",
        "- A formal result requires every repeat to pass fixed work, SLO, temperature, slope, power-drift and TPOT-drift gates.",
        "",
        "| Strategy | Stable repeats | W | J/token | Peak C | Rise C | Slope C/min | Power drift % | TPOT drift % | Power vs native % | Energy vs native % | Power vs MaxClock % | Energy vs MaxClock % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['label']} | {row['stable_repeats']}/{row['repeats']} | "
            f"{fmt(row['avg_power_w'])} | {fmt(row['energy_per_token_j'], 3)} | "
            f"{fmt(row['temperature_peak_c'])} | {fmt(row['temperature_rise_median_c'])} | "
            f"{fmt(row['temperature_slope_median_c_per_min'], 3)} | {fmt(row['power_drift_median_pct'])} | "
            f"{fmt(row['tpot_drift_median_pct'])} | {fmt(row['power_delta_vs_native_pct'])} | "
            f"{fmt(row['energy_delta_vs_native_pct'])} | {fmt(row['power_delta_vs_max_clock_pct'])} | "
            f"{fmt(row['energy_delta_vs_max_clock_pct'])} |"
        )
    lines += [
        "",
        "A lower average power value is not sufficient for a win. A candidate is thermally confirmed only when `all_repeats_stable` is true and its declared SLO remains feasible.",
        "",
        "Per-trial values and individual gate outcomes are retained in `thermal_trials.json` and `trial_thermal.csv`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/thor_thermal_stability_v1.json")
    parser.add_argument("--preset", choices=("quick", "full"), required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config_path = resolve(args.config).resolve()
    config = load(config_path)
    run_dir = resolve(args.run_dir).resolve()
    expected = [
        run_dir / "results" / f"result_{profile}_{repeat}_{strategy['name']}.json"
        for repeat in range(int(config["presets"][args.preset]["repeats"]))
        for profile in config["presets"][args.preset]["profiles"]
        for strategy in config["strategies"]
    ]
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(f"Thermal campaign is incomplete; missing {len(missing)} result files")
    rows = [trial_row(load(path), config, args.preset, run_dir) for path in expected]
    summary = aggregate(rows, config)
    write_csv(run_dir / "trial_thermal.csv", rows)
    write_csv(run_dir / "thermal_summary.csv", summary)
    (run_dir / "thermal_trials.json").write_text(json.dumps(rows, indent=2) + "\n")
    manifest = {
        "schema": "energyinfra-thor-thermal-analysis-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "trials": len(rows),
        "strategies": len(summary),
    }
    (run_dir / "thermal_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (run_dir / "THERMAL_RESULTS.md").write_text(report(summary, rows, config, args.preset))
    print(json.dumps({"trials": len(rows), "output": str(run_dir / "THERMAL_RESULTS.md")}, indent=2))


if __name__ == "__main__":
    main()
