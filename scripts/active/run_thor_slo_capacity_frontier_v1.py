#!/usr/bin/env python3
"""Build and run an SLO-specific serving-capacity frontier on Thor."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.active import run_thor_joint_serving_stage1b_v1 as serving


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_slo_capacity_frontier_v1.json"
PHASES = ("pilot", "refine", "confirm", "strict_confirm")


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def immutable(path: Path, value: Any) -> None:
    if path.exists() and load(path) != value:
        raise RuntimeError(f"Protocol artifact changed; use a new output directory: {path}")
    if not path.exists():
        serving.atomic(path, value)


def hkey(hardware: dict[str, Any]) -> str:
    return json.dumps(hardware, sort_keys=True, separators=(",", ":"))


def normalized(values: list[float], value: float) -> float:
    low, high = min(values), max(values)
    return 0.0 if high == low else (value - low) / (high - low)


def static_candidates(frontier: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    table_path = resolve(frontier["static_table"])
    validation_path = resolve(frontier["static_validation"])
    workload_weights = {
        (int(item["prompt_tokens"]), int(item["output_tokens"])): int(item["weight"])
        for item in base["workload_mix"]
    }
    grouped: dict[str, dict[str, Any]] = {}
    with table_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for source_index, row in enumerate(rows):
        workload = tuple(int(value) for value in json.loads(row["workload"]))
        if workload not in workload_weights:
            continue
        hardware = json.loads(row["hardware"])
        if hardware.get("mode") != "cap":
            continue
        key = hkey(hardware)
        value = grouped.setdefault(key, {
            "hardware": hardware,
            "points": set(),
            "workloads": {},
            "source_rows": [],
        })
        value["points"].add(row["point"])
        value["source_rows"].append(source_index)
        value["workloads"][workload] = {
            name: float(row[name])
            for name in ("avg_power_w", "energy_per_token_j", "ttft_p95_ms", "tpot_p95_ms")
        }
    candidates = []
    for value in grouped.values():
        if set(value["workloads"]) != set(workload_weights):
            continue
        request_weight = sum(workload_weights.values())
        output_weight = sum(workload_weights[w] * w[1] for w in workload_weights)
        metrics = {
            "avg_power_w": sum(
                workload_weights[w] * value["workloads"][w]["avg_power_w"]
                for w in workload_weights
            ) / request_weight,
            "energy_per_token_j": sum(
                workload_weights[w] * w[1] * value["workloads"][w]["energy_per_token_j"]
                for w in workload_weights
            ) / output_weight,
            "ttft_p95_ms": sum(
                workload_weights[w] * value["workloads"][w]["ttft_p95_ms"]
                for w in workload_weights
            ) / request_weight,
            "tpot_p95_ms": sum(
                workload_weights[w] * w[1] * value["workloads"][w]["tpot_p95_ms"]
                for w in workload_weights
            ) / output_weight,
        }
        candidates.append({
            "hardware": value["hardware"],
            "source_points": sorted(value["points"]),
            "source_rows": value["source_rows"],
            "metrics": metrics,
            "roles": [],
        })
    if not candidates:
        raise RuntimeError("No complete cap candidates match the serving workload mix")
    objectives = ("avg_power_w", "energy_per_token_j", "ttft_p95_ms", "tpot_p95_ms")
    for candidate in candidates:
        candidate["static_pareto"] = not any(
            all(other["metrics"][key] <= candidate["metrics"][key] for key in objectives)
            and any(other["metrics"][key] < candidate["metrics"][key] for key in objectives)
            for other in candidates if other is not candidate
        )
    role_choice = {
        "power_min": min(candidates, key=lambda item: item["metrics"]["avg_power_w"]),
        "energy_min": min(candidates, key=lambda item: item["metrics"]["energy_per_token_j"]),
        "latency_min": min(
            candidates,
            key=lambda item: item["metrics"]["ttft_p95_ms"] + item["metrics"]["tpot_p95_ms"],
        ),
    }
    pareto = [candidate for candidate in candidates if candidate["static_pareto"]]
    columns = {key: [item["metrics"][key] for item in pareto] for key in objectives}
    role_choice["knee"] = min(
        pareto,
        key=lambda item: math.sqrt(sum(
            normalized(columns[key], item["metrics"][key]) ** 2 for key in objectives
        )),
    )
    queue = base["hardware_points"]["queue_power_min"]
    role_choice["queue_reference"] = min(
        candidates,
        key=lambda item: sum(
            item["hardware"].get(key) != queue.get(key)
            for key in ("gpu_hz", "emc_hz", "cpu_khz", "cpu_floor_khz")
        ),
    )
    selected = []
    by_key: dict[str, dict[str, Any]] = {}
    for role in frontier["selection_roles"]:
        candidate = role_choice[role]
        key = hkey(candidate["hardware"])
        if key not in by_key:
            if len(selected) >= int(frontier["candidate_limit"]):
                continue
            copy = {**candidate, "roles": []}
            selected.append(copy)
            by_key[key] = copy
        by_key[key]["roles"].append(role)
    for index, candidate in enumerate(selected, 1):
        hardware = dict(candidate["hardware"])
        if int(hardware.get("cpu_floor_khz", 1)) <= 0:
            hardware.pop("cpu_floor_khz", None)
        candidate["hardware_for_run"] = hardware
        candidate["name"] = (
            f"static_c{index}_g{int(hardware['gpu_hz']) // 1000000}_"
            f"e{int(hardware['emc_hz']) // 1000000}_f{int(hardware.get('cpu_floor_khz', 0))}"
        )
    return {
        "schema": "energyinfra-thor-capacity-candidates-v1",
        "static_table": str(table_path),
        "static_table_sha256": serving.file_hash(table_path),
        "static_validation": str(validation_path),
        "static_validation_sha256": serving.file_hash(validation_path),
        "source_rows": len(rows),
        "complete_cap_candidates": len(candidates),
        "workload_weights": {f"{p}x{o}": weight for (p, o), weight in workload_weights.items()},
        "role_assignments": {role: hkey(candidate["hardware"]) for role, candidate in role_choice.items()},
        "selected": selected,
        "all_candidates": candidates,
    }


def strategy_space(frontier: dict[str, Any], base: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    baseline_names = set(frontier["baseline_strategies"])
    strategies = [dict(item) for item in base["strategies"] if item["name"] in baseline_names]
    if {item["name"] for item in strategies} != baseline_names:
        raise RuntimeError("A requested Stage 1B baseline is missing")
    parallels = [int(value) for value in frontier["parallel_by_role"]["default"]]
    for candidate in catalog["selected"]:
        role_label = "+".join(candidate["roles"])
        for parallel in parallels:
            strategies.append({
                "name": f"joint_{candidate['name']}_p{parallel}",
                "label": f"Joint {candidate['name']} P{parallel} ({role_label})",
                "hardware": candidate["name"],
                "parallel": parallel,
                "batch": 512 if parallel == 8 else 256,
                "ubatch": 256,
                "method_class": "joint_static_frontier",
            })
    return strategies


def phase_dir(frontier: dict[str, Any], phase: str) -> Path:
    return resolve(frontier["output_dir"]) / phase


def summary_path(frontier: dict[str, Any], phase: str) -> Path:
    return phase_dir(frontier, phase) / "serving" / "summary.json"


def experiment_path(frontier: dict[str, Any], phase: str) -> Path:
    return phase_dir(frontier, phase) / "experiment.json"


def rate_for(profile: str) -> int:
    return int(profile.rsplit("_", 1)[1])


def prior_rows(frontier: dict[str, Any], phases: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for phase in phases:
        path = summary_path(frontier, phase)
        if path.exists():
            rows.extend(load(path))
    return rows


def feasibility_by_rate(frontier: dict[str, Any], phases: tuple[str, ...]) -> dict[str, dict[int, bool]]:
    base = load(resolve(frontier["base_serving_config"]))
    observations: dict[str, dict[int, bool]] = {slo["name"]: {} for slo in base["slo_profiles"]}
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_rows(frontier, phases):
        grouped[rate_for(row["profile"])].append(row)
    for slo in base["slo_profiles"]:
        for rate, rows in grouped.items():
            observations[slo["name"]][rate] = any(
                row["slo"][slo["name"]]["all_repeats_feasible"] for row in rows
            )
    return observations


def midpoint(low: int, high: int, quantum: int) -> int | None:
    value = int(round(((low + high) / 2) / quantum) * quantum)
    if value <= low:
        value = low + quantum
    if value >= high:
        value = high - quantum
    return value if low < value < high else None


def phase_rates(frontier: dict[str, Any], phase: str) -> list[int]:
    if phase == "pilot":
        return [int(value) for value in frontier["pilot"]["offered_output_tps"]]
    if phase == "strict_confirm":
        return [int(value) for value in frontier["strict_confirm"]["offered_output_tps"]]
    previous = ("pilot",) if phase == "refine" else ("pilot", "refine")
    missing = [name for name in previous if not summary_path(frontier, name).exists()]
    if missing:
        raise RuntimeError(f"Missing prior summaries for {phase}: {missing}")
    observed = feasibility_by_rate(frontier, previous)
    if phase == "refine":
        cfg = frontier["refine"]
        quantum = int(cfg["rate_quantum_tps"])
        rates = set()
        for values in observed.values():
            feasible = sorted(rate for rate, state in values.items() if state)
            low = max(feasible) if feasible else int(cfg["minimum_rate_tps"])
            infeasible = sorted(rate for rate, state in values.items() if not state and rate > low)
            high = min(infeasible) if infeasible else int(cfg["maximum_rate_tps"])
            value = midpoint(low, high, quantum)
            if value is not None and value not in values:
                rates.add(value)
        if not rates:
            raise RuntimeError("No unresolved capacity bracket remains after pilot")
        return sorted(rates)
    rates = set()
    for values in observed.values():
        feasible = sorted(rate for rate, state in values.items() if state)
        infeasible = sorted(rate for rate, state in values.items() if not state)
        if feasible:
            low = max(feasible)
            rates.add(low)
            higher = [rate for rate in infeasible if rate > low]
            if higher:
                rates.add(min(higher))
        elif infeasible:
            rates.add(min(infeasible))
    if not rates:
        raise RuntimeError("No confirmation rates can be derived")
    return sorted(rates)


def confirmation_strategies(frontier: dict[str, Any], strategies: list[dict[str, Any]], rates: list[int]) -> list[dict[str, Any]]:
    rows = prior_rows(frontier, ("pilot", "refine"))
    base = load(resolve(frontier["base_serving_config"]))
    baseline = set(frontier["baseline_strategies"])
    score: dict[str, int] = defaultdict(int)
    threshold = float(frontier["confirm"]["screen_keep_attainment"])
    for row in rows:
        if rate_for(row["profile"]) not in rates:
            continue
        for slo in base["slo_profiles"]:
            result = row["slo"][slo["name"]]
            if result["attainment"] >= threshold or result["pareto"]:
                score[row["strategy"]] += 1
    names = list(baseline)
    ranked = sorted(
        (name for name in score if name not in baseline),
        key=lambda name: (-score[name], name),
    )
    names.extend(ranked[:max(0, int(frontier["confirm"]["maximum_strategies"]) - len(names))])
    selected = [item for item in strategies if item["name"] in set(names)]
    if not any(item["method_class"] == "joint_static_frontier" for item in selected):
        joint = next(item for item in strategies if item["method_class"] == "joint_static_frontier")
        selected.append(joint)
    return selected


def build_experiment(frontier_path: Path, phase: str) -> tuple[dict[str, Any], Path, Path]:
    frontier = load(frontier_path)
    if frontier.get("schema") != "energyinfra-thor-slo-capacity-frontier-v1":
        raise RuntimeError("Unsupported capacity-frontier schema")
    base_path = resolve(frontier["base_serving_config"])
    base = load(base_path)
    directory = phase_dir(frontier, phase)
    directory.mkdir(parents=True, exist_ok=True)
    catalog_path = resolve(frontier["output_dir"]) / "candidate_catalog.json"
    if phase == "strict_confirm":
        if not catalog_path.exists():
            raise RuntimeError(f"Missing frozen candidate catalog: {catalog_path}")
        catalog = load(catalog_path)
    else:
        catalog = static_candidates(frontier, base)
        catalog.update({
            "frontier_config": str(frontier_path),
            "frontier_config_sha256": serving.file_hash(frontier_path),
            "base_serving_config": str(base_path),
            "base_serving_config_sha256": serving.file_hash(base_path),
            "controller_source_sha256": serving.file_hash(Path(__file__).resolve()),
        })
        immutable(catalog_path, catalog)
    rates = phase_rates(frontier, phase)
    strategies = strategy_space(frontier, base, catalog)
    if phase in ("confirm", "strict_confirm"):
        strategies = confirmation_strategies(frontier, strategies, rates)
    phase_cfg = frontier[phase]
    serving_dir = directory / "serving"
    experiment = {
        key: base[key]
        for key in (
            "joint_resource_config", "hardware_config", "resource_screen_summary", "runtime",
            "server", "workload_mix", "slo_profiles", "required_attainment", "related_work",
        )
    }
    hardware_points = {
        key: dict(value) for key, value in base["hardware_points"].items()
    }
    for candidate in catalog["selected"]:
        hardware_points[candidate["name"]] = candidate["hardware_for_run"]
    experiment.update({
        "schema": "energyinfra-thor-joint-serving-stage1b-v2",
        "seed": int(base["seed"]) + {
            "pilot": 2000,
            "refine": 3000,
            "confirm": 4000,
            "strict_confirm": 5000,
        }[phase],
        "output_dir": str(serving_dir.relative_to(ROOT)),
        "presets": {
            phase: {
                "duration_s": int(phase_cfg["duration_s"]),
                "repeats": int(phase_cfg["repeats"]),
                "profiles": [f"rate_{rate:03d}" for rate in rates],
            }
        },
        "load_profiles": {
            f"rate_{rate:03d}": {"offered_output_tps": float(rate)} for rate in rates
        },
        "hardware_points": hardware_points,
        "strategies": strategies,
        "capacity_provenance": {
            "phase": phase,
            "rates": rates,
            "candidate_catalog_sha256": serving.object_hash(catalog),
            "frontier_config_sha256": serving.file_hash(frontier_path),
            "stage1b_anchor_summary": frontier["stage1b_full_summary"],
        },
    })
    target = experiment_path(frontier, phase)
    immutable(target, experiment)
    return frontier, target, serving_dir


def analyze(frontier_path: Path, phase: str) -> None:
    frontier = load(frontier_path)
    derived = load(experiment_path(frontier, phase))
    rows = load(summary_path(frontier, phase))
    report = {"schema": "energyinfra-thor-capacity-analysis-v1", "phase": phase, "slos": []}
    lines = [f"# Thor SLO capacity frontier: {phase}", ""]
    for slo in derived["slo_profiles"]:
        name = slo["name"]
        rates = []
        for profile in derived["presets"][phase]["profiles"]:
            rate = rate_for(profile)
            selected = [row for row in rows if row["profile"] == profile]
            feasible = [row for row in selected if row["slo"][name]["all_repeats_feasible"]]
            pareto = [row["strategy"] for row in feasible if row["slo"][name]["pareto"]]
            rates.append({
                "offered_output_tps": rate,
                "feasible": bool(feasible),
                "feasible_strategies": [row["strategy"] for row in feasible],
                "pareto_strategies": pareto,
            })
        feasible_rates = [item["offered_output_tps"] for item in rates if item["feasible"]]
        capacity = max(feasible_rates) if feasible_rates else None
        report["slos"].append({"slo": slo, "screen_capacity_tps": capacity, "rates": rates})
        lines += [
            f"## {name} ({slo['ttft_ms']}/{slo['tpot_ms']} ms)", "",
            f"Screen capacity: **{capacity if capacity is not None else 'none'} output-token/s**", "",
            "| Offered tps | Feasible | Feasible strategies | Pareto strategies |",
            "|---:|---:|---|---|",
        ]
        for item in rates:
            lines.append(
                f"| {item['offered_output_tps']} | {'yes' if item['feasible'] else 'no'} | "
                f"{', '.join(item['feasible_strategies']) or '--'} | "
                f"{', '.join(item['pareto_strategies']) or '--'} |"
            )
        lines.append("")
    if phase in ("pilot", "refine"):
        try:
            report["next_rates"] = phase_rates(frontier, "refine" if phase == "pilot" else "confirm")
        except RuntimeError as exc:
            report["next_rates_error"] = str(exc)
    directory = phase_dir(frontier, phase)
    serving.atomic(directory / "capacity_frontier.json", report)
    (directory / "CAPACITY.md").write_text("\n".join(lines) + "\n")
    serving.emit(event="capacity_report", phase=phase, output=str(directory / "CAPACITY.md"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "report"))
    parser.add_argument("--phase", choices=PHASES, default="pilot")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = resolve(args.config)
    frontier, derived, serving_dir = build_experiment(config_path, args.phase)
    if args.stage == "prepare":
        if os.geteuid() == 0:
            raise RuntimeError("Prepare must run without sudo")
        serving.prepare(derived, args.phase, serving_dir)
        serving.emit(
            event="capacity_prepared", phase=args.phase,
            rates=load(derived)["capacity_provenance"]["rates"],
            strategies=len(load(derived)["strategies"]), run_dir=str(serving_dir),
        )
    elif args.stage == "run":
        if os.geteuid() != 0:
            raise RuntimeError("Run requires sudo -E for hardware controls")
        serving.execute(derived, args.phase, serving_dir)
        analyze(config_path, args.phase)
        serving.release_ownership(phase_dir(frontier, args.phase))
    else:
        if os.geteuid() == 0:
            raise RuntimeError("Report must run without sudo")
        serving.report(derived, args.phase, serving_dir)
        analyze(config_path, args.phase)


if __name__ == "__main__":
    main()
