#!/usr/bin/env python3
"""Enumerate workload-specific Thor policies with queue-aware SLO constraints.

The output is a model-predicted frontier. It is a shortlist generator, not a
substitute for fixed-trace hardware validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/experiments/thor_queue_aware_pareto_v1.json"


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
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
    if not ordered:
        raise ValueError("empty percentile input")
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def workload(row: dict[str, Any]) -> tuple[int, int]:
    value = row["workload"]
    return int(value[0]), int(value[1])


def source_candidates(config: dict[str, Any]) -> tuple[list[tuple[int, int]], list[list[dict[str, Any]]]]:
    path = resolve(config["frontier_validation"])
    rows = load(path)
    scenario = config["source_filter"]["scenario"]
    limit = float(config["source_filter"]["tpot_limit_ms"])
    targets = [tuple(map(int, item)) for item in config["workloads"]]
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {item: {} for item in targets}
    for row in rows:
        if row.get("strategy") != "frontier_point" or row.get("scenario") != scenario:
            continue
        if abs(float(row.get("tpot_limit_ms")) - limit) > 1e-9:
            continue
        key = workload(row)
        if key not in grouped:
            continue
        hardware = canonical(row["hardware"])
        previous = grouped[key].get(hardware)
        rank = (int(row.get("requests") or 0), int(row.get("blocks") or 0), -float(row["energy_per_token_j"]))
        if previous is None:
            grouped[key][hardware] = row
        else:
            old_rank = (
                int(previous.get("requests") or 0),
                int(previous.get("blocks") or 0),
                -float(previous["energy_per_token_j"]),
            )
            if rank > old_rank:
                grouped[key][hardware] = row
    missing = [item for item in targets if not grouped[item]]
    if missing:
        raise RuntimeError(f"missing static candidates for {missing}")
    candidates = [
        sorted(grouped[item].values(), key=lambda row: canonical(row["hardware"]))
        for item in targets
    ]
    return targets, candidates


def load_traces(config: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], dict[str, str]]:
    directory = resolve(config["trace_dir"])
    protocol = load(directory / "protocol.json")
    traces = []
    provenance = {str((directory / "protocol.json").relative_to(ROOT)): sha256(directory / "protocol.json")}
    for repeat in range(len(protocol["trace_hashes"])):
        path = directory / f"trace_{repeat}.json"
        trace = load(path)
        traces.append(trace)
        provenance[str(path.relative_to(ROOT))] = sha256(path)
    return traces, provenance


def load_serving_runs(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    directory = resolve(config["serving_results_dir"])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    provenance = {}
    for path in sorted(directory.glob("result_*_*.json")):
        value = load(path)
        grouped[value["strategy"]].append(value)
        provenance[str(path.relative_to(ROOT))] = sha256(path)
    for runs in grouped.values():
        runs.sort(key=lambda row: int(row["repeat"]))
    if not grouped:
        raise RuntimeError(f"no serving results in {directory}")
    return grouped, provenance


def candidate_lookup(
    workloads: list[tuple[int, int]], candidates: list[list[dict[str, Any]]]
) -> dict[tuple[tuple[int, int], str], tuple[int, dict[str, Any]]]:
    return {
        (workload_key, canonical(row["hardware"])): (index, row)
        for workload_key, rows in zip(workloads, candidates)
        for index, row in enumerate(rows)
    }


def calibrate(
    config: dict[str, Any],
    workloads: list[tuple[int, int]],
    candidates: list[list[dict[str, Any]]],
    runs: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    lookup = candidate_lookup(workloads, candidates)
    selected = set(config["calibration_strategies"])
    latency_samples: dict[tuple[tuple[int, int], str], dict[str, list[float]]] = defaultdict(
        lambda: {"ttft": [], "tpot": [], "control": []}
    )
    energy_rows = []
    matched_responses = 0
    for strategy in selected:
        for run in runs.get(strategy, []):
            active_proxy_j = 0.0
            busy_s = 0.0
            for response in run["responses"]:
                if response.get("dropped") or not response.get("metrics"):
                    continue
                request = response["request"]
                key = (int(request["prompt_length"]), int(request["output_length"]))
                signature = canonical(response["hardware"])
                match = lookup.get((key, signature))
                if match is None:
                    continue
                _, static = match
                metrics = response["metrics"]
                bucket = latency_samples[(key, signature)]
                bucket["ttft"].append(float(metrics["prefill_and_first_sample_ms"]))
                bucket["tpot"].append(float(metrics["tpot_ms"]))
                bucket["control"].append(float(metrics.get("control_latency_ms") or 0.0))
                service_s = (float(metrics["total_time_ms"]) + float(metrics.get("control_latency_ms") or 0.0)) / 1000.0
                active_proxy_j += float(static["avg_power_w"]) * service_s
                busy_s += service_s
                matched_responses += 1
            idle_s = max(0.0, float(run["duration_with_drain_s"]) - busy_s)
            if active_proxy_j > 0:
                energy_rows.append((active_proxy_j, idle_s, float(run["total_energy_j"])))

    ratio_bounds = tuple(map(float, config["latency_ratio_bounds"]))
    ttft_by_workload: dict[tuple[int, int], list[float]] = defaultdict(list)
    tpot_by_workload: dict[tuple[int, int], list[float]] = defaultdict(list)
    control_by_mode: dict[str, list[float]] = defaultdict(list)
    exact_control: dict[str, float] = {}
    for (key, signature), samples in latency_samples.items():
        _, static = lookup[(key, signature)]
        ttft_ratio = percentile(samples["ttft"], 0.95) / float(static["ttft_p95_ms"])
        tpot_ratio = percentile(samples["tpot"], 0.95) / float(static["tpot_p95_ms"])
        ttft_by_workload[key].append(min(ratio_bounds[1], max(ratio_bounds[0], ttft_ratio)))
        tpot_by_workload[key].append(min(ratio_bounds[1], max(ratio_bounds[0], tpot_ratio)))
        exact_control[signature] = percentile(samples["control"], 0.95)
        control_by_mode[str(static["hardware"].get("mode", "unknown"))].extend(samples["control"])
    ttft_ratio = {"x".join(map(str, key)): float(np.median(values)) for key, values in ttft_by_workload.items()}
    tpot_ratio = {"x".join(map(str, key)): float(np.median(values)) for key, values in tpot_by_workload.items()}
    mode_control = {key: percentile(values, 0.95) for key, values in control_by_mode.items()}
    global_control = percentile([item for values in control_by_mode.values() for item in values], 0.95)

    low_scale, high_scale = map(float, config["active_power_scale_bounds"])
    low_idle, high_idle = map(float, config["idle_power_bounds_w"])
    active = np.asarray([row[0] for row in energy_rows], dtype=np.float64)
    idle = np.asarray([row[1] for row in energy_rows], dtype=np.float64)
    target = np.asarray([row[2] for row in energy_rows], dtype=np.float64)
    best = None
    for scale in np.linspace(low_scale, high_scale, 901):
        denominator = float(np.dot(idle, idle))
        idle_power = float(np.dot(idle, target - scale * active) / denominator) if denominator else low_idle
        idle_power = min(high_idle, max(low_idle, idle_power))
        predicted = scale * active + idle_power * idle
        rmse = float(np.sqrt(np.mean(np.square(predicted - target))))
        if best is None or rmse < best[0]:
            best = (rmse, float(scale), idle_power)
    if best is None:
        raise RuntimeError("energy calibration has no matched runs")
    rmse, active_scale, idle_power = best
    mape = float(np.mean(np.abs((active_scale * active + idle_power * idle - target) / target)) * 100.0)
    return {
        "matched_responses": matched_responses,
        "energy_calibration_runs": len(energy_rows),
        "ttft_ratio_by_workload": ttft_ratio,
        "tpot_ratio_by_workload": tpot_ratio,
        "control_p95_ms_by_hardware": exact_control,
        "control_p95_ms_by_mode": mode_control,
        "global_control_p95_ms": global_control,
        "active_power_scale": active_scale,
        "idle_power_w": idle_power,
        "energy_fit_rmse_j": rmse,
        "energy_fit_mape_pct": mape,
    }


def candidate_arrays(
    workloads: list[tuple[int, int]],
    candidates: list[list[dict[str, Any]]],
    calibration: dict[str, Any],
) -> list[dict[str, np.ndarray]]:
    result = []
    for key, rows in zip(workloads, candidates):
        label = "x".join(map(str, key))
        ttft_factor = float(calibration["ttft_ratio_by_workload"].get(label, 1.0))
        tpot_factor = float(calibration["tpot_ratio_by_workload"].get(label, 1.0))
        values = defaultdict(list)
        for row in rows:
            hardware = row["hardware"]
            signature = canonical(hardware)
            mode = str(hardware.get("mode", "unknown"))
            control = calibration["control_p95_ms_by_hardware"].get(
                signature,
                calibration["control_p95_ms_by_mode"].get(mode, calibration["global_control_p95_ms"]),
            )
            ttft = float(row["ttft_p95_ms"]) * ttft_factor
            tpot = float(row["tpot_p95_ms"]) * tpot_factor
            service = control + ttft + tpot * max(0, key[1] - 1)
            values["ttft"].append(ttft)
            values["tpot"].append(tpot)
            values["control"].append(control)
            values["service"].append(service)
            values["power"].append(float(row["avg_power_w"]))
        result.append({name: np.asarray(items, dtype=np.float64) for name, items in values.items()})
    return result


def decode_policy_ids(ids: np.ndarray, sizes: list[int]) -> np.ndarray:
    result = np.empty((len(sizes), ids.size), dtype=np.int16)
    remainder = ids.copy()
    for position in range(len(sizes) - 1, -1, -1):
        result[position] = remainder % sizes[position]
        remainder //= sizes[position]
    return result


def simulate_chunk(
    policy: np.ndarray,
    arrays: list[dict[str, np.ndarray]],
    workloads: list[tuple[int, int]],
    traces: list[list[dict[str, Any]]],
    profiles: list[dict[str, Any]],
    config: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    count = policy.shape[1]
    selected = [
        {name: values[indexes] for name, values in bucket.items()}
        for indexes, bucket in zip(policy, arrays)
    ]
    lookup = {key: index for index, key in enumerate(workloads)}
    profile_pass = np.ones((len(profiles), count), dtype=bool)
    total_energy = np.zeros(count, dtype=np.float64)
    total_duration = np.zeros(count, dtype=np.float64)
    total_requests = sum(len(trace) for trace in traces)
    ttft_samples = np.empty((total_requests, count), dtype=np.float32)
    tpot_samples = np.empty((total_requests, count), dtype=np.float32)
    queue_samples = np.empty((total_requests, count), dtype=np.float32)
    sample_index = 0
    horizon_ms = float(config["arrival_horizon_s"]) * 1000.0
    required = float(config["required_observed_attainment"])
    for trace in traces:
        finish = np.zeros(count, dtype=np.float64)
        active_proxy = np.zeros(count, dtype=np.float64)
        busy_ms = np.zeros(count, dtype=np.float64)
        repeat_passed = np.zeros((len(profiles), count), dtype=np.int16)
        for request in trace:
            key = (int(request["prompt_length"]), int(request["output_length"]))
            bucket = selected[lookup[key]]
            arrival = float(request["scheduled_time"]) * 1000.0
            start = np.maximum(arrival, finish)
            wait = start - arrival
            ttft = wait + bucket["control"] + bucket["ttft"]
            completion = wait + bucket["service"]
            finish = start + bucket["service"]
            ttft_samples[sample_index] = ttft
            tpot_samples[sample_index] = bucket["tpot"]
            queue_samples[sample_index] = wait
            sample_index += 1
            active_proxy += bucket["power"] * bucket["service"] / 1000.0
            busy_ms += bucket["service"]
            output = int(request["output_length"])
            for profile_index, profile in enumerate(profiles):
                ttft_limit = float(profile["ttft_limit_ms"])
                tpot_limit = float(profile["tpot_limit_ms"])
                e2e_limit = ttft_limit + tpot_limit * max(0, output - 1)
                repeat_passed[profile_index] += (
                    (ttft <= ttft_limit)
                    & (bucket["tpot"] <= tpot_limit)
                    & (completion <= e2e_limit)
                )
        duration_s = np.maximum(horizon_ms, finish) / 1000.0
        idle_s = np.maximum(0.0, duration_s - busy_ms / 1000.0)
        total_energy += (
            float(calibration["active_power_scale"]) * active_proxy
            + float(calibration["idle_power_w"]) * idle_s
        )
        total_duration += duration_s
        required_count = math.ceil(required * len(trace) - 1e-12)
        profile_pass &= repeat_passed >= required_count
    try:
        ttft_p95 = np.quantile(ttft_samples, 0.95, axis=0, method="higher")
        tpot_p95 = np.quantile(tpot_samples, 0.95, axis=0, method="higher")
        queue_p95 = np.quantile(queue_samples, 0.95, axis=0, method="higher")
    except TypeError:
        ttft_p95 = np.quantile(ttft_samples, 0.95, axis=0, interpolation="higher")
        tpot_p95 = np.quantile(tpot_samples, 0.95, axis=0, interpolation="higher")
        queue_p95 = np.quantile(queue_samples, 0.95, axis=0, interpolation="higher")
    return {
        "profile_pass": profile_pass,
        "total_energy_j": total_energy,
        "avg_power_w": total_energy / total_duration,
        "duration_s": total_duration,
        "ttft_p95_ms": ttft_p95,
        "tpot_p95_ms": tpot_p95,
        "queue_p95_ms": queue_p95,
    }


def exact_observed_goodput(run_list: list[dict[str, Any]], profile: dict[str, Any], required: float) -> tuple[float, bool]:
    passed = 0
    offered = 0
    repeat_ok = []
    for run in run_list:
        repeat_passed = 0
        for response in run["responses"]:
            offered += 1
            metrics = response.get("metrics") or {}
            request = response["request"]
            output = int(request["output_length"])
            e2e_limit = float(profile["ttft_limit_ms"]) + float(profile["tpot_limit_ms"]) * max(0, output - 1)
            ok = bool(
                not response.get("dropped")
                and int(metrics.get("output_tokens") or 0) >= output
                and float(metrics.get("ttft_end_to_end_ms") or math.inf) <= float(profile["ttft_limit_ms"])
                and float(metrics.get("tpot_ms") or math.inf) <= float(profile["tpot_limit_ms"])
                and float(metrics.get("arrival_to_completion_ms") or math.inf) <= e2e_limit
            )
            passed += int(ok)
            repeat_passed += int(ok)
        repeat_ok.append(repeat_passed / len(run["responses"]) >= required)
    return passed / offered, all(repeat_ok)


def policy_for_strategy(
    strategy: str,
    runs: dict[str, list[dict[str, Any]]],
    workloads: list[tuple[int, int]],
    candidates: list[list[dict[str, Any]]],
) -> list[int] | None:
    lookup = candidate_lookup(workloads, candidates)
    mapping = {}
    for run in runs.get(strategy, []):
        for response in run["responses"]:
            request = response["request"]
            key = (int(request["prompt_length"]), int(request["output_length"]))
            match = lookup.get((key, canonical(response["hardware"])))
            if match is not None:
                mapping[key] = match[0]
    if any(key not in mapping for key in workloads):
        return None
    return [mapping[key] for key in workloads]


def model_validation(
    config: dict[str, Any],
    workloads: list[tuple[int, int]],
    candidates: list[list[dict[str, Any]]],
    arrays: list[dict[str, np.ndarray]],
    traces: list[list[dict[str, Any]]],
    runs: dict[str, list[dict[str, Any]]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    profiles = config["profiles"]
    rows = []
    holdout_errors = []
    for strategy in config["calibration_strategies"] + config["holdout_strategies"]:
        indexes = policy_for_strategy(strategy, runs, workloads, candidates)
        if indexes is None:
            continue
        policy = np.asarray(indexes, dtype=np.int16).reshape(len(workloads), 1)
        predicted = simulate_chunk(policy, arrays, workloads, traces, profiles, config, calibration)
        actual_energy = sum(float(run["total_energy_j"]) for run in runs[strategy])
        actual_duration = sum(float(run["duration_with_drain_s"]) for run in runs[strategy])
        energy_error = 100.0 * (float(predicted["total_energy_j"][0]) / actual_energy - 1.0)
        profile_rows = []
        maximum_goodput_error = 0.0
        for profile_index, profile in enumerate(profiles):
            actual_goodput, actual_feasible = exact_observed_goodput(
                runs[strategy], profile, float(config["required_observed_attainment"])
            )
            predicted_feasible = bool(predicted["profile_pass"][profile_index, 0])
            predicted_goodput = None
            maximum_goodput_error = max(
                maximum_goodput_error,
                0.0 if predicted_feasible == actual_feasible else 100.0,
            )
            profile_rows.append({
                "profile": profile["name"],
                "actual_goodput": actual_goodput,
                "actual_feasible": actual_feasible,
                "predicted_goodput": predicted_goodput,
                "predicted_feasible": predicted_feasible,
            })
        role = "holdout" if strategy in config["holdout_strategies"] else "calibration"
        row = {
            "strategy": strategy,
            "role": role,
            "policy_indices": indexes,
            "actual_energy_j": actual_energy,
            "predicted_energy_j": float(predicted["total_energy_j"][0]),
            "energy_error_pct": energy_error,
            "actual_avg_power_w": actual_energy / actual_duration,
            "predicted_avg_power_w": float(predicted["avg_power_w"][0]),
            "profiles": profile_rows,
        }
        rows.append(row)
        if role == "holdout":
            holdout_errors.append((abs(energy_error), maximum_goodput_error))
    max_energy = max((item[0] for item in holdout_errors), default=math.inf)
    max_goodput = max((item[1] for item in holdout_errors), default=math.inf)
    return {
        "rows": rows,
        "holdout_max_abs_energy_error_pct": max_energy,
        "holdout_max_feasibility_disagreement_pp": max_goodput,
        "passed": bool(
            max_energy <= float(config["holdout_max_energy_error_pct"])
            and max_goodput <= float(config["holdout_max_goodput_error_pp"])
        ),
    }


def pareto_indices(energy: np.ndarray, power: np.ndarray) -> np.ndarray:
    order = np.lexsort((power, energy))
    selected = []
    best_power = math.inf
    for index in order:
        value = float(power[index])
        if value < best_power - 1e-9:
            selected.append(int(index))
            best_power = value
    return np.asarray(selected, dtype=np.int64)


def compact_policy(
    role: str,
    position: int,
    policy: np.ndarray,
    metrics: dict[str, np.ndarray],
    workloads: list[tuple[int, int]],
    candidates: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    indexes = [int(value) for value in policy[:, position]]
    return {
        "role": role,
        "policy_indices": indexes,
        "hardware_by_workload": {
            "x".join(map(str, key)): candidates[workload_index][candidate_index]["hardware"]
            for workload_index, (key, candidate_index) in enumerate(zip(workloads, indexes))
        },
        "predicted_total_energy_j": float(metrics["total_energy_j"][position]),
        "predicted_avg_power_w": float(metrics["avg_power_w"][position]),
        "predicted_duration_s": float(metrics["duration_s"][position]),
        "predicted_ttft_p95_ms": float(metrics["ttft_p95_ms"][position]),
        "predicted_tpot_p95_ms": float(metrics["tpot_p95_ms"][position]),
        "predicted_queue_p95_ms": float(metrics["queue_p95_ms"][position]),
        "hardware_validated": False,
    }


def shortlist(
    policy: np.ndarray,
    metrics: dict[str, np.ndarray],
    baseline_energy: float,
    limit: int,
    workloads: list[tuple[int, int]],
    candidates: list[list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    frontier = pareto_indices(metrics["total_energy_j"], metrics["avg_power_w"])
    energy = metrics["total_energy_j"][frontier]
    power = metrics["avg_power_w"][frontier]
    roles: dict[int, set[str]] = defaultdict(set)
    roles[int(frontier[int(np.argmin(energy))])].add("energy_min")
    no_regression = frontier[energy <= baseline_energy]
    if no_regression.size:
        roles[int(no_regression[int(np.argmin(metrics["avg_power_w"][no_regression]))])].add(
            "power_min_without_energy_regression"
        )
    energy_norm = (energy - energy.min()) / max(1e-12, float(energy.max() - energy.min()))
    power_norm = (power - power.min()) / max(1e-12, float(power.max() - power.min()))
    roles[int(frontier[int(np.argmin(np.sqrt(np.square(energy_norm) + np.square(power_norm))))])].add("knee")
    remaining = max(0, limit - len(roles))
    if remaining and frontier.size:
        for item in np.linspace(0, frontier.size - 1, remaining + 2, dtype=int)[1:-1]:
            roles[int(frontier[item])].add("epsilon_frontier")
    selected = []
    for position in sorted(roles, key=lambda item: (float(metrics["total_energy_j"][item]), float(metrics["avg_power_w"][item]))):
        selected.append(
            compact_policy(
                "+".join(sorted(roles[position])), position, policy, metrics, workloads, candidates
            )
        )
    return selected, frontier


def explore(config_path: Path) -> Path:
    started = time.time()
    config = load(config_path)
    output = resolve(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    workloads, candidates = source_candidates(config)
    traces, trace_provenance = load_traces(config)
    runs, result_provenance = load_serving_runs(config)
    calibration = calibrate(config, workloads, candidates, runs)
    arrays = candidate_arrays(workloads, candidates, calibration)
    validation = model_validation(config, workloads, candidates, arrays, traces, runs, calibration)
    sizes = [len(rows) for rows in candidates]
    total = math.prod(sizes)
    chunk_size = int(config["chunk_size"])
    profiles = config["profiles"]
    collected = [defaultdict(list) for _ in profiles]
    print(json.dumps({
        "event": "search_start",
        "candidate_counts": sizes,
        "policy_count": total,
        "model_holdout_passed": validation["passed"],
    }), flush=True)
    for start in range(0, total, chunk_size):
        stop = min(total, start + chunk_size)
        ids = np.arange(start, stop, dtype=np.int64)
        policy = decode_policy_ids(ids, sizes)
        metrics = simulate_chunk(policy, arrays, workloads, traces, profiles, config, calibration)
        for profile_index in range(len(profiles)):
            mask = metrics["profile_pass"][profile_index]
            if not np.any(mask):
                continue
            bucket = collected[profile_index]
            bucket["policy"].append(policy[:, mask])
            for name in (
                "total_energy_j", "avg_power_w", "duration_s", "ttft_p95_ms", "tpot_p95_ms", "queue_p95_ms"
            ):
                bucket[name].append(metrics[name][mask])
        if start == 0 or stop == total or (start // chunk_size) % 5 == 0:
            print(json.dumps({"event": "search_progress", "processed": stop, "total": total}), flush=True)

    baseline_rows = runs[config["power_energy_baseline"]]
    baseline_energy = sum(float(run["total_energy_j"]) for run in baseline_rows)
    profile_results = []
    all_shortlist = []
    for profile, bucket in zip(profiles, collected):
        if not bucket:
            profile_results.append({"profile": profile, "feasible_policies": 0, "frontier": [], "shortlist": []})
            continue
        policy = np.concatenate(bucket["policy"], axis=1)
        metrics = {
            name: np.concatenate(bucket[name])
            for name in (
                "total_energy_j", "avg_power_w", "duration_s", "ttft_p95_ms", "tpot_p95_ms", "queue_p95_ms"
            )
        }
        selected, frontier_positions = shortlist(
            policy,
            metrics,
            baseline_energy,
            int(config["shortlist_limit"]),
            workloads,
            candidates,
        )
        frontier = [
            compact_policy("pareto", int(position), policy, metrics, workloads, candidates)
            for position in frontier_positions
        ]
        for row in selected:
            row["profile"] = profile["name"]
            all_shortlist.append(row)
        profile_results.append({
            "profile": profile,
            "feasible_policies": int(policy.shape[1]),
            "pareto_policies": len(frontier),
            "frontier": frontier,
            "shortlist": selected,
        })

    payload = {
        "schema": config["schema"],
        "status": "predicted_candidates_require_hardware_validation",
        "workloads": [list(item) for item in workloads],
        "candidate_counts": sizes,
        "policy_count": total,
        "calibration": calibration,
        "model_validation": validation,
        "baseline": {
            "strategy": config["power_energy_baseline"],
            "measured_total_energy_j": baseline_energy,
        },
        "profiles": profile_results,
        "elapsed_s": time.time() - started,
        "provenance": {
            str(resolve(config["frontier_validation"]).relative_to(ROOT)): sha256(resolve(config["frontier_validation"])),
            **trace_provenance,
            **result_provenance,
        },
    }
    result_path = output / "queue_aware_pareto.json"
    atomic_json(result_path, payload)
    atomic_json(output / "hardware_validation_shortlist.json", all_shortlist)
    atomic_json(output / "protocol.json", {
        "schema": config["schema"],
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "result": str(result_path),
        "result_sha256": sha256(result_path),
        "hardware_frequency_changes": False,
        "model_holdout_passed": validation["passed"],
    })

    with (output / "shortlist.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "profile", "role", "policy_indices", "predicted_total_energy_j", "predicted_avg_power_w",
            "predicted_duration_s", "predicted_ttft_p95_ms", "predicted_tpot_p95_ms",
            "predicted_queue_p95_ms", "hardware_validated",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_shortlist:
            writer.writerow({key: row.get(key) for key in fields})
    print(json.dumps({"event": "complete", "result": str(result_path), "elapsed_s": payload["elapsed_s"]}), flush=True)
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    explore(args.config.resolve())


if __name__ == "__main__":
    main()
