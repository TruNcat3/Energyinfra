#!/usr/bin/env python3
"""Measure Thor phase-aware DVFS with scheduled prefill transitions and fallbacks."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import statistics
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_energyinfra_v2 import Hardware, Power, atomic
from run_thor_moe_placement_power_v1 import release_ownership
from run_thor_moe_runtime_gate_v1 import file_hash, load, resolve, run_request, start_server, stop_server


def fmt(value: Any, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower, upper = int(index), min(len(ordered) - 1, int(index) + 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def object_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def wait_until(target: float) -> None:
    while True:
        remaining = target - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def peak_power(power: Power, start: float, end: float) -> float | None:
    with power.lock:
        values = [value for timestamp, value in power.samples if start <= timestamp <= end]
    return max(values) if values else None


def load_rows(journal: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not journal.exists():
        return rows
    for line in journal.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(row["strategy"], row["trace"], int(row["repeat"]))] = row
    return rows


def append_row(journal: Path, row: dict[str, Any]) -> None:
    with journal.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def token_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int):
        return value
    return 0


def stream_request(
    url: str,
    workload: dict[str, Any],
    seed: int,
    timeout: float,
    hardware: Hardware,
    decode_hardware: dict[str, Any],
    switch_decode: bool,
) -> dict[str, Any]:
    """Stream one request and switch to decode hardware at the first output token."""
    rng = __import__("random").Random(seed)
    payload = {
        "prompt": [100 + rng.randrange(100000) for _ in range(int(workload["prompt_tokens"]))],
        "n_predict": int(workload["output_tokens"]),
        "temperature": 0,
        "ignore_eos": True,
        "cache_prompt": False,
        "stream": True,
    }
    started = time.monotonic()
    first_token = None
    switch_started = None
    switch_finished = None
    switch_error = None
    observed_tokens = 0
    final: dict[str, Any] = {}
    try:
        request = urllib.request.Request(
            url + "/completion", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                value = json.loads(data)
                final = value
                observed_tokens += token_count(value.get("tokens"))
                has_output = observed_tokens > 0 or bool(value.get("content"))
                if first_token is None and has_output:
                    first_token = time.monotonic()
                    switch_started = first_token
                    if switch_decode:
                        try:
                            hardware.apply(decode_hardware)
                        except Exception as exc:
                            switch_error = repr(exc)
                        finally:
                            switch_finished = time.monotonic()
                if value.get("stop"):
                    break
        ended = time.monotonic()
        timings = final.get("timings") or {}
        predicted = int(final.get("tokens_predicted", observed_tokens))
        if first_token is None:
            first_token = ended
        predicted_ms = timings.get("predicted_ms")
        tpot = (
            float(predicted_ms) / max(1, predicted - 1)
            if predicted_ms is not None else
            (ended - first_token) * 1000.0 / max(1, predicted - 1)
        )
        exact = predicted == int(workload["output_tokens"]) and switch_error is None
        return {
            "prompt": workload["name"],
            "prompt_tokens": workload["prompt_tokens"],
            "requested_output_tokens": workload["output_tokens"],
            "output_tokens": predicted,
            "completion_exact": exact,
            "ttft_ms": (first_token - started) * 1000.0,
            "tpot_ms": tpot,
            "elapsed_ms": (ended - started) * 1000.0,
            "phase_switch_latency_ms": None if switch_finished is None else (switch_finished - switch_started) * 1000.0,
            "phase_switch_enabled": switch_decode,
            "phase_switch_error": switch_error,
            "error": None,
        }
    except BaseException as exc:
        return {
            "prompt": workload["name"],
            "prompt_tokens": workload["prompt_tokens"],
            "requested_output_tokens": workload["output_tokens"],
            "output_tokens": observed_tokens,
            "completion_exact": False,
            "ttft_ms": None,
            "tpot_ms": None,
            "elapsed_ms": (time.monotonic() - started) * 1000.0,
            "phase_switch_latency_ms": None,
            "phase_switch_enabled": switch_decode,
            "phase_switch_error": switch_error,
            "error": repr(exc),
        }


def run_trace(
    url: str,
    trace: dict[str, Any],
    workloads: dict[str, Any],
    seed: int,
    timeout: float,
    hardware: Hardware,
    strategy: dict[str, Any],
    power: Power,
    measurement: dict[str, Any],
) -> dict[str, Any]:
    power.wait_after(time.monotonic())
    epoch = time.monotonic() + float(measurement["launch_lead_s"])
    prefill_lead_s = float(measurement["prefill_lead_s"])
    minimum_output_tokens = int(strategy.get("minimum_output_tokens", 0))
    fallback_hardware = strategy.get("fallback", strategy["prefill"])
    responses: list[dict[str, Any]] = []
    for request_id, request in enumerate(trace["requests"]):
        scheduled = epoch + float(request["scheduled_time_s"])
        workload_name = request["workload"]
        workload = {"name": workload_name, **workloads[workload_name]}
        phase_enabled = bool(strategy.get("phase_aware", False)) and int(workload["output_tokens"]) >= minimum_output_tokens
        start_hardware = strategy["prefill"] if phase_enabled else fallback_hardware
        if strategy.get("phase_aware", False):
            wait_until(max(time.monotonic(), scheduled - prefill_lead_s))
            applied_start = hardware.apply(start_hardware)
        else:
            applied_start = hardware.state()
        wait_until(scheduled)
        dispatched = time.monotonic()
        result = stream_request(
            url, workload, seed + request_id * 7919, timeout,
            hardware, strategy["decode"], phase_enabled,
        )
        ended = time.monotonic()
        response = {
            "request_id": request_id,
            "workload": workload_name,
            "scheduled_time_s": request["scheduled_time_s"],
            "dispatch_lag_ms": (dispatched - scheduled) * 1000.0,
            "ttft_end_to_end_ms": None if result["ttft_ms"] is None else (dispatched - scheduled) * 1000.0 + result["ttft_ms"],
            "end_to_end_ms": (ended - scheduled) * 1000.0,
            "completed_monotonic": ended,
            "start_hardware": applied_start,
            "start_policy": "phase_prefill" if phase_enabled else "static_fallback",
            **result,
        }
        responses.append(response)
    horizon = epoch + float(trace["duration_s"])
    wait_until(horizon)
    drain_end = max(horizon, max(item["completed_monotonic"] for item in responses))
    fixed = power.energy(epoch, horizon, int(measurement["minimum_samples"]))
    drained = power.energy(epoch, drain_end, int(measurement["minimum_samples"]))
    completed = sum(item["completion_exact"] for item in responses)
    completed_by_horizon = sum(
        item["completion_exact"] and item["completed_monotonic"] <= horizon for item in responses
    )
    total_tokens = sum(int(item["output_tokens"]) for item in responses if item["completion_exact"])
    for item in responses:
        item.pop("completed_monotonic", None)
    return {
        "trace_sha256": object_hash(trace),
        "duration_s": trace["duration_s"],
        "offered": len(responses),
        "completed_exact": completed,
        "completed_by_horizon": completed_by_horizon,
        "comparable_fixed_horizon": completed_by_horizon == len(responses),
        "output_tokens": total_tokens,
        "fixed_total_energy_j": fixed["total_energy_j"],
        "fixed_avg_power_w": fixed["avg_power_w"],
        "fixed_peak_power_w": peak_power(power, epoch, horizon),
        "fixed_energy_per_output_token_j": None if total_tokens <= 0 else fixed["total_energy_j"] / total_tokens,
        "with_drain_total_energy_j": drained["total_energy_j"],
        "with_drain_duration_s": drain_end - epoch,
        "telemetry_observed_max_gap_s": max(fixed["telemetry_observed_max_gap_s"], drained["telemetry_observed_max_gap_s"]),
        "responses": responses,
    }


def aggregate_rows(rows: list[dict[str, Any]], repeats: int, slos: list[dict[str, Any]], required: float) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["strategy"], row["trace"])].append(row)
    aggregates = []
    for (strategy, trace), group in sorted(groups.items()):
        responses = [response for row in group for response in row["responses"]]
        first = group[0]
        item: dict[str, Any] = {
            "strategy": strategy,
            "label": first["label"],
            "trace": trace,
            "repeats": len(group),
            "expected_repeats": repeats,
            "complete": len(group) == repeats,
            "fixed_horizon_success_rate": statistics.fmean(row["completed_by_horizon"] / row["offered"] for row in group),
            "p95_ttft_end_to_end_ms": percentile([float(x["ttft_end_to_end_ms"]) for x in responses if x["ttft_end_to_end_ms"] is not None], 0.95),
            "p95_tpot_ms": percentile([float(x["tpot_ms"]) for x in responses if x["tpot_ms"] is not None], 0.95),
            "p95_phase_switch_latency_ms": percentile([float(x["phase_switch_latency_ms"]) for x in responses if x["phase_switch_latency_ms"] is not None], 0.95),
            "phase_switch_success_rate": statistics.fmean(x["phase_switch_error"] is None for x in responses),
        }
        for metric in ("fixed_total_energy_j", "fixed_avg_power_w", "fixed_peak_power_w", "fixed_energy_per_output_token_j"):
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            item[f"mean_{metric}"] = statistics.fmean(values) if values else None
        for slo in slos:
            passed = sum(
                bool(x["completion_exact"])
                and x["ttft_end_to_end_ms"] is not None
                and x["tpot_ms"] is not None
                and float(x["tpot_ms"]) <= float(slo["tpot_ms"])
                and float(x["ttft_end_to_end_ms"]) <= float(slo["ttft_ms"][x["workload"]])
                for x in responses
            )
            attainment = passed / len(responses)
            item[f"attainment_{slo['name']}"] = attainment
            item[f"feasible_{slo['name']}"] = bool(item["complete"] and item["fixed_horizon_success_rate"] == 1.0 and attainment >= required)
        aggregates.append(item)
    baselines = {row["trace"]: row for row in aggregates if row["strategy"] == "static_native_dynamic"}
    for row in aggregates:
        baseline = baselines.get(row["trace"])
        for metric in ("fixed_total_energy_j", "fixed_avg_power_w", "fixed_peak_power_w"):
            value = row[f"mean_{metric}"]
            reference = None if baseline is None else baseline[f"mean_{metric}"]
            row[f"{metric}_delta_vs_native_pct"] = None if value is None or not reference else (value / reference - 1.0) * 100.0
    return aggregates


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("mean_fixed_total_energy_j", "mean_fixed_avg_power_w", "mean_fixed_peak_power_w", "p95_ttft_end_to_end_ms", "p95_tpot_ms")
    return all(left[key] <= right[key] for key in keys) and any(left[key] < right[key] for key in keys)


def analyze(aggregates: list[dict[str, Any]], slos: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frontiers, winners = [], []
    for slo in slos:
        for trace in sorted({row["trace"] for row in aggregates}):
            feasible = [row for row in aggregates if row["trace"] == trace and row[f"feasible_{slo['name']}"]]
            frontier = [row for row in feasible if not any(dominates(other, row) for other in feasible if other is not row)]
            frontiers.append({"slo": slo["name"], "trace": trace, "feasible": [x["strategy"] for x in feasible], "pareto": [x["strategy"] for x in frontier]})
            for objective, metric in (("minimum_fixed_energy", "mean_fixed_total_energy_j"), ("minimum_average_power", "mean_fixed_avg_power_w"), ("minimum_peak_power", "mean_fixed_peak_power_w")):
                if feasible:
                    winner = min(feasible, key=lambda x: x[metric])
                    winners.append({"slo": slo["name"], "trace": trace, "objective": objective, "strategy": winner["strategy"], "energy_delta_pct": winner["fixed_total_energy_j_delta_vs_native_pct"], "average_power_delta_pct": winner["fixed_avg_power_w_delta_vs_native_pct"], "peak_power_delta_pct": winner["fixed_peak_power_w_delta_vs_native_pct"], "ttft_p95_ms": winner["p95_ttft_end_to_end_ms"], "tpot_p95_ms": winner["p95_tpot_ms"]})
    return frontiers, winners


def render(preset: str, aggregates: list[dict[str, Any]], frontiers: list[dict[str, Any]], winners: list[dict[str, Any]]) -> str:
    lines = ["# Thor MoE phase-aware serving v2", "", f"Preset: `{preset}`. Prefill transitions are scheduled early; short outputs use a static fallback.", "", "## Baseline and strategy metrics", "", "| Trace | Strategy | Energy J | Avg W | Peak W | TTFT p95 ms | TPOT p95 ms | Switch p95 ms |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        lines.append(f"| {row['trace']} | {row['strategy']} | {fmt(row['mean_fixed_total_energy_j'])} | {fmt(row['mean_fixed_avg_power_w'])} | {fmt(row['mean_fixed_peak_power_w'])} | {fmt(row['p95_ttft_end_to_end_ms'])} | {fmt(row['p95_tpot_ms'])} | {fmt(row['p95_phase_switch_latency_ms'])} |")
    lines += ["", "## SLO-constrained winners", "", "| SLO | Trace | Objective | Strategy | Energy delta | Avg-power delta | Peak-power delta | TTFT p95 | TPOT p95 |", "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in winners:
        lines.append(f"| {row['slo']} | {row['trace']} | {row['objective']} | {row['strategy']} | {fmt(row['energy_delta_pct'], 1)}% | {fmt(row['average_power_delta_pct'], 1)}% | {fmt(row['peak_power_delta_pct'], 1)}% | {fmt(row['ttft_p95_ms'])} | {fmt(row['tpot_p95_ms'])} |")
    lines += ["", "## Pareto sets", "", "| SLO | Trace | Feasible | Pareto strategies |", "|---|---|---:|---|"]
    for row in frontiers:
        lines.append(f"| {row['slo']} | {row['trace']} | {len(row['feasible'])} | {', '.join(row['pareto']) or '-'} |")
    lines += ["", "The first token is the phase boundary. This v2 intentionally excludes concurrent traces because hardware frequency is global on Thor; concurrent phase-aware control is a separate experiment.", "", "Quality scope: random-weight fixed-output systems measurement; no model-quality claim.", ""]
    return "\n".join(lines)


def write_outputs(config: dict[str, Any], preset: str, run_dir: Path) -> None:
    rows = list(load_rows(run_dir / "rows.jsonl").values())
    if not rows:
        raise RuntimeError("No phase-aware rows available")
    aggregates = aggregate_rows(rows, int(config["presets"][preset]["repeats"]), config["slo_profiles"], float(config["measurement"]["required_attainment"]))
    frontiers, winners = analyze(aggregates, config["slo_profiles"])
    atomic(run_dir / "results.json", rows)
    atomic(run_dir / "aggregate.json", aggregates)
    atomic(run_dir / "pareto.json", frontiers)
    atomic(run_dir / "recommendations.json", winners)
    fields = [key for key in rows[0] if key not in ("responses", "server_command", "prefill_hardware", "decode_hardware")]
    with (run_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    with (run_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    (run_dir / "RESULTS.md").write_text(render(preset, aggregates, frontiers, winners))
    print(json.dumps({"event": "report", "rows": len(rows), "output": str(run_dir / "RESULTS.md")}), flush=True)


def execute(config_path: Path, config: dict[str, Any], preset: str, model: Path, run_dir: Path) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("Phase-aware serving requires sudo -E")
    runtime = config["runtime"]
    binary = resolve(runtime["server_binary"]).resolve()
    if not model.is_file() or not binary.is_file():
        raise FileNotFoundError(model if not model.is_file() else binary)
    manifest = resolve(runtime["manifest"])
    if manifest.is_file() and load(manifest).get("base_commit") != runtime["base_commit"]:
        raise RuntimeError("Pinned runtime base commit mismatch")
    hardware_config = load(resolve(config["hardware_config"]))
    hardware_config["measurement"]["abort_temperature_c"] = config["measurement"]["abort_temperature_c"]
    strategies = {item["name"]: item for item in config["strategies"]}
    selected = [strategies[name] for name in config["presets"][preset]["strategies"]]
    repeats = int(config["presets"][preset]["repeats"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "server_logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "telemetry").mkdir(parents=True, exist_ok=True)
    journal = run_dir / "rows.jsonl"
    existing = load_rows(journal)
    model_sha256 = file_hash(model)
    snapshot_path = run_dir / "hardware_snapshot.json"
    pending_path = run_dir / "restore_pending.json"

    def interrupted(signum: int, frame: Any) -> None:
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (run_dir / "campaign.lock").open("a") as campaign_lock:
            fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with open("/run/energyinfra-v2-hardware.lock", "a") as hardware_lock:
                fcntl.flock(hardware_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                hw = Hardware(hardware_config)
                if pending_path.exists() and load(pending_path).get("pending"):
                    raise RuntimeError("Hardware restoration is pending; run restore first")
                snapshot = hw.state()
                atomic(snapshot_path, snapshot)
                atomic(pending_path, {"pending": True})
                try:
                    total = len(selected) * len(config["traces"]) * repeats
                    print(json.dumps({"event": "start", "preset": preset, "trials": total}), flush=True)
                    process, url, command, server_handle = start_server(binary, model, runtime, config["placement"], int(runtime["port"]), run_dir / "server_logs" / "server_all_gpu.log")
                    try:
                        for strategy in selected:
                            pending = [(trace, repeat) for trace in config["traces"] for repeat in range(repeats) if not existing.get((strategy["name"], trace["name"], repeat), {}).get("comparable_fixed_horizon")]
                            if not pending:
                                continue
                            hw.apply(strategy["prefill"])
                            warmup = run_request(url, {"name": "warmup", "prompt_tokens": 128, "output_tokens": 16}, 17, float(runtime["request_timeout_s"]))
                            if not warmup["completion_exact"]:
                                raise RuntimeError(f"warmup failed for {strategy['name']}: {warmup}")
                            for trace, repeat in pending:
                                measured = None
                                last_error = None
                                for attempt in range(1, int(config["measurement"]["telemetry_attempts"]) + 1):
                                    power = None
                                    try:
                                        prefill_applied = hw.apply(strategy["prefill"])
                                        time.sleep(float(config["measurement"]["settle_after_apply_s"]))
                                        power = Power(run_dir / "telemetry", hardware_config)
                                        measured = run_trace(url, trace, config["workloads"], int(config["seed"]) + repeat * 1009, float(runtime["request_timeout_s"]), hw, strategy, power, config["measurement"])
                                        measured["telemetry_attempt"] = attempt
                                        measured["temperature_latest_c"] = power.thermal_state()["latest_c"]
                                        measured["temperature_max_c"] = power.thermal_state()["maximum_c"]
                                        measured["prefill_hardware"] = prefill_applied
                                        measured["decode_hardware"] = hw.state()
                                        if power.thermal_state()["exceeded"]:
                                            raise RuntimeError(f"thermal safety limit reached: {measured['temperature_max_c']} C")
                                        break
                                    except RuntimeError as exc:
                                        last_error = exc
                                        measured = None
                                    finally:
                                        if power is not None:
                                            power.close()
                                if measured is None:
                                    raise RuntimeError(f"trace failed after retries for {strategy['name']}/{trace['name']}: {last_error}")
                                row = {"schema": "energyinfra-thor-moe-phase-aware-result-v2", "preset": preset, "strategy": strategy["name"], "label": strategy["label"], "trace": trace["name"], "repeat": repeat, "model": str(model), "model_sha256": model_sha256, "server_command": command, **measured}
                                append_row(journal, row)
                                existing[(strategy["name"], trace["name"], repeat)] = row
                                print(json.dumps({"event": "trial", "strategy": strategy["name"], "trace": trace["name"], "repeat": repeat, "energy_j": row["fixed_total_energy_j"], "avg_power_w": row["fixed_avg_power_w"], "comparable": row["comparable_fixed_horizon"]}), flush=True)
                    finally:
                        stop_server(process, server_handle)
                    atomic(run_dir / "manifest.json", {"schema": "energyinfra-thor-moe-phase-aware-manifest-v2", "config": str(config_path), "config_sha256": file_hash(config_path), "preset": preset, "model": str(model), "model_sha256": model_sha256, "binary": str(binary), "binary_sha256": file_hash(binary), "strategies": [item["name"] for item in selected], "trace_hashes": {trace["name"]: object_hash(trace) for trace in config["traces"]}, "trials_expected": total, "quality_scope": config["quality_scope"], "protocol_scope": config["protocol_scope"]})
                    write_outputs(config, preset, run_dir)
                finally:
                    hw.restore(snapshot)
                    atomic(pending_path, {"pending": False})
    finally:
        release_ownership(run_dir)
    print(json.dumps({"event": "complete", "run_dir": str(run_dir)}, indent=2), flush=True)


def restore(config: dict[str, Any], run_dir: Path) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("Restore requires sudo -E")
    snapshot = run_dir / "hardware_snapshot.json"
    if not snapshot.exists():
        raise FileNotFoundError(snapshot)
    with open("/run/energyinfra-v2-hardware.lock", "a") as hardware_lock:
        fcntl.flock(hardware_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Hardware(load(resolve(config["hardware_config"]))).restore(load(snapshot))
        atomic(run_dir / "restore_pending.json", {"pending": False})
    release_ownership(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("run", "report", "restore"))
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--model", default="models/moe_random_v1/Qwen3MoE-13B-A1.8B-random-v151936-Q4_K_M.gguf")
    parser.add_argument("--config", default="configs/experiments/thor_moe_phase_aware_serving_v2.json")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()
    config_path = resolve(args.config).resolve()
    config = load(config_path)
    run_dir = resolve(args.run_dir or (Path(config["output_dir"]) / args.preset)).resolve()
    if args.stage == "run":
        execute(config_path, config, args.preset, resolve(args.model).resolve(), run_dir)
    elif args.stage == "report":
        write_outputs(config, args.preset, run_dir)
        release_ownership(run_dir)
    else:
        restore(config, run_dir)


if __name__ == "__main__":
    main()
