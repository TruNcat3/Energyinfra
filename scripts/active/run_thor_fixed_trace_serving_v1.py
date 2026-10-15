#!/usr/bin/env python3
"""Compare serving policies on frozen, capacity-calibrated request traces."""

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import random
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.active import run_cpu_floor_v1 as floor


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_q4_long_generation_fixed_trace_v1.json"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def emit(**value):
    print(json.dumps(value, sort_keys=True), flush=True)


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = max(0, min(len(values) - 1, math.ceil(fraction * len(values)) - 1))
    return values[index]


def wilson(passed, offered):
    if not offered:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = passed / offered
    denominator = 1 + z * z / offered
    center = (p + z * z / (2 * offered)) / denominator
    radius = z * math.sqrt(p * (1 - p) / offered + z * z / (4 * offered * offered)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def load_inputs(config_path):
    experiment = json.loads(config_path.read_text())
    if experiment["schema"] != "energyinfra-fixed-trace-serving-v1":
        raise RuntimeError("Unsupported fixed-trace serving schema")
    static_config_path = resolve(experiment["static_config"])
    settings = json.loads(static_config_path.read_text())
    diagnostic = json.loads(resolve(settings["diagnostic_config"]).read_text())
    cfg = copy.deepcopy(json.loads(resolve(diagnostic["base_config"]).read_text()))
    cfg["model"]["path"] = str(resolve(diagnostic["model_path"]))
    cfg["measurement"].update(diagnostic["measurement"])
    cfg["measurement"].update(settings["measurement"])
    return experiment, settings, diagnostic, cfg, static_config_path


def load_frozen_traces(experiment):
    directory = resolve(experiment["calibration_trace_dir"])
    protocol_path = directory / "protocol.json"
    if not protocol_path.exists():
        raise RuntimeError("Confirmed calibration trace protocol is missing")
    protocol = json.loads(protocol_path.read_text())
    traces = []
    for repeat, expected_hash in enumerate(protocol["trace_hashes"]):
        path = directory / f"trace_{repeat}.json"
        if not path.exists():
            raise RuntimeError(f"Frozen trace is missing: {path}")
        trace = json.loads(path.read_text())
        if floor.v2.digest(trace) != expected_hash:
            raise RuntimeError(f"Frozen trace hash changed: {path}")
        traces.append(trace)
    return directory, protocol_path, traces


def build_plans(experiment, settings, cfg, traces):
    parent = resolve(experiment["parent_run_dir"])
    comparison_path = parent / "comparison.json"
    if not comparison_path.exists():
        raise RuntimeError("Frozen static comparison is missing")
    comparisons = json.loads(comparison_path.read_text())
    known = {tuple(workload) for workload in floor.workloads(settings, cfg, "full")}
    selectable = set(experiment["strategies"]) - {"dynamic", "max_clock", "max_clock_idle_dynamic"}
    plans = {}
    for trace in traces:
        arrivals = [request["scheduled_time"] for request in trace]
        if not arrivals or arrivals != sorted(arrivals) or arrivals[0] < 0 or arrivals[-1] > experiment["duration_s"]:
            raise RuntimeError("Invalid frozen arrival trace")
        for request in trace:
            workload = (request["prompt_length"], request["output_length"])
            if workload not in known:
                raise RuntimeError(f"Trace workload {workload} is outside the frozen static table")
            for strategy in selectable:
                row = next((item for item in comparisons
                            if tuple(item["workload"]) == workload
                            and item["strategy"] == strategy
                            and item["scenario"] == experiment["scenario"]
                            and item["tpot_limit_ms"] == experiment["tpot_slo_ms"]), None)
                if not row or row["status"] != "pass_observed":
                    raise RuntimeError(f"Serving blocked: {strategy} lacks validated feasibility for {workload}")
                plans[(strategy, workload)] = row["hardware"]
    return plans, comparison_path


def pareto_flags(rows):
    feasible = [row for row in rows if row["all_repeats_observed_slo_pass"]]
    objectives = ("avg_power_w", "energy_per_token_j", "ttft_end_to_end_p95_ms", "tpot_p95_ms")
    for row in rows:
        row["serving_pareto"] = False
    for candidate in feasible:
        dominated = any(
            all(other[key] <= candidate[key] for key in objectives)
            and any(other[key] < candidate[key] for key in objectives)
            for other in feasible if other is not candidate
        )
        candidate["serving_pareto"] = not dominated


def summarize(output, experiment, repeats):
    manifest = json.loads((output / "protocol.json").read_text())
    groups = defaultdict(dict)
    for path in output.glob("result_*.json"):
        result = json.loads(path.read_text())
        groups[result["strategy"]][result["repeat"]] = result
    expected = set(range(repeats))
    rows = []
    for strategy in manifest["strategies"]:
        by_repeat = groups[strategy]
        results = [by_repeat[index] for index in sorted(by_repeat)]
        if not results:
            continue
        complete = set(by_repeat) == expected
        fixed = complete and all(result["comparable_fixed_work"] for result in results)
        energy = sum(result["total_energy_j"] for result in results)
        duration = sum(result["duration_with_drain_s"] for result in results)
        offered = sum(result["offered"] for result in results)
        passed = sum(result["passed"] for result in results)
        tokens = sum(result["output_tokens"] for result in results)
        metrics = [response.get("metrics", {}) for result in results for response in result["responses"]]
        attainment = passed / offered if offered else 0.0
        interval = wilson(passed, offered)
        row = {
            "strategy": strategy,
            "repeats": len(results),
            "complete": complete,
            "comparable_fixed_work": fixed,
            "offered": offered,
            "completed": sum(result["completed"] for result in results),
            "dropped": sum(result["dropped"] for result in results),
            "passed": passed,
            "observed_attainment": attainment,
            "wilson_ci95_low": interval[0],
            "wilson_ci95_high": interval[1],
            "all_repeats_observed_slo_pass": complete and all(
                result["observed_attainment"] >= experiment["required_observed_attainment"]
                for result in results
            ),
            "ttft_end_to_end_p95_ms": percentile([item.get("ttft_end_to_end_ms") for item in metrics], 0.95),
            "queue_wait_p95_ms": percentile([item.get("queue_wait_ms") for item in metrics], 0.95),
            "tpot_p95_ms": percentile([item.get("tpot_ms") for item in metrics], 0.95),
            "total_energy_j": energy,
            "avg_power_w": energy / duration,
            "fixed_horizon_avg_power_w": sum(result["fixed_horizon"]["total_energy_j"] for result in results) /
                                          (len(results) * experiment["duration_s"]),
            "energy_per_token_j": energy / tokens if tokens else None,
            "duration_with_drain_s": duration,
            "switch_count": sum(len(result["switches"]) for result in results),
        }
        for baseline in ("dynamic", "max_clock", "max_clock_idle_dynamic"):
            base = groups[baseline]
            comparable = fixed and set(base) == expected and all(
                base[index]["comparable_fixed_work"]
                and base[index]["trace_sha256"] == by_repeat[index]["trace_sha256"]
                and base[index]["output_tokens"] == by_repeat[index]["output_tokens"]
                for index in expected
            )
            row[f"total_energy_delta_vs_{baseline}_pct"] = (
                100 * (energy / sum(item["total_energy_j"] for item in base.values()) - 1)
                if comparable else None
            )
        rows.append(row)
    pareto_flags(rows)
    floor.v2.atomic(output / "summary.json", rows)
    if rows:
        with (output / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    number = lambda value: "--" if value is None else f"{value:.3f}"
    lines = [
        f"# Thor Q4 fixed-trace serving comparison: {experiment['trace']}",
        "",
        "The frozen input traces are identical for every strategy.",
        "FIFO queueing, the full 600-second arrival horizon, idle energy and configuration switches are included.",
        "A lower power or energy result with failed SLO is not a constrained win.",
        "Serving Pareto membership is computed only among strategies passing every repeat.",
        "",
        "| Strategy | Completed/offered | SLO % | Every repeat passes | TTFT p95 ms | TPOT p95 ms | W | J/token | Energy vs Dynamic % | Energy vs MaxClock % | Serving Pareto |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['strategy']} | {row['completed']}/{row['offered']} | "
            f"{100 * row['observed_attainment']:.2f} | {row['all_repeats_observed_slo_pass']} | "
            f"{number(row['ttft_end_to_end_p95_ms'])} | {number(row['tpot_p95_ms'])} | "
            f"{number(row['avg_power_w'])} | {number(row['energy_per_token_j'])} | "
            f"{number(row['total_energy_delta_vs_dynamic_pct'])} | "
            f"{number(row['total_energy_delta_vs_max_clock_pct'])} | {row['serving_pareto']} |"
        )
    lines.extend([
        "",
        "summary.csv contains Wilson intervals, fixed-horizon power, drain duration and switch counts.",
        "Random-weight model quality is outside this experiment.",
    ])
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    emit(serving_report=str(output / "RESULTS.md"))


def execute(args, experiment, settings, diagnostic, cfg, static_config_path):
    trace_dir, trace_protocol_path, traces = load_frozen_traces(experiment)
    plans, comparison_path = build_plans(experiment, settings, cfg, traces)
    output = resolve(experiment["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    parent = resolve(experiment["parent_run_dir"])
    provenance = [parent / "protocol.json", parent / "selection.json", comparison_path,
                  trace_protocol_path, *[trace_dir / f"trace_{index}.json" for index in range(len(traces))]]
    identity = {
        "schema": experiment["schema"],
        "strategies": experiment["strategies"],
        "trace": experiment["trace"],
        "scenario": experiment["scenario"],
        "tpot_slo_ms": experiment["tpot_slo_ms"],
        "duration_s": experiment["duration_s"],
        "required_observed_attainment": experiment["required_observed_attainment"],
        "trace_hashes": [floor.v2.digest(trace) for trace in traces],
        "source_sha256": floor.v2.file_hash(Path(__file__).resolve()),
        "static_config_sha256": floor.v2.file_hash(static_config_path),
        "provenance": {str(path.relative_to(ROOT)): floor.v2.file_hash(path) for path in provenance},
    }
    manifest = output / "protocol.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("Fixed-trace serving protocol changed; use a new output directory")
    floor.v2.atomic(manifest, identity)
    if args.stage == "report":
        summarize(output, experiment, len(traces))
        return
    if os.geteuid() != 0:
        raise RuntimeError("Run serving/restore with sudo -E")
    tasks = floor.profile_jobs(settings, cfg, "full")
    with (output / "campaign.lock").open("a") as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hw = floor.FloorHardware(cfg)
        snapshot_path = output / "hardware_snapshot.json"
        pending_path = output / "restore_pending.json"
        if args.stage == "restore":
            if not snapshot_path.exists():
                raise RuntimeError("No fixed-trace serving hardware snapshot exists")
            hw.restore(json.loads(snapshot_path.read_text()))
            floor.v2.atomic(pending_path, {"pending": False})
            emit(hardware_restored=True)
            return
        if pending_path.exists() and json.loads(pending_path.read_text()).get("pending"):
            raise RuntimeError("Restoration pending; run restore before resuming")
        environment = floor.validation.validate_hardware(hw, cfg, diagnostic, tasks)
        environment_path = output / "environment.json"
        if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
            raise RuntimeError("Model/runtime/nvpmodel/platform changed")
        floor.v2.atomic(environment_path, environment)
        snapshot = hw.state()
        floor.v2.atomic(snapshot_path, snapshot)
        floor.v2.atomic(pending_path, {"pending": True})
        engine = power = None
        try:
            signal.signal(signal.SIGTERM, floor.validation.interrupted)
            signal.signal(signal.SIGINT, floor.validation.interrupted)
            with open("/run/energyinfra-v2-hardware.lock", "a") as hardware_lock:
                fcntl.flock(hardware_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                hw.apply({"mode": "dynamic"})
                engine = floor.v2.Tokens(cfg)
                power = floor.phase.Telemetry(output, cfg)
                live_settings = copy.deepcopy(settings)
                live_settings["serving"]["duration_s"] = experiment["duration_s"]
                live_settings["serving"]["repeats"] = len(traces)
                live_settings["_diagnostic"] = diagnostic
                trial_args = SimpleNamespace(trace=experiment["trace"], scenario=experiment["scenario"],
                                             slo=experiment["tpot_slo_ms"])
                for repeat, trace in enumerate(traces):
                    order = list(experiment["strategies"])
                    random.Random(cfg["seed"] + 31001 + repeat).shuffle(order)
                    for strategy in order:
                        target = output / f"result_{repeat}_{strategy}.json"
                        if target.exists():
                            continue
                        emit(stage="fixed_trace_serving", event="trial_start", repeat=repeat,
                             strategy=strategy, offered=len(trace))
                        result = floor.trial(engine, power, hw, cfg, live_settings, trial_args,
                                             trace, strategy, plans)
                        floor.v2.atomic(target, {"repeat": repeat, **result})
                        emit(stage="fixed_trace_serving", event="trial_complete", repeat=repeat,
                             strategy=strategy, attainment=result["observed_attainment"],
                             energy_j=result["total_energy_j"])
                        floor.trim_telemetry(power)
                summarize(output, experiment, len(traces))
        except BaseException as exc:
            floor.v2.atomic(output / "last_failure.json",
                            {"stage": args.stage, "error": repr(exc), "time": time.time()})
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                if power:
                    power.close()
            finally:
                try:
                    if engine:
                        engine.model.close()
                finally:
                    hw.restore(snapshot)
                    floor.v2.atomic(pending_path, {"pending": False})
                    emit(hardware_restored=True)
        emit(stage="fixed_trace_serving", complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "report", "restore"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = resolve(args.config)
    execute(args, *load_inputs(config_path))


if __name__ == "__main__":
    main()
