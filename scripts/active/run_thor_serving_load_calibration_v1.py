#!/usr/bin/env python3
"""Calibrate a fixed serving arrival rate before comparing DVFS policies."""

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
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from scripts.active import run_cpu_floor_v1 as floor
from src.experiments.run_serving_benchmark import RequestGenerator


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_q4_long_generation_load_calibration_v1.json"


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def atomic(path, value):
    floor.v2.atomic(path, value)


def emit(**value):
    print(json.dumps(value, sort_keys=True), flush=True)


def scaled_traces(settings, cfg, experiment, phase, scale):
    duration = phase["duration_s"]
    traces = []
    for repeat in range(phase["repeats"]):
        base = RequestGenerator(experiment["trace"], duration, cfg["seed"] + repeat).generate_requests()
        trace = []
        for request in base:
            scheduled = request["scheduled_time"] * scale
            if scheduled > duration:
                break
            trace.append(dict(request, scheduled_time=scheduled))
        if not trace:
            raise RuntimeError(f"Arrival scale {scale:g} produced an empty trace")
        traces.append(trace)
    return traces


def percentile(values, fraction):
    values = sorted(v for v in values if v is not None)
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


def scale_name(scale):
    return f"arrival_x{scale:g}".replace(".", "p")


def summarize(phase_dir, experiment, phase):
    rows = []
    for directory in sorted(phase_dir.glob("arrival_x*")):
        protocol_path = directory / "protocol.json"
        if not protocol_path.exists():
            continue
        protocol = json.loads(protocol_path.read_text())
        results = [json.loads(path.read_text()) for path in sorted(directory.glob("result_*.json"))]
        if not results:
            continue
        responses = [response for result in results for response in result["responses"]]
        offered = sum(result["offered"] for result in results)
        passed = sum(result["passed"] for result in results)
        completed = sum(result["completed"] for result in results)
        energy = sum(result["total_energy_j"] for result in results)
        tokens = sum(result["output_tokens"] for result in results)
        duration = sum(result["duration_with_drain_s"] for result in results)
        complete = len(results) == phase["repeats"] and {r["repeat"] for r in results} == set(range(phase["repeats"]))
        attainment = passed / offered if offered else 0.0
        every_repeat = complete and all(
            result["observed_attainment"] >= experiment["required_observed_attainment"]
            for result in results
        )
        metrics = [response.get("metrics", {}) for response in responses]
        ttft = [item.get("ttft_end_to_end_ms") for item in metrics]
        tpot = [item.get("tpot_ms") for item in metrics]
        queue = [item.get("queue_wait_ms") for item in metrics]
        rows.append({
            "arrival_scale": protocol["arrival_scale"],
            "repeats": len(results),
            "complete": complete,
            "offered": offered,
            "completed": completed,
            "passed": passed,
            "observed_attainment": attainment,
            "wilson_ci95_low": wilson(passed, offered)[0],
            "wilson_ci95_high": wilson(passed, offered)[1],
            "every_repeat_observed_slo_pass": every_repeat,
            "offered_requests_per_s": offered / (len(results) * phase["duration_s"]),
            "ttft_end_to_end_p95_ms": percentile(ttft, 0.95),
            "queue_wait_p95_ms": percentile(queue, 0.95),
            "tpot_p95_ms": percentile(tpot, 0.95),
            "avg_power_w": energy / duration,
            "fixed_horizon_avg_power_w": sum(r["fixed_horizon"]["total_energy_j"] for r in results) /
                                          (len(results) * phase["duration_s"]),
            "energy_per_token_j": energy / tokens if tokens else None,
            "duration_with_drain_s": duration,
        })
    rows.sort(key=lambda row: row["arrival_scale"])
    candidate = next((row for row in rows if row["every_repeat_observed_slo_pass"]), None)
    atomic(phase_dir / "summary.json", rows)
    if rows:
        with (phase_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Thor long-generation arrival calibration",
        "",
        f"Phase: {phase_dir.name}",
        f"MaxClock only; TPOT <= {experiment['tpot_slo_ms']:g} ms and scenario {experiment['scenario']}.",
        "Arrival scale multiplies every original inter-arrival timestamp; larger means lighter load.",
        "This phase locates a load boundary and does not establish a DVFS energy win.",
        "",
        "| Arrival scale | Repeats | Completed/offered | SLO % | Every repeat passes | TTFT p95 ms | Queue p95 ms | TPOT p95 ms | W | J/token |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        number = lambda value: "--" if value is None else f"{value:.3f}"
        lines.append(
            f"| {row['arrival_scale']:g} | {row['repeats']} | {row['completed']}/{row['offered']} | "
            f"{100 * row['observed_attainment']:.2f} | {row['every_repeat_observed_slo_pass']} | "
            f"{number(row['ttft_end_to_end_p95_ms'])} | {number(row['queue_wait_p95_ms'])} | "
            f"{number(row['tpot_p95_ms'])} | {number(row['avg_power_w'])} | {number(row['energy_per_token_j'])} |"
        )
    lines.extend(["", "Observed highest-load candidate: " +
                  (f"arrival scale {candidate['arrival_scale']:g}." if candidate else "none in this scan.")])
    if phase_dir.name in ("coarse", "refine"):
        lines.append("A scan candidate must be rerun in the confirm phase before policy comparison.")
    (phase_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
    emit(calibration_report=str(phase_dir / "RESULTS.md"),
         observed_candidate=None if candidate is None else candidate["arrival_scale"])


def load_inputs(config_path):
    experiment = json.loads(config_path.read_text())
    if experiment["schema"] != "energyinfra-serving-load-calibration-v1":
        raise RuntimeError("Unsupported calibration schema")
    static_config_path = resolve(experiment["static_config"])
    settings = json.loads(static_config_path.read_text())
    diagnostic = json.loads(resolve(settings["diagnostic_config"]).read_text())
    cfg = copy.deepcopy(json.loads(resolve(diagnostic["base_config"]).read_text()))
    cfg["model"]["path"] = str(resolve(diagnostic["model_path"]))
    cfg["measurement"].update(diagnostic["measurement"])
    cfg["measurement"].update(settings["measurement"])
    return experiment, settings, diagnostic, cfg, static_config_path


def run(args, experiment, settings, diagnostic, cfg, static_config_path):
    phase = copy.deepcopy(experiment[args.phase])
    if args.phase == "confirm":
        if args.scale is None or args.scale <= 0:
            raise ValueError("Confirm requires --scale > 0")
        phase["arrival_scales"] = [args.scale]
    elif args.scale is not None:
        raise ValueError("--scale is only valid for confirm")
    scales = [float(scale) for scale in phase["arrival_scales"]]
    if not scales or any(scale < 1 for scale in scales):
        raise ValueError("Arrival scales must be >= 1")
    parent = resolve(experiment["parent_run_dir"])
    required_parent = [parent / "protocol.json", parent / "selection.json", parent / "comparison.json"]
    missing = [str(path) for path in required_parent if not path.exists()]
    if missing:
        raise RuntimeError(f"Frozen parent campaign is incomplete: {missing}")
    output = resolve(experiment["output_dir"])
    phase_dir = output / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": experiment["schema"],
        "phase": args.phase,
        "phase_settings": phase,
        "trace": experiment["trace"],
        "scenario": experiment["scenario"],
        "tpot_slo_ms": experiment["tpot_slo_ms"],
        "required_observed_attainment": experiment["required_observed_attainment"],
        "strategy": experiment["strategy"],
        "source_sha256": floor.v2.file_hash(Path(__file__).resolve()),
        "static_config_sha256": floor.v2.file_hash(static_config_path),
        "parent_files": {str(path.relative_to(ROOT)): floor.v2.file_hash(path) for path in required_parent},
        "trace_source_sha256": floor.v2.file_hash(ROOT / "src/experiments/run_serving_benchmark.py"),
    }
    protocol = phase_dir / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != identity:
        raise RuntimeError("Calibration protocol changed; use a new output directory")
    atomic(protocol, identity)
    if args.stage == "report":
        summarize(phase_dir, experiment, phase)
        return
    if os.geteuid() != 0:
        raise RuntimeError("Run calibration/restore with sudo -E")
    tasks = floor.profile_jobs(settings, cfg, "full")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "campaign.lock").open("a") as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hw = floor.FloorHardware(cfg)
        snapshot_path = output / "hardware_snapshot.json"
        pending_path = output / "restore_pending.json"
        if args.stage == "restore":
            if not snapshot_path.exists():
                raise RuntimeError("No calibration hardware snapshot exists")
            hw.restore(json.loads(snapshot_path.read_text()))
            atomic(pending_path, {"pending": False})
            emit(hardware_restored=True)
            return
        if pending_path.exists() and json.loads(pending_path.read_text()).get("pending"):
            raise RuntimeError("Restoration pending; run restore before resuming")
        environment = floor.validation.validate_hardware(hw, cfg, diagnostic, tasks)
        environment_path = output / "environment.json"
        if environment_path.exists() and json.loads(environment_path.read_text()) != environment:
            raise RuntimeError("Model/runtime/nvpmodel/platform changed during calibration")
        atomic(environment_path, environment)
        snapshot = hw.state()
        atomic(snapshot_path, snapshot)
        atomic(pending_path, {"pending": True})
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
                live_settings["serving"]["duration_s"] = phase["duration_s"]
                live_settings["serving"]["repeats"] = phase["repeats"]
                live_settings["_diagnostic"] = diagnostic
                order = list(scales)
                random.Random(cfg["seed"] + 29001).shuffle(order)
                for scale in order:
                    traces = scaled_traces(live_settings, cfg, experiment, phase, scale)
                    directory = phase_dir / scale_name(scale)
                    directory.mkdir(parents=True, exist_ok=True)
                    scale_protocol = {
                        "arrival_scale": scale,
                        "duration_s": phase["duration_s"],
                        "repeats": phase["repeats"],
                        "trace_hashes": [floor.v2.digest(trace) for trace in traces],
                        "phase_protocol_sha256": floor.v2.file_hash(protocol),
                    }
                    scale_manifest = directory / "protocol.json"
                    if scale_manifest.exists() and json.loads(scale_manifest.read_text()) != scale_protocol:
                        raise RuntimeError(f"Trace protocol changed for arrival scale {scale:g}")
                    atomic(scale_manifest, scale_protocol)
                    for repeat, trace in enumerate(traces):
                        atomic(directory / f"trace_{repeat}.json", trace)
                        target = directory / f"result_{repeat}_{experiment['strategy']}.json"
                        if target.exists():
                            continue
                        emit(stage="calibrate", event="trial_start", phase=args.phase,
                             arrival_scale=scale, repeat=repeat, offered=len(trace))
                        trial_args = SimpleNamespace(trace=experiment["trace"], scenario=experiment["scenario"],
                                                     slo=experiment["tpot_slo_ms"])
                        result = floor.trial(engine, power, hw, cfg, live_settings, trial_args, trace,
                                             experiment["strategy"], {})
                        atomic(target, {"repeat": repeat, "arrival_scale": scale, **result})
                        emit(stage="calibrate", event="trial_complete", phase=args.phase,
                             arrival_scale=scale, repeat=repeat,
                             attainment=result["observed_attainment"], energy_j=result["total_energy_j"])
                        floor.trim_telemetry(power)
                summarize(phase_dir, experiment, phase)
        except BaseException as exc:
            atomic(output / "last_failure.json", {"stage": args.stage, "phase": args.phase,
                                                   "error": repr(exc), "time": time.time()})
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
                    atomic(pending_path, {"pending": False})
                    emit(hardware_restored=True)
        emit(stage="calibrate", phase=args.phase, complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "report", "restore"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=["coarse", "refine", "confirm"], default="coarse")
    parser.add_argument("--scale", type=float)
    args = parser.parse_args()
    config_path = resolve(args.config)
    experiment, settings, diagnostic, cfg, static_config_path = load_inputs(config_path)
    run(args, experiment, settings, diagnostic, cfg, static_config_path)


if __name__ == "__main__":
    main()
