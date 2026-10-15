#!/usr/bin/env python3
"""Measure persistent llama-server resource/power policies on frozen arrival traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.active.run_energyinfra_v2 import Hardware, Power, atomic, file_hash


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_joint_serving_stage1b_v2.json"
STOP = threading.Event()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def object_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def emit(**values: Any) -> None:
    print(json.dumps(values, sort_keys=True), flush=True)


def percentile(values: list[float], fraction: float) -> float | None:
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    return ordered[max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))]


def wilson(passed: int, offered: int) -> list[float]:
    if not offered:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = passed / offered
    denominator = 1 + z * z / offered
    center = (p + z * z / (2 * offered)) / denominator
    radius = z * math.sqrt(p * (1 - p) / offered + z * z / (4 * offered * offered)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def on_signal(signum: int, _frame: Any) -> None:
    STOP.set()
    raise KeyboardInterrupt(f"received signal {signum}")


def load_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], Path, Path]:
    experiment = load_json(config_path)
    if experiment.get("schema") not in {
        "energyinfra-thor-joint-serving-stage1b-v1",
        "energyinfra-thor-joint-serving-stage1b-v2",
    }:
        raise RuntimeError("Unsupported Stage 1B schema")
    joint_path = resolve(experiment["joint_resource_config"])
    joint = load_json(joint_path)
    hardware = load_json(resolve(experiment["hardware_config"]))
    output = resolve(experiment["output_dir"])
    runtime_cfg = experiment.get("runtime")
    runtime_manifest_path = (
        resolve(runtime_cfg["manifest"])
        if runtime_cfg else resolve(joint["output_dir"]) / "runtime_manifest.json"
    )
    resource_summary_path = resolve(experiment["resource_screen_summary"])
    runtime_manifest = load_json(runtime_manifest_path)
    resource_summary = load_json(resource_summary_path)
    server = resolve(runtime_cfg["server_binary"] if runtime_cfg else joint["runtime"]["server_binary"])
    model = resolve(joint["model"]["path"])
    if runtime_cfg:
        if runtime_manifest.get("schema") != "energyinfra-thor-stage1b-patched-runtime-v1":
            raise RuntimeError("Stage 1B requires the isolated patched runtime manifest")
        if runtime_manifest.get("base_commit") != runtime_cfg["base_commit"]:
            raise RuntimeError("Patched runtime base commit mismatch")
        patch = resolve(runtime_cfg["patch"])
        source = resolve(runtime_cfg["source_dir"]) / "common/chat.cpp"
        if runtime_manifest.get("patch_sha256") != file_hash(patch):
            raise RuntimeError("Patched runtime patch hash mismatch")
        if runtime_manifest.get("patched_chat_sha256") != file_hash(source):
            raise RuntimeError("Patched runtime source hash mismatch")
    elif runtime_manifest["actual_commit"] != joint["runtime"]["commit"]:
        raise RuntimeError("Runtime commit does not match joint resource configuration")
    if runtime_manifest["server"]["sha256"] != file_hash(server):
        raise RuntimeError("llama-server binary hash differs from runtime manifest")
    if resource_summary.get("completed_jobs") != resource_summary.get("expected_jobs"):
        raise RuntimeError("Stage 1A resource screen is incomplete")
    for path in (server, model):
        if not path.exists():
            raise FileNotFoundError(path)
    paths = {
        "config": config_path,
        "joint": joint_path,
        "runtime_manifest": runtime_manifest_path,
        "resource_summary": resource_summary_path,
    }
    return experiment, joint, hardware, paths, server, model


def preset_settings(experiment: dict[str, Any], preset: str) -> tuple[int, int, list[str]]:
    settings = experiment["presets"][preset]
    return int(settings["duration_s"]), int(settings["repeats"]), list(settings["profiles"])


def weighted_workloads(experiment: dict[str, Any]) -> list[tuple[int, int]]:
    values = []
    for item in experiment["workload_mix"]:
        values.extend(
            [(int(item["prompt_tokens"]), int(item["output_tokens"]))] * int(item["weight"])
        )
    if not values:
        raise RuntimeError("Empty workload mix")
    return values


def make_coupled_stratified_trace(
    experiment: dict[str, Any], preset: str, profile: str, repeat: int
) -> dict[str, Any]:
    generation = experiment["trace_generation"]
    target_tps = float(experiment["load_profiles"][profile]["offered_output_tps"])
    blocks = int(generation.get("workload_blocks", 1))
    if blocks < 1:
        raise RuntimeError("trace_generation.workload_blocks must be positive")
    final_fraction = float(generation.get("final_arrival_fraction", 0.99))
    if not 0.0 < final_fraction < 1.0:
        raise RuntimeError("trace_generation.final_arrival_fraction must be between zero and one")
    rng = random.Random(
        int(experiment["seed"])
        + repeat * 1009
        + int(generation.get("seed_offset", 0))
    )
    choices = weighted_workloads(experiment)
    requests = []
    for _ in range(blocks):
        block = list(choices)
        rng.shuffle(block)
        for prompt, output in block:
            requests.append({
                "request_id": len(requests),
                "prompt_tokens": prompt,
                "output_tokens": output,
                "prompt_seed": rng.randrange(1, 2**31),
            })
    raw_arrivals = []
    current = 0.0
    for _ in requests:
        current += rng.expovariate(1.0)
        raw_arrivals.append(current)
    normalized_arrivals = [value / raw_arrivals[-1] for value in raw_arrivals]
    offered_tokens = sum(request["output_tokens"] for request in requests)
    duration = offered_tokens / target_tps
    for request, normalized in zip(requests, normalized_arrivals):
        request["scheduled_time_s"] = round(normalized * duration * final_fraction, 6)
    coupling_sha256 = object_hash({
        "requests": [
            {
                "request_id": request["request_id"],
                "prompt_tokens": request["prompt_tokens"],
                "output_tokens": request["output_tokens"],
                "prompt_seed": request["prompt_seed"],
            }
            for request in requests
        ],
        "normalized_arrivals": [round(value, 12) for value in normalized_arrivals],
    })
    return {
        "schema": "energyinfra-thor-joint-serving-trace-v1",
        "trace_generation": "coupled_stratified_v1",
        "coupling_sha256": coupling_sha256,
        "preset": preset,
        "profile": profile,
        "repeat": repeat,
        "duration_s": duration,
        "target_output_tps": target_tps,
        "offered_output_tokens": offered_tokens,
        "realized_offered_output_tps": offered_tokens / duration,
        "requests": requests,
    }


def make_interactive_think_time_trace(
    experiment: dict[str, Any], preset: str, profile: str, repeat: int
) -> dict[str, Any]:
    """Create a serial single-user trace with an explicit think-time gap."""
    generation = experiment["trace_generation"]
    trace = make_coupled_stratified_trace(experiment, preset, profile, repeat)
    think_time_s = float(generation.get("think_time_s", 30.0))
    tail_guard_s = float(generation.get("tail_guard_s", think_time_s))
    if think_time_s <= 0.0 or tail_guard_s <= 0.0:
        raise RuntimeError("interactive think-time values must be positive")
    for index, request in enumerate(trace["requests"]):
        request["scheduled_time_s"] = round(index * think_time_s, 6)
    trace["duration_s"] = round(
        max(trace["duration_s"], (len(trace["requests"]) - 1) * think_time_s + tail_guard_s),
        6,
    )
    trace["trace_generation"] = "interactive_think_time_v1"
    trace["realized_offered_output_tps"] = trace["offered_output_tokens"] / trace["duration_s"]
    trace["coupling_sha256"] = object_hash({
        "requests": trace["requests"],
        "duration_s": trace["duration_s"],
        "think_time_s": think_time_s,
    })
    return trace


def make_trace(experiment: dict[str, Any], preset: str, profile: str, repeat: int) -> dict[str, Any]:
    mode = experiment.get("trace_generation", {}).get("mode")
    if mode == "interactive_think_time_v1":
        return make_interactive_think_time_trace(experiment, preset, profile, repeat)
    if mode == "coupled_stratified_v1":
        return make_coupled_stratified_trace(experiment, preset, profile, repeat)
    duration, _, _ = preset_settings(experiment, preset)
    target_tps = float(experiment["load_profiles"][profile]["offered_output_tps"])
    target_tokens = int(round(duration * target_tps))
    rng = random.Random(int(experiment["seed"]) + repeat * 1009 + sum(map(ord, profile)))
    choices = weighted_workloads(experiment)
    requests = []
    offered_tokens = 0
    while offered_tokens < target_tokens:
        prompt, output = rng.choice(choices)
        requests.append({
            "request_id": len(requests),
            "prompt_tokens": prompt,
            "output_tokens": output,
            "prompt_seed": rng.randrange(1, 2**31),
        })
        offered_tokens += output
    raw_arrivals = []
    current = 0.0
    for _ in requests:
        current += rng.expovariate(1.0)
        raw_arrivals.append(current)
    final_arrival = max(0.0, duration - 1.0)
    scale = final_arrival / raw_arrivals[-1]
    for request, arrival in zip(requests, raw_arrivals):
        request["scheduled_time_s"] = round(arrival * scale, 6)
    return {
        "schema": "energyinfra-thor-joint-serving-trace-v1",
        "preset": preset,
        "profile": profile,
        "repeat": repeat,
        "duration_s": duration,
        "target_output_tps": target_tps,
        "offered_output_tokens": offered_tokens,
        "realized_offered_output_tps": offered_tokens / duration,
        "requests": requests,
    }


def trace_path(run_dir: Path, profile: str, repeat: int) -> Path:
    return run_dir / "traces" / f"trace_{profile}_{repeat}.json"


def protocol_identity(
    config_path: Path,
    experiment: dict[str, Any],
    paths: dict[str, Path],
    server: Path,
    model: Path,
    preset: str,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    duration, repeats, profiles = preset_settings(experiment, preset)
    stat = model.stat()
    return {
        "schema": experiment["schema"],
        "preset": preset,
        "duration_s": duration,
        "repeats": repeats,
        "profiles": profiles,
        "strategies": experiment["strategies"],
        "slo_profiles": experiment["slo_profiles"],
        "required_attainment": experiment["required_attainment"],
        "config": str(config_path),
        "config_sha256": file_hash(config_path),
        "source_sha256": file_hash(Path(__file__).resolve()),
        "joint_config_sha256": file_hash(paths["joint"]),
        "runtime_manifest_sha256": file_hash(paths["runtime_manifest"]),
        "resource_summary_sha256": file_hash(paths["resource_summary"]),
        "server": {"path": str(server), "sha256": file_hash(server)},
        "model": {"path": str(model), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "trace_hashes": {f"{trace['profile']}:{trace['repeat']}": object_hash(trace) for trace in traces},
        "measurement_scope": "tegrastats VIN input, fixed arrival horizon plus drain",
        "quality_scope": "random weights; systems behavior only",
    }


def prepare(config_path: Path, preset: str, run_dir: Path) -> None:
    experiment, _joint, _hardware, paths, server, model = load_inputs(config_path)
    duration, repeats, profiles = preset_settings(experiment, preset)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "traces").mkdir(exist_ok=True)
    traces = []
    for repeat in range(repeats):
        for profile in profiles:
            trace = make_trace(experiment, preset, profile, repeat)
            path = trace_path(run_dir, profile, repeat)
            if path.exists() and load_json(path) != trace:
                raise RuntimeError(f"Frozen trace changed: {path}")
            if not path.exists():
                atomic(path, trace)
            traces.append(trace)
    identity = protocol_identity(config_path, experiment, paths, server, model, preset, traces)
    protocol = run_dir / "protocol.json"
    if protocol.exists() and load_json(protocol) != identity:
        raise RuntimeError("Stage 1B protocol changed; use a new --run-dir")
    if not protocol.exists():
        atomic(protocol, identity)
    emit(event="prepared", preset=preset, duration_s=duration, traces=len(traces), run_dir=str(run_dir))


def validate_prepared(config_path: Path, preset: str, run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[dict[str, Any]]]:
    experiment, joint, hardware, paths, server, model = load_inputs(config_path)
    _duration, repeats, profiles = preset_settings(experiment, preset)
    traces = []
    for repeat in range(repeats):
        for profile in profiles:
            path = trace_path(run_dir, profile, repeat)
            if not path.exists():
                raise RuntimeError("Frozen traces are missing; run prepare first")
            trace = load_json(path)
            if trace != make_trace(experiment, preset, profile, repeat):
                raise RuntimeError(f"Frozen trace validation failed: {path}")
            traces.append(trace)
    expected = protocol_identity(config_path, experiment, paths, server, model, preset, traces)
    protocol = run_dir / "protocol.json"
    if not protocol.exists():
        raise RuntimeError("Stage 1B protocol validation failed; rerun prepare in a new directory")
    persisted = load_json(protocol)
    if persisted != expected:
        persisted_compatible = dict(persisted)
        expected_compatible = dict(expected)
        persisted_compatible.pop("source_sha256", None)
        expected_compatible.pop("source_sha256", None)
        if persisted_compatible != expected_compatible:
            raise RuntimeError("Stage 1B protocol validation failed; rerun prepare in a new directory")
    return experiment, hardware, server, model, traces


def prompt_tokens(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [100 + rng.randrange(100000) for _ in range(length)]


def wait_until(target: float) -> None:
    while True:
        if STOP.is_set():
            raise InterruptedError("campaign interrupted")
        remaining = target - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.05, remaining))


def post_json(url: str, payload: dict[str, Any] | None, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Server:
    def __init__(self, binary: Path, model: Path, settings: dict[str, Any], strategy: dict[str, Any], log: Path):
        self.settings = settings
        self.strategy = strategy
        self.url = f"http://{settings['host']}:{settings['port']}"
        context = int(settings["per_slot_context_tokens"]) * int(strategy["parallel"])
        self.command = [
            str(binary), "-m", str(model), "-c", str(context),
            "-b", str(strategy["batch"]), "-ub", str(strategy["ubatch"]),
            "-t", str(settings["threads"]), "-tb", str(settings["threads"]),
            "-ngl", "999", "-np", str(strategy["parallel"]), "--cont-batching",
            "--no-cache-prompt", "--host", str(settings["host"]),
            "--port", str(settings["port"]), "--reasoning-format", "none",
            "--skip-chat-parsing",
        ]
        log.parent.mkdir(parents=True, exist_ok=True)
        self.handle = log.open("w")
        self.proc = subprocess.Popen(
            self.command,
            cwd=ROOT,
            stdout=self.handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + float(settings["startup_timeout_s"])
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self.close()
                raise RuntimeError(f"llama-server exited during startup; see {log}")
            try:
                if post_json(self.url + "/health", None, 1).get("status") == "ok":
                    return
            except (OSError, ValueError, urllib.error.URLError):
                pass
            time.sleep(0.2)
        self.close()
        raise RuntimeError(f"llama-server startup timed out; see {log}")

    def warmup(self) -> None:
        response = post_json(
            self.url + "/completion",
            {
                "prompt": prompt_tokens(128, 17),
                "n_predict": 16,
                "temperature": 0,
                "ignore_eos": True,
                "cache_prompt": False,
                "stream": False,
                "logit_bias": [
                    [int(token_id), -100.0]
                    for token_id in self.settings.get("suppressed_token_ids", [])
                ],
            },
            float(self.settings["request_timeout_s"]),
        )
        if int(response.get("tokens_predicted", -1)) != 16:
            raise RuntimeError("llama-server warmup did not produce fixed output tokens")

    def close(self) -> None:
        if getattr(self, "proc", None) is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if getattr(self, "handle", None) is not None and not self.handle.closed:
            self.handle.close()


def request_once(server: Server, request: dict[str, Any], epoch: float) -> dict[str, Any]:
    scheduled = epoch + float(request["scheduled_time_s"])
    wait_until(scheduled)
    dispatched = time.monotonic()
    token_times: list[float] = []
    final: dict[str, Any] = {}
    received_final_stop = False
    final_stop_time = None
    error = None
    try:
        payload = {
            "prompt": prompt_tokens(int(request["prompt_tokens"]), int(request["prompt_seed"])),
            "n_predict": int(request["output_tokens"]),
            "temperature": 0,
            "ignore_eos": True,
            "cache_prompt": False,
            "stream": True,
            "logit_bias": [
                [int(token_id), -100.0]
                for token_id in server.settings.get("suppressed_token_ids", [])
            ],
        }
        http_request = urllib.request.Request(
            server.url + "/completion",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(http_request, timeout=float(server.settings["request_timeout_s"])) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                value = json.loads(line[6:])
                now = time.monotonic()
                if value.get("tokens"):
                    token_times.extend([now] * len(value["tokens"]))
                if value.get("stop"):
                    final = value
                    received_final_stop = True
                    final_stop_time = now
    except BaseException as exc:
        error = repr(exc)
    ended = time.monotonic()
    predicted = int(final.get("tokens_predicted", len(token_times)))
    required_stop_type = str(server.settings.get("required_stop_type", "limit"))
    exact = bool(
        error is None
        and received_final_stop
        and final.get("stop_type") == required_stop_type
        and predicted == int(request["output_tokens"])
    )
    first = token_times[0] if token_times else None
    stream_event_coverage = len(token_times) / predicted if predicted > 0 else 0.0
    return {
        "request_id": request["request_id"],
        "scheduled_time_s": request["scheduled_time_s"],
        "prompt_tokens": request["prompt_tokens"],
        "requested_output_tokens": request["output_tokens"],
        "output_tokens": predicted,
        "stream_token_events": len(token_times),
        "stream_event_gap": predicted - len(token_times),
        "stream_event_coverage": stream_event_coverage,
        "received_final_stop": received_final_stop,
        "completion_exact": exact,
        "dispatch_lag_ms": (dispatched - scheduled) * 1000.0,
        "ttft_end_to_end_ms": None if first is None else (first - scheduled) * 1000.0,
        "ttft_after_dispatch_ms": None if first is None else (first - dispatched) * 1000.0,
        "tpot_ms": None if first is None or final_stop_time is None or predicted < 2 else
                   (final_stop_time - first) * 1000.0 / (predicted - 1),
        "end_to_end_ms": (ended - scheduled) * 1000.0,
        "server_timings": final.get("timings"),
        "stop_type": final.get("stop_type"),
        "error": error,
    }


def apply_hardware(hw: Hardware, point: dict[str, Any]) -> dict[str, Any]:
    state = hw.apply(point)
    floor = point.get("cpu_floor_khz")
    if floor is None:
        return state
    available = set(hw.paths.cpu_available_hz())
    if int(floor) not in available:
        raise RuntimeError(f"Unsupported CPU floor {floor}; available={sorted(available)}")
    for path in hw.cpus:
        Hardware.domain(
            path,
            int(floor),
            int(point["cpu_khz"]),
            hw.p["cpu_dynamic_governor"],
            True,
        )
    return hw.state()


def run_trace(server: Server, trace: dict[str, Any], power: Power, minimum_samples: int) -> dict[str, Any]:
    requests = trace["requests"]
    epoch = time.monotonic() + float(server.settings["launch_lead_s"])
    workers = max(1, len(requests))
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = [executor.submit(request_once, server, request, epoch) for request in requests]
    try:
        responses = [future.result() for future in concurrent.futures.as_completed(futures)]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    horizon = epoch + float(trace["duration_s"])
    wait_until(horizon)
    ended = max(time.monotonic(), horizon)
    fixed_horizon = power.energy(epoch, horizon, minimum_samples)
    with_drain = power.energy(epoch, ended, minimum_samples)
    responses.sort(key=lambda value: value["request_id"])
    minimum_coverage = float(server.settings.get("minimum_stream_event_coverage", 0.98))
    incomplete = [
        response for response in responses
        if not response["completion_exact"]
        or response["stream_event_coverage"] < minimum_coverage
    ]
    if incomplete:
        details = [
            {
                "request_id": response["request_id"],
                "requested": response["requested_output_tokens"],
                "predicted": response["output_tokens"],
                "events": response["stream_token_events"],
                "event_coverage": response["stream_event_coverage"],
                "final_stop": response["received_final_stop"],
                "stop_type": response["stop_type"],
                "error": response["error"],
            }
            for response in incomplete[:8]
        ]
        raise RuntimeError(
            f"fixed-work integrity gate failed for {len(incomplete)}/{len(responses)} requests: {details}"
        )
    return {
        "measurement_start_monotonic": epoch,
        "duration_s": trace["duration_s"],
        "duration_with_drain_s": ended - epoch,
        "fixed_horizon": fixed_horizon,
        **with_drain,
        "responses": responses,
        "offered": len(requests),
        "completed_exact": sum(response["completion_exact"] for response in responses),
        "offered_output_tokens": sum(request["output_tokens"] for request in requests),
        "completed_output_tokens": sum(response["output_tokens"] for response in responses if response["completion_exact"]),
        "comparable_fixed_work": all(response["completion_exact"] for response in responses),
    }


def result_path(run_dir: Path, profile: str, repeat: int, strategy: str) -> Path:
    return run_dir / "results" / f"result_{profile}_{repeat}_{strategy}.json"


def release_ownership(path: Path) -> None:
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    for current, directories, files in os.walk(path):
        os.chown(current, int(uid), int(gid))
        for name in directories + files:
            os.chown(Path(current) / name, int(uid), int(gid))


def execute(config_path: Path, preset: str, run_dir: Path) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("Stage 1B run requires sudo -E for hardware controls")
    experiment, hardware_cfg, server_binary, model, traces = validate_prepared(config_path, preset, run_dir)
    results_dir = run_dir / "results"
    logs_dir = run_dir / "server_logs"
    telemetry_dir = run_dir / "telemetry"
    for directory in (results_dir, logs_dir, telemetry_dir):
        directory.mkdir(parents=True, exist_ok=True)
    jobs = [(trace, strategy) for trace in traces for strategy in experiment["strategies"]]
    pending = [job for job in jobs if not result_path(run_dir, job[0]["profile"], job[0]["repeat"], job[1]["name"]).exists()]
    if not pending:
        emit(event="nothing_pending", jobs=len(jobs))
        report(config_path, preset, run_dir)
        return
    with (run_dir / "campaign.lock").open("a") as campaign_lock:
        fcntl.flock(campaign_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with open("/run/energyinfra-v2-hardware.lock", "a") as hardware_lock:
            fcntl.flock(hardware_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            hw = Hardware(hardware_cfg)
            snapshot = hw.state()
            snapshot_path = run_dir / "hardware_snapshot.json"
            pending_path = run_dir / "restore_pending.json"
            if pending_path.exists() and load_json(pending_path).get("pending"):
                raise RuntimeError("Hardware restoration is pending; run restore before resuming")
            atomic(snapshot_path, snapshot)
            atomic(pending_path, {"pending": True})
            power = None
            active_server = None
            try:
                signal.signal(signal.SIGTERM, on_signal)
                signal.signal(signal.SIGINT, on_signal)
                order = list(pending)
                random.Random(int(experiment["seed"]) + (1 if preset == "quick" else 2)).shuffle(order)
                emit(event="start", preset=preset, total_jobs=len(jobs), pending_jobs=len(order))
                for index, (trace, strategy) in enumerate(order, 1):
                    if STOP.is_set():
                        raise InterruptedError("campaign interrupted")
                    profile = trace["profile"]
                    repeat = int(trace["repeat"])
                    target = result_path(run_dir, profile, repeat, strategy["name"])
                    point = experiment["hardware_points"][strategy["hardware"]]
                    emit(event="trial_start", progress=f"{index}/{len(order)}", profile=profile,
                         repeat=repeat, strategy=strategy["name"])
                    applied = apply_hardware(hw, point)
                    log = logs_dir / f"server_{profile}_{repeat}_{strategy['name']}.log"
                    telemetry_attempts = 2
                    for attempt in range(1, telemetry_attempts + 1):
                        if STOP.is_set():
                            raise InterruptedError("campaign interrupted")
                        power = Power(telemetry_dir, hardware_cfg)
                        active_server = Server(server_binary, model, experiment["server"], strategy, log)
                        try:
                            active_server.warmup()
                            time.sleep(float(experiment["server"]["settle_s"]))
                            measured = run_trace(
                                active_server,
                                trace,
                                power,
                                int(hardware_cfg["measurement"]["min_samples"]),
                            )
                            result = {
                                "schema": "energyinfra-thor-joint-serving-result-v2",
                                "preset": preset,
                                "profile": profile,
                                "repeat": repeat,
                                "trace_sha256": object_hash(trace),
                                "strategy": strategy,
                                "hardware_point": point,
                                "applied_hardware": applied,
                                "server_command": active_server.command,
                                "telemetry_attempt": attempt,
                                **measured,
                            }
                            atomic(target, result)
                            break
                        except RuntimeError as exc:
                            retryable = "Telemetry sample gap exceeds protocol limit" in str(exc)
                            if not retryable or attempt >= telemetry_attempts:
                                raise
                            emit(event="trial_retry", reason="telemetry_sample_gap",
                                 profile=profile, repeat=repeat, strategy=strategy["name"],
                                 next_attempt=attempt + 1)
                        finally:
                            active_server.close()
                            active_server = None
                            power.close()
                            power = None
                    emit(event="trial_complete", profile=profile, repeat=repeat,
                         strategy=strategy["name"], completed=result["completed_exact"],
                         offered=result["offered"], avg_power_w=result["fixed_horizon"]["avg_power_w"])
                    time.sleep(float(experiment["server"]["cooldown_s"]))
                report(config_path, preset, run_dir)
            except BaseException as exc:
                atomic(run_dir / "last_failure.json", {"error": repr(exc), "time": time.time()})
                raise
            finally:
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                if active_server is not None:
                    active_server.close()
                if power is not None:
                    power.close()
                hw.restore(snapshot)
                atomic(pending_path, {"pending": False})
                release_ownership(run_dir)
                emit(event="hardware_restored")


def request_slo_pass(response: dict[str, Any], slo: dict[str, Any]) -> bool:
    return bool(
        response["completion_exact"]
        and response["ttft_end_to_end_ms"] is not None
        and response["tpot_ms"] is not None
        and response["ttft_end_to_end_ms"] <= float(slo["ttft_ms"])
        and response["tpot_ms"] <= float(slo["tpot_ms"])
    )


def aggregate(experiment: dict[str, Any], preset: str, run_dir: Path) -> list[dict[str, Any]]:
    _duration, repeats, profiles = preset_settings(experiment, preset)
    grouped: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for path in (run_dir / "results").glob("result_*.json"):
        value = load_json(path)
        if value.get("schema") != "energyinfra-thor-joint-serving-result-v2":
            raise RuntimeError(f"Non-v2 or invalid result in Stage 1B v2 directory: {path}")
        if not value.get("comparable_fixed_work"):
            raise RuntimeError(f"Fixed-work integrity failed in persisted result: {path}")
        grouped[(value["profile"], value["strategy"]["name"])][int(value["repeat"])] = value
    rows = []
    strategy_map = {strategy["name"]: strategy for strategy in experiment["strategies"]}
    for profile in profiles:
        for strategy_name, strategy in strategy_map.items():
            by_repeat = grouped[(profile, strategy_name)]
            values = [by_repeat[index] for index in sorted(by_repeat)]
            if not values:
                continue
            responses = [response for value in values for response in value["responses"]]
            valid = [response for response in responses if response["completion_exact"]]
            fixed_duration = sum(float(value["duration_s"]) for value in values)
            fixed_energy = sum(float(value["fixed_horizon"]["total_energy_j"]) for value in values)
            drain_energy = sum(float(value["total_energy_j"]) for value in values)
            drain_duration = sum(float(value["duration_with_drain_s"]) for value in values)
            completed_tokens = sum(int(value["completed_output_tokens"]) for value in values)
            row = {
                "profile": profile,
                "strategy": strategy_name,
                "label": strategy["label"],
                "method_class": strategy["method_class"],
                "hardware": strategy["hardware"],
                "parallel": strategy["parallel"],
                "batch": strategy["batch"],
                "ubatch": strategy["ubatch"],
                "repeats": len(values),
                "complete_repeats": set(by_repeat) == set(range(repeats)),
                "comparable_fixed_work": len(values) == repeats and all(value["comparable_fixed_work"] for value in values),
                "offered": sum(int(value["offered"]) for value in values),
                "completed_exact": sum(int(value["completed_exact"]) for value in values),
                "offered_output_tokens": sum(int(value["offered_output_tokens"]) for value in values),
                "completed_output_tokens": completed_tokens,
                "stream_event_gap_total": sum(int(r["stream_event_gap"]) for r in responses),
                "stream_event_coverage_min": min(
                    (float(r["stream_event_coverage"]) for r in responses), default=None
                ),
                "fixed_horizon_energy_j": fixed_energy,
                "fixed_horizon_avg_power_w": fixed_energy / fixed_duration,
                "drain_energy_j": drain_energy,
                "drain_avg_power_w": drain_energy / drain_duration,
                "energy_per_completed_token_j": drain_energy / completed_tokens if completed_tokens else None,
                "output_throughput_with_drain_tps": completed_tokens / drain_duration if drain_duration else None,
                "ttft_p50_ms": percentile([r["ttft_end_to_end_ms"] for r in valid], 0.50),
                "ttft_p95_ms": percentile([r["ttft_end_to_end_ms"] for r in valid], 0.95),
                "tpot_p50_ms": percentile([r["tpot_ms"] for r in valid], 0.50),
                "tpot_p95_ms": percentile([r["tpot_ms"] for r in valid], 0.95),
                "dispatch_lag_p95_ms": percentile([r["dispatch_lag_ms"] for r in responses], 0.95),
                "slo": {},
            }
            for slo in experiment["slo_profiles"]:
                passed = sum(request_slo_pass(response, slo) for response in responses)
                repeat_attainment = []
                passed_tokens = 0
                for value in values:
                    repeat_passed = [response for response in value["responses"] if request_slo_pass(response, slo)]
                    repeat_attainment.append(len(repeat_passed) / value["offered"] if value["offered"] else 0.0)
                    passed_tokens += sum(response["output_tokens"] for response in repeat_passed)
                interval = wilson(passed, len(responses))
                row["slo"][slo["name"]] = {
                    "passed": passed,
                    "attainment": passed / len(responses) if responses else 0.0,
                    "wilson_ci95": interval,
                    "all_repeats_feasible": row["complete_repeats"] and all(
                        value >= float(experiment["required_attainment"]) for value in repeat_attainment
                    ),
                    "goodput_tps": passed_tokens / fixed_duration,
                    "pareto": False,
                }
            rows.append(row)
    for profile in profiles:
        baseline = next((row for row in rows if row["profile"] == profile and row["strategy"] == "native_dynamic_serial"), None)
        for row in [item for item in rows if item["profile"] == profile]:
            comparable = bool(baseline and baseline["comparable_fixed_work"] and row["comparable_fixed_work"])
            row["fixed_energy_delta_vs_native_pct"] = (
                100.0 * (row["fixed_horizon_energy_j"] / baseline["fixed_horizon_energy_j"] - 1.0)
                if comparable else None
            )
            row["energy_per_token_delta_vs_native_pct"] = (
                100.0 * (row["energy_per_completed_token_j"] / baseline["energy_per_completed_token_j"] - 1.0)
                if comparable else None
            )
        for slo in experiment["slo_profiles"]:
            feasible = [
                row for row in rows
                if row["profile"] == profile
                and row["slo"][slo["name"]]["all_repeats_feasible"]
                and row["energy_per_completed_token_j"] is not None
            ]
            objectives = ("fixed_horizon_avg_power_w", "energy_per_completed_token_j", "ttft_p95_ms", "tpot_p95_ms")
            for candidate in feasible:
                dominated = any(
                    all(other[key] <= candidate[key] for key in objectives)
                    and any(other[key] < candidate[key] for key in objectives)
                    for other in feasible if other is not candidate
                )
                candidate["slo"][slo["name"]]["pareto"] = not dominated
    return rows


def number(value: Any, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def render(experiment: dict[str, Any], preset: str, rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Thor persistent joint serving Stage 1B: {preset}",
        "",
        "Every strategy receives the same frozen arrival traces and exact prompt/output token counts.",
        "VIN energy includes the fixed arrival horizon and a separately reported drain interval; model loading is excluded.",
        "Random weights make this a systems comparison, not a model-quality comparison.",
        "",
        "## Related-work mapping",
        "",
        "| Work | Mechanism | Experimental mapping | Scope |",
        "|---|---|---|---|",
    ]
    for work in experiment.get("related_work", []):
        lines.append(
            f"| [{work['name']}]({work['url']}) | {work['mechanism']} | {work['mapping']} | {work['scope']} |"
        )
    lines += [
        "",
        "The Orca and Sarathi rows are mechanism-aligned proxies only. They must not be reported as native-system reproductions.",
        "",
    ]
    for profile in experiment["presets"][preset]["profiles"]:
        lines += [
            f"## Load: {profile}",
            "",
            "| Strategy | HW | P/B/U | Exact | W | J/token | Fixed energy vs native | TTFT p95 ms | TPOT p95 ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        selected = [row for row in rows if row["profile"] == profile]
        for row in selected:
            lines.append(
                f"| {row['label']} | {row['hardware']} | {row['parallel']}/{row['batch']}/{row['ubatch']} | "
                f"{row['completed_exact']}/{row['offered']} | {number(row['fixed_horizon_avg_power_w'])} | "
                f"{number(row['energy_per_completed_token_j'], 4)} | {number(row['fixed_energy_delta_vs_native_pct'])}% | "
                f"{number(row['ttft_p95_ms'])} | {number(row['tpot_p95_ms'])} |"
            )
        lines += [
            "",
            "| Strategy | Strict attainment/Pareto | Interactive attainment/Pareto | Relaxed attainment/Pareto |",
            "|---|---:|---:|---:|",
        ]
        slo_names = [slo["name"] for slo in experiment["slo_profiles"]]
        for row in selected:
            cells = []
            for name in slo_names:
                result = row["slo"][name]
                status = "P" if result["pareto"] else ("F" if result["all_repeats_feasible"] else "X")
                cells.append(f"{100 * result['attainment']:.1f}%/{status}")
            lines.append(f"| {row['label']} | {cells[0]} | {cells[1]} | {cells[2]} |")
        lines += [
            "",
            "`P` = SLO-feasible Pareto point, `F` = feasible but dominated, `X` = infeasible.",
            "",
        ]
    return "\n".join(lines) + "\n"


def report(config_path: Path, preset: str, run_dir: Path) -> None:
    experiment, _joint, _hardware, _paths, _server, _model = load_inputs(config_path)
    rows = aggregate(experiment, preset, run_dir)
    atomic(run_dir / "summary.json", rows)
    flat = []
    for row in rows:
        value = {key: item for key, item in row.items() if key != "slo"}
        for name, result in row["slo"].items():
            for key, item in result.items():
                value[f"{name}_{key}"] = item
        flat.append(value)
    if flat:
        with (run_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    (run_dir / "RESULTS.md").write_text(render(experiment, preset, rows))
    emit(event="report", rows=len(rows), output=str(run_dir / "RESULTS.md"))


def restore(config_path: Path, preset: str, run_dir: Path) -> None:
    if os.geteuid() != 0:
        raise RuntimeError("Restore requires sudo -E")
    _experiment, _joint, hardware_cfg, _paths, _server, _model = load_inputs(config_path)
    snapshot_path = run_dir / "hardware_snapshot.json"
    if not snapshot_path.exists():
        raise RuntimeError("No hardware snapshot exists")
    with open("/run/energyinfra-v2-hardware.lock", "a") as hardware_lock:
        fcntl.flock(hardware_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Hardware(hardware_cfg).restore(load_json(snapshot_path))
    atomic(run_dir / "restore_pending.json", {"pending": False})
    release_ownership(run_dir)
    emit(event="hardware_restored")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "report", "restore"])
    parser.add_argument("--preset", choices=["quick", "full"], default="quick")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config_path = resolve(args.config)
    experiment = load_json(config_path)
    run_dir = resolve(args.run_dir or (Path(experiment["output_dir"]) / args.preset))
    if args.stage == "prepare":
        prepare(config_path, args.preset, run_dir)
    elif args.stage == "run":
        execute(config_path, args.preset, run_dir)
    elif args.stage == "report":
        report(config_path, args.preset, run_dir)
    else:
        restore(config_path, args.preset, run_dir)


if __name__ == "__main__":
    main()
