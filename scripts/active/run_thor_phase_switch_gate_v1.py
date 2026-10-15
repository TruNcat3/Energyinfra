#!/usr/bin/env python3
"""Paired fixed-work gate for first-token prefill/decode DVFS switching on Thor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import signal
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "active"))

import run_energyinfra_v2 as v2  # noqa: E402
import run_phase_diagnostic_v1 as phase  # noqa: E402

DEFAULT_CONFIG = ROOT / "configs/experiments/thor_phase_switch_gate_v1.json"
STOP = False


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def emit(**values: Any) -> None:
    print(json.dumps(values, sort_keys=True), flush=True)


def on_signal(signum: int, _frame: Any) -> None:
    global STOP
    STOP = True
    emit(event="signal", signum=signum, action="finish_current_token_request_then_restore")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def mean_ci95(values: list[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def load_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    experiment = load_json(config_path)
    diagnostic_path = resolve(experiment["diagnostic_config"])
    settings = load_json(diagnostic_path)
    base_path = resolve(settings["base_config"])
    base = load_json(base_path)
    base["model"]["path"] = str(resolve(settings["model_path"]))
    base["measurement"].update(settings["measurement"])
    return experiment, settings, base, diagnostic_path


def selected_matrix(experiment: dict[str, Any], preset: str) -> tuple[list[list[int]], list[dict[str, Any]], int]:
    workloads = experiment["workloads"]
    candidates = experiment["decode_candidates"]
    repeats = int(experiment["repeats"])
    if preset == "quick":
        quick = experiment["quick"]
        workloads = quick["workloads"]
        wanted = set(quick["candidate_names"])
        candidates = [candidate for candidate in candidates if candidate["name"] in wanted]
        repeats = int(quick["repeats"])
    return workloads, candidates, repeats


def protocol(
    config_path: Path,
    experiment: dict[str, Any],
    settings: dict[str, Any],
    diagnostic_path: Path,
    base: dict[str, Any],
    preset: str,
) -> dict[str, Any]:
    model_path = resolve(base["model"]["path"])
    stat = model_path.stat()
    return {
        "schema": "energyinfra-phase-switch-gate-protocol-v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "preset": preset,
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "diagnostic_config": str(diagnostic_path),
        "diagnostic_config_sha256": sha256(diagnostic_path),
        "model": {"path": str(model_path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "platform_version": settings.get("platform_version"),
        "prefill_point": experiment["prefill_point"],
        "decode_candidates": experiment["decode_candidates"],
        "workloads": experiment["workloads"],
        "measurement_scope": (
            "tegrastats VIN input energy from generation start through restoration of the prefill point; "
            "fixed generated token count; first-token-to-decode and decode-to-prefill control overhead included"
        ),
        "ttft_definition": "generation start to first sampled token; excludes post-first-token DVFS switch",
        "tpot_definition": "mean inter-token latency after first token; includes the first decode-switch interval",
        "quality_scope": "random-weight model; systems and fixed-work comparison only",
    }


def validate_or_write_protocol(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        existing = load_json(path)
        immutable = ["preset", "config_sha256", "diagnostic_config_sha256", "model"]
        if any(existing.get(key) != expected.get(key) for key in immutable):
            raise RuntimeError("Existing run protocol differs; use a new --run-dir")
        return
    atomic_json(path, expected)


def apply_checked(hw: Any, point: dict[str, Any], label: str) -> dict[str, Any]:
    result = hw.apply(point)
    if not isinstance(result, dict) or result.get("ok") is False or result.get("errors"):
        raise RuntimeError(f"hardware apply failed for {label}: {result}")
    return result


def run_request(
    engine: Any,
    hw: Any,
    prefill: dict[str, Any],
    decode: dict[str, Any] | None,
    prompt_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    if prompt_tokens < 1 or output_tokens < 2 or prompt_tokens + output_tokens > engine.ctx:
        raise ValueError("Invalid exact token lengths or context overflow")
    prompt = (engine.base * math.ceil(prompt_tokens / len(engine.base)))[:prompt_tokens]
    engine.model.reset()
    generator = engine.model.generate(prompt, temp=0.0, top_p=1.0, repeat_penalty=1.0, reset=True)
    tokens: list[int] = []
    times: list[float] = []
    switch_result = None
    switch_start = None
    switch_end = None
    restore_start = None
    restore_end = None
    start = time.monotonic()
    try:
        for index in range(output_tokens):
            tokens.append(int(next(generator)))
            times.append(time.monotonic())
            if index == 0 and decode is not None:
                switch_start = time.monotonic()
                switch_result = apply_checked(hw, decode, "decode")
                switch_end = time.monotonic()
    finally:
        generator.close()
    generation_end = time.monotonic()
    if decode is not None:
        restore_start = time.monotonic()
        restore_result = apply_checked(hw, prefill, "prefill restore")
        restore_end = time.monotonic()
    else:
        restore_result = None
    end = restore_end or generation_end
    itl_ms = [(right - left) * 1000 for left, right in zip(times, times[1:])]
    return {
        "prompt_tokens": len(prompt),
        "output_tokens": len(tokens),
        "token_sha256": v2.digest(tokens),
        "start": start,
        "generation_end": generation_end,
        "end": end,
        "ttft_ms": (times[0] - start) * 1000,
        "tpot_ms": (times[-1] - times[0]) * 1000 / (len(times) - 1),
        "first_itl_ms": itl_ms[0],
        "itl_p95_ms": percentile(itl_ms, 0.95),
        "itl_p99_ms": percentile(itl_ms, 0.99),
        "total_generation_ms": (generation_end - start) * 1000,
        "cycle_ms": (end - start) * 1000,
        "switch_latency_ms": None if switch_start is None else (switch_end - switch_start) * 1000,
        "restore_latency_ms": None if restore_start is None else (restore_end - restore_start) * 1000,
        "switch_result": switch_result,
        "restore_result": restore_result,
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{number}") from exc
    return rows


def release_ownership(path: Path) -> None:
    uid = os.environ.get("SUDO_UID")
    gid = os.environ.get("SUDO_GID")
    if uid is None or gid is None:
        return
    owner = (int(uid), int(gid))
    for item in [path, *path.rglob("*")]:
        try:
            os.chown(item, *owner)
        except FileNotFoundError:
            pass


def execute(config_path: Path, run_dir: Path, preset: str) -> None:
    if os.geteuid() != 0:
        raise PermissionError("run requires sudo for hardware controls")
    experiment, settings, base, diagnostic_path = load_inputs(config_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = run_dir / "protocol.json"
    validate_or_write_protocol(
        protocol_path, protocol(config_path, experiment, settings, diagnostic_path, base, preset)
    )
    workloads, candidates, repeats = selected_matrix(experiment, preset)
    known_points = phase.points(settings)
    prefill_name = experiment["prefill_point"]
    if prefill_name not in known_points:
        raise KeyError(f"Unknown prefill point: {prefill_name}")
    for candidate in candidates:
        if candidate["point"] is not None and candidate["point"] not in known_points:
            raise KeyError(f"Unknown decode point: {candidate['point']}")

    jobs = [
        {"workload": workload, "candidate": candidate, "repeat": repeat}
        for repeat in range(repeats)
        for workload in workloads
        for candidate in candidates
    ]
    random.Random(int(experiment["seed"])).shuffle(jobs)
    raw_path = run_dir / "raw_blocks.jsonl"
    rows = read_rows(raw_path)
    completed = {
        (tuple(row["workload"]), row["candidate"], int(row["repeat"])) for row in rows
    }
    pending = [
        job for job in jobs
        if (tuple(job["workload"]), job["candidate"]["name"], job["repeat"]) not in completed
    ]
    emit(event="start", preset=preset, completed=len(completed), pending=len(pending), run_dir=str(run_dir))
    if not pending:
        report(config_path, run_dir)
        release_ownership(run_dir)
        return

    hw = phase.IsolatedHardware(base)
    snapshot = hw.state()
    engine = None
    telemetry = None
    prefill = known_points[prefill_name]
    try:
        apply_checked(hw, prefill, prefill_name)
        engine = v2.Tokens(base)
        warmup = experiment["warmup"]
        for _ in range(int(warmup["requests"])):
            engine.run(int(warmup["prompt_tokens"]), int(warmup["output_tokens"]))
        telemetry_dir = run_dir / "telemetry"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry = phase.Telemetry(telemetry_dir, base)
        time.sleep(float(experiment["measurement"]["settle_s"]))
        with raw_path.open("a", buffering=1) as stream:
            for index, job in enumerate(pending, 1):
                if STOP:
                    break
                workload = [int(value) for value in job["workload"]]
                candidate = job["candidate"]
                decode = None if candidate["point"] is None else known_points[candidate["point"]]
                apply_checked(hw, prefill, prefill_name)
                time.sleep(float(experiment["measurement"]["settle_s"]))
                block_start = time.monotonic()
                requests = [
                    run_request(engine, hw, prefill, decode, workload[0], workload[1])
                    for _ in range(int(experiment["requests_per_block"]))
                ]
                block_end = time.monotonic()
                energy = telemetry.energy(
                    block_start,
                    block_end,
                    int(experiment["measurement"]["minimum_power_samples"]),
                )
                clocks = telemetry.clock_summary(block_start, block_end, decode or prefill, settings)
                state = hw.state()
                temperatures = [
                    value for key, value in state.items()
                    if "temperature" in key and isinstance(value, (int, float))
                ]
                maximum_temperature = max(temperatures) if temperatures else state.get("temperature_c")
                row = {
                    "schema": "energyinfra-phase-switch-block-v1",
                    "workload": workload,
                    "candidate": candidate["name"],
                    "decode_point": candidate["point"],
                    "prefill_point": prefill_name,
                    "repeat": int(job["repeat"]),
                    "block_start": block_start,
                    "block_end": block_end,
                    "duration_s": block_end - block_start,
                    "requests": requests,
                    "energy": energy,
                    "clocks": clocks,
                    "temperature_c": maximum_temperature,
                }
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                emit(
                    event="block",
                    progress=f"{index}/{len(pending)}",
                    workload=workload,
                    candidate=candidate["name"],
                    repeat=job["repeat"],
                    energy_j=round(energy["total_energy_j"], 3),
                    avg_power_w=round(energy["avg_power_w"], 3),
                    tpot_ms=round(statistics.fmean(r["tpot_ms"] for r in requests), 3),
                )
                limit = float(experiment["measurement"]["abort_temperature_c"])
                if maximum_temperature is not None and maximum_temperature >= limit:
                    raise RuntimeError(f"Temperature safety limit reached: {maximum_temperature} C")
    finally:
        if telemetry is not None:
            telemetry.close()
        restore_result = hw.restore(snapshot)
        emit(event="restore", result=restore_result)
        release_ownership(run_dir)
    if STOP:
        emit(event="stopped", completed=len(read_rows(raw_path)), expected=len(jobs), resumable=True)
        return
    report(config_path, run_dir)
    release_ownership(run_dir)


def aggregate(rows: list[dict[str, Any]], experiment: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[tuple[int, int], str], list[dict[str, Any]]] = defaultdict(list)
    baselines: dict[tuple[tuple[int, int], int], float] = {}
    for row in rows:
        workload = tuple(row["workload"])
        grouped[(workload, row["candidate"])].append(row)
        if row["candidate"] == "no_switch":
            count = len(row["requests"])
            baselines[(workload, int(row["repeat"]))] = row["energy"]["total_energy_j"] / count

    summaries = []
    for (workload, candidate), blocks in sorted(grouped.items()):
        requests = [request for block in blocks for request in block["requests"]]
        total_requests = len(requests)
        total_energy = sum(block["energy"]["total_energy_j"] for block in blocks)
        total_duration = sum(block["duration_s"] for block in blocks)
        output_tokens = sum(request["output_tokens"] for request in requests)
        paired = []
        for block in blocks:
            key = (workload, int(block["repeat"]))
            if key in baselines:
                current = block["energy"]["total_energy_j"] / len(block["requests"])
                paired.append((current / baselines[key] - 1) * 100)
        delta_mean, delta_low, delta_high = mean_ci95(paired)
        ttft = [request["ttft_ms"] for request in requests]
        tpot = [request["tpot_ms"] for request in requests]
        first_itl = [request["first_itl_ms"] for request in requests]
        switches = [
            request["switch_latency_ms"] for request in requests
            if request["switch_latency_ms"] is not None
        ]
        restores = [
            request["restore_latency_ms"] for request in requests
            if request["restore_latency_ms"] is not None
        ]
        attainment = {}
        for ttft_slo in experiment["slo"]["ttft_ms"]:
            for tpot_slo in experiment["slo"]["tpot_ms"]:
                key = f"ttft{ttft_slo}_tpot{tpot_slo}"
                attainment[key] = sum(
                    one_ttft <= ttft_slo and one_tpot <= tpot_slo
                    for one_ttft, one_tpot in zip(ttft, tpot)
                ) / total_requests
        summaries.append({
            "workload": list(workload),
            "candidate": candidate,
            "blocks": len(blocks),
            "requests": total_requests,
            "energy_per_request_j": total_energy / total_requests,
            "energy_per_output_token_j": total_energy / output_tokens,
            "avg_power_w": total_energy / total_duration,
            "ttft_p50_ms": percentile(ttft, 0.50),
            "ttft_p95_ms": percentile(ttft, 0.95),
            "tpot_p50_ms": percentile(tpot, 0.50),
            "tpot_p95_ms": percentile(tpot, 0.95),
            "first_itl_p95_ms": percentile(first_itl, 0.95),
            "switch_latency_p50_ms": percentile(switches, 0.50),
            "switch_latency_p95_ms": percentile(switches, 0.95),
            "restore_latency_p50_ms": percentile(restores, 0.50),
            "paired_energy_delta_pct_mean": delta_mean,
            "paired_energy_delta_pct_ci95_low": delta_low,
            "paired_energy_delta_pct_ci95_high": delta_high,
            "attainment": attainment,
        })

    required = float(experiment["slo"]["required_attainment"])
    gate = experiment["decision_gate"]
    primary_key = f"ttft{gate['primary_ttft_ms']}_tpot{gate['primary_tpot_ms']}"
    by_workload: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        by_workload[tuple(item["workload"])].append(item)
    for items in by_workload.values():
        feasible = [item for item in items if item["attainment"].get(primary_key, 0) >= required]
        for item in items:
            item["primary_slo"] = primary_key
            item["primary_slo_feasible"] = item in feasible
            item["energy_gate_pass"] = bool(
                item["candidate"] != "no_switch"
                and item["primary_slo_feasible"]
                and item["paired_energy_delta_pct_ci95_high"] is not None
                and item["paired_energy_delta_pct_ci95_high"] < 0
            )
            item["pareto"] = item in feasible and not any(
                other is not item
                and other["energy_per_output_token_j"] <= item["energy_per_output_token_j"]
                and other["avg_power_w"] <= item["avg_power_w"]
                and other["tpot_p95_ms"] <= item["tpot_p95_ms"]
                and (
                    other["energy_per_output_token_j"] < item["energy_per_output_token_j"]
                    or other["avg_power_w"] < item["avg_power_w"]
                    or other["tpot_p95_ms"] < item["tpot_p95_ms"]
                )
                for other in feasible
            )
    return {"rows": summaries, "primary_slo": primary_key}


def render_markdown(summary: dict[str, Any], status: dict[str, Any]) -> str:
    lines = [
        "# Thor phase-switch gate v1",
        "",
        f"Status: **{status['completed_blocks']}/{status['expected_blocks']} blocks** "
        f"(`{status['state']}`).",
        "",
        "Fixed output work is compared against `no_switch`. Energy includes both phase transitions; "
        "TTFT excludes the transition after the first token, while TPOT includes it.",
        "",
        "| Workload | Candidate | W | J/token | Energy vs no-switch (95% CI) | TTFT p95 ms | TPOT p95 ms | First ITL p95 ms | Switch p50 ms | Primary SLO | Pareto | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in summary["rows"]:
        delta = row["paired_energy_delta_pct_mean"]
        low = row["paired_energy_delta_pct_ci95_low"]
        high = row["paired_energy_delta_pct_ci95_high"]
        if delta is None:
            delta_text = "n/a"
        elif low is None or high is None:
            delta_text = f"{delta:+.3f}% (n=1)"
        else:
            delta_text = f"{delta:+.3f}% [{low:+.3f}, {high:+.3f}]"
        switch = row["switch_latency_p50_ms"]
        lines.append(
            f"| {row['workload'][0]}/{row['workload'][1]} | {row['candidate']} | "
            f"{row['avg_power_w']:.3f} | {row['energy_per_output_token_j']:.4f} | {delta_text} | "
            f"{row['ttft_p95_ms']:.2f} | {row['tpot_p95_ms']:.2f} | {row['first_itl_p95_ms']:.2f} | "
            f"{'n/a' if switch is None else f'{switch:.2f}'} | "
            f"{'pass' if row['primary_slo_feasible'] else 'fail'} | "
            f"{'yes' if row['pareto'] else 'no'} | {'pass' if row['energy_gate_pass'] else 'no'} |"
        )
    winners = [row for row in summary["rows"] if row["energy_gate_pass"]]
    lines += [
        "",
        "## Decision",
        "",
        (
            "At least one phase-switched point has a negative upper 95% confidence bound for energy "
            "and passes the primary SLO; it is eligible for fixed-trace serving validation."
            if winners else
            "No phase-switched point currently clears the fixed-work energy and primary-SLO gate. "
            "Lower average power alone is not treated as an energy-efficiency win."
        ),
        "",
        "This random-weight model supports systems conclusions only; it does not measure output quality.",
        "",
    ]
    return "\n".join(lines)


def report(config_path: Path, run_dir: Path) -> None:
    experiment, _settings, _base, _diagnostic_path = load_inputs(config_path)
    rows = read_rows(run_dir / "raw_blocks.jsonl")
    if not rows:
        raise RuntimeError("No raw blocks found")
    protocol_data = load_json(run_dir / "protocol.json")
    workloads, candidates, repeats = selected_matrix(experiment, protocol_data["preset"])
    expected = len(workloads) * len(candidates) * repeats
    status = {
        "state": "complete" if len(rows) == expected else "partial",
        "completed_blocks": len(rows),
        "expected_blocks": expected,
    }
    summary = aggregate(rows, experiment)
    summary.update(status)
    atomic_json(run_dir / "summary.json", summary)
    (run_dir / "RESULTS.md").write_text(render_markdown(summary, status))
    emit(event="report", run_dir=str(run_dir), **status)


def restore(config_path: Path) -> None:
    if os.geteuid() != 0:
        raise PermissionError("restore requires sudo")
    _experiment, settings, base, _diagnostic_path = load_inputs(config_path)
    hw = phase.IsolatedHardware(base)
    result = hw.apply(phase.points(settings)["dynamic"])
    emit(event="dynamic_restore", result=result)
    if not isinstance(result, dict) or result.get("ok") is False or result.get("errors"):
        raise RuntimeError(f"Dynamic restore failed: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["run", "report", "restore"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir")
    parser.add_argument("--preset", choices=["quick", "full"], default="full")
    args = parser.parse_args()
    config_path = resolve(args.config)
    experiment = load_json(config_path)
    run_dir = resolve(args.run_dir or experiment["output_dir"])
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    if args.stage == "run":
        execute(config_path, run_dir, args.preset)
    elif args.stage == "report":
        report(config_path, run_dir)
    else:
        restore(config_path)


if __name__ == "__main__":
    main()
