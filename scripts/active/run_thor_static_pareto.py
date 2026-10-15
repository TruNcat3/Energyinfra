#!/usr/bin/env python3
"""Reproduce Energyinfra's static-table -> Pareto-search workflow on Jetson Thor.

The workflow has two independent stages:

1. profile:
   Lock CPU/GPU/EMC frequencies, run the same workloads repeatedly, and write
   rows compatible with ``build_workload_rate_table.py``.
2. analyze:
   Aggregate the static table, compute Pareto ranks, and select a knee-point
   configuration for every workload and phase.

Run as root because the Jetson frequency controls are exposed through sysfs
and debugfs.  The default matrix is intentionally smaller than the original
research sweep; use ``--quick`` for a smoke-sized matrix and increase the
frequency/workload/repeat options for a full table.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.llama_cpp_runner import LlamaCppRunner
from src.controller.cap_controller import CapController
from src.controller.pareto_selector import ParetoSelector
from src.metrics.metrics_collector import MetricsCollector
from src.ratetable.build_workload_rate_table import (
    aggregate_lock_mode,
    compute_pareto_ranks,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger(__name__)


DEFAULT_WORKLOADS: Dict[str, Dict[str, int]] = {
    "p64_o64": {"prompt_length": 64, "output_length": 64},
    "p128_o512": {"prompt_length": 128, "output_length": 512},
    "p512_o128": {"prompt_length": 512, "output_length": 128},
    "p512_o512": {"prompt_length": 512, "output_length": 512},
    "p1024_o512": {"prompt_length": 1024, "output_length": 512},
    "p2048_o128": {"prompt_length": 2048, "output_length": 128},
    "p2048_o512": {"prompt_length": 2048, "output_length": 512},
}

THOR_EMC_DEFAULTS = [665, 2750, 3200, 4266]


def parse_int_list(value: str) -> List[int]:
    """Parse a comma-separated positive integer list."""
    result: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(float(item))
        if parsed <= 0:
            raise ValueError(f"frequency must be positive: {item}")
        result.append(parsed)
    if not result:
        raise ValueError(f"empty list: {value!r}")
    return result


def unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted(set(int(value) for value in values))


def evenly_spaced(values: Sequence[int], levels: int) -> List[int]:
    """Select approximately evenly spaced values while preserving endpoints."""
    ordered = unique_sorted(values)
    if not ordered:
        return []
    if levels <= 1 or len(ordered) <= levels:
        return ordered

    selected: List[int] = []
    for index in range(levels):
        position = round(index * (len(ordered) - 1) / (levels - 1))
        selected.append(ordered[position])
    return unique_sorted(selected)


def resolve_frequencies(
    controller: CapController,
    gpu_freqs: Optional[Sequence[int]],
    gpu_levels: int,
    emc_freqs: Sequence[int],
    cpu_freq: int,
) -> Tuple[List[int], List[int], int]:
    """Resolve requested MHz values to actual Thor-supported setpoints."""
    paths = controller.paths

    if gpu_freqs:
        resolved_gpu = [
            int(paths.resolve_gpu_hz(value) // 1_000_000)
            for value in gpu_freqs
        ]
    else:
        available_gpu = [
            int(value // 1_000_000) for value in paths.gpu_available_hz()
        ]
        if not available_gpu:
            available_gpu = [306, 510, 714, 918, 1122, 1326, 1530, 1575]
        selected_gpu = evenly_spaced(available_gpu, gpu_levels)
        resolved_gpu = [
            int(paths.resolve_gpu_hz(value) // 1_000_000)
            for value in selected_gpu
        ]

    resolved_emc = [
        int(paths.resolve_emc_hz(value) // 1_000_000)
        for value in emc_freqs
    ]
    resolved_cpu = int(paths.resolve_cpu_hz(cpu_freq) // 1_000)

    return unique_sorted(resolved_gpu), unique_sorted(resolved_emc), resolved_cpu


def select_workloads(spec: Optional[str], quick: bool) -> Dict[str, Dict[str, int]]:
    if spec:
        selected: Dict[str, Dict[str, int]] = {}
        for name in (item.strip() for item in spec.split(",")):
            if name not in DEFAULT_WORKLOADS:
                raise ValueError(
                    f"unknown workload {name!r}; choose from "
                    f"{', '.join(DEFAULT_WORKLOADS)}"
                )
            selected[name] = DEFAULT_WORKLOADS[name]
        return selected

    if quick:
        names = ["p64_o64", "p512_o128"]
        return {name: DEFAULT_WORKLOADS[name] for name in names}
    return dict(DEFAULT_WORKLOADS)


def energy_fields(
    collector: MetricsCollector,
    energy: Dict,
) -> Dict[str, float]:
    """Copy the repository energy fields and expose optional VIN power."""
    fields = {
        "avg_power_w": float(energy.get("avg_power_w", 0) or 0),
        "max_power_w": float(energy.get("max_power_w", 0) or 0),
        "total_energy_j": float(energy.get("total_energy_j", 0) or 0),
        "avg_gpu_soc_w": float(energy.get("avg_gpu_soc_w", 0) or 0),
        "avg_cpu_cv_w": float(energy.get("avg_cpu_cv_w", 0) or 0),
        "avg_sys_5v0_w": float(energy.get("avg_sys_5v0_w", 0) or 0),
        "avg_temp_cpu_c": float(energy.get("avg_temp_cpu_c", 0) or 0),
        "avg_gpu_freq_mhz": float(energy.get("avg_gpu_freq_mhz", 0) or 0),
        "avg_emc_freq_mhz": float(energy.get("avg_emc_freq_mhz", 0) or 0),
        "avg_vin_w": float(energy.get("avg_vin_w", 0) or 0),
        "power_samples": int(energy.get("n_samples", 0) or 0),
    }

    # VIN is the Thor module+carrier input power.  Keep it as an additional
    # column without changing the original rate-table power definition.
    fields.setdefault("avg_input_power_w", 0.0)
    try:
        samples = collector.get_samples_df()
        for column in ("vin_w", "input_power_w", "vin_power_w"):
            if column in samples.columns and not samples[column].empty:
                fields["avg_input_power_w"] = float(samples[column].mean())
                break
        if not fields["avg_input_power_w"] and energy.get("avg_vin_w"):
            fields["avg_input_power_w"] = float(energy["avg_vin_w"])
    except Exception:
        pass
    return fields


def run_one_static_measurement(
    runner: LlamaCppRunner,
    collector: MetricsCollector,
    controller: CapController,
    workload_name: str,
    workload: Dict[str, int],
    phase: str,
    gpu_mhz: int,
    emc_mhz: int,
    cpu_mhz: int,
    repeat: int,
    run_id: int,
    post_roll_ms: int,
) -> Dict:
    """Run one locked-frequency observation and return a rate-table row."""
    measured_prompt = 16 if phase == "decode" else workload["prompt_length"]
    tag = (
        f"thor_static_{run_id}_{workload_name}_{phase}_"
        f"g{gpu_mhz}_e{emc_mhz}_c{cpu_mhz}_r{repeat}_{time.time_ns()}"
    )

    collector.start_collection(tag=tag)
    try:
        result = runner.run_single_inference(
            prompt_length=measured_prompt,
            output_length=workload["output_length"],
            benchmark_mode=True,
        )
    finally:
        if post_roll_ms > 0:
            time.sleep(post_roll_ms / 1000.0)
        collector.stop_collection()

    energy = energy_fields(collector, collector.compute_energy())
    state = controller.read_actual_state()

    output_tokens = int(result.get("output_tokens", 0) or 0)
    total_energy = energy["total_energy_j"]
    if output_tokens > 0 and total_energy > 0:
        energy_per_token = total_energy / output_tokens
        tokens_per_joule = output_tokens / total_energy
    else:
        energy_per_token = 0.0
        tokens_per_joule = 0.0

    row = {
        "timestamp": datetime.now().isoformat(),
        "config_name": f"GPU{gpu_mhz}_EMC{emc_mhz}_CPU{cpu_mhz}",
        "gpu_freq_mhz": int(round(
            energy.get("avg_gpu_freq_mhz")
            or state.get("actual_gpu_mhz")
            or gpu_mhz
        )),
        "cpu_freq_mhz": int(state.get("actual_cpu_mhz") or cpu_mhz),
        "emc_freq_mhz": int(round(
            energy.get("avg_emc_freq_mhz")
            or state.get("actual_emc_mhz")
            or emc_mhz
        )),
        "target_gpu_mhz": gpu_mhz,
        "target_emc_mhz": emc_mhz,
        "target_cpu_mhz": cpu_mhz,
        "workload": workload_name,
        "prompt_length": workload["prompt_length"],
        "output_length": workload["output_length"],
        "phase": phase,
        "repeat": repeat,
        "model": Path(str(runner.model_path)).stem,
        "runtime": "llama.cpp",
        "experiment": "thor_static_lock",
        "measured_prompt_tokens": measured_prompt,
        "ttft_ms": float(result.get("ttft_ms", 0) or 0),
        "tpot_ms": float(result.get("tpot_ms", 0) or 0),
        "total_time_ms": float(result.get("total_time_ms", 0) or 0),
        "output_tokens": output_tokens,
        "tokens_per_second": float(result.get("tokens_per_second", 0) or 0),
        "energy_per_token_j": energy_per_token,
        "tokens_per_joule": tokens_per_joule,
        "temperature_start_c": float(state.get("temperature_c") or 0),
        "temperature_end_c": float(state.get("temperature_c") or 0),
        **energy,
    }
    return row


def profile(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    controller = CapController()
    requested_gpu = parse_int_list(args.gpu_freqs) if args.gpu_freqs else None
    requested_emc = (
        parse_int_list(args.emc_freqs)
        if args.emc_freqs
        else list(THOR_EMC_DEFAULTS)
    )
    gpu_freqs, emc_freqs, cpu_freq = resolve_frequencies(
        controller=controller,
        gpu_freqs=requested_gpu,
        gpu_levels=args.gpu_levels,
        emc_freqs=requested_emc,
        cpu_freq=args.cpu_freq,
    )

    workloads = select_workloads(args.workloads, args.quick)
    phases = [item.strip() for item in args.phases.split(",") if item.strip()]
    repeats = min(args.repeats, 1) if args.quick else args.repeats
    if args.quick and not args.gpu_freqs:
        gpu_freqs = evenly_spaced(gpu_freqs, 4)
    if args.quick and not args.emc_freqs:
        emc_freqs = [value for value in (2750, 4266) if value in emc_freqs]
    if args.quick:
        phases = ["decode"]

    total = len(gpu_freqs) * len(emc_freqs) * len(workloads) * len(phases) * repeats
    LOGGER.info(
        "Thor static profile: %d runs = %d GPU x %d EMC x %d workloads x "
        "%d phases x %d repeats",
        total,
        len(gpu_freqs),
        len(emc_freqs),
        len(workloads),
        len(phases),
        repeats,
    )
    LOGGER.info("Resolved GPU frequencies: %s MHz", gpu_freqs)
    LOGGER.info("Resolved EMC frequencies: %s MHz", emc_freqs)
    LOGGER.info("Resolved CPU frequency: %s MHz", cpu_freq)

    LOGGER.info("Loading model: %s", args.model)
    runner = LlamaCppRunner(
        model_path=args.model,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
    )
    collector = MetricsCollector(
        interval_ms=args.interval_ms,
        output_dir=args.raw_log_dir,
    )

    results: List[Dict] = []
    run_id = 0
    try:
        for gpu_mhz in gpu_freqs:
            for emc_mhz in emc_freqs:
                lock_result = controller.lock_config(gpu_mhz, emc_mhz, cpu_freq)
                if not lock_result.get("ok", False):
                    raise RuntimeError(f"lock failed: {lock_result}")
                time.sleep(args.settle_ms / 1000.0)
                LOGGER.info(
                    "Locked target GPU=%d EMC=%d CPU=%d MHz; "
                    "configured GPU=%s EMC=%s CPU=%s MHz; "
                    "idle GPU=%s EMC=%s CPU=%s MHz",
                    gpu_mhz,
                    emc_mhz,
                    cpu_freq,
                    lock_result.get("configured_gpu_mhz"),
                    lock_result.get("configured_emc_mhz"),
                    lock_result.get("configured_cpu_mhz"),
                    lock_result.get("actual_gpu_mhz"),
                    lock_result.get("actual_emc_mhz"),
                    lock_result.get("actual_cpu_mhz"),
                )
                runner.warmup(args.warmup_runs)

                for phase in phases:
                    for workload_name, workload in workloads.items():
                        for repeat in range(repeats):
                            run_id += 1
                            LOGGER.info(
                                "[%d/%d] %s %s %s repeat=%d",
                                run_id,
                                total,
                                f"GPU{gpu_mhz}_EMC{emc_mhz}_CPU{cpu_freq}",
                                workload_name,
                                phase,
                                repeat,
                            )
                            try:
                                row = run_one_static_measurement(
                                    runner=runner,
                                    collector=collector,
                                    controller=controller,
                                    workload_name=workload_name,
                                    workload=workload,
                                    phase=phase,
                                    gpu_mhz=gpu_mhz,
                                    emc_mhz=emc_mhz,
                                    cpu_mhz=cpu_freq,
                                    repeat=repeat,
                                    run_id=run_id,
                                    post_roll_ms=args.post_roll_ms,
                                )
                                results.append(row)
                                LOGGER.info(
                                    "  TPOT=%.2f ms TPS=%.2f P=%.2f W "
                                    "E/tok=%.5f J samples=%d",
                                    row["tpot_ms"],
                                    row["tokens_per_second"],
                                    row["avg_power_w"],
                                    row["energy_per_token_j"],
                                    row["power_samples"],
                                )
                            except Exception as exc:
                                LOGGER.exception("  measurement failed: %s", exc)
                                results.append(
                                    {
                                        "timestamp": datetime.now().isoformat(),
                                        "config_name": (
                                            f"GPU{gpu_mhz}_EMC{emc_mhz}_CPU{cpu_freq}"
                                        ),
                                        "gpu_freq_mhz": gpu_mhz,
                                        "cpu_freq_mhz": cpu_freq,
                                        "emc_freq_mhz": emc_mhz,
                                        "target_gpu_mhz": gpu_mhz,
                                        "target_emc_mhz": emc_mhz,
                                        "target_cpu_mhz": cpu_freq,
                                        "workload": workload_name,
                                        "prompt_length": workload["prompt_length"],
                                        "output_length": workload["output_length"],
                                        "phase": phase,
                                        "repeat": repeat,
                                        "model": Path(args.model).stem,
                                        "runtime": "llama.cpp",
                                        "experiment": "thor_static_lock",
                                        "output_tokens": 0,
                                        "total_energy_j": 0.0,
                                        "energy_per_token_j": 0.0,
                                        "error": str(exc),
                                    }
                                )
    finally:
        restore_result = controller.restore_dynamic()
        LOGGER.info("Restored dynamic mode: %s", restore_result)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"thor_static_profiling_{timestamp}.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    LOGGER.info("Saved %d rows to %s", len(results), output_path)
    return output_path


def json_value(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def analyze(input_csv: str, output_dir: str) -> Dict[str, str]:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    raw = pd.read_csv(input_path)
    valid = raw[
        (pd.to_numeric(raw["output_tokens"], errors="coerce") > 0)
        & (pd.to_numeric(raw["energy_per_token_j"], errors="coerce") > 0)
        & (pd.to_numeric(raw["tpot_ms"], errors="coerce") > 0)
    ].copy()
    if valid.empty:
        raise RuntimeError("static table has no valid rows")

    lock_agg = aggregate_lock_mode(valid)
    lock_agg = compute_pareto_ranks(lock_agg)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rate_table_path = out_dir / f"thor_lock_rate_table_{timestamp}.parquet"
    ranked_path = out_dir / f"thor_lock_pareto_ranked_{timestamp}.csv"
    frontier_path = out_dir / f"thor_pareto_frontiers_{timestamp}.csv"
    selection_path = out_dir / f"thor_pareto_selections_{timestamp}.json"
    report_path = out_dir / f"thor_pareto_report_{timestamp}.md"

    lock_agg.to_parquet(rate_table_path, index=False)
    lock_agg.to_csv(ranked_path, index=False)

    model = str(valid["model"].iloc[0])
    selector = ParetoSelector(
        lock_rate_table_path=str(rate_table_path),
        model=model,
    )

    frontier_rows: List[Dict] = []
    selections: List[Dict] = []
    for phase in sorted(valid["phase"].dropna().unique()):
        phase_data = valid[valid["phase"] == phase]
        for prompt_length, output_length in sorted(
            phase_data[["prompt_length", "output_length"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        ):
            frontier = selector.compute_frontier(
                int(prompt_length),
                int(output_length),
                phase=str(phase),
                source="lock",
            )
            if frontier.n_frontier == 0:
                continue

            for row in frontier.frontier_df.to_dict(orient="records"):
                frontier_rows.append(
                    {
                        "model": model,
                        "phase": phase,
                        "prompt_length": int(prompt_length),
                        "output_length": int(output_length),
                        **{key: json_value(value) for key, value in row.items()},
                    }
                )

            selected = selector.select_from_frontier(
                frontier,
                strategy="knee_point",
            )
            if selected:
                selections.append(
                    {
                        "model": model,
                        "phase": str(phase),
                        "prompt_length": int(prompt_length),
                        "output_length": int(output_length),
                        "frontier_size": frontier.n_frontier,
                        **{
                            key: json_value(value)
                            for key, value in selected.items()
                        },
                    }
                )

    pd.DataFrame(frontier_rows).to_csv(frontier_path, index=False)
    with selection_path.open("w") as handle:
        json.dump(selections, handle, indent=2, ensure_ascii=True)

    report_lines = [
        "# Thor Static Table Pareto Report",
        "",
        f"- Input: `{input_path}`",
        f"- Model: `{model}`",
        f"- Valid raw rows: {len(valid)}",
        f"- Aggregated lock configurations: {len(lock_agg)}",
        f"- Pareto points: {len(frontier_rows)}",
        f"- Selected workload-phase points: {len(selections)}",
        "",
        "| Phase | Workload | GPU MHz | EMC MHz | CPU MHz | E/token J | TPOT ms | Power W |",
        "|:---|:---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in selections:
        report_lines.append(
            f"| {item['phase']} | p{item['prompt_length']}_o{item['output_length']} | "
            f"{item.get('gpu_freq_mhz', 0)} | "
            f"{item.get('emc_freq_mhz', 0)} | "
            f"{item.get('cpu_freq_mhz', 0)} | "
            f"{item.get('energy_per_token_j', 0):.5f} | "
            f"{item.get('tpot_ms', 0):.2f} | "
            f"{item.get('avg_power_w', 0):.2f} |"
        )
    report_path.write_text("\n".join(report_lines) + "\n")

    outputs = {
        "rate_table": str(rate_table_path),
        "ranked_table": str(ranked_path),
        "frontiers": str(frontier_path),
        "selections": str(selection_path),
        "report": str(report_path),
    }
    LOGGER.info("Pareto analysis outputs: %s", json.dumps(outputs, indent=2))
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Thor static profiling and Pareto analysis"
    )
    parser.add_argument(
        "--mode",
        choices=("profile", "analyze", "all"),
        default="profile",
    )
    parser.add_argument(
        "--model",
        default="models/gguf/EnergyInfra-random-Qwen2.5-7B-Q4_K_M.gguf",
    )
    parser.add_argument("--input-csv")
    parser.add_argument("--output-dir", default="data/energy_profiling")
    parser.add_argument("--raw-log-dir", default="data/raw_logs")
    parser.add_argument("--gpu-freqs", help="requested GPU MHz, comma separated")
    parser.add_argument(
        "--gpu-levels",
        type=int,
        default=11,
        help="number of evenly spaced GPU levels when --gpu-freqs is omitted",
    )
    parser.add_argument(
        "--emc-freqs",
        help="requested Thor EMC MHz, comma separated; defaults to all 4 setpoints",
    )
    parser.add_argument("--cpu-freq", type=int, default=1497)
    parser.add_argument("--workloads", help="comma-separated workload names")
    parser.add_argument("--phases", default="decode,mixed")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--settle-ms", type=int, default=300)
    parser.add_argument("--post-roll-ms", type=int, default=50)
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-threads", type=int, default=4)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.mode in ("profile", "all"):
        profile_path = profile(args)
        if args.mode == "all":
            analyze(profile_path, args.output_dir)
    else:
        input_csv = args.input_csv
        if not input_csv:
            raise SystemExit("--input-csv is required with --mode analyze")
        analyze(input_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
