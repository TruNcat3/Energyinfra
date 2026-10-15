#!/usr/bin/env python3
"""Prepare targeted coupled-trace Thor capacity and TPOT-stability runs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.active import run_thor_joint_serving_stage1b_v1 as serving


DEFAULT_CONFIG = ROOT / "configs/experiments/thor_prompt_aware_targeted_v1.json"


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def immutable(path: Path, value: Any) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != serialized:
            raise RuntimeError(f"Protocol artifact changed; use a new output directory: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized)
    temporary.replace(path)


def analysis_config(run_dir: Path, scenario: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "energyinfra-thor-prompt-aware-slo-v1",
        "inputs": [str((run_dir / "results").relative_to(ROOT))],
        "output_dir": str((run_dir / f"prompt_aware_{analysis['name']}").relative_to(ROOT)),
        "ttft_schedules": [
            {
                "name": "tiered_strict",
                "limits_ms": {"128": 500, "512": 750, "1024": 1000},
            },
            {
                "name": "tiered_balanced",
                "limits_ms": {"128": 750, "512": 1000, "1024": 1500},
            },
        ],
        "tpot_thresholds_ms": analysis["tpot_thresholds_ms"],
        "required_attainment": 0.95,
        "expected_repeats": 3,
        "require_all_repeats": True,
        "require_each_prompt_bucket": bool(analysis["require_each_prompt_bucket"]),
        "pareto_objectives": [
            "avg_power_w",
            "energy_per_token_j",
            "ttft_p95_ms",
            "tpot_p95_ms",
        ],
    }


def prepare_one(config_path: Path, config: dict[str, Any], name: str) -> None:
    scenario = config["scenarios"][name]
    base_path = resolve(config["base_experiment"])
    experiment = copy.deepcopy(load(base_path))
    run_dir = resolve(scenario["output_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    rates = [int(value) for value in scenario["rates_tps"]]
    profiles = [f"rate_{rate:03d}" for rate in rates]
    selected = set(scenario["strategies"])
    experiment["seed"] = int(experiment["seed"]) + int(scenario["seed_offset"])
    experiment["output_dir"] = str(run_dir.relative_to(ROOT))
    experiment["trace_generation"]["workload_blocks"] = int(scenario["workload_blocks"])
    experiment["presets"] = {
        "full": {"duration_s": 120, "repeats": 3, "profiles": profiles}
    }
    experiment["load_profiles"] = {
        profile: {"offered_output_tps": float(rate)}
        for profile, rate in zip(profiles, rates)
    }
    experiment["strategies"] = [
        strategy for strategy in experiment["strategies"] if strategy["name"] in selected
    ]
    missing = selected - {strategy["name"] for strategy in experiment["strategies"]}
    if missing:
        raise RuntimeError(f"Unknown strategies for {name}: {sorted(missing)}")
    experiment["targeted_protocol"] = {
        "schema": config["schema"],
        "scenario": name,
        "source_config": str(config_path),
        "source_config_sha256": digest(config_path),
        "base_experiment": str(base_path),
        "base_experiment_sha256": digest(base_path),
    }
    experiment_path = run_dir / "experiment.json"
    immutable(experiment_path, experiment)
    for analysis in scenario["analyses"]:
        immutable(
            run_dir / f"analysis_{analysis['name']}.json",
            analysis_config(run_dir, scenario, analysis),
        )
    serving.prepare(experiment_path, "full", run_dir)
    print(json.dumps({
        "event": "targeted_prepared",
        "scenario": name,
        "rates": rates,
        "strategies": len(experiment["strategies"]),
        "traces": len(rates) * 3,
        "trials": len(rates) * 3 * len(experiment["strategies"]),
        "run_dir": str(run_dir),
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario key from the targeted config, or all",
    )
    args = parser.parse_args()
    if os.geteuid() == 0:
        raise RuntimeError("Preparation must run without sudo")
    config_path = resolve(args.config)
    config = load(config_path)
    if config.get("schema") != "energyinfra-thor-prompt-aware-targeted-v1":
        raise RuntimeError("Unsupported targeted experiment schema")
    if args.scenario != "all" and args.scenario not in config["scenarios"]:
        raise RuntimeError(
            f"Unknown scenario {args.scenario!r}; available: {sorted(config['scenarios'])}"
        )
    names = list(config["scenarios"]) if args.scenario == "all" else [args.scenario]
    for name in names:
        prepare_one(config_path, config, name)


if __name__ == "__main__":
    main()
