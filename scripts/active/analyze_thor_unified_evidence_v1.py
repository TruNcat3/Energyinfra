#!/usr/bin/env python3
"""Build a scenario-scoped Thor evidence table from completed protocol outputs.

This is an offline reader. It never loads a model, changes hardware controls, or
modifies an input protocol directory. The detailed CSV is the source of truth;
the Markdown report intentionally summarizes each scenario separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


EDGE_SLO_PROFILES = (
    ("interactive_strict", "interactive_strict_passed", "interactive_strict_attainment", "interactive_strict_all_repeats_feasible"),
    ("interactive", "interactive_passed", "interactive_attainment", "interactive_all_repeats_feasible"),
    ("edge_relaxed", "edge_relaxed_passed", "edge_relaxed_attainment", "edge_relaxed_all_repeats_feasible"),
)

OUTPUT_FIELDS = (
    "scenario",
    "source_kind",
    "workload",
    "slo_profile",
    "strategy",
    "point",
    "repeats",
    "complete_repeats",
    "fixed_work_ok",
    "offered",
    "completed",
    "passed",
    "observed_attainment",
    "all_repeats_slo_pass",
    "avg_power_w",
    "energy_per_token_j",
    "ttft_p95_ms",
    "tpot_p95_ms",
    "total_energy_j",
    "energy_delta_vs_native_pct",
    "energy_delta_vs_max_clock_pct",
    "source_file",
    "evidence_status",
    "notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def text(value: object) -> str:
    return "" if value is None else str(value)


def number(value: object) -> str:
    raw = text(value).strip()
    if not raw:
        return ""
    try:
        parsed = float(raw)
    except ValueError:
        return raw
    if parsed.is_integer():
        return str(int(parsed))
    return f"{parsed:.6f}".rstrip("0").rstrip(".")


def as_float(value: object) -> float | None:
    raw = text(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def as_bool(value: object) -> bool:
    return text(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def display(value: object, digits: int = 3) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "--"
    return f"{parsed:.{digits}f}"


def relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def source_record(root: Path, path: Path, kind: str) -> dict[str, object]:
    record: dict[str, object] = {"kind": kind, "path": relative(root, path)}
    if not path.is_file():
        record.update({"status": "missing", "rows": 0})
        return record
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record.update({"status": "available", "rows": len(read_csv(path)), "sha256": digest})
    return record


def base_row(
    *,
    scenario: str,
    source_kind: str,
    workload: str,
    slo_profile: str,
    strategy: str,
    point: str,
    source_file: str,
    evidence_status: str,
    notes: str,
) -> dict[str, str]:
    row = {field: "" for field in OUTPUT_FIELDS}
    row.update(
        {
            "scenario": scenario,
            "source_kind": source_kind,
            "workload": workload,
            "slo_profile": slo_profile,
            "strategy": strategy,
            "point": point,
            "source_file": source_file,
            "evidence_status": evidence_status,
            "notes": notes,
        }
    )
    return row


def add_edge_rows(root: Path, rows: list[dict[str, str]], scenario: str, path: Path) -> None:
    if not path.is_file():
        return
    for item in read_csv(path):
        for profile, passed_key, attainment_key, repeat_key in EDGE_SLO_PROFILES:
            row = base_row(
                scenario=scenario,
                source_kind="fixed_trace_serving",
                workload=item.get("profile", ""),
                slo_profile=profile,
                strategy=item.get("label") or item.get("strategy", ""),
                point=item.get("hardware", ""),
                source_file=relative(root, path),
                evidence_status="full",
                notes="Scenario-local result; do not rank against queue-aware or P8 rows.",
            )
            row.update(
                {
                    "repeats": number(item.get("repeats")),
                    "complete_repeats": text(item.get("complete_repeats")),
                    "fixed_work_ok": text(item.get("comparable_fixed_work")),
                    "offered": number(item.get("offered")),
                    "completed": number(item.get("completed_exact")),
                    "passed": number(item.get(passed_key)),
                    "observed_attainment": number(item.get(attainment_key)),
                    "all_repeats_slo_pass": text(item.get(repeat_key)),
                    "avg_power_w": number(item.get("fixed_horizon_avg_power_w")),
                    "energy_per_token_j": number(item.get("energy_per_completed_token_j")),
                    "ttft_p95_ms": number(item.get("ttft_p95_ms")),
                    "tpot_p95_ms": number(item.get("tpot_p95_ms")),
                    "total_energy_j": number(item.get("fixed_horizon_energy_j")),
                    "energy_delta_vs_native_pct": number(item.get("energy_per_token_delta_vs_native_pct")),
                }
            )
            rows.append(row)


def add_queue_rows(root: Path, rows: list[dict[str, str]], scenario: str, path: Path) -> None:
    if not path.is_file():
        return
    for item in read_csv(path):
        row = base_row(
            scenario=scenario,
            source_kind="queue_aware_serving",
            workload="frozen_arrival_trace",
            slo_profile="joint_slo",
            strategy=item.get("strategy", ""),
            point=item.get("strategy", ""),
            source_file=relative(root, path),
            evidence_status="full",
            notes="Queue/arrival-specific result; not a single-user edge result.",
        )
        row.update(
            {
                "repeats": number(item.get("repeats")),
                "complete_repeats": text(item.get("complete")),
                "fixed_work_ok": text(item.get("comparable_fixed_work")),
                "offered": number(item.get("offered")),
                "completed": number(item.get("completed")),
                "passed": number(item.get("passed")),
                "observed_attainment": number(item.get("observed_attainment")),
                "all_repeats_slo_pass": text(item.get("all_repeats_observed_slo_pass")),
                "avg_power_w": number(item.get("avg_power_w")),
                "energy_per_token_j": number(item.get("energy_per_token_j")),
                "ttft_p95_ms": number(item.get("ttft_end_to_end_p95_ms")),
                "tpot_p95_ms": number(item.get("tpot_p95_ms")),
                "total_energy_j": number(item.get("total_energy_j")),
                "energy_delta_vs_native_pct": number(item.get("total_energy_delta_vs_dynamic_pct")),
                "energy_delta_vs_max_clock_pct": number(item.get("total_energy_delta_vs_max_clock_pct")),
            }
        )
        rows.append(row)


def add_phase_rows(root: Path, rows: list[dict[str, str]], path: Path) -> None:
    if not path.is_file():
        return
    for item in read_csv(path):
        row = base_row(
            scenario="phase_static_validation",
            source_kind="static_candidate_validation",
            workload=item.get("workload", ""),
            slo_profile="see_slo_attainment.csv",
            strategy=item.get("role", ""),
            point=item.get("point", ""),
            source_file=relative(root, path),
            evidence_status="validation",
            notes="Static candidate validation; not phase-switch serving.",
        )
        row.update(
            {
                "repeats": number(item.get("blocks")),
                "complete_repeats": text(item.get("complete")),
                "fixed_work_ok": text(item.get("complete")),
                "avg_power_w": number(item.get("avg_power_w")),
                "energy_per_token_j": number(item.get("energy_per_token_j")),
                "ttft_p95_ms": number(item.get("ttft_p95_ms")),
                "tpot_p95_ms": number(item.get("tpot_p95_ms")),
                "total_energy_j": number(item.get("total_energy_j")),
            }
        )
        rows.append(row)


def add_pilot_rows(root: Path, rows: list[dict[str, str]], scenario: str, path: Path) -> None:
    if not path.is_file():
        return
    for item in read_csv(path):
        row = base_row(
            scenario=scenario,
            source_kind="matched_pilot",
            workload=item.get("workload", ""),
            slo_profile="declared_pilot_slo",
            strategy=item.get("point", ""),
            point=item.get("point", ""),
            source_file=relative(root, path),
            evidence_status="pilot",
            notes="Within-runtime sensitivity pilot; no model-quality conclusion.",
        )
        row.update(
            {
                "repeats": number(item.get("blocks")),
                "complete_repeats": text(item.get("complete")),
                "fixed_work_ok": text(item.get("complete")),
                "avg_power_w": number(item.get("avg_power_w")),
                "energy_per_token_j": number(item.get("energy_per_token_j")),
                "ttft_p95_ms": number(item.get("ttft_p95_ms")),
                "tpot_p95_ms": number(item.get("tpot_p95_ms")),
                "total_energy_j": number(item.get("total_energy_j")),
            }
        )
        rows.append(row)


def write_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    materialized = list(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(materialized)


def markdown_table(headers: list[str], values: Iterable[Iterable[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(text(value) for value in row) + " |" for row in values)
    return lines


def edge_summary(rows: list[dict[str, str]], scenario: str) -> list[str]:
    selected = [row for row in rows if row["scenario"] == scenario]
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        groups[row["slo_profile"]].append(row)
    lines = [f"## {scenario}", "", "The table reports feasible anchors within this scenario only.", ""]
    values = []
    for profile in sorted(groups):
        group = groups[profile]
        feasible = [row for row in group if as_bool(row["all_repeats_slo_pass"])]
        energy = min(feasible, key=lambda row: as_float(row["energy_per_token_j"]) or float("inf")) if feasible else None
        power = min(feasible, key=lambda row: as_float(row["avg_power_w"]) or float("inf")) if feasible else None
        values.append(
            (
                profile,
                f"{len(feasible)}/{len(group)}",
                energy["strategy"] if energy else "--",
                display(energy["energy_per_token_j"], 3) if energy else "--",
                power["strategy"] if power else "--",
                display(power["avg_power_w"], 2) if power else "--",
            )
        )
    lines.extend(markdown_table(["SLO profile", "all-repeat feasible", "energy anchor", "J/token", "power anchor", "W"], values))
    return lines


def queue_summary(rows: list[dict[str, str]], scenario: str) -> list[str]:
    selected = [row for row in rows if row["scenario"] == scenario]
    lines = [f"## {scenario}", ""]
    values = []
    for row in selected:
        values.append(
            (
                row["strategy"],
                row["all_repeats_slo_pass"] or "--",
                display(row["observed_attainment"], 3),
                display(row["avg_power_w"], 2),
                display(row["energy_per_token_j"], 3),
                display(row["ttft_p95_ms"], 1),
                display(row["tpot_p95_ms"], 1),
                display(row["energy_delta_vs_max_clock_pct"], 2),
            )
        )
    lines.extend(markdown_table(["Strategy", "All repeats", "Attainment", "W", "J/token", "TTFT P95", "TPOT P95", "Energy vs MaxClock %"], values))
    return lines


def pilot_summary(rows: list[dict[str, str]], scenario: str) -> list[str]:
    selected = [row for row in rows if row["scenario"] == scenario and row["workload"] == "[64, 64]"]
    lines = [f"## {scenario}", "", "Only the 64/64 anchor is shown here; `evidence_rows.csv` contains all workloads.", ""]
    values = []
    for row in selected:
        values.append((row["strategy"], row["repeats"], display(row["avg_power_w"], 2), display(row["energy_per_token_j"], 3), display(row["ttft_p95_ms"], 1), display(row["tpot_p95_ms"], 1)))
    lines.extend(markdown_table(["Point", "Blocks", "W", "J/token", "TTFT P95", "TPOT P95"], values))
    return lines


def render_report(rows: list[dict[str, str]], sources: list[dict[str, object]], root: Path) -> str:
    lines = [
        "# Thor unified evidence v1",
        "",
        "This is an offline, scenario-scoped aggregation of completed Thor protocol summaries.",
        "It does not reselect hardware points and does not rank serial, bursty, queue-aware or P8 results together.",
        "",
        f"- Data root: `{root}`",
        f"- Generated: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Detailed rows: `{len(rows)}`",
        "",
        "## Source status",
        "",
    ]
    lines.extend(markdown_table(["Kind", "Path", "Status", "Rows", "SHA256"], ((s.get("kind"), s.get("path"), s.get("status"), s.get("rows", 0), text(s.get("sha256", ""))[:12]) for s in sources)))
    lines.extend(["", "## Reading rules", "", "- A feasible anchor is selected only from rows whose declared all-repeat SLO flag is true.", "- Empty or missing source files remain missing; this report never substitutes another platform or scenario.", "- Pilot and static-validation rows are descriptive and do not establish online serving gains.", "- Use `evidence_rows.csv` for the complete table and the source protocol for SLO semantics.", ""])
    for scenario in ("edge_serial", "edge_bursty", "edge_confirmation_serial", "edge_confirmation_bursty"):
        lines.extend(edge_summary(rows, scenario))
        lines.append("")
    for scenario in ("queue_aware_validation", "queue_aware_confirmation"):
        if any(row["scenario"] == scenario for row in rows):
            lines.extend(queue_summary(rows, scenario))
            lines.append("")
    for scenario in ("phase_static_validation", "precision_q4", "precision_q5", "precision_q8", "scale_7b", "scale_14b"):
        if any(row["scenario"] == scenario for row in rows):
            if scenario == "phase_static_validation":
                lines.extend([f"## {scenario}", "", "Static validation is retained as a diagnostic table; consult the phase SLO file before claiming feasibility.", ""])
            else:
                lines.extend(pilot_summary(rows, scenario))
            lines.append("")
    lines.extend(["## Conclusion boundary", "", "This report is an evidence index, not a universal Thor ranking. A positive queue-aware result remains scoped to its frozen arrival trace; phase-switch rows remain separate from static candidate validation; precision and model-scale rows remain within-runtime pilots with synthetic weights."])
    return "\n".join(lines) + "\n"


def build(args: argparse.Namespace) -> None:
    root = Path(args.data_root).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    sources: list[dict[str, object]] = []

    edge_specs = (
        ("edge_serial", "fixed_trace_serving", root / "thor_edge_interactive_think_v1/full/summary.csv"),
        ("edge_bursty", "fixed_trace_serving", root / "thor_edge_low_concurrency_v1/full/summary.csv"),
        ("edge_confirmation_serial", "targeted_confirmation", root / "thor_edge_confirmation_v1/serial/full/summary.csv"),
        ("edge_confirmation_bursty", "targeted_confirmation", root / "thor_edge_confirmation_v1/bursty/full/summary.csv"),
    )
    for scenario, kind, path in edge_specs:
        sources.append(source_record(root, path, kind))
        add_edge_rows(root, rows, scenario, path)

    queue_specs = (
        ("queue_aware_validation", root / "thor_queue_aware_hardware_validation_v1/summary.csv"),
        ("queue_aware_confirmation", root / "thor_queue_aware_confirmation_v1/summary.csv"),
    )
    for scenario, path in queue_specs:
        sources.append(source_record(root, path, "queue_aware_serving"))
        add_queue_rows(root, rows, scenario, path)

    phase_path = root / "thor_phase_validation_full_v1/summary.csv"
    sources.append(source_record(root, phase_path, "static_candidate_validation"))
    add_phase_rows(root, rows, phase_path)

    pilot_specs = (
        ("precision_q4", root / "thor_precision_pareto_v1/q4_k_m_pilot/validation_summary.csv"),
        ("precision_q5", root / "thor_precision_pareto_v1/q5_k_m_pilot/validation_summary.csv"),
        ("precision_q8", root / "thor_precision_pareto_v1/q8_0_pilot/validation_summary.csv"),
        ("scale_7b", root / "thor_model_scale_v1/7b_pilot/validation_summary.csv"),
        ("scale_14b", root / "thor_model_scale_v1/14b_pilot/validation_summary.csv"),
    )
    for scenario, path in pilot_specs:
        sources.append(source_record(root, path, "matched_pilot"))
        add_pilot_rows(root, rows, scenario, path)

    write_csv(out / "evidence_rows.csv", rows)
    (out / "evidence_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (out / "RESULTS.md").write_text(render_report(rows, sources, root))
    print(json.dumps({"rows": len(rows), "sources": len(sources), "output": str(out)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/protocol_v2", help="Thor protocol_v2 directory")
    parser.add_argument("--out-dir", default="reports/thor_unified_evidence_v1", help="Output directory")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
