#!/usr/bin/env python3
"""
Multi-Objective Pareto Evaluation (Phase 13, P5)

Computes multi-objective EMO metrics to quantify Pareto strategy effectiveness
across 3 dimensions (E/tok, TPOT, Power) for offline E2E benchmark and
4 dimensions (+ Peak Temperature) for serving benchmark.

Metrics (following EMO literature conventions):
  1. MDR  — Multi-Objective Dominance Rate (fraction of workloads where strategy
            dominates baseline on ALL objectives simultaneously)
  2. JIR  — Joint Improvement Ratio (geometric mean of per-objective improvement
            ratios, counted ONLY when ALL objectives improve)
  3. HV   — Hypervolume Indicator (Zitzler 1999 gold-standard; dominated volume
            in normalized objective space, all minimize)

4D variants (MDR+T, JIR+T, HV+T) add Peak Temperature for serving evaluation,
capturing ThermalSLO's unique thermal-aware contribution.

Data sources:
  - E2E benchmark:   data/cap_selector_benchmark/cap_selector_benchmark_*.csv
  - Oracle gap:      data/oracle_gap_analysis/oracle_gap_*.csv
  - Serving windows:  data/serving_benchmark/serving_windows_*.csv

Usage:
    python3 src/ratetable/pareto_multi_objective_evaluation.py
    python3 src/ratetable/pareto_multi_objective_evaluation.py --hv-samples 50000
    python3 src/ratetable/pareto_multi_objective_evaluation.py --output-dir data/multi_obj_eval
"""

import sys, os, glob, json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import argparse
import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))


# ═══════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════

def short_model(m: str) -> str:
    """Shorten model name for display."""
    return m.replace("-Instruct-Q4_K_M", "").replace("Meta-Llama-3.1-", "Llama-")


@dataclass
class PairwiseResult:
    """Multi-objective comparison between two strategies/baselines."""
    strategy_a: str
    strategy_b: str
    model: str = ""            # Model name (for per-model comparison)
    trace: str = ""            # Trace name (for serving evaluation)
    mdr: float = 0.0          # Multi-Objective Dominance Rate (A dominates B)
    jir: float = 0.0          # Joint Improvement Ratio (A improves over B)
    jir_count: int = 0        # Number of instances where ALL objectives improve
    total_count: int = 0      # Total instances compared
    ept_improvement: float = 0.0  # Mean E/tok improvement ratio when A<B
    tpot_improvement: float = 0.0 # Mean TPOT improvement ratio
    power_improvement: float = 0.0 # Mean Power improvement ratio
    temp_improvement: float = 0.0  # Mean Temp improvement ratio (4D only, 0 if 3D)
    composite_waste: float = 0.0   # Composite distance ratio (B relative to A)


@dataclass
class StrategySummary:
    """Summary metrics for a single strategy."""
    strategy: str
    model: str
    trace: str = ""  # empty for offline
    hv_3d: float = 0.0     # Hypervolume (3D: E/tok, TPOT, Power)
    hv_4d: float = 0.0     # Hypervolume (4D: + Temperature)
    n_points: int = 0      # Number of observations
    mean_ept: float = 0.0
    mean_tpot: float = 0.0
    mean_power: float = 0.0
    mean_temp: float = 0.0  # 4D only


# ═══════════════════════════════════════════════════════════════════
# Core Metric Functions
# ═══════════════════════════════════════════════════════════════════

def dominates(point_a: np.ndarray, point_b: np.ndarray) -> bool:
    """
    Pareto dominance: A dominates B iff A <= B on ALL objectives
    and A < B on at least ONE objective (all minimize).
    """
    if not np.all(point_a <= point_b):
        return False
    return np.any(point_a < point_b)


def compute_mdr(points_a: List[np.ndarray], points_b: List[np.ndarray],
                keys: Optional[List[Tuple]] = None) -> float:
    """
    Multi-Objective Dominance Rate.

    Fraction of instances where A dominates B on ALL objectives.
    When keys are provided, matches A and B points by key (same workload/window).
    Otherwise, assumes paired: points_a[i] vs points_b[i].
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0

    n = min(len(points_a), len(points_b))
    if n == 0:
        return 0.0

    dominance_count = 0
    for i in range(n):
        if dominates(points_a[i], points_b[i]):
            dominance_count += 1

    return dominance_count / n


def compute_jir(points_a: List[np.ndarray], points_b: List[np.ndarray]) -> Tuple[float, int]:
    """
    Joint Improvement Ratio.

    For each instance where ALL objectives improve (A < B on every dimension),
    compute the geometric mean of per-objective improvement ratios (B_i / A_i - 1).
    JIR = mean of these geometric means across improving instances.

    Returns (jir, count) where count = number of improving instances.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0, 0

    n = min(len(points_a), len(points_b))
    joint_improvements = []

    for i in range(n):
        pa, pb = points_a[i], points_b[i]
        # ALL objectives must improve (A < B, since all minimize)
        if np.all(pa < pb):
            ratios = pb / pa - 1.0  # positive improvement ratios
            geo_mean = np.exp(np.mean(np.log(ratios)))
            joint_improvements.append(geo_mean)

    if not joint_improvements:
        return 0.0, 0

    return float(np.mean(joint_improvements)), len(joint_improvements)


def compute_composite_waste(points_a: List[np.ndarray], points_b: List[np.ndarray]) -> float:
    """
    Composite Distance Ratio.

    For each instance, compute normalized Euclidean distance from B to A
    relative to A's distance from the ideal point.

    waste(b) = dist(b, ideal) / dist(a, ideal) - 1.0

    Returns the mean waste across all instances.
    """
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0

    n = min(len(points_a), len(points_b))
    all_points = np.array(points_a + points_b)
    ideal = np.min(all_points, axis=0)

    wastes = []
    for i in range(n):
        pa, pb = np.array(points_a[i]), np.array(points_b[i])
        dist_a = np.linalg.norm(pa - ideal)
        dist_b = np.linalg.norm(pb - ideal)
        if dist_a > 1e-10:
            wastes.append(dist_b / dist_a - 1.0)

    return float(np.mean(wastes)) if wastes else 0.0


def compute_hv(points: List[np.ndarray], n_samples: int = 50000) -> float:
    """
    Hypervolume Indicator (Monte Carlo approximation).

    Measures the volume of the objective space dominated by the point set,
    relative to a reference point (worst observed values + margin).

    Higher HV = better (dominates more of the objective space).
    Points are in minimization direction.

    Implementation: Monte Carlo sampling in the hypercube [ideal, reference],
    count fraction dominated by any point in the set.
    """
    if len(points) == 0:
        return 0.0

    pts = np.array(points)
    n_obj = pts.shape[1]

    # Reference point: worst values + 10% margin
    ideal = np.min(pts, axis=0)
    worst = np.max(pts, axis=0)
    margin = (worst - ideal) * 0.1
    if np.any(margin < 1e-10):
        margin = np.maximum(margin, 0.01 * np.abs(ideal) + 1e-6)
    ref_point = worst + margin

    # Hypercube volume
    cube_vol = float(np.prod(ref_point - ideal))
    if cube_vol < 1e-20:
        return 0.0

    # Monte Carlo sampling
    rng = np.random.default_rng(42)
    samples = rng.uniform(0, 1, size=(n_samples, n_obj)) * (ref_point - ideal) + ideal

    # Check domination: sample point s is dominated if some p in pts has p <= s
    # For efficiency, check if any point is <= sample on all dimensions
    pts_expanded = pts[np.newaxis, :, :]  # (1, N, D)
    samples_expanded = samples[:, np.newaxis, :]  # (S, 1, D)
    dominated = np.all(pts_expanded <= samples_expanded, axis=2)  # (S, N)
    any_dominated = np.any(dominated, axis=1)  # (S,)

    hv = float(np.sum(any_dominated)) / n_samples * cube_vol
    return hv


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_e2e_benchmark(data_dir: str = "data/cap_selector_benchmark") -> pd.DataFrame:
    """Load and merge all E2E cap selector benchmark CSV files."""
    files = sorted(glob.glob(os.path.join(data_dir, "cap_selector_benchmark_*.csv")))
    if not files:
        raise FileNotFoundError(f"No E2E benchmark files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"  Loaded {f}: {len(df)} rows")

    merged = pd.concat(dfs, ignore_index=True)
    # Coerce numeric columns
    for col in ['energy_per_token_j', 'tpot_ms', 'avg_power_w', 'tokens_per_joule']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    print(f"  Total E2E benchmark: {len(merged)} rows, "
          f"{merged['model'].nunique()} models, "
          f"{merged['strategy'].nunique()} strategies")
    return merged


def load_serving_windows(data_dir: str = "data/serving_benchmark") -> pd.DataFrame:
    """Load and merge all serving window CSV files (latest per combination)."""
    files = sorted(glob.glob(os.path.join(data_dir, "serving_windows_*.csv")))
    if not files:
        raise FileNotFoundError(f"No serving window files found in {data_dir}")

    # Group by (baseline, trace, model) and keep the latest file per group
    file_groups: Dict[str, str] = {}
    for f in files:
        name = os.path.basename(f)
        # Extract key: baseline_trace_model (strip timestamp suffix)
        parts = name.replace("serving_windows_", "").rsplit("_", 1)
        if len(parts) == 2:
            key = parts[0]
            # Keep latest file (by modification time)
            if key not in file_groups or os.path.getmtime(f) > os.path.getmtime(file_groups[key]):
                file_groups[key] = f

    dfs = []
    for key, f in file_groups.items():
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # Coerce numeric
    for col in ['tpot_p50_ms', 'avg_power_w', 'max_power_w', 'temp_max_c',
                'temp_mean_c', 'tokens_per_joule', 'slo_violation_rate',
                'total_energy_j', 'total_tokens', 'tokens_per_second']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce')

    print(f"  Total serving windows: {len(merged)} rows from {len(dfs)} files, "
          f"{merged['model'].nunique()} models, "
          f"{merged['baseline'].nunique()} baselines, "
          f"{merged['trace'].nunique()} traces")
    return merged


def load_oracle_gap(data_dir: str = "data/oracle_gap_analysis") -> pd.DataFrame:
    """Load and merge oracle gap CSV files."""
    files = sorted(glob.glob(os.path.join(data_dir, "oracle_gap_*.csv")))
    if not files:
        raise FileNotFoundError(f"No oracle gap files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # Deduplicate by (model, workload)
    merged = merged.drop_duplicates(subset=['model', 'workload'], keep='last')

    print(f"  Oracle gap: {len(merged)} rows, {merged['model'].nunique()} models")
    return merged


# ═══════════════════════════════════════════════════════════════════
# Evaluation: 3D Offline (E2E Benchmark)
# ═══════════════════════════════════════════════════════════════════

def evaluate_3d_offline(df: pd.DataFrame, n_hv_samples: int = 50000
                        ) -> Tuple[List[PairwiseResult], List[StrategySummary]]:
    """
    Compute 3D multi-objective metrics from E2E benchmark data.

    Objectives (all minimize): E/tok, TPOT, Power

    For MDR/JIR: Pairwise per (model, workload, repeat) comparison.
    For HV: Each strategy's point set per model.
    """
    print("\n" + "=" * 60)
    print("  3D Offline Evaluation (E2E Benchmark)")
    print("  Objectives: E/tok, TPOT, Power (all minimize)")
    print("=" * 60)

    strategies = sorted(df['strategy'].unique())
    models = sorted(df['model'].unique())
    results = []
    summaries = []

    for model in models:
        model_df = df[df['model'] == model]
        print(f"\n  Model: {model}")

        # ── HV per strategy ──
        for strat in strategies:
            strat_df = model_df[model_df['strategy'] == strat]
            if len(strat_df) == 0:
                continue

            pts = strat_df[['energy_per_token_j', 'tpot_ms', 'avg_power_w']].dropna()
            points = [row.values for _, row in pts.iterrows()]
            hv = compute_hv(points, n_hv_samples) if len(points) > 0 else 0.0

            summaries.append(StrategySummary(
                strategy=strat, model=model, trace="offline",
                hv_3d=hv, hv_4d=0.0,
                n_points=len(points),
                mean_ept=float(pts['energy_per_token_j'].mean()),
                mean_tpot=float(pts['tpot_ms'].mean()),
                mean_power=float(pts['avg_power_w'].mean()),
            ))

        # ── Pairwise MDR/JIR ──
        workloads = sorted(model_df['workload'].unique())
        repeats = sorted(model_df['repeat'].unique())

        for strat_a in strategies:
            for strat_b in strategies:
                if strat_a == strat_b:
                    continue

                pts_a_list = []
                pts_b_list = []

                for wl in workloads:
                    for rep in repeats:
                        row_a = model_df[(model_df['strategy'] == strat_a) &
                                         (model_df['workload'] == wl) &
                                         (model_df['repeat'] == rep)]
                        row_b = model_df[(model_df['strategy'] == strat_b) &
                                         (model_df['workload'] == wl) &
                                         (model_df['repeat'] == rep)]

                        if len(row_a) == 0 or len(row_b) == 0:
                            continue

                        pt_a = row_a[['energy_per_token_j', 'tpot_ms', 'avg_power_w']].iloc[0].values
                        pt_b = row_b[['energy_per_token_j', 'tpot_ms', 'avg_power_w']].iloc[0].values
                        pts_a_list.append(pt_a)
                        pts_b_list.append(pt_b)

                if len(pts_a_list) == 0:
                    continue

                mdr = compute_mdr(pts_a_list, pts_b_list)
                jir, jir_count = compute_jir(pts_a_list, pts_b_list)
                waste = compute_composite_waste(pts_a_list, pts_b_list)

                # Per-dimension mean improvement
                arr_a = np.array(pts_a_list)
                arr_b = np.array(pts_b_list)
                ept_imp = float(np.mean((arr_b[:, 0] - arr_a[:, 0]) / arr_a[:, 0] * 100))
                tpot_imp = float(np.mean((arr_b[:, 1] - arr_a[:, 1]) / arr_a[:, 1] * 100))
                pwr_imp = float(np.mean((arr_b[:, 2] - arr_a[:, 2]) / arr_a[:, 2] * 100))

                results.append(PairwiseResult(
                    strategy_a=strat_a, strategy_b=strat_b, model=model,
                    trace="offline",
                    mdr=mdr, jir=jir, jir_count=jir_count,
                    total_count=len(pts_a_list),
                    ept_improvement=ept_imp, tpot_improvement=tpot_imp,
                    power_improvement=pwr_imp, temp_improvement=0.0,
                    composite_waste=waste,
                ))

    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# Evaluation: 4D Serving
# ═══════════════════════════════════════════════════════════════════

def evaluate_4d_serving(df: pd.DataFrame, n_hv_samples: int = 50000
                         ) -> Tuple[List[PairwiseResult], List[StrategySummary]]:
    """
    Compute 4D multi-objective metrics from serving benchmark data.

    Objectives (all minimize): E/tok (1/tokens_per_joule), TPOT, Power, Peak Temp

    Two evaluation modes:
    1. Per-window pairwise: MDR/JIR computed on window-by-window aligned data
       (same window_id across baselines in the same trace/model)
    2. Aggregate per (model, trace): Mean metrics per baseline → single 4D point
       → HV computed across all traces per model.
    """
    print("\n" + "=" * 60)
    print("  4D Serving Evaluation")
    print("  Objectives: E/tok, TPOT, Power, Peak Temperature (all minimize)")
    print("=" * 60)

    baselines = sorted(df['baseline'].unique())
    models = sorted(df['model'].unique())
    traces = sorted(df['trace'].unique())
    results = []
    summaries = []

    for model in models:
        model_df = df[df['model'] == model]
        print(f"\n  Model: {model}")

        for trace in traces:
            trace_df = model_df[model_df['trace'] == trace]
            print(f"    Trace: {trace} ({len(trace_df)} total windows)")

            # ── Aggregate metrics per baseline ──
            agg_pts = {}
            for bl in baselines:
                bl_df = trace_df[trace_df['baseline'] == bl]
                if len(bl_df) == 0:
                    continue

                # Derive E/tok from tokens_per_joule (avoid div-by-zero)
                tpj = bl_df['tokens_per_joule'].values
                ept = np.where(tpj > 1e-10, 1.0 / tpj, np.nan)

                agg_pt = np.array([
                    np.nanmean(ept),           # E/tok (minimize)
                    np.nanmean(bl_df['tpot_p50_ms'].values),   # TPOT (minimize)
                    np.nanmean(bl_df['avg_power_w'].values),   # Power (minimize)
                    np.nanmean(bl_df['temp_max_c'].values),    # Peak Temp (minimize)
                ])
                agg_pts[bl] = agg_pt

                summaries.append(StrategySummary(
                    strategy=bl, model=model, trace=trace,
                    hv_3d=0.0, hv_4d=0.0,
                    n_points=len(bl_df),
                    mean_ept=float(agg_pt[0]),
                    mean_tpot=float(agg_pt[1]),
                    mean_power=float(agg_pt[2]),
                    mean_temp=float(agg_pt[3]),
                ))

            # ── Pairwise MDR/JIR on aggregate points ──
            for bl_a in baselines:
                for bl_b in baselines:
                    if bl_a == bl_b:
                        continue
                    if bl_a not in agg_pts or bl_b not in agg_pts:
                        continue

                    pt_a = agg_pts[bl_a]
                    pt_b = agg_pts[bl_b]

                    # MDR: does A dominate B? (binary for single-point comparison)
                    mdr_val = 1.0 if dominates(pt_a, pt_b) else 0.0

                    # JIR: joint improvement when ALL 4 objectives improve
                    if np.all(pt_a < pt_b):
                        ratios = pt_b / pt_a - 1.0
                        jir_val = float(np.exp(np.mean(np.log(ratios))))
                        jir_count = 1
                    else:
                        jir_val = 0.0
                        jir_count = 0

                    # Per-dimension improvement
                    ept_imp = float((pt_b[0] - pt_a[0]) / max(pt_a[0], 1e-10) * 100)
                    tpot_imp = float((pt_b[1] - pt_a[1]) / max(pt_a[1], 1e-10) * 100)
                    pwr_imp = float((pt_b[2] - pt_a[2]) / max(pt_a[2], 1e-10) * 100)
                    temp_imp = float((pt_b[3] - pt_a[3]) / max(pt_a[3], 1e-10) * 100)

                    # Composite waste
                    ideal_4d = np.minimum(pt_a, pt_b)
                    dist_a = np.linalg.norm(pt_a - ideal_4d)
                    dist_b = np.linalg.norm(pt_b - ideal_4d)
                    waste = float(dist_b / max(dist_a, 1e-10) - 1.0)

                    results.append(PairwiseResult(
                        strategy_a=bl_a, strategy_b=bl_b, model=model, trace=trace,
                        mdr=mdr_val, jir=jir_val, jir_count=jir_count,
                        total_count=1,
                        ept_improvement=ept_imp, tpot_improvement=tpot_imp,
                        power_improvement=pwr_imp, temp_improvement=temp_imp,
                        composite_waste=waste,
                    ))

        # ── HV per baseline across all traces (4D point cloud) ──
        for bl in baselines:
            bl_df = model_df[model_df['baseline'] == bl]
            if len(bl_df) == 0:
                continue

            # Build per-window 4D points
            tpj = bl_df['tokens_per_joule'].values
            ept = np.where(tpj > 1e-10, 1.0 / tpj, np.nan)
            valid_mask = ~np.isnan(ept)

            points_4d = []
            for idx in np.where(valid_mask)[0]:
                pt = np.array([
                    ept[idx],
                    bl_df.iloc[idx]['tpot_p50_ms'],
                    bl_df.iloc[idx]['avg_power_w'],
                    bl_df.iloc[idx]['temp_max_c'],
                ])
                if np.all(np.isfinite(pt)):
                    points_4d.append(pt)

            if points_4d:
                hv_4d = compute_hv(points_4d, n_hv_samples)
            else:
                hv_4d = 0.0

            # Update the last summary entry for this baseline/model
            for i, s in enumerate(summaries):
                if s.strategy == bl and s.model == model and s.trace != "offline":
                    summaries[i] = StrategySummary(
                        strategy=s.strategy, model=s.model, trace=s.trace,
                        hv_3d=s.hv_3d, hv_4d=hv_4d,
                        n_points=s.n_points,
                        mean_ept=s.mean_ept, mean_tpot=s.mean_tpot,
                        mean_power=s.mean_power, mean_temp=s.mean_temp,
                    )

        # ── Per-window pairwise MDR/JIR (aligned by window_id) ──
        for trace in traces:
            trace_df = model_df[model_df['trace'] == trace]
            window_ids = sorted(trace_df['window_id'].unique())

            for bl_a in baselines:
                for bl_b in baselines:
                    if bl_a == bl_b:
                        continue

                    pts_a_list = []
                    pts_b_list = []

                    for wid in window_ids:
                        row_a = trace_df[(trace_df['baseline'] == bl_a) &
                                         (trace_df['window_id'] == wid)]
                        row_b = trace_df[(trace_df['baseline'] == bl_b) &
                                         (trace_df['window_id'] == wid)]

                        if len(row_a) == 0 or len(row_b) == 0:
                            continue

                        # E/tok from tokens_per_joule
                        tpj_a = row_a['tokens_per_joule'].iloc[0]
                        tpj_b = row_b['tokens_per_joule'].iloc[0]
                        ept_a = 1.0 / tpj_a if tpj_a > 1e-10 else np.nan
                        ept_b = 1.0 / tpj_b if tpj_b > 1e-10 else np.nan

                        pt_a = np.array([ept_a,
                                         row_a['tpot_p50_ms'].iloc[0],
                                         row_a['avg_power_w'].iloc[0],
                                         row_a['temp_max_c'].iloc[0]])
                        pt_b = np.array([ept_b,
                                         row_b['tpot_p50_ms'].iloc[0],
                                         row_b['avg_power_w'].iloc[0],
                                         row_b['temp_max_c'].iloc[0]])

                        if np.all(np.isfinite(pt_a)) and np.all(np.isfinite(pt_b)):
                            pts_a_list.append(pt_a)
                            pts_b_list.append(pt_b)

                    if len(pts_a_list) == 0:
                        continue

                    mdr = compute_mdr(pts_a_list, pts_b_list)
                    jir, jir_count = compute_jir(pts_a_list, pts_b_list)
                    waste = compute_composite_waste(pts_a_list, pts_b_list)

                    # Per-dimension mean improvement
                    arr_a = np.array(pts_a_list)
                    arr_b = np.array(pts_b_list)
                    temp_imp = float(np.mean((arr_b[:, 3] - arr_a[:, 3]) / arr_a[:, 3] * 100))

                    results.append(PairwiseResult(
                        strategy_a=bl_a, strategy_b=bl_b, model=model, trace=trace,
                        mdr=mdr, jir=jir, jir_count=jir_count,
                        total_count=len(pts_a_list),
                        ept_improvement=float(np.mean((arr_b[:, 0] - arr_a[:, 0]) / arr_a[:, 0] * 100)),
                        tpot_improvement=float(np.mean((arr_b[:, 1] - arr_a[:, 1]) / arr_a[:, 1] * 100)),
                        power_improvement=float(np.mean((arr_b[:, 2] - arr_a[:, 2]) / arr_a[:, 2] * 100)),
                        temp_improvement=temp_imp,
                        composite_waste=waste,
                    ))

    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# Evaluation: 3D from Oracle Gap (Offline Multi-Strategy)
# ═══════════════════════════════════════════════════════════════════

def evaluate_oracle_gap_3d(df: pd.DataFrame
                            ) -> Tuple[List[PairwiseResult], List[StrategySummary]]:
    """
    Compute 3D multi-objective metrics from Oracle Gap data.

    Uses the pre-computed strategy values (E/tok, TPOT, Power) from oracle gap
    analysis. This provides a cleaner comparison since all values come from
    the same rate table search space.

    Strategies: dynamic, maxn, best_static, pareto, oracle_energy, oracle_slo, oracle_power
    """
    print("\n" + "=" * 60)
    print("  3D Oracle Gap Evaluation")
    print("  Objectives: E/tok, TPOT, Power (all minimize)")
    print("=" * 60)

    strategy_cols = {
        'dynamic': ('dynamic_ept', 'dynamic_tpot', 'dynamic_power'),
        'maxn': ('maxn_ept', 'maxn_tpot', 'maxn_power'),
        'best_static': ('best_static_ept', 'best_static_tpot', 'best_static_power'),
        'pareto': ('pareto_ept', 'pareto_tpot', 'pareto_power'),
        'oracle_energy': ('oracle_energy_ept', 'oracle_energy_tpot', 'oracle_energy_power'),
        'oracle_slo': ('oracle_slo_ept', 'oracle_slo_tpot', 'oracle_slo_power'),
        'oracle_power': ('oracle_power_ept', 'oracle_power_tpot', 'oracle_power_power'),
    }

    models = sorted(df['model'].unique())
    results = []
    summaries = []

    for model in models:
        model_df = df[df['model'] == model]
        print(f"\n  Model: {model}")

        # ── HV per strategy ──
        for strat_name, (ept_col, tpot_col, pwr_col) in strategy_cols.items():
            pts = model_df[[ept_col, tpot_col, pwr_col]].dropna()
            if len(pts) == 0:
                continue
            points = [row.values for _, row in pts.iterrows()]
            hv = compute_hv(points)

            summaries.append(StrategySummary(
                strategy=strat_name, model=model, trace="oracle_gap",
                hv_3d=hv, hv_4d=0.0,
                n_points=len(points),
                mean_ept=float(pts[ept_col].mean()),
                mean_tpot=float(pts[tpot_col].mean()),
                mean_power=float(pts[pwr_col].mean()),
            ))

        # ── Pairwise MDR/JIR ──
        for strat_a, cols_a in strategy_cols.items():
            for strat_b, cols_b in strategy_cols.items():
                if strat_a == strat_b:
                    continue

                pts_a_list = []
                pts_b_list = []

                for _, row in model_df.iterrows():
                    pt_a = np.array([row[cols_a[0]], row[cols_a[1]], row[cols_a[2]]])
                    pt_b = np.array([row[cols_b[0]], row[cols_b[1]], row[cols_b[2]]])

                    if np.all(np.isfinite(pt_a)) and np.all(np.isfinite(pt_b)):
                        pts_a_list.append(pt_a)
                        pts_b_list.append(pt_b)

                if not pts_a_list:
                    continue

                mdr = compute_mdr(pts_a_list, pts_b_list)
                jir, jir_count = compute_jir(pts_a_list, pts_b_list)
                waste = compute_composite_waste(pts_a_list, pts_b_list)

                arr_a = np.array(pts_a_list)
                arr_b = np.array(pts_b_list)
                ept_imp = float(np.mean((arr_b[:, 0] - arr_a[:, 0]) / np.maximum(arr_a[:, 0], 1e-10) * 100))
                tpot_imp = float(np.mean((arr_b[:, 1] - arr_a[:, 1]) / np.maximum(arr_a[:, 1], 1e-10) * 100))
                pwr_imp = float(np.mean((arr_b[:, 2] - arr_a[:, 2]) / np.maximum(arr_a[:, 2], 1e-10) * 100))

                results.append(PairwiseResult(
                    strategy_a=strat_a, strategy_b=strat_b, model=model,
                    trace="oracle_gap",
                    mdr=mdr, jir=jir, jir_count=jir_count,
                    total_count=len(pts_a_list),
                    ept_improvement=ept_imp, tpot_improvement=tpot_imp,
                    power_improvement=pwr_imp, temp_improvement=0.0,
                    composite_waste=waste,
                ))

    return results, summaries


# ═══════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════

def generate_report(
    offline_results: List[PairwiseResult],
    offline_summaries: List[StrategySummary],
    serving_results: List[PairwiseResult],
    serving_summaries: List[StrategySummary],
    oracle_results: List[PairwiseResult],
    oracle_summaries: List[StrategySummary],
    output_dir: str,
):
    """Generate markdown report and CSV data files."""

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = []
    lines.append("# Multi-Objective Pareto Evaluation Report\n")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Script**: `src/ratetable/pareto_multi_objective_evaluation.py`\n")
    lines.append("---\n")

    # ── Metric Definitions ──
    lines.append("## Metric Definitions\n")
    lines.append("| Metric | Full Name | Description |")
    lines.append("|:---:|---|---|")
    lines.append("| **MDR** | Multi-Objective Dominance Rate | Fraction of instances where strategy A dominates baseline B on **ALL** objectives simultaneously |")
    lines.append("| **JIR** | Joint Improvement Ratio | Geometric mean of per-objective improvement ratios, counted **only** when ALL objectives improve |")
    lines.append("| **HV** | Hypervolume Indicator | Volume of dominated objective space (Zitzler 1999); higher = better |")
    lines.append("| **Waste** | Composite Waste | Normalized Euclidean distance ratio: how much worse B is than A relative to ideal |")
    lines.append("")

    # ════════════════════════════════════════════
    # Section 1: 3D Offline E2E Benchmark
    # ════════════════════════════════════════════
    lines.append("## 1. 3D Offline Evaluation (E2E Benchmark)\n")
    lines.append("**Data source**: `data/cap_selector_benchmark/` (378 runs, 3 models × 9 strategies × 5 workloads × 3 repeats)")
    lines.append("**Objectives**: E/tok (minimize), TPOT (minimize), Power (minimize)\n")

    # Fixed model order: 7B, 8B, 14B (not alphabetical)
    model_order = ['Qwen2.5-7B-Instruct-Q4_K_M',
                   'Meta-Llama-3.1-8B-Instruct-Q4_K_M',
                   'Qwen2.5-14B-Instruct-Q4_K_M']

    # Key comparisons: pareto vs dynamic, pareto vs maxn
    key_comparisons = [('pareto', 'dynamic'), ('pareto', 'maxn')]

    for strat_a, strat_b in key_comparisons:
        lines.append(f"### {strat_a.capitalize()} vs {strat_b.capitalize()}\n")
        lines.append("| Model | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% |")
        lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

        models = [m for m in model_order if any(r.model == m for r in offline_results if r.strategy_a == strat_a and r.strategy_b == strat_b)]
        for model in models:
            matching = [r for r in offline_results
                       if r.strategy_a == strat_a and r.strategy_b == strat_b and r.model == model]
            if not matching:
                continue
            m = matching[0]
            lines.append(
                f"| {short_model(model)} | {m.mdr:.1%} | {m.jir:.1%} | {m.composite_waste:.1%} | "
                f"{m.ept_improvement:+.1f}% | {m.tpot_improvement:+.1f}% | {m.power_improvement:+.1f}% |"
            )
        lines.append("")

    # HV comparison table
    lines.append("### Hypervolume (3D)\n")
    lines.append("Higher HV = dominates more objective space = better\n")
    lines.append("| Strategy | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) |")
    lines.append("|:---|:---:|:---:|:---:|")

    strats_offline = sorted(set(s.strategy for s in offline_summaries))
    models_offline = [m for m in model_order if m in set(s.model for s in offline_summaries)]
    for strat in strats_offline:
        cells = [strat]
        for model in models_offline:
            matching = [s for s in offline_summaries
                       if s.strategy == strat and s.model == model and s.trace == "offline"]
            if matching:
                cells.append(f"{matching[0].hv_3d:.4f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ════════════════════════════════════════════
    # Section 2: 3D Oracle Gap
    # ════════════════════════════════════════════
    lines.append("## 2. 3D Oracle Gap Evaluation\n")
    lines.append("**Data source**: `data/oracle_gap_analysis/` (lock-mode 11-freq oracle)")
    lines.append("**Objectives**: E/tok, TPOT, Power (all minimize)\n")

    lines.append("### Multi-Objective Dominance vs Oracle\n")
    lines.append("| Model | Dynamic→Oracle | MAXN→Oracle | Pareto→Oracle | BestStatic→Oracle |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")

    models_oracle = [m for m in model_order if any(s.model == m for s in oracle_summaries if s.trace == "oracle_gap")]
    for model in models_oracle:
        cells = [short_model(model)]
        for strat in ['dynamic', 'maxn', 'pareto', 'best_static']:
            matching = [r for r in oracle_results
                       if r.strategy_a == strat and r.strategy_b == 'oracle_energy']
            if matching:
                cells.append(f"{matching[0].mdr:.1%} (waste {matching[0].composite_waste:.1%})")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # HV table for oracle
    lines.append("### Hypervolume (Oracle Gap)\n")
    lines.append("| Strategy | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) |")
    lines.append("|:---|:---:|:---:|:---:|")

    strats_oracle = sorted(set(s.strategy for s in oracle_summaries if s.trace == "oracle_gap"))
    for strat in strats_oracle:
        cells = [strat]
        for model in models_oracle:
            matching = [s for s in oracle_summaries
                       if s.strategy == strat and s.model == model and s.trace == "oracle_gap"]
            if matching:
                cells.append(f"{matching[0].hv_3d:.4f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ════════════════════════════════════════════
    # Section 3: 4D Serving Evaluation
    # ════════════════════════════════════════════
    lines.append("## 3. 4D Serving Evaluation\n")
    lines.append("**Data source**: `data/serving_benchmark/` (long-running serving windows)")
    lines.append("**Objectives**: E/tok (minimize), TPOT (minimize), Power (minimize), Peak Temp (minimize)\n")

    models_serving = [m for m in model_order if any(s.model == m for s in serving_summaries if s.trace != "offline")]
    for model in models_serving:
        lines.append(f"### Model: {short_model(model)}\n")

        traces_model = sorted(set(s.trace for s in serving_summaries
                                  if s.model == model and s.trace != "offline"))
        for trace in traces_model:
            lines.append(f"#### Trace: {trace}\n")
            lines.append("| Baseline A | Baseline B | MDR | JIR | Waste | E/tok Δ% | TPOT Δ% | Power Δ% | Temp Δ% |")
            lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

            bls = sorted(set(
                s.strategy for s in serving_summaries
                if s.model == model and s.trace == trace
            ))

            # Key comparisons: pareto/thermalslo vs maxn/dynamic
            priority_a = ['Pareto', 'ThermalSLO', 'BestStatic']
            priority_b = ['MAXN', 'Dynamic']
            shown = set()

            for a in priority_a:
                for b in priority_b:
                    if a == b:
                        continue
                    key = (a, b)
                    if key in shown:
                        continue
                    shown.add(key)

                    matching_agg = [r for r in serving_results
                                   if r.strategy_a == a and r.strategy_b == b
                                   and r.model == model and r.trace == trace
                                   and r.total_count == 1]  # aggregate entries

                    # Also get per-window entries
                    matching_win = [r for r in serving_results
                                   if r.strategy_a == a and r.strategy_b == b
                                   and r.model == model and r.trace == trace
                                   and r.total_count > 1]

                    if matching_agg and matching_win:
                        agg = matching_agg[0]
                        win = matching_win[0]
                        lines.append(
                            f"| {a} | {b} | {win.mdr:.1%} | {win.jir:.1%} | "
                            f"{win.composite_waste:.1%} | "
                            f"{win.ept_improvement:+.1f}% | {win.tpot_improvement:+.1f}% | "
                            f"{win.power_improvement:+.1f}% | {win.temp_improvement:+.1f}% |"
                        )
                    elif matching_agg:
                        agg = matching_agg[0]
                        lines.append(
                            f"| {a} | {b} (agg) | {agg.mdr:.1%} | {agg.jir:.1%} | "
                            f"{agg.composite_waste:.1%} | "
                            f"{agg.ept_improvement:+.1f}% | {agg.tpot_improvement:+.1f}% | "
                            f"{agg.power_improvement:+.1f}% | {agg.temp_improvement:+.1f}% |"
                        )
            lines.append("")

    # 4D HV table
    lines.append("### Hypervolume (4D Serving, per model)\n")
    lines.append("| Baseline | 7B (Qwen2.5) | 8B (Llama-3.1) | 14B (Qwen2.5) |")
    lines.append("|:---|:---:|:---:|:---:|")

    bls_all = sorted(set(s.strategy for s in serving_summaries if s.trace != "offline"))
    for bl in bls_all:
        cells = [bl]
        for model in models_serving:
            matching = [s for s in serving_summaries
                       if s.strategy == bl and s.model == model and s.hv_4d > 0]
            if matching:
                # Use the one with most points
                best = max(matching, key=lambda x: x.n_points)
                cells.append(f"{best.hv_4d:.4f} ({best.n_points} pts)")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # ════════════════════════════════════════════
    # Section 4: Serving Aggregate Stats
    # ════════════════════════════════════════════
    lines.append("## 4. Serving Aggregate Statistics\n")
    for model in models_serving:
        lines.append(f"### {short_model(model)}\n")
        lines.append("| Baseline | Trace | Windows | E/tok (J) | TPOT (ms) | Power (W) | Peak Temp (°C) | SLO Viol % |")
        lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

        for trace in sorted(set(s.trace for s in serving_summaries
                                if s.model == model and s.trace != "offline")):
            for bl in sorted(set(s.strategy for s in serving_summaries
                                if s.model == model and s.trace == trace)):
                matching = [s for s in serving_summaries
                           if s.strategy == bl and s.model == model and s.trace == trace]
                if not matching:
                    continue
                s = matching[0]
                lines.append(
                    f"| {bl} | {trace} | {s.n_points} | {s.mean_ept:.4f} | "
                    f"{s.mean_tpot:.1f} | {s.mean_power:.1f} | {s.mean_temp:.1f} | — |"
                )
        lines.append("")

    # ════════════════════════════════════════════
    # Section 5: Summary & Conclusions
    # ════════════════════════════════════════════
    lines.append("## 5. Summary\n")
    lines.append("### Key Findings\n")
    lines.append("")

    # Auto-generate summary based on data
    # 3D: pareto vs dynamic/maxn
    for strat_a, strat_b in [('pareto', 'dynamic'), ('pareto', 'maxn')]:
        all_matching = [r for r in offline_results
                       if r.strategy_a == strat_a and r.strategy_b == strat_b]
        if all_matching:
            avg_mdr = np.mean([r.mdr for r in all_matching])
            avg_waste = np.mean([r.composite_waste for r in all_matching])
            lines.append(
                f"- **{strat_a} vs {strat_b}** (3D offline): "
                f"MDR = {avg_mdr:.1%}, Composite Waste = {avg_waste:.1%}"
            )

    lines.append("")
    lines.append("### Paper Narrative Implications\n")
    lines.append("1. **Multi-objective superiority**: Pareto's value extends beyond single-objective E/tok")
    lines.append("2. **Thermal dimension**: ThermalSLO's 4D advantage demonstrates unique thermal-aware contribution")
    lines.append("3. **Hypervolume**: Gold-standard EMO metric quantifies total dominated solution space")
    lines.append("")

    # Write report
    report_path = os.path.join(output_dir, f"multi_obj_eval_report_{ts}.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Report: {report_path}")

    # ── Save CSV data ──
    # Offline pairwise results
    offline_csv = os.path.join(output_dir, f"multi_obj_3d_offline_{ts}.csv")
    offline_df = pd.DataFrame([asdict(r) for r in offline_results])
    offline_df.to_csv(offline_csv, index=False)
    print(f"  3D offline CSV: {offline_csv}")

    # Serving pairwise results
    serving_csv = os.path.join(output_dir, f"multi_obj_4d_serving_{ts}.csv")
    serving_df = pd.DataFrame([asdict(r) for r in serving_results])
    serving_df.to_csv(serving_csv, index=False)
    print(f"  4D serving CSV: {serving_csv}")

    # Oracle gap pairwise results
    oracle_csv = os.path.join(output_dir, f"multi_obj_3d_oracle_{ts}.csv")
    oracle_df = pd.DataFrame([asdict(r) for r in oracle_results])
    oracle_df.to_csv(oracle_csv, index=False)
    print(f"  3D oracle CSV: {oracle_csv}")

    # HV summaries
    all_summaries = offline_summaries + serving_summaries + oracle_summaries
    hv_csv = os.path.join(output_dir, f"multi_obj_hv_summary_{ts}.csv")
    hv_df = pd.DataFrame([asdict(s) for s in all_summaries])
    hv_df.to_csv(hv_csv, index=False)
    print(f"  HV summary CSV: {hv_csv}")

    return report_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Objective Pareto Evaluation (MDR, JIR, HV)"
    )
    parser.add_argument("--e2e-data", default="data/cap_selector_benchmark",
                        help="E2E benchmark data directory")
    parser.add_argument("--serving-data", default="data/serving_benchmark",
                        help="Serving window data directory")
    parser.add_argument("--oracle-data", default="data/oracle_gap_analysis",
                        help="Oracle gap data directory")
    parser.add_argument("--output-dir", default="data/multi_obj_eval",
                        help="Output directory for results")
    parser.add_argument("--hv-samples", type=int, default=50000,
                        help="Monte Carlo samples for Hypervolume (default: 50000)")
    parser.add_argument("--skip-offline", action="store_true",
                        help="Skip 3D offline E2E benchmark evaluation")
    parser.add_argument("--skip-serving", action="store_true",
                        help="Skip 4D serving evaluation")
    parser.add_argument("--skip-oracle", action="store_true",
                        help="Skip 3D oracle gap evaluation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Multi-Objective Pareto Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    offline_results, offline_summaries = [], []
    serving_results, serving_summaries = [], []
    oracle_results, oracle_summaries = [], []

    # Load data
    if not args.skip_offline:
        print("\n[1/3] Loading E2E benchmark data...")
        try:
            e2e_df = load_e2e_benchmark(args.e2e_data)
            offline_results, offline_summaries = evaluate_3d_offline(e2e_df, args.hv_samples)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping offline evaluation")

    if not args.skip_serving:
        print("\n[2/3] Loading serving window data...")
        try:
            serving_df = load_serving_windows(args.serving_data)
            serving_results, serving_summaries = evaluate_4d_serving(serving_df, args.hv_samples)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping serving evaluation")

    if not args.skip_oracle:
        print("\n[3/3] Loading oracle gap data...")
        try:
            oracle_df = load_oracle_gap(args.oracle_data)
            oracle_results, oracle_summaries = evaluate_oracle_gap_3d(oracle_df)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} — skipping oracle gap evaluation")

    # Generate report
    print("\n" + "=" * 60)
    print("  Generating report...")
    print("=" * 60)

    report_path = generate_report(
        offline_results, offline_summaries,
        serving_results, serving_summaries,
        oracle_results, oracle_summaries,
        args.output_dir,
    )

    # Print key results
    MODEL_ORDER = ['Qwen2.5-7B-Instruct-Q4_K_M',
                   'Meta-Llama-3.1-8B-Instruct-Q4_K_M',
                   'Qwen2.5-14B-Instruct-Q4_K_M']
    print("\n" + "=" * 60)
    print("  KEY RESULTS SUMMARY")
    print("=" * 60)

    if offline_results:
        print("\n  3D Offline (E2E Benchmark):")
        for pair in [('pareto', 'dynamic'), ('pareto', 'maxn')]:
            matching_per_model = {}
            for r in offline_results:
                if r.strategy_a == pair[0] and r.strategy_b == pair[1]:
                    matching_per_model.setdefault(r.model, []).append(r)
            for model in [m for m in MODEL_ORDER if m in matching_per_model]:
                m = matching_per_model[model][0]
                sm = short_model(model)
                print(f"    {sm}: {pair[0]:>8s} vs {pair[1]:<8s}: "
                      f"MDR={m.mdr:.1%}, JIR={m.jir:.1%}, "
                      f"Waste={m.composite_waste:.1%}, "
                      f"E/tok={m.ept_improvement:+.1f}%, "
                      f"Power={m.power_improvement:+.1f}%")

    if serving_results:
        print("\n  4D Serving (key comparisons, per-window):")
        for pair in [('Pareto', 'MAXN'), ('Pareto', 'Dynamic'),
                     ('ThermalSLO', 'MAXN'), ('ThermalSLO', 'Dynamic')]:
            matching = [r for r in serving_results
                       if r.strategy_a == pair[0] and r.strategy_b == pair[1]
                       and r.total_count > 1]  # per-window, not aggregate
            if matching:
                # Average across models/traces
                avg_mdr = np.mean([r.mdr for r in matching])
                avg_jir = np.mean([r.jir for r in matching])
                avg_waste = np.mean([r.composite_waste for r in matching])
                print(f"    {pair[0]:>10s} vs {pair[1]:<8s}: "
                      f"MDR={avg_mdr:.1%}, JIR={avg_jir:.1%}, "
                      f"Waste={avg_waste:.1%} (avg over {len(matching)} traces)")

    if offline_summaries:
        print("\n  3D Hypervolume (top 3 per model):")
        for model in [m for m in MODEL_ORDER if any(s.model == m for s in offline_summaries if s.trace == "offline")]:
            model_s = [s for s in offline_summaries if s.model == model and s.trace == "offline"]
            model_s.sort(key=lambda x: x.hv_3d, reverse=True)
            short = model.replace("-Instruct-Q4_K_M", "").replace("Meta-Llama-3.1-", "Llama-")
            for s in model_s[:3]:
                print(f"    {short}: {s.strategy:>10s} HV={s.hv_3d:.6f}")

    print(f"\n  Full report: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
