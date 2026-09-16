#!/usr/bin/env python3
"""
Thermal-SLO Feedback Controller (Phase 13, P2)

Runtime feedback controller that adjusts GPU frequency cap based on
SLO compliance and thermal state. Operates on fixed-size time windows
(10s default), making one cap adjustment decision per window.

Architecture:
  Offline Phase-aware Characterization (lock rate table)
      ↓
  Workload-aware Initial Cap Selection (WorkloadCapSelector)
      ↓
  Runtime SLO/Thermal Feedback Controller (THIS MODULE) ← Phase 13 core

Control Rules (priority from high to low):
  1. SLO Violation   → violation_rate > 10% or p95_TPOT > 0.95×SLO → raise cap
  2. Thermal Protect → temp > 85°C → raise to max cap (finish fast, then idle cool)
  3. Thermal Opport  → temp > 75°C + slo_slack ok + K consecutive windows → lower cap
  4. Power Exceed    → avg_power > budget + slo_slack ok + K consecutive → lower cap
  5. Cooldown        → time since last cap change < 20s → skip
  6. Default         → keep current cap

Key Design Decisions:
  - Thermal protection RAISES (not lowers) frequency: high temp needs fast
    completion so GPU can enter idle and cool down
  - Hysteresis (K consecutive windows) prevents oscillation
  - EMC/CPU fixed: only GPU cap adjusted to isolate control variable
  - Window interval 10s: balances responsiveness with stability

Usage:
    from src.controller.thermal_slo_controller import ThermalSLOCapController

    controller = ThermalSLOCapController(
        cap_rate_table_path='data/rate_tables/cap_rate_table_*.parquet',
        lock_rate_table_path='data/rate_tables/lock_rate_table_*.parquet',
        model='Qwen2.5-7B-Instruct-Q4_K_M',
        cap_controller=cc,  # CapController instance
    )

    # Initial cap selection for first request
    init_cap = controller.select_initial_cap(prompt_length=512, output_length=128)

    # Per-window update (call every ~10s)
    window_state = controller.update_cap(
        current_temperature_c=72.3,
        temperature_slope_c_per_s=0.2,
        tpot_p50_ms=45.0, tpot_p95_ms=52.0, tpot_p99_ms=58.0,
        avg_power_w=42.0, max_power_w=48.0,
        tokens_per_second=22.1,
        n_requests=3,
        prompt_length=512, output_length=128,
    )
    print(f'Action: {window_state.cap_action}, Cap: {window_state.current_cap_mhz}MHz')
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

import time
import logging
import numpy as np
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.controller.cap_controller import CapController
from src.controller.workload_cap_selector import WorkloadCapSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CapAction(Enum):
    """Controller action types."""
    RAISE = 'raise'
    LOWER = 'lower'
    KEEP = 'keep'
    RAISE_MAX = 'raise_max'  # Emergency thermal raise


@dataclass
class ControllerConfig:
    """Configuration for the Thermal-SLO feedback controller."""
    # Available GPU cap levels (ascending order)
    gpu_caps: List[int] = field(default_factory=lambda: [
        408, 510, 612, 714, 816, 918, 1020, 1122, 1224, 1300
    ])

    # SLO constraints
    tpot_slo_ms: float = 50.0

    # Power budget
    power_budget_w: float = 50.0

    # Thermal thresholds
    thermal_warning_c: float = 75.0    # Start considering cap reduction
    thermal_critical_c: float = 85.0   # Emergency: raise to max

    # Temperature rate-of-change threshold
    temp_slope_threshold: float = 0.5  # °C/s — fast warming triggers caution

    # Window parameters
    window_interval_s: float = 10.0

    # Hysteresis: require K consecutive windows before lowering cap
    lower_hysteresis_k: int = 3

    # Cooldown: minimum seconds between cap changes
    cooldown_s: float = 20.0

    # SLO violation thresholds
    slo_violation_rate_threshold: float = 0.10   # >10% violation → raise
    slo_tpot_p95_ratio: float = 0.95              # p95 > 0.95×SLO → raise

    # Pareto strategy for initial cap selection
    initial_cap_strategy: str = 'pareto'


@dataclass
class WindowState:
    """State snapshot for one control window."""
    window_id: int = 0
    timestamp: float = 0.0

    # Current cap state
    current_cap_mhz: int = 1300
    cap_action: CapAction = CapAction.KEEP
    prev_cap_mhz: int = 1300

    # Thermal state
    temperature_c: float = 0.0
    temperature_slope_c_per_s: float = 0.0
    temperature_max_c: float = 0.0

    # Performance state
    tpot_p50_ms: float = 0.0
    tpot_p95_ms: float = 0.0
    tpot_p99_ms: float = 0.0
    tokens_per_second: float = 0.0

    # SLO compliance
    slo_violation_count: int = 0
    slo_violation_rate: float = 0.0
    total_requests: int = 0

    # Power state
    avg_power_w: float = 0.0
    max_power_w: float = 0.0

    # Internal controller state
    consecutive_lower_windows: int = 0
    last_cap_change_time: float = 0.0
    cap_switch_count: int = 0

    # Workload context
    prompt_length: int = 0
    output_length: int = 0


class ThermalSLOCapController:
    """Thermal-SLO feedback controller for dynamic GPU frequency capping.

    Integrates with WorkloadCapSelector for initial cap selection and
    CapController for actual hardware frequency changes.
    """

    def __init__(
        self,
        cap_rate_table_path: str,
        lock_rate_table_path: str,
        model: str,
        cap_controller: Optional[CapController] = None,
        config: Optional[ControllerConfig] = None,
    ):
        """Initialize the Thermal-SLO controller.

        Args:
            cap_rate_table_path: Path to cap rate table (glob pattern supported)
            lock_rate_table_path: Path to lock rate table (glob pattern supported)
            model: Model name for rate table lookup
            cap_controller: CapController instance for hardware control
            config: Controller configuration (uses defaults if None)
        """
        self.model = model
        self.config = config or ControllerConfig()
        self.cap_controller = cap_controller

        # Resolve glob patterns
        if '*' in str(cap_rate_table_path):
            cap_files = sorted(Path('data/rate_tables').glob(
                Path(cap_rate_table_path).name))
        elif Path(cap_rate_table_path).exists():
            cap_files = [Path(cap_rate_table_path)]
        else:
            cap_files = []

        if '*' in str(lock_rate_table_path):
            lock_files = sorted(Path('data/rate_tables').glob(
                Path(lock_rate_table_path).name))
        elif Path(lock_rate_table_path).exists():
            lock_files = [Path(lock_rate_table_path)]
        else:
            lock_files = []

        # Initialize workload-aware selector for initial cap choices
        try:
            self.cap_selector = WorkloadCapSelector(
                cap_rate_table_path=str(cap_files[-1]) if cap_files else '',
                lock_rate_table_path=str(lock_files[-1]) if lock_files else '',
                model=model,
            )
            logger.info(f"WorkloadCapSelector loaded for {model}")
        except Exception as e:
            logger.warning(f"WorkloadCapSelector init failed: {e}. "
                           "Using default cap selection.")
            self.cap_selector = None

        # Internal state
        self._current_cap_mhz: int = self.config.gpu_caps[-1]  # Start at max
        self._window_id: int = 0
        self._consecutive_lower_windows: int = 0
        self._last_cap_change_time: float = 0.0
        self._cap_switch_count: int = 0
        self._window_history: List[WindowState] = []

        logger.info(f"ThermalSLOCapController initialized: "
                    f"{len(self.config.gpu_caps)} caps, "
                    f"SLO={self.config.tpot_slo_ms}ms, "
                    f"thermal_warn={self.config.thermal_warning_c}°C, "
                    f"thermal_crit={self.config.thermal_critical_c}°C")

    def select_initial_cap(
        self,
        prompt_length: int,
        output_length: int,
    ) -> Dict:
        """Select initial GPU cap based on workload characteristics.

        Uses WorkloadCapSelector with Pareto strategy as starting point.

        Args:
            prompt_length: Input prompt length in tokens
            output_length: Expected output length in tokens

        Returns:
            Dict with gpu_cap_mhz and predicted metrics
        """
        if self.cap_selector:
            try:
                result = self.cap_selector.select(
                    prompt_length=prompt_length,
                    output_length=output_length,
                    strategy=self.config.initial_cap_strategy,
                )
                cap = result.get('gpu_cap_mhz', self.config.gpu_caps[-1])
                # Snap to nearest available cap
                cap = self._snap_to_cap(cap)
            except Exception as e:
                logger.warning(f"Initial cap selection failed: {e}")
                cap = self.config.gpu_caps[-1]
        else:
            cap = self.config.gpu_caps[-1]

        self._current_cap_mhz = cap
        self._last_cap_change_time = time.time()

        logger.info(f"Initial cap: {cap}MHz for p={prompt_length} o={output_length}")

        return {
            'gpu_cap_mhz': cap,
            'action': 'initial',
            'window_id': 0,
        }

    def update_cap(
        self,
        # Thermal inputs
        current_temperature_c: float,
        temperature_slope_c_per_s: float = 0.0,
        temperature_max_c: float = 0.0,
        # Performance inputs
        tpot_p50_ms: float = 0.0,
        tpot_p95_ms: float = 0.0,
        tpot_p99_ms: float = 0.0,
        tokens_per_second: float = 0.0,
        # SLO inputs
        slo_violation_count: int = 0,
        total_requests: int = 0,
        # Power inputs
        avg_power_w: float = 0.0,
        max_power_w: float = 0.0,
        # Workload context
        prompt_length: int = 0,
        output_length: int = 0,
        # Override for testing
        n_requests: int = 0,
    ) -> WindowState:
        """Update GPU cap based on current window state.

        Called once per control window (~10s). Evaluates control rules
        in priority order and returns the resulting WindowState.

        Args:
            current_temperature_c: Current GPU temperature
            temperature_slope_c_per_s: Temperature change rate
            temperature_max_c: Peak temperature in this window
            tpot_p50_ms: Median TPOT in this window
            tpot_p95_ms: P95 TPOT in this window
            tpot_p99_ms: P99 TPOT in this window
            tokens_per_second: Average TPS in this window
            slo_violation_count: Number of SLO-violating requests
            total_requests: Total requests in this window
            avg_power_w: Average power consumption
            max_power_w: Peak power consumption
            prompt_length: Prompt length of current workload
            output_length: Output length of current workload
            n_requests: Number of requests in this window

        Returns:
            WindowState with the controller's decision
        """
        self._window_id += 1
        now = time.time()

        # Build window state (pre-decision)
        slo_violation_rate = (slo_violation_count / total_requests
                              if total_requests > 0 else 0.0)

        state = WindowState(
            window_id=self._window_id,
            timestamp=now,
            current_cap_mhz=self._current_cap_mhz,
            prev_cap_mhz=self._current_cap_mhz,
            temperature_c=current_temperature_c,
            temperature_slope_c_per_s=temperature_slope_c_per_s,
            temperature_max_c=max(temperature_max_c, current_temperature_c),
            tpot_p50_ms=tpot_p50_ms,
            tpot_p95_ms=tpot_p95_ms,
            tpot_p99_ms=tpot_p99_ms,
            tokens_per_second=tokens_per_second,
            slo_violation_count=slo_violation_count,
            slo_violation_rate=slo_violation_rate,
            total_requests=max(total_requests, n_requests),
            avg_power_w=avg_power_w,
            max_power_w=max_power_w,
            consecutive_lower_windows=self._consecutive_lower_windows,
            last_cap_change_time=self._last_cap_change_time,
            cap_switch_count=self._cap_switch_count,
            prompt_length=prompt_length,
            output_length=output_length,
        )

        # Compute SLO slack (how much TPOT headroom exists)
        slo_slack = (self.config.tpot_slo_ms - tpot_p50_ms) / self.config.tpot_slo_ms
        # slo_slack > 0 means we're under SLO, < 0 means violating

        # ── Rule 1: SLO Violation → Raise cap ──
        if (slo_violation_rate > self.config.slo_violation_rate_threshold or
                tpot_p95_ms > self.config.slo_tpot_p95_ratio * self.config.tpot_slo_ms):
            new_cap = self._step_cap(self._current_cap_mhz, direction='raise')
            if new_cap != self._current_cap_mhz:
                state.cap_action = CapAction.RAISE
                self._apply_cap_change(new_cap, now, state)
                logger.info(f"[W{self._window_id}] SLO VIOLATION: "
                            f"violation_rate={slo_violation_rate:.1%}, "
                            f"p95_TPOT={tpot_p95_ms:.1f}ms → RAISE to {new_cap}MHz")
                return state

        # ── Rule 2: Thermal Protection → Raise to max ──
        if current_temperature_c >= self.config.thermal_critical_c:
            max_cap = self.config.gpu_caps[-1]
            if self._current_cap_mhz != max_cap:
                state.cap_action = CapAction.RAISE_MAX
                self._apply_cap_change(max_cap, now, state)
                logger.warning(f"[W{self._window_id}] THERMAL CRITICAL: "
                               f"{current_temperature_c:.1f}°C → RAISE_MAX to {max_cap}MHz")
                return state

        # ── Rule 5: Cooldown check (before lower decisions) ──
        time_since_change = now - self._last_cap_change_time
        if time_since_change < self.config.cooldown_s:
            state.cap_action = CapAction.KEEP
            logger.debug(f"[W{self._window_id}] COOLDOWN: "
                         f"{time_since_change:.1f}s < {self.config.cooldown_s}s → KEEP")
            self._window_history.append(state)
            return state

        # ── Rule 3: Thermal Opportunity → Lower cap (with hysteresis) ──
        if (current_temperature_c >= self.config.thermal_warning_c and
                slo_slack >= 0.1 and  # At least 10% SLO headroom
                self._current_cap_mhz > self.config.gpu_caps[0]):  # Not at minimum
            self._consecutive_lower_windows += 1
            consecutive_before = self._consecutive_lower_windows
            if self._consecutive_lower_windows >= self.config.lower_hysteresis_k:
                new_cap = self._step_cap(self._current_cap_mhz, direction='lower')
                if new_cap != self._current_cap_mhz:
                    state.cap_action = CapAction.LOWER
                    self._apply_cap_change(new_cap, now, state)
                    logger.info(f"[W{self._window_id}] THERMAL OPPORTUNITY: "
                                f"temp={current_temperature_c:.1f}°C, "
                                f"slo_slack={slo_slack:.1%}, "
                                f"consecutive={consecutive_before} → "
                                f"LOWER to {new_cap}MHz")
                    return state
            state.consecutive_lower_windows = self._consecutive_lower_windows
            state.cap_action = CapAction.KEEP
            logger.debug(f"[W{self._window_id}] THERMAL OPPORTUNE but hysteresis: "
                         f"{self._consecutive_lower_windows}/{self.config.lower_hysteresis_k} → KEEP")
            self._window_history.append(state)
            return state
        else:
            # Reset counter if condition not met
            self._consecutive_lower_windows = 0

        # ── Rule 4: Power Exceed → Lower cap (with hysteresis) ──
        if (avg_power_w > self.config.power_budget_w and
                slo_slack >= 0.1 and
                self._current_cap_mhz > self.config.gpu_caps[0]):
            self._consecutive_lower_windows += 1
            if self._consecutive_lower_windows >= self.config.lower_hysteresis_k:
                new_cap = self._step_cap(self._current_cap_mhz, direction='lower')
                if new_cap != self._current_cap_mhz:
                    state.cap_action = CapAction.LOWER
                    self._apply_cap_change(new_cap, now, state)
                    logger.info(f"[W{self._window_id}] POWER EXCEED: "
                                f"avg_pwr={avg_power_w:.1f}W > {self.config.power_budget_w}W, "
                                f"slo_slack={slo_slack:.1%} → LOWER to {new_cap}MHz")
                    return state
            state.consecutive_lower_windows = self._consecutive_lower_windows
            state.cap_action = CapAction.KEEP
            self._window_history.append(state)
            return state
        else:
            if avg_power_w <= self.config.power_budget_w:
                self._consecutive_lower_windows = 0

        # ── Rule 6: Default → Keep ──
        state.cap_action = CapAction.KEEP
        self._window_history.append(state)
        return state

    def _step_cap(self, current: int, direction: str) -> int:
        """Step GPU cap one level up or down.

        Args:
            current: Current cap in MHz
            direction: 'raise' or 'lower'

        Returns:
            New cap in MHz (clamped to available caps)
        """
        idx = self.config.gpu_caps.index(current) if current in self.config.gpu_caps else -1
        if idx < 0:
            # Not in list, snap to nearest
            idx = int(np.argmin(np.abs(np.array(self.config.gpu_caps) - current)))

        if direction == 'raise':
            new_idx = min(idx + 1, len(self.config.gpu_caps) - 1)
        elif direction == 'lower':
            new_idx = max(idx - 1, 0)
        else:
            return current

        return self.config.gpu_caps[new_idx]

    def _snap_to_cap(self, target: int) -> int:
        """Snap arbitrary MHz to nearest available cap level."""
        if target in self.config.gpu_caps:
            return target
        idx = int(np.argmin(np.abs(np.array(self.config.gpu_caps) - target)))
        return self.config.gpu_caps[idx]

    def _apply_cap_change(self, new_cap: int, timestamp: float, state: WindowState):
        """Apply a cap change to hardware and update internal state."""
        state.prev_cap_mhz = self._current_cap_mhz
        state.current_cap_mhz = new_cap
        self._current_cap_mhz = new_cap
        self._last_cap_change_time = timestamp
        self._consecutive_lower_windows = 0  # Reset hysteresis on any change
        self._cap_switch_count += 1
        state.cap_switch_count = self._cap_switch_count

        # Apply to hardware if controller available
        if self.cap_controller:
            try:
                self.cap_controller.set_cap(
                    gpu_cap_mhz=new_cap,
                    emc_cap_mhz=3199,
                    cpu_cap_mhz=1036,
                )
            except Exception as e:
                logger.error(f"Hardware cap change failed: {e}")

    def get_state(self) -> Dict:
        """Get current controller state summary."""
        return {
            'current_cap_mhz': self._current_cap_mhz,
            'window_id': self._window_id,
            'cap_switch_count': self._cap_switch_count,
            'consecutive_lower_windows': self._consecutive_lower_windows,
            'last_cap_change_time': self._last_cap_change_time,
        }

    def get_history(self) -> List[WindowState]:
        """Get full window state history."""
        return self._window_history

    def reset(self):
        """Reset controller state (e.g., for new experiment)."""
        self._current_cap_mhz = self.config.gpu_caps[-1]
        self._window_id = 0
        self._consecutive_lower_windows = 0
        self._last_cap_change_time = 0.0
        self._cap_switch_count = 0
        self._window_history = []


def simulate_synthetic_trace(
    controller: ThermalSLOCapController,
    trace: List[Dict],
) -> List[WindowState]:
    """Run controller against a synthetic trace for validation.

    Args:
        controller: ThermalSLOCapController instance
        trace: List of dicts with per-window metrics:
            - temperature_c, temperature_slope, tpot_p50, tpot_p95, ...
            - n_requests, slo_violations, avg_power_w, ...

    Returns:
        List of WindowState records
    """
    states = []

    for i, metrics in enumerate(trace):
        state = controller.update_cap(
            current_temperature_c=metrics.get('temperature_c', 65.0),
            temperature_slope_c_per_s=metrics.get('temperature_slope', 0.0),
            temperature_max_c=metrics.get('temperature_max', metrics.get('temperature_c', 65.0)),
            tpot_p50_ms=metrics.get('tpot_p50', 40.0),
            tpot_p95_ms=metrics.get('tpot_p95', 45.0),
            tpot_p99_ms=metrics.get('tpot_p99', 50.0),
            tokens_per_second=metrics.get('tps', 25.0),
            slo_violation_count=metrics.get('slo_violations', 0),
            total_requests=metrics.get('n_requests', 3),
            avg_power_w=metrics.get('avg_power_w', 40.0),
            max_power_w=metrics.get('max_power_w', 48.0),
            prompt_length=metrics.get('prompt_length', 512),
            output_length=metrics.get('output_length', 128),
            n_requests=metrics.get('n_requests', 3),
        )
        states.append(state)

    return states


def generate_warming_trace(
    n_windows: int = 36,
    base_temp_c: float = 55.0,
    peak_temp_c: float = 82.0,
    tpot_base_ms: float = 38.0,
    power_base_w: float = 38.0,
) -> List[Dict]:
    """Generate a synthetic warming trace for controller validation.

    Simulates a long-running inference session where temperature
    gradually rises from base to peak over n_windows.

    Args:
        n_windows: Number of windows (each ~10s)
        base_temp_c: Starting temperature
        peak_temp_c: Peak temperature
        tpot_base_ms: Base TPOT (increases with temperature)
        power_base_w: Base power (increases with temperature)

    Returns:
        List of dicts for simulate_synthetic_trace()
    """
    trace = []
    for i in range(n_windows):
        progress = i / max(n_windows - 1, 1)
        temp = base_temp_c + (peak_temp_c - base_temp_c) * progress
        temp_slope = (peak_temp_c - base_temp_c) / (n_windows * 10.0) * (1 + 0.5 * np.random.random())

        # Temperature causes slight performance degradation
        tpot = tpot_base_ms + (temp - base_temp_c) * 0.3 + np.random.normal(0, 1)
        power = power_base_w + (temp - base_temp_c) * 0.15 + np.random.normal(0, 0.5)

        # SLO violations increase with temperature
        slo_violations = 0 if tpot < 47.5 else (1 if tpot < 50 else np.random.randint(1, 4))
        n_requests = np.random.randint(2, 5)

        trace.append({
            'temperature_c': round(temp, 1),
            'temperature_slope': round(temp_slope, 3),
            'temperature_max': round(temp + np.random.uniform(0, 1), 1),
            'tpot_p50': round(tpot, 1),
            'tpot_p95': round(tpot * 1.1 + np.random.uniform(0, 3), 1),
            'tpot_p99': round(tpot * 1.2 + np.random.uniform(0, 5), 1),
            'tps': round(1000.0 / max(tpot, 1), 1),
            'slo_violations': int(slo_violations),
            'n_requests': n_requests,
            'avg_power_w': round(power, 1),
            'max_power_w': round(power + np.random.uniform(2, 8), 1),
            'prompt_length': 512,
            'output_length': 128,
        })

    return trace


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Thermal-SLO Controller Validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--validate', action='store_true',
                        help='Run synthetic trace validation')
    parser.add_argument('--trace-type', type=str, default='warming',
                        choices=['warming', 'bursty', 'stable'],
                        help='Synthetic trace type')
    parser.add_argument('--n-windows', type=int, default=36,
                        help='Number of windows for synthetic trace')
    parser.add_argument('--tpot-slo-ms', type=float, default=50.0,
                        help='TPOT SLO in ms')
    parser.add_argument('--power-budget-w', type=float, default=50.0,
                        help='Power budget in W')
    args = parser.parse_args()

    if args.validate:
        print("=" * 70)
        print("  Thermal-SLO Controller: Synthetic Trace Validation")
        print("=" * 70)

        config = ControllerConfig(
            tpot_slo_ms=args.tpot_slo_ms,
            power_budget_w=args.power_budget_w,
            thermal_warning_c=75.0,
            thermal_critical_c=85.0,
            lower_hysteresis_k=3,
            cooldown_s=20.0,
        )

        controller = ThermalSLOCapController(
            cap_rate_table_path='data/rate_tables/cap_rate_table_with_savings_*.parquet',
            lock_rate_table_path='data/rate_tables/lock_rate_table_*.parquet',
            model='Qwen2.5-7B-Instruct-Q4_K_M',
            config=config,
        )

        # Generate synthetic trace
        print(f"\nGenerating {args.trace_type} trace ({args.n_windows} windows)...")
        trace = generate_warming_trace(
            n_windows=args.n_windows,
            base_temp_c=55.0,
            peak_temp_c=82.0,
        )

        # Run simulation
        print(f"Running controller simulation...")
        states = simulate_synthetic_trace(controller, trace)

        # Print timeline
        print(f"\n{'W':>3} {'Temp°C':>7} {'Slope':>7} {'Cap':>6} {'Action':>8} "
              f"{'TPOT50':>7} {'TPOT95':>7} {'Pwr':>5} {'Switch':>6}")
        print("─" * 75)
        for s in states:
            print(f"{s.window_id:3d} {s.temperature_c:7.1f} {s.temperature_slope_c_per_s:7.3f} "
                  f"{s.current_cap_mhz:6d} {s.cap_action.value:>8} "
                  f"{s.tpot_p50_ms:7.1f} {s.tpot_p95_ms:7.1f} "
                  f"{s.avg_power_w:5.1f} {s.cap_switch_count:6d}")

        # Summary
        actions = [s.cap_action for s in states]
        temps = [s.temperature_c for s in states]
        caps = [s.current_cap_mhz for s in states]
        print(f"\nSummary:")
        print(f"  Actions: {Counter(a.value for a in actions)}")
        print(f"  Temperature: {min(temps):.1f} → {max(temps):.1f}°C")
        print(f"  Cap range: {min(caps)} → {max(caps)} MHz")
        print(f"  Total switches: {states[-1].cap_switch_count}")
        print(f"\n✅ Validation complete")

    else:
        print("Use --validate to run synthetic trace validation")
