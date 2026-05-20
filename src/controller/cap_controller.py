#!/usr/bin/env python3
"""
Cap Controller for Jetson Devices

Provides three frequency control modes:
- lock mode: Force exact frequency (for offline profiling)
- cap mode: Set frequency upper bound with dynamic governor (for online control)
- dynamic mode: Restore default dynamic governors (for baseline)

Key insight: cap mode does NOT force exact frequency. It sets max_freq while
keeping the governor dynamic, so actual frequency may be lower than the cap.
This is the intended behavior — the governor adapts to real-time load.

Usage:
    from src.controller.cap_controller import CapController

    cc = CapController()

    # Offline profiling: lock exact frequency
    cc.lock_config(gpu_mhz=816, emc_mhz=204, cpu_mhz=1036)

    # Online control: set frequency cap
    cc.set_cap(gpu_cap_mhz=1020, emc_cap_mhz=665, cpu_cap_mhz=1497)

    # Restore defaults
    cc.restore_dynamic()

    # Read current state
    state = cc.read_actual_state()
"""

import time
import logging
from typing import Dict, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Frequency tables ──
GPU_FREQS_HZ = {
    306: 306000000, 408: 408000000, 510: 510000000, 612: 612000000,
    714: 714000000, 816: 816000000, 918: 918000000, 1020: 1020000000,
    1122: 1122000000, 1224: 1224000000, 1300: 1300500000,
}
EMC_HZ = {204: 204000000, 665: 665600000, 2133: 2133000000, 3199: 3199000000}
CPU_HZ = {1036: 1036800, 1497: 1497600}

GPU_PATH = '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu'
CPU_PATH = '/sys/devices/system/cpu/cpu0/cpufreq'
EMC_MAX = '/sys/kernel/debug/emc/max_rate'
EMC_MIN = '/sys/kernel/debug/emc/min_rate'
EMC_CLK = '/sys/kernel/debug/clk/emc/clk_rate'

MIN_GPU_MHZ = 306
MIN_EMC_MHZ = 204
MIN_CPU_MHZ = 1036
MAX_GPU_MHZ = 1300
MAX_EMC_MHZ = 3199
MAX_CPU_MHZ = 1497


class CapController:
    """Three-mode frequency controller: lock / cap / dynamic."""

    def __init__(self):
        self._mode = 'dynamic'
        self._last_lock: Dict = {}
        self._last_cap: Dict = {}

    # ── Mode 1: Lock (profiling only) ──────────────────────────────────

    def lock_config(self, gpu_mhz: int, emc_mhz: int, cpu_mhz: int) -> Dict:
        """
        Force exact GPU/EMC/CPU frequencies. Uses performance governor.
        Only for offline profiling — not for online control.
        """
        t0 = time.monotonic()
        actual_gpu = self._lock_gpu(gpu_mhz)
        actual_emc = self._lock_emc(emc_mhz)
        actual_cpu = self._lock_cpu(cpu_mhz)
        latency_ms = (time.monotonic() - t0) * 1000

        self._mode = 'lock'
        self._last_lock = {
            'target_gpu_mhz': gpu_mhz, 'target_emc_mhz': emc_mhz, 'target_cpu_mhz': cpu_mhz,
            'actual_gpu_mhz': actual_gpu, 'actual_emc_mhz': actual_emc, 'actual_cpu_mhz': actual_cpu,
            'apply_latency_ms': round(latency_ms, 1),
            'governor_gpu': 'performance', 'governor_cpu': 'userspace',
        }
        logger.info(f"LOCK: GPU={actual_gpu} EMC={actual_emc} CPU={actual_cpu} ({latency_ms:.0f}ms)")
        return self._last_lock

    # ── Mode 2: Cap (online control) ───────────────────────────────────

    def set_cap(self, gpu_cap_mhz: int, emc_cap_mhz: int, cpu_cap_mhz: int,
                gpu_min_mhz: int = MIN_GPU_MHZ, emc_min_mhz: int = MIN_EMC_MHZ,
                cpu_min_mhz: int = MIN_CPU_MHZ) -> Dict:
        """
        Set frequency upper bounds while keeping dynamic governors.
        Actual frequency may be lower than cap — this is intended.
        """
        t0 = time.monotonic()
        actual_gpu = self._cap_gpu(gpu_cap_mhz, gpu_min_mhz)
        actual_emc = self._cap_emc(emc_cap_mhz, emc_min_mhz)
        actual_cpu = self._cap_cpu(cpu_cap_mhz, cpu_min_mhz)
        latency_ms = (time.monotonic() - t0) * 1000

        self._mode = 'cap'
        self._last_cap = {
            'target_gpu_cap_mhz': gpu_cap_mhz, 'target_emc_cap_mhz': emc_cap_mhz,
            'target_cpu_cap_mhz': cpu_cap_mhz,
            'gpu_min_mhz': gpu_min_mhz, 'emc_min_mhz': emc_min_mhz, 'cpu_min_mhz': cpu_min_mhz,
            'actual_gpu_mhz': actual_gpu, 'actual_emc_mhz': actual_emc, 'actual_cpu_mhz': actual_cpu,
            'apply_latency_ms': round(latency_ms, 1),
            'governor_gpu': self._read_gpu_governor(),
            'governor_cpu': self._read_cpu_governor(),
            'control_mode': 'cap',
        }
        logger.info(f"CAP: GPU<={gpu_cap_mhz} EMC<={emc_cap_mhz} CPU<={cpu_cap_mhz} "
                    f"(actual: {actual_gpu}/{actual_emc}/{actual_cpu} {latency_ms:.0f}ms)")
        return self._last_cap

    # ── Mode 3: Dynamic (restore defaults) ─────────────────────────────

    def restore_dynamic(self) -> Dict:
        """Restore default dynamic governors and full frequency range."""
        t0 = time.monotonic()
        self._restore_gpu_dynamic()
        self._restore_cpu_dynamic()
        self._restore_emc_dynamic()
        latency_ms = (time.monotonic() - t0) * 1000

        self._mode = 'dynamic'
        state = self.read_actual_state()
        state['apply_latency_ms'] = round(latency_ms, 1)
        logger.info(f"DYNAMIC restored ({latency_ms:.0f}ms)")
        return state

    # ── Read state ──────────────────────────────────────────────────────

    def read_actual_state(self) -> Dict:
        """Read current frequency, governor, and temperature."""
        gpu_freq = self._read_gpu_freq()
        emc_freq = self._read_emc_freq()
        cpu_freq = self._read_cpu_freq()
        gpu_gov = self._read_gpu_governor()
        cpu_gov = self._read_cpu_governor()
        temp = self._read_temperature()

        return {
            'control_mode': self._mode,
            'actual_gpu_mhz': gpu_freq,
            'actual_emc_mhz': emc_freq,
            'actual_cpu_mhz': cpu_freq,
            'governor_gpu': gpu_gov,
            'governor_cpu': cpu_gov,
            'temperature_c': temp,
        }

    @property
    def mode(self) -> str:
        return self._mode

    # ── GPU helpers ─────────────────────────────────────────────────────

    def _lock_gpu(self, mhz: int) -> int:
        hz = GPU_FREQS_HZ.get(mhz, mhz * 1000000)
        self._write(f'{GPU_PATH}/governor', 'performance')
        self._write(f'{GPU_PATH}/max_freq', str(hz))
        self._write(f'{GPU_PATH}/min_freq', str(hz))
        time.sleep(0.3)
        return self._read_gpu_freq()

    def _cap_gpu(self, cap_mhz: int, min_mhz: int) -> int:
        cap_hz = GPU_FREQS_HZ.get(cap_mhz, cap_mhz * 1000000)
        min_hz = GPU_FREQS_HZ.get(min_mhz, min_mhz * 1000000)
        try:
            self._write(f'{GPU_PATH}/governor', 'simple_ondemand')
        except OSError:
            self._write(f'{GPU_PATH}/governor', 'performance')
        self._write(f'{GPU_PATH}/min_freq', str(min_hz))
        self._write(f'{GPU_PATH}/max_freq', str(cap_hz))
        return self._read_gpu_freq()

    def _restore_gpu_dynamic(self):
        min_hz = GPU_FREQS_HZ[MIN_GPU_MHZ]
        max_hz = GPU_FREQS_HZ[MAX_GPU_MHZ]
        self._write(f'{GPU_PATH}/min_freq', str(min_hz))
        self._write(f'{GPU_PATH}/max_freq', str(max_hz))
        try:
            self._write(f'{GPU_PATH}/governor', 'simple_ondemand')
        except OSError:
            self._write(f'{GPU_PATH}/governor', 'performance')

    def _read_gpu_freq(self) -> int:
        return int(self._read(f'{GPU_PATH}/cur_freq')) // 1000000

    def _read_gpu_governor(self) -> str:
        return self._read(f'{GPU_PATH}/governor').strip()

    # ── EMC helpers ─────────────────────────────────────────────────────

    def _lock_emc(self, mhz: int) -> int:
        hz = EMC_HZ.get(mhz, mhz * 1000000)
        max_hz = max(EMC_HZ.values())
        self._write(EMC_MAX, str(max_hz))
        self._write(EMC_MIN, str(min(EMC_HZ.values())))
        self._write(EMC_MIN, str(hz))
        self._write(EMC_MAX, str(hz))
        time.sleep(0.3)
        return self._read_emc_freq()

    def _cap_emc(self, cap_mhz: int, min_mhz: int) -> int:
        cap_hz = EMC_HZ.get(cap_mhz, cap_mhz * 1000000)
        min_hz = EMC_HZ.get(min_mhz, min_mhz * 1000000)
        max_hz = max(EMC_HZ.values())
        self._write(EMC_MAX, str(max_hz))
        self._write(EMC_MIN, str(min_hz))
        self._write(EMC_MAX, str(cap_hz))
        return self._read_emc_freq()

    def _restore_emc_dynamic(self):
        self._write(EMC_MIN, str(min(EMC_HZ.values())))
        self._write(EMC_MAX, str(max(EMC_HZ.values())))

    def _read_emc_freq(self) -> int:
        return int(self._read(EMC_CLK)) // 1000000

    # ── CPU helpers ─────────────────────────────────────────────────────

    def _lock_cpu(self, mhz: int) -> int:
        hz = CPU_HZ.get(mhz, mhz * 1000)
        for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
            gov = cpu / 'cpufreq' / 'scaling_governor'
            mx = cpu / 'cpufreq' / 'scaling_max_freq'
            mn = cpu / 'cpufreq' / 'scaling_min_freq'
            if gov.exists():
                try:
                    self._write(str(gov), 'userspace')
                    self._write(str(mx), str(hz))
                    self._write(str(mn), str(hz))
                except PermissionError:
                    pass
        time.sleep(0.1)
        return self._read_cpu_freq()

    def _cap_cpu(self, cap_mhz: int, min_mhz: int) -> int:
        cap_hz = CPU_HZ.get(cap_mhz, cap_mhz * 1000)
        min_hz = CPU_HZ.get(min_mhz, min_mhz * 1000)
        for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
            gov = cpu / 'cpufreq' / 'scaling_governor'
            mx = cpu / 'cpufreq' / 'scaling_max_freq'
            mn = cpu / 'cpufreq' / 'scaling_min_freq'
            if gov.exists():
                try:
                    self._write(str(gov), 'schedutil')
                    self._write(str(mn), str(min_hz))
                    self._write(str(mx), str(cap_hz))
                except PermissionError:
                    pass
        return self._read_cpu_freq()

    def _restore_cpu_dynamic(self):
        max_hz = max(CPU_HZ.values())
        min_hz = min(CPU_HZ.values())
        for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
            gov = cpu / 'cpufreq' / 'scaling_governor'
            mx = cpu / 'cpufreq' / 'scaling_max_freq'
            mn = cpu / 'cpufreq' / 'scaling_min_freq'
            if gov.exists():
                try:
                    self._write(str(gov), 'schedutil')
                    self._write(str(mx), str(max_hz))
                    self._write(str(mn), str(min_hz))
                except PermissionError:
                    pass

    def _read_cpu_freq(self) -> int:
        return int(self._read(f'{CPU_PATH}/scaling_cur_freq')) // 1000

    def _read_cpu_governor(self) -> str:
        return self._read(f'{CPU_PATH}/scaling_governor').strip()

    # ── Temperature ─────────────────────────────────────────────────────

    def _read_temperature(self) -> float:
        try:
            val = self._read('/sys/devices/virtual/thermal/thermal_zone0/temp')
            return float(val.strip()) / 1000.0
        except Exception:
            return 0.0

    # ── Low-level I/O ───────────────────────────────────────────────────

    @staticmethod
    def _write(path: str, value: str):
        with open(path, 'w') as f:
            f.write(value)

    @staticmethod
    def _read(path: str) -> str:
        with open(path, 'r') as f:
            return f.read().strip()
