#!/usr/bin/env python3
"""
Metrics Collector for Jetson Devices
Manages tegrastats process for collecting power, temperature, and frequency metrics.

Actual tegrastats output format (JetPack r36.x, Jetson AGX Orin):
  05-13-2026 18:21:47 RAM 16226/62841MB (lfb 44x4MB) SWAP 142/31420MB (cached 0MB)
  CPU [3%@1036,0%@1036,...] GR3D_FREQ 0%
  cpu@49.25C soc2@44.468C soc0@46.125C tj@49.25C soc1@45.468C
  VDD_GPU_SOC 3194mW/3194mW VDD_CPU_CV 399mW/399mW VIN_SYS_5V0 4206mW/4206mW
"""

import re
import subprocess
import logging
import time
import signal
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect system metrics via tegrastats on Jetson devices."""

    def __init__(self, interval_ms: int = 500, output_dir: str = "data/raw_logs"):
        self.interval_ms = interval_ms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.process = None
        self.log_file = None
        self.is_collecting = False

        self.tegrastats_path = self._find_tegrastats()
        self._samples: List[Dict] = []

    def _find_tegrastats(self) -> Optional[str]:
        for path in ['/usr/bin/tegrastats', 'tegrastats']:
            try:
                subprocess.run([path, '--help'], capture_output=True, timeout=5)
                return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        return None

    # ── Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def parse_tegrastats_line(line: str) -> Optional[Dict]:
        """Parse one line of tegrastats output into a metrics dict."""
        line = line.strip()
        if not line or 'RAM' not in line:
            return None

        m: Dict = {}

        # RAM: RAM 16226/62841MB
        ram = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if ram:
            m['ram_used_mb'] = int(ram.group(1))
            m['ram_total_mb'] = int(ram.group(2))

        # SWAP: SWAP 142/31420MB
        swap = re.search(r'SWAP\s+(\d+)/(\d+)MB', line)
        if swap:
            m['swap_used_mb'] = int(swap.group(1))
            m['swap_total_mb'] = int(swap.group(2))

        # CPU: [3%@1036,0%@1036,...]  → per-core util% and freq
        cpu_block = re.search(r'CPU\s+\[([^\]]+)\]', line)
        if cpu_block:
            cores = cpu_block.group(1).split(',')
            m['cpu_cores_online'] = len(cores)
            freqs = []
            utils = []
            for c in cores:
                parts = c.strip().split('@')
                try:
                    utils.append(float(parts[0].replace('%', '')))
                    freqs.append(int(parts[1]))
                except (IndexError, ValueError):
                    pass
            if freqs:
                m['cpu_freq_mhz'] = freqs[0]  # All cores same freq on Orin
                m['cpu_util_avg'] = sum(utils) / len(utils)

        # GPU freq: GR3D_FREQ 0%
        gpu_freq = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if gpu_freq:
            m['gpu_util_pct'] = int(gpu_freq.group(1))

        # Temperature: cpu@49.25C  soc2@44.468C  tj@49.25C  etc.
        for match in re.finditer(r'(\w+)@([\d.]+)C', line):
            name, temp = match.group(1), float(match.group(2))
            if name == 'cpu' or name == 'tj':
                m[f'temp_{name}_c'] = temp
            else:
                m[f'temp_{name}_c'] = temp

        # Power: VDD_GPU_SOC 3194mW/3194mW  VDD_CPU_CV 399mW/399mW  VIN_SYS_5V0 4206mW/4206mW
        power_total = 0
        for match in re.finditer(r'(VDD_\w+|VIN_\w+)\s+(\d+)mW/(\d+)mW', line):
            rail, avg_mw, _ = match.group(1), int(match.group(2)), int(match.group(3))
            m[f'power_{rail}_mw'] = avg_mw
            power_total += avg_mw
        m['power_total_mw'] = power_total
        m['power_total_w'] = power_total / 1000.0

        return m if m else None

    # ── Collection Control ───────────────────────────────────────────────

    def start_collection(self, tag: str = "") -> Optional[str]:
        """Start tegrastats in background. Returns log file path."""
        if not self.tegrastats_path:
            logger.error("tegrastats not found")
            return None
        if self.is_collecting:
            return str(self.log_file)

        ts = int(time.time())
        fname = f"tegrastats_{tag}_{ts}.log" if tag else f"tegrastats_{ts}.log"
        self.log_file = self.output_dir / fname
        self._samples = []

        cmd = [self.tegrastats_path, '--interval', str(self.interval_ms),
               '--logfile', str(self.log_file)]
        self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.is_collecting = True
        time.sleep(0.3)  # Let first sample arrive
        logger.info(f"tegrastats started: interval={self.interval_ms}ms, log={self.log_file}")
        return str(self.log_file)

    def stop_collection(self) -> bool:
        """Stop tegrastats and parse the log file."""
        if not self.is_collecting or not self.process:
            return False

        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

        self.is_collecting = False
        time.sleep(0.2)

        # Parse the log file
        if self.log_file and self.log_file.exists():
            self._parse_log_file(self.log_file)

        logger.info(f"tegrastats stopped. Parsed {len(self._samples)} samples from {self.log_file}")
        return True

    def _parse_log_file(self, path: Path):
        """Parse tegrastats log file into self._samples."""
        with open(path) as f:
            for line in f:
                parsed = self.parse_tegrastats_line(line)
                if parsed:
                    self._samples.append(parsed)

    def get_samples_df(self) -> pd.DataFrame:
        """Return all parsed samples as DataFrame."""
        if not self._samples:
            return pd.DataFrame()
        return pd.DataFrame(self._samples)

    # ── Energy Calculation ───────────────────────────────────────────────

    def compute_energy(self) -> Dict:
        """
        Compute energy metrics from collected samples.

        Returns dict with:
            avg_power_w, max_power_w, total_energy_j,
            energy_per_second, gpu_power_w, cpu_power_w,
            temp_cpu_c, temp_tj_c, n_samples
        """
        df = self.get_samples_df()
        if df.empty:
            return {}

        n = len(df)
        duration = n * (self.interval_ms / 1000.0)

        result = {'n_samples': n, 'duration_s': round(duration, 2)}

        # Total power
        if 'power_total_mw' in df.columns:
            result['avg_power_w'] = round(df['power_total_mw'].mean() / 1000.0, 3)
            result['max_power_w'] = round(df['power_total_mw'].max() / 1000.0, 3)
            result['total_energy_j'] = round(df['power_total_mw'].mean() * duration / 1000.0, 3)

        # Per-rail power
        for rail in ['VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0']:
            col = f'power_{rail}_mw'
            if col in df.columns:
                short = rail.replace('VDD_', '').replace('VIN_', '').lower()
                result[f'avg_{short}_w'] = round(df[col].mean() / 1000.0, 3)

        # Temperature
        for tcol in ['temp_cpu_c', 'temp_tj_c']:
            if tcol in df.columns:
                result[f'avg_{tcol}'] = round(df[tcol].mean(), 1)
                result[f'max_{tcol}'] = round(df[tcol].max(), 1)

        return result

    def __enter__(self):
        self.start_collection()
        return self

    def __exit__(self, *args):
        self.stop_collection()


def main():
    """Quick test: collect for 5 seconds and print energy."""
    mc = MetricsCollector(interval_ms=500)
    mc.start_collection(tag="test")
    logger.info("Collecting for 5 seconds...")
    time.sleep(5)
    mc.stop_collection()

    energy = mc.compute_energy()
    print("\nEnergy Metrics:")
    for k, v in energy.items():
        print(f"  {k}: {v}")

    df = mc.get_samples_df()
    if not df.empty:
        print(f"\nFirst 3 samples:")
        print(df.head(3).to_string())


if __name__ == "__main__":
    main()
