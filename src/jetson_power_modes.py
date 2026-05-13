#!/usr/bin/env python3
"""
Jetson Power Mode Management
Manage nvpmodel and jetson_clocks for baseline comparisons.
"""

import subprocess
import re
import logging
import time
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_current_nvpmodel() -> Optional[str]:
    """Parse nvpmodel -q to get current power mode name."""
    try:
        result = subprocess.run(['nvpmodel', '-q'], capture_output=True, text=True, timeout=10)
        output = result.stdout + result.stderr
        # Parse "NV Power Mode: MODE_NAME"
        match = re.search(r'NV Power Mode:\s*(.+)', output)
        if match:
            return match.group(1).strip()
    except FileNotFoundError:
        logger.warning("nvpmodel not found — not running on Jetson?")
    except Exception as e:
        logger.error(f"Failed to query nvpmodel: {e}")
    return None


def list_nvpmodel_modes() -> Dict[str, int]:
    """Parse /etc/nvpmodel.conf to list available modes."""
    modes = {}
    conf_path = Path('/etc/nvpmodel.conf')
    if not conf_path.exists():
        logger.warning(f"{conf_path} not found")
        return modes

    try:
        content = conf_path.read_text()
        # Parse sections like [MODE_0] and PARAM_NAME = VALUE
        current_id = None
        current_name = None
        for line in content.splitlines():
            line = line.strip()
            # Match [MODE_N]
            mode_match = re.match(r'\[MODE_(\d+)\]', line)
            if mode_match:
                if current_id is not None and current_name:
                    modes[current_name] = current_id
                current_id = int(mode_match.group(1))
                current_name = None
                continue
            # Match APPEND ... NAME
            name_match = re.match(r'APPEND\s+.*\bNAME\s+(\S+)', line)
            if name_match and current_id is not None:
                current_name = name_match.group(1)
        # Don't forget the last mode
        if current_id is not None and current_name:
            modes[current_name] = current_id
    except Exception as e:
        logger.error(f"Failed to parse nvpmodel.conf: {e}")

    return modes


def set_nvpmodel_mode(mode_id: int) -> bool:
    """Set nvpmodel to specified mode (requires sudo)."""
    try:
        result = subprocess.run(
            ['sudo', 'nvpmodel', '-m', str(mode_id)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info(f"Set nvpmodel to mode {mode_id}")
            return True
        else:
            logger.error(f"nvpmodel set failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to set nvpmodel: {e}")
        return False


def enable_jetson_clocks() -> bool:
    """Enable jetson_clocks max frequency override (requires sudo)."""
    try:
        result = subprocess.run(
            ['sudo', 'jetson_clocks'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info("Enabled jetson_clocks")
            return True
        else:
            logger.error(f"jetson_clocks enable failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to enable jetson_clocks: {e}")
        return False


def disable_jetson_clocks() -> bool:
    """Restore jetson_clocks to default frequencies (requires sudo)."""
    try:
        result = subprocess.run(
            ['sudo', 'jetson_clocks', '--restore'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            logger.info("Restored jetson_clocks defaults")
            return True
        else:
            logger.error(f"jetson_clocks restore failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Failed to restore jetson_clocks: {e}")
        return False


def _get_frequency_sysfs(target: str) -> Optional[int]:
    """Read current frequency from sysfs (no sudo needed)."""
    paths = {
        'gpu': '/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/cur_freq',
        'cpu': '/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq',
        'emc': '/sys/kernel/debug/bpmp/debug/emc_rate',
    }
    path = paths.get(target)
    if not path:
        return None
    try:
        with open(path, 'r') as f:
            freq_khz = int(f.read().strip())
            return freq_khz // 1000  # Convert to MHz
    except Exception as e:
        logger.debug(f"Failed to read {target} freq from sysfs: {e}")
        return None


def record_current_power_state() -> Dict:
    """Snapshot current power state for later restoration."""
    state = {
        'nvpmodel': get_current_nvpmodel(),
        'gpu_freq_mhz': _get_frequency_sysfs('gpu'),
        'cpu_freq_mhz': _get_frequency_sysfs('cpu'),
        'emc_freq_mhz': _get_frequency_sysfs('emc'),
        'timestamp': time.time()
    }
    logger.info(f"Recorded power state: nvpmodel={state['nvpmodel']}, "
                f"gpu={state['gpu_freq_mhz']}MHz, cpu={state['cpu_freq_mhz']}MHz, "
                f"emc={state['emc_freq_mhz']}MHz")
    return state


def restore_power_state(state: Dict) -> bool:
    """Restore previously recorded power state."""
    success = True

    # Restore nvpmodel
    if state.get('nvpmodel'):
        modes = list_nvpmodel_modes()
        mode_id = modes.get(state['nvpmodel'])
        if mode_id is not None:
            success &= set_nvpmodel_mode(mode_id)
            time.sleep(2)

    # Restore jetson_clocks (disable to reset to nvpmodel defaults)
    disable_jetson_clocks()
    time.sleep(1)

    logger.info(f"Restored power state from {state.get('nvpmodel', 'unknown')}")
    return success


def apply_baseline_config(baseline_config: Dict) -> bool:
    """Apply a baseline configuration from baselines.yaml."""
    apply_spec = baseline_config.get('apply', {})
    success = True

    # Set nvpmodel
    nvpmodel = apply_spec.get('nvpmodel')
    if nvpmodel and nvpmodel != 'MAXN':
        modes = list_nvpmodel_modes()
        mode_id = modes.get(nvpmodel)
        if mode_id is not None:
            success &= set_nvpmodel_mode(mode_id)
            time.sleep(2)
    elif nvpmodel == 'MAXN' or not nvpmodel:
        # Default to MAXN for full frequency range
        modes = list_nvpmodel_modes()
        mode_id = modes.get('MAXN')
        if mode_id is not None:
            success &= set_nvpmodel_mode(mode_id)
            time.sleep(2)

    # Apply jetson_clocks
    if apply_spec.get('jetson_clocks'):
        success &= enable_jetson_clocks()
    else:
        disable_jetson_clocks()
    time.sleep(1)

    return success
