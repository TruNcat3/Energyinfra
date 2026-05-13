#!/usr/bin/env python3
"""
Frequency Controller for Jetson Devices
Manages GPU, CPU, and EMC frequency settings on Jetson Orin devices.
"""

import subprocess
import logging
import time
from typing import Dict, Optional, Tuple
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FrequencyController:
    """
    Controller for managing GPU, CPU, and EMC frequencies on Jetson devices.

    Supports multiple frequency control methods:
    - hybrid: Read via sysfs (no sudo required), set via jetson_clocks (sudo required) - RECOMMENDED
    - sysfs: Direct /sys filesystem manipulation for both read and set
    - jetson_clocks: Official NVIDIA tool for both read and set (requires sudo)
    """

    def __init__(self, config: Dict, control_method: str = "hybrid"):
        """
        Initialize the frequency controller.

        Args:
            config: Platform configuration dictionary
            control_method: Frequency control method ('jetson_clocks', 'sysfs', 'hybrid')
                           'hybrid' reads via sysfs (no sudo) and sets via jetson_clocks (sudo)
        """
        self.config = config
        self.control_method = control_method
        self.frequencies = config.get('frequencies', {})

        # Frequency paths for sysfs method (corrected for actual Jetson paths)
        self.freq_paths = {
            'gpu': '/sys/class/devfreq/17000000.gpu',
            'cpu': '/sys/devices/system/cpu/cpu0/cpufreq',  # Use cpu0 direct path
            'emc': '/sys/class/devfreq/17000000.emc'  # May not exist, will fallback to jetson_clocks
        }

        # Default frequency storage
        self.default_frequencies = {
            'gpu': None,
            'cpu': None,
            'emc': None
        }

        # Load frequency presets from config
        self.presets = {}
        if self.frequencies:
            for target in ['gpu', 'cpu', 'emc']:
                if target in self.frequencies and 'presets' in self.frequencies[target]:
                    self.presets[target] = self.frequencies[target]['presets']

    def get_frequency(self, target: str) -> int:
        """
        Get current frequency for specified target.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')

        Returns:
            Current frequency in MHz, or -1 if failed
        """
        if target not in ['gpu', 'cpu', 'emc']:
            logger.error(f"Invalid target: {target}")
            return -1

        try:
            if self.control_method == 'jetson_clocks':
                return self._get_frequency_jetson_clocks(target)
            elif self.control_method == 'sysfs':
                return self._get_frequency_sysfs(target)
            elif self.control_method == 'hybrid':
                # Hybrid mode: read via sysfs (no sudo needed)
                return self._get_frequency_sysfs(target)
            else:
                logger.error(f"Unsupported control method: {self.control_method}")
                return -1
        except Exception as e:
            logger.error(f"Failed to get {target} frequency: {e}")
            return -1

    def _get_frequency_jetson_clocks(self, target: str) -> int:
        """
        Get frequency using jetson_clocks command.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')

        Returns:
            Current frequency in MHz
        """
        try:
            result = subprocess.run(
                ['jetson_clocks', '--show'],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"jetson_clocks failed: {result.stderr}")

            # Parse output to extract frequency
            frequency = self._parse_jetson_clocks_output(result.stdout, target)
            return frequency

        except subprocess.TimeoutExpired:
            logger.error("jetson_clocks command timed out")
            return -1
        except FileNotFoundError:
            logger.error("jetson_clocks command not found")
            return -1

    def _get_frequency_sysfs(self, target: str) -> int:
        """
        Get frequency using sysfs interface.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')

        Returns:
            Current frequency in MHz
        """
        try:
            if target == 'cpu':
                freq_file = f"{self.freq_paths[target]}/scaling_cur_freq"
            else:
                freq_file = f"{self.freq_paths[target]}/cur_freq"

            if not Path(freq_file).exists():
                logger.error(f"Frequency file not found: {freq_file}")
                return -1

            with open(freq_file, 'r') as f:
                freq_hz = int(f.read().strip())
                return freq_hz // 1000  # Convert Hz to MHz

        except Exception as e:
            logger.error(f"Failed to read frequency from sysfs: {e}")
            return -1

    def _parse_jetson_clocks_output(self, output: str, target: str) -> int:
        """
        Parse jetson_clocks output to extract frequency.

        Args:
            output: jetson_clocks command output
            target: Target device ('gpu', 'cpu', 'emc')

        Returns:
            Frequency in MHz
        """
        # Parse based on target
        for line in output.split('\n'):
            if target.upper() in line and 'MHz' in line:
                # Extract frequency value
                parts = line.split()
                for part in parts:
                    try:
                        freq = int(part.replace('MHz', ''))
                        return freq
                    except ValueError:
                        continue

        logger.warning(f"Could not parse {target} frequency from jetson_clocks output")
        return -1

    def set_frequency(self, target: str, frequency: int) -> bool:
        """
        Set frequency for specified target.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')
            frequency: Frequency in MHz

        Returns:
            True if successful, False otherwise
        """
        if target not in ['gpu', 'cpu', 'emc']:
            logger.error(f"Invalid target: {target}")
            return False

        try:
            if self.control_method == 'jetson_clocks':
                return self._set_frequency_jetson_clocks(target, frequency)
            elif self.control_method == 'sysfs':
                return self._set_frequency_sysfs(target, frequency)
            elif self.control_method == 'hybrid':
                # Hybrid mode: set via jetson_clocks (requires sudo)
                return self._set_frequency_jetson_clocks(target, frequency)
            else:
                logger.error(f"Unsupported control method: {self.control_method}")
                return False
        except Exception as e:
            logger.error(f"Failed to set {target} frequency to {frequency} MHz: {e}")
            return False

    def _set_frequency_jetson_clocks(self, target: str, frequency: int) -> bool:
        """
        Set frequency using jetson_clocks command.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')
            frequency: Frequency in MHz

        Returns:
            True if successful, False otherwise
        """
        try:
            # Save current frequency if not already saved
            if self.default_frequencies[target] is None:
                self.default_frequencies[target] = self.get_frequency(target)
                logger.info(f"Saved default {target} frequency: {self.default_frequencies[target]} MHz")

            # Construct jetson_clocks command
            cmd = ['jetson_clocks', '--store']

            if target == 'gpu':
                cmd.extend(['--gpu', str(frequency)])
            elif target == 'cpu':
                cmd.extend(['--cpu', str(frequency)])
            elif target == 'emc':
                cmd.extend(['--emc', str(frequency)])

            logger.info(f"Setting {target} frequency to {frequency} MHz (requires sudo)")

            # Add sudo to the command for jetson_clocks
            sudo_cmd = ['sudo'] + cmd

            result = subprocess.run(
                sudo_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                logger.error(f"jetson_clocks failed: {result.stderr}")
                return False

            # Verify the frequency was set correctly
            time.sleep(1)  # Wait for frequency change to take effect
            actual_freq = self.get_frequency(target)
            tolerance = self.config.get('frequency_control', {}).get('tolerance_mhz', {}).get(target, 10)

            if abs(actual_freq - frequency) > tolerance:
                logger.warning(f"{target} frequency set to {frequency} MHz but actual is {actual_freq} MHz")
                return False

            logger.info(f"Successfully set {target} frequency to {actual_freq} MHz")
            return True

        except subprocess.TimeoutExpired:
            logger.error("jetson_clocks command timed out")
            return False
        except FileNotFoundError:
            logger.error("jetson_clocks command not found")
            return False

    def _set_frequency_sysfs(self, target: str, frequency: int) -> bool:
        """
        Set frequency using sysfs interface.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')
            frequency: Frequency in MHz

        Returns:
            True if successful, False otherwise
        """
        try:
            # Save current frequency if not already saved
            if self.default_frequencies[target] is None:
                self.default_frequencies[target] = self.get_frequency(target)
                logger.info(f"Saved default {target} frequency: {self.default_frequencies[target]} MHz")

            # Convert MHz to Hz for sysfs
            freq_hz = frequency * 1000

            if target == 'cpu':
                freq_file = f"{self.freq_paths[target]}/scaling_setspeed"
                min_freq_file = f"{self.freq_paths[target]}/scaling_min_freq"
                max_freq_file = f"{self.freq_paths[target]}/scaling_max_freq"

                # Set min and max frequencies
                with open(min_freq_file, 'w') as f:
                    f.write(str(freq_hz))
                with open(max_freq_file, 'w') as f:
                    f.write(str(freq_hz))
            else:
                freq_file = f"{self.freq_paths[target]}/min_freq"

                # Set minimum frequency (which sets the target frequency)
                with open(freq_file, 'w') as f:
                    f.write(str(freq_hz))

            # Verify the frequency was set correctly
            time.sleep(1)  # Wait for frequency change to take effect
            actual_freq = self.get_frequency(target)

            if actual_freq == -1 or abs(actual_freq - frequency) > 10:
                logger.warning(f"{target} frequency set to {frequency} MHz but actual is {actual_freq} MHz")
                return False

            logger.info(f"Successfully set {target} frequency to {actual_freq} MHz")
            return True

        except PermissionError:
            logger.error(f"Permission denied when setting {target} frequency. Try running with sudo.")
            return False
        except Exception as e:
            logger.error(f"Failed to set {target} frequency via sysfs: {e}")
            return False

    def set_frequency_preset(self, target: str, preset: str) -> bool:
        """
        Set frequency using predefined preset.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')
            preset: Preset name ('low', 'mid', 'high')

        Returns:
            True if successful, False otherwise
        """
        if preset not in ['low', 'mid', 'high']:
            logger.error(f"Invalid preset: {preset}")
            return False

        if target not in self.presets or preset not in self.presets[target]:
            logger.error(f"Preset {preset} not found for {target}")
            return False

        frequency = self.presets[target][preset]
        return self.set_frequency(target, frequency)

    def set_all_frequencies(self, gpu_freq: int, cpu_freq: int, emc_freq: int) -> bool:
        """
        Set all frequencies at once.

        Args:
            gpu_freq: GPU frequency in MHz
            cpu_freq: CPU frequency in MHz
            emc_freq: EMC frequency in MHz

        Returns:
            True if all successful, False otherwise
        """
        logger.info(f"Setting all frequencies - GPU: {gpu_freq} MHz, CPU: {cpu_freq} MHz, EMC: {emc_freq} MHz")

        results = []
        results.append(self.set_frequency('gpu', gpu_freq))
        results.append(self.set_frequency('cpu', cpu_freq))
        results.append(self.set_frequency('emc', emc_freq))

        success = all(results)
        if success:
            logger.info("Successfully set all frequencies")
        else:
            logger.error("Failed to set some frequencies")

        return success

    def set_all_frequencies_preset(self, gpu_preset: str, cpu_preset: str, emc_preset: str) -> bool:
        """
        Set all frequencies using presets.

        Args:
            gpu_preset: GPU preset ('low', 'mid', 'high')
            cpu_preset: CPU preset ('low', 'mid', 'high')
            emc_preset: EMC preset ('low', 'mid', 'high')

        Returns:
            True if all successful, False otherwise
        """
        gpu_freq = self.presets.get('gpu', {}).get(gpu_preset)
        cpu_freq = self.presets.get('cpu', {}).get(cpu_preset)
        emc_freq = self.presets.get('emc', {}).get(emc_preset)

        if None in [gpu_freq, cpu_freq, emc_freq]:
            logger.error("Invalid preset combination")
            return False

        return self.set_all_frequencies(gpu_freq, cpu_freq, emc_freq)

    def save_defaults(self) -> bool:
        """
        Save current frequencies as defaults.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Saving current frequencies as defaults")

        success = True
        for target in ['gpu', 'cpu', 'emc']:
            current_freq = self.get_frequency(target)
            if current_freq != -1:
                self.default_frequencies[target] = current_freq
                logger.info(f"Saved default {target} frequency: {current_freq} MHz")
            else:
                logger.error(f"Failed to get {target} frequency")
                success = False

        return success

    def restore_defaults(self) -> bool:
        """
        Restore saved default frequencies.

        Returns:
            True if all successful, False otherwise
        """
        logger.info("Restoring default frequencies")

        success = True
        for target in ['gpu', 'cpu', 'emc']:
            if self.default_frequencies[target] is not None:
                if not self.set_frequency(target, self.default_frequencies[target]):
                    logger.error(f"Failed to restore {target} frequency")
                    success = False
            else:
                logger.warning(f"No default frequency saved for {target}")

        if success:
            logger.info("Successfully restored all default frequencies")
        else:
            logger.error("Failed to restore some default frequencies")

        return success

    def get_all_frequencies(self) -> Dict[str, int]:
        """
        Get current frequencies for all targets.

        Returns:
            Dictionary with current frequencies for gpu, cpu, emc
        """
        frequencies = {}
        for target in ['gpu', 'cpu', 'emc']:
            frequencies[target] = self.get_frequency(target)

        return frequencies

    def verify_frequency_support(self) -> Dict[str, bool]:
        """
        Verify if frequency control is supported for each target.

        Returns:
            Dictionary with support status for each target
        """
        support_status = {}

        for target in ['gpu', 'cpu', 'emc']:
            # Try to get current frequency
            current_freq = self.get_frequency(target)
            support_status[target] = current_freq != -1

        logger.info(f"Frequency support status: {support_status}")
        return support_status

    def get_available_frequencies(self, target: str) -> list:
        """
        Get available frequency options for a target.

        Args:
            target: Target device ('gpu', 'cpu', 'emc')

        Returns:
            List of available frequencies in MHz
        """
        if target not in self.frequencies or 'available' not in self.frequencies[target]:
            logger.warning(f"No available frequencies defined for {target}")
            return []

        return self.frequencies[target]['available']


def main():
    """
    Test function for frequency controller.
    """
    # Example configuration
    config = {
        'frequencies': {
            'gpu': {
                'available': [378, 846, 1428],
                'presets': {
                    'low': 378,
                    'mid': 846,
                    'high': 1428
                }
            },
            'cpu': {
                'available': [1020, 1479, 2015],
                'presets': {
                    'low': 1020,
                    'mid': 1479,
                    'high': 2015
                }
            },
            'emc': {
                'available': [133, 1600, 2133],
                'presets': {
                    'low': 133,
                    'mid': 1600,
                    'high': 2133
                }
            }
        },
        'frequency_control': {
            'tolerance_mhz': {
                'gpu': 10,
                'cpu': 10,
                'emc': 10
            }
        }
    }

    # Create controller
    controller = FrequencyController(config, control_method='jetson_clocks')

    # Test frequency control
    print("Testing frequency control...")

    # Get current frequencies
    current_freqs = controller.get_all_frequencies()
    print(f"Current frequencies: {current_freqs}")

    # Save defaults
    controller.save_defaults()

    # Verify support
    support = controller.verify_frequency_support()
    print(f"Frequency support: {support}")

    # Get available frequencies
    for target in ['gpu', 'cpu', 'emc']:
        available = controller.get_available_frequencies(target)
        print(f"Available {target} frequencies: {available}")


if __name__ == "__main__":
    main()