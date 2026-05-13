#!/usr/bin/env python3
"""
Metrics Collector for Jetson Devices
Manages tegrastats process for collecting power, temperature, and frequency metrics.
"""

import subprocess
import logging
import time
import signal
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collector for system metrics using tegrastats on Jetson devices.

    Collects:
    - Power consumption (GPU, CPU, total)
    - Temperature (GPU, CPU, SoC)
    - Frequency (GPU, CPU, EMC)
    - Memory usage (RAM, GPU)
    - CPU utilization
    """

    def __init__(self, interval_ms: int = 1000, output_dir: str = "data/raw_logs"):
        """
        Initialize metrics collector.

        Args:
            interval_ms: Sampling interval in milliseconds (default: 1000ms)
            output_dir: Directory to save log files
        """
        self.interval_ms = interval_ms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Process management
        self.process = None
        self.log_file = None
        self.is_collecting = False
        self.data_queue = queue.Queue()

        # Tegrastats path
        self.tegrastats_path = self._find_tegrastats()

        if not self.tegrastats_path:
            logger.error("tegrastats not found. Please ensure JetPack is installed.")

        # Metrics storage
        self.metrics_data = []

        # Thread for parsing output
        self.parse_thread = None

    def _find_tegrastats(self) -> Optional[str]:
        """
        Find tegrastats executable.

        Returns:
            Path to tegrastats, or None if not found
        """
        possible_paths = [
            '/usr/bin/tegrastats',
            '/usr/local/bin/tegrastats',
            'tegrastats'  # Try system PATH
        ]

        for path in possible_paths:
            try:
                result = subprocess.run(
                    [path, '--help'],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info(f"Found tegrastats at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        return None

    def start_collection(self, log_filename: Optional[str] = None) -> Optional[str]:
        """
        Start collecting metrics in background.

        Args:
            log_filename: Optional filename for log output. If None, auto-generate.

        Returns:
            Path to log file, or None if failed
        """
        if not self.tegrastats_path:
            logger.error("Cannot start collection: tegrastats not found")
            return None

        if self.is_collecting:
            logger.warning("Collection already in progress")
            return str(self.log_file)

        # Generate log filename if not provided
        if log_filename is None:
            timestamp = int(time.time())
            log_filename = f"tegrastats_{timestamp}.log"

        self.log_file = self.output_dir / log_filename

        try:
            # Start tegrastats process
            cmd = [
                self.tegrastats,
                '--interval', str(self.interval_ms),
                '--logfile', str(self.log_file)
            ]

            logger.info(f"Starting tegrastats with interval {self.interval_ms}ms")
            logger.info(f"Logging to: {self.log_file}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            self.is_collecting = True

            # Start parsing thread
            self.parse_thread = threading.Thread(
                target=self._parse_output,
                daemon=True
            )
            self.parse_thread.start()

            logger.info("Metrics collection started successfully")
            return str(self.log_file)

        except Exception as e:
            logger.error(f"Failed to start metrics collection: {e}")
            return None

    def _parse_output(self):
        """
        Parse tegrastats output and store in queue.
        """
        if not self.process:
            return

        try:
            for line in self.process.stdout:
                try:
                    metrics = self.parse_line(line)
                    if metrics:
                        timestamp = time.time()
                        metrics['timestamp'] = timestamp
                        self.data_queue.put(metrics)
                except Exception as e:
                    logger.warning(f"Error parsing line: {e}, line: {line}")

        except Exception as e:
            logger.error(f"Error in parsing thread: {e}")

    def parse_line(self, line: str) -> Optional[Dict]:
        """
        Parse a single line of tegrastats output.

        Args:
            line: Single line from tegrastats output

        Returns:
            Dictionary of parsed metrics, or None if parsing failed
        """
        try:
            # Remove leading/trailing whitespace
            line = line.strip()

            if not line or line.startswith('tegrastats'):
                return None

            metrics = {}

            # Parse different metric types based on tegrastats format
            # This is a generic parser - may need adjustment for different JetPack versions

            # RAM usage: RAM 1234/15518MB
            if 'RAM' in line:
                ram_parts = [p for p in line.split() if 'RAM' in p or 'MB' in p]
                if ram_parts:
                    ram_str = ram_parts[0].replace('RAM', '').replace('MB', '')
                    if '/' in ram_str:
                        used, total = map(int, ram_str.split('/'))
                        metrics['ram_used_mb'] = used
                        metrics['ram_total_mb'] = total
                        metrics['ram_usage_percent'] = (used / total) * 100 if total > 0 else 0

            # Power: Power 12345/12345mW
            if 'Power' in line:
                power_parts = [p for p in line.split() if 'Power' in p or 'mW' in p]
                if power_parts:
                    power_str = power_parts[0].replace('Power', '').replace('mW', '')
                    if '/' in power_str:
                        # Try to extract different power values
                        values = [int(v) for v in power_str.split('/') if v.isdigit()]
                        if values:
                            metrics['total_power_mw'] = values[0]
                            if len(values) > 1:
                                metrics['cpu_power_mw'] = values[0]
                                metrics['gpu_power_mw'] = values[1]

            # Temperature: CPU 12345C ... GPU 12345C ... AO 12345C ... thermal
            temp_parts = [p for p in line.split() if 'C' in p]
            for part in temp_parts:
                temp_str = part.replace('C', '')
                try:
                    temp_value = int(temp_str)
                    if 'CPU' in part:
                        metrics['cpu_temp_c'] = temp_value
                    elif 'GPU' in part:
                        metrics['gpu_temp_c'] = temp_value
                    elif 'thermal' in part or 'temp' in part:
                        metrics['soc_temp_c'] = temp_value
                except ValueError:
                    pass

            # Frequency: GPU 12345MHz ... CPU 12345MHz ... EMC 12345MHz
            freq_parts = [p for p in line.split() if 'MHz' in p]
            for part in freq_parts:
                freq_str = part.replace('MHz', '')
                try:
                    freq_value = int(freq_str)
                    if 'GPU' in part:
                        metrics['gpu_freq_mhz'] = freq_value
                    elif 'CPU' in part:
                        metrics['cpu_freq_mhz'] = freq_value
                    elif 'EMC' in part or 'emc' in part:
                        metrics['emc_freq_mhz'] = freq_value
                except ValueError:
                    pass

            # CPU usage: CPU 12% 23% ...
            cpu_usage_parts = [p for p in line.split() if '%' in p and 'CPU' in line.split(p)[0]]
            if cpu_usage_parts:
                try:
                    cpu_usage = int(cpu_usage_parts[0].replace('%', ''))
                    metrics['cpu_usage_percent'] = cpu_usage
                except ValueError:
                    pass

            return metrics if metrics else None

        except Exception as e:
            logger.warning(f"Error parsing tegrastats line: {e}")
            return None

    def get_metrics(self) -> List[Dict]:
        """
        Get all collected metrics.

        Returns:
            List of metric dictionaries
        """
        metrics_list = []

        while not self.data_queue.empty():
            try:
                metrics = self.data_queue.get_nowait()
                metrics_list.append(metrics)
            except queue.Empty:
                break

        self.metrics_data.extend(metrics_list)
        return metrics_list

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Get collected metrics as pandas DataFrame.

        Returns:
            DataFrame with all collected metrics
        """
        self.get_metrics()  # Update from queue

        if not self.metrics_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.metrics_data)

        # Ensure timestamp is the first column
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        return df

    def stop_collection(self) -> bool:
        """
        Stop metrics collection and cleanup.

        Returns:
            True if successful, False otherwise
        """
        if not self.is_collecting:
            logger.warning("Collection not in progress")
            return False

        try:
            logger.info("Stopping metrics collection...")

            # Send SIGTERM to tegrastats process
            if self.process:
                self.process.send_signal(signal.SIGTERM)
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("tegrastats did not terminate gracefully, killing...")
                    self.process.kill()
                    self.process.wait()

            # Wait for parsing thread to finish
            if self.parse_thread:
                self.parse_thread.join(timeout=5)

            self.is_collecting = False

            logger.info("Metrics collection stopped")
            logger.info(f"Collected {len(self.metrics_data)} data points")
            logger.info(f"Log file saved to: {self.log_file}")

            return True

        except Exception as e:
            logger.error(f"Error stopping metrics collection: {e}")
            return False

    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for collected metrics.

        Returns:
            Dictionary with summary statistics
        """
        df = self.get_metrics_dataframe()

        if df.empty:
            return {}

        summary = {}

        # Temperature statistics
        temp_columns = ['cpu_temp_c', 'gpu_temp_c', 'soc_temp_c']
        for col in temp_columns:
            if col in df.columns:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_max'] = df[col].max()
                summary[f'{col}_min'] = df[col].min()

        # Power statistics
        power_columns = ['total_power_mw', 'cpu_power_mw', 'gpu_power_mw']
        for col in power_columns:
            if col in df.columns:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_max'] = df[col].max()
                summary[f'{col}_min'] = df[col].min()

        # Frequency statistics
        freq_columns = ['gpu_freq_mhz', 'cpu_freq_mhz', 'emc_freq_mhz']
        for col in freq_columns:
            if col in df.columns:
                summary[f'{col}_mean'] = df[col].mean()
                summary[f'{col}_std'] = df[col].std()

        return summary

    def save_metrics(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Save collected metrics to file.

        Args:
            filename: Optional filename for saved metrics. If None, auto-generate.

        Returns:
            Path to saved file, or None if failed
        """
        df = self.get_metrics_dataframe()

        if df.empty:
            logger.warning("No metrics data to save")
            return None

        if filename is None:
            timestamp = int(time.time())
            filename = f"metrics_{timestamp}.parquet"

        output_path = self.output_dir / filename

        try:
            # Save as Parquet for efficient storage
            df.to_parquet(output_path)
            logger.info(f"Metrics saved to: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return None

    def calculate_energy_consumption(self, start_time: Optional[float] = None,
                                   end_time: Optional[float] = None) -> Dict:
        """
        Calculate energy consumption from collected metrics.

        Args:
            start_time: Start timestamp (if None, use first data point)
            end_time: End timestamp (if None, use last data point)

        Returns:
            Dictionary with energy consumption metrics
        """
        df = self.get_metrics_dataframe()

        if df.empty or 'total_power_mw' not in df.columns:
            return {}

        # Filter by time range
        if start_time is not None or end_time is not None:
            if start_time is not None and end_time is not None:
                df = df[(df.index >= start_time) & (df.index <= end_time)]
            elif start_time is not None:
                df = df[df.index >= start_time]
            elif end_time is not None:
                df = df[df.index <= end_time]

        if df.empty:
            return {}

        # Calculate duration
        if len(df) > 1:
            duration_sec = (df.index[-1] - df.index[0])
        else:
            duration_sec = self.interval_ms / 1000.0

        # Calculate total energy: Power (mW) * Time (s) / 1000 = Energy (mJ)
        total_power_mw = df['total_power_mw'].mean()
        energy_mj = total_power_mw * duration_sec
        energy_j = energy_mj / 1000.0  # Convert to Joules

        return {
            'duration_sec': duration_sec,
            'avg_power_mw': total_power_mw,
            'avg_power_w': total_power_mw / 1000.0,
            'total_energy_j': energy_j,
            'total_energy_mj': energy_mj
        }

    def __enter__(self):
        """Context manager entry."""
        self.start_collection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_collection()


def main():
    """
    Test function for metrics collector.
    """
    # Create collector with 1 second interval
    collector = MetricsCollector(interval_ms=1000)

    try:
        # Start collection for 10 seconds
        logger.info("Starting 10-second test collection...")

        with collector:
            time.sleep(10)

            # Get summary statistics
            summary = collector.get_summary_statistics()
            logger.info(f"Summary statistics: {summary}")

            # Calculate energy consumption
            energy = collector.calculate_energy_consumption()
            logger.info(f"Energy consumption: {energy}")

            # Save metrics
            saved_file = collector.save_metrics()
            if saved_file:
                logger.info(f"Metrics saved to: {saved_file}")

        logger.info("Test completed successfully")

    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    main()