#!/usr/bin/env python3
"""
System Monitoring Script for Jetson Orin
无需sudo权限的系统监控脚本，使用tegrastats和nvidia-smi
"""

import subprocess
import time
import json
import re
from pathlib import Path
from datetime import datetime
import threading

class JetsonSystemMonitor:
    """Monitor Jetson Orin system metrics without sudo"""

    def __init__(self, interval_ms=1000, output_dir="data/monitoring"):
        self.interval_ms = interval_ms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitoring = False
        self.monitor_thread = None
        self.metrics_history = []

    def parse_tegrastats_line(self, line):
        """Parse tegrastats output line for Jetson Orin"""

        # Jetson Orin tegrastats output example:
        # RAM 13123/62841MB (lfb 63x4MB) SWAP 9/31420MB (cached 0MB) CPU [3%@729,4%@729,1%@729,4%@729,1%@729,3%@729,2%@729,1%@729,4%@729,2%@729,1%@729,1%@729] GR3D_FREQ 0% cpu@48.156C soc2@43.562C soc0@45.125C tj@48.156C soc1@44.343C VDD_GPU_SOC 3193mW/3193mW VDD_CPU_CV 399mW/399mW VIN_SYS_5V0 4306mW/4306mW

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'ram_usage_mb': 0,
            'ram_total_mb': 0,
            'cpu_usage_percent': [],
            'cpu_freq_mhz': [],
            'gpu_freq_percent': 0,
            'temperature_c': 0,
            'power_gpu_mw': 0,
            'power_cpu_mw': 0,
            'power_total_mw': 0
        }

        try:
            # Parse RAM usage
            ram_match = re.search(r'RAM (\d+)/(\d+)MB', line)
            if ram_match:
                metrics['ram_usage_mb'] = int(ram_match.group(1))
                metrics['ram_total_mb'] = int(ram_match.group(2))

            # Parse CPU usage and frequency
            cpu_match = re.search(r'CPU \[(.*?)\]', line)
            if cpu_match:
                cpu_data = cpu_match.group(1).split(',')
                for core_data in cpu_data:
                    core_match = re.search(r'(\d+)%@(\d+)', core_data.strip())
                    if core_match:
                        metrics['cpu_usage_percent'].append(int(core_match.group(1)))
                        metrics['cpu_freq_mhz'].append(int(core_match.group(2)))

            # Parse GPU frequency (GR3D_FREQ)
            gpu_freq_match = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_freq_match:
                metrics['gpu_freq_percent'] = int(gpu_freq_match.group(1))

            # Parse temperature (look for tj@ which is junction temperature)
            temp_match = re.search(r'tj@(\d+\.\d+)C', line)
            if temp_match:
                metrics['temperature_c'] = float(temp_match.group(1))

            # Parse power consumption
            # VDD_GPU_SOC: GPU/SOC power
            gpu_power_match = re.search(r'VDD_GPU_SOC (\d+)mW/(\d+)mW', line)
            if gpu_power_match:
                metrics['power_gpu_mw'] = int(gpu_power_match.group(1))

            # VDD_CPU_CV: CPU power
            cpu_power_match = re.search(r'VDD_CPU_CV (\d+)mW/(\d+)mW', line)
            if cpu_power_match:
                metrics['power_cpu_mw'] = int(cpu_power_match.group(1))

            # VIN_SYS_5V0: Total system power
            total_power_match = re.search(r'VIN_SYS_5V0 (\d+)mW/(\d+)mW', line)
            if total_power_match:
                metrics['power_total_mw'] = int(total_power_match.group(1))

        except Exception as e:
            print(f"Error parsing tegrastats line: {e}")

        return metrics

    def get_gpu_info(self):
        """Get GPU information from nvidia-smi (limited support on Jetson Orin)"""

        gpu_info = {
            'name': 'Unknown',
            'temperature_c': 0,
            'power_usage_w': 0,
            'power_limit_w': 0,
            'memory_usage_mb': 0,
            'memory_total_mb': 0,
            'gpu_util_percent': 0,
            'available': False
        }

        try:
            # On Jetson Orin, nvidia-smi has limited support
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                gpu_info['name'] = result.stdout.strip()
                gpu_info['available'] = True

            # Note: Most other metrics are not available via nvidia-smi on Jetson Orin
            # They are available through tegrastats instead

        except Exception as e:
            print(f"Error getting GPU info: {e}")

        return gpu_info

    def monitor_once(self):
        """Get current system metrics"""

        metrics = {
            'timestamp': datetime.now().isoformat(),
            'tegrastats': None,
            'gpu': None
        }

        # Get tegrastats data (one sample)
        try:
            # Start tegrastats in background, get one line, then kill it
            tegrastats_proc = subprocess.Popen(
                ['tegrastats', '--interval', str(self.interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait a bit for data
            time.sleep(1.5)

            # Get output and kill process
            tegrastats_proc.terminate()
            try:
                stdout, stderr = tegrastats_proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                tegrastats_proc.kill()
                stdout, stderr = tegrastats_proc.communicate()

            if stdout:
                lines = stdout.strip().split('\n')
                # Use the last complete line with RAM and CPU data
                for line in reversed(lines):
                    if 'RAM' in line and 'CPU' in line and 'GR3D_FREQ' in line:
                        metrics['tegrastats'] = self.parse_tegrastats_line(line)
                        break

        except Exception as e:
            print(f"Error getting tegrastats data: {e}")

        # Get GPU info
        metrics['gpu'] = self.get_gpu_info()

        return metrics

    def start_monitoring(self, duration_seconds=60):
        """Start monitoring for a specific duration"""

        self.monitoring = True
        self.metrics_history = []
        start_time = time.time()

        print(f"Starting system monitoring for {duration_seconds} seconds...")
        print(f"Sampling interval: {self.interval_ms}ms")

        while self.monitoring and (time.time() - start_time) < duration_seconds:
            try:
                metrics = self.monitor_once()
                self.metrics_history.append(metrics)

                # Print current status
                if metrics['tegrastats']:
                    ts = metrics['tegrastats']
                    cpu_avg = sum(ts['cpu_usage_percent']) / len(ts['cpu_usage_percent']) if ts['cpu_usage_percent'] else 0
                    total_power_w = ts['power_total_mw'] / 1000.0 if ts['power_total_mw'] > 0 else 0
                    print(f"\r[{metrics['timestamp']}] {ts['temperature_c']:.1f}°C, {total_power_w:.1f}W | CPU: {cpu_avg:.1f}%, GPU: {ts['gpu_freq_percent']}%", end='')
                else:
                    print(f"\r[{metrics['timestamp']}] No data available", end='')

                time.sleep(self.interval_ms / 1000.0)

            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Error during monitoring: {e}")
                time.sleep(1)

        self.monitoring = False
        print(f"\nMonitoring completed. Collected {len(self.metrics_history)} samples.")

        return self.metrics_history

    def save_results(self, filename_suffix=""):
        """Save monitoring results to file"""

        if not self.metrics_history:
            print("No metrics to save")
            return None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"system_monitor_{timestamp}{filename_suffix}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump({
                'monitoring_start': self.metrics_history[0]['timestamp'] if self.metrics_history else None,
                'monitoring_end': self.metrics_history[-1]['timestamp'] if self.metrics_history else None,
                'total_samples': len(self.metrics_history),
                'interval_ms': self.interval_ms,
                'data': self.metrics_history
            }, f, indent=2)

        print(f"Monitoring results saved to: {filepath}")
        return filepath

    def generate_summary(self):
        """Generate summary statistics from collected metrics"""

        if not self.metrics_history:
            return {}

        summary = {
            'total_samples': len(self.metrics_history),
            'duration_seconds': 0,
            'temperatures': {
                'avg_c': 0,
                'max_c': 0,
                'min_c': 0
            },
            'power': {
                'avg_total_w': 0,
                'max_total_w': 0,
                'avg_gpu_w': 0,
                'avg_cpu_w': 0
            },
            'cpu': {
                'avg_usage_percent': 0,
                'max_usage_percent': 0,
                'avg_freq_mhz': 0
            },
            'gpu': {
                'avg_freq_percent': 0,
                'max_freq_percent': 0
            },
            'memory': {
                'avg_usage_mb': 0,
                'max_usage_mb': 0,
                'avg_total_mb': 0
            }
        }

        temps = []
        total_powers = []
        gpu_powers = []
        cpu_powers = []
        cpu_usages = []
        cpu_freqs = []
        gpu_freqs = []
        ram_usages = []
        ram_totals = []

        for metrics in self.metrics_history:
            if metrics['tegrastats']:
                ts = metrics['tegrastats']

                if ts['temperature_c'] > 0:
                    temps.append(ts['temperature_c'])

                if ts['power_total_mw'] > 0:
                    total_powers.append(ts['power_total_mw'] / 1000.0)  # Convert to watts

                if ts['power_gpu_mw'] > 0:
                    gpu_powers.append(ts['power_gpu_mw'] / 1000.0)  # Convert to watts

                if ts['power_cpu_mw'] > 0:
                    cpu_powers.append(ts['power_cpu_mw'] / 1000.0)  # Convert to watts

                if ts['cpu_usage_percent']:
                    cpu_usages.extend(ts['cpu_usage_percent'])
                    cpu_freqs.extend(ts['cpu_freq_mhz'])

                if ts['gpu_freq_percent'] >= 0:
                    gpu_freqs.append(ts['gpu_freq_percent'])

                if ts['ram_usage_mb'] > 0:
                    ram_usages.append(ts['ram_usage_mb'])

                if ts['ram_total_mb'] > 0:
                    ram_totals.append(ts['ram_total_mb'])

        # Calculate temperature statistics
        if temps:
            summary['temperatures']['avg_c'] = sum(temps) / len(temps)
            summary['temperatures']['max_c'] = max(temps)
            summary['temperatures']['min_c'] = min(temps)

        # Calculate power statistics
        if total_powers:
            summary['power']['avg_total_w'] = sum(total_powers) / len(total_powers)
            summary['power']['max_total_w'] = max(total_powers)

        if gpu_powers:
            summary['power']['avg_gpu_w'] = sum(gpu_powers) / len(gpu_powers)

        if cpu_powers:
            summary['power']['avg_cpu_w'] = sum(cpu_powers) / len(cpu_powers)

        # Calculate CPU statistics
        if cpu_usages:
            summary['cpu']['avg_usage_percent'] = sum(cpu_usages) / len(cpu_usages)
            summary['cpu']['max_usage_percent'] = max(cpu_usages)

        if cpu_freqs:
            summary['cpu']['avg_freq_mhz'] = sum(cpu_freqs) / len(cpu_freqs)

        # Calculate GPU statistics
        if gpu_freqs:
            summary['gpu']['avg_freq_percent'] = sum(gpu_freqs) / len(gpu_freqs)
            summary['gpu']['max_freq_percent'] = max(gpu_freqs)

        # Calculate memory statistics
        if ram_usages:
            summary['memory']['avg_usage_mb'] = sum(ram_usages) / len(ram_usages)
            summary['memory']['max_usage_mb'] = max(ram_usages)

        if ram_totals:
            summary['memory']['avg_total_mb'] = sum(ram_totals) / len(ram_totals)

        # Calculate duration
        if len(self.metrics_history) >= 2:
            start = datetime.fromisoformat(self.metrics_history[0]['timestamp'])
            end = datetime.fromisoformat(self.metrics_history[-1]['timestamp'])
            summary['duration_seconds'] = (end - start).total_seconds()

        return summary

def main():
    """Main function to run system monitoring"""

    print("=" * 60)
    print("Jetson Orin System Monitor")
    print("=" * 60)

    # Check if required tools are available
    tools_ok = True

    if subprocess.run(['which', 'tegrastats'], capture_output=True).returncode != 0:
        print("❌ tegrastats not found")
        tools_ok = False
    else:
        print("✅ tegrastats available")

    if subprocess.run(['which', 'nvidia-smi'], capture_output=True).returncode != 0:
        print("❌ nvidia-smi not found")
        tools_ok = False
    else:
        print("✅ nvidia-smi available")

    if not tools_ok:
        print("\nRequired tools not available. Exiting.")
        return 1

    # Create monitor
    monitor = JetsonSystemMonitor(interval_ms=1000)

    # Get initial system info
    print("\n📊 Current System Status:")
    initial_metrics = monitor.monitor_once()

    if initial_metrics['gpu']:
        gpu = initial_metrics['gpu']
        print(f"  GPU: {gpu['name']}")

    if initial_metrics['tegrastats']:
        ts = initial_metrics['tegrastats']
        cpu_avg = sum(ts['cpu_usage_percent']) / len(ts['cpu_usage_percent']) if ts['cpu_usage_percent'] else 0
        cpu_freq_avg = sum(ts['cpu_freq_mhz']) / len(ts['cpu_freq_mhz']) if ts['cpu_freq_mhz'] else 0
        total_power = ts['power_total_mw'] / 1000.0 if ts['power_total_mw'] > 0 else 0

        print(f"  Temperature: {ts['temperature_c']:.1f}°C")
        print(f"  Total Power: {total_power:.1f}W")
        print(f"  CPU Usage: {cpu_avg:.1f}% @ {cpu_freq_avg:.0f}MHz")
        print(f"  GPU Frequency: {ts['gpu_freq_percent']}%")
        print(f"  RAM Usage: {ts['ram_usage_mb']}MB / {ts['ram_total_mb']}MB")

    # Ask for monitoring duration
    print("\n⏱️  Starting 30-second monitoring session...")
    print("   (Press Ctrl+C to stop early)")

    try:
        # Monitor for 30 seconds
        history = monitor.start_monitoring(duration_seconds=30)

        # Save results
        filepath = monitor.save_results()

        # Generate and print summary
        summary = monitor.generate_summary()

        print("\n📈 Monitoring Summary:")
        print(f"  Duration: {summary['duration_seconds']:.1f}s")
        print(f"  Samples: {summary['total_samples']}")

        if summary['temperatures']['avg_c'] > 0:
            print(f"\n  Temperature:")
            print(f"    Avg: {summary['temperatures']['avg_c']:.1f}°C")
            print(f"    Range: {summary['temperatures']['min_c']:.1f}°C - {summary['temperatures']['max_c']:.1f}°C")

        if summary['power']['avg_total_w'] > 0:
            print(f"\n  Power:")
            print(f"    Avg Total: {summary['power']['avg_total_w']:.1f}W (max: {summary['power']['max_total_w']:.1f}W)")
            print(f"    Avg GPU: {summary['power']['avg_gpu_w']:.1f}W")
            print(f"    Avg CPU: {summary['power']['avg_cpu_w']:.1f}W")

        if summary['cpu']['avg_usage_percent'] > 0:
            print(f"\n  CPU:")
            print(f"    Avg Usage: {summary['cpu']['avg_usage_percent']:.1f}% (max: {summary['cpu']['max_usage_percent']:.1f}%)")
            print(f"    Avg Frequency: {summary['cpu']['avg_freq_mhz']:.0f}MHz")

        if summary['gpu']['avg_freq_percent'] >= 0:
            print(f"\n  GPU:")
            print(f"    Avg Frequency: {summary['gpu']['avg_freq_percent']:.1f}% (max: {summary['gpu']['max_freq_percent']:.1f}%)")

        if summary['memory']['avg_usage_mb'] > 0:
            print(f"\n  Memory:")
            print(f"    Avg Usage: {summary['memory']['avg_usage_mb']:.0f}MB (max: {summary['memory']['max_usage_mb']:.0f}MB)")
            print(f"    Total: {summary['memory']['avg_total_mb']:.0f}MB")

        # Save summary
        if filepath:
            summary_path = filepath.parent / (filepath.stem + '_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n  📁 Summary saved to: {summary_path}")

        print("\n✅ System monitoring completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\nMonitoring interrupted by user")
        if monitor.metrics_history:
            monitor.save_results('_interrupted')
        return 0
    except Exception as e:
        print(f"\n❌ Error during monitoring: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())