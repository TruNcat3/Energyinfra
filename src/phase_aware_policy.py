#!/usr/bin/env python3
"""
Phase-Aware DVFS Policy
实现离线 phase-aware 策略模拟，不做真实在线控制
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PhaseAwarePolicy:
    """Phase-Aware DVFS Policy for LLM Inference"""

    def __init__(self, selector_table_path: str = "data/rate_tables/selector_table.parquet",
                 config_path: str = "configs/selector.yaml"):
        self.selector_table_path = Path(selector_table_path)
        self.config_path = Path(config_path)

        # Load selector table
        try:
            self.selector_df = pd.read_parquet(self.selector_table_path)
            logger.info(f"Loaded selector table: {len(self.selector_df)} configurations")
        except Exception as e:
            logger.error(f"Failed to load selector table: {e}")
            self.selector_df = pd.DataFrame()

        # Load configuration
        self.config = self.load_config()

        # Default switching overhead (ms)
        self.default_switching_overhead = {
            'gpu': 50,
            'cpu': 20,
            'emc': 80,
            'phase_boundary': 30
        }

    def load_config(self) -> Dict:
        """Load selector configuration"""

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    import yaml
                    config = yaml.safe_load(f)
                logger.info(f"Loaded config from: {self.config_path}")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")

        # Default configuration
        return {
            'switching_overhead_ms': {
                'gpu_low_to_high': 50,
                'gpu_high_to_low': 45,
                'cpu_low_to_high': 20,
                'cpu_high_to_low': 15,
                'emc_low_to_high': 80,
                'emc_high_to_low': 75,
                'phase_boundary': 30
            },
            'break_even_tokens': 64,
            'slo_margin_threshold_ms': 100,
            'energy_saving_threshold_j': 0.01
        }

    def find_phase_config(self, workload: Dict, phase: str) -> Optional[Dict]:
        """Find optimal configuration for specific phase"""

        if self.selector_df.empty:
            return None

        # Filter by phase
        phase_workload = workload.copy()
        phase_workload['phase'] = phase

        # Find matching bucket
        bucket_key = self.create_bucket_key(phase_workload)
        phase_bucket = self.selector_df[
            (self.selector_df['bucket_key'] == bucket_key) &
            (self.selector_df['phase'] == phase)
        ]

        if phase_bucket.empty:
            # Try to find nearest bucket with same phase
            phase_bucket = self.selector_df[self.selector_df['phase'] == phase]
            if phase_bucket.empty:
                # Fallback to any bucket
                phase_bucket = self.selector_df

        # Select best config for this phase
        if not phase_bucket.empty:
            # For prefill, prioritize TTFT (latency)
            # For decode, prioritize energy efficiency
            if phase == 'prefill':
                # Minimize TTFT
                best_config = phase_bucket.loc[phase_bucket['ttft_ms_median'].idxmin()]
            else:
                # Use phase-appropriate energy column
                energy_col = self._get_energy_column(phase_bucket, phase)
                best_config = phase_bucket.loc[phase_bucket[energy_col].idxmin()]

            # Use phase-appropriate energy for predicted metrics
            energy_col = self._get_energy_column(phase_bucket, phase)

            return {
                'gpu_freq_mhz': int(best_config['gpu_freq_mhz']),
                'cpu_freq_mhz': int(best_config['cpu_freq_mhz']),
                'emc_freq_mhz': int(best_config['emc_freq_mhz']),
                'predicted_metrics': {
                    'ttft_ms': float(best_config['ttft_ms_median']),
                    'tpot_ms': float(best_config['tpot_ms_median']),
                    'energy_per_token_j': float(best_config[energy_col]),
                    'avg_power_w': float(best_config['avg_power_w_median'])
                }
            }

        return None

    def _get_energy_column(self, df: pd.DataFrame, phase: str) -> str:
        """Determine the phase-appropriate energy column"""
        phase_col_map = {
            'prefill': 'energy_per_input_token_j_median',
            'decode': 'energy_per_output_token_j_median',
            'mixed': 'energy_per_output_token_j_median'
        }
        target_col = phase_col_map.get(phase, 'energy_per_output_token_j_median')
        if target_col in df.columns:
            return target_col
        if 'energy_objective_used' in df.columns:
            obj_col = df['energy_objective_used'].iloc[0]
            if obj_col in df.columns:
                return obj_col
        return 'energy_per_token_j_median'

    def create_bucket_key(self, workload: Dict) -> str:
        """Create bucket key from workload"""

        model = workload.get('model', 'synthetic_qwen_7b_int4')
        runtime = workload.get('runtime', 'synthetic')
        batch_size = workload.get('batch_size', 1)
        prompt_length = workload.get('prompt_length', workload.get('prompt_len', 512))
        output_length = workload.get('output_length', workload.get('output_len', 128))
        phase = workload.get('phase', 'mixed')
        concurrency = workload.get('concurrency', 1)

        return f"{model}|{runtime}|{batch_size}|{prompt_length}|{output_length}|{phase}|{concurrency}"

    def estimate_switching_cost(self, prefill_config: Dict, decode_config: Dict) -> Dict:
        """Estimate frequency switching cost"""

        switching_cost = {
            'overhead_ms': 0,
            'energy_overhead_j': 0,
            'components': []
        }

        # GPU switching
        if prefill_config['gpu_freq_mhz'] != decode_config['gpu_freq_mhz']:
            if prefill_config['gpu_freq_mhz'] < decode_config['gpu_freq_mhz']:
                overhead = self.config['switching_overhead_ms'].get('gpu_low_to_high', 50)
            else:
                overhead = self.config['switching_overhead_ms'].get('gpu_high_to_low', 45)

            switching_cost['overhead_ms'] += overhead
            switching_cost['components'].append({
                'component': 'gpu',
                'from_freq': prefill_config['gpu_freq_mhz'],
                'to_freq': decode_config['gpu_freq_mhz'],
                'overhead_ms': overhead
            })

        # CPU switching
        if prefill_config['cpu_freq_mhz'] != decode_config['cpu_freq_mhz']:
            if prefill_config['cpu_freq_mhz'] < decode_config['cpu_freq_mhz']:
                overhead = self.config['switching_overhead_ms'].get('cpu_low_to_high', 20)
            else:
                overhead = self.config['switching_overhead_ms'].get('cpu_high_to_low', 15)

            switching_cost['overhead_ms'] += overhead
            switching_cost['components'].append({
                'component': 'cpu',
                'from_freq': prefill_config['cpu_freq_mhz'],
                'to_freq': decode_config['cpu_freq_mhz'],
                'overhead_ms': overhead
            })

        # EMC switching
        if prefill_config['emc_freq_mhz'] != decode_config['emc_freq_mhz']:
            if prefill_config['emc_freq_mhz'] < decode_config['emc_freq_mhz']:
                overhead = self.config['switching_overhead_ms'].get('emc_low_to_high', 80)
            else:
                overhead = self.config['switching_overhead_ms'].get('emc_high_to_low', 75)

            switching_cost['overhead_ms'] += overhead
            switching_cost['components'].append({
                'component': 'emc',
                'from_freq': prefill_config['emc_freq_mhz'],
                'to_freq': decode_config['emc_freq_mhz'],
                'overhead_ms': overhead
            })

        return switching_cost

    def calculate_energy_saving(self, prefill_config: Dict, decode_config: Dict,
                               single_config: Dict, output_length: int,
                               prompt_length: int = 512) -> float:
        """Calculate energy saving from phase-aware vs single config"""

        # Phase-aware energy
        phase_aware_energy = (
            prefill_config['predicted_metrics']['energy_per_token_j'] * prompt_length +
            decode_config['predicted_metrics']['energy_per_token_j'] * output_length
        )

        # Single config energy
        single_config_energy = (
            single_config['predicted_metrics']['energy_per_token_j'] * (prompt_length + output_length)
        )

        return single_config_energy - phase_aware_energy

    def should_enable_phase_aware(self, workload: Dict, slo: Dict) -> Dict:
        """Determine if phase-aware DVFS should be enabled"""

        # Find phase-specific configs
        prefill_config = self.find_phase_config(workload, 'prefill')
        decode_config = self.find_phase_config(workload, 'decode')
        single_config = self.find_phase_config(workload, 'mixed')

        if not all([prefill_config, decode_config, single_config]):
            return {
                'policy': 'single_config',
                'reason': 'insufficient_config_data',
                'switch_enabled': False
            }

        output_length = workload.get('output_length', workload.get('output_len', 128))
        prompt_length = workload.get('prompt_length', workload.get('prompt_len', 512))

        # Calculate switching cost
        switching_cost = self.estimate_switching_cost(prefill_config, decode_config)

        # Calculate energy saving
        energy_saving = self.calculate_energy_saving(
            prefill_config, decode_config, single_config, output_length, prompt_length
        )

        # Estimate switching energy overhead (simplified)
        avg_power = (prefill_config['predicted_metrics']['avg_power_w'] +
                    decode_config['predicted_metrics']['avg_power_w']) / 2
        switching_energy_overhead = (avg_power * switching_cost['overhead_ms']) / 1000

        # Net energy saving
        net_energy_saving = energy_saving - switching_energy_overhead

        # Check SLO margin
        slo_margin = slo.get('ttft_ms', 1000) - max(
            prefill_config['predicted_metrics']['ttft_ms'],
            decode_config['predicted_metrics']['ttft_ms']
        )

        # Decision logic
        break_even_tokens = self.config.get('break_even_tokens', 64)
        energy_threshold = self.config.get('energy_saving_threshold_j', 0.01)
        slo_margin_threshold = self.config.get('slo_margin_threshold_ms', 100)

        enable_phase_aware = (
            net_energy_saving > energy_threshold and
            switching_cost['overhead_ms'] < slo_margin and
            output_length > break_even_tokens
        )

        result = {
            'policy': 'phase_aware' if enable_phase_aware else 'single_config',
            'prefill_config': prefill_config,
            'decode_config': decode_config,
            'single_config': single_config,
            'switch_enabled': enable_phase_aware,
            'switching_cost': switching_cost,
            'energy_saving': net_energy_saving,
            'break_even_tokens': break_even_tokens,
            'selection_reason': ''
        }

        if enable_phase_aware:
            result['selection_reason'] = 'phase_aware_energy_saving'
        else:
            if output_length <= break_even_tokens:
                result['selection_reason'] = 'output_too_short'
            elif net_energy_saving <= energy_threshold:
                result['selection_reason'] = 'insufficient_energy_saving'
            elif switching_cost['overhead_ms'] >= slo_margin:
                result['selection_reason'] = 'switching_cost_too_high'
            else:
                result['selection_reason'] = 'single_config_preferred'

        return result

    def select_phase_aware_config(self, workload: Dict, slo: Dict) -> Dict:
        """Main phase-aware configuration selection"""

        # Determine if phase-aware should be enabled
        decision = self.should_enable_phase_aware(workload, slo)

        if decision['switch_enabled']:
            return {
                'policy': 'phase_aware',
                'prefill_config': {
                    'gpu_freq_mhz': decision['prefill_config']['gpu_freq_mhz'],
                    'cpu_freq_mhz': decision['prefill_config']['cpu_freq_mhz'],
                    'emc_freq_mhz': decision['prefill_config']['emc_freq_mhz']
                },
                'decode_config': {
                    'gpu_freq_mhz': decision['decode_config']['gpu_freq_mhz'],
                    'cpu_freq_mhz': decision['decode_config']['cpu_freq_mhz'],
                    'emc_freq_mhz': decision['decode_config']['emc_freq_mhz']
                },
                'switch_enabled': True,
                'switching_cost_ms': decision['switching_cost']['overhead_ms'],
                'expected_energy_saving_j': decision['energy_saving'],
                'break_even_tokens': decision['break_even_tokens'],
                'selection_reason': decision['selection_reason']
            }
        else:
            # Return single config
            return {
                'policy': 'single_config',
                'config': {
                    'gpu_freq_mhz': decision['single_config']['gpu_freq_mhz'],
                    'cpu_freq_mhz': decision['single_config']['cpu_freq_mhz'],
                    'emc_freq_mhz': decision['single_config']['emc_freq_mhz']
                },
                'switch_enabled': False,
                'selection_reason': decision['selection_reason'],
                'predicted_metrics': decision['single_config']['predicted_metrics']
            }


def main():
    """Main function for testing phase-aware policy"""

    # Example usage
    policy = PhaseAwarePolicy()

    # Example workload and SLO
    workload = {
        "model": "synthetic_qwen_7b_int4",
        "runtime": "synthetic",
        "batch_size": 1,
        "prompt_length": 512,
        "output_length": 128,
        "phase": "mixed",
        "concurrency": 1
    }

    slo = {
        "ttft_ms": 1000,
        "tpot_ms": 80,
        "max_power_w": 40,
        "max_temp_c": 80
    }

    # Select phase-aware configuration
    result = policy.select_phase_aware_config(workload, slo)

    print("Phase-Aware Configuration Selection Result:")
    print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())