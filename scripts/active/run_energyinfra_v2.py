#!/usr/bin/env python3
"""Isolated token-accurate, SLO-constrained EnergyInfra extension.

Legacy experiments are intentionally not rewritten. `fine` matches the later
GPU-only CAP matrix shape, not the original measurement or selector algorithm.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import csv
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SCHEMA = "energyinfra-token-v2"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def percentile(values, q):
    a = sorted(values)
    x = (len(a) - 1) * q
    lo = math.floor(x)
    hi = math.ceil(x)
    return a[lo] + (a[hi] - a[lo]) * (x - lo)


def wilson(k, n):
    if not n:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = k / n
    mid = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [mid-half, mid+half]


def config_id(c):
    return c["mode"] if c["mode"] != "cap" else f"g{c['gpu_hz']}_e{c['emc_hz']}_c{c['cpu_khz']}"


def matrix(cfg, cohort):
    p = cfg["platform"]
    emc = p["emc_hz"] if cohort == "full" else [max(p["emc_hz"])]
    cpu = p["cpu_khz"] if cohort == "full" else [1458000]
    caps = [dict(mode="cap", gpu_hz=g, emc_hz=e, cpu_khz=c)
            for g, e, c in itertools.product(p["gpu_hz"], emc, cpu)]
    return caps + [dict(mode="dynamic"), dict(mode="max_clock")]


class Hardware:
    def __init__(self, cfg):
        from src.controller.jetson_sysfs import JetsonPaths
        self.paths = JetsonPaths.discover()
        self.p = cfg["platform"]
        if self.p["emc_control"] != "cap_only":
            raise RuntimeError("v2 hardware adapter currently requires cap_only EMC")
        if self.p["model_contains"].lower() not in self.paths.platform_model.lower():
            raise RuntimeError("Platform model mismatch")
        if not platform.release().startswith(self.p["kernel_prefix"]):
            raise RuntimeError("Kernel profile mismatch; create a version-specific config")
        if self.paths.platform_config["os_version"] != self.p["os_version"]:
            raise RuntimeError("OS version profile mismatch")
        self.gpu = self.paths.gpu_devfreq
        self.cpus = self.paths.cpu_policies
        self.emc = self.paths.emc_cap
        if not self.gpu or not self.cpus or not self.emc:
            raise RuntimeError("Required control paths missing")
        if not set(self.p["gpu_hz"]) <= set(self.paths.gpu_available_hz()):
            raise RuntimeError("Unsupported GPU grid")
        if not set(self.p["cpu_khz"]) <= set(self.paths.cpu_available_hz()):
            raise RuntimeError("Unsupported CPU grid")
        for node, attr, requested in [(self.gpu, "available_governors", [self.p['gpu_dynamic_governor'], self.p['gpu_lock_governor']])]:
            if not set(requested) <= set((node / attr).read_text().split()):
                raise RuntimeError("Unsupported GPU governors")
        for cpu in self.cpus:
            if not {self.p['cpu_dynamic_governor'], self.p['cpu_lock_governor']} <= set((cpu / 'scaling_available_governors').read_text().split()):
                raise RuntimeError("Unsupported CPU governors")

    def state(self):
        return {"gpu": self.read_domain(self.gpu, False),
                "cpus": {str(c): self.read_domain(c, True) for c in self.cpus},
                "emc_hz": int(self.emc.read_text())}

    @staticmethod
    def read_domain(path, cpu):
        prefix = "scaling_" if cpu else ""
        return {"min": int((path / (prefix+'min_freq')).read_text()),
                "max": int((path / (prefix+'max_freq')).read_text()),
                "governor": (path / (prefix+'governor')).read_text().strip()}

    @staticmethod
    def domain(path, minimum, maximum, governor, cpu=False):
        prefix = "scaling_" if cpu else ""
        lo, hi = path / (prefix+'min_freq'), path / (prefix+'max_freq')
        # Expand before contracting, so transitions cannot invert the range.
        hi.write_text(str(max(int(hi.read_text()), maximum, minimum)))
        lo.write_text(str(minimum))
        hi.write_text(str(maximum))
        (path / (prefix+'governor')).write_text(governor)
        observed = Hardware.read_domain(path, cpu)
        if observed != dict(min=minimum, max=maximum, governor=governor):
            raise RuntimeError(f"Configured boundary mismatch: {path}: {observed}")

    def apply(self, c):
        g = self.paths.gpu_available_hz()
        cpu = self.paths.cpu_available_hz()
        mode = c['mode']
        gmax = c.get('gpu_hz', max(g))
        cmax = c.get('cpu_khz', max(cpu))
        locked = mode == 'max_clock'
        self.domain(self.gpu, gmax if locked else min(g), gmax,
                    self.p['gpu_lock_governor' if locked else 'gpu_dynamic_governor'])
        for path in self.cpus:
            self.domain(path, cmax if locked else min(cpu), cmax,
                        self.p['cpu_lock_governor' if locked else 'cpu_dynamic_governor'], True)
        e = c.get('emc_hz', max(self.p['emc_hz']))
        self.emc.write_text(str(e))
        if int(self.emc.read_text()) != e:
            raise RuntimeError("EMC cap readback mismatch")
        return self.state()

    def restore(self, snapshot):
        errors = []
        for path, state, cpu in [(self.gpu, snapshot['gpu'], False)] + [(Path(p), v, True) for p, v in snapshot['cpus'].items()]:
            try:
                self.domain(path, state['min'], state['max'], state['governor'], cpu)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        try:
            self.emc.write_text(str(snapshot['emc_hz']))
            if int(self.emc.read_text()) != snapshot['emc_hz']:
                raise RuntimeError("EMC restore readback mismatch")
        except Exception as exc:
            errors.append(f"EMC: {exc}")
        if errors:
            raise RuntimeError('; '.join(errors))


class Power:
    """Receive-timestamped tegrastats; raw lines retained for clock/thermal audit."""
    def __init__(self, directory, cfg):
        self.cfg = cfg['measurement']
        self.max_sample_gap_s = float(
            os.environ.get('ENERGYINFRA_MAX_SAMPLE_GAP_S', self.cfg['max_sample_gap_s'])
        )
        self.rail = cfg['platform']['power_rail']
        self.samples = []
        self.error = None
        self.lock = threading.Lock()
        self.file = (directory / f"telemetry_{uuid.uuid4().hex}.jsonl").open('w')
        self.proc = subprocess.Popen(['stdbuf', '-oL', 'tegrastats', '--interval', str(self.cfg['interval_ms'])],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        try:
            for line in self.proc.stdout:
                now = time.monotonic()
                match = re.search(r'(?<!\S)' + re.escape(self.rail) + r'\s+(\d+)mW/', line)
                value = int(match.group(1))/1000 if match else None
                self.file.write(json.dumps(dict(t=now, power_w=value, raw=line.rstrip()))+'\n')
                self.file.flush()
                if value is not None:
                    with self.lock:
                        self.samples.append((now, value))
        except Exception as exc:
            self.error = str(exc)

    def wait_after(self, t):
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            with self.lock:
                if self.samples and self.samples[-1][0] >= t:
                    return
            if self.error or self.proc.poll() is not None:
                raise RuntimeError(f"Telemetry stopped: {self.error}")
            time.sleep(0.02)
        raise RuntimeError(f"No bracketing {self.rail} power sample within 10s")

    def energy(self, start, end, minimum_samples=2):
        self.wait_after(end)
        with self.lock:
            samples = list(self.samples)
        ts = [s[0] for s in samples]
        left = bisect.bisect_right(ts, start)-1
        right = bisect.bisect_left(ts, end)
        if left < 0 or right >= len(ts) or end <= start:
            raise RuntimeError("Energy window is not bracketed")
        data = samples[left:right+1]
        inside = sum(start <= t <= end for t, _ in data)
        if inside < minimum_samples:
            raise RuntimeError("Insufficient samples in measurement window")
        gaps = [b[0]-a[0] for a, b in zip(data, data[1:])]
        observed_max_gap_s = max(gaps, default=0.0)
        gap_count = sum(gap > self.cfg['max_sample_gap_s'] for gap in gaps)
        if observed_max_gap_s > self.max_sample_gap_s:
            raise RuntimeError("Telemetry sample gap exceeds protocol limit")
        energy = 0.0
        for (ta, pa), (tb, pb) in zip(data, data[1:]):
            a, b = max(ta, start), min(tb, end)
            if b <= a:
                continue
            va = pa + (pb-pa)*(a-ta)/(tb-ta)
            vb = pa + (pb-pa)*(b-ta)/(tb-ta)
            energy += (va+vb)*0.5*(b-a)
        return dict(total_energy_j=energy, avg_power_w=energy/(end-start), power_samples=inside,
                    power_timestamp_source='host_monotonic_line_receipt',
                    telemetry_max_sample_gap_s=self.max_sample_gap_s,
                    telemetry_observed_max_gap_s=observed_max_gap_s,
                    telemetry_gap_count=gap_count)

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.thread.join(timeout=5)
        self.file.close()


class Tokens:
    def __init__(self, cfg):
        from llama_cpp import Llama
        model = dict(cfg['model'])
        path = ROOT / model.pop('path')
        self.model = Llama(model_path=str(path), seed=cfg['seed'], verbose=True, **model)
        self.ctx = model['n_ctx']
        # Low-level generate does not stop at EOG. This is deliberately a fixed
        # number of model token steps, including any special tokens, not chat.
        self.base = self.model.tokenize(b'The following is a discussion about technology and science. ', add_bos=False)
        if not self.base:
            raise RuntimeError("Empty tokenizer prompt seed")

    def run(self, pl, ol):
        if pl < 1 or ol < 2 or pl+ol > self.ctx:
            raise ValueError("Invalid exact token lengths or context overflow")
        prompt = (self.base * math.ceil(pl/len(self.base)))[:pl]
        self.model.reset()  # n_tokens=0 prevents generate's prefix-cache reuse.
        start = time.monotonic()
        generator = self.model.generate(prompt, temp=0.0, top_p=1.0, repeat_penalty=1.0, reset=True)
        tokens, times = [], []
        try:
            for _ in range(ol):
                tokens.append(int(next(generator)))
                times.append(time.monotonic())
        finally:
            generator.close()
        end = time.monotonic()
        return dict(prompt_tokens=len(prompt), output_tokens=len(tokens), token_ids=tokens,
                    prompt_sha256=digest(prompt), token_times=times, start=start, end=end,
                    ttft_ms=(times[0]-start)*1000,
                    tpot_ms=(times[-1]-times[0])*1000/(len(times)-1),
                    itl_ms=[(b-a)*1000 for a, b in zip(times, times[1:])],
                    prefill_and_first_sample_ms=(times[0]-start)*1000,
                    total_time_ms=(end-start)*1000,
                    generation_policy='fixed_model_token_steps_including_special_tokens',
                    cache_policy='reset_before_every_request')


def measure(engine, power, hw, cfg, workload):
    pl, ol = workload
    for _ in range(cfg['measurement']['warmup_requests']):
        engine.run(pl, ol)
    power.wait_after(time.monotonic())
    requests = []
    start = time.monotonic()
    while not requests or time.monotonic()-start < cfg['measurement']['min_block_s']:
        requests.append(engine.run(pl, ol))
    end = time.monotonic()
    energy = power.energy(start, end, cfg['measurement']['min_samples'])
    total_tokens = sum(r['output_tokens'] for r in requests)
    return dict(valid=True, workload=list(workload), requests=requests,
                start=start, end=end, measured_duration_s=end-start,
                output_tokens=total_tokens,
                ttft_ms=statistics.median(r['ttft_ms'] for r in requests),
                tpot_ms=statistics.median(r['tpot_ms'] for r in requests),
                energy_per_token_j=energy['total_energy_j']/total_tokens,
                state_after=hw.state(), **energy)


def rows(directory, stage):
    return [json.loads(p.read_text()) for p in sorted((directory/stage).glob('*.json'))]


def passed(request, cfg, slo, scenario, queue_ms=0):
    return request['tpot_ms'] <= slo and (scenario == 'tpot_only' or request['ttft_ms']+queue_ms <= cfg['interactive_ttft_ms'])


def summarize(records, cfg, slo, scenario):
    requests = [r for row in records for r in row['requests']]
    energy = sum(r['total_energy_j'] for r in records)
    duration = sum(r['measured_duration_s'] for r in records)
    k = sum(passed(r, cfg, slo, scenario) for r in requests)
    return dict(n_requests=len(requests), n_blocks=len(records), slo_pass=k, attainment=k/len(requests),
                attainment_ci95=wilson(k, len(requests)),
                ttft_ms=statistics.median(r['ttft_ms'] for r in requests),
                tpot_ms=statistics.median(r['tpot_ms'] for r in requests),
                tpot_p95_ms=percentile([r['tpot_ms'] for r in requests], .95),
                avg_power_w=energy/duration, total_energy_j=energy,
                energy_per_token_j=energy/sum(r['output_tokens'] for r in records))


def select(directory, cfg, cohort):
    source = rows(directory, 'profile_'+cohort)
    wl = cfg['workloads'] if cohort == 'full' else cfg['fine_workloads']
    expected = len(matrix(cfg, cohort))*len(wl)*cfg['profile_repeats']
    if len(source) != expected or not all(r.get('valid') for r in source):
        raise RuntimeError(f"Incomplete profile: {len(source)}/{expected}; all points must be valid")
    groups = defaultdict(list)
    for row in source:
        groups[(tuple(row['workload']), config_id(row['hardware']))].append(row)
    table = []
    for (workload, cid), group in groups.items():
        metrics = summarize(group, cfg, cfg['tpot_slos_ms'][0], 'tpot_only')
        table.append(dict(workload=list(workload), hardware=group[0]['hardware'], config_id=cid, **metrics))
    selections, frontier = [], []
    objectives = ['avg_power_w', 'energy_per_token_j', 'tpot_ms']
    for workload in wl:
        candidates = [r for r in table if r['workload'] == workload and r['hardware']['mode'] == 'cap']
        for row in candidates:
            dominated = any(all(other[k] <= row[k] for k in objectives) and any(other[k] < row[k] for k in objectives) for other in candidates)
            if not dominated:
                frontier.append(row)
        for slo, scenario in itertools.product(cfg['tpot_slos_ms'], ['tpot_only', 'interactive_1s']):
            feasible = [r for r in candidates if r['tpot_ms'] <= slo and (scenario == 'tpot_only' or r['ttft_ms'] <= cfg['interactive_ttft_ms'])]
            for name, metric in [('min_power', 'avg_power_w'), ('min_energy', 'energy_per_token_j')]:
                best = min(feasible, key=lambda r: (r[metric], r['tpot_ms'], r['config_id'])) if feasible else None
                selections.append(dict(workload=workload, slo_ms=slo, scenario=scenario, strategy=name,
                                       predicted_feasible=best is not None, hardware=best['hardware'] if best else None,
                                       prediction=best, feasibility_rule='profile_median_not_online_guarantee'))
    atomic(directory/f'selection_{cohort}.json', dict(schema=SCHEMA, cohort=cohort, profile_digest=digest(source),
                                                   table=table, frontier=frontier, selections=selections))
    return selections


def jobs(cfg, stage, cohort, directory):
    if stage == 'calibrate':
        configs = matrix(cfg, 'full')
        configs = [configs[-2], configs[-1], configs[0], configs[-3]]
        work = cfg['calibration_workloads']
        repeats = 2
    elif stage == 'profile':
        configs = matrix(cfg, cohort)
        work = cfg['workloads'] if cohort == 'full' else cfg['fine_workloads']
        repeats = cfg['profile_repeats']
    else:
        selection = json.loads((directory/f'selection_{cohort}.json').read_text())['selections']
        pairs = {(tuple(s['workload']), config_id(s['hardware'])): s['hardware'] for s in selection if s['hardware']}
        work = cfg['workloads'] if cohort == 'full' else cfg['fine_workloads']
        for w in work:
            for mode in ['dynamic', 'max_clock']:
                pairs[(tuple(w), mode)] = dict(mode=mode)
        result = [dict(workload=list(w), hardware=c, repeat=r) for (w, _), c in sorted(pairs.items()) for r in range(cfg['validation_repeats'])]
        random.Random(cfg['seed']+1000).shuffle(result)
        return result
    result = [dict(workload=w, hardware=c, repeat=r) for w, c, r in itertools.product(work, configs, range(repeats))]
    random.Random(cfg['seed']+(0 if stage == 'calibrate' else 100)).shuffle(result)
    return result


def report(directory, cfg, cohort):
    selections = json.loads((directory/f'selection_{cohort}.json').read_text())['selections']
    records = rows(directory, 'validate_'+cohort)
    grouped = defaultdict(list)
    for r in records:
        if r.get('valid'):
            grouped[(tuple(r['workload']), config_id(r['hardware']))].append(r)
    result = []
    for s in selections:
        w = tuple(s['workload'])
        baseline = {}
        for mode in ['dynamic', 'max_clock']:
            group = grouped.get((w, mode), [])
            if len(group) >= cfg['validation_repeats']:
                baseline[mode] = summarize(group, cfg, s['slo_ms'], s['scenario'])
        for name, hardware in [(s['strategy'], s['hardware']), ('dynamic', dict(mode='dynamic')), ('max_clock', dict(mode='max_clock'))]:
            item = dict(workload=list(w), slo_ms=s['slo_ms'], scenario=s['scenario'], strategy=name, hardware=hardware)
            group = grouped.get((w, config_id(hardware)), []) if hardware else []
            if len(group) < cfg['validation_repeats']:
                item.update(status='infeasible' if hardware is None else 'not_validated')
            else:
                item.update(summarize(group, cfg, s['slo_ms'], s['scenario']))
                item['status'] = 'pass' if item['attainment'] >= cfg['required_attainment'] else 'fail'
                for mode, base in baseline.items():
                    for metric in ['avg_power_w', 'energy_per_token_j', 'ttft_ms', 'tpot_ms']:
                        item[f'{metric}_delta_vs_{mode}_pct'] = (item[metric]/base[metric]-1)*100 if base[metric] else None
            if item not in result:
                result.append(item)
    atomic(directory/f'comparison_{cohort}.json', result)
    keys = list(dict.fromkeys(k for row in result for k in row))
    with (directory/f'comparison_{cohort}.csv').open('w') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(result)
    return result


def serve(engine, power, hw, cfg, directory, cohort, args):
    # Import the existing trace generator rather than inventing arrival patterns.
    from src.experiments.run_serving_benchmark import RequestGenerator
    comparison = report(directory, cfg, cohort)
    selection = json.loads((directory/f'selection_{cohort}.json').read_text())['selections']
    settings = cfg['serving']
    candidates = {tuple(s['workload']): s['hardware'] for s in selection
                  if s['slo_ms'] == args.slo and s['scenario'] == args.scenario and s['strategy'] == args.strategy}
    runs = list(range(settings['repeats']))
    for repeat in runs:
        trace = RequestGenerator(args.trace, settings['duration_s'], cfg['seed']+repeat).generate_requests()
        trace_path = directory/f'trace_{args.trace}_{repeat}.json'
        if trace_path.exists() and json.loads(trace_path.read_text()) != trace:
            raise RuntimeError("Existing trace differs")
        atomic(trace_path, trace)
        if args.strategy not in ['dynamic', 'max_clock']:
            for req in trace:
                w = (req['prompt_length'], req['output_length'])
                acceptable = any(r['workload'] == list(w) and r['slo_ms'] == args.slo and r['scenario'] == args.scenario and r['strategy'] == args.strategy and r['status'] == 'pass' for r in comparison)
                if not candidates.get(w) or not acceptable:
                    raise RuntimeError(f"Serving blocked: no validated feasible {args.strategy} for {w}")
        key = f'{args.trace}_{args.strategy}_{args.slo}_{args.scenario}_{repeat}'
        output = directory/'serving'/f'{key}.json'
        output.parent.mkdir(exist_ok=True)
        if output.exists():
            continue
        hw.apply(dict(mode='dynamic'))
        engine.run(64, 64)
        power.wait_after(time.monotonic())
        start = time.monotonic()
        index, queue, results = 0, [], []
        previous = None
        while index < len(trace) or queue:
            now = time.monotonic()-start
            while index < len(trace) and trace[index]['scheduled_time'] <= now:
                req = dict(trace[index], request_id=index)
                if len(queue) >= settings['queue_limit']:
                    results.append(dict(request=req, dropped=True))
                else:
                    queue.append(req)
                index += 1
            if not queue:
                if index < len(trace):
                    time.sleep(min(.05, max(0, trace[index]['scheduled_time']-now)))
                continue
            req = queue.pop(0)
            w = (req['prompt_length'], req['output_length'])
            c = dict(mode=args.strategy) if args.strategy in ['dynamic', 'max_clock'] else candidates[w]
            if c != previous:
                hw.apply(c)
                previous = c
            result = engine.run(*w)
            result['queue_ms'] = (result['start']-start-req['scheduled_time'])*1000
            result['slo_pass'] = passed(result, cfg, args.slo, args.scenario, result['queue_ms'])
            results.append(dict(request=req, dropped=False, hardware=c, metrics=result))
        # Fixed arrival horizon includes idle time, followed by complete drain.
        end = max(time.monotonic(), start+settings['duration_s'])
        while time.monotonic() < end:
            time.sleep(min(.1, end-time.monotonic()))
        complete = [r['metrics'] for r in results if not r['dropped']]
        energy = power.energy(start, end, cfg['measurement']['min_samples'])
        horizon = power.energy(start, start+settings['duration_s'], cfg['measurement']['min_samples'])
        k = sum(r['slo_pass'] for r in complete)
        atomic(output, dict(schema=SCHEMA, strategy=args.strategy, trace=args.trace, repeat=repeat,
                            slo_ms=args.slo, scenario=args.scenario, results=results,
                            offered=len(trace), completed=len(complete), dropped=len(trace)-len(complete),
                            attainment=k/len(trace), attainment_ci95=wilson(k,len(trace)),
                            meets_attainment=k/len(trace)>=cfg['required_attainment'],
                            ttft_with_queue_p95_ms=percentile([r['ttft_ms']+r['queue_ms'] for r in complete], .95),
                            tpot_p95_ms=percentile([r['tpot_ms'] for r in complete], .95),
                            duration_with_drain_s=end-start, fixed_horizon=horizon,
                            energy_per_token_j=energy['total_energy_j']/sum(r['output_tokens'] for r in complete), **energy))
        print(json.dumps(dict(serving_complete=key)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan','calibrate','profile','select','validate','report','serve','restore'])
    parser.add_argument('--config', type=Path, default=ROOT/'configs/experiments/thor_protocol_v2.json')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--cohort', choices=['fine','full'], default='full')
    parser.add_argument('--strategy', choices=['dynamic','max_clock','min_power','min_energy'], default='min_energy')
    parser.add_argument('--trace', choices=['short_chat','long_generation','bursty_mixed'], default='short_chat')
    parser.add_argument('--slo', type=int, choices=[50,100,250], default=50)
    parser.add_argument('--scenario', choices=['tpot_only','interactive_1s'], default='tpot_only')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if cfg['schema'] != SCHEMA:
        raise RuntimeError('Unsupported protocol schema')
    directory = args.run_dir.resolve()
    if args.stage == 'plan':
        print(json.dumps({c: dict(hardware_configs=len(matrix(cfg,c))-2,
                                  workloads=len(cfg['workloads'] if c=='full' else cfg['fine_workloads']),
                                  blocks_including_baselines=len(jobs(cfg,'profile',c,directory))) for c in ['fine','full']}, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'campaign.lock').open('a') as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_hash = file_hash(__file__)
        identity = dict(config=cfg, implementation_sha256=source_hash,
                        trace_source_sha256=file_hash(ROOT/'src/experiments/run_serving_benchmark.py'))
        protocol_path = directory/'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != identity:
            raise RuntimeError('Protocol/source changed; use a new run directory')
        atomic(protocol_path, identity)
        if args.stage == 'select':
            select(directory, cfg, args.cohort)
            return
        if args.stage == 'report':
            report(directory, cfg, args.cohort)
            return
        if os.geteuid() != 0:
            raise RuntimeError('Hardware stages require foreground sudo or the nohup launcher')
        with open('/run/energyinfra-v2-hardware.lock', 'a') as hardware_lock:
            fcntl.flock(hardware_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            hw = Hardware(cfg)
            snapshot_path = directory/'hardware_snapshot.json'
            pending = directory/'restore_pending.json'
            if args.stage == 'restore':
                hw.restore(json.loads(snapshot_path.read_text()))
                atomic(pending, dict(pending=False))
                return
            if pending.exists() and json.loads(pending.read_text())['pending']:
                raise RuntimeError('Previous restore incomplete: run restore stage first')
            if args.stage != 'calibrate':
                gate = directory/'calibration.json'
                if not gate.exists() or not json.loads(gate.read_text()).get('passed'):
                    raise RuntimeError('Calibration gate not passed')
            model_path = ROOT/cfg['model']['path']
            environment = dict(model_sha256=file_hash(model_path), kernel=platform.release(),
                               llama_cpp_python=importlib.metadata.version('llama-cpp-python'),
                               platform=hw.paths.describe(),
                               l4t=Path('/etc/nv_tegra_release').read_text() if Path('/etc/nv_tegra_release').exists() else None)
            env_path = directory/'environment.json'
            if env_path.exists() and json.loads(env_path.read_text()) != environment:
                raise RuntimeError('Model/runtime/system changed; use a new run directory')
            atomic(env_path, environment)
            snapshot = hw.state()
            atomic(snapshot_path, snapshot)
            atomic(pending, dict(pending=True))
            def interrupted(signum, frame):
                raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM, interrupted)
            signal.signal(signal.SIGINT, interrupted)
            power = None
            try:
                engine = Tokens(cfg)
                power = Power(directory, cfg)
                power.wait_after(time.monotonic())
                if args.stage == 'serve':
                    serve(engine, power, hw, cfg, directory, args.cohort, args)
                    return
                stage = args.stage if args.stage == 'calibrate' else args.stage+'_'+args.cohort
                target = directory/stage
                target.mkdir(exist_ok=True)
                tasks = jobs(cfg, args.stage, args.cohort, directory)
                atomic(directory/f'{stage}_manifest.json', tasks)
                for index, task in enumerate(tasks):
                    output = target/(digest(task)+'.json')
                    if output.exists():
                        continue
                    control = hw.apply(task['hardware'])
                    time.sleep(cfg['measurement']['settle_s'])
                    result = measure(engine, power, hw, cfg, task['workload'])
                    atomic(output, dict(schema=SCHEMA, task=task, hardware=task['hardware'], configured=control, **result))
                    atomic(directory/'progress.json', dict(stage=stage, completed_index=index+1, total=len(tasks), last_task=task))
                    print(json.dumps(dict(stage=stage, completed=index+1, total=len(tasks), task=task,
                                          power_w=result['avg_power_w'], j_per_token=result['energy_per_token_j'])), flush=True)
                if args.stage == 'calibrate':
                    atomic(directory/'calibration.json', dict(passed=True, blocks=len(tasks),
                           checks=['exact_model_token_steps','energy_bracketing','minimum_samples','configured_boundaries'],
                           limitations=['VIN not externally calibrated','host receipt timestamps','no production text-quality validation']))
                if args.stage == 'validate':
                    report(directory, cfg, args.cohort)
            except BaseException as exc:
                atomic(directory/'last_failure.json', dict(stage=args.stage, error=repr(exc), time=time.time()))
                raise
            finally:
                try:
                    if power:
                        power.close()
                finally:
                    hw.restore(snapshot)
                    atomic(pending, dict(pending=False))


if __name__ == '__main__':
    main()
