#!/usr/bin/env python3
"""CPU-floor extension: characterize, select, validate, then serve.

The joint hardware-vector Pareto table is explicitly an extension, not the
original GPU-only selector or thermal controller. Existing code/data stay fixed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import statistics
import time

import run_energyinfra_v2 as v2
import run_phase_diagnostic_v1 as phase
import run_phase_validation_v1 as validation


DEFAULT_CONFIG = v2.ROOT/'configs/experiments/thor_cpu_floor_v1.json'


def emit(**values):
    print(json.dumps(values), flush=True)


def points(settings):
    grid = settings['grid']
    result = dict(dynamic=dict(mode='dynamic'), max_clock=dict(mode='max_clock'))
    for g in grid['gpu_hz']:
        for e in grid['emc_hz']:
            for floor in grid['cpu_floor_khz']:
                name = f'cap_g{g//1000000}_e{e//1000000}_floor{floor//1000}'
                result[name] = dict(mode='cap', gpu_hz=g, emc_hz=e,
                                    cpu_khz=grid['cpu_cap_khz'], cpu_floor_khz=floor)
    for g in grid['locked_reference_gpu_hz']:
        name = f'lock_g{g//1000000}_c{grid["cpu_cap_khz"]//1000}'
        result[name] = dict(mode='isolated', gpu_hz=g, emc_hz=max(grid['emc_hz']),
                            cpu_khz=grid['cpu_cap_khz'], cpu_mode='lock')
    return result


def workloads(settings, cfg, preset):
    if preset == 'quick':
        return settings['quick']['workloads']
    return settings['pilot_workloads'] if preset == 'pilot' else cfg['workloads']


def repeats(settings, preset, stage):
    spec = settings['quick'] if preset == 'quick' else settings
    return spec['profile_repeats' if stage == 'profile' else 'validation_repeats']


def preference_profiles(settings, objectives):
    profiles = settings.get('pareto', {}).get('preference_profiles', {})
    reserved = {'dynamic','max_clock','max_clock_idle_dynamic','min_energy','min_power','frontier_point'}
    result = {}
    for name, weights in profiles.items():
        if name in reserved or not name:
            raise ValueError(f'Reserved/empty preference profile name: {name!r}')
        if set(weights)-set(objectives):
            raise ValueError(f'Preference {name} uses metrics outside Pareto objectives')
        if any(not isinstance(value,(int,float)) or value<0 for value in weights.values()):
            raise ValueError(f'Preference {name} weights must be finite non-negative numbers')
        total = sum(weights.values())
        if total <= 0:
            raise ValueError(f'Preference {name} has zero total weight')
        result[name] = {key:value/total for key,value in weights.items()}
    return result


def profile_jobs(settings, cfg, preset):
    available = points(settings)
    names = settings['quick']['point_names'] if preset == 'quick' else list(available)
    result = []
    for repeat in range(repeats(settings, preset, 'profile')):
        batch = [dict(point=n, hardware=available[n], workload=w, repeat=repeat)
                 for w in workloads(settings, cfg, preset) for n in names]
        random.Random(cfg['seed']+12100+repeat).shuffle(batch)
        result.extend(batch)
    return result


class FloorHardware(phase.IsolatedHardware):
    def apply(self, point):
        if 'cpu_floor_khz' not in point:
            return super().apply(point)
        minimum = point['cpu_floor_khz'] or min(self.paths.cpu_available_hz())
        maximum = point['cpu_khz']
        if minimum > maximum or minimum not in self.paths.cpu_available_hz():
            raise RuntimeError('Unsupported CPU floor/range')
        self.domain(self.gpu, min(self.paths.gpu_available_hz()), point['gpu_hz'],
                    self.p['gpu_dynamic_governor'])
        for path in self.cpus:
            self.domain(path, minimum, maximum, self.p['cpu_dynamic_governor'], True)
        self.emc.write_text(str(point['emc_hz']))
        if int(self.emc.read_text()) != point['emc_hz']:
            raise RuntimeError('EMC cap readback mismatch')
        return self.state()


def trim_telemetry(power):
    # Only after every window in a committed block/trial has been analyzed.
    # Raw JSONL is retained; the live reader does not need hours of RAM history.
    with power.lock:
        power.samples = power.samples[-4:]
        power.clocks = power.clocks[-4:]


def thermal_guard(hw, cfg):
    value = hw.paths.temperature_c()
    if value is not None and value >= cfg['measurement']['abort_temperature_c']:
        raise RuntimeError('Thermal guard reached; no silent frequency substitution')


def characterize(engine, power, hw, out, cfg, settings, diagnostic, preset):
    tasks = profile_jobs(settings, cfg, preset)
    directory = out/'profile'
    directory.mkdir(exist_ok=True)
    v2.atomic(out/'profile_jobs.json', tasks)
    completed = 0
    for task in tasks:
        target = directory/(v2.digest(task)+'.json')
        if target.exists():
            completed += 1
            continue
        thermal_guard(hw, cfg)
        emit(stage='profile', event='block_start', completed=completed, total=len(tasks), task=task)
        configured = hw.apply(task['hardware'])
        time.sleep(cfg['measurement']['settle_s'])
        result = phase.measure(engine, power, hw, cfg, diagnostic, task)
        if result['clocks']['emc_above_cap_samples']:
            raise RuntimeError('Measured EMC cap excursion; retain raw log and investigate')
        v2.atomic(target, dict(configured=configured, **result))
        completed += 1
        status = dict(stage='profile', completed=completed, total=len(tasks), task=task)
        v2.atomic(out/'progress.json', status)
        emit(event='block_complete', **status)
        trim_telemetry(power)


def select(out, settings, cfg, preset):
    records = v2.rows(out, 'profile')
    expected = {v2.digest(t) for t in profile_jobs(settings, cfg, preset)}
    observed = {v2.digest(r['task']) for r in records}
    if observed != expected or len(records) != len(expected) or not all(r['valid'] for r in records):
        raise RuntimeError(f'Complete declared grid required before selection: {len(records)}/{len(expected)}')
    groups = defaultdict(list)
    for row in records:
        groups[(tuple(row['task']['workload']), row['task']['point'])].append(row)
    table = [dict(workload=list(w), point=name, hardware=rows[0]['task']['hardware'],
                  **validation.aggregate(rows)) for (w, name), rows in sorted(groups.items())]
    objectives = tuple(settings.get('pareto', {}).get(
        'objectives', ('avg_power_w', 'energy_per_token_j', 'tpot_ms')))
    supported = {'avg_power_w','energy_per_token_j','ttft_ms','tpot_ms'}
    if not objectives or len(set(objectives))!=len(objectives) or set(objectives)-supported:
        raise ValueError('Pareto objectives must be unique supported lower-is-better metrics')
    profiles = preference_profiles(settings, objectives)
    frontiers, selections = [], []
    for w in workloads(settings, cfg, preset):
        candidates = [r for r in table if r['workload']==w and r['point'] not in ('dynamic', 'max_clock')]
        for scenario, ttft in settings['ttft_scenarios_ms'].items():
            for slo in settings['tpot_slos_ms']:
                feasible = [r for r in candidates if r['tpot_ms']<=slo and (ttft is None or r['ttft_ms']<=ttft)]
                frontier = [r for r in feasible if not any(
                    all(other[k]<=r[k] for k in objectives) and any(other[k]<r[k] for k in objectives)
                    for other in feasible)]
                frontiers.append(dict(workload=w, scenario=scenario, tpot_limit_ms=slo,
                                      ttft_limit_ms=ttft, points=frontier))
                for strategy, objective in [('min_energy', 'energy_per_token_j'), ('min_power', 'avg_power_w')]:
                    best = min(frontier, key=lambda r:(r[objective], r['tpot_ms'], r['point'])) if frontier else None
                    selections.append(dict(workload=w, scenario=scenario, tpot_limit_ms=slo,
                                           ttft_limit_ms=ttft, strategy=strategy,
                                           point=best['point'] if best else None,
                                           hardware=best['hardware'] if best else None,
                                           prediction=best, selection_role='frontier_anchor',
                                           status='predicted_feasible' if best else 'infeasible'))
                bounds = {key:(min(r[key] for r in frontier),max(r[key] for r in frontier))
                          for key in objectives} if frontier else {}
                for strategy, weights in profiles.items():
                    def score(row):
                        value = 0.0
                        for key, weight in weights.items():
                            low, high = bounds[key]
                            value += weight*(0.0 if high==low else (row[key]-low)/(high-low))
                        return value
                    best = min(frontier,key=lambda r:(score(r),r['tpot_ms'],r['point'])) if frontier else None
                    prediction = dict(best,preference_score=score(best)) if best else None
                    selections.append(dict(workload=w,scenario=scenario,tpot_limit_ms=slo,
                                           ttft_limit_ms=ttft,strategy=strategy,
                                           point=best['point'] if best else None,
                                           hardware=best['hardware'] if best else None,
                                           prediction=prediction,selection_role='preference_profile',
                                           preference_weights=weights,
                                           status='predicted_feasible' if best else 'infeasible'))
    result = dict(scope='cpu_floor_joint_vector_extension_not_original_GPU_only_knee',
                  feasibility='profile medians; independent validation required',
                  grid_complete=True, profile_digest=v2.digest(records), table=table,
                  objectives=list(objectives), frontiers=frontiers, selections=selections)
    target = out/'selection.json'
    if target.exists() and json.loads(target.read_text()) != result:
        raise RuntimeError('Frozen selection differs; use a new campaign, do not reselect on validation data')
    v2.atomic(target, result)
    phase.write_csv(out/'static_table.csv', [dict(r, hardware=json.dumps(r['hardware'])) for r in table])
    emit(selection_complete=True, points=len(table), selected_pairs=len({(tuple(s['workload']), s['point'])
                                                                       for s in selections if s['point']}))
    return result


def validation_jobs(out, settings, cfg, preset):
    selection = json.loads((out/'selection.json').read_text())
    if settings.get('pareto', {}).get('validation_scope') == 'frontier_union':
        pairs = {(tuple(frontier['workload']), point['point']):point['hardware']
                 for frontier in selection['frontiers'] for point in frontier['points']}
    else:
        pairs = {(tuple(s['workload']), s['point']):s['hardware']
                 for s in selection['selections'] if s['point']}
    for w in workloads(settings, cfg, preset):
        for name in ('dynamic', 'max_clock'):
            pairs[(tuple(w), name)] = dict(mode=name)
    result = []
    for repeat in range(repeats(settings, preset, 'validate')):
        batch = []
        for (w, name), point in sorted(pairs.items()):
            count = 1 if preset == 'quick' else settings['validation_requests_by_output_tokens'][str(w[1])]
            batch.append(dict(workload=list(w), point=name, hardware=point, repeat=repeat,
                              requests_per_block=count))
        random.Random(cfg['seed']+13100+repeat).shuffle(batch)
        result.extend(batch)
    return result


def validate(engine, power, hw, out, cfg, settings, diagnostic, preset):
    tasks = validation_jobs(out, settings, cfg, preset)
    directory = out/'validate'
    directory.mkdir(exist_ok=True)
    v2.atomic(out/'validation_jobs.json', tasks)
    completed = 0
    for task in tasks:
        target = directory/(v2.digest(task)+'.json')
        if target.exists():
            completed += 1
            continue
        thermal_guard(hw, cfg)
        emit(stage='validate', event='block_start', completed=completed, total=len(tasks), task=task)
        configured = hw.apply(task['hardware'])
        time.sleep(cfg['measurement']['settle_s'])
        result = validation.measure(engine, power, hw, cfg, diagnostic, settings, task, configured)
        v2.atomic(target, result)
        completed += 1
        status = dict(stage='validate', completed=completed, total=len(tasks), task=task)
        v2.atomic(out/'progress.json', status)
        emit(event='block_complete', **status)
        trim_telemetry(power)


def report(out, settings, cfg, preset):
    selection = json.loads((out/'selection.json').read_text())
    selections = selection['selections']
    groups = defaultdict(dict)
    for r in v2.rows(out, 'validate'):
        task = r['task']
        key = (tuple(task['workload']), task['point'])
        if task['repeat'] in groups[key]:
            raise RuntimeError('Duplicate validation repeat')
        groups[key][task['repeat']] = r
    expected = set(range(repeats(settings, preset, 'validate')))
    comparisons, summaries = [], []
    metrics = {}
    for (w, name), by_repeat in sorted(groups.items()):
        metrics[(w, name)] = validation.aggregate(list(by_repeat.values()))
        summaries.append(dict(workload='/'.join(map(str,w)), point=name,
                              complete=set(by_repeat)==expected, **metrics[(w,name)]))
    cache = {}

    def evaluate(s, family_primary):
        w, name = tuple(s['workload']), s['point']
        base_item = {k:s[k] for k in ('workload','scenario','tpot_limit_ms','ttft_limit_ms','strategy','point','hardware')}
        for key in ('selection_role','preference_weights'):
            if key in s:
                base_item[key]=s[key]
        by_repeat = groups.get((w,name), {})
        if not name or set(by_repeat)!=expected:
            return dict(base_item, status='infeasible' if not name else 'not_validated')
        rows = [by_repeat[r] for r in sorted(expected)]
        qs = [q for r in rows for q in r['requests']]
        k = sum(q['tpot_ms']<=s['tpot_limit_ms'] and
                (s['ttft_limit_ms'] is None or q['ttft_ms']<=s['ttft_limit_ms']) for q in qs)
        item = dict(base_item, **metrics[(w,name)], passed=k, observed_attainment=k/len(qs),
                    request_wilson_ci95=v2.wilson(k,len(qs)),
                    status='pass_observed' if k/len(qs)>=settings['required_observed_attainment'] else 'fail')
        for baseline in ('dynamic','max_clock'):
            base = groups.get((w,baseline), {})
            if set(base)!=expected:
                continue
            bs = [base[r] for r in sorted(expected)]
            if any(a['output_tokens']!=b['output_tokens'] or
                   [q['prompt_sha256'] for q in a['requests']]!=[q['prompt_sha256'] for q in b['requests']]
                   for a,b in zip(rows,bs)):
                raise RuntimeError('Mismatched fixed work in paired validation')
            primary = (family_primary and s['strategy']=='min_energy' and s['scenario']=='interactive_1s' and
                       s['tpot_limit_ms']==50 and baseline=='max_clock')
            for metric in ('energy_per_token_j','avg_power_w'):
                alpha = settings['statistics']['family_alpha']
                if primary and metric=='energy_per_token_j':
                    alpha /= len(workloads(settings,cfg,preset))
                key = (w,name,baseline,metric,alpha)
                if key not in cache:
                    cache[key] = validation.paired_interval(rows,bs,metric,alpha,
                        settings['statistics']['bootstrap_replicates'],int(v2.digest(key)[:16],16))
                for suffix,value in cache[key].items():
                    item[f'{metric}_vs_{baseline}_{suffix}'] = value
        return item

    comparisons = [evaluate(s,True) for s in selections]
    frontier_specs=[]
    seen=set()
    for frontier in selection['frontiers']:
        for point in frontier['points']:
            key=(tuple(frontier['workload']),frontier['scenario'],frontier['tpot_limit_ms'],
                 frontier['ttft_limit_ms'],point['point'])
            if key in seen:
                continue
            seen.add(key)
            frontier_specs.append(dict(workload=frontier['workload'],scenario=frontier['scenario'],
                tpot_limit_ms=frontier['tpot_limit_ms'],ttft_limit_ms=frontier['ttft_limit_ms'],
                strategy='frontier_point',point=point['point'],hardware=point['hardware'],
                selection_role='pareto_frontier_member'))
    frontier_results=[evaluate(s,False) for s in frontier_specs]
    v2.atomic(out/'comparison.json', comparisons)
    v2.atomic(out/'frontier_validation.json', frontier_results)
    v2.atomic(out/'validation_summary.json', summaries)
    phase.write_csv(out/'comparison.csv', [dict(r,hardware=json.dumps(r.get('hardware'))) for r in comparisons])
    phase.write_csv(out/'frontier_validation.csv',
                    [dict(r,hardware=json.dumps(r.get('hardware'))) for r in frontier_results])
    phase.write_csv(out/'validation_summary.csv', summaries)
    lines=['# CPU-floor extension: independent validation','',
           'Profile data form Pareto sets; disjoint validation data report them without reselection.',
           'Pareto sets contain non-dominated tradeoffs, not a unique optimum.',
           f"Validation scope: {settings.get('pareto',{}).get('validation_scope','selected_anchors')}.",
           'pass_observed means observed attainment >=95%, not a future or serving guarantee.',
           'Request Wilson intervals assume independence; requests within a block may be correlated.',
           'Energy/power intervals resample paired repeat blocks. The primary family is min_energy,',
           'TPOT 50 ms, TTFT 1 s versus MaxClock across the declared workloads (Bonferroni adjustment).',
           'Other intervals are nominal 95%, with no multiple-comparison adjustment.','',
           '| Workload | Point | Blocks | TTFT ms | TPOT ms | W | J/token |',
           '|---|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['workload']} | {r['point']} | {r['blocks']} | "+
                     ' | '.join(f'{r[k]:.3f}' for k in ('ttft_ms','tpot_ms','avg_power_w','energy_per_token_j'))+' |')
    lines.extend(['','All declared SLO scenarios, anchors and frozen preference profiles are in comparison.csv.',
                  'Every independently measured profile-frontier member is in frontier_validation.csv.',
                  'Long-input TTFT failures remain failures at 1 s. No implicit fallback or threshold relaxation.',
                  'This is not an unchanged original controller or a prefill/decode switching experiment.'])
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    emit(validation_report=str(out/'RESULTS.md'))
    return comparisons


def trial(engine, power, hw, cfg, settings, args, trace, strategy, plans):
    initial = dict(mode='max_clock' if strategy=='max_clock' else 'dynamic')
    hw.apply(initial)
    engine.run(64,64)
    time.sleep(cfg['measurement']['settle_s'])
    power.wait_after(time.monotonic())
    current = initial
    switches, responses, queue = [], [], []
    index = 0
    start = time.monotonic()
    horizon = settings['serving']['duration_s']

    def change(point, reason):
        nonlocal current
        if point == current:
            return 0.0
        before = time.monotonic()
        state = hw.apply(point)
        after = time.monotonic()
        switches.append(dict(start=before,end=after,latency_ms=(after-before)*1000,
                             reason=reason,hardware=point,configured=state))
        current = point
        return (after-before)*1000

    while index<len(trace) or queue or time.monotonic()<start+horizon:
        now = time.monotonic()
        while index<len(trace) and trace[index]['scheduled_time']<=now-start:
            req = dict(trace[index], request_id=index)
            if len(queue)>=settings['serving']['queue_limit']:
                responses.append(dict(request=req,dropped=True))
            else:
                queue.append(req)
            index += 1
        if not queue:
            # Keep the historical MaxClock baseline fixed. The explicit
            # MaxClock-idle-dynamic control isolates idle release from tuning.
            if strategy!='max_clock':
                change(dict(mode='dynamic'),'queue_empty_idle_release')
            if index>=len(trace) and time.monotonic()>=start+horizon:
                break
            wake = start+trace[index]['scheduled_time'] if index<len(trace) else start+horizon
            time.sleep(min(.02,max(0,wake-time.monotonic())))
            continue
        req = queue.pop(0)
        thermal_guard(hw,cfg)
        dispatch = time.monotonic()
        w = (req['prompt_length'],req['output_length'])
        if strategy=='dynamic':
            point=dict(mode='dynamic')
        elif strategy in ('max_clock','max_clock_idle_dynamic'):
            point=dict(mode='max_clock')
        else:
            point=plans[(strategy,w)]
        control_ms=change(point,'request_dispatch')
        result=engine.run(*w)
        arrival=start+req['scheduled_time']
        result.update(queue_wait_ms=max(0,(dispatch-arrival)*1000),control_latency_ms=control_ms,
                      ttft_end_to_end_ms=(result['token_times'][0]-arrival)*1000,
                      arrival_to_completion_ms=(result['end']-arrival)*1000)
        ttft_limit=settings['ttft_scenarios_ms'][args.scenario]
        result['slo_pass']=(result['tpot_ms']<=args.slo and
                            (ttft_limit is None or result['ttft_end_to_end_ms']<=ttft_limit))
        responses.append(dict(request=req,dropped=False,hardware=point,metrics=result))
    # Include the final idle release even if the queue drained beyond horizon.
    if strategy!='max_clock':
        change(dict(mode='dynamic'),'final_idle_release')
    end=time.monotonic()
    energy=power.energy(start,end,cfg['measurement']['min_samples'])
    fixed_horizon=power.energy(start,start+horizon,cfg['measurement']['min_samples'])
    complete=[r['metrics'] for r in responses if not r['dropped']]
    tokens=sum(q['output_tokens'] for q in complete)
    passed=sum(q['slo_pass'] for q in complete)
    scope_point=dict(emc_hz=max(cfg['platform']['emc_hz']))
    clocks=power.clock_summary(start,end,scope_point,settings['_diagnostic'])
    if clocks['temperature_max_c'] is not None and clocks['temperature_max_c']>=cfg['measurement']['abort_temperature_c']:
        raise RuntimeError('Serving thermal guard; raw telemetry retained, trial not committed')
    return dict(strategy=strategy,trace_sha256=v2.digest(trace),trace=args.trace,scenario=args.scenario,
                tpot_limit_ms=args.slo,ttft_limit_ms=settings['ttft_scenarios_ms'][args.scenario],
                responses=responses,switches=switches,offered=len(trace),completed=len(complete),
                dropped=len(trace)-len(complete),passed=passed,observed_attainment=passed/len(trace),
                request_wilson_ci95=v2.wilson(passed,len(trace)),duration_with_drain_s=end-start,
                comparable_fixed_work=len(complete)==len(trace),output_tokens=tokens,
                energy_per_token_j=energy['total_energy_j']/tokens if tokens else None,
                ttft_with_queue_p95_ms=v2.percentile([q['ttft_end_to_end_ms'] for q in complete],.95) if complete else None,
                tpot_p95_ms=v2.percentile([q['tpot_ms'] for q in complete],.95) if complete else None,
                fixed_horizon=fixed_horizon,clocks=clocks,raw_telemetry=str(Path(power.file.name).resolve()),**energy)


def serving_directory(out,args):
    return out/'serving'/f'{args.trace}_{args.scenario}_{args.slo:g}'


def serve(engine,power,hw,out,cfg,settings,args):
    if args.preset!='full':
        raise RuntimeError('Serving requires the full declared workload table; quick/pilot do not substitute nearest workloads')
    from src.experiments.run_serving_benchmark import RequestGenerator
    strategies=args.strategies or settings['serving']['strategies']
    if len(set(strategies))!=len(strategies):
        raise ValueError('Duplicate serving strategy')
    comparisons=report(out,settings,cfg,args.preset)
    traces=[RequestGenerator(args.trace,settings['serving']['duration_s'],cfg['seed']+r).generate_requests()
            for r in range(settings['serving']['repeats'])]
    known={tuple(w) for w in workloads(settings,cfg,args.preset)}
    plans={}
    for trace in traces:
        if not trace:
            raise RuntimeError('Trace generator returned no requests')
        arrivals=[r['scheduled_time'] for r in trace]
        if arrivals!=sorted(arrivals) or arrivals[0]<0 or arrivals[-1]>settings['serving']['duration_s']:
            raise RuntimeError('Invalid fixed arrival trace')
        for req in trace:
            w=(req['prompt_length'],req['output_length'])
            if w not in known:
                raise RuntimeError(f'Trace workload {w} is outside the static table; no nearest-workload fallback')
            for strategy in strategies:
                if strategy in ('dynamic','max_clock','max_clock_idle_dynamic'):
                    continue
                row=next((r for r in comparisons if tuple(r['workload'])==w and
                          r['strategy']==strategy and r['scenario']==args.scenario and r['tpot_limit_ms']==args.slo),None)
                if not row or row['status']!='pass_observed':
                    raise RuntimeError(f'Serving blocked: {strategy} lacks independently validated feasibility for {w}')
                plans[(strategy,w)]=row['hardware']
    directory=serving_directory(out,args)
    directory.mkdir(parents=True,exist_ok=True)
    identity=dict(strategies=strategies,trace=args.trace,scenario=args.scenario,slo_ms=args.slo,
                  trace_hashes=[v2.digest(t) for t in traces],selection_sha256=v2.file_hash(out/'selection.json'),
                  validation_sha256=v2.file_hash(out/'comparison.json'),
                  trace_source_sha256=v2.file_hash(v2.ROOT/'src/experiments/run_serving_benchmark.py'))
    manifest=directory/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:
        raise RuntimeError('Serving protocol changed; use a new serving scenario/campaign')
    v2.atomic(manifest,identity)
    for repeat,trace in enumerate(traces):
        v2.atomic(directory/f'trace_{repeat}.json',trace)
        order=list(strategies)
        random.Random(cfg['seed']+14100+repeat).shuffle(order)
        for strategy in order:
            target=directory/f'result_{repeat}_{strategy}.json'
            if target.exists():
                continue
            emit(stage='serve',event='trial_start',repeat=repeat,strategy=strategy,offered=len(trace))
            result=trial(engine,power,hw,cfg,settings,args,trace,strategy,plans)
            v2.atomic(target,dict(repeat=repeat,**result))
            emit(stage='serve',event='trial_complete',repeat=repeat,strategy=strategy,
                 attainment=result['observed_attainment'],energy_j=result['total_energy_j'])
            trim_telemetry(power)
    serving_report(out,settings,args)


def serving_report(out,settings,args):
    directory=serving_directory(out,args)
    manifest=json.loads((directory/'protocol.json').read_text())
    groups=defaultdict(dict)
    for path in directory.glob('result_*.json'):
        r=json.loads(path.read_text())
        if r['repeat'] in groups[r['strategy']]:
            raise RuntimeError('Duplicate serving repeat')
        groups[r['strategy']][r['repeat']]=r
    expected=set(range(settings['serving']['repeats']))
    summary=[]
    for strategy in manifest['strategies']:
        by_repeat=groups[strategy]
        rows=[by_repeat[r] for r in sorted(by_repeat)]
        if not rows:
            continue
        complete=set(by_repeat)==expected
        fixed=complete and all(r['comparable_fixed_work'] for r in rows)
        energy=sum(r['total_energy_j'] for r in rows)
        duration=sum(r['duration_with_drain_s'] for r in rows)
        offered=sum(r['offered'] for r in rows)
        passed=sum(r['passed'] for r in rows)
        tokens=sum(r['output_tokens'] for r in rows)
        item=dict(strategy=strategy,repeats=len(rows),complete=complete,comparable_fixed_work=fixed,
                  offered=offered,completed=sum(r['completed'] for r in rows),dropped=sum(r['dropped'] for r in rows),
                  observed_attainment=passed/offered,all_repeats_observed_slo_pass=complete and all(
                      r['observed_attainment']>=settings['required_observed_attainment'] for r in rows),
                  total_energy_j=energy,avg_power_w=energy/duration,
                  energy_per_token_j=energy/tokens if tokens else None,
                  duration_with_drain_s=duration,
                  fixed_horizon_avg_power_w=sum(r['fixed_horizon']['total_energy_j'] for r in rows)/
                                            (len(rows)*settings['serving']['duration_s']),
                  switch_count=sum(len(r['switches']) for r in rows))
        for baseline in ('dynamic','max_clock','max_clock_idle_dynamic'):
            base=groups[baseline]
            comparable=fixed and set(base)==expected and all(
                base[n]['comparable_fixed_work'] and base[n]['trace_sha256']==by_repeat[n]['trace_sha256'] and
                base[n]['output_tokens']==by_repeat[n]['output_tokens'] for n in expected)
            item[f'total_energy_delta_vs_{baseline}_pct']=(100*(energy/sum(r['total_energy_j'] for r in base.values())-1)
                                                          if comparable else None)
        summary.append(item)
    v2.atomic(directory/'summary.json',summary)
    phase.write_csv(directory/'summary.csv',summary)
    lines=['# CPU-floor serving extension','',
           'Same original RequestGenerator arrivals for every strategy; FIFO queue and full drain.',
           'Energy includes idle periods and every measured configuration switch. Drops count as SLO failures.',
           'Energy comparisons are withheld unless the complete fixed work and repeat set match.',
           'MaxClock remains fixed while idle. MaxClock-idle-dynamic isolates the effect of idle release.',
           'These are observed attainment rates, not population guarantees.','',
           '| Strategy | Repeats | Completed/offered | SLO % | Every repeat passes | W | J/token | Energy vs Dynamic % |',
           '|---|---:|---:|---:|---|---:|---:|---:|']
    for r in summary:
        j='--' if r['energy_per_token_j'] is None else f"{r['energy_per_token_j']:.3f}"
        delta=r['total_energy_delta_vs_dynamic_pct']
        d='--' if delta is None else f'{delta:.3f}'
        lines.append(f"| {r['strategy']} | {r['repeats']} | {r['completed']}/{r['offered']} | "
                     f"{100*r['observed_attainment']:.2f} | {r['all_repeats_observed_slo_pass']} | {r['avg_power_w']:.3f} | {j} | {d} |")
    lines.extend(['','summary.csv includes both MaxClock baselines, fixed-horizon power and switch counts.',
                  'Lower energy with failed SLO is not a constrained win. CPU/GPU model quality is not evaluated.'])
    (directory/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    emit(serving_report=str(directory/'RESULTS.md'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['plan','run','profile','select','validate','report','serve','serve-report','restore'])
    ap.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--preset',choices=['quick','pilot','full'],default='full')
    ap.add_argument('--trace',choices=['short_chat','long_generation','bursty_mixed'],default='short_chat')
    ap.add_argument('--slo',type=float,default=50)
    ap.add_argument('--scenario',choices=['tpot_only','interactive_1s','long_context_2s'],default='interactive_1s')
    ap.add_argument('--strategies',nargs='+')
    args=ap.parse_args()
    settings=json.loads(args.config.read_text())
    if settings['schema']!='energyinfra-cpu-floor-v1':
        raise RuntimeError('Unsupported CPU-floor schema')
    objectives=tuple(settings.get('pareto',{}).get(
        'objectives',('avg_power_w','energy_per_token_j','tpot_ms')))
    profiles=preference_profiles(settings,objectives)
    allowed={'dynamic','max_clock','max_clock_idle_dynamic','min_energy','min_power',*profiles}
    if args.strategies and set(args.strategies)-allowed:
        raise ValueError(f'Unknown serving strategies: {sorted(set(args.strategies)-allowed)}')
    diagnostic=json.loads(phase.root_path(settings['diagnostic_config']).read_text())
    cfg=copy.deepcopy(json.loads(phase.root_path(diagnostic['base_config']).read_text()))
    cfg['model']['path']=str(phase.root_path(diagnostic['model_path']))
    cfg['measurement'].update(diagnostic['measurement'])
    cfg['measurement'].update(settings['measurement'])
    if args.slo not in settings['tpot_slos_ms']:
        raise ValueError('SLO is outside the declared configuration')
    tasks=profile_jobs(settings,cfg,args.preset)
    if args.stage=='plan':
        emit(preset=args.preset,configurations=len({t['point'] for t in tasks}),
             workloads=workloads(settings,cfg,args.preset),profile_blocks=len(tasks),
             validation_repeats=repeats(settings,args.preset,'validate'),
             validation_scope=settings.get('pareto',{}).get('validation_scope','selected_anchors'),
             preference_profiles=profiles,
             validation_blocks='determined by frozen frontier/workload union after profiling',
             source_is_original_GPU_only_selector=False,changes_hardware=False)
        return
    out=args.run_dir.resolve()
    if out==v2.ROOT or out==phase.root_path(diagnostic['model_path']).parent:
        raise RuntimeError('Use a new campaign directory')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        emit(worker_pid=os.getpid(),stage=args.stage,run_dir=str(out))
        manifest=out/'protocol.json'
        if args.stage=='restore':
            cfg=json.loads(manifest.read_text())['base_config']
        else:
            files=[Path(__file__).resolve(),Path(v2.__file__).resolve(),Path(phase.__file__).resolve(),
                   Path(validation.__file__).resolve(),v2.ROOT/'src/controller/jetson_sysfs.py',
                   v2.ROOT/'src/controller/platform_config.py',v2.ROOT/'src/experiments/run_serving_benchmark.py']
            identity=dict(settings=settings,diagnostic=diagnostic,base_config=cfg,preset=args.preset,
                          sources={str(p.relative_to(v2.ROOT)):v2.file_hash(p) for p in files})
            if manifest.exists() and json.loads(manifest.read_text())!=identity:
                raise RuntimeError('Source/config/preset changed; use a new run directory')
            v2.atomic(manifest,identity)
        if args.stage=='select':
            select(out,settings,cfg,args.preset)
            return
        if args.stage=='report':
            report(out,settings,cfg,args.preset)
            return
        if args.stage=='serve-report':
            serving_report(out,settings,args)
            return
        if os.geteuid()!=0:
            raise RuntimeError('Use foreground sudo or the sudo nohup launcher for hardware stages')
        signal.signal(signal.SIGTERM,validation.interrupted)
        signal.signal(signal.SIGINT,validation.interrupted)
        with open('/run/energyinfra-v2-hardware.lock','a') as hardware_lock:
            fcntl.flock(hardware_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            hw=FloorHardware(cfg)
            pending,snapshot_path=out/'restore_pending.json',out/'hardware_snapshot.json'
            if args.stage=='restore':
                hw.restore(json.loads(snapshot_path.read_text()))
                v2.atomic(pending,dict(pending=False))
                emit(hardware_restored=True)
                return
            if pending.exists() and json.loads(pending.read_text())['pending']:
                raise RuntimeError('Restoration pending; run restore before resuming')
            environment=validation.validate_hardware(hw,cfg,diagnostic,tasks)
            for point in points(settings).values():
                floor=point.get('cpu_floor_khz',0)
                if floor and (floor not in hw.paths.cpu_available_hz() or floor>point['cpu_khz']):
                    raise RuntimeError('Unsupported CPU floor in configuration')
            env_path=out/'environment.json'
            if env_path.exists() and json.loads(env_path.read_text())!=environment:
                raise RuntimeError('Model/runtime/nvpmodel/platform changed')
            v2.atomic(env_path,environment)
            snapshot=hw.state()
            v2.atomic(snapshot_path,snapshot)
            v2.atomic(pending,dict(pending=True))
            engine=power=None
            try:
                hw.apply(dict(mode='dynamic'))
                engine=v2.Tokens(cfg)
                power=phase.Telemetry(out,cfg)
                if args.stage in ('run','profile'):
                    characterize(engine,power,hw,out,cfg,settings,diagnostic,args.preset)
                if args.stage=='run':
                    select(out,settings,cfg,args.preset)
                if args.stage in ('run','validate'):
                    validate(engine,power,hw,out,cfg,settings,diagnostic,args.preset)
                if args.stage=='serve':
                    live_settings=dict(settings,_diagnostic=diagnostic)
                    serve(engine,power,hw,out,cfg,live_settings,args)
            except BaseException as exc:
                v2.atomic(out/'last_failure.json',dict(stage=args.stage,error=repr(exc),time=time.time()))
                raise
            finally:
                signal.signal(signal.SIGTERM,signal.SIG_IGN)
                signal.signal(signal.SIGINT,signal.SIG_IGN)
                try:
                    if power:
                        power.close()
                finally:
                    try:
                        if engine:
                            engine.model.close()
                    finally:
                        hw.restore(snapshot)
                        v2.atomic(pending,dict(pending=False))
                        emit(hardware_restored=True)
            signal.signal(signal.SIGTERM,validation.interrupted)
            signal.signal(signal.SIGINT,validation.interrupted)
            if args.stage in ('run','validate'):
                report(out,settings,cfg,args.preset)
            emit(stage=args.stage,complete=True)


if __name__=='__main__':
    main()
