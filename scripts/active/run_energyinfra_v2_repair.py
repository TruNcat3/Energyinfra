#!/usr/bin/env python3
"""Supplement failed minimum-power candidates without changing parent results.

Preregister candidates with 10% latency headroom in every profiling request,
then screen in fixed rank order and independently confirm the first passing one.
No fallback is labelled feasible. This is not the original EnergyInfra selector.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import signal
import time

import run_energyinfra_v2 as v2


def make_plan(parent, cfg):
    comparison = json.loads((parent/'comparison_full.json').read_text())
    failed = [r for r in comparison if r['strategy']=='min_power' and r['status']=='fail']
    groups = defaultdict(list)
    for row in v2.rows(parent, 'profile_full'):
        if row.get('valid') and row['hardware']['mode']=='cap':
            groups[(tuple(row['workload']), v2.config_id(row['hardware']))].append(row)
    cases = []
    for row in failed:
        candidates = []
        for (workload, cid), records in groups.items():
            if workload != tuple(row['workload']) or len(records) != cfg['profile_repeats']:
                continue
            requests = [r for block in records for r in block['requests']]
            if not requests or any(r['tpot_ms'] > .9*row['slo_ms'] for r in requests):
                continue
            if row['scenario']!='tpot_only' and any(r['ttft_ms'] > .9*cfg['interactive_ttft_ms'] for r in requests):
                continue
            if cid == v2.config_id(row['hardware']):
                continue
            metrics = v2.summarize(records, cfg, row['slo_ms'], row['scenario'])
            candidates.append(dict(hardware=records[0]['hardware'], config_id=cid, profile=metrics))
        candidates.sort(key=lambda c:(c['profile']['avg_power_w'], c['profile']['tpot_ms'], c['config_id']))
        cases.append(dict(workload=row['workload'], slo_ms=row['slo_ms'], scenario=row['scenario'],
                          original=row, candidates=candidates[:3],
                          status='planned' if candidates else 'no_headroom_candidate'))
    return dict(schema='energyinfra-repair-v1', source='full CAP profile', headroom_fraction=.1,
                criterion='all profiling requests within 90 percent of applicable latency limits',
                max_candidates_per_case=3, independent_blocks_per_stage=10, cases=cases)


def collect_groups(out, stage):
    grouped = defaultdict(list)
    for row in v2.rows(out, stage):
        if row.get('valid'):
            grouped[(tuple(row['workload']), v2.config_id(row['hardware']))].append(row)
    return grouped


def screen_choices(plan, out, cfg):
    groups = collect_groups(out, 'screen')
    result = []
    for case in plan['cases']:
        assessed, chosen = [], None
        for candidate in case['candidates']:
            records = groups.get((tuple(case['workload']), candidate['config_id']), [])
            if len(records) != 10:
                raise RuntimeError('Screening incomplete; finish screen before confirm')
            metrics = v2.summarize(records, cfg, case['slo_ms'], case['scenario'])
            passed = metrics['attainment'] >= cfg['required_attainment']
            assessed.append(dict(candidate=candidate, metrics=metrics, passed=passed))
            if passed and chosen is None:
                chosen = candidate
        result.append(dict(case=case, assessed=assessed, chosen=chosen))
    return result


def tasks_for(plan, out, cfg, stage):
    pairs = {}
    if stage=='screen':
        for case in plan['cases']:
            for candidate in case['candidates']:
                pairs[(tuple(case['workload']), candidate['config_id'])] = candidate['hardware']
    else:
        choices = screen_choices(plan, out, cfg)
        v2.atomic(out/'screen_decisions.json', choices)
        for choice in choices:
            if choice['chosen']:
                c = choice['chosen']
                pairs[(tuple(choice['case']['workload']), c['config_id'])] = c['hardware']
    for case in plan['cases']:
        for mode in ['dynamic', 'max_clock']:
            pairs[(tuple(case['workload']), mode)] = dict(mode=mode)
    tasks = [dict(workload=list(w), hardware=c, repeat=r) for (w, _), c in sorted(pairs.items()) for r in range(10)]
    random.Random(cfg['seed']+(2000 if stage=='screen' else 3000)).shuffle(tasks)
    return tasks


def write_report(plan, out, cfg):
    choices = screen_choices(plan, out, cfg)
    groups = collect_groups(out, 'confirm')
    results = []
    for choice in choices:
        case, candidate = choice['case'], choice['chosen']
        item = dict(workload=case['workload'], scenario=case['scenario'], slo_ms=case['slo_ms'],
                    original_status=case['original']['status'], selected=candidate,
                    status='no_screen_pass' if candidate is None else 'not_confirmed', baselines={})
        for mode in ['dynamic', 'max_clock']:
            records = groups.get((tuple(case['workload']),mode), [])
            if len(records)==10:
                item['baselines'][mode] = v2.summarize(records,cfg,case['slo_ms'],case['scenario'])
        if candidate:
            records = groups.get((tuple(case['workload']),candidate['config_id']), [])
            if len(records)==10:
                metrics = v2.summarize(records,cfg,case['slo_ms'],case['scenario'])
                item.update(metrics=metrics, status='pass' if metrics['attainment']>=cfg['required_attainment'] else 'fail')
                item['deltas_pct'] = {mode:{k:(metrics[k]/base[k]-1)*100 for k in ['avg_power_w','energy_per_token_j','ttft_ms','tpot_ms']} for mode,base in item['baselines'].items()}
        results.append(item)
    v2.atomic(out/'comparison.json',results)
    lines=['# Independent confirmation of headroom candidates', '',
           'Negative deltas mean lower values. Screening results are not confirmation results.',
           'No claim of original-selector reproduction or battery-life improvement.', '',
           '| Workload | SLO ms | Scenario | Status | Pass/requests | W | J/token | Power vs Dynamic % | Energy vs Dynamic % | Power vs max_clock % | Energy vs max_clock % |',
           '|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m=r.get('metrics',{})
        d=r.get('deltas_pct',{}).get('dynamic',{})
        x=r.get('deltas_pct',{}).get('max_clock',{})
        def fmt(value):
            return '-' if value is None else f'{value:.3f}'
        lines.append('| '+' | '.join([str(r['workload']),str(r['slo_ms']),r['scenario'],r['status'],
                     f"{m.get('slo_pass','-')}/{m.get('n_requests','-')}",fmt(m.get('avg_power_w')),
                     fmt(m.get('energy_per_token_j')),fmt(d.get('avg_power_w')),fmt(d.get('energy_per_token_j')),
                     fmt(x.get('avg_power_w')),fmt(x.get('energy_per_token_j'))])+' |')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['plan','screen','confirm','report','restore'])
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    parent=args.parent.resolve()
    out=args.run_dir.resolve()
    if out==parent:
        raise RuntimeError('Use a separate supplementary run directory')
    identity=json.loads((parent/'protocol.json').read_text())
    cfg=identity['config']
    if v2.file_hash(v2.__file__)!=identity['implementation_sha256']:
        raise RuntimeError('Base measurement code changed since parent campaign')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest_path=out/'protocol.json'
        provenance=dict(parent=str(parent), parent_protocol=identity, repair_source_sha256=v2.file_hash(__file__),
                        parent_comparison_sha256=v2.file_hash(parent/'comparison_full.json'))
        if manifest_path.exists() and json.loads(manifest_path.read_text())!=provenance:
            raise RuntimeError('Supplement protocol changed; use a new output directory')
        v2.atomic(manifest_path,provenance)
        plan_path=out/'plan.json'
        if plan_path.exists():
            plan=json.loads(plan_path.read_text())
        else:
            plan=make_plan(parent,cfg)
            v2.atomic(plan_path,plan)
        if args.stage=='plan':
            print(json.dumps(dict(cases=len(plan['cases']),screen_blocks=len(tasks_for(plan,out,cfg,'screen')),
                                  details=[{k:c[k] for k in ['workload','slo_ms','scenario','status']} for c in plan['cases']]),indent=2))
            return
        if args.stage=='report':
            write_report(plan,out,cfg)
            return
        if os.geteuid()!=0:
            raise RuntimeError('Use foreground sudo or the supplementary launcher')
        with open('/run/energyinfra-v2-hardware.lock','a') as hardware_lock:
            fcntl.flock(hardware_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            hw=v2.Hardware(cfg)
            pending=out/'restore_pending.json'
            snapshot_path=out/'hardware_snapshot.json'
            if args.stage=='restore':
                hw.restore(json.loads(snapshot_path.read_text()))
                v2.atomic(pending,dict(pending=False))
                return
            if pending.exists() and json.loads(pending.read_text())['pending']:
                raise RuntimeError('Restore pending; run restore first')
            parent_pending=parent/'restore_pending.json'
            if not parent_pending.exists() or json.loads(parent_pending.read_text())['pending']:
                raise RuntimeError('Parent hardware restoration not confirmed')
            env=json.loads((parent/'environment.json').read_text())
            if (v2.file_hash(v2.ROOT/cfg['model']['path'])!=env['model_sha256'] or
                platform.release()!=env['kernel'] or
                importlib.metadata.version('llama-cpp-python')!=env['llama_cpp_python']):
                raise RuntimeError('Model/runtime differs from parent campaign')
            tasks=tasks_for(plan,out,cfg,args.stage)
            v2.atomic(out/(args.stage+'_manifest.json'),tasks)
            target=out/args.stage
            target.mkdir(exist_ok=True)
            snapshot=hw.state()
            v2.atomic(snapshot_path,snapshot)
            v2.atomic(pending,dict(pending=True))
            def interrupt(signum,frame):
                raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM,interrupt)
            signal.signal(signal.SIGINT,interrupt)
            power=None
            try:
                engine=v2.Tokens(cfg)
                power=v2.Power(out,cfg)
                power.wait_after(time.monotonic())
                for i,task in enumerate(tasks):
                    path=target/(v2.digest(task)+'.json')
                    if path.exists():
                        continue
                    configured=hw.apply(task['hardware'])
                    time.sleep(cfg['measurement']['settle_s'])
                    measurement=v2.measure(engine,power,hw,cfg,task['workload'])
                    v2.atomic(path,dict(task=task,hardware=task['hardware'],configured=configured,**measurement))
                    v2.atomic(out/'progress.json',dict(stage=args.stage,completed=i+1,total=len(tasks),task=task))
                    print(json.dumps(dict(stage=args.stage,completed=i+1,total=len(tasks),task=task)),flush=True)
                if args.stage=='confirm':
                    write_report(plan,out,cfg)
                else:
                    v2.atomic(out/'screen_decisions.json',screen_choices(plan,out,cfg))
            except BaseException as exc:
                v2.atomic(out/'last_failure.json',dict(stage=args.stage,error=repr(exc)))
                raise
            finally:
                try:
                    if power:
                        power.close()
                finally:
                    hw.restore(snapshot)
                    v2.atomic(pending,dict(pending=False))


if __name__=='__main__':
    main()
