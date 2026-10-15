#!/usr/bin/env python3
"""Build an isolated serving cohort from validated parent and repair records.

Reuses the v2 token, telemetry and replay implementation without changing it.
This is an extension comparison, not legacy ThermalSLO/BestStatic reproduction.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import subprocess
import sys

import run_energyinfra_v2 as v2

# Reporting-only predecessor: measurement, imports and scheduling are unchanged.
REPORT_COMPATIBLE_ADAPTER_SHA256 = {
    '79cccaa5b7895e5131e820d3e6dd7d6dc140df05c2582fc3dbbaa4babc8990de',
}


def read(path):
    return json.loads(path.read_text())


def group(row):
    return (tuple(row['workload']), v2.config_id(row['hardware']))


def prepare(args):
    parent, repair, out = args.parent.resolve(), args.repair.resolve(), args.run_dir.resolve()
    if out in (parent, repair):
        raise RuntimeError('Serving must use a separate output directory')
    protocol = read(parent/'protocol.json')
    if v2.file_hash(v2.__file__) != protocol['implementation_sha256']:
        raise RuntimeError('Base protocol implementation has changed')
    repair_protocol = read(repair/'protocol.json')
    if repair_protocol['parent'] != str(parent) or repair_protocol['parent_protocol'] != protocol:
        raise RuntimeError('Repair belongs to a different parent campaign')
    if repair_protocol['parent_comparison_sha256'] != v2.file_hash(parent/'comparison_full.json'):
        raise RuntimeError('Parent comparison changed since repair')
    for source in (parent, repair):
        if read(source/'restore_pending.json')['pending']:
            raise RuntimeError(f'Hardware restoration pending in {source}')
    cfg = protocol['config']
    selection = copy.deepcopy(read(parent/'selection_full.json'))
    repaired = read(repair/'comparison.json')
    confirm = v2.rows(repair, 'confirm')
    original = v2.rows(parent, 'validate_full')
    replacement_groups = {}
    replacements = []
    for result in repaired:
        if result['status'] != 'pass':
            continue
        candidate = result['selected']
        w = tuple(result['workload'])
        for mode in [candidate['config_id'], 'dynamic', 'max_clock']:
            records = [r for r in confirm if group(r)==(w,mode) and r.get('valid')]
            if len(records) != 10:
                raise RuntimeError('Repair confirmation missing candidate or matched baseline blocks')
            replacement_groups[(w,mode)] = records
        records = replacement_groups[(w,candidate['config_id'])]
        metrics = v2.summarize(records,cfg,result['slo_ms'],result['scenario'])
        if metrics['attainment'] < cfg['required_attainment']:
            raise RuntimeError('Repair pass disagrees with raw confirmation requests')
        matches = [s for s in selection['selections'] if tuple(s['workload'])==w and
                   s['slo_ms']==result['slo_ms'] and s['scenario']==result['scenario'] and s['strategy']=='min_power']
        if len(matches)!=1:
            raise RuntimeError('Ambiguous or missing repaired selection')
        selected = matches[0]
        replacements.append(dict(workload=list(w),slo_ms=result['slo_ms'],scenario=result['scenario'],
                                 previous=selected['hardware'],replacement=candidate['hardware']))
        selected.update(hardware=candidate['hardware'],prediction=candidate['profile'],
                        predicted_feasible=True,confirmation=metrics,
                        feasibility_rule='independent_headroom_confirmation_observed_attainment')
    effective = [r for r in original if group(r) not in replacement_groups]
    effective.extend(r for records in replacement_groups.values() for r in records)
    # Freeze content fingerprints, including every imported validation block.
    provenance = dict(schema='energyinfra-serving-import-v1', parent=str(parent), repair=str(repair),
                      parent_protocol=protocol, repair_protocol=repair_protocol,
                      selection_sha256=v2.digest(selection), records_sha256=v2.digest(effective),
                      environment=read(parent/'environment.json'),
                      replacements=replacements, adapter_sha256=v2.file_hash(__file__))
    out.mkdir(parents=True,exist_ok=True)
    manifest = out/'serving_import.json'
    if manifest.exists():
        stored = read(manifest)
        comparable = dict(stored)
        if stored.get('adapter_sha256') in REPORT_COMPATIBLE_ADAPTER_SHA256:
            comparable['adapter_sha256'] = provenance['adapter_sha256']
        if comparable!=provenance:
            raise RuntimeError('Imported evidence changed; choose a new serving directory')
        if not (out/'import_complete.json').exists():
            raise RuntimeError('Previous import interrupted; choose a new serving directory')
        return cfg
    if any(out.iterdir()):
        raise RuntimeError('New serving directory must be empty')
    v2.atomic(manifest,provenance)
    v2.atomic(out/'protocol.json',protocol)
    v2.atomic(out/'environment.json',provenance['environment'])
    calibration=read(parent/'calibration.json')
    if not calibration.get('passed'):
        raise RuntimeError('Parent calibration gate did not pass')
    calibration['inherited_from']=str(parent/'calibration.json')
    v2.atomic(out/'calibration.json',calibration)
    v2.atomic(out/'selection_full.json',selection)
    target=out/'validate_full'
    target.mkdir()
    for index,row in enumerate(effective):
        v2.atomic(target/f'imported_{index:05d}.json',row)
    # Recompute reports from imported raw evidence; do not trust pass labels alone.
    v2.report(out,cfg,'full')
    v2.atomic(out/'import_complete.json',dict(records=len(effective),repaired_cases=len(replacements)))
    return cfg


def summarize(out, trace, slo, scenario):
    rows=[r for r in v2.rows(out,'serving') if r['trace']==trace and r['slo_ms']==slo and r['scenario']==scenario]
    results=[]
    for strategy in ['dynamic','max_clock','min_power','min_energy']:
        records=[r for r in rows if r['strategy']==strategy]
        if not records:
            continue
        offered=sum(r['offered'] for r in records)
        completed=sum(r['completed'] for r in records)
        passed=sum(x['metrics']['slo_pass'] for r in records for x in r['results'] if not x['dropped'])
        energy=sum(r['total_energy_j'] for r in records)
        duration=sum(r['duration_with_drain_s'] for r in records)
        tokens=sum(x['metrics']['output_tokens'] for r in records for x in r['results'] if not x['dropped'])
        results.append(dict(strategy=strategy,repeats=len(records),offered=offered,completed=completed,
                            dropped=offered-completed,slo_pass=passed,attainment=passed/offered,
                            total_energy_j=energy,duration_with_drain_s=duration,avg_power_w=energy/duration,
                            energy_per_token_j=energy/tokens if tokens else None,
                            fixed_horizon_energy_j=sum(r['fixed_horizon']['total_energy_j'] for r in records)))
    for row in results:
        for baseline in [b for b in results if b['strategy'] in ['dynamic','max_clock']]:
            comparable=(row['repeats']==baseline['repeats'] and row['offered']==baseline['offered'] and
                        row['dropped']==0 and baseline['dropped']==0)
            row[f'work_matched_vs_{baseline["strategy"]}']=comparable
            row[f'total_energy_delta_vs_{baseline["strategy"]}_pct']=(row['total_energy_j']/baseline['total_energy_j']-1)*100 if comparable else None
    name=f'summary_{trace}_{slo}_{scenario}'
    v2.atomic(out/(name+'.json'),results)
    lines=['# Matched-arrival serving comparison','',
           'Power includes idle and drain. Compare total energy only for matched completed work.',
           'This is not the original EnergyInfra online-controller reproduction.','',
           '| Strategy | Repeats | Completed/offered | SLO pass % | W | Total J | J/token | Energy vs Dynamic % | Energy vs max_clock % |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def number(v):
        return '-' if v is None else f'{v:.3f}'
    for r in results:
        lines.append('| '+' | '.join([r['strategy'],str(r['repeats']),f"{r['completed']}/{r['offered']}",
                     number(r['attainment']*100),number(r['avg_power_w']),number(r['total_energy_j']),
                     number(r['energy_per_token_j']),number(r.get('total_energy_delta_vs_dynamic_pct')),
                     number(r.get('total_energy_delta_vs_max_clock_pct'))])+' |')
    (out/(name+'.md')).write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','run','report'])
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--repair',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--trace',choices=['short_chat','long_generation','bursty_mixed'],default='short_chat')
    ap.add_argument('--slo',type=int,choices=[50,100,250],default=50)
    ap.add_argument('--scenario',choices=['tpot_only','interactive_1s'],default='interactive_1s')
    args=ap.parse_args()
    out=args.run_dir.resolve()
    # Lock beside the target, so prepare can require an empty target directory.
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_name(out.name+'.orchestrator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg=prepare(args)
        if args.stage=='prepare':
            print(f'Prepared immutable serving evidence in {out}',flush=True)
            return
        if args.stage=='report':
            summarize(out,args.trace,args.slo,args.scenario)
            return
        # Use the same original trace generator and enforce eligibility up front.
        from src.experiments.run_serving_benchmark import RequestGenerator
        comparison=v2.report(out,cfg,'full')
        workloads=set()
        for repeat in range(cfg['serving']['repeats']):
            for r in RequestGenerator(args.trace,cfg['serving']['duration_s'],cfg['seed']+repeat).generate_requests():
                workloads.add((r['prompt_length'],r['output_length']))
        for strategy in ['min_power','min_energy']:
            for w in workloads:
                if not any(tuple(r['workload'])==w and r['scenario']==args.scenario and r['slo_ms']==args.slo and
                           r['strategy']==strategy and r['status']=='pass' for r in comparison):
                    raise RuntimeError(f'No confirmed feasible {strategy} for {w}; refusing to relax SLO')
        strategies=['dynamic','max_clock','min_power','min_energy']
        random.Random(cfg['seed']+4000).shuffle(strategies)
        schedule=dict(trace=args.trace,slo_ms=args.slo,scenario=args.scenario,strategies=strategies,
                      repeats_per_strategy=cfg['serving']['repeats'],
                      limitation='strategy-level random order; repeats are not interleaved')
        v2.atomic(out/f'schedule_{args.trace}_{args.slo}_{args.scenario}.json',schedule)
        child=None
        import signal
        def stop(signum,frame):
            if child is not None and child.poll() is None:
                child.terminate()
                child.wait()
            raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        for strategy in strategies:
            print(f'Serving start: {strategy} {args.trace} SLO={args.slo} {args.scenario}',flush=True)
            command=[sys.executable,'-u',str(Path(v2.__file__).resolve()),'serve','--run-dir',str(out),
                     '--config',str(out/'serving_config.json'),'--cohort','full','--strategy',strategy,
                     '--trace',args.trace,'--slo',str(args.slo),'--scenario',args.scenario]
            v2.atomic(out/'serving_config.json',cfg)
            child=subprocess.Popen(command,cwd=v2.ROOT)
            code=child.wait()
            if code:
                raise RuntimeError(f'{strategy} exited {code}; no later strategy was launched')
            summarize(out,args.trace,args.slo,args.scenario)
        print(f'Serving comparison complete: {out}',flush=True)


if __name__=='__main__':
    main()
