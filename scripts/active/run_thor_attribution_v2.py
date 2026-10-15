#!/usr/bin/env python3
"""Small model-shape ablation and isolated original-controller replay.

No legacy data is overwritten. This does not rerun the full static search.
"""
from collections import Counter,defaultdict
from dataclasses import asdict
import argparse
import copy
import fcntl
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import statistics
import threading
import time
from types import SimpleNamespace

import run_energyinfra_v2 as v2
from build_paired_random_qwen import build_pair


def fixed_update(self,**kw):
    """Explicit variant: preserve power streak when thermal opportunity is false.

    Original priority, cooldown, thermal opportunity and step logic are retained.
    This method is used only by the named offline replay variant, never imported
    into an existing original-policy run.
    """
    import src.controller.thermal_slo_controller as original
    now=original.time.time()
    self._window_id+=1
    temp=kw['current_temperature_c']
    p50=kw.get('tpot_p50_ms',0)
    p95=kw.get('tpot_p95_ms',0)
    count=kw.get('total_requests',0)
    violations=kw.get('slo_violation_count',0)
    rate=violations/count if count else 0
    state=original.WindowState(window_id=self._window_id,timestamp=now,current_cap_mhz=self._current_cap_mhz,
        prev_cap_mhz=self._current_cap_mhz,temperature_c=temp,
        temperature_slope_c_per_s=kw.get('temperature_slope_c_per_s',0),
        temperature_max_c=max(temp,kw.get('temperature_max_c',0)),tpot_p50_ms=p50,tpot_p95_ms=p95,
        tpot_p99_ms=kw.get('tpot_p99_ms',0),tokens_per_second=kw.get('tokens_per_second',0),
        slo_violation_count=violations,slo_violation_rate=rate,total_requests=max(count,kw.get('n_requests',0)),
        avg_power_w=kw.get('avg_power_w',0),max_power_w=kw.get('max_power_w',0),
        consecutive_lower_windows=self._consecutive_lower_windows,last_cap_change_time=self._last_cap_change_time,
        cap_switch_count=self._cap_switch_count,prompt_length=kw.get('prompt_length',0),output_length=kw.get('output_length',0))
    c=self.config
    slack=(c.tpot_slo_ms-p50)/c.tpot_slo_ms
    if rate>c.slo_violation_rate_threshold or p95>c.slo_tpot_p95_ratio*c.tpot_slo_ms:
        cap=self._step_cap(self._current_cap_mhz,'raise')
        if cap!=self._current_cap_mhz:
            state.cap_action=original.CapAction.RAISE
            self._apply_cap_change(cap,now,state)
            return state
    if temp>=c.thermal_critical_c and self._current_cap_mhz!=c.gpu_caps[-1]:
        state.cap_action=original.CapAction.RAISE_MAX
        self._apply_cap_change(c.gpu_caps[-1],now,state)
        return state
    if now-self._last_cap_change_time<c.cooldown_s:
        self._window_history.append(state)
        return state
    thermal_ok=temp>=c.thermal_warning_c and slack>=.1 and self._current_cap_mhz>c.gpu_caps[0]
    power_ok=kw.get('avg_power_w',0)>c.power_budget_w and slack>=.1 and self._current_cap_mhz>c.gpu_caps[0]
    if thermal_ok:
        self._consecutive_lower_windows+=1
        if self._consecutive_lower_windows>=c.lower_hysteresis_k:
            state.cap_action=original.CapAction.LOWER
            self._apply_cap_change(self._step_cap(self._current_cap_mhz,'lower'),now,state)
        else:
            state.consecutive_lower_windows=self._consecutive_lower_windows
            self._window_history.append(state)
        return state
    # Difference from original: do not erase an eligible power-only streak.
    if not power_ok:
        self._consecutive_lower_windows=0
    if power_ok:
        self._consecutive_lower_windows+=1
        if self._consecutive_lower_windows>=c.lower_hysteresis_k:
            state.cap_action=original.CapAction.LOWER
            self._apply_cap_change(self._step_cap(self._current_cap_mhz,'lower'),now,state)
        else:
            state.consecutive_lower_windows=self._consecutive_lower_windows
            self._window_history.append(state)
        return state
    self._window_history.append(state)
    return state


def replay(out,cfg):
    import src.controller.thermal_slo_controller as original
    class PowerHysteresisFixed(original.ThermalSLOCapController):
        update_cap=fixed_update
    gpu=[v//1000000 for v in cfg['platform']['gpu_hz']]
    config=original.ControllerConfig(gpu_caps=gpu,tpot_slo_ms=50,power_budget_w=50)
    scenarios={
      'power_only': [dict(temp=60,p50=30,p95=32,power=70) for _ in range(6)],
      'cool_idle': [dict(temp=60,p50=30,p95=32,power=35) for _ in range(6)],
      'slo_pressure': [dict(temp=60,p50=49,p95=55,power=70) for _ in range(6)],
      'thermal_warning': [dict(temp=77,p50=30,p95=32,power=70) for _ in range(6)],
      'thermal_critical': [dict(temp=86,p50=30,p95=32,power=70) for _ in range(3)],
      'broken_power_streak': [dict(temp=60,p50=30,p95=32,power=p) for p in [70,70,35,70,70,70]]}
    # Missing explicit paths mean no table, not the legacy empty-string cwd bug.
    missing=out/'reserved_missing_table.parquet'
    if missing.exists() or missing.is_symlink():
        raise RuntimeError('Reserved missing-table path must not exist')
    saved_time=original.time
    results=[]
    try:
        for name,windows in scenarios.items():
            for label,cls in [('original',original.ThermalSLOCapController),('power_hysteresis_fixed',PowerHysteresisFixed)]:
                clock=SimpleNamespace(value=0.0)
                original.time=SimpleNamespace(time=lambda:clock.value)
                controller=cls(str(missing),str(missing),'offline-synthetic',None,config=copy.deepcopy(config))
                controller._current_cap_mhz=1359
                actions=[]
                for i,w in enumerate(windows):
                    clock.value=60+10*i
                    state=controller.update_cap(current_temperature_c=w['temp'],tpot_p50_ms=w['p50'],tpot_p95_ms=w['p95'],
                          avg_power_w=w['power'],total_requests=10,slo_violation_count=2 if w['p95']>50 else 0)
                    data=asdict(state)
                    data['cap_action']=state.cap_action.value
                    actions.append(dict(input=w,state=data))
                results.append(dict(scenario=name,variant=label,windows=actions))
    finally:
        original.time=saved_time
    v2.atomic(out/'controller_replay.json',dict(scope='synthetic open-loop controller inputs, NOT measured performance/energy',
                    original_source_sha256=v2.file_hash(v2.ROOT/'src/controller/thermal_slo_controller.py'),cases=results))
    print(json.dumps([dict(scenario=r['scenario'],variant=r['variant'],actions=[w['state']['cap_action'] for w in r['windows']]) for r in results],indent=2))


class ObservedClocks:
    def __init__(self,hw):
        self.hw=hw
        self.samples=[]
        self.done=threading.Event()
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def loop(self):
        while not self.done.is_set():
            p=self.hw.paths
            self.samples.append(dict(t=time.monotonic(),gpu_mhz=p.gpu_current_mhz(),emc_mhz=p.emc_current_mhz(),
                                     first_cpu_policy_mhz=p.cpu_current_mhz(),temperature_c=p.temperature_c()))
            self.done.wait(.25)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self,*unused):
        self.done.set()
        self.thread.join()


def report(out):
    grouped=defaultdict(list)
    for r in v2.rows(out,'model_probe'):
        grouped[(r['vocab'],tuple(r['workload']),v2.config_id(r['hardware']))].append(r)
    summary=[]
    for (vocab,workload,cid),rs in sorted(grouped.items()):
        energy=sum(r['total_energy_j'] for r in rs)
        duration=sum(r['measured_duration_s'] for r in rs)
        tokens=sum(r['output_tokens'] for r in rs)
        requests=[q for r in rs for q in r['requests']]
        summary.append(dict(vocab=vocab,workload=list(workload),config_id=cid,blocks=len(rs),
                      ttft_ms=statistics.median(q['ttft_ms'] for q in requests),
                      tpot_ms=statistics.median(q['tpot_ms'] for q in requests),avg_power_w=energy/duration,
                      energy_per_token_j=energy/tokens,complete=len(rs)==3))
    v2.atomic(out/'model_comparison.json',summary)
    lines=['# Matched random model-shape probe','',
           'Only complete groups have three blocks. Do not mix these models with historical weights.',
           'Both tokenizers are synthetic; generated token IDs, not textual quality, define the workload.','',
           '| Vocabulary | Workload | Hardware | Blocks | TTFT ms | TPOT ms | W | J/token |',
           '|---:|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['vocab']} | {r['workload']} | {r['config_id']} | {r['blocks']} | {r['ttft_ms']:.3f} | {r['tpot_ms']:.3f} | {r['avg_power_w']:.3f} | {r['energy_per_token_j']:.3f} |")
    (out/'MODEL_RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['build','controller-replay','model-probe','report','restore'])
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--run-dir',type=Path,required=True)
    args=ap.parse_args()
    parent,out=args.parent.resolve(),args.run_dir.resolve()
    if out==parent:
        raise RuntimeError('Use an independent attribution directory')
    cfg=json.loads((parent/'protocol.json').read_text())['config']
    out.mkdir(parents=True,exist_ok=True)
    with (out/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(parent=str(parent),config=cfg,source=v2.file_hash(__file__),
                      builder=v2.file_hash(Path(__file__).with_name('build_paired_random_qwen.py')),
                      measurement=v2.file_hash(v2.__file__),
                      controller=v2.file_hash(v2.ROOT/'src/controller/thermal_slo_controller.py'))
        manifest=out/'protocol.json'
        if manifest.exists() and json.loads(manifest.read_text())!=identity:
            raise RuntimeError('Attribution protocol changed; use a new run directory')
        v2.atomic(manifest,identity)
        if args.stage=='build':
            v2.atomic(out/'models.json',build_pair(out/'models'))
            return
        if args.stage=='controller-replay':
            replay(out,cfg)
            return
        if args.stage=='report':
            report(out)
            return
        if os.geteuid()!=0:
            raise RuntimeError('Hardware probes require foreground sudo')
        with open('/run/energyinfra-v2-hardware.lock','a') as hl:
            fcntl.flock(hl,fcntl.LOCK_EX|fcntl.LOCK_NB)
            hw=v2.Hardware(cfg)
            pending=out/'restore_pending.json'
            snapshot_file=out/'hardware_snapshot.json'
            if args.stage=='restore':
                hw.restore(json.loads(snapshot_file.read_text()))
                v2.atomic(pending,dict(pending=False))
                return
            if pending.exists() and json.loads(pending.read_text())['pending']:
                raise RuntimeError('Hardware restore pending')
            models=json.loads((out/'models.json').read_text())
            fingerprints={v:v2.file_hash(p) for v,p in models.items()}
            environment=dict(models=fingerprints,llama_cpp_python=importlib.metadata.version('llama-cpp-python'),
                             platform=hw.paths.describe(),l4t=Path('/etc/nv_tegra_release').read_text())
            env=out/'environment.json'
            if env.exists() and json.loads(env.read_text())!=environment:
                raise RuntimeError('Models or environment changed')
            v2.atomic(env,environment)
            caps=[dict(mode='dynamic'),dict(mode='max_clock')]
            caps.extend(dict(mode='cap',gpu_hz=g,emc_hz=e,cpu_khz=1458000) for g,e in
                        [(738000000,4266000000),(1485000000,4266000000),(1485000000,2750000000)])
            workloads=[[64,64],[512,512],[2048,128]]
            snapshot=hw.state()
            v2.atomic(snapshot_file,snapshot)
            v2.atomic(pending,dict(pending=True))
            (out/'model_probe').mkdir(exist_ok=True)
            def stop(signum,frame):
                raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM,stop)
            signal.signal(signal.SIGINT,stop)
            engine=None
            power=None
            completed=0
            try:
                for repeat in range(3):
                    model_order=[32768,152064]
                    random.Random(cfg['seed']+6000+repeat).shuffle(model_order)
                    for vocab in model_order:
                        tasks=[dict(vocab=vocab,workload=w,hardware=c,repeat=repeat) for w in workloads for c in caps]
                        random.Random(cfg['seed']+repeat).shuffle(tasks)
                        todo=[t for t in tasks if not (out/'model_probe'/(v2.digest(t)+'.json')).exists()]
                        completed+=len(tasks)-len(todo)
                        if not todo:
                            continue
                        hw.apply(dict(mode='dynamic'))
                        local=copy.deepcopy(cfg)
                        local['model']['path']=models[str(vocab)]
                        engine=v2.Tokens(local)
                        power=v2.Power(out,local)
                        power.wait_after(time.monotonic())
                        for task in todo:
                            hw.apply(task['hardware'])
                            time.sleep(local['measurement']['settle_s'])
                            with ObservedClocks(hw) as observed:
                                result=v2.measure(engine,power,hw,local,task['workload'])
                            v2.atomic(out/'model_probe'/(v2.digest(task)+'.json'),dict(task=task,vocab=vocab,
                                      hardware=task['hardware'],observed_clocks=observed.samples,**result))
                            completed+=1
                            v2.atomic(out/'progress.json',dict(completed=completed,total=90,task=task))
                            print(json.dumps(dict(completed=completed,total=90,task=task)),flush=True)
                        power.close()
                        power=None
                        engine.model.close()
                        engine=None
                        gc.collect()
                report(out)
            except BaseException as exc:
                v2.atomic(out/'last_failure.json',dict(error=repr(exc)))
                raise
            finally:
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


if __name__=='__main__':
    main()
