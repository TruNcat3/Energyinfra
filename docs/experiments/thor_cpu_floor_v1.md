# CPU-floor-aware characterization and serving extension

This keeps the EnergyInfra sequence: static characterization, per-workload
SLO-constrained multi-objective selection, disjoint validation, then serving.
It is explicitly a hardware-vector extension, NOT the unchanged original
GPU-only knee selector or thermal controller. Existing source/configurations,
models and historical results are not modified. It does not implement frequency
switching between prefill and decode.

## Declared hardware space

Load `configs/experiments/thor_cpu_floor_v1.json`. The referenced diagnostic
profile supplies Thor/L4T R38.2.2, Ubuntu/kernel guards and the full-vocabulary
paired random model. Other platforms need their own supported profile and
frequencies; no nearest-frequency substitution is allowed.

The grid crosses GPU caps 738/1116/1485/1575 MHz, EMC caps 2750/3200/4266 MHz,
and CPU floors minimum-supported/486/972 MHz, with a fixed CPU upper cap of
1458 MHz and dynamic CPU/GPU governors. Floor value 0 in configuration means
the minimum supported CPU frequency, NOT 0 MHz and not necessarily the vendor's
boot default. This minimum-floor control reproduces the existing Thor v2 CAP
settings. The changed floor is applied to every discovered CPU policy.

Also retain GPU 1485/1575 plus CPU 1458 locked references at EMC cap 4266,
Dynamic and MaxClock: 40 total configurations. EMC is never claimed to be
locked. The CPU upper cap is intentionally held at 1458 in the new grid; this
is not exhaustive search over all supported GPU/CPU frequencies or all upper
caps. Boundaries, governors, actual tegrastats clocks and temperatures are logged.

Presets:
- quick: 6 configurations, one 512/128 workload, one profile repeat, one
  validation repeat. A plumbing check, not evidence of optimality or gains.
- pilot: all 40 configurations, three representative workloads, three profile
  repeats = 360 profile blocks, then selected-pair validation.
- full: all 40 configurations and the original v2 twelve-workload list, three
  profile repeats = 1,440 profile blocks, then selected-pair validation.

The full grid includes every combination declared above, not just a few
hand-picked winning points. Each profile block warms up, then runs complete
requests for at least five seconds. Long output requests can take longer.
Validation uses 20 independent repeat blocks per selected hardware/workload
pair, plus both baselines. Every pair uses fixed measured request counts based
on output length. Validation size is determined only after static selection.

## Run the stages

Start with the short pipeline:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_cpu_floor_nohup.sh run \
  --preset quick --run-dir data/protocol_v2/thor_cpu_floor_quick_v1
```

After inspecting quick results, use a separate directory for the full pipeline:

```bash
sudo -E bash scripts/active/run_cpu_floor_nohup.sh run \
  --preset full --run-dir data/protocol_v2/thor_cpu_floor_full_v1
```

`run` executes profile -> select -> validate -> report, but NEVER starts serving
automatically. The full pipeline is a larger campaign; it is not another
162-point diagnostic. To review the static table before validation, run the
`profile`, `select`, `validate` stages separately with identical configuration,
preset and run directory. `plan` prints scope without changing hardware:

```bash
.venv-thor/bin/python scripts/active/run_cpu_floor_v1.py plan \
  --preset full --run-dir data/protocol_v2/thor_cpu_floor_full_v1
```

Launchers require foreground sudo so no password prompt blocks nohup. Logs and
the actual Python PID are printed. Resume with the identical command; committed
blocks/trials are skipped. Source, model, system, nvpmodel and configuration
fingerprints are fixed. Global hardware locking prevents concurrent experiments.
Use the printed SIGTERM command, not SIGKILL. In case of interrupted restoration:

```bash
sudo -E .venv-thor/bin/python scripts/active/run_cpu_floor_v1.py restore \
  --run-dir data/protocol_v2/thor_cpu_floor_full_v1
```

## Selection and independent validation

Every workload is solved independently. For each TPOT limit 50/75/100 ms and
TTFT scenario (TPOT-only, 1 s, 2 s), filter the static table by median latency
and compute the complete nondominated set over average power, J/output-token
and TPOT. From that set freeze the minimum-energy and minimum-power choices.
There is no weighted score or unlabelled substitution for the original knee
algorithm. Dynamic/MaxClock remain comparison baselines; unchanged minimum-floor
CAP and locked reference points may legitimately win the selection. The solver
does not force a new CPU-floor point to win.

Static feasibility is only a prediction. Independent validation never reranks
candidates or automatically falls back to another point. Failures remain in
`comparison.json/csv`. Primary energy evidence is min_energy at TPOT 50 ms,
TTFT 1 s against MaxClock across declared workloads, using Bonferroni-adjusted
paired-block bootstrap intervals. Secondary intervals are nominal 95%, without
multiple-comparison correction. Selection, raw tables and all Pareto points
are saved in `selection.json`; `static_table.csv` contains the whole grid.
`validation_summary.csv` and `RESULTS.md` show independent measurements.

Observed >=95% request attainment is not a future guarantee. Request Wilson
intervals assume independence and can be optimistic with within-block
correlation. A candidate failing TTFT 1 s is not silently moved to TTFT 2 s.
Energy/power improvements do not imply SLO success or a battery-runtime gain.

## Serving: matched arrivals, switching and idle energy

Only a full campaign can enter serving. First review validation outcomes, then:

```bash
sudo -E bash scripts/active/run_cpu_floor_nohup.sh serve \
  --preset full --run-dir data/protocol_v2/thor_cpu_floor_full_v1 \
  --trace short_chat --slo 50 --scenario interactive_1s
```

This reuses the original RequestGenerator and its unscaled arrival times, a
600-second horizon and three seeds. Every strategy receives the same requests.
There is no nearest-workload fallback. Optimized strategies are blocked before
trials if any trace workload lacks independently observed SLO feasibility.
This gate is not a guarantee against online queue violations. To study another
scenario, explicitly select `long_context_2s`, `tpot_only`, or another declared
TPOT limit; do not relabel the same run as passing a different protocol.

The five default strategies are:
- Dynamic: existing dynamic baseline throughout the trial.
- MaxClock: existing maximum-clock baseline, also held while idle.
- MaxClock-idle-dynamic: MaxClock during requests, Dynamic when queue is empty.
  This named extra control isolates idle-release savings from design-point tuning.
- min_energy: the validated per-workload minimum-energy choice, idle Dynamic.
- min_power: the validated per-workload minimum-power choice, idle Dynamic.

The floor remains active while processing queued requests, then is released
when the queue empties. A change in workload can select another GPU/EMC/floor
vector. Every measured switch is timestamped; request TTFT starts at scheduled
arrival, so queue waiting, configuration writes and request preparation are
included. No switch settling delay is hidden outside request timing. Idle
release is part of the measured energy window, including final release after
an overloaded queue drains. Model load and the initial warmup are excluded.

VIN energy is continuously integrated over the arrival horizon plus full queue
drain. A separate fixed-horizon energy measurement is retained. Queue overflow
counts as dropped/SLO-failed work; cross-strategy energy deltas are withheld if
the complete requested work does not match. Both aggregate attainment and
whether every repeat reached the observed target are reported. The approach
does not claim a continuous-batching server or calibrated battery measurements.

Results are under `serving/<trace>_<scenario>_<slo>/`, including trace hashes,
individual requests, switches, summaries and comparisons against Dynamic,
MaxClock and MaxClock-idle-dynamic. Any explicitly restricted `--strategies`
set is frozen in that serving protocol and must not be changed on resume.
Run other traces sequentially, not concurrently. The inherited long/bursty
arrival patterns may overload this engine; report their failures rather than
changing arrival rates until a strategy appears successful.

This change installs no packages and launches no benchmarks by itself. Existing
NSight diagnostic and frozen-candidate validation scripts remain available.
