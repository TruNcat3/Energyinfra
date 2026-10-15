# Thor queue-aware Pareto exploration v1

## Objective

Replace static single-request feasibility with frozen-trace queue-aware
feasibility. Latency is a hard constraint. Total fixed-work energy and average
power are the two Pareto objectives.

This stage performs no hardware frequency changes. Its output is a predicted
shortlist that must be validated by the existing fixed-trace hardware protocol.

## Search space

The four long-generation workload classes retain every measured static
frontier candidate. The current Cartesian product contains approximately
1.04 million workload-specific policies.

Every policy is replayed over all three frozen `arrival_x5` traces. The model
includes FIFO queueing, p95 control latency, the 600-second arrival horizon,
queue drain, active energy and idle energy.

## Feasibility

A policy is predicted feasible only when every repeat reaches at least 95%
joint attainment for completion, TTFT, TPOT and derived end-to-end latency.

After feasibility filtering, the exact two-objective frontier minimizes:

- total fixed-work energy in joules;
- average power over arrival horizon plus drain.

The shortlist includes energy-minimum, power-minimum without energy regression,
the normalized knee, and epsilon-spaced frontier representatives.

## Calibration and validation

Energy-, power- and latency-weighted hardware runs calibrate the latency and
energy model. Balanced-weighted is excluded from fitting and used as a holdout.
The report records holdout error and never labels predicted points as hardware
validated.

## Run

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_queue_aware_pareto_nohup.sh
```

Outputs:

```text
data/protocol_v2/thor_queue_aware_pareto_v1/queue_aware_pareto.json
data/protocol_v2/thor_queue_aware_pareto_v1/hardware_validation_shortlist.json
data/protocol_v2/thor_queue_aware_pareto_v1/shortlist.csv
data/protocol_v2/thor_queue_aware_pareto_v1/protocol.json
```

The next hardware stage should run only unique shortlist mappings, paired on
the same frozen traces against Dynamic, MaxClock and MaxClockIdleDynamic.
