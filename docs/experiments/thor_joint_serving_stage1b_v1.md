# Thor persistent joint serving Stage 1B

## Objective

Stage 1B crosses the Stage 1A resource knees with native dynamic, MaxClock, and the validated
QueuePowerMin hardware state. It measures request-level TTFT, TPOT, SLO attainment, fixed-horizon VIN
power, drain energy, and joules per completed output token using one persistent CUDA `llama-server`.

The experiment uses identical deterministic prompt-token arrays, requested output lengths, and frozen
arrival times for every strategy. Model loading and warmup are outside the energy window. EOS is ignored
so every successful request completes the same fixed output work.

## Baselines and related work

The comparison includes native dynamic and MaxClock controls, the EnergyInfra DVFS-only point,
continuous-batching-only points, and joint resource/power points. The continuous batching comparison is
mechanism-aligned with [Orca](https://www.usenix.org/conference/osdi22/presentation/yu). The 256/256
batch-microbatch candidates are informed by the Stage 1A result and are only a mechanism proxy for
[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal).

[vLLM](https://dl.acm.org/doi/10.1145/3600006.3613165) is listed qualitatively because this llama.cpp
runtime does not implement PagedAttention. Results must not be presented as native Orca, Sarathi, or
vLLM reproductions.

## Quick gate

Prepare frozen traces without sudo:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh prepare quick
```

Run after preparation completes:

```bash
sudo -E bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh run quick
```

Quick uses one 60-second saturation trace and eight strategies. It normally takes about 8-12 minutes.

## Full experiment

```bash
bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh prepare full
sudo -E bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh run full
```

Full uses light, serial-saturation, and batching-heavy loads with three frozen repeats. Results are
evaluated under 500/30 ms, 1000/50 ms, and 2000/100 ms TTFT/TPOT profiles. A strategy joins an SLO Pareto
set only if every repeat reaches at least 95% joint attainment.

## Recovery

The runner restores the original GPU, CPU, and EMC state in `finally`, including graceful SIGTERM. If
the process is killed ungracefully, restore with:

```bash
sudo -E bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh restore quick
```

Use `full` instead of `quick` for the full run directory.
