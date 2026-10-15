# Thor prompt-aware coupled trace v1

This experiment removes cross-rate workload-composition noise from the
prompt-aware SLO capacity study.

Within each repetition, every offered rate receives the same ten requests,
prompt seeds, workload order, and normalized inter-arrival sequence. The ten
requests exactly reproduce the configured 4:3:2:1 workload weights and contain
2432 output tokens. A rate changes only the time scale:

`trace duration = 2432 output tokens / offered output-token rate`.

The six rates are 18, 20, 22, 24, 26, and 28 output-token/s. Eight baseline and
joint strategies run over three repetitions, for 144 trials. Trace durations
range from about 86.9 to 135.1 seconds.

Prepare without root:

```bash
bash scripts/active/run_thor_prompt_aware_coupled_nohup.sh prepare
```

Run with hardware-control permission:

```bash
sudo -E bash scripts/active/run_thor_prompt_aware_coupled_nohup.sh run
```

After the hardware run, execute the prompt-aware offline analysis:

```bash
bash scripts/active/run_thor_prompt_aware_coupled_nohup.sh analyze
```
