# Thor Q4 fixed-trace long-generation serving v1

This experiment consumes the three confirmed arrival-scale-5 traces without
regenerating or modifying them. Every strategy receives the same requests at the
same scheduled times. Strategy order is independently randomized per repeat.

The comparison includes Dynamic, MaxClock, MaxClock with dynamic idle release,
and four preference views of the frozen static Pareto sets. FIFO queueing, idle
energy, complete drain and measured hardware-switch overhead are included.

Run the complete comparison with:

```bash
cd /home/wt/energyinfra/Energyinfra-main
sudo -E bash scripts/active/run_thor_fixed_trace_serving_nohup.sh run
```

The run is resumable and takes about 3.5 hours because it contains 21 fixed
600-second trials. Its final report is:

```text
data/protocol_v2/thor_q4_long_generation_fixed_trace_v1/RESULTS.md
```

If an ungraceful stop leaves restoration pending, run:

```bash
sudo -E bash scripts/active/run_thor_fixed_trace_serving_nohup.sh restore
```

No single Pareto member is declared universally optimal. The report preserves
power, energy per token, TTFT and TPOT separately and marks non-dominated serving
outcomes only among strategies passing every repeat.
