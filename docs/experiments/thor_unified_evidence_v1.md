# Thor unified evidence aggregation v1

## Purpose

This is the first P0 close-out step for Thor. It reads completed protocol
summaries and produces one detailed evidence table without merging scenarios
that have different arrival processes, SLO contracts or measurement horizons.
It performs no inference, model loading or hardware control.

The report is deliberately different from a global leaderboard:

- serial think-time and bursty low-concurrency remain separate;
- queue-aware results remain separate from single-user edge results;
- phase static validation is not treated as phase-switch serving;
- precision and model-scale pilots remain within-runtime sensitivity evidence;
- missing inputs are reported as missing rather than replaced by another platform.

## Run on the Thor server

Run as the normal user after the protocol directories are complete:

```bash
cd /home/wt/energyinfra/Energyinfra-main
python3 scripts/active/analyze_thor_unified_evidence_v1.py \
  --data-root data/protocol_v2 \
  --out-dir reports/thor_unified_evidence_v1
```

If the analysis script is being run from a separate source checkout, point
`--data-root` at the experiment machine's absolute `data/protocol_v2` path.
The input directories are never modified.

## Outputs

```text
reports/thor_unified_evidence_v1/RESULTS.md
reports/thor_unified_evidence_v1/evidence_rows.csv
reports/thor_unified_evidence_v1/evidence_sources.json
```

`evidence_rows.csv` is the complete scenario-scoped table. `RESULTS.md`
contains feasible anchors and baseline-oriented queue summaries. The source
manifest records row counts and SHA256 for every input summary, including
missing files.

## Interpretation

An edge anchor is considered feasible only when the source protocol's declared
all-repeat SLO flag is true. For queue-aware rows, the report preserves the
protocol's joint SLO attainment and every-repeat flag. Static validation and
pilot rows do not receive an inferred SLO result from another protocol.

The output is a close-out index, not a new experiment and not a replacement
for the protocol-specific `RESULTS.md` files. Any claim of cross-scenario
benefit must still use the same model, runtime, trace and SLO.
