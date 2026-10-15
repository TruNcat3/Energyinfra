# Thor persistent joint serving Stage 1B v2

## Why v2 is required

The completed v1 full campaign is not valid for comparison. The pinned llama-server routed raw
`/completion` output through its fallback Content-only PEG parser. Random-weight model output can be
valid generated text while failing that parser, which terminated streams before their requested token
count. v2 isolates the fix in a separate binary and never modifies the official Stage 1A runtime.

The v2 runner accepts fixed work only when it receives the final stop frame, `stop_type=limit`, and the
exact requested `tokens_predicted` count. Token ID 0 is suppressed because it is the random model's sole
empty-text vocabulary entry. Under concurrency, llama-server may still coalesce or omit a small number of
Content-only text deltas even though its final generated-token count is exact, so SSE coverage is recorded
separately and must remain at least 98%. TPOT spans the first visible token through the final stop frame and
uses the generated-token count as its denominator. One failure aborts the trial before a result file is
written. Existing failed or v1 files are never resumed or included.

## Build the isolated runtime

Run without sudo:

```bash
cd /home/wt/energyinfra/Energyinfra-main
bash scripts/active/run_thor_stage1b_runtime_prepare_nohup.sh
```

The build is seeded from the validated official `4df29be` build, then only the patched common parser
and dependent server targets are rebuilt in an isolated directory. The source copy, binary, patch hash,
source hash, base commit, seed build, and toolchain are recorded in
`data/protocol_v2/thor_joint_serving_stage1b_runtime_v1/runtime_manifest.json`.

## Mandatory quick gate

After the runtime build completes:

```bash
bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh prepare quick
sudo -E bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh run quick
```

Only continue if all eight quick jobs finish without the fixed-work integrity gate firing.

## Full campaign

```bash
bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh prepare full
sudo -E bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh run full
```

The full v2 campaign contains 72 trials: three load profiles, three frozen repeats, and eight strategies.
Outputs are written under `data/protocol_v2/thor_joint_serving_stage1b_v2_coverage`; earlier directories
are retained only as diagnostic evidence.

## Recovery

```bash
sudo -E bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh restore quick
sudo -E bash scripts/active/run_thor_joint_serving_stage1b_nohup.sh restore full
```

Use only the command matching the interrupted preset.
