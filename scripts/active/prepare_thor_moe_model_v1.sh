#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR="${1:-${ENERGYINFRA_MOE_MODEL_DIR:-$ROOT/models/moe_v1}}"
NAME="Qwen3-30B-A3B-Q4_K_M.gguf"
URL="https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/e4d4bafdfb96a411a163846265362aceb0b9c63a/$NAME"
REVISION="e4d4bafdfb96a411a163846265362aceb0b9c63a"
EXPECTED_SIZE=18556685824
EXPECTED_SHA256="0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5"
RESERVE_BYTES=$((8 * 1024 * 1024 * 1024))

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
FINAL="$OUTPUT_DIR/$NAME"
PART="$FINAL.part"
MANIFEST="$OUTPUT_DIR/Qwen3-30B-A3B-Q4_K_M.manifest.json"

verify_file() {
  local path="$1"
  local actual_size actual_sha
  actual_size="$(stat -c %s "$path")"
  [[ "$actual_size" -eq "$EXPECTED_SIZE" ]] || return 1
  actual_sha="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual_sha" == "$EXPECTED_SHA256" ]]
}

write_manifest() {
  python3 - "$FINAL" "$MANIFEST" "$URL" "$REVISION" "$EXPECTED_SHA256" <<'PY'
import datetime
import json
import pathlib
import sys

model, manifest, url, revision, digest = sys.argv[1:]
path = pathlib.Path(model)
payload = {
    "schema": "energyinfra-thor-moe-model-v1",
    "model": str(path),
    "bytes": path.stat().st_size,
    "sha256": digest,
    "source_url": url,
    "source_revision": revision,
    "prepared_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "hardware_frequency_changes": False,
}
target = pathlib.Path(manifest)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(target)
print(json.dumps(payload, sort_keys=True))
PY
}

if [[ -f "$FINAL" ]]; then
  echo "Existing model found; verifying size and SHA256: $FINAL"
  if verify_file "$FINAL"; then
    write_manifest
    echo "Model already ready: $FINAL"
    exit 0
  fi
  echo "Existing final model failed integrity validation; refusing to overwrite it." >&2
  exit 1
fi

partial_size=0
if [[ -f "$PART" ]]; then
  partial_size="$(stat -c %s "$PART")"
  if (( partial_size > EXPECTED_SIZE )); then
    echo "Partial file is larger than expected; remove it manually: $PART" >&2
    exit 1
  fi
fi

remaining=$((EXPECTED_SIZE - partial_size))
available="$(df -PB1 "$OUTPUT_DIR" | awk 'NR == 2 {print $4}')"
required=$((remaining + RESERVE_BYTES))
if (( available < required )); then
  printf 'Insufficient free space: available=%s required=%s output=%s\n' \
    "$available" "$required" "$OUTPUT_DIR" >&2
  exit 1
fi

printf 'Downloading %s bytes (%s already present) to %s\n' \
  "$EXPECTED_SIZE" "$partial_size" "$PART"
HF_CLI="$ROOT/.venv-hf/bin/hf"
if [[ -x "$HF_CLI" ]]; then
  echo "Using Hugging Face Xet downloader with adaptive concurrency."
  if [[ -f "$PART" ]]; then
    echo "Preserving legacy partial download until Xet output passes integrity validation: $PART"
  fi
  HF_HOME="$OUTPUT_DIR/.hf-home" \
  HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}" \
    "$HF_CLI" download \
      Qwen/Qwen3-30B-A3B-GGUF \
      "$NAME" \
      --revision "$REVISION" \
      --local-dir "$OUTPUT_DIR"

  if ! verify_file "$FINAL"; then
    echo "Xet output failed size or SHA256 validation: $FINAL" >&2
    exit 1
  fi
  rm -f "$PART"
  rm -rf "$OUTPUT_DIR/.hf-home" "$OUTPUT_DIR/.cache/huggingface"
  write_manifest
  echo "Model preparation complete: $FINAL"
  exit 0
fi

downloaded=false
for attempt in $(seq 1 100); do
  echo "Direct download attempt $attempt/100; requesting a fresh signed URL."
  if command -v wget >/dev/null 2>&1; then
    if wget \
      --continue \
      --timeout=60 \
      --tries=1 \
      --progress=dot:giga \
      --output-document="$PART" \
      "$URL"; then
      downloaded=true
      break
    fi
  elif curl \
    --http1.1 \
    --fail \
    --location \
    --continue-at - \
    --no-progress-meter \
    --output "$PART" \
    "$URL"; then
    downloaded=true
    break
  fi
  echo "Attempt $attempt failed; refreshing the signed URL in 5 seconds." >&2
  sleep 5
done

if [[ "$downloaded" != true ]]; then
  echo "Download failed after 100 fresh-URL attempts; preserving: $PART" >&2
  exit 1
fi

actual_size="$(stat -c %s "$PART")"
if [[ "$actual_size" -ne "$EXPECTED_SIZE" ]]; then
  echo "Downloaded size mismatch: actual=$actual_size expected=$EXPECTED_SIZE" >&2
  exit 1
fi

echo "Download complete; verifying SHA256."
if ! verify_file "$PART"; then
  echo "SHA256 validation failed; preserving partial file for inspection: $PART" >&2
  exit 1
fi

mv "$PART" "$FINAL"
write_manifest
echo "Model preparation complete: $FINAL"

