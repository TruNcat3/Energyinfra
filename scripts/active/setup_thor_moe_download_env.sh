#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$ROOT/.venv-hf"

python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade \
  'pip==26.2.1' \
  'huggingface_hub==2.2.0' \
  'hf-xet==1.7.0'

printf 'Hugging Face Xet environment ready: %s\n' "$VENV"

