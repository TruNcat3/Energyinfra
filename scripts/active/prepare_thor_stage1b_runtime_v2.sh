#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${THOR_STAGE1B_CONFIG:-$ROOT/configs/experiments/thor_joint_serving_stage1b_v2.json}"
PYTHON="$ROOT/.venv-thor/bin/python"
JOBS="${LLAMA_BUILD_JOBS:-4}"

if [[ "$EUID" -eq 0 ]]; then
  printf 'Do not build llama.cpp with sudo. Run this launcher as the workspace user.\n' >&2
  exit 2
fi
for command in cmake patch sha256sum; do
  command -v "$command" >/dev/null 2>&1 || { printf 'Missing build command: %s\n' "$command" >&2; exit 2; }
done
NVCC="${CUDACXX:-$(command -v nvcc 2>/dev/null || true)}"
if [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]]; then NVCC=/usr/local/cuda/bin/nvcc; fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
  printf 'Missing CUDA compiler. Set CUDACXX or install nvcc at /usr/local/cuda/bin/nvcc.\n' >&2
  exit 2
fi
export CUDACXX="$NVCC"
export CUDA_HOME="${CUDA_HOME:-$(cd -- "$(dirname -- "$NVCC")/.." && pwd)}"
export PATH="$(dirname -- "$NVCC"):$PATH"

readarray -t VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
config=Path(sys.argv[1]).resolve()
root=config.parents[2]
runtime=json.load(config.open())['runtime']
for key in ('base_commit','cuda_architectures','base_source_dir','source_dir','base_build_dir','build_dir','server_binary','patch','manifest'):
    value=runtime[key]
    print(value if key in ('base_commit','cuda_architectures') else root/value)
PY
)
BASE_COMMIT="${VALUES[0]}"
CUDA_ARCHITECTURES="${VALUES[1]}"
BASE_SOURCE="${VALUES[2]}"
SOURCE_DIR="${VALUES[3]}"
BASE_BUILD="${VALUES[4]}"
BUILD_DIR="${VALUES[5]}"
SERVER_BIN="${VALUES[6]}"
PATCH_FILE="${VALUES[7]}"
MANIFEST="${VALUES[8]}"
PATCH_SHA="$(sha256sum "$PATCH_FILE" | cut -d' ' -f1)"
PATCH_MARKER="$SOURCE_DIR/.energyinfra-stage1b-patch-sha256"
SEED_MARKER="$BUILD_DIR/.energyinfra-stage1b-seed"

if [[ ! -f "$BASE_SOURCE/CMakeLists.txt" ]]; then
  printf 'Missing validated base source: %s\n' "$BASE_SOURCE" >&2
  exit 2
fi
ACTUAL_COMMIT="$(cat "$BASE_SOURCE/.energyinfra-source-commit" 2>/dev/null || true)"
if [[ -z "$ACTUAL_COMMIT" && -d "$BASE_SOURCE/.git" ]]; then
  ACTUAL_COMMIT="$(git -C "$BASE_SOURCE" rev-parse HEAD)"
fi
if [[ "$ACTUAL_COMMIT" != "$BASE_COMMIT" ]]; then
  printf 'Base source commit mismatch: got %s, expected %s\n' "$ACTUAL_COMMIT" "$BASE_COMMIT" >&2
  exit 2
fi

if [[ ! -e "$SOURCE_DIR" ]]; then
  mkdir -p "$(dirname -- "$SOURCE_DIR")"
  cp -a --reflink=auto "$BASE_SOURCE" "$SOURCE_DIR"
fi
if [[ ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
  printf 'Invalid patched source directory: %s\n' "$SOURCE_DIR" >&2
  exit 2
fi
if [[ -f "$PATCH_MARKER" ]]; then
  if [[ "$(cat "$PATCH_MARKER")" != "$PATCH_SHA" ]]; then
    printf 'Patch provenance changed; use a new source/build directory.\n' >&2
    exit 2
  fi
else
  patch --directory "$SOURCE_DIR" --strip=1 --forward --batch < "$PATCH_FILE"
  printf '%s\n' "$PATCH_SHA" > "$PATCH_MARKER"
fi

if [[ ! -f "$BASE_BUILD/CMakeCache.txt" ]]; then
  printf 'Missing validated base build: %s\n' "$BASE_BUILD" >&2
  exit 2
fi
EXPECTED_SEED="$BASE_BUILD|$BASE_SOURCE|$SOURCE_DIR"
if [[ -f "$SEED_MARKER" ]]; then
  if [[ "$(cat "$SEED_MARKER")" != "$EXPECTED_SEED" ]]; then
    printf 'Build seed provenance changed; use a new build directory.\n' >&2
    exit 2
  fi
else
  mkdir -p "$BUILD_DIR"
  cp -a "$BASE_BUILD/." "$BUILD_DIR/"
  while IFS= read -r -d '' path; do
    sed -i -e "s|$BASE_SOURCE|$SOURCE_DIR|g" -e "s|$BASE_BUILD|$BUILD_DIR|g" "$path"
    relative="${path#"$BUILD_DIR"/}"
    touch --reference "$BASE_BUILD/$relative" "$path"
  done < <(rg -l0 --hidden --fixed-strings -e "$BASE_SOURCE" -e "$BASE_BUILD" "$BUILD_DIR" || true)
  printf '%s\n' "$EXPECTED_SEED" > "$SEED_MARKER"
fi

# Do not reconfigure this relocated build. Regeneration updates global flags.make files and
# needlessly invalidates every CUDA object. The validated seed already has the required sm_110
# configuration; Make will rebuild chat.cpp and relink only its dependent server targets.
cmake --build "$BUILD_DIR" --config Release --parallel "$JOBS" --target llama-server
if [[ ! -x "$SERVER_BIN" ]]; then
  printf 'Expected server binary was not produced: %s\n' "$SERVER_BIN" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$MANIFEST")"
"$PYTHON" - "$CONFIG" "$SOURCE_DIR" "$BUILD_DIR" "$SERVER_BIN" "$PATCH_FILE" "$MANIFEST" "$ACTUAL_COMMIT" <<'PY'
import hashlib, json, os, platform, subprocess, sys, tempfile, time
from pathlib import Path
config, source, build, server, patch, manifest=map(Path,sys.argv[1:7])
actual_commit=sys.argv[7]
cfg=json.load(config.open())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def run(*command):
    return subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True).stdout.strip()
value={
    'schema':'energyinfra-thor-stage1b-patched-runtime-v1',
    'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'config':str(config),'config_sha256':digest(config),
    'repository':cfg['runtime']['repository'],
    'base_commit':actual_commit,
    'patch':str(patch),'patch_sha256':digest(patch),
    'source_dir':str(source),'build_dir':str(build),
    'base_build_dir':str((config.parents[2]/cfg['runtime']['base_build_dir']).resolve()),
    'patched_chat_sha256':digest(source/'common/chat.cpp'),
    'server':{'path':str(server),'sha256':digest(server),'version':run(str(server),'--version')},
    'toolchain':{
        'cmake':run('cmake','--version').splitlines()[0],
        'nvcc':run('nvcc','--version').splitlines()[-1],
        'cuda_architectures':str(cfg['runtime']['cuda_architectures']),
        'machine':platform.machine(),'kernel':platform.release(),
    },
}
fd,temporary=tempfile.mkstemp(prefix=manifest.name+'.',dir=manifest.parent)
try:
    with os.fdopen(fd,'w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True); stream.write('\n')
    os.replace(temporary,manifest)
finally:
    if os.path.exists(temporary): os.unlink(temporary)
print(json.dumps({'server':str(server),'manifest':str(manifest),'patch_sha256':value['patch_sha256']},sort_keys=True))
PY
printf 'Patched Stage 1B runtime prepared. No model inference or hardware change was performed.\n'
