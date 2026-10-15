#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${THOR_JOINT_CONFIG:-$ROOT/configs/experiments/thor_joint_resource_power_v1.json}"
PYTHON="$ROOT/.venv-thor/bin/python"
JOBS="${LLAMA_BUILD_JOBS:-4}"

if [[ "$EUID" -eq 0 ]]; then
  printf 'Do not build llama.cpp with sudo. Run this launcher as the workspace user.\n' >&2
  exit 2
fi

for command in git cmake sha256sum; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Missing required build command: %s\n' "$command" >&2
    exit 2
  fi
done
NVCC="${CUDACXX:-$(command -v nvcc 2>/dev/null || true)}"
if [[ -z "$NVCC" && -x /usr/local/cuda/bin/nvcc ]]; then
  NVCC=/usr/local/cuda/bin/nvcc
fi
if [[ -z "$NVCC" || ! -x "$NVCC" ]]; then
  printf 'Missing CUDA compiler. Expected nvcc on PATH or at /usr/local/cuda/bin/nvcc.\n' >&2
  exit 2
fi
CUDA_HOME="${CUDA_HOME:-$(cd -- "$(dirname -- "$NVCC")/.." && pwd)}"
export CUDA_HOME
export CUDACXX="$NVCC"
export PATH="$(dirname -- "$NVCC"):$PATH"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing project Python: %s\n' "$PYTHON" >&2
  exit 2
fi

readarray -t VALUES < <("$PYTHON" - "$CONFIG" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1]).resolve().parents[2]
cfg=json.load(open(sys.argv[1]))
runtime=cfg['runtime']
for value in (
    runtime['repository'], runtime['commit'],
    str(runtime['cuda_architectures']),
    str(root / runtime['source_dir']), str(root / runtime['build_dir']),
    str(root / runtime['server_binary']), str(root / runtime['batched_bench_binary']),
    str(root / cfg['output_dir']),
):
    print(value)
PY
)

REPOSITORY="${VALUES[0]}"
COMMIT="${VALUES[1]}"
CUDA_ARCHITECTURES="${VALUES[2]}"
SOURCE_DIR="${VALUES[3]}"
BUILD_DIR="${VALUES[4]}"
SERVER_BIN="${VALUES[5]}"
BATCHED_BENCH_BIN="${VALUES[6]}"
OUTPUT_DIR="${VALUES[7]}"
SOURCE_MARKER="$SOURCE_DIR/.energyinfra-source-commit"

printf 'Pinned llama.cpp commit: %s\n' "$COMMIT"
printf 'Source: %s\nBuild: %s\n' "$SOURCE_DIR" "$BUILD_DIR"
printf 'CUDA compiler: %s\n' "$NVCC"

if [[ -d "$SOURCE_DIR/.git" ]]; then
  CURRENT="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$CURRENT" && "$CURRENT" != "$COMMIT" ]]; then
    printf 'Existing source is at %s, expected %s. Refusing to modify it.\n' "$CURRENT" "$COMMIT" >&2
    exit 2
  fi
else
  if [[ -e "$SOURCE_DIR" && -n "$(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    MARKED_COMMIT="$(cat "$SOURCE_MARKER" 2>/dev/null || true)"
    if [[ "$MARKED_COMMIT" != "$COMMIT" || ! -f "$SOURCE_DIR/CMakeLists.txt" ]]; then
      printf 'Unverified non-git source directory: %s\n' "$SOURCE_DIR" >&2
      exit 2
    fi
  else
    mkdir -p "$SOURCE_DIR"
    git -C "$SOURCE_DIR" init
    git -C "$SOURCE_DIR" remote add origin "$REPOSITORY"
  fi
fi

if [[ -d "$SOURCE_DIR/.git" ]]; then
  CURRENT="$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$CURRENT" != "$COMMIT" ]]; then
    git -C "$SOURCE_DIR" fetch --depth 1 origin "$COMMIT"
    git -C "$SOURCE_DIR" checkout --detach FETCH_HEAD
  fi
  CURRENT="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
else
  CURRENT="$(cat "$SOURCE_MARKER")"
fi
if [[ "$CURRENT" != "$COMMIT" ]]; then
  printf 'Source checkout mismatch: got %s, expected %s\n' "$CURRENT" "$COMMIT" >&2
  exit 1
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON

cmake --build "$BUILD_DIR" --config Release --parallel "$JOBS" \
  --target llama-server llama-batched-bench

if [[ ! -x "$SERVER_BIN" || ! -x "$BATCHED_BENCH_BIN" ]]; then
  printf 'Expected binaries were not produced.\nserver=%s\nbatched_bench=%s\n' \
    "$SERVER_BIN" "$BATCHED_BENCH_BIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON" - "$CONFIG" "$SOURCE_DIR" "$BUILD_DIR" "$SERVER_BIN" "$BATCHED_BENCH_BIN" "$OUTPUT_DIR" "$CURRENT" <<'PY'
import hashlib, json, os, platform, subprocess, sys, tempfile, time
from pathlib import Path

config, source, build, server, bench, output = map(Path, sys.argv[1:7])
actual_commit=sys.argv[7]
cfg=json.load(config.open())

def run(*command):
    result=subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
    return result.stdout.strip()

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

manifest={
    'schema':'energyinfra-thor-llama-server-runtime-v1',
    'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'config':str(config),
    'config_sha256':digest(config),
    'repository':cfg['runtime']['repository'],
    'expected_commit':cfg['runtime']['commit'],
    'actual_commit':actual_commit,
    'source_kind':'git' if (source/'.git').is_dir() else 'verified_archive',
    'source_dir':str(source),
    'build_dir':str(build),
    'server':{
        'path':str(server), 'sha256':digest(server),
        'version':run(str(server),'--version'),
        'help_sha256':hashlib.sha256(run(str(server),'--help').encode()).hexdigest(),
    },
    'batched_bench':{
        'path':str(bench), 'sha256':digest(bench),
        'version':run(str(bench),'--version'),
    },
    'toolchain':{
        'cmake':run('cmake','--version').splitlines()[0],
        'nvcc':run('nvcc','--version').splitlines()[-1],
        'cuda_architectures':str(cfg['runtime']['cuda_architectures']),
        'machine':platform.machine(),
        'kernel':platform.release(),
    },
}
output.mkdir(parents=True, exist_ok=True)
target=output/'runtime_manifest.json'
fd, temporary=tempfile.mkstemp(prefix=target.name+'.', dir=output)
try:
    with os.fdopen(fd,'w') as stream:
        json.dump(manifest,stream,indent=2,sort_keys=True)
        stream.write('\n')
    os.replace(temporary,target)
finally:
    if os.path.exists(temporary): os.unlink(temporary)
print(json.dumps({'server':str(server),'batched_bench':str(bench),'manifest':str(target)},sort_keys=True))
PY

printf 'Runtime preparation complete. No model inference or hardware frequency changes were performed.\n'
