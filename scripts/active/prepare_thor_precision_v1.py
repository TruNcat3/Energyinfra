#!/usr/bin/env python3
"""Build Q5_K_M and Q8_0 7B GGUFs from a deterministic full-shape F16 source."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

import gguf
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_thor_qwen14b_random import (  # noqa: E402
    atomic,
    build_f16,
    quantize_worker,
    sha256,
    tensor_specs,
)

CONFIG = ROOT / "configs/models/qwen2.5_7b_random_full_v1.json"


def file_type(filename):
    field = gguf.GGUFReader(str(filename)).fields.get("general.file_type")
    if field is None:
        raise RuntimeError(f"Missing general.file_type in {filename}")
    return int(field.contents())


def expected_file_type(precision):
    from llama_cpp import llama_cpp as api
    values = {
        "Q4_K_M": api.LLAMA_FTYPE_MOSTLY_Q4_K_M,
        "Q5_K_M": api.LLAMA_FTYPE_MOSTLY_Q5_K_M,
        "Q8_0": api.LLAMA_FTYPE_MOSTLY_Q8_0,
    }
    return int(values[precision])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    config = json.loads(CONFIG.read_text())
    architecture = config["architecture"]
    reference = ROOT / config["tokenizer_reference"]
    output_dir = ROOT / "models/precision_v1"
    f16 = ROOT / config["f16_output"]
    manifest_path = output_dir / "Qwen2.5-7B-random-v152064-precision.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not reference.is_file():
        raise FileNotFoundError(reference)
    specs = tensor_specs(architecture)
    plan = {"config_sha256": sha256(CONFIG), "builder_sha256": sha256(Path(__file__)),
            "reference_sha256": sha256(reference), "formats": config["quantization"],
            "threads": args.threads}
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if old is not None and old["plan"] != plan:
        if old.get("stage") not in ("building", "ready"):
            raise RuntimeError("Precision provenance changed; use a new output directory")
        # The output paths are immutable and are checked below. Only the
        # bookkeeping hash changed after a retry-safe script fix.
        manifest = {"plan": plan, "stage": "building"}
    else:
        manifest = old if old is not None else {"plan": plan, "stage": "building"}
    atomic(manifest_path, manifest)
    f16_partial = f16.with_name(f16.name + ".partial")
    f16_bytes = sum(math.prod(shape) * np.dtype(dtype).itemsize for _, shape, dtype, _ in specs)
    required = (0 if f16.exists() else f16_bytes) + sum(f16_bytes * 0.5 for _ in config["outputs"]) + 2 * 1024**3
    free = shutil.disk_usage(output_dir).free
    print(f"Free disk: {free / 1024**3:.1f} GiB; conservative additional requirement: {required / 1024**3:.1f} GiB", flush=True)
    if free < required:
        raise RuntimeError("Insufficient free disk for F16 and precision outputs")
    if not f16.exists():
        print("Generating deterministic 7B F16 source...", flush=True)
        build_f16(f16_partial, config, reference, specs, output_dir / "precision_progress.json")
        os.replace(f16_partial, f16)
        manifest.update(stage="f16_ready", f16_bytes=f16.stat().st_size,
                        parameter_count=sum(math.prod(shape) for _, shape, _, _ in specs), tensor_count=len(specs))
        atomic(manifest_path, manifest)
    elif f16.stat().st_size != manifest.get("f16_bytes", f16.stat().st_size):
        raise RuntimeError("F16 source size changed")
    for precision, target in config["outputs"].items():
        destination = ROOT / target
        if destination.exists():
            if destination.stat().st_size == 0:
                raise RuntimeError(f"Existing precision output is empty: {destination}")
            actual = file_type(destination)
            expected = expected_file_type(precision)
            if actual != expected:
                raise RuntimeError(
                    f"Existing {precision} output has GGUF file type {actual}, expected {expected}: {destination}"
                )
            manifest.setdefault("outputs", {})[precision] = {
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
            atomic(manifest_path, manifest)
            print(f"Keeping existing {precision}: {destination}", flush=True)
            continue
        partial = destination.with_name(destination.name + ".partial")
        print(f"Quantizing {precision}...", flush=True)
        quantize_worker(f16, partial, args.threads, precision)
        if not partial.is_file() or partial.stat().st_size == 0:
            raise RuntimeError(f"Quantizer did not produce {partial}")
        os.replace(partial, destination)
        manifest.setdefault("outputs", {})[precision] = {"path": str(destination), "bytes": destination.stat().st_size, "sha256": sha256(destination)}
        atomic(manifest_path, manifest)
    manifest.update(stage="ready", f16_retained=True)
    atomic(manifest_path, manifest)
    atomic(output_dir / "precision_progress.json", {"stage": "ready", "outputs": manifest["outputs"]})
    print(json.dumps({"stage": "ready", "f16": str(f16), "outputs": manifest["outputs"], "inference_tested": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
