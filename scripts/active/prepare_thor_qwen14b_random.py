#!/usr/bin/env python3
"""Build full-shape synthetic Qwen2.5-14B without downloading weights.

Reuse only tokenizer metadata from the existing 7B GGUF. Generate one tensor
at a time, then quantize using the installed llama.cpp Python C bindings.
This command does not write hardware frequency controls or run inference.
"""

import argparse
import ctypes
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import gguf
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/models/qwen2.5_14b_random_full_v1.json"


def path(value):
    return (ROOT / value).resolve()


def sha256(filename):
    result = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def atomic(filename, value):
    temporary = filename.with_name(filename.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, filename)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def tensor_specs(arch):
    h = arch["hidden_size"]
    ff = arch["intermediate_size"]
    vocab = arch["vocab_size"]
    kv = h // arch["num_attention_heads"] * arch["num_key_value_heads"]
    result = [
        ("token_embd.weight", (vocab, h), np.float16, "random"),
        ("output_norm.weight", (h,), np.float32, "ones"),
        ("output.weight", (vocab, h), np.float16, "random"),
    ]
    for layer in range(arch["num_hidden_layers"]):
        prefix = f"blk.{layer}."
        for name, shape, dtype, initialization in [
            ("attn_norm.weight", (h,), np.float32, "ones"),
            ("attn_q.weight", (h, h), np.float16, "random"),
            ("attn_k.weight", (kv, h), np.float16, "random"),
            ("attn_v.weight", (kv, h), np.float16, "random"),
            ("attn_q.bias", (h,), np.float32, "zeros"),
            ("attn_k.bias", (kv,), np.float32, "zeros"),
            ("attn_v.bias", (kv,), np.float32, "zeros"),
            ("attn_output.weight", (h, h), np.float16, "random"),
            ("ffn_norm.weight", (h,), np.float32, "ones"),
            ("ffn_gate.weight", (ff, h), np.float16, "random"),
            ("ffn_up.weight", (ff, h), np.float16, "random"),
            ("ffn_down.weight", (h, ff), np.float16, "random"),
        ]:
            result.append((prefix + name, shape, dtype, initialization))
    return result


def build_f16(filename, config, reference, specs, progress):
    arch = config["architecture"]
    reader = gguf.GGUFReader(str(reference))
    tokens = reader.fields.get("tokenizer.ggml.tokens")
    if tokens is None or len(tokens.data) != arch["vocab_size"]:
        raise RuntimeError("The tokenizer reference must contain all 152064 tokens")
    writer = gguf.GGUFWriter(str(filename), "qwen2")
    writer.add_string("general.name", "EnergyInfra random Qwen2.5-14B full vocabulary")
    # gguf-python exposes GGUFValueType but not the llama.cpp file-type enum.
    # The numeric file type is part of llama.cpp's public C API instead.
    writer.add_uint32("general.file_type", int(gguf_file_type_f16()))
    for key, value in {
        "qwen2.context_length": arch["max_position_embeddings"],
        "qwen2.embedding_length": arch["hidden_size"],
        "qwen2.block_count": arch["num_hidden_layers"],
        "qwen2.feed_forward_length": arch["intermediate_size"],
        "qwen2.attention.head_count": arch["num_attention_heads"],
        "qwen2.attention.head_count_kv": arch["num_key_value_heads"],
        "qwen2.rope.dimension_count": arch["hidden_size"] // arch["num_attention_heads"],
    }.items():
        writer.add_uint32(key, value)
    writer.add_float32("qwen2.attention.layer_norm_rms_epsilon", arch["rms_norm_eps"])
    writer.add_float32("qwen2.rope.freq_base", arch["rope_theta"])
    tokenizer = {}
    for key, field in reader.fields.items():
        if key.startswith("tokenizer."):
            value = field.contents()
            subtype = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
            writer.add_key_value(key, value, field.types[0], subtype)
            tokenizer[key] = value
    tokenizer_hash = hashlib.sha256(
        json.dumps(tokenizer, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    for name, shape, dtype, _ in specs:
        writer.add_tensor_info(name, shape, np.dtype(dtype), math.prod(shape) * np.dtype(dtype).itemsize)
    started = time.monotonic()
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for index, (name, shape, dtype, initialization) in enumerate(specs, 1):
            print(f"tensor {index}/{len(specs)}: {name} {shape}", flush=True)
            atomic(progress, {"stage": "generate_f16", "tensor": name, "completed": index - 1,
                              "total": len(specs), "elapsed_s": time.monotonic() - started})
            if initialization == "ones":
                array = np.ones(shape, dtype=dtype)
            elif initialization == "zeros":
                array = np.zeros(shape, dtype=dtype)
            else:
                tensor_seed = int.from_bytes(
                    hashlib.sha256(f"{config['seed']}:{name}".encode("ascii")).digest()[:8], "little"
                )
                rng = np.random.default_rng(tensor_seed)
                array = np.empty(shape, dtype=dtype)
                for first in range(0, shape[0], 256):
                    count = min(256, shape[0] - first)
                    block = rng.standard_normal((count, *shape[1:]), dtype=np.float32)
                    block *= arch["initializer_range"]
                    array[first:first + count] = block
            writer.write_tensor_data(array)
            del array
    finally:
        writer.close()
    with filename.open("rb") as stream:
        os.fsync(stream.fileno())
    return tokenizer_hash


def gguf_file_type_f16():
    from llama_cpp import llama_cpp as api
    return api.LLAMA_FTYPE_MOSTLY_F16


def gguf_file_type_q4_k_m():
    from llama_cpp import llama_cpp as api
    return api.LLAMA_FTYPE_MOSTLY_Q4_K_M


def gguf_file_type(quantization):
    values = {
        "Q4_K_M": gguf_file_type_q4_k_m,
        "Q5_K_M": lambda: __import__("llama_cpp").llama_cpp.LLAMA_FTYPE_MOSTLY_Q5_K_M,
        "Q8_0": lambda: __import__("llama_cpp").llama_cpp.LLAMA_FTYPE_MOSTLY_Q8_0,
    }
    try:
        return values[quantization]()
    except KeyError as exc:
        raise ValueError(f"unsupported quantization: {quantization}") from exc


def quantize_worker(source, destination, threads, quantization="Q4_K_M"):
    from llama_cpp import llama_cpp as api

    params = api.llama_model_quantize_default_params()
    params.nthread = threads
    params.ftype = int(gguf_file_type(quantization))
    params.quantize_output_tensor = True
    params.allow_requantize = False
    params.pure = False
    api.llama_backend_init()
    try:
        result = api.llama_model_quantize(os.fsencode(source), os.fsencode(destination), ctypes.byref(params))
        if result != 0:
            raise RuntimeError(f"llama_model_quantize returned {result}")
    finally:
        api.llama_backend_free()


def prepare(args):
    import llama_cpp
    from llama_cpp import llama_cpp as api

    config = json.loads(path(args.config).read_text())
    arch = config["architecture"]
    if (arch["model_type"], arch["hidden_size"], arch["intermediate_size"],
        arch["num_hidden_layers"], arch["num_attention_heads"], arch["num_key_value_heads"],
        arch["vocab_size"], arch["tie_word_embeddings"], arch["use_sliding_window"]) != (
            "qwen2", 5120, 13824, 48, 40, 8, 152064, False, False):
        raise RuntimeError("This builder is restricted to the full Qwen2.5-14B dense architecture")
    if config["quantization"] != "Q4_K_M":
        raise RuntimeError("This experiment requires Q4_K_M")
    reference = path(config["tokenizer_reference"])
    output = path(config["output"])
    if output == reference:
        raise RuntimeError("Refusing to overwrite the 7B tokenizer reference")
    if not reference.is_file():
        raise FileNotFoundError(f"Existing full-vocabulary 7B model is required: {reference}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.parent / ".qwen14b_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = output.with_suffix(".json")
        progress = output.parent / "prepare_progress.json"
        f16 = output.with_name("Qwen2.5-14B-random-v152064-F16.gguf")
        temporary_f16 = f16.with_name(f16.name + ".partial")
        temporary_q4 = output.with_name(output.name + ".partial")
        print("Hashing the 7B tokenizer reference and installed quantizer for provenance...", flush=True)
        plan = {
            "config": config, "builder_sha256": sha256(__file__),
            "tokenizer_reference_sha256": sha256(reference),
            "numpy_version": np.__version__, "llama_cpp_version": llama_cpp.__version__,
            "quantizer_library_sha256": sha256(api._lib._name),
            "threads": args.threads,
            "rng": "PCG64, SHA256(seed:name) little-endian first 8 bytes, 256-row float32 chunks",
        }
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is not None and manifest["plan"] != plan:
            if manifest.get("stage") != "building":
                raise RuntimeError("Preparation provenance changed; refusing to overwrite existing model artifacts")
            # The previous run failed before committing an F16 or Q4 artifact.
            # Permit this incomplete manifest to adopt the fixed builder hash.
            manifest = {"plan": plan, "stage": "building"}
            atomic(manifest_path, manifest)
        if output.exists():
            if (manifest is None or manifest.get("stage") != "ready" or
                    output.stat().st_size != manifest.get("output_bytes")):
                raise RuntimeError("An uncommitted/different output exists; refusing to overwrite it")
            print(f"Reusing committed model: {output}", flush=True)
            return
        if manifest is not None and manifest.get("stage") == "ready":
            raise RuntimeError("Committed model is missing; use a separate output path to rebuild")
        if f16.exists() and (manifest is None or manifest.get("stage") != "f16_ready"):
            raise RuntimeError(f"Uncommitted intermediate exists: {f16}; inspect/remove only this intermediate before retrying")
        if manifest is None:
            manifest = {"plan": plan, "stage": "building"}
            atomic(manifest_path, manifest)
        # These are exclusively this builder's unfinished outputs, never the 7B model.
        temporary_f16.unlink(missing_ok=True)
        temporary_q4.unlink(missing_ok=True)
        specs = tensor_specs(arch)
        f16_bytes = sum(math.prod(shape) * np.dtype(dtype).itemsize for _, shape, dtype, _ in specs)
        required = (0 if f16.exists() else f16_bytes) + int(f16_bytes * 0.45) + 2 * 1024**3
        free = shutil.disk_usage(output.parent).free
        print(f"Free disk: {free / 1024**3:.1f} GiB; conservative additional requirement: {required / 1024**3:.1f} GiB", flush=True)
        if free < required:
            raise RuntimeError("Insufficient free disk for F16 plus Q4_K_M; no frequency controls were changed")
        if not f16.exists():
            tokenizer_hash = build_f16(temporary_f16, config, reference, specs, progress)
            os.replace(temporary_f16, f16)
            manifest.update(stage="f16_ready", tokenizer_metadata_sha256=tokenizer_hash,
                            parameter_count=sum(math.prod(shape) for _, shape, _, _ in specs),
                            tensor_count=len(specs), f16_bytes=f16.stat().st_size)
            atomic(manifest_path, manifest)
        elif f16.stat().st_size != manifest["f16_bytes"]:
            raise RuntimeError("F16 intermediate size changed; refusing to quantize")
        print(f"Quantizing with installed llama.cpp, {args.threads} CPU threads...", flush=True)
        atomic(progress, {"stage": "quantize", "source": str(f16), "output": str(temporary_q4)})
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--quantize-input", str(f16), "--quantize-output", str(temporary_q4),
                   "--threads", str(args.threads), "--quantization", "Q4_K_M"]
        child = subprocess.Popen(command)
        try:
            returncode = child.wait()
            if returncode:
                raise RuntimeError(f"Quantizer exited with status {returncode}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        if not temporary_q4.is_file() or temporary_q4.stat().st_size == 0:
            raise RuntimeError("Quantizer did not produce a nonempty GGUF")
        print("Recording the output SHA256...", flush=True)
        output_hash = sha256(temporary_q4)
        with temporary_q4.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_q4, output)
        manifest.update(stage="ready", output_sha256=output_hash, output_bytes=output.stat().st_size,
                        f16_retained=args.keep_f16, completed_unix_s=time.time())
        atomic(manifest_path, manifest)
        atomic(progress, {"stage": "ready", "output": str(output), "parameter_count": manifest["parameter_count"]})
        if not args.keep_f16:
            f16.unlink()
        print(json.dumps({"model": str(output), "manifest": str(manifest_path),
                          "parameter_count": manifest["parameter_count"],
                          "quantization": "Q4_K_M", "inference_tested": False}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--keep-f16", action="store_true")
    parser.add_argument("--quantize-input", help=argparse.SUPPRESS)
    parser.add_argument("--quantize-output", help=argparse.SUPPRESS)
    parser.add_argument("--quantization", choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
                        default="Q4_K_M", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.quantize_input or args.quantize_output:
        if not (args.quantize_input and args.quantize_output):
            parser.error("Both internal quantizer paths are required")
        quantize_worker(args.quantize_input, args.quantize_output, args.threads, args.quantization)
        return
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        prepare(args)
    except KeyboardInterrupt:
        print("Preparation interrupted. Committed F16 can be reused; no hardware frequency controls were written.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
