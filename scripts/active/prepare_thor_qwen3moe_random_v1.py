#!/usr/bin/env python3
"""Build a deterministic, scaled Qwen3-MoE GGUF without network weights.

The probe preserves the official 30B-A3B hidden size, attention geometry,
128-expert routing, top-8 activation, and expert width. It uses 20 rather than
48 blocks so the F16-to-Q4_K_M construction fits on the Thor development disk.
It is a systems probe, not a quality model. No hardware controls are written.
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
DEFAULT_CONFIG = "configs/models/qwen3moe_13b_a18b_random_v1.json"
MAX_RANDOM_BLOCK_ELEMENTS = 32 * 1024 * 1024


def resolve(value):
    return (ROOT / value).resolve()


def sha256(filename):
    digest = hashlib.sha256()
    with open(filename, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(filename, value):
    temporary = filename.with_name(filename.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, filename)


def tensor_specs(arch):
    hidden = arch["hidden_size"]
    head_dim = arch["head_dim"]
    query = head_dim * arch["num_attention_heads"]
    key_value = head_dim * arch["num_key_value_heads"]
    experts = arch["num_experts"]
    expert_ff = arch["moe_intermediate_size"]
    vocab = arch["vocab_size"]
    specs = [
        ("token_embd.weight", (vocab, hidden), np.float16, "random"),
        ("output_norm.weight", (hidden,), np.float32, "ones"),
        ("output.weight", (vocab, hidden), np.float16, "random"),
    ]
    for layer in range(arch["num_hidden_layers"]):
        prefix = f"blk.{layer}."
        specs.extend([
            (prefix + "attn_norm.weight", (hidden,), np.float32, "ones"),
            (prefix + "attn_q.weight", (query, hidden), np.float16, "random"),
            (prefix + "attn_k.weight", (key_value, hidden), np.float16, "random"),
            (prefix + "attn_v.weight", (key_value, hidden), np.float16, "random"),
            (prefix + "attn_output.weight", (hidden, query), np.float16, "random"),
            (prefix + "attn_q_norm.weight", (head_dim,), np.float32, "ones"),
            (prefix + "attn_k_norm.weight", (head_dim,), np.float32, "ones"),
            (prefix + "ffn_norm.weight", (hidden,), np.float32, "ones"),
            (prefix + "ffn_gate_inp.weight", (experts, hidden), np.float16, "random"),
            (prefix + "ffn_gate_exps.weight", (experts, expert_ff, hidden), np.float16, "random"),
            (prefix + "ffn_up_exps.weight", (experts, expert_ff, hidden), np.float16, "random"),
            (prefix + "ffn_down_exps.weight", (experts, hidden, expert_ff), np.float16, "random"),
        ])
    return specs


def add_metadata(writer, arch):
    writer.add_name("EnergyInfra random Qwen3-MoE 13B-A1.8B systems probe")
    writer.add_description(
        "Random-weight Qwen3-MoE systems probe preserving 128 experts and top-8 routing"
    )
    writer.add_context_length(arch["max_position_embeddings"])
    writer.add_embedding_length(arch["hidden_size"])
    writer.add_block_count(arch["num_hidden_layers"])
    writer.add_feed_forward_length(arch["intermediate_size"])
    writer.add_expert_feed_forward_length(arch["moe_intermediate_size"])
    writer.add_head_count(arch["num_attention_heads"])
    writer.add_head_count_kv(arch["num_key_value_heads"])
    writer.add_key_length(arch["head_dim"])
    writer.add_value_length(arch["head_dim"])
    writer.add_rope_dimension_count(arch["head_dim"])
    writer.add_rope_freq_base(arch["rope_theta"])
    writer.add_layer_norm_rms_eps(arch["rms_norm_eps"])
    writer.add_expert_count(arch["num_experts"])
    writer.add_expert_used_count(arch["num_experts_per_tok"])
    writer.add_causal_attention(True)
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16))
    writer.add_quantization_version(2)


def add_tokenizer(writer, vocab):
    sys.path.insert(0, str(ROOT / "scripts"))
    from create_random_gguf import build_qwen2_tokens

    tokens, scores, token_types = build_qwen2_tokens(vocab)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(151643)
    writer.add_eos_token_id(151645)
    writer.add_unk_token_id(151643)
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)


def random_array(config, name, shape, dtype, initialization):
    if initialization == "ones":
        return np.ones(shape, dtype=dtype)
    seed = int.from_bytes(
        hashlib.sha256(f"{config['seed']}:{name}".encode("ascii")).digest()[:8],
        "little",
    )
    rng = np.random.default_rng(seed)
    result = np.empty(shape, dtype=dtype)
    tail = math.prod(shape[1:]) if len(shape) > 1 else 1
    rows = max(1, min(shape[0], MAX_RANDOM_BLOCK_ELEMENTS // tail))
    for first in range(0, shape[0], rows):
        count = min(rows, shape[0] - first)
        block = rng.standard_normal((count, *shape[1:]), dtype=np.float32)
        block *= np.float32(config["architecture"]["initializer_range"])
        result[first:first + count] = block
    return result


def build_f16(filename, config, specs, progress):
    arch = config["architecture"]
    writer = gguf.GGUFWriter(str(filename), arch["gguf_architecture"])
    add_metadata(writer, arch)
    add_tokenizer(writer, arch["vocab_size"])
    for name, shape, dtype, _ in specs:
        size = math.prod(shape) * np.dtype(dtype).itemsize
        writer.add_tensor_info(name, shape, np.dtype(dtype), size)
    started = time.monotonic()
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        for index, (name, shape, dtype, initialization) in enumerate(specs, 1):
            print(f"tensor {index}/{len(specs)}: {name} {shape}", flush=True)
            atomic(progress, {
                "stage": "generate_f16",
                "tensor": name,
                "completed": index - 1,
                "total": len(specs),
                "elapsed_s": time.monotonic() - started,
            })
            array = random_array(config, name, shape, dtype, initialization)
            writer.write_tensor_data(array)
            del array
    finally:
        writer.close()
    with filename.open("rb") as stream:
        os.fsync(stream.fileno())


def quantize_worker(source, destination, threads):
    from llama_cpp import llama_cpp as api

    params = api.llama_model_quantize_default_params()
    params.nthread = threads
    params.ftype = int(api.LLAMA_FTYPE_MOSTLY_Q4_K_M)
    params.quantize_output_tensor = True
    params.allow_requantize = False
    params.pure = False
    api.llama_backend_init()
    try:
        result = api.llama_model_quantize(
            os.fsencode(source), os.fsencode(destination), ctypes.byref(params)
        )
        if result != 0:
            raise RuntimeError(f"llama_model_quantize returned {result}")
    finally:
        api.llama_backend_free()


def validate_config(config):
    arch = config["architecture"]
    expected = (
        "qwen3_moe", "qwen3moe", 151936, 2048, 6144, 768, 20,
        32, 4, 128, 128, 8, False,
    )
    actual = (
        arch["model_type"], arch["gguf_architecture"], arch["vocab_size"],
        arch["hidden_size"], arch["intermediate_size"],
        arch["moe_intermediate_size"], arch["num_hidden_layers"],
        arch["num_attention_heads"], arch["num_key_value_heads"],
        arch["head_dim"], arch["num_experts"], arch["num_experts_per_tok"],
        arch["tie_word_embeddings"],
    )
    if actual != expected or config["quantization"] != "Q4_K_M":
        raise RuntimeError("Configuration is not the pinned random Qwen3-MoE systems probe")


def prepare(args):
    import llama_cpp
    from llama_cpp import llama_cpp as api

    config_path = resolve(args.config)
    config = json.loads(config_path.read_text())
    validate_config(config)
    output = resolve(config["output"])
    f16 = resolve(config["f16_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".json")
    progress = output.parent / "prepare_progress.json"
    partial_f16 = f16.with_suffix(f16.suffix + ".partial")
    partial_q4 = output.with_suffix(output.suffix + ".partial")

    with (output.parent / ".qwen3moe_random_prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = {
            "schema": config["schema"],
            "config": config,
            "builder_sha256": sha256(__file__),
            "tokenizer_helper_sha256": sha256(ROOT / "scripts/create_random_gguf.py"),
            "numpy_version": np.__version__,
            "llama_cpp_version": llama_cpp.__version__,
            "quantizer_library_sha256": sha256(api._lib._name),
            "threads": args.threads,
            "rng": "PCG64, SHA256(seed:tensor-name), float32 generation, F16 storage",
        }
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if manifest is not None and manifest.get("plan") != plan:
            raise RuntimeError("Random MoE provenance changed; use a new output directory")
        if output.exists():
            if manifest is None or manifest.get("stage") != "ready":
                raise RuntimeError("Uncommitted output exists; refusing to overwrite it")
            print(json.dumps({"model": str(output), "manifest": str(manifest_path),
                              "reused": True}, indent=2), flush=True)
            return
        if manifest is not None and manifest.get("stage") == "ready":
            raise RuntimeError("Committed output is missing; use a new output directory")
        if f16.exists() and (manifest is None or manifest.get("stage") != "f16_ready"):
            raise RuntimeError("Uncommitted F16 intermediate exists; refusing to quantize it")
        if manifest is None:
            manifest = {"plan": plan, "stage": "building"}
            atomic(manifest_path, manifest)
        partial_f16.unlink(missing_ok=True)
        partial_q4.unlink(missing_ok=True)

        specs = tensor_specs(config["architecture"])
        parameters = sum(math.prod(shape) for _, shape, _, _ in specs)
        active_experts = config["architecture"]["num_experts_per_tok"]
        total_experts = config["architecture"]["num_experts"]
        expert_parameters = sum(
            math.prod(shape) for name, shape, _, _ in specs if "_exps.weight" in name
        )
        active_parameters = parameters - expert_parameters + (
            expert_parameters * active_experts // total_experts
        )
        f16_bytes = sum(
            math.prod(shape) * np.dtype(dtype).itemsize for _, shape, dtype, _ in specs
        )
        required = (0 if f16.exists() else f16_bytes) + int(f16_bytes * 0.42) + 2 * 1024**3
        free = shutil.disk_usage(output.parent).free
        print(json.dumps({
            "parameter_count": parameters,
            "active_parameter_estimate": active_parameters,
            "f16_tensor_bytes": f16_bytes,
            "free_bytes": free,
            "required_bytes": required,
        }, sort_keys=True), flush=True)
        if free < required:
            raise RuntimeError("Insufficient free disk for random MoE F16 plus Q4_K_M")

        if not f16.exists():
            build_f16(partial_f16, config, specs, progress)
            os.replace(partial_f16, f16)
            manifest.update(
                stage="f16_ready",
                parameter_count=parameters,
                active_parameter_estimate=active_parameters,
                tensor_count=len(specs),
                f16_bytes=f16.stat().st_size,
            )
            atomic(manifest_path, manifest)
        elif f16.stat().st_size != manifest["f16_bytes"]:
            raise RuntimeError("F16 intermediate size changed; refusing to quantize")

        atomic(progress, {"stage": "quantize", "source": str(f16), "output": str(partial_q4)})
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--quantize-input", str(f16),
            "--quantize-output", str(partial_q4),
            "--threads", str(args.threads),
        ]
        child = subprocess.Popen(command)
        try:
            returncode = child.wait()
            if returncode:
                raise RuntimeError(f"Quantizer exited with status {returncode}")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=15)
        if not partial_q4.is_file() or partial_q4.stat().st_size == 0:
            raise RuntimeError("Quantizer did not produce a nonempty GGUF")

        print("Recording Q4_K_M SHA256...", flush=True)
        output_hash = sha256(partial_q4)
        with partial_q4.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial_q4, output)
        manifest.update(
            stage="ready",
            output_sha256=output_hash,
            output_bytes=output.stat().st_size,
            f16_retained=False,
            completed_unix_s=time.time(),
            inference_tested=False,
        )
        atomic(manifest_path, manifest)
        atomic(progress, {"stage": "ready", "output": str(output)})
        f16.unlink()
        print(json.dumps({
            "model": str(output),
            "manifest": str(manifest_path),
            "parameter_count": parameters,
            "active_parameter_estimate": active_parameters,
            "quantization": "Q4_K_M",
            "inference_tested": False,
        }, indent=2), flush=True)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"received signal {signum}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quantize-input", help=argparse.SUPPRESS)
    parser.add_argument("--quantize-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.quantize_input or args.quantize_output:
        if not (args.quantize_input and args.quantize_output):
            parser.error("Both internal quantizer paths are required")
        quantize_worker(args.quantize_input, args.quantize_output, args.threads)
        return
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        prepare(args)
    except KeyboardInterrupt:
        print("Preparation interrupted; committed F16 can be reused.", flush=True)
        raise SystemExit(130)


if __name__ == "__main__":
    main()

