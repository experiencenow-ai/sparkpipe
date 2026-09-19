import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from t1_reference_common import f32_to_bf16_u16, read_fixture  # noqa: E402

DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    muse
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   6u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-05f
#define SPARK_LLM_POST_NORM_EPSILON             1e-08f
#define SPARK_LLM_INTERMEDIATE_DIMENSION        8u
#define SPARK_LLM_ATTENTION_HEAD_COUNT          2u
#define SPARK_LLM_KV_HEAD_COUNT                 2u
#define SPARK_LLM_HEAD_DIMENSION                4u
#define SPARK_LLM_ATTENTION_PERIOD              4u
#define SPARK_LLM_FULL_ATTENTION_PHASE          3u
#define SPARK_LLM_FULL_ATTENTION_LAYER_COUNT    1u
#define SPARK_LLM_SLIDING_LAYER_COUNT           5u
#define SPARK_LLM_SLIDING_WINDOW                2u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        128u
#define SPARK_LLM_ROPE_THETA                    500000.0f
#define SPARK_LLM_QK_SCALE_FACTOR               3.87f
#define SPARK_LLM_OUTPUT_MULTIPLIER             0.19611613513818404f
#define SPARK_LLM_FINAL_LOGIT_SOFTCAP           20.0f
#define SPARK_LLM_BOS_TOKEN_ID                  30u
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
"""

ROPE_THETA = 500000.0


def config_document():
    return {
        "text_config": {
            "hidden_size": 16, "num_hidden_layers": 6, "vocab_size": 32,
            "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
            "intermediate_size": 8, "num_attention_heads": 2,
            "num_key_value_heads": 2, "head_dim": 4, "sliding_window": 2,
            "qk_scale_factor": 3.87, "output_multiplier": 0.19611613513818404,
            "final_logit_softcapping": 20.0, "bos_token_id": 30,
            "eos_token_id": 31, "max_position_embeddings": 128,
            "rope_parameters": {"rope_theta": ROPE_THETA, "rope_type": "default"},
            "layer_types": ["sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention",
                            "sliding_attention", "sliding_attention"],
            "layer_rope_theta": [ROPE_THETA, ROPE_THETA, ROPE_THETA, 0,
                                 ROPE_THETA, ROPE_THETA],
        }
    }


def bf16(shape, rng, scale=0.05):
    return f32_to_bf16_u16(rng.standard_normal(shape).astype(np.float32)
                           * scale)


def build_tensors():
    rng = np.random.default_rng(19)
    t = {}
    t["model.language_model.embed_tokens.weight"] = bf16((32, 16), rng)
    t["model.language_model.norm.weight"] = bf16((16,), rng)
    head = bf16((32, 16), rng, scale=0.5)
    head[31] = np.zeros(16, dtype=np.uint16)
    t["lm_head.weight"] = head
    for layer in range(6):
        p = f"model.language_model.layers.{layer}."
        t[p + "input_layernorm.weight"] = bf16((16,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16((16,), rng)
        t[p + "pre_feedforward_layernorm.weight"] = bf16((16,), rng)
        t[p + "post_feedforward_layernorm.weight"] = bf16((16,), rng)
        t[p + "self_attn.q_proj.weight"] = bf16((8, 16), rng)
        t[p + "self_attn.k_proj.weight"] = bf16((8, 16), rng)
        t[p + "self_attn.v_proj.weight"] = bf16((8, 16), rng)
        t[p + "self_attn.gate_proj.weight"] = bf16((8, 16), rng)
        t[p + "self_attn.o_proj.weight"] = bf16((16, 8), rng)
        t[p + "mlp.gate_proj.weight"] = bf16((8, 16), rng)
        t[p + "mlp.up_proj.weight"] = bf16((8, 16), rng)
        t[p + "mlp.down_proj.weight"] = bf16((16, 8), rng)
    return t


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = tensors[name]
        raw = array.tobytes()
        header[name] = {"dtype": "BF16", "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as fh:
        fh.write(len(header_bytes).to_bytes(8, "little"))
        fh.write(header_bytes)
        for raw in blobs:
            fh.write(raw)


def write_checkpoint(directory, config=None):
    os.makedirs(directory, exist_ok=True)
    write_safetensors(os.path.join(directory, "model.safetensors"),
                      build_tensors())
    document = config if config is not None else config_document()
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(document, fh)


def write_header(path, text=DEFINES):
    with open(path, "w") as fh:
        fh.write(text)


def write_prompts(path):
    document = {"prompts": [{
        "name": "synth_muse",
        "prompt_token_ids": [3, 7, 11, 5],
        "new_tokens": 3,
        "capture_layers": [0, 3, 5],
    }]}
    with open(path, "w") as fh:
        json.dump(document, fh)


def run_generator(checkpoint, header, prompts, output):
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", "muse", "--checkpoint", checkpoint,
         "--header", header, "--prompts", prompts, "--output", output],
        capture_output=True, text=True)
    return result


def compare(reference, candidate):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
         "compare", "--reference", reference, "--candidate", candidate],
        capture_output=True, text=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def load_engine(checkpoint, header):
    from t1_reference_common import parse_llm_defines
    import t1_reference_muse
    defines = parse_llm_defines(header)
    config = json.load(open(os.path.join(checkpoint, "config.json")))
    if "text_config" in config:
        config = config["text_config"]
    return t1_reference_muse.ENGINE_CLASS(checkpoint, defines, config)


def engine_behavior_checks(checkpoint, header):
    engine = load_engine(checkpoint, header)
    caches = {}
    streams = None
    for position, token in enumerate([3, 7, 11, 5]):
        streams = engine.decode_step(token, position, {}, caches, {})
    expect(len(caches[0]) == 4 and len(caches[3]) == 4,
           "kv cache must hold every decoded position")
    token, score = engine.logits(streams)
    expect(0 <= token < 32, "logits token outside synthetic vocabulary")
    expect(score <= 20.0, "softcapped score must not exceed the cap")
    expect(abs(engine.softcapped(0.0)) < 1e-6, "softcap(0) must be 0")
    probe = engine.softcapped(1000.0)
    expect(abs(probe - 20.0) < 1e-3, "softcap must saturate at the cap")
    try:
        engine.decode_step(9999, 0, {}, {}, {})
    except ValueError as error:
        expect("outside vocabulary" in str(error), "wrong vocab guard message")
    else:
        raise AssertionError("out-of-vocabulary token must fail loud")
    bad_types = config_document()
    bad_types["text_config"]["layer_types"] = ["full_attention"] * 6
    bad_dir = os.path.join(checkpoint, "bad_types")
    write_checkpoint(bad_dir, bad_types)
    try:
        load_engine(bad_dir, header)
    except Exception as error:
        expect("period-4" in str(error), "wrong layer-type guard message")
    else:
        raise AssertionError("layer-type pattern violation must fail loud")
    bad_theta = config_document()
    bad_theta["text_config"]["layer_rope_theta"] = [0.0] * 6
    theta_dir = os.path.join(checkpoint, "bad_theta")
    write_checkpoint(theta_dir, bad_theta)
    try:
        load_engine(theta_dir, header)
    except Exception as error:
        expect("layer_rope_theta" in str(error), "wrong rope guard message")
    else:
        raise AssertionError("rope/full-attention disagreement must fail loud")


def main():
    workspace = tempfile.mkdtemp(prefix="t1ref-muse-test-")
    try:
        checkpoint = os.path.join(workspace, "checkpoint")
        write_checkpoint(checkpoint)
        header = os.path.join(workspace, "llm_defines.h")
        write_header(header)
        prompts = os.path.join(workspace, "prompts.json")
        write_prompts(prompts)
        engine_behavior_checks(checkpoint, header)
        out_a = os.path.join(workspace, "run_a")
        out_b = os.path.join(workspace, "run_b")
        for output in (out_a, out_b):
            result = run_generator(checkpoint, header, prompts, output)
            expect(result.returncode == 0,
                   f"generator failed: {result.stderr}")
        fixture_a = os.path.join(out_a, "muse", "synth_muse.t1r")
        fixture_b = os.path.join(out_b, "muse", "synth_muse.t1r")
        expect(os.path.exists(fixture_a), "fixture missing after generation")
        _, arrays = read_fixture(fixture_a)
        expect("pos0000_layer0000_streams" in arrays, "anchor capture missing")
        expect("pos0005_layer0003_streams" in arrays,
               "full-attention anchor capture missing")
        expect(not np.array_equal(arrays["pos0000_layer0000_streams"],
                                  arrays["pos0000_layer0005_streams"]),
               "per-layer snap capture must differ across layers")
        expect(open(fixture_a, "rb").read() == open(fixture_b, "rb").read(),
               "generator is not byte-deterministic")
        manifest = json.load(open(os.path.join(out_a, "muse", "MANIFEST.json")))
        expect(manifest["fixtures"]["synth_muse.t1r"]["sha256"] ==
               hashlib.sha256(open(fixture_a, "rb").read()).hexdigest(),
               "manifest sha mismatch")
        expect(manifest["defines_config_mismatches"] == [],
               "clean run must record zero defines/config mismatches")
        passed = compare(fixture_a, fixture_b)
        expect(passed.returncode == 0,
               f"identical fixtures must PASS: {passed.stdout} {passed.stderr}")
        corrupted = os.path.join(workspace, "synth_muse_corrupt.t1r")
        corrupt = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "t1_reference_compare.py"),
             "corrupt-fixture", "--source", fixture_a, "--target", corrupted,
             "--array", "pos0002_layer0000_streams", "--offset", "9"],
            capture_output=True, text=True)
        expect(corrupt.returncode == 0, f"corrupt-fixture failed: {corrupt.stderr}")
        diverged = compare(fixture_a, corrupted)
        expect(diverged.returncode == 1, "corrupted fixture must FAIL")
        expect("pos0002_layer0000_streams" in diverged.stdout,
               f"FAIL must name the corrupted array: {diverged.stdout}")
        bad_header = os.path.join(workspace, "llm_defines_bad.h")
        with open(header) as fh:
            text = fh.read()
        broken = text.replace("SPARK_LLM_HIDDEN_DIMENSION              16u",
                              "SPARK_LLM_HIDDEN_DIMENSION              17u")
        expect(broken != text, "hidden define not found in header")
        write_header(bad_header, broken)
        mismatch = run_generator(checkpoint, bad_header, prompts,
                                 os.path.join(workspace, "run_c"))
        expect(mismatch.returncode != 0,
               "defines/config disagreement must fail loud")
        expect("HIDDEN_DIMENSION" in mismatch.stderr,
               f"failure must name the mismatched define: {mismatch.stderr}")
        shutil.rmtree(workspace, ignore_errors=True)
        print("PASS t1_reference_muse synthetic proof: engine guards, "
              "determinism, snap capture, manifest sha, negative control, "
              "defines/config fail-closed")
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
