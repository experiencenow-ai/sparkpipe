import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from t1_reference_common import (bf16_round_f32, bf16_to_f32,  # noqa: E402
                                 f32_to_bf16_u16, parse_llm_defines,
                                 read_fixture)
from t1_reference_minimax import (MiniMaxEngine,  # noqa: E402
                                  bf16_expand, bf16_round_u16)


def conversion_bit_equivalence():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(4096).astype(np.float32) * rng.choice(
        [1e-3, 1.0, 100.0, 1e8, 1e-8], 4096)
    for value in (0.0, -0.0, np.inf, -np.inf, np.nan):
        x[0:8] = value
    expect(np.array_equal(bf16_round_u16(x), f32_to_bf16_u16(x)),
           "fast bf16 rounding must be bit-identical to the framework")
    u = f32_to_bf16_u16(x)
    expect(np.array_equal(bf16_expand(u).view(np.uint32),
                          bf16_to_f32(u).view(np.uint32)),
           "fast bf16 expand must be bit-identical to the framework")

HIDDEN = 16
LAYERS = 2
VOCAB = 32
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 8
INTERMEDIATE = 32
EPS = 1e-6
THETA = 5000000.0
Q_DIM = HEADS * HEAD_DIM
KV_DIM = KV_HEADS * HEAD_DIM
EOT = 31

DEFINES = """#pragma once

#define SPARK_LLM_FAMILY_TAG                    minimax
#define SPARK_LLM_HIDDEN_DIMENSION              16u
#define SPARK_LLM_LAYER_COUNT                   2u
#define SPARK_LLM_OUTPUT_VOCAB_COUNT            32u
#define SPARK_LLM_MAXIMUM_CONTEXT_TOKENS        64u
#define SPARK_LLM_RMS_NORM_EPSILON              1e-06f
#define SPARK_LLM_END_OF_TEXT_TOKEN_ID          31u
#define SPARK_LLM_BEGIN_OF_TEXT_TOKEN_ID        30u
#define SPARK_LLM_ATTENTION_HEAD_COUNT          4u
#define SPARK_LLM_KV_HEAD_COUNT                 2u
#define SPARK_LLM_HEAD_DIMENSION                8u
#define SPARK_LLM_INTERMEDIATE_DIMENSION        32u
#define SPARK_LLM_ROPE_THETA                    5000000.0f
#define SPARK_LLM_MROPE_SECTION_TEMPORAL        1u
#define SPARK_LLM_MROPE_SECTION_HEIGHT          2u
#define SPARK_LLM_MROPE_SECTION_WIDTH           1u
#define SPARK_LLM_MROPE_INTERLEAVED             1u
#define SPARK_LLM_TIED_WORD_EMBEDDINGS          0u
#define SPARK_LLM_MODEL_SOURCE_URI              "MiniMaxAI/MiniMax-H3"
"""


def config_document():
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "model_type": "qwen3_vl",
        "tie_word_embeddings": False,
        "vision_config": {"depth": 2, "hidden_size": 8},
        "text_config": {
            "hidden_size": HIDDEN, "num_hidden_layers": LAYERS,
            "vocab_size": VOCAB, "rms_norm_eps": EPS,
            "num_attention_heads": HEADS, "num_key_value_heads": KV_HEADS,
            "head_dim": HEAD_DIM, "intermediate_size": INTERMEDIATE,
            "max_position_embeddings": 64, "eos_token_id": EOT,
            "bos_token_id": 30, "hidden_act": "silu",
            "rope_theta": THETA,
            "rope_scaling": {"mrope_interleaved": True,
                             "mrope_section": [1, 2, 1],
                             "rope_type": "default"},
        },
    }


def bf16(name, shape, rng, scale=0.05):
    del name
    return f32_to_bf16_u16(rng.standard_normal(shape).astype(np.float32) * scale)


def build_tensors():
    rng = np.random.default_rng(19)
    t = {}
    t["model.visual.patch_embed.proj.weight"] = bf16("v", (4, 16), rng)
    t["model.language_model.embed_tokens.weight"] = bf16("e", (VOCAB, HIDDEN),
                                                         rng)
    t["model.language_model.norm.weight"] = bf16("n", (HIDDEN,), rng)
    head = bf16("lm", (VOCAB, HIDDEN), rng, scale=0.5)
    head[EOT] = np.zeros(HIDDEN, dtype=np.uint16)
    t["lm_head.weight"] = head
    for layer in range(LAYERS):
        p = f"model.language_model.layers.{layer}."
        t[p + "input_layernorm.weight"] = bf16("il", (HIDDEN,), rng)
        t[p + "post_attention_layernorm.weight"] = bf16("pl", (HIDDEN,), rng)
        t[p + "self_attn.q_proj.weight"] = bf16("q", (Q_DIM, HIDDEN), rng)
        t[p + "self_attn.k_proj.weight"] = bf16("k", (KV_DIM, HIDDEN), rng)
        t[p + "self_attn.v_proj.weight"] = bf16("v", (KV_DIM, HIDDEN), rng)
        t[p + "self_attn.o_proj.weight"] = bf16("o", (HIDDEN, Q_DIM), rng)
        t[p + "self_attn.q_norm.weight"] = bf16("qn", (HEAD_DIM,), rng)
        t[p + "self_attn.k_norm.weight"] = bf16("kn", (HEAD_DIM,), rng)
        t[p + "mlp.gate_proj.weight"] = bf16("g", (INTERMEDIATE, HIDDEN), rng)
        t[p + "mlp.up_proj.weight"] = bf16("u", (INTERMEDIATE, HIDDEN), rng)
        t[p + "mlp.down_proj.weight"] = bf16("d", (HIDDEN, INTERMEDIATE), rng)
    return t


def write_safetensors(path, tensors):
    header = {}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        array = tensors[name]
        dtype = "BF16" if array.dtype == np.uint16 else "F32"
        itemsize = 2 if array.dtype == np.uint16 else 4
        raw = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
        blobs.append(raw)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - len(header_bytes) % 8) % 8
    header_bytes += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for raw in blobs:
            fh.write(raw)


def write_checkpoint(directory, tensors=None):
    os.makedirs(directory, exist_ok=True)
    write_safetensors(os.path.join(directory, "model.safetensors"),
                      tensors if tensors is not None else build_tensors())
    with open(os.path.join(directory, "config.json"), "w") as fh:
        json.dump(config_document(), fh)


def write_header(path, text=DEFINES):
    with open(path, "w") as fh:
        fh.write(text)


def write_prompts(path):
    document = {"prompts": [{
        "name": "synth_a",
        "prompt_token_ids": [3, 7, 11, 5],
        "new_tokens": 3,
        "capture_layers": [0, 1],
    }]}
    with open(path, "w") as fh:
        json.dump(document, fh)


def run_generator(checkpoint, header, prompts, output):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_decoder.py"),
         "--family", "minimax", "--checkpoint", checkpoint,
         "--header", header, "--prompts", prompts, "--output", output],
        capture_output=True, text=True)


def compare(reference, candidate):
    return subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "t1_reference_compare.py"),
         "compare", "--reference", reference, "--candidate", candidate],
        capture_output=True, text=True)


def expect(condition, message):
    if not condition:
        raise AssertionError(message)


def hand_rope(rows, position):
    out = np.array(rows, dtype=np.float64)
    half = HEAD_DIM // 2
    for head in range(out.shape[0]):
        for j in range(half):
            freq = THETA ** (-2.0 * j / HEAD_DIM)
            angle = position * freq
            real = rows[head, j]
            imag = rows[head, j + half]
            out[head, j] = real * math.cos(angle) - imag * math.sin(angle)
            out[head, j + half] = imag * math.cos(angle) + real * math.sin(angle)
    return out.astype(np.float32)


def hand_rms(rows, gain):
    out = np.zeros_like(rows)
    for r in range(rows.shape[0]):
        acc = 0.0
        for c in range(rows.shape[1]):
            acc += float(rows[r, c]) * float(rows[r, c])
        scale = 1.0 / math.sqrt(acc / HEAD_DIM + EPS)
        for c in range(rows.shape[1]):
            out[r, c] = float(rows[r, c]) * scale * float(gain[c])
    return out


def hand_layer0(engine, token, position, history):
    p = "model.language_model.layers.0."
    x = bf16_to_f32(engine.st.raw(
        "model.language_model.embed_tokens.weight")[token].astype(np.uint16))
    x = bf16_round_f32(x)
    gain_in = bf16_to_f32(engine.st.raw(p + "input_layernorm.weight"))
    normed = bf16_round_f32(x / math.sqrt(
        float((x * x).sum()) / HIDDEN + EPS) * gain_in)
    wq = bf16_to_f32(engine.st.raw(p + "self_attn.q_proj.weight"))
    wk = bf16_to_f32(engine.st.raw(p + "self_attn.k_proj.weight"))
    wv = bf16_to_f32(engine.st.raw(p + "self_attn.v_proj.weight"))
    q = bf16_round_f32(wq @ normed).reshape(HEADS, HEAD_DIM)
    k = bf16_round_f32(wk @ normed).reshape(KV_HEADS, HEAD_DIM)
    v = bf16_round_f32(wv @ normed).reshape(KV_HEADS, HEAD_DIM)
    q = hand_rope(hand_rms(q, bf16_to_f32(engine.st.raw(
        p + "self_attn.q_norm.weight"))), position)
    k = hand_rope(hand_rms(k, bf16_to_f32(engine.st.raw(
        p + "self_attn.k_norm.weight"))), position)
    history = history + [(bf16_round_f32(k.reshape(-1)),
                          bf16_round_f32(v.reshape(-1)))]
    out = np.zeros((HEADS, HEAD_DIM), dtype=np.float32)
    for h in range(HEADS):
        kvh = h // (HEADS // KV_HEADS)
        scores = np.array([
            float(history[t][0].reshape(KV_HEADS, HEAD_DIM)[kvh] @ q[h])
            / math.sqrt(HEAD_DIM) for t in range(len(history))],
            dtype=np.float64)
        top = scores.max()
        weights = np.exp(scores - top)
        weights = weights / weights.sum()
        for t in range(len(history)):
            out[h] += weights[t] * history[t][1].reshape(KV_HEADS,
                                                         HEAD_DIM)[kvh]
    attended = bf16_round_f32(out.reshape(-1))
    wo = bf16_to_f32(engine.st.raw(p + "self_attn.o_proj.weight"))
    streams = bf16_round_f32(x + bf16_round_f32(wo @ attended))
    gain_post = bf16_to_f32(engine.st.raw(p + "post_attention_layernorm.weight"))
    normed2 = bf16_round_f32(streams / math.sqrt(
        float((streams * streams).sum()) / HIDDEN + EPS) * gain_post)
    wg = bf16_to_f32(engine.st.raw(p + "mlp.gate_proj.weight"))
    wu = bf16_to_f32(engine.st.raw(p + "mlp.up_proj.weight"))
    wd = bf16_to_f32(engine.st.raw(p + "mlp.down_proj.weight"))
    gate = bf16_round_f32(wg @ normed2)
    up = bf16_round_f32(wu @ normed2)
    act = bf16_round_f32(gate * (1.0 / (1.0 + np.exp(-gate))) * up)
    return (bf16_round_f32(streams + bf16_round_f32(wd @ act)), history)


def hand_check(engine):
    caches = [None] * LAYERS
    states = {}
    capture = {}
    history = []
    for position, token in enumerate([7, 11]):
        seen = {}
        streams = engine.decode_step(token, position, states, caches, capture,
                                     lambda i, s: seen.setdefault(i, s.copy()))
        expect(0 in seen, "capture_streams did not report layer 0")
        reference, history = hand_layer0(engine, token, position, history)
        delta = np.abs(seen[0] - reference).max()
        expect(delta < 1e-5,
               f"hand layer0 check diverged at position {position}: {delta}")
        expect(np.array_equal(streams, seen[LAYERS - 1]),
               "decode_step result disagrees with final layer stream")


def main():
    workspace = tempfile.mkdtemp(prefix="t1ref-minimax-")
    try:
        conversion_bit_equivalence()
        checkpoint = os.path.join(workspace, "checkpoint")
        write_checkpoint(checkpoint)
        header = os.path.join(workspace, "llm_defines.h")
        write_header(header)
        prompts = os.path.join(workspace, "prompts.json")
        write_prompts(prompts)
        out_a = os.path.join(workspace, "run_a")
        out_b = os.path.join(workspace, "run_b")
        result_a = run_generator(checkpoint, header, prompts, out_a)
        expect(result_a.returncode == 0,
               f"generator failed: {result_a.stderr}")
        result_b = run_generator(checkpoint, header, prompts, out_b)
        expect(result_b.returncode == 0,
               f"second generator run failed: {result_b.stderr}")
        fixture_a = os.path.join(out_a, "minimax", "synth_a.t1r")
        fixture_b = os.path.join(out_b, "minimax", "synth_a.t1r")
        expect(os.path.exists(fixture_a), "fixture missing after generation")
        expect(open(fixture_a, "rb").read() == open(fixture_b, "rb").read(),
               "generator is not byte-deterministic")
        _, arrays = read_fixture(fixture_a)
        for position in range(7):
            for layer in (0, 1):
                expect(f"pos{position:04d}_layer{layer:04d}_streams" in arrays,
                       f"anchor capture missing at pos {position} layer {layer}")
        expect(len(arrays["generated_token_ids"]) == 3,
               "generated token count disagrees with new_tokens")
        expect(not any("route_ids" in name for name in arrays),
               "dense text path must not emit routed-expert arrays")
        manifest = json.load(open(os.path.join(out_a, "minimax",
                                               "MANIFEST.json")))
        import hashlib
        expect(manifest["fixtures"]["synth_a.t1r"]["sha256"] ==
               hashlib.sha256(open(fixture_a, "rb").read()).hexdigest(),
               "manifest sha mismatch")
        expect(manifest["defines_config_mismatches"] == [],
               "defines/config closure must be empty for the synthetic pair")
        passed = compare(fixture_a, fixture_b)
        expect(passed.returncode == 0,
               f"identical fixtures must PASS: {passed.stdout} {passed.stderr}")
        corrupted = os.path.join(workspace, "synth_a_corrupt.t1r")
        corrupt = subprocess.run(
            [sys.executable,
             os.path.join(ROOT, "tools", "t1_reference_compare.py"),
             "corrupt-fixture", "--source", fixture_a, "--target", corrupted,
             "--array", "pos0002_layer0001_streams", "--offset", "3"],
            capture_output=True, text=True)
        expect(corrupt.returncode == 0, f"corrupt-fixture failed: {corrupt.stderr}")
        diverged = compare(fixture_a, corrupted)
        expect(diverged.returncode == 1, "corrupted fixture must FAIL")
        expect("pos0002_layer0001_streams" in diverged.stdout,
               f"FAIL must name the corrupted array: {diverged.stdout}")
        for old, new, needle in (
                ("SPARK_LLM_INTERMEDIATE_DIMENSION        32u",
                 "SPARK_LLM_INTERMEDIATE_DIMENSION        33u",
                 "INTERMEDIATE_DIMENSION"),
                ("SPARK_LLM_MROPE_SECTION_HEIGHT          2u",
                 "SPARK_LLM_MROPE_SECTION_HEIGHT          3u",
                 "MROPE_SECTION_HEIGHT")):
            bad_header = os.path.join(workspace, "llm_defines_bad.h")
            write_header(bad_header, DEFINES.replace(old, new))
            mismatch = run_generator(checkpoint, bad_header, prompts,
                                     os.path.join(workspace, "run_bad"))
            expect(mismatch.returncode != 0,
                   "defines/config disagreement must fail loud")
            expect(needle in mismatch.stderr,
                   f"failure must name {needle}: {mismatch.stderr}")
        os.remove(os.path.join(workspace, "llm_defines_bad.h"))
        engine = MiniMaxEngine(checkpoint, parse_llm_defines(header),
                               config_document()["text_config"])
        hand_check(engine)
        bad_tensors = build_tensors()
        bad_tensors["model.language_model.layers.0.self_attn.q_proj.weight"] = \
            bf16("q2", (Q_DIM + HEAD_DIM, HIDDEN), np.random.default_rng(3))
        bad_checkpoint = os.path.join(workspace, "checkpoint_bad")
        write_checkpoint(bad_checkpoint, bad_tensors)
        try:
            MiniMaxEngine(bad_checkpoint, parse_llm_defines(header),
                          config_document()["text_config"])
        except ValueError as error:
            expect("q_proj" in str(error),
                   f"shape failure must name the tensor: {error}")
        else:
            raise AssertionError("wrong q_proj shape must fail loud")
        missing_tensors = build_tensors()
        del missing_tensors["model.language_model.layers.1.mlp.down_proj.weight"]
        missing_checkpoint = os.path.join(workspace, "checkpoint_missing")
        write_checkpoint(missing_checkpoint, missing_tensors)
        try:
            MiniMaxEngine(missing_checkpoint, parse_llm_defines(header),
                          config_document()["text_config"])
        except KeyError as error:
            expect("down_proj" in str(error),
                   f"missing tensor must be named: {error}")
        else:
            raise AssertionError("missing tensor must fail loud")
        shutil.rmtree(workspace, ignore_errors=True)
        print("PASS t1_reference_minimax synthetic proof: determinism, "
              "manifest sha, negative control, defines/config fail-closed, "
              "hand layer check, checkpoint shape fail-closed")
        return 0
    except AssertionError:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
