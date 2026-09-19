import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t1_reference_common import (bf16_round_f32, bf16_to_f32, f32_to_bf16_u16,
                                 parse_llm_defines, rmsnorm, sha256_file,
                                 sigmoid)
import t1_reference_muse


def oracle_scaleless(x, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                       + np.float32(eps))


def oracle_centered(st, x, name, eps):
    w = bf16_to_f32(st.raw(name))
    normed = oracle_scaleless(x, eps)
    return bf16_round_f32(normed * (np.float32(1.0) + w))


QK_FACTOR = 3.87


def oracle_qk_norm_scale(rows, eps, factor):
    out = np.empty_like(rows)
    for h in range(rows.shape[0]):
        normed = bf16_round_f32(oracle_scaleless(rows[h:h + 1], eps))
        out[h] = bf16_round_f32(normed * np.float32(factor))
    return out


def oracle_rope(rows, position, theta, index, layer_types):
    if layer_types[index] != "sliding_attention":
        return rows
    dim = rows.shape[1]
    half = dim // 2
    exponents = np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)
    inv = np.float32(1.0) / (np.float32(theta) ** exponents)
    angles = np.float32(position) * inv
    cos = bf16_round_f32(np.cos(angles))
    sin = bf16_round_f32(np.sin(angles))
    cos = np.concatenate([cos, cos])
    sin = np.concatenate([sin, sin])
    out = np.empty_like(rows)
    for h in range(rows.shape[0]):
        rotated = np.concatenate([-rows[h, half:], rows[h, :half]])
        left = bf16_round_f32(rows[h] * cos)
        right = bf16_round_f32(rotated * sin)
        out[h] = bf16_round_f32(left + right)
    return out


def full_matmul_linear(st, x, name, hidden):
    return bf16_round_f32(bf16_to_f32(st.raw(name + ".weight")) @ x)


def oracle_layer_zero(st, streams, layer_prefix, hidden, heads, kv_heads,
                      head_dim, eps, post_eps, theta, layer_types, index,
                      linear):
    q_rows = heads * head_dim
    kv_rows = kv_heads * head_dim
    x = oracle_centered(st, streams, layer_prefix + "input_layernorm.weight",
                        eps)
    q = linear(st, x, layer_prefix + "self_attn.q_proj", hidden)
    k = linear(st, x, layer_prefix + "self_attn.k_proj", hidden)
    v = linear(st, x, layer_prefix + "self_attn.v_proj", hidden)
    q_heads = q.reshape(heads, head_dim).copy()
    k_heads = k.reshape(kv_heads, head_dim).copy()
    v_heads = v.reshape(kv_heads, head_dim).copy()
    q_heads = oracle_qk_norm_scale(q_heads, eps, QK_FACTOR)
    k_heads = bf16_round_f32(np.stack([
        oracle_scaleless(k_heads[h:h + 1], eps)[0]
        for h in range(kv_heads)]))
    q_heads = oracle_rope(q_heads, 0, theta, index, layer_types)
    k_heads = oracle_rope(k_heads, 0, theta, index, layer_types)
    attended = np.zeros(q_rows, dtype=np.float32)
    group = heads // kv_heads
    for h in range(heads):
        kvh = h // group
        scores = np.zeros(1, dtype=np.float32)
        for j in range(1):
            total = np.float32(0.0)
            for d in range(head_dim):
                total += np.float32(q_heads[h, d]) * np.float32(k_heads[kvh, d])
            scores[j] = total * np.float32(head_dim ** -0.5)
        peak = scores.max()
        weights = np.exp(scores - peak)
        weights = weights / weights.sum()
        for d in range(head_dim):
            acc = np.float32(0.0)
            for j in range(1):
                acc += np.float32(weights[j]) * np.float32(v_heads[kvh, d])
            attended[h * head_dim + d] = acc
    attended = bf16_round_f32(attended)
    gate = linear(st, x, layer_prefix + "self_attn.gate_proj", hidden)
    gate = bf16_round_f32(sigmoid(gate))
    gated = bf16_round_f32(attended * gate)
    attention = linear(st, gated, layer_prefix + "self_attn.o_proj", q_rows)
    normed = oracle_centered(
        st, attention, layer_prefix + "post_attention_layernorm.weight",
        post_eps)
    after_attn = bf16_round_f32(streams + normed)
    x2 = oracle_centered(st, after_attn,
                         layer_prefix + "pre_feedforward_layernorm.weight",
                         eps)
    gate_mlp = linear(st, x2, layer_prefix + "mlp.gate_proj", hidden)
    up = linear(st, x2, layer_prefix + "mlp.up_proj", hidden)
    silu = bf16_round_f32(gate_mlp * sigmoid(gate_mlp))
    activated = bf16_round_f32(silu * up)
    down = linear(st, activated, layer_prefix + "mlp.down_proj",
                  activated.shape[0])
    normed2 = oracle_centered(
        st, down, layer_prefix + "post_feedforward_layernorm.weight", post_eps)
    return bf16_round_f32(after_attn + normed2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--header", required=True)
    parser.add_argument("--token", type=int, default=954)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    started = time.time()
    defines = parse_llm_defines(arguments.header)
    config = json.load(open(os.path.join(arguments.checkpoint,
                                         "config.json")))
    if "text_config" in config:
        config = config["text_config"]
    engine = t1_reference_muse.ENGINE_CLASS(arguments.checkpoint, defines,
                                            config)
    index_path = os.path.join(arguments.checkpoint,
                              "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]
    layer_keys = [k for k in weight_map if ".layers." in k]
    text_keys = [k for k in weight_map if "language_model" in k
                 or k == "lm_head.weight"]
    closure = {
        "defines_config_mismatches": engine.mismatches,
        "checked_defines": sorted(name for name, _, _
                                  in t1_reference_muse.DEFINES_VS_CONFIG),
        "recorded_only_defines": sorted(name for name, _, _
                                        in t1_reference_muse.DEFINES_RECORDED_ONLY),
        "config_sha256": sha256_file(os.path.join(arguments.checkpoint,
                                                  "config.json")),
        "index_sha256": sha256_file(index_path),
        "llm_defines_sha256": sha256_file(arguments.header),
        "census": {"total_tensors": len(weight_map),
                   "text_tensors": len(text_keys),
                   "layer_tensors": len(layer_keys),
                   "vision_tensors": len(weight_map) - len(text_keys),
                   "layers": engine.layers,
                   "full_attention_layers": [i for i, kind in
                                             enumerate(engine.layer_types)
                                             if kind == "full_attention"],
                   "dense_mlp": True,
                   "moe_expert_tensors": 0},
    }
    engine_streams = engine.embed(arguments.token)
    caches = {i: [] for i in range(engine.layers)}
    oracle_arguments = (engine.st, engine_streams,
                        t1_reference_muse.PREFIX + "0.", engine.hidden,
                        engine.heads, engine.kv_heads, engine.head_dim,
                        engine.eps, engine.post_eps, engine.theta,
                        engine.layer_types, 0)
    engine_value = engine.forward_layer(0, engine_streams, caches, 0)
    oracle_value = oracle_layer_zero(
        *oracle_arguments,
        linear=lambda st, x, name, hidden: engine.linear(x, name))
    oracle_bits = f32_to_bf16_u16(oracle_value)
    engine_bits = f32_to_bf16_u16(engine_value)
    divergent = np.nonzero(oracle_bits != engine_bits)[0]
    bitwise_stage = lambda a, b: bool(np.array_equal(f32_to_bf16_u16(a),
                                                     f32_to_bf16_u16(b)))
    probe_rows = bf16_round_f32(np.arange(256, dtype=np.float32)
                                .reshape(2, 128) * np.float32(0.01))
    stage_checks = {
        "embed_norm_bitwise": bitwise_stage(
            engine.embed(arguments.token),
            bf16_round_f32(oracle_scaleless(
                bf16_to_f32(engine.st.raw_rows(
                    t1_reference_muse.EMBED, arguments.token, 1)[0]),
                engine.eps))),
        "centered_norm_bitwise": bitwise_stage(
            engine.centered_norm(engine_streams,
                                 t1_reference_muse.PREFIX
                                 + "0.input_layernorm.weight", engine.eps),
            oracle_centered(engine.st, engine_streams,
                            t1_reference_muse.PREFIX
                            + "0.input_layernorm.weight", engine.eps)),
        "rope_bitwise": bitwise_stage(
            engine.apply_rope(probe_rows.copy(), 7),
            oracle_rope(probe_rows.copy(), 7, engine.theta,
                        0, ["sliding_attention"])),
    }
    hand_check = {
        "scope": "layer 0 (sliding_attention, rope, position 0), "
                 "independent orchestration vs engine.forward_layer; "
                 "both sides share the engine blocked-linear accumulation "
                 "primitive, so any composite divergence is a math defect",
        "token": arguments.token,
        "bitwise_stage_checks": stage_checks,
        "composite_bitwise_equal": bool(len(divergent) == 0),
        "divergent_elements": int(len(divergent)),
        "divergent_of": int(len(engine_value)),
        "divergent_indices": divergent[:16].tolist(),
        "max_abs_delta_f32": float(np.max(np.abs(oracle_value
                                                 - engine_value))),
        "engine_norm": float(np.linalg.norm(engine_value)),
        "oracle_norm": float(np.linalg.norm(oracle_value)),
    }
    if not all(stage_checks.values()):
        raise ValueError(f"hand-check bitwise stage failed: {hand_check}")
    if divergent.size:
        raise ValueError(f"hand-check layer 0 not bitwise equal: "
                         f"{hand_check}")
    full = oracle_layer_zero(*oracle_arguments,
                             linear=lambda st, x, name, hidden:
                             full_matmul_linear(st, x, name, hidden))
    full_bits = f32_to_bf16_u16(full)
    full_divergent = np.nonzero(full_bits != engine_bits)[0]
    accumulation = {
        "experiment": "oracle layer 0 with full-tensor matmul vs engine "
                      "4096-row slab streaming; divergence is numpy "
                      "accumulation order, not architecture",
        "divergent_elements": int(len(full_divergent)),
        "divergent_of": int(len(engine_value)),
        "max_abs_delta_f32": float(np.max(np.abs(full - engine_value))),
    }
    probes = {}
    streams = engine.embed(arguments.token)
    for i in range(engine.layers):
        streams = engine.forward_layer(i, streams, caches, 0)
        if i in (0, 3, 49, 51):
            probes[f"layer{i}"] = {
                "kind": engine.layer_types[i],
                "norm": float(np.linalg.norm(streams)),
                "finite": bool(np.isfinite(streams).all()),
            }
    token, score = engine.logits(streams)
    bad_header = arguments.header + ".bad"
    with open(arguments.header) as fh:
        text = fh.read()
    broken = text.replace("SPARK_LLM_QK_SCALE_FACTOR SPARK_MUSE_GLIMMER_MODEL_ATTN_QK_SCALE_FACTOR",
                          "SPARK_LLM_QK_SCALE_FACTOR 3.88f")
    if broken == text:
        raise ValueError("negative-control edit did not apply")
    with open(bad_header, "w") as fh:
        fh.write(broken)
    negative = {"define": "SPARK_LLM_QK_SCALE_FACTOR"}
    try:
        bad_defines = parse_llm_defines(bad_header)
        t1_reference_muse.ENGINE_CLASS(arguments.checkpoint, bad_defines,
                                       config)
        negative["raised"] = False
    except Exception as error:
        negative["raised"] = True
        negative["error"] = str(error)
    finally:
        os.remove(bad_header)
    if not negative["raised"]:
        raise ValueError("negative control did not fail loud")
    if "QK_SCALE_FACTOR" not in negative["error"]:
        raise ValueError(f"negative control error must name the define: "
                         f"{negative['error']}")
    receipt = {
        "family": "muse",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "wall_seconds": round(time.time() - started, 1),
        "closure": closure,
        "hand_check_layer0": hand_check,
        "accumulation_experiment": accumulation,
        "layer_probes_position0": probes,
        "position0_top1": {"token": int(token), "softcapped_score": float(score)},
        "negative_control": negative,
    }
    with open(arguments.output, "w") as fh:
        json.dump(receipt, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({"status": "PASS", "output": arguments.output,
                      "closure_mismatches": len(closure["defines_config_mismatches"]),
                      "hand_check_bitwise": hand_check["bitwise_equal"],
                      "negative_control": negative["raised"],
                      "position0_top1": receipt["position0_top1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
