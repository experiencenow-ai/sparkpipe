#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

HEADER = Path(__file__).resolve().parent.parent / (
    "model-families/ling/include/sparkpipe/spark_ling_model.h")

SPEC_BW_GB_S = 273.0
MEASURED_BW_GB_S = 242.1
MEASURED_NVME_GB_S = 4.686
PROBE_SOURCE = "spark0 240.3 / spark1 243.9 GB/s, 1.5 GiB stream read, 2026-09-13"
CLASSIFICATION = "analytical estimate (measured bandwidth input)"


def defines(text):
    found = {}
    pattern = re.compile(r"^#define\s+(SPARK_LING_MODEL_[A-Z0-9_]+)\s+([0-9]+)u?\s*$", re.M)
    for name, value in pattern.findall(text):
        found[name] = int(value)
    return found


def require(d, name):
    if name not in d:
        sys.exit("missing " + name + " in " + HEADER.name)
    return d[name]


def build_table(d, context_tokens, expert_bytes_per_element, tp_degree):
    hidden = require(d, "SPARK_LING_MODEL_HIDDEN_DIMENSION")
    layers = require(d, "SPARK_LING_MODEL_LAYER_COUNT")
    vocab = require(d, "SPARK_LING_MODEL_OUTPUT_VOCAB_COUNT")
    heads = require(d, "SPARK_LING_MODEL_MLA_HEAD_COUNT")
    latent = require(d, "SPARK_LING_MODEL_MLA_LATENT_DIMENSION")
    rope = require(d, "SPARK_LING_MODEL_MLA_QK_ROPE_HEAD_DIMENSION")
    nope = require(d, "SPARK_LING_MODEL_MLA_QK_NOPE_HEAD_DIMENSION")
    value_dim = require(d, "SPARK_LING_MODEL_MLA_VALUE_HEAD_DIMENSION")
    mla_layers = require(d, "SPARK_LING_MODEL_MLA_LAYER_COUNT")
    kda_layers = require(d, "SPARK_LING_MODEL_KDA_LAYER_COUNT")
    kda_heads = require(d, "SPARK_LING_MODEL_KDA_HEAD_COUNT")
    kda_key = require(d, "SPARK_LING_MODEL_KDA_HEAD_KEY_DIMENSION")
    kda_value = require(d, "SPARK_LING_MODEL_KDA_HEAD_VALUE_DIMENSION")
    kda_state_element_bytes = require(d, "SPARK_LING_MODEL_KDA_STATE_ELEMENT_BYTES")
    kda_conv = require(d, "SPARK_LING_MODEL_KDA_CONV_KERNEL")
    experts = require(d, "SPARK_LING_MODEL_MOE_EXPERT_COUNT")
    top_k = require(d, "SPARK_LING_MODEL_MOE_TOP_K")
    expert_intermediate = require(d, "SPARK_LING_MODEL_MOE_INTERMEDIATE_DIMENSION")
    shared_experts = require(d, "SPARK_LING_MODEL_MOE_SHARED_EXPERT_COUNT")
    dense_intermediate = require(d, "SPARK_LING_MODEL_DENSE_INTERMEDIATE_DIMENSION")
    first_routed = require(d, "SPARK_LING_MODEL_FIRST_ROUTED_LAYER")
    dense_layers = require(d, "SPARK_LING_MODEL_FIRST_DENSE_LAYER_COUNT")
    kv_bits = require(d, "SPARK_LING_MODEL_KV_BITS")
    bf16 = 2
    scale_block = require(d, "SPARK_LING_MODEL_FP8_SCALE_BLOCK")
    bf16_element_bytes = require(d, "SPARK_LING_MODEL_BF16_ELEMENT_BYTES")

    local_heads = heads // tp_degree
    local_kda_heads = kda_heads // tp_degree
    query_dim = heads * (nope + rope)
    latent_row = latent + rope
    kv_slot_bytes = latent_row * kv_bits // 8
    kda_state_bytes = kda_heads * kda_key * kda_value * kda_state_element_bytes

    def linear_bytes(rows, input_dim, element_bytes, replicate):
        row_slice = rows if replicate else (rows + tp_degree - 1) // tp_degree
        return row_slice * input_dim * element_bytes

    def local_linear_bytes(local_rows, input_dim, element_bytes):
        return local_rows * input_dim * element_bytes

    rows = []
    mla_weight = 0
    mla_weight += linear_bytes(query_dim, hidden, bf16, False)
    mla_weight += linear_bytes(latent_row, hidden, bf16, True)
    mla_weight += local_heads * latent * nope * bf16
    mla_weight += local_heads * latent * value_dim * bf16
    mla_weight += linear_bytes(heads, hidden, bf16, False)
    mla_weight += local_linear_bytes(local_heads * value_dim, hidden, bf16)
    mla_weight += 3 * hidden * bf16
    rows.append(("mla attention spine", mla_layers, mla_weight))
    kda_weight = 0
    kda_weight += local_linear_bytes(2 * local_kda_heads * kda_key + local_kda_heads * kda_value + local_kda_heads, hidden, bf16)
    kda_weight += local_linear_bytes(local_kda_heads * kda_key, hidden, bf16)
    kda_weight += local_linear_bytes(local_kda_heads * kda_value, hidden, bf16)
    kda_weight += local_linear_bytes(local_kda_heads * kda_value, hidden, bf16)
    kda_weight += 3 * local_kda_heads * kda_key * kda_conv * bf16
    kda_weight += 4 * local_kda_heads * kda_value * bf16
    rows.append(("kda spine", kda_layers, kda_weight))
    routed_layers = layers - first_routed
    expert_w1_rows = 2 * expert_intermediate
    expert_bytes = (expert_w1_rows * hidden + hidden * expert_intermediate) * expert_bytes_per_element
    expert_bytes += (expert_w1_rows * ((hidden + scale_block - 1) // scale_block) + hidden * ((expert_intermediate + scale_block - 1) // scale_block)) * 4
    shared_weight = 0
    shared_weight += linear_bytes(2 * shared_experts * expert_intermediate, hidden, bf16, False)
    shared_weight += linear_bytes(shared_experts * expert_intermediate, hidden, bf16, False)
    router_weight = linear_bytes(experts, hidden, bf16, True) + experts * 4
    routed_weight = router_weight + shared_weight
    rows.append(("routed-layer spine (router replicated + shared expert)", routed_layers, routed_weight))
    dense_weight = 0
    dense_weight += linear_bytes(2 * dense_intermediate, hidden, bf16, False)
    dense_weight += linear_bytes(dense_intermediate, hidden, bf16, False)
    dense_weight += hidden * bf16
    rows.append(("dense ffn", dense_layers, dense_weight))
    head_weight = (vocab + tp_degree - 1) // tp_degree * hidden * bf16_element_bytes
    rows.append(("lm head (final stage)", 1, head_weight))
    rows.append(("embedding gather", 1, hidden * bf16_element_bytes))

    state_rows = []
    kda_state_rank_bytes = kda_heads * kda_key * kda_value * kda_state_element_bytes // tp_degree
    state_rows.append(("kda recurrent state read+write", kda_layers, 2 * kda_state_rank_bytes))
    state_rows.append(("kda conv window read+write", kda_layers, 2 * 3 * local_kda_heads * kda_key * kda_conv * bf16))
    kv_read = mla_layers * context_tokens * kv_slot_bytes
    state_rows.append(("mla latent kv read+write", 1, kv_read + mla_layers * kv_slot_bytes))
    active_experts = top_k * expert_bytes
    local_expert_bytes = (active_experts + tp_degree - 1) // tp_degree
    state_rows.append(("active expert weights (hot resident)", 1, local_expert_bytes))
    return rows, state_rows, local_expert_bytes, active_experts


def gigabytes(value):
    return value / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=16)
    parser.add_argument("--expert-codec", choices=["bf16", "fp8", "mxfp4"], default="bf16")
    parser.add_argument("--bw-gb-s", type=float, default=MEASURED_BW_GB_S)
    parser.add_argument("--receipt-tok-s", type=float, default=0.0)
    args = parser.parse_args()

    expert_element_bytes = {"bf16": 2.0, "fp8": 1.125, "mxfp4": 0.5625}[args.expert_codec]
    d = defines(HEADER.read_text())
    table, state_rows, local_expert_bytes, active_experts = build_table(
        d, args.context, expert_element_bytes, args.tp)

    total = sum(count * value for _, count, value in table)
    total += sum(count * value for _, count, value in state_rows)
    print("ling decode roofline, B1, TP%d, context %d tokens, expert codec %s" % (
        args.tp, args.context, args.expert_codec))
    print("classification: %s" % CLASSIFICATION)
    print("bandwidth: measured %.1f GB/s of %.0f GB/s spec (%.1f%%) [%s]" % (
        args.bw_gb_s, SPEC_BW_GB_S, 100.0 * args.bw_gb_s / SPEC_BW_GB_S, PROBE_SOURCE))
    print()
    print("%-52s %8s %14s" % ("block", "count", "bytes/token"))
    for name, count, value in table:
        print("%-52s %8d %14s" % (name, count, "%.3f MB" % (gigabytes(count * value) * 1e3)))
    for name, count, value in state_rows:
        print("%-52s %8s %14s" % (name, "-", "%.3f MB" % (gigabytes(count * value) * 1e3)))
    print("%-52s %8s %14s" % ("TOTAL per rank per token", "-", "%.3f GB" % gigabytes(total)))
    print()
    ceiling = args.bw_gb_s * 1e9 / total
    print("ceiling (hot resident experts): %.1f tok/s per rank at %.1f GB/s [ESTIMATE]" % (
        ceiling, args.bw_gb_s))
    cold_ceiling = args.bw_gb_s * 1e9 / (total - local_expert_bytes) if local_expert_bytes < total else 0.0
    cold_stream = active_experts / (MEASURED_NVME_GB_S * 1e9)
    print("active expert set per token: %.1f MB fleet, %.3f MB rank slice" % (
        gigabytes(active_experts) * 1e3, gigabytes(local_expert_bytes) * 1e3))
    print("ceiling (cold experts, NVMe-streamed at %.2f GB/s): %.1f tok/s [ESTIMATE]" % (
        MEASURED_NVME_GB_S, min(ceiling, 1.0 / cold_stream)))
    if args.receipt_tok_s > 0.0:
        print("receipted %.4f tok/s -> %.1f%% of measured roofline" % (
            args.receipt_tok_s, 100.0 * args.receipt_tok_s * total / (args.bw_gb_s * 1e9)))
    else:
        print("receipted tok/s: none (daemon-gated) -> percent of roofline N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
