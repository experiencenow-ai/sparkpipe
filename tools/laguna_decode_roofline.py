#!/usr/bin/env python3
import argparse
import re
import sys
from pathlib import Path

HEADER = Path(__file__).resolve().parent.parent / (
    "model-families/laguna/include/sparkpipe/spark_laguna_model.h")

SPEC_BW_GB_S = 273.0
MEASURED_BW_GB_S = 242.1
MEASURED_NVME_GB_S = 4.686
PROBE_SOURCE = "spark0 240.3 / spark1 243.9 GB/s, 1.5 GiB stream read, 2026-09-13"
CLASSIFICATION = "analytical estimate (measured bandwidth input)"


def defines(text):
    found = {}
    pattern = re.compile(r"^#define\s+(SPARK_LAGUNA_MODEL_[A-Z0-9_]+)\s+([0-9]+)u?\s*$", re.M)
    for name, value in pattern.findall(text):
        found[name] = int(value)
    return found


def require(d, name):
    if name not in d:
        sys.exit("missing " + name + " in " + HEADER.name)
    return d[name]


def build_table(d, context_tokens, expert_bytes_per_element, tp_degree):
    hidden = require(d, "SPARK_LAGUNA_MODEL_HIDDEN_DIMENSION")
    layers = require(d, "SPARK_LAGUNA_MODEL_LAYER_COUNT")
    vocab = require(d, "SPARK_LAGUNA_MODEL_OUTPUT_VOCAB_COUNT")
    head_dim = require(d, "SPARK_LAGUNA_MODEL_ATTENTION_HEAD_DIMENSION")
    kv_heads = require(d, "SPARK_LAGUNA_MODEL_ATTENTION_KV_HEAD_COUNT")
    full_q_heads = require(d, "SPARK_LAGUNA_MODEL_ATTENTION_Q_HEAD_COUNT_FULL")
    sliding_q_heads = require(d, "SPARK_LAGUNA_MODEL_ATTENTION_Q_HEAD_COUNT_SLIDING")
    window = require(d, "SPARK_LAGUNA_MODEL_SLIDING_WINDOW")
    full_layers = require(d, "SPARK_LAGUNA_MODEL_FULL_LAYER_COUNT")
    sliding_layers = require(d, "SPARK_LAGUNA_MODEL_SLIDING_LAYER_COUNT")
    experts = require(d, "SPARK_LAGUNA_MODEL_MOE_EXPERT_COUNT")
    top_k = require(d, "SPARK_LAGUNA_MODEL_MOE_TOP_K")
    expert_intermediate = require(d, "SPARK_LAGUNA_MODEL_MOE_INTERMEDIATE_DIMENSION")
    shared_experts = require(d, "SPARK_LAGUNA_MODEL_MOE_SHARED_EXPERT_COUNT")
    dense_intermediate = require(d, "SPARK_LAGUNA_MODEL_DENSE_INTERMEDIATE_DIMENSION")
    first_routed = require(d, "SPARK_LAGUNA_MODEL_FIRST_ROUTED_LAYER")
    kv_bits = require(d, "SPARK_LAGUNA_MODEL_KV_BITS")
    bf16 = 2
    scale_block = require(d, "SPARK_LAGUNA_MODEL_FP8_SCALE_BLOCK")

    local = lambda value: (value + tp_degree - 1) // tp_degree
    local_kv_heads = local(kv_heads)
    kv_slot_bytes = local_kv_heads * 2 * head_dim * kv_bits // 8
    routed_layers = layers - first_routed
    dense_layers = first_routed

    rows = []
    full_weight = 0
    full_weight += local(full_q_heads + 2 * kv_heads) * head_dim * hidden * bf16
    full_weight += local(full_q_heads) * head_dim * hidden * bf16
    full_weight += local(full_q_heads) * head_dim * bf16
    full_weight += (full_q_heads + kv_heads) * head_dim * bf16
    full_weight += 3 * hidden * bf16
    rows.append(("full attention spine", full_layers, full_weight))
    sliding_weight = 0
    sliding_weight += local(sliding_q_heads + 2 * kv_heads) * head_dim * hidden * bf16
    sliding_weight += local(sliding_q_heads) * head_dim * hidden * bf16
    sliding_weight += local(sliding_q_heads) * head_dim * bf16
    sliding_weight += (sliding_q_heads + kv_heads) * head_dim * bf16
    sliding_weight += 3 * hidden * bf16
    rows.append(("sliding attention spine", sliding_layers, sliding_weight))
    expert_w1_rows = 2 * expert_intermediate
    expert_bytes = (expert_w1_rows * hidden + hidden * expert_intermediate) * expert_bytes_per_element
    expert_bytes += (expert_w1_rows * ((hidden + scale_block - 1) // scale_block) + hidden * ((expert_intermediate + scale_block - 1) // scale_block)) * 4
    shared_weight = 0
    shared_weight += local(2 * shared_experts * expert_intermediate) * hidden * bf16
    shared_weight += local(shared_experts * expert_intermediate) * hidden * bf16
    router_weight = experts * hidden * bf16 + experts * 4
    rows.append(("routed-layer spine (router + shared expert)", routed_layers, router_weight + shared_weight))
    dense_weight = 0
    dense_weight += local(2 * dense_intermediate) * hidden * bf16
    dense_weight += local(dense_intermediate) * hidden * bf16
    rows.append(("dense ffn (first layer)", dense_layers, dense_weight))
    head_weight = local(vocab) * hidden * bf16
    rows.append(("lm head", 1, head_weight))
    rows.append(("embedding gather", 1, hidden * bf16))

    state_rows = []
    sliding_kv = sliding_layers * min(context_tokens, window) * kv_slot_bytes
    full_kv = full_layers * context_tokens * kv_slot_bytes
    state_rows.append(("sliding kv read+write (window %d)" % window, 1, sliding_kv + sliding_layers * kv_slot_bytes))
    state_rows.append(("full attention kv read+write", 1, full_kv + full_layers * kv_slot_bytes))
    active_experts = top_k * expert_bytes
    local_expert_bytes = (active_experts + tp_degree - 1) // tp_degree
    state_rows.append(("active expert weights (hot resident)", 1, local_expert_bytes))
    return rows, state_rows, local_expert_bytes, active_experts


def gigabytes(value):
    return value / 1e9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=8)
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
    print("laguna decode roofline, B1, TP%d, context %d tokens, expert codec %s" % (
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
    cold = active_experts / (MEASURED_NVME_GB_S * 1e9)
    print("active expert set per token: %.1f MB fleet, %.3f MB rank slice" % (
        gigabytes(active_experts) * 1e3, gigabytes(local_expert_bytes) * 1e3))
    print("ceiling (cold experts, NVMe-streamed at %.2f GB/s): %.1f tok/s [ESTIMATE]" % (
        MEASURED_NVME_GB_S, min(ceiling, 1.0 / cold)))
    if args.receipt_tok_s > 0.0:
        print("receipted %.4f tok/s -> %.1f%% of measured roofline" % (
            args.receipt_tok_s, 100.0 * args.receipt_tok_s * total / (args.bw_gb_s * 1e9)))
    else:
        print("receipted tok/s: none (daemon-gated) -> percent of roofline N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
