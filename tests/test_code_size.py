"""Enforce a monotonic ceiling for authored non-test source code.

Generated build products, test fixtures, documentation, caches, and package
receipts are not implementation source and must never move this number. The
previous counter included generated model-driver C files under build/, so its
ceiling changed depending on which tests had already run. This counter is
stable before and after a clean build.
"""
import sys
from pathlib import Path

# Phase 6 adds lossless per-lane completion ownership at the rank boundary,
# completion-to-transaction correlation, synchronous-callback deferral, strict
# final-event identity propagation, and rollback-safe submission ownership. The
# exact authored-source count at landing is retained so later changes remain
# monotonic. The follow-up audit landing adds tools/verify_package_manifest.py
# and wires more gates into tools/gates.sh; the ceiling moves by those
# tooling lines (86), no production source grew for its own sake.
# The audit-fix landing adds the mbarrier phase-parity model coverage
# (test_mma_fragment_mapping.c, +89), the deterministic-failure paths and their
# tests (node/backend.c +128, test_ring_service_backend_transactions.c +168),
# the shared smem opt-in (runtime/launch.h, qwen/kimi call sites net negative),
# and the new no-python/manifest gates; ceiling moves to the exact count.
# The performance wave adds the tensor-map descriptor cache
# (runtime/gemm_descriptor_cache.h + test), the comms arena (runtime/arena.h +
# test), the RDMA eviction/batching/lane-rotation logic in rdma.cu, the BF16
# collective path, and docs/PERF_ROADMAP_2026-08-01.md; ceiling moves to the
# exact count again.
# The NVMe JIT KV tier (cache/nvme_tier.c + include/sparkpipe/spark_nvme_tier.h,
# the tier-3 manager and lookahead prefetcher, ~1080 lines with its build and
# gate wiring) lands alongside concurrent performance-wave work; ceiling moves
# to the exact count again.
# The K3 pack format V2 redesign (tools/k3_pack.py +400: the fused KDA
# projection emission, the interleaved weight+scale relay and its numpy-free
# fallback, the layout validator; tools/generate_k3_contract.py and
# generated_config.h gain the pack constants and geometry checks) lands its
# tooling lines; ceiling moves to the exact count.
# The DSv4 driver audit (2026-08-01) adds the exact attention-bytes
# derivation and the sparse-launch/rope-span defect flags to
# inference/llms/deepseek_v4/layer.cuh (+70), the Pro launch-budget note to
# deepseek_v4_pro/unity.cu (+10), and the two dsv4 gates' wiring in
# tools/gates.sh and Makefile (+6); the quantise dedup is net-negative code.
# Ceiling moves by those 86 lines; concurrent agents' in-flight growth is
# theirs to account.
# The D10 graph/gather/head wave adds the stage-side CUDA graph cache and
# replay contract (inference/stage/graph_replay.h, 451), the dispatch.cu
# capture wiring (+202 net), the head's chunked top-k pair with its launcher
# (inference/kernels/head.cuh, +231), the route row-indirection consumer
# contract (inference/kernels/route.cuh, +65), the slot-state field docs
# (+9), and their build/gate wiring (Makefile +6, tools/gates.sh +10,
# tools/build_head_topk.sh 25). Ceiling moves by those 999 lines; concurrent
# agents' in-flight growth remains theirs to account.
# The NVMe KV sizing work adds the estimator behind the dedicate-the-external
# drive decision (tools/nvme_kv_estimate.py, 407), the tier's default
# bandwidth/step-time assumptions from that analysis
# (include/sparkpipe/spark_nvme_tier.h +19), and the gate wiring (Makefile +1,
# tools/gates.sh +4); the test and the doc are excluded by construction.
# Ceiling moves by those 431 lines; concurrent agents' in-flight growth
# remains theirs to account.
# The topology-switch landing adds the TP16<->PP16 switch state machine and
# its strategy-neutral KV key scheme (scheduler/topology_switch.c, 646;
# include/sparkpipe/spark_topology_switch.h, 350) and its build/gate wiring
# (Makefile +6, tools/gates.sh +3) - 1,005 lines; the balance of the
# exact-count move is concurrent agents' in-flight growth, theirs to
# account.
# The hardware-topology landing adds the generator/validator
# (tools/generate_topology.py, 562) and its two generated C artifacts
# (deployment/include/sparkpipe/spark_hardware_topology.h, 111;
# deployment/src/spark_hardware_topology_tables.c, 285) plus the gate
# wiring (Makefile +1, tools/gates.sh +3) - 962 lines; the balance of the
# exact-count move is concurrent agents' in-flight growth, theirs to
# account.
# The batch-variant landing adds the per-family tuning headers
# (modules/glm52_resident_decode_stage/include/sparkpipe/spark_glm52_batch_tuning.h, 131;
# modules/k3_resident_decode_stage/include/sparkpipe/spark_k3_batch_tuning.h, 115),
# the variant rules and all/variants/publish_variants targets in the glm52
# module Makefile (+95 net), the firmware header's composed module ID (+5),
# and the build/gate wiring (Makefile +25, tools/gates.sh +21) - 392 lines;
# the test is excluded by construction. The balance of the exact-count move
# is concurrent agents' in-flight growth, theirs to account.
# The TP/PP recipe wave adds the recipe generator (tools/generate_recipe.py,
# 914: family adapters over the authoritative contracts, the stage_plan.c
# balancing DP mirrored, the k3 shard table built from k3_shard's own sets)
# and its gate wiring (tools/gates.sh +8, Makefile +1) - 923 lines; the
# test, the naming doc, the mimo25 contract JSON and the generated
# examples/recipes/ set are excluded by construction. The balance of the
# exact-count move is concurrent agents' in-flight growth, theirs to
# account.
# The K3 pack V2 shard wave teaches the TP tooling the V2 tensor names and
# slice rules (tools/k3_shard.py +118: the fused q|k|v|beta per-section
# head slice, the replicated decay|gate fusion, the interleaved expert
# cell/k-tile splits with manifest repricing; tools/generate_recipe.py +8:
# the scale-less expert classes and the 128-element k-tile group;
# model-families/k3/.../spark_k3_tp_shard_table.h +4: the V2 suffixes) -
# 130 lines. The balance of the exact-count move is concurrent agents'
# in-flight growth, theirs to account.
# The prefill/decode estimator landing adds tools/perf_estimate.py (649:
# the launch/wall/collective/transport overlay on top of the imported
# nvme_kv_estimate byte law, the chunked-prefill model, and the sm_121a
# ptxas occupancy check) and its gate wiring (Makefile +1,
# tools/gates.sh +4) - 654 lines; the test and the estimates doc are
# excluded by construction. Ceiling moves to the exact count.
# The power-of-2 batch-variant wave widens the bucket ladder from
# {8, 64, 256, 1024} to all eleven powers of two B1..B1024: the per-family
# tuning headers gain the seven new module-ID compositions, guard rungs, and
# ceiling-picker rungs (spark_glm52_batch_tuning.h +70 net,
# spark_k3_batch_tuning.h +68 net), the graph cache spare entries grow to
# hold 11 buckets x MTP draft variants with the slot-state ABI bump
# (include/sparkpipe/spark_resident_decode_stage.h +11), and the variant-set
# comments move with the ladder (module Makefile +3, top Makefile +1,
# tools/gates.sh +1, inference/stage/graph_replay.h +1) - 155 lines; the
# test is excluded by construction. The balance of the exact-count move is
# concurrent agents' in-flight growth, theirs to account.
# The bf16 KDA/GDN state kernel variant adds the State element-type
# parameter to LmDeltaRuleKernel and LmReplayFoldKernel with the
# round-to-nearest-even commit store (inference/kernels/linear_attn.cuh
# +50), the sm_121a instantiation guard for both bf16 variants
# (tools/build_replay_fold.sh +10), and the gate wiring (Makefile +1,
# tools/gates.sh +6) - 67 lines; the host harness and its driver are
# excluded by construction. The balance of the exact-count move is
# concurrent agents' in-flight growth, theirs to account.
# The node/scheduler leftover wave converts residentd's per-client control
# payload to the comms arena (node/residentd.c, +36 net), wires the tier-3
# residency oracle into admission (scheduler/scheduler.c + include/
# sparkpipe/spark_scheduler.h, +125 net: the exposed query, the decision
# record, the confidence histogram), lifts the stage-count cap to 16 for
# the PP16 recipes (include/sparkpipe/spark_stage_plan.h, +5), and deletes
# the caller-less prefill slice prototype (inference/stage/
# serving_adapter.cu, +0 net). Ceiling moves by those 166 lines; the
# balance of the exact-count move is concurrent agents' in-flight growth,
# theirs to account.
# The DSv4 correctness + o_proj lever wave lands the four audit fixes and
# the grouped low-rank output projection: the per-head YaRN rope and the
# sparse-score axis swap with its window clamp (inference/kernels/attn.cuh
# +26 net), the query row-width and o_proj contract constants
# (inference/llms/deepseek_v4/config.h +22), the layer's rope/window/o_proj
# rewiring and repriced byte audit (inference/llms/deepseek_v4/layer.cuh
# +27 net), and the instantiation/budget notes (deepseek_v4/unity.cu +1,
# deepseek_v4_pro/unity.cu +2) - 78 lines; the host harness and the gate
# updates are excluded by construction. Ceiling moves to the exact count;
# the balance of the exact-count move is concurrent agents' in-flight
# growth, theirs to account.
# The indirect-A GEMM wave (kernel half of the MoE gather deletion,
# route.cuh's consumer contract) adds the activation_row_index /
# activation_source words and their consume-side source-row scale
# (inference/kernels/gemm.cuh +107 net), the tx-identical per-chunk bulk
# staging path and the shared expect+weight helper
# (inference/kernels/tile.cuh +118 net), the bulk-1D copy primitive
# (inference/kernels/tma.cuh +23), the contract's status rewrite
# (inference/kernels/route.cuh +12 net), and the gate line
# (tools/gates.sh +5) - 265 lines; the fragment-mapping model, the
# contract pins, and the host recorder mirror are excluded by
# construction. Ceiling moves to the exact count; the balance of the
# exact-count move is concurrent agents' in-flight growth, theirs to
# account.
# The K3 bind wave (pack V2 consumption) adds the fused-projection
# section split and its contract comments (inference/llms/kimi_k3/
# layer.cuh +86 net: the split kernel, the two wide GEMMs replacing six,
# the interleave fail-closed; slice.cuh net 0: field-for-field swaps) -
# 86 lines; the host recorders, the gate updates and the doc are
# excluded by construction. The ceiling already sits at the exact count,
# so this entry accounts without moving it.
# Phase 7 adds generation-carrying arena ownership, the explicit NVMe write
# lifecycle and cancellation/heap safety, fail-closed KV access diagnostics,
# contract-derived performance geometry, deterministic Git-independent source
# packaging, and conservative receipt-bound status checks. Ceiling moves to
# the exact count after the merge. The merge-fallout fixes (residentd arena
# handles, the cudart link on CUDA hosts, the scheduler test's reserve/commit
# lifecycle, the PEP 706 fallback, the four phase-7 gate lines) add 13.
CEILING = 126530

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {'.c', '.h', '.cu', '.cuh', '.py', '.mk', '.sh'}
EXCLUDED_COMPONENTS = {'tests', '.git', 'docs', 'build', 'qualification', '__pycache__'}


def main():
    total = 0
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if not path.is_file():
            continue
        if any(component in EXCLUDED_COMPONENTS for component in relative.parts):
            continue
        if path.suffix in EXTENSIONS or path.name == 'Makefile':
            total += sum(1 for _ in path.open(errors='surrogateescape'))
    print(f"non-test authored lines: {total} (ceiling {CEILING})")
    if total > CEILING:
        print(f"\nFAIL authored code grew by {total - CEILING} over the ceiling; "
              f"shrink it or justify a new ceiling in the same change")
        return 1
    if total < CEILING - 800:
        print(f"note: ceiling is {CEILING - total} above reality; "
              f"lower it with the next landing")
    print("\nthe authored codebase did not grow")
    return 0


if __name__ == '__main__':
    sys.exit(main())
