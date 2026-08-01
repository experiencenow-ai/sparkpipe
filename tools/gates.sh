#!/bin/sh
# Every gate, each checked on its exit code.
#
# This exists because a previous run reported nine passing static_asserts from a
# translation unit that had failed to compile: the echo was not gated on the
# status. A gate that cannot fail is not a gate.
ok=0; bad=0; skipped=0
run() {
	if eval "$2" >/dev/null 2>&1
	then printf "  %-26s PASS\n" "$1"; ok=$((ok+1))
	else printf "  %-26s FAIL\n" "$1"; bad=$((bad+1))
	fi
}
run_cuda() { if [ -x "${CUDA_HOME:-/opt/cuda}/bin/nvcc" ]; then run "$1" "$2"; else printf "  %-26s SKIP (nvcc unavailable)\n" "$1"; skipped=$((skipped+1)); fi; }
run_cuda "ptx capability gate" "python3 tests/test_ptx_capability_gate.py"
run "complete host inventory" "make -s test"
# Phase 7: required-KV access failures are structured device errors that fail
# the stream - dense, selected, sparse-score, summary and refinement paths
# exercised against a host harness with planted faults.
run "kv failure host"      "python3 tests/test_kv_failure_host.py"
# Phase 7: --use_fast_math must not silently rewrite the numerics the
# contracts pin (the build-policy audit's list of forbidden flags).
run "cuda math policy"     "python3 tests/test_cuda_math_policy.py"
# Phase 7: STATUS.md is receipt-bound - it may only claim what the gates
# actually executed.
run "status truth"         "python3 tests/test_status_truth.py"
run "mma fragment mapping" "gcc -O2 -Wall -Wextra -o /tmp/g_f tests/test_mma_fragment_mapping.c && /tmp/g_f"
run "model constants"      "gcc -O2 -Wall -Wextra -I. -Imodel-families/glm52/include -o /tmp/g_c tests/test_model_constants.c && /tmp/g_c"
run "sub-byte packing"     "gcc -O2 -Wall -Wextra -o /tmp/g_p tests/test_pack.c && /tmp/g_p"
run "free dequant"         "gcc -O2 -Wall -Wextra -o /tmp/g_d tests/test_dequant.c && /tmp/g_d"
run "reference oracle"     "gcc -O2 -Wall -Wextra -Itests -o /tmp/g_r tests/test_reference.c -lm && /tmp/g_r"
run "weight binding"       "gcc -O2 -Wall -Wextra -I. -Imodules/glm52_resident_decode_stage/include -Iinclude -Ideployment/include -Imodel-families/glm52/include -o /tmp/g_b tests/test_pack_bind.c && /tmp/g_b"
run "sidebands"            "gcc -O2 -Wall -Wextra -I. -o /tmp/g_s tests/test_sideband.c && /tmp/g_s"
run "kv cache"             "gcc -O2 -Wall -Wextra -I. -o /tmp/g_kv tests/test_cache.c && /tmp/g_kv"
run "kv geometry"          "g++ -std=c++17 -fsyntax-only -Wall -Wextra -I. -Imodel-families/glm52/include tests/test_kv_geometry.cc"
run "workspace layout"     "gcc -O2 -Wall -Wextra -I. -o /tmp/g_w tests/test_group_gemm_workspace.c && /tmp/g_w"
run "tensor map geometry"  "gcc -O2 -Wall -Wextra -I. -o /tmp/g_t tests/test_tensor_map_geometry.c && /tmp/g_t"
run "tensor map encode"    "gcc -O2 -Wall -Wextra -I. -Itests/cuda_driver_stub -o /tmp/g_e tests/test_tensor_map_encode.c tests/cuda_driver_stub/stub.c && /tmp/g_e"
run "launch planning"      "g++ -std=c++17 -O2 -Wall -Wextra -I. -D__host__= -D__device__= -o /tmp/g_l tests/test_launch.c && /tmp/g_l"
run "config coverage"      "python3 tests/test_config_coverage.py"
# Carried forward from #514, whose patch targeted a file the rewrite deleted.
# The hazard survived the rewrite with a different failure mode: the old tile
# staged out-of-row data past the K bound, the new one drops the tail via
# k_tiles = input_dimension / TILE_K. Both are wrong output with no crash.
run "gemm K alignment"     "python3 tests/test_gemm_k_alignment.py"
run "rope pairing"         "python3 tests/test_rope_pairing.py"
run "layer kinds"          "python3 tests/test_layer_kinds.py"
run "situ activation"      "python3 tests/test_situ_activation.py"
run "kda decay bound"      "python3 tests/test_kda_decay.py"
run "kernel launches"      "python3 tests/test_kernel_launches.py"
run "mla absorption"       "python3 tests/test_mla_absorption.py"
run "expert grouping"      "python3 tests/test_expert_grouping.py"
# The kernel half of the MoE gather deletion (route.cuh's row-indirection
# contract): the row-map words, the ragged-tail clamp, source-following
# scales, and both staging paths declaring the same barrier bytes. The
# arithmetic model behind the pins is the mma fragment mapping gate above.
run "grouped moe contracts" "python3 tests/test_grouped_moe_source_contracts.py"
# The real kernels, run on a CPU. Not a reimplementation: kda_host.cu includes
# inference/kernels/linear_attn.cuh unmodified and gives it a grid. Reverting
# either of the two bugs this path had - the undecayed prediction, the dropped
# dt_bias - takes the relative error from 2e-3 to 3e-1 and 6e-2.
run "kda on host"          "python3 tests/test_kda_host.py"
# The bf16-state variant of the same kernels, both instantiations run side by
# side: bit-identical per-step math, the commit store exactly round-to-
# nearest-even, the divergence from arbitrary fp32 state inside the geometric
# envelope over 64 commits, and the replay fold byte-exact against serial
# decode. This is the kernel-side contract the kda_state_bf16 flag waits on.
run "kda bf16 state"       "python3 tests/test_kda_bf16_state.py"
# The routing path had three defects, all found by reading and none by running.
# Emitting the biased score as the weight produces 9 failures here; skipping the
# renormalisation produces 17.
run "router on host"       "python3 tests/test_router_host.py"
run "router fp32 contract"  "python3 tests/test_router_precision_contract.py"
# Six more kernels the other two harnesses do not reach, including the MoE
# finalize whose launch was wrong four ways and compiled.
run "layer on host"        "python3 tests/test_layer_host.py"
# The MLA store and attention over a paged cache, two sequences with interleaved
# pages so ignoring the page table is visible.
run "mla on host"          "python3 tests/test_mla_host.py"
# The per-head KV pair qwen_3_6 and mimo_2_5 actually launch: store packing
# [K|V] into the slot, then GQA decode. Both drivers previously stored a buffer
# nothing wrote and attended through the MLA latent kernel, which returns the
# key's prefix as the value.
run "gqa on host"          "python3 tests/test_gqa_host.py"
# A whole Qwen 3.6 layer on a CPU: the GEMM recorded, every other kernel the
# one that ships. Catches the wiring defects per-kernel tests cannot - a cache
# stored from a buffer nothing wrote, a state pool half the kernel's stride.
run "qwen36 layer on host" "python3 tests/test_qwen36_layer_host.py"
# MiMo 2.5's two attention branches: full and sliding-window, same recorder
# pattern, plus the 0.707 value scale and the narrower-than-key value width
# only this driver has.
run "mimo25 layer on host" "python3 tests/test_mimo25_layer_host.py"
# Dataflow, not arithmetic. Every per-kernel harness passes and an audit still
# found three defects in which buffer feeds which kernel. Reintroducing the
# shared-expert overwrite makes this fail.
run "layer dataflow"       "python3 tests/test_layer_dataflow.py"
# The checkpoint quantises the routed experts and nothing else, because only
# they saw quantisation-aware training. Putting attention back on Format fails.
run "k3 quant recipe"      "python3 tests/test_k3_quant_recipe.py"
run "glm52 precision"      "python3 tests/test_glm52_quantized_cuda_contract.py"
run "glm52 unity precision" "python3 tests/test_glm52_unity_precision_contract.py"
# The whole GLM layer, executed. The kv projections write the cache slot
# layout directly - a reintroduced join or a swapped slot offset fails the
# stored-slot check - and the head's norm must fold the residual stream, which
# it did not do until this harness disagreed with it.
run "glm52 layer on host"  "python3 tests/test_glm52_layer_host.py"
# A whole layer, executed. Found a divide-by-zero in production code on its
# first successful run, and catches the shared-expert overwrite by seeing the
# routed value missing from the output rather than by reading the source.
run "k3 layer on host"     "python3 tests/test_k3_layer_host.py"
run "k3 slice on host"     "python3 tests/test_k3_slice_host.py"
run "k3 engine on host"    "python3 tests/test_k3_engine.py"
run "k3 driver contracts"  "python3 tests/test_k3_driver_contracts.py"
run "k3 pack"              "python3 tests/test_k3_pack.py"
# The V2 layout - fused KDA section tables, the zero-padding interleave grid,
# the 128-byte alignment - is integer arithmetic, proven here with no numpy.
run "k3 pack layout"       "python3 tests/test_k3_pack_layout.py"
run "k3 tp shard"          "python3 tests/test_k3_shard.py"
run "k3 shard table"       "python3 tests/test_k3_shard_table.py"
# The recipe generator: deterministic bytes, geometry-hash invalidation, TP
# shard coverage against k3_shard's own table, PP balance against a
# brute-force optimum - then --check that the committed examples/recipes/
# set is current for the committed contracts (a stale content-hash in a
# filename fails here before a node ever picks it up).
run "recipe generation"    "python3 tests/test_recipe_generation.py"
run "recipes current"      "python3 tools/generate_recipe.py --check"
run "k3 stage doorway"     "gcc -Iinclude -Imodel-families/k3/include -Imodel-families/glm52/include -Wall -Werror -DNDEBUG -c modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_validation.c -o /tmp/g_k3v.o"
run "mimo25 stage doorway"  "gcc -Iinclude -Wall -Werror -DNDEBUG -c modules/mimo25_resident_decode_stage/source/spark_mimo25_resident_decode_stage_validation.c -o /tmp/g_mimo25v.o"
run "qwen36 stage doorway"  "gcc -Iinclude -Wall -Werror -DNDEBUG -c modules/qwen36_resident_decode_stage/source/spark_qwen36_resident_decode_stage_validation.c -o /tmp/g_qwen36v.o"
run "dsv4 stage doorway"  "gcc -Iinclude -Wall -Werror -DNDEBUG -c modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_validation.c -o /tmp/g_dsv4v.o"
run "stage dispatch host compile" "g++ -x c++ -std=c++17 -fsyntax-only -Wall -Wextra -Werror -I. -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -Itests/cuda_stub inference/stage/dispatch.cu"
run "hybrid kv arithmetic"   "make -s build/test_hybrid_kv_arithmetic && ./build/test_hybrid_kv_arithmetic"
run "uniform-profile admit"  "make -s build/test_uniform_profile_admit && ./build/test_uniform_profile_admit"
run "null seam link+run"   "make -s build/test_null_seam_link && ./build/test_null_seam_link"
run "seam symbol parity"   "sh tools/seam_parity.sh"
run "stage module + model"  "gcc -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -Wall -Werror -DNDEBUG -c inference/stage/module.c -o /tmp/g_mod.o && gcc -Iinclude -Imodules/glm52_resident_decode_stage/include -Imodules/glm52_resident_decode_stage/source -Imodel-families/glm52/include -Wall -Werror -DNDEBUG -c modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_validation.c -o /tmp/g_val.o"
run "k3 kv seam"          "gcc -O2 -Wall -Wextra -Iinclude -Imodel-families/glm52/include -Imodel-families/k3/include -o /tmp/g_k3kv tests/test_k3_kv_cache.c cache/kv_cache.c && /tmp/g_k3kv"
run "state pool"          "gcc -O2 -Wall -Wextra -I. -o /tmp/g_sp tests/test_state_pool.c && /tmp/g_sp"
run "comms arena"         "gcc -O2 -Wall -Wextra -Werror -I. -o /tmp/g_ar tests/test_arena.c && /tmp/g_ar"
run "node daemons compile"  "make -s build/sparkpipe_glm52_cuda_residentd build/sparkpipe_glm52_ring_rank_daemon"
run "code size"           "python3 tests/test_code_size.py"
run "dry naming law"       "python3 tests/test_dry_law.py"
# The batch-variant contract: one source tree, eleven power-of-two
# capacity-ceiling modules (b1..b1024) per resident family from ONE recipe
# template, every per-bucket constant in the family's tuning header, every
# name spelled once. A per-bucket fork, a respelled module ID, or a .PHONY
# target nobody defined all fail here. The ceiling/tile/module-ID selection
# runs compiled, per bucket, on the host.
run "batch variants"       "python3 tests/test_batch_variants.py"
# The grouped selection path has no model in this tree, so nothing instantiates
# it and nothing would notice it failing to compile.
run_cuda "grouped topk builds"  "sh tools/build_grouped_topk.sh"
run_cuda "replay fold builds"   "sh tools/build_replay_fold.sh"
run_cuda "head topk builds"     "sh tools/build_head_topk.sh"
run "head topk on host"    "python3 tests/test_head_host.py"
run_cuda "nvcc: sm_121a build"  "sh tools/build.sh"
run "makefile parses"      "make -n all"
run "makefile: tools"      "make -n tools"
run "makefile: backend"    "make -n glm52_ring_service_backend"
run "makefile: variants"   "make -n -C modules/glm52_resident_decode_stage variants"
run "makefile: variant publish" "make -n -C modules/glm52_resident_decode_stage publish_variants"
run "core boundaries"      "make -s audit-boundaries"
run "no python in production" "python3 tests/test_no_python_in_production.py"
run "package manifest"     "python3 tools/verify_package_manifest.py"
printf "  ---- %d pass, %d skip, %d fail\n" "$ok" "$skipped" "$bad"
[ "$bad" -eq 0 ]
