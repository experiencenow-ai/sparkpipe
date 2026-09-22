#!/usr/bin/env bash
# qwen38max_multidev_build_artifacts.sh — build the lane-2 wrapper's
# PREBUILT artifact set on THIS node (multidev M3 build arm).
#
# A GPU-owned queue job (glm5_next build precedent): nvcc compiles CUDA
# and the module publish step executes retained-receipt GPU validation,
# so a CPU-only reservation is insufficient. Produces the coherent
# artifact set the wrapper's QMAX_PREBUILT_DIR consumes, in a persistent
# node-local directory:
#
#   /home/<host>/sparkdata/qwen38max.tp16/build-latest/
#     sparkpipe_model_residentd  model_serving_adapter.so
#     model_driver.so            hidden_transport.so   weightd_warm
#
# Same build chain as the wrapper's inline path (the family's proven
# qwen38_tp4_build.sh flow): root targets, module archive+adapter with
# the pinned serving revision, module publish (whole-stack smoke tier on
# the rank-0 placed pack, the only pack a single-node TP1 validation may
# touch), then sparkpipe_model_compile for the driver — the driver
# compile resolves the module from build/module_library, so the publish
# MUST precede it. Atomic: builds into build-partial.$$ and renames on
# success. Queue cmd stays BARE.
set -euo pipefail

WORLD=16
EXPERT_CODEC="fp8"
MODEL_REVISION="d2dc35658bcf77e66643428cb52e774cc3b5bd29"
CONTRACT="model_contracts/qwen38_authoritative.json"
OUT_REL="sparkdata/qwen38max.tp16/build-latest"

fail() { echo "qwen38max-build-artifacts: $*" >&2; exit 1; }

ATTEMPT="${SPARK_QUEUE_ATTEMPT:?run through the authoritative spark queue}"
case "$ATTEMPT" in
  *[!0-9a-f]*|""|?????????????????????????????????*) fail "bad attempt id" ;;
esac
[ "${#ATTEMPT}" -eq 32 ] || fail "bad attempt id length"
# Single-node targeted form or the 16-node --per-node fleet form (each
# node builds its own artifacts).
case "${SPARK_QUEUE_SIZE:-1}" in 1|16) ;; *) fail "size must be 1 or 16" ;; esac

HOST="$(hostname)"
case "$HOST" in spark[0-9a-f]) ;; *) fail "unexpected hostname '$HOST'" ;; esac
# This node's OWN placed pack (one pack per node: rank i lives on
# spark{hex(i)} — tools/qwen38max_multidev_pack_emit.sh placement). The
# publish validates the module against the same pack this node's driver
# will attach; a rank0 hardcode would fail the pack check on 15/16 nodes.
NODE_RANK="$((16#${HOST#spark}))"
PACK="/home/$HOST/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank$NODE_RANK.sp"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="/home/$HOST/$OUT_REL"
PARTIAL="$OUT/../build-partial.$$"
rm -rf "$PARTIAL"
mkdir -p "$PARTIAL"
trap 'rm -rf "$PARTIAL"' EXIT

[ -r "$PACK" ] || fail "placed pack for this node missing: $PACK (operator-placed NVMe set; no pack, no publish)"

PATH="/usr/local/cuda/bin:$PATH"
export PATH
CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
START="$(date +%s)"

make -C "$CHECKOUT" -j4 \
  build/sparkpipe_model_residentd \
  build/sparkpipe_model_compile \
  build/weightd_warm \
  hidden_transport_spark_host_rdma_verbs
make -C "$CHECKOUT/modules/qwen38_max_resident_decode_stage" -j4 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  archive adapter
ADAPTER="$CHECKOUT/build/modules/qwen38_max_resident_decode_stage/$EXPERT_CODEC/libqwen38_max_serving_adapter_$EXPERT_CODEC.so"
[ -f "$ADAPTER" ] || fail "adapter not built: $ADAPTER"
# Publish the validated module into build/module_library: the driver
# compile below resolves the module by exact identity from the library
# and fails with MODULE_NOT_VALIDIFIED without this. Whole-stack smoke
# tier (STAGE_COUNT=1, all 92 layers, TP1, MAS=8) on this node's own
# placed pack — the validator's admitted single-node tier (qwen38_27b
# publish precedent: tp4-rank0 pack + standalone whole-stack).
make -C "$CHECKOUT/modules/qwen38_max_resident_decode_stage" -j2 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  STAGE_PACK_PATH="$PACK" \
  STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=92 \
  MTP_LAYER_COUNT=0 MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 \
  ALLOW_UNQUALIFIED_EXECUTION=1 \
  publish
"$CHECKOUT/build/sparkpipe_model_compile" \
  --model "$CHECKOUT/examples/model_descriptions/qwen38_max_resident_decode_stage_firmware.json" \
  --stage qwen38_max_resident_decode_stage \
  --library "$CHECKOUT/build/module_library" \
  --output "$PARTIAL/driver" \
  --cc /usr/bin/cc \
  --include "$CHECKOUT/include" \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
  --cc-arg -lcuda \
  --cc-arg -lcudart \
  --cc-arg -lstdc++ \
  --cc-arg -lm \
  --cc-arg -ldl \
  --cc-arg -pthread
install -m 0755 "$CHECKOUT/build/sparkpipe_model_residentd" "$PARTIAL/"
install -m 0755 "$CHECKOUT/build/weightd_warm" "$PARTIAL/"
install -m 0644 "$ADAPTER" "$PARTIAL/model_serving_adapter.so"
install -m 0644 "$PARTIAL/driver/model_driver.so" "$PARTIAL/"
install -m 0644 \
  "$CHECKOUT/build/libhidden_transport_spark_host_rdma_verbs.so" \
  "$PARTIAL/hidden_transport.so"
rm -rf "$PARTIAL/driver"
rm -rf "$OUT" && mv "$PARTIAL" "$OUT"
END="$(date +%s)"
echo "qwen38max-build-artifacts: $HOST -> $OUT in $((END - START))s"
for artifact in sparkpipe_model_residentd model_serving_adapter.so \
    model_driver.so hidden_transport.so weightd_warm; do
  [ -f "$OUT/$artifact" ] || fail "missing artifact: $OUT/$artifact"
done
echo "BUILD-DONE host=$HOST seconds=$((END - START))"
