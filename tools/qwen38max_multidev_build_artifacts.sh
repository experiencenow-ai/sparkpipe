#!/usr/bin/env bash
# qwen38max_multidev_build_artifacts.sh — build the lane-2 wrapper's
# PREBUILT artifact set on THIS node (multidev M3 build arm).
#
# A CPU queue job (nvcc compiles CUDA but needs no GPU device) that
# produces the coherent artifact set the wrapper's QMAX_PREBUILT_DIR
# consumes, in a persistent node-local directory:
#
#   /home/<host>/sparkdata/qwen38max.tp16/build-latest/
#     sparkpipe_model_residentd  model_serving_adapter.so
#     model_driver.so            hidden_transport.so   weightd_warm
#
# Same build chain as the wrapper's inline path (the family's proven
# qwen38_tp4_build.sh flow): root targets, module archive+adapter with
# the pinned serving revision, then sparkpipe_model_compile for the
# driver. Atomic: builds into build-partial.$$ and renames on success.
# Queue cmd stays BARE.
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

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="/home/$HOST/$OUT_REL"
PARTIAL="$OUT/../build-partial.$$"
rm -rf "$PARTIAL"
mkdir -p "$PARTIAL"
trap 'rm -rf "$PARTIAL"' EXIT

PATH="/usr/local/cuda/bin:$PATH"
export PATH
CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
START="$(date +%s)"

make -C "$CHECKOUT" -j2 \
  build/sparkpipe_model_residentd \
  build/sparkpipe_model_compile \
  build/weightd_warm \
  hidden_transport_spark_host_rdma_verbs
make -C "$CHECKOUT/modules/qwen38_max_resident_decode_stage" -j2 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  archive adapter
ADAPTER="$CHECKOUT/build/modules/qwen38_max_resident_decode_stage/$EXPERT_CODEC/libqwen38_max_serving_adapter_$EXPERT_CODEC.so"
[ -f "$ADAPTER" ] || fail "adapter not built: $ADAPTER"
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
