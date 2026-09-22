#!/usr/bin/env bash
# ling_multidev_build.sh — the lane-9 ling family firmware build (M3
# build arm). Runs INSIDE one per-node queue job on the synced checkout;
# produces the five artifacts tools/ling_multidev_run_family.sh consumes
# through LING_PREBUILT_DIR, so serving attaches never pay the nvcc bill
# (the qwen38_27b lane-1 M3 flow, PR #1133):
#
#   FIRMWARE/sparkpipe_model_residentd   the common resident launcher
#   FIRMWARE/weightd_warm                the shared-socket warmer
#   FIRMWARE/model_serving_adapter.so    libling_serving_adapter_<codec>.so
#   FIRMWARE/model_driver.so             the linked ling stage driver
#   FIRMWARE/hidden_transport.so         libhidden_transport_...verbs.so
#   FIRMWARE/SOURCE_COMMIT + SHA256SUMS  the coherence receipt
#
# The build recipe is the wrapper's compile branch verbatim (keep the two
# in lockstep; the lane test pins the shared toolchain targets).
#
# Env:
#   LING_EXPERT_CODEC   bf16 (default) | fp8 - which placed arm to build
#   LING_FIRMWARE_ROOT  output dir (default
#                       /home/<host>/sparkdata/ling-lane9-firmware)
set -euo pipefail

FAMILY="ling"
LANE=9
EXPERT_CODEC="${LING_EXPERT_CODEC:-bf16}"
MODEL_REVISION="e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"
CONTRACT="model_contracts/ling_authoritative.json"
FIRMWARE="examples/model_descriptions/ling_resident_decode_stage_firmware.json"
MODULE="modules/ling_resident_decode_stage"

case "$EXPERT_CODEC" in
  bf16|fp8) ;;
  *) echo "ling-$FAMILY-lane$LANE build: LING_EXPERT_CODEC must be bf16 or fp8" >&2; exit 2 ;;
esac

HOST="$(hostname)"
case "$HOST" in
  spark*) ;;
  *) echo "ling-$FAMILY-lane$LANE build: not on a spark node: $HOST" >&2; exit 2 ;;
esac
FIRMWARE_ROOT="${LING_FIRMWARE_ROOT:-/home/$HOST/sparkdata/ling-lane9-firmware}"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(git -C "$CHECKOUT" rev-parse HEAD)"

PATH="/usr/local/cuda/bin:$PATH"
export PATH

CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
make -C "$CHECKOUT" -j4 \
  build/sparkpipe_model_residentd \
  build/sparkpipe_model_compile \
  build/weightd_warm \
  hidden_transport_spark_host_rdma_verbs
make -C "$CHECKOUT/$MODULE" -j4 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  archive adapter
ADAPTER="$CHECKOUT/build/modules/ling_resident_decode_stage/$EXPERT_CODEC/libling_serving_adapter_$EXPERT_CODEC.so"
[ -f "$ADAPTER" ] || { echo "adapter not built: $ADAPTER" >&2; exit 1; }
BUILD_DRIVER="$CHECKOUT/build/ling-driver.$$"
rm -rf "$BUILD_DRIVER"
"$CHECKOUT/build/sparkpipe_model_compile" \
  --model "$CHECKOUT/$FIRMWARE" \
  --stage ling_resident_decode_stage \
  --library "$CHECKOUT/build/module_library" \
  --output "$BUILD_DRIVER" \
  --cc /usr/bin/cc \
  --include "$CHECKOUT/include" \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
  --cc-arg -lcuda \
  --cc-arg -lcudart \
  --cc-arg -lstdc++ \
  --cc-arg -lm \
  --cc-arg -ldl \
  --cc-arg -pthread

STAGE="$FIRMWARE_ROOT.new.$$"
rm -rf "$STAGE"
mkdir -p "$STAGE"
install -m 0755 "$CHECKOUT/build/sparkpipe_model_residentd" "$STAGE/"
install -m 0755 "$CHECKOUT/build/weightd_warm" "$STAGE/"
install -m 0644 "$ADAPTER" "$STAGE/model_serving_adapter.so"
install -m 0644 "$BUILD_DRIVER/model_driver.so" "$STAGE/model_driver.so"
install -m 0644 \
  "$CHECKOUT/build/libhidden_transport_spark_host_rdma_verbs.so" \
  "$STAGE/hidden_transport.so"
printf '%s\n' "$COMMIT" > "$STAGE/SOURCE_COMMIT"
( cd "$STAGE" && sha256sum sparkpipe_model_residentd weightd_warm \
    model_serving_adapter.so model_driver.so hidden_transport.so \
    > SHA256SUMS )
rm -rf "$BUILD_DRIVER"
mv "$STAGE" "$FIRMWARE_ROOT"
echo "ling-$FAMILY-lane$LANE build: firmware at $FIRMWARE_ROOT commit $COMMIT codec $EXPERT_CODEC"
