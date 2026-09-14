#!/bin/sh
set -eu
cd /tmp/t1qmax/tree
REPOSITORY_ROOT=/tmp/t1qmax/tree make -C modules/qwen38_max_resident_decode_stage archive > /tmp/t1qmax/make_q38.log 2>&1
REPOSITORY_ROOT=/tmp/t1qmax/tree EXPERT_CODEC=nvfp4 \
	MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
	CONTRACT_SHA256=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1) \
	make -C modules/glm5_next_resident_decode_stage archive > /tmp/t1qmax/make_g5n.log 2>&1
make build/libsparkpipe_core.a build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so > /tmp/t1qmax/make_rest.log 2>&1
nvcc -std=c++17 -O3 -arch=sm_121a \
	-I. -Iinclude -Imodel-families/common/include -Imodel-families/qwen38_max/include \
	-Imodules/qwen38_max_resident_decode_stage/include \
	-Imodules/qwen38_max_resident_decode_stage/source \
	-DSPARK_QWEN38_MAX_MODULE_BUILD=1 \
	-DSPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT=0u \
	-DQWEN38_MODEL_REVISION="d2dc35658bcf77e66643428cb52e774cc3b5bd29" \
	tools/t1_qmax_harness.c \
	build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a \
	build/modules/glm5_next_resident_decode_stage/nvfp4/libglm5_next_resident_decode_stage_nvfp4.a \
	build/libsparkpipe_core.a \
	-L/usr/local/cuda/lib64 -lcudart -lcuda -o /tmp/t1qmax/t1_qmax_harness
cp build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so /tmp/t1qmax/
chmod +x /tmp/t1qmax/t1_qmax_harness /tmp/t1qmax/sparkpipe_weightd
ls -la /tmp/t1qmax/t1_qmax_harness /tmp/t1qmax/libhidden_transport_spark_host_rdma_verbs.so /tmp/t1qmax/sparkpipe_weightd
