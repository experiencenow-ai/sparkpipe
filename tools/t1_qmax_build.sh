#!/bin/sh
set -eu
cd /tmp/t1qmax/tree
contract_sha=$(sha256sum model_contracts/qwen38_authoritative.json | cut -d' ' -f1)
nvcc -std=c++17 -O2 \
	-I. -Iinclude -Imodel-families/common/include -Imodel-families/qwen38_max/include \
	-Imodules/qwen38_max_resident_decode_stage/include \
	-Imodules/qwen38_max_resident_decode_stage/source \
	-DSPARK_QWEN38_MAX_MODULE_BUILD=1 \
	-DSPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT=0u \
	-DQWEN38_MODEL_REVISION="d2dc35658bcf77e66643428cb52e774cc3b5bd29" \
	-DQWEN38_CONTRACT_SHA256="$contract_sha" \
	tools/t1_qmax_harness.c \
	modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_cuda.cu \
	modules/qwen38_max_resident_decode_stage/source/spark_qwen38_max_resident_decode_stage_module.c \
	runtime/stage_module_lifecycle.c runtime/stagepack_format.c \
	runtime/spark_weightd.c runtime/spark_weightd_manifest.c runtime/spark_weightd_lease.c \
	runtime/spark_weightd_attach.c runtime/spark_weightd_map.c runtime/spark_weightd_spine.c \
	runtime/spark_weightd_worker.c runtime/spark_weightd_lazy_pack.c \
	cache/store/stage_kv_client.c cache/store/kv_store.c \
	model-families/qwen38_max/src/spark_qwen38_max_work_control.c \
	ring/transport/hidden_transport.c ring/transport/tp_collective.c \
	ring/transport/tp_device_collective.c \
	-L/usr/local/cuda/lib64 -lcudart -o /tmp/t1qmax/t1_qmax_harness
make build/libhidden_transport_spark_host_rdma_verbs.so build/sparkpipe_weightd > /tmp/t1qmax/make.log 2>&1
cp build/libhidden_transport_spark_host_rdma_verbs.so build/sparkpipe_weightd /tmp/t1qmax/
chmod +x /tmp/t1qmax/t1_qmax_harness /tmp/t1qmax/sparkpipe_weightd
ls -la /tmp/t1qmax/t1_qmax_harness /tmp/t1qmax/libhidden_transport_spark_host_rdma_verbs.so /tmp/t1qmax/sparkpipe_weightd
