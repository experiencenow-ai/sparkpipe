MODULE_FAMILY := qwen38
include ../../runtime/weightd_sources.mk

ifndef EXPERT_CODEC
$(error EXPERT_CODEC is required: fp8)
endif
QWEN38_MAX_EXPERT_CODECS := fp8
ifeq ($(filter $(EXPERT_CODEC),$(QWEN38_MAX_EXPERT_CODECS)),)
$(error unsupported EXPERT_CODEC '$(EXPERT_CODEC)': choose $(QWEN38_MAX_EXPERT_CODECS))
endif
ifndef MODEL_REVISION
$(error MODEL_REVISION is required and must be the exact source snapshot revision)
endif
ifndef CONTRACT_SHA256
$(error CONTRACT_SHA256 is required and must identify the exact model package contract)
endif

MODULE_IDENTIFIER := spark.qwen38.resident_decode_stage.fp8.h8192.l92.gdn69.e512k10.v1
MODULE_TARGET := cuda.sm121.qwen38.resident_decode_stage.fp8
MODEL_HEADER := ../../model-families/qwen38_max/include/sparkpipe/spark_qwen38_max_model.h
MODULE_ENTRY_PREFIX := SparkQwen38MaxResidentDecodeStage
MODULE_HOST_SOURCE := $(CURDIR)/source/spark_qwen38_max_resident_decode_stage_module.c
MODULE_CUDA_SOURCE := $(CURDIR)/source/spark_qwen38_max_resident_decode_stage_cuda.cu
MODULE_ADDITIONAL_HOST_SOURCES = \
	$(REPOSITORY_ROOT)/runtime/stage_module_lifecycle.c \
	$(REPOSITORY_ROOT)/runtime/stagepack_format.c \
	$(addprefix $(REPOSITORY_ROOT)/,$(SPARKPIPE_WEIGHTD_SOURCES)) \
	$(REPOSITORY_ROOT)/cache/store/stage_kv_client.c \
	$(REPOSITORY_ROOT)/cache/store/kv_store.c \
	$(REPOSITORY_ROOT)/model-families/qwen38_max/src/spark_qwen38_max_work_control.c \
	$(REPOSITORY_ROOT)/ring/transport/hidden_transport.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_collective.c \
	$(REPOSITORY_ROOT)/ring/transport/tp_device_collective.c
# Adapter provenance pins: the vendor FP8 checkpoint revision and the exact
# package contract, checked at adapter compile time (the adapter #errors
# without both, so a drifted build cannot link silently). Passed in by the
# build invocation (MODEL_REVISION / CONTRACT_SHA256), the glm5_next
# convention. REPOSITORY_ROOT is only defined inside
# resident_decode_stage_rules.mk, so this Makefile must never expand it
# eagerly: the previous in-Makefile contract digest ran $(shell sha256sum
# $(REPOSITORY_ROOT)/...) at parse time - before the include - which pinned
# an EMPTY contract sha into every artifact (build-r2 attempt 1c1077ce).
MTP_LAYER_COUNT ?= 0
MODULE_COMPILE_FLAGS := \
	-DSPARK_QWEN38_MAX_MODULE_BUILD=1 \
	-DSPARK_LLM_MTP_LAYER_COUNT=$(MTP_LAYER_COUNT)u \
	-DQWEN38_MODEL_REVISION=\"$(MODEL_REVISION)\" \
	-DQWEN38_CONTRACT_SHA256=\"$(CONTRACT_SHA256)\"
MODULE_INCLUDE_FLAGS = \
	-I"$(REPOSITORY_ROOT)/include" \
	-I"$(REPOSITORY_ROOT)/src" \
	-I"$(REPOSITORY_ROOT)/model-families/common/include" \
	-I"$(REPOSITORY_ROOT)/model-families/qwen38_max/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_max_resident_decode_stage/include" \
	-I"$(REPOSITORY_ROOT)/modules/qwen38_max_resident_decode_stage/source" \
	-I"$(REPOSITORY_ROOT)"

# Codec-partitioned build outputs (glm5_next layout): the wrapper resolves the
# adapter at build/modules/qwen38_max_resident_decode_stage/<codec>/lib..._so.
# ?= keeps the expansion deferred until resident_decode_stage_rules.mk has
# defined REPOSITORY_ROOT; the rules default would name the family
# (qwen38_resident_decode_stage, without the _max) and drop the codec leg.
BUILD_DIRECTORY ?= $(REPOSITORY_ROOT)/build/modules/qwen38_max_resident_decode_stage/$(EXPERT_CODEC)
MODULE_ARCHIVE ?= $(BUILD_DIRECTORY)/libqwen38_max_resident_decode_stage_$(EXPERT_CODEC).a
GPU_VALIDATOR := validation/validate_qwen38_max_resident_decode_stage_cuda.sh

STAGE_PACK_PATH ?= $(REPOSITORY_ROOT)/build/stagepacks/qwen38_stage.qwen38sp
STAGE_COUNT ?= 16
STAGE_INDEX ?= 0
STAGE_FIRST_LAYER ?= 0
STAGE_LAYER_COUNT ?= 23
MAX_ACTIVE_SEQUENCES ?= 32
PIPELINE_SLOT_COUNT ?= 2
KV_BLOCK_COUNT ?= 4096
GDN_SNAPSHOT_SLOT_COUNT ?= 0
KV_PROVIDER ?= none
KV_SERVICE_ADDRESS ?= none
KV_SOCKET_PATH ?= none
KV_POOL_BYTES ?= 0
KV_WORKER_COUNT ?= 0
ALLOW_UNQUALIFIED_EXECUTION ?= 0

RUNTIME_CONFIGURATION := \
	SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION=$(ALLOW_UNQUALIFIED_EXECUTION) \
	SPARK_QWEN38_MAX_STAGE_PACK_PATH=$(STAGE_PACK_PATH) \
	SPARK_QWEN38_MAX_STAGE_COUNT=$(STAGE_COUNT) \
	SPARK_QWEN38_MAX_STAGE_INDEX=$(STAGE_INDEX) \
	SPARK_QWEN38_MAX_STAGE_FIRST_LAYER=$(STAGE_FIRST_LAYER) \
	SPARK_QWEN38_MAX_STAGE_LAYER_COUNT=$(STAGE_LAYER_COUNT) \
	SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES=$(MAX_ACTIVE_SEQUENCES) \
	SPARK_QWEN38_MAX_STAGE_PIPELINE_SLOTS=$(PIPELINE_SLOT_COUNT) \
	SPARK_QWEN38_MAX_STAGE_KV_BLOCKS=$(KV_BLOCK_COUNT) \
	SPARK_QWEN38_MAX_STAGE_KV_STORE=$(KV_PROVIDER) \
	SPARK_QWEN38_MAX_STAGE_KV_SERVICE=$(KV_SERVICE_ADDRESS) \
	SPARK_QWEN38_MAX_STAGE_KV_SOCKET=$(KV_SOCKET_PATH) \
	SPARK_QWEN38_MAX_STAGE_KV_POOL_BYTES=$(KV_POOL_BYTES) \
	SPARK_QWEN38_MAX_STAGE_KV_WORKERS=$(KV_WORKER_COUNT) \
	SPARK_QWEN38_MAX_STAGE_MTP=$(MTP_LAYER_COUNT) \
	SPARK_QWEN38_MAX_STAGE_GDN_SNAPSHOT_SLOTS=$(GDN_SNAPSHOT_SLOT_COUNT)

include ../resident_decode_stage_rules.mk

# Serving adapter: the shared object the residentd dlopens. Nothing else in
# the tree builds it (build-r2: "No rule to make target 'adapter'" on all 16
# nodes); this is the glm5_next module-local pattern, plus the real CUDA
# runtime link this adapter needs (it includes cuda_runtime.h directly).
ADAPTER_LIBRARY := $(BUILD_DIRECTORY)/libqwen38_max_serving_adapter_$(EXPERT_CODEC).so

.PHONY: adapter
adapter:
	$(MAKE) -C $(REPOSITORY_ROOT) core model_common
	@mkdir -p "$(BUILD_DIRECTORY)"
	$(CC) $(CFLAGS) $(MODULE_POSIX_FLAGS) -fPIC -shared \
		-I$(REPOSITORY_ROOT)/src -I"$(CUDA_HOME)/include" $(MODULE_INCLUDE_FLAGS) \
		$(MODULE_COMPILE_FLAGS) \
		source/spark_qwen38_max_serving_adapter.c \
		$(REPOSITORY_ROOT)/build/libsparkpipe_runtime.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_model_common.a \
		$(REPOSITORY_ROOT)/build/libsparkpipe_core.a \
		-L"$(CUDA_HOME)/lib64" -lcudart -ldl -pthread \
		-o "$(ADAPTER_LIBRARY)"
