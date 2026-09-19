#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"

#define _GNU_SOURCE 1
#include <cuda_runtime.h>

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)


int SparkGlm5NextLaunchMeshCopyDown(void *stream, volatile void *destination, const void *source, uint64_t bytes)
{
	(void)stream; (void)destination; (void)source; (void)bytes;
	return(0);
}

int SparkGlm5NextLaunchMeshGuard(void *stream, volatile void *error_word, void *output)
{
	(void)stream; (void)error_word; (void)output;
	return(0);
}

extern uint32_t cuda_stub_mesh_publish_calls;
extern uint32_t cuda_stub_mesh_publish_null_seq_cell;
extern uint32_t cuda_stub_mesh_publish_null_epoch_cell;

static uint64_t mock_client_alive = 1u;

SparkStatus SparkWeightdClientConnect(const char *socket_path, SparkWeightdClient **client, SparkWeightdHelloResult *hello_out)
{
	(void)socket_path; (void)hello_out;
	*client = (SparkWeightdClient *)calloc(1u, 64u);
	return(SPARK_STATUS_OK);
}

void SparkWeightdClientClose(SparkWeightdClient *client)
{
	free(client);
}

uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client)
{
	(void)client;
	return((uint32_t)mock_client_alive);
}

SparkStatus SparkWeightdClientMeshBroadcast(SparkWeightdClient *client, uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset, uint32_t length, uint64_t seq_value, uint64_t seq_remote_offset, uint64_t timeout_nanoseconds)
{
	(void)client; (void)peer_mask; (void)source_offset; (void)remote_offset;
	(void)length; (void)seq_value; (void)seq_remote_offset; (void)timeout_nanoseconds;
	return(SPARK_STATUS_OK);
}

static uint32_t mock_combine_calls;

static SparkStatus TestCombineFusedBf16(void *combine_context, void *destination_device, const void *const *source_devices, uint32_t source_count, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context; (void)destination_device; (void)source_devices;
	(void)source_count; (void)active_sequence_count; (void)hidden_dimension; (void)cuda_stream;
	mock_combine_calls++;
	return(SPARK_STATUS_OK);
}

static SparkStatus TestCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)source_device;
	return(TestCombineFusedBf16(combine_context,destination_device,0,0,active_sequence_count,hidden_dimension,cuda_stream));
}

static SparkStatus TestCombineU64Max(void *combine_context, uint64_t *destination_device, const uint64_t *source_device, uint32_t element_count, void *cuda_stream)
{
	(void)combine_context; (void)destination_device; (void)source_device; (void)element_count; (void)cuda_stream;
	mock_combine_calls++;
	return(SPARK_STATUS_OK);
}

static void TestComplete(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	(void)context; (void)completion;
}

int main(void)
{
	SparkTpDeviceCollectiveConfig config;
	(void)setenv("SPARK_WEIGHTD_SOCKET","/tmp/tp_collective_mock.sock",1);
	SparkTpDeviceCollective collective;
	SparkStatus status;
	void *mesh_buffer;
	void *device_scratch;
	uint8_t *peer_tails;

	cuda_stub_mesh_publish_calls = 0u;
	cuda_stub_mesh_publish_null_seq_cell = 0u;
	cuda_stub_mesh_publish_null_epoch_cell = 0u;
	mock_combine_calls = 0u;

	mesh_buffer = calloc(1u, (size_t)SPARK_WEIGHTD_MESH_REGION_BYTES);
	if ( mesh_buffer == 0 )
	{
		fprintf(stderr,"mesh buffer alloc failed\n");
		return(1);
	}

	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	config.tp_degree = 2u;
	config.tp_rank = 0u;
	config.local_hidden_dimension = 64u;
	config.max_active_sequence_count = 4u;
	config.connect_timeout_milli = 1000u;
	config.operation_timeout_milli = 2000u;
	config.collective_identifier = 0u;
	config.combine_bf16_function = TestCombineBf16;
	config.combine_fused_bf16_function = TestCombineFusedBf16;
	config.combine_u64_max_function = TestCombineU64Max;

	memset(&collective,0,sizeof(collective));
	status = SparkTpDeviceCollectiveCreate(&config, &collective);
	CHECK(status == SPARK_STATUS_OK, "create");

	cudaMalloc(&device_scratch, 4096u);
	status = SparkTpDeviceCollectivePrepareReceiveBf16(&collective, mesh_buffer, 2u, 64u, 0u, 0);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"prepare status=%d\n",(int)status);
	CHECK(status == SPARK_STATUS_OK, "prepare receive");

	peer_tails = (uint8_t *)mesh_buffer + (1u * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK) * SPARK_WEIGHTD_MESH_SLOT_BYTES;

	{
		SparkTpDeviceCollectiveSubmission submission;
		memset(&submission,0,sizeof(submission));
		submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
		submission.descriptor_bytes = sizeof(submission);
		submission.slot_index = 0u;
		submission.active_sequence_count = 2u;
		submission.ordinal = 1u;
		submission.local_device = device_scratch;
		submission.full_device = device_scratch;
		submission.cuda_stream = (void *)0x1;
		submission.completion_function = TestComplete;

		status = SparkTpDeviceCollectiveEnqueue(&collective, &submission,
			SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
		CHECK( cuda_stub_mesh_publish_calls >= 1u, "eager round publishes (kernel publish ran)");
		CHECK( cuda_stub_mesh_publish_null_seq_cell == 0u, "eager publish had a live seq_cell (no NULL atomic)");
		CHECK( cuda_stub_mesh_publish_null_epoch_cell == 0u, "eager publish had a live epoch_cell");
	}

	(void)peer_tails;

	SparkTpDeviceCollectiveDestroy(&collective);
	free(mesh_buffer);

	fprintf(stderr,"%s: %u checks, %u failures (publish=%u combine=%u)\n",
		"test_tp_device_collective_mock", test_checks, test_failures,
		cuda_stub_mesh_publish_calls, mock_combine_calls);
	return( test_failures != 0u ? 1 : 0 );
}
