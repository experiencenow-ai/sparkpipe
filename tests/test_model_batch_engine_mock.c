#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "mock_model_resident_client.h"
#include "sparkpipe/spark_model_batch_engine.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif

#define TEST_RANKS 3u

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

typedef struct TestBatchState
{
	uint32_t token_events;
	uint32_t terminal_events;
	uint32_t terminal_status;
	uint32_t tokens[64];
} TestBatchState;

static void TestBatchEvent(void *context, const SparkModelBatchEvent *event)
{
	TestBatchState *s = (TestBatchState *)context;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN )
	{
		if ( s->token_events < 64u )
			s->tokens[s->token_events] = event->token_id;
		s->token_events++;
	}
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED ||
	     event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED ||
	     event->kind == SPARK_MODEL_BATCH_EVENT_ERROR )
	{
		s->terminal_events++;
		s->terminal_status = event->status;
	}
}

static const char *const TestTransportHosts[TEST_RANKS] =
{
	"mock-stage-a","mock-stage-b","mock-stage-c"
};

static void TestWriteDeployment(const char *path, const char *runtime_root)
{
	TestModelResidentDeploymentFixture fixture;
	const char *runtime_roots[TEST_RANKS];
	uint32_t stage_indices[TEST_RANKS];
	SparkModelResidentEndpoint endpoints[TEST_RANKS];
	uint32_t rank;
	for (rank=0u; rank<TEST_RANKS; rank++)
	{
		runtime_roots[rank] = runtime_root;
		stage_indices[rank] = rank;
		memset(&endpoints[rank],0,sizeof(endpoints[rank]));
		endpoints[rank].abi_version = SPARK_MODEL_RESIDENT_ENDPOINT_ABI_VERSION;
		endpoints[rank].descriptor_bytes = SPARK_MODEL_RESIDENT_ENDPOINT_BYTES;
		endpoints[rank].kind = SPARK_MODEL_RESIDENT_ENDPOINT_KIND_TCP;
		endpoints[rank].tcp_host = TestTransportHosts[rank];
		endpoints[rank].tcp_port = (uint32_t)(59100u + rank);
	}
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = TEST_MODEL_RESIDENT_TRANSPORT_PATH;
	fixture.transport_mode = "host-rdma";
	fixture.node_target = "test.model.serving.target";
	fixture.adapter_configuration_path = "tests/fixtures/model_serving_adapter_config.json";
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = TestTransportHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 4u;
	fixture.runtime_limits.max_active_sequence_count = 8u;
	fixture.runtime_limits.max_input_row_count = 32u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 256u;
	fixture.runtime_limits.kv_physical_page_capacity = 256u;
	fixture.control_port_base = 59200u;
	fixture.node_count = TEST_RANKS;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
}

static SparkModelBatchEngine *TestConnect(const SparkModelResidentDeployment *deployment, TestBatchState *state, const char *runtime_root)
{
	SparkModelBatchEngineConfiguration configuration;
	SparkModelBatchEngine *engine;
	SparkStatus status;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	configuration.descriptor_bytes = sizeof(configuration);
	configuration.connect_timeout_ms = 1000u;
	configuration.request_capacity = 8u;
	configuration.max_context_tokens = 256u;
	configuration.max_prefill_rows_per_submission = 4u;
	configuration.maximum_messages_per_rank_per_progress = 8u;
	configuration.deployment = deployment;
	configuration.runtime_root = runtime_root;
	configuration.event_function = TestBatchEvent;
	configuration.event_context = state;
	engine = 0;
	status = SparkModelBatchEngineConnect(&configuration,&engine);
	CHECK(status == SPARK_STATUS_OK, "batch engine connect");
	return(engine);
}

int main(void)
{
	SparkModelResidentDeployment deployment;
	SparkModelBatchEngine *engine;
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle;
	TestBatchState state;
	SparkStatus status;
	char path[512];
	char runtime_root[256];
	uint32_t prompt[4] = { 11u, 12u, 13u, 14u };
	uint32_t step;

	MockResidentClientReset();
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	(void)snprintf(path,sizeof(path),"%s/mock-batch-deployment.json",runtime_root);
	TestWriteDeployment(path,runtime_root);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == SPARK_STATUS_OK);

	deployment.eos_token_count = 1u;
	deployment.eos_token_ids[0] = 154820u;
	memset(&state,0,sizeof(state));
	engine = TestConnect(&deployment,&state,runtime_root);
	if ( engine == 0 )
		return(1);

	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.request_id = 1u;
	request.sequence_id = 500u;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = 4u;
	request.output_token_budget = 2u;
	handle = 0;
	status = SparkModelBatchEngineSubmit(engine,&request,&handle);
	CHECK(status == SPARK_STATUS_OK, "submit request");

	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u, 1u);
	for (step=0u; step<400u && state.terminal_events == 0u; step++)
	{
		SparkModelBatchEngineView view;
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		if ( step == 20u || step == 100u )
			if ( SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK )
				fprintf(stderr,"DBG step=%u active=%u queued_prefill=%u ready_decode=%u\n",
					step,(unsigned)view.live_request_count,
					(unsigned)view.queued_prefill_count,(unsigned)view.ready_decode_count);
	}

	CHECK( state.token_events != 0u,
		"the request produced tokens through the full stack");
	CHECK( state.terminal_events != 0u,
		"the request reached a terminal event");

	SparkModelBatchEngineDestroy(engine);
	SparkModelResidentDeploymentReset(&deployment);
	MockResidentClientReset();
	(void)unlink(path);

	fprintf(stderr,"%s: %u checks, %u failures (tokens=%u terminals=%u)\n",
		"test_model_batch_engine_mock", test_checks, test_failures,
		state.token_events, state.terminal_events);
	return( test_failures != 0u ? 1 : 0 );
}
