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
#define TEST_MAX_REQUESTS 8u

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
	uint32_t token_events[TEST_MAX_REQUESTS + 1u];
	uint32_t completed_events[TEST_MAX_REQUESTS + 1u];
	uint32_t error_events[TEST_MAX_REQUESTS + 1u];
	uint32_t total_terminals;
} TestBatchState;

static void TestBatchEvent(void *context, const SparkModelBatchEvent *event)
{
	TestBatchState *s = (TestBatchState *)context;
	uint64_t id = event->request_id;
	if ( id > TEST_MAX_REQUESTS )
		return;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_TOKEN )
		s->token_events[id]++;
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_COMPLETED )
	{
		s->completed_events[id]++;
		s->total_terminals++;
	}
	if ( event->kind == SPARK_MODEL_BATCH_EVENT_REQUEST_CANCELLED ||
	     event->kind == SPARK_MODEL_BATCH_EVENT_ERROR )
	{
		s->error_events[id]++;
		s->total_terminals++;
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

static void TestSubmitPrompt(SparkModelBatchEngine *engine, uint64_t request_id, uint64_t sequence_id, uint32_t budget,const uint32_t *prompt,uint32_t prompt_count)
{
	SparkModelBatchSubmitRequest request;
	SparkModelBatchRequestHandle handle;
	SparkStatus status;
	memset(&request,0,sizeof(request));
	request.abi_version = SPARK_MODEL_BATCH_ENGINE_ABI_VERSION;
	request.descriptor_bytes = sizeof(request);
	request.request_id = request_id;
	request.sequence_id = sequence_id;
	request.prompt_token_ids = prompt;
	request.prompt_token_count = prompt_count;
	request.output_token_budget = budget;
	handle = 0;
	status = SparkModelBatchEngineSubmit(engine,&request,&handle);
	CHECK(status == SPARK_STATUS_OK, "submit request");
}

static void TestSubmit(SparkModelBatchEngine *engine, uint64_t request_id, uint64_t sequence_id, uint32_t budget)
{
	static const uint32_t prompt[4] = {11u,12u,13u,14u};
	TestSubmitPrompt(engine,request_id,sequence_id,budget,prompt,4u);
}

/* One drive step: engine progress (which pumps the pipeline and the mock
 * residents' reconnect logic), then the mock answers everything in flight.
 * The 1ms pause lets the engine's busy-retry backoff elapse on failure
 * scenarios without making the happy path slow. */
static void TestDrive(SparkModelBatchEngine *engine, uint32_t steps)
{
	uint32_t step;
	for (step=0u; step<steps; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
}

static void TestDriveUntilTerminal(SparkModelBatchEngine *engine, TestBatchState *state, uint32_t terminals, uint32_t max_steps)
{
	uint32_t step;
	for (step=0u; step<max_steps && state->total_terminals < terminals; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
}

static void TestScenarioHappyPath(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,2u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK( state.token_events[1] != 0u,
		"happy: the request produced tokens through the full stack");
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"happy: the request completed cleanly");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioRankDiesMidDecode(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	uint32_t step,tokens_before;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,4u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	for (step=0u; step<400u && state.token_events[1] == 0u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
	CHECK( state.token_events[1] != 0u, "chaos: prefill produced a token before the kill");
	tokens_before = state.token_events[1];
	MockResidentClientDisconnect(1u);
	TestDriveUntilTerminal(engine,&state,1u,2000u);
	CHECK( state.completed_events[1] == 0u && state.error_events[1] == 1u,
		"disconnect: started stream fails exactly once");
	CHECK( state.token_events[1] == tokens_before,
		"disconnect: emitted tokens are never replayed");
	{
		SparkModelBatchEngineView view;
		CHECK( SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK &&
			view.inflight_submission_count == 0u &&
			view.pipeline.active_transaction_count == 0u,
			"disconnect: old submissions and transactions retire before reuse");
	}
	TestSubmit(engine,2u,501u,2u);
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK( state.completed_events[2] == 1u && state.error_events[2] == 0u,
		"chaos: a fresh request completes after the recovery");
	SparkModelBatchEngineDestroy(engine);
}

static uint32_t TestWaitLane(SparkModelBatchEngine *engine,uint64_t request_id,uint64_t position,SparkModelServingLane *lane)
{
	for (uint32_t step = 0u; step < 2000u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine,8u);
		if ( MockResidentClientLastLane(0u,lane) != 0u &&
		     lane->request_id == request_id && lane->sequence_position == position )
			return(1u);
		(void)MockResidentClientDriveAll();
		usleep(1000);
	}
	return(0u);
}

static void TestScenarioCachedPrefixSessionReset(const SparkModelResidentDeployment *deployment,const char *runtime_root)
{
	static const uint32_t prompt[8] = {11u,12u,13u,14u,15u,16u,17u,18u};
	TestBatchState state = {0};
	SparkModelBatchEngine *engine;
	SparkModelServingLane canonical = {0},cached = {0},rebuilt = {0};
	MockResidentClientReset();
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestSubmit(engine,1u,500u,1u);
	CHECK(TestWaitLane(engine,1u,0u,&canonical) != 0u &&
		canonical.cache_publish_token_count == 4u,
		"prefix reset: capture canonical first block identity");
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK(state.completed_events[1] == 1u,"prefix reset: warm request completes");
	TestSubmitPrompt(engine,2u,501u,1u,prompt,8u);
	CHECK(TestWaitLane(engine,2u,4u,&cached) != 0u &&
		cached.cache_prefix_token_count == 4u &&
		(cached.flags & SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX) != 0u,
		"prefix reset: next request actually uses cached prefix");
	CHECK(state.token_events[2] == 0u,"prefix reset: session dies before emitted token");
	MockResidentClientDisconnect(1u);
	CHECK(TestWaitLane(engine,2u,0u,&rebuilt) != 0u &&
		rebuilt.cache_prefix_token_count == 0u && rebuilt.cache_publish_token_count == 4u,
		"prefix reset: recovered session recomputes first block");
	CHECK(memcmp(&canonical.cache_publish_identity,&rebuilt.cache_publish_identity,
		sizeof(canonical.cache_publish_identity)) == 0,
		"prefix reset: rebuilt digest equals canonical prompt digest");
	TestDriveUntilTerminal(engine,&state,2u,400u);
	CHECK(state.completed_events[2] == 1u && state.error_events[2] == 0u &&
		state.token_events[2] == 1u,"prefix reset: recovered request completes exactly once");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioRankKilledAndRevived(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,3u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	/* The rank is dead from the start (residentd down, agent respawning). */
	MockResidentClientKill(1u);
	TestDrive(engine,50u);
	CHECK( state.total_terminals == 0u,
		"chaos: a dead rank holds the request without failing it");
	MockResidentClientRevive(1u);
	TestDriveUntilTerminal(engine,&state,1u,2000u);
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"chaos: request completed once the killed rank revived");
	SparkModelBatchEngineDestroy(engine);
}

/* A rank answering BUSY is transient backpressure, not a fault: the
 * pipeline must NOT fail-stop (that reconnects every rank and resets every
 * engine session, killing all in-flight chains fleet-wide). The request
 * retries and completes; no rank ever reconnects. */
static void TestScenarioRankBusyBackpressure(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	uint32_t step;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_BUSY);
	TestSubmit(engine,1u,500u,2u);
	for (step=0u; step<50u; step++)
	{
		(void)SparkModelBatchEngineProgress(engine, 8u);
		(void)MockResidentClientDriveAll();
		usleep(2000);
	}
	CHECK( state.total_terminals == 0u,
		"busy: a BUSY rank holds the request without failing it");
	MockResidentClientScriptSubmitStatus(1u,SPARK_STATUS_OK);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	{
		SparkModelBatchEngineView view;
		if ( SparkModelBatchEngineGetView(engine,&view) == SPARK_STATUS_OK )
			fprintf(stderr,"DBG busy-end live=%u tokens=%u terminals=%u active_txn=%u submitted=%llu completed=%llu admitted=%llu rejected=%llu\n",
				(unsigned)view.live_request_count,(unsigned)state.token_events[1],
				(unsigned)state.total_terminals,
				(unsigned)view.pipeline.active_transaction_count,
				(unsigned long long)view.pipeline.submitted_count,
				(unsigned long long)view.pipeline.completed_count,
				(unsigned long long)view.pipeline.admitted_count,
				(unsigned long long)view.pipeline.rejected_count);
	}
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"busy: the request completes once the rank drains");
	{
		uint32_t rank;
		uint32_t reconnected = 0u;
		for (rank=0u; rank<TEST_RANKS; rank++)
			if ( MockResidentClientGeneration(rank) != 1u )
				reconnected = 1u;
		CHECK( reconnected == 0u,
			"busy: no rank reconnected — no pipeline fail-stop on backpressure");
	}
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioEosEarlyStop(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,8u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetTokenStart(154820u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestDriveUntilTerminal(engine,&state,1u,400u);
	CHECK( state.token_events[1] == 1u,
		"eos: generation stopped at the EOS token instead of the budget");
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"eos: the request completed cleanly on EOS");
	SparkModelBatchEngineDestroy(engine);
}

static void TestScenarioTwoRequestsRankDies(const SparkModelResidentDeployment *deployment, const char *runtime_root)
{
	TestBatchState state;
	SparkModelBatchEngine *engine;
	MockResidentClientReset();
	memset(&state,0,sizeof(state));
	engine = TestConnect(deployment,&state,runtime_root);
	if ( engine == 0 )
		return;
	TestSubmit(engine,1u,500u,3u);
	TestSubmit(engine,2u,501u,3u);
	MockResidentClientSetAutoTokens(1u);
	MockResidentClientSetFinalRank(TEST_RANKS - 1u,1u);
	TestDrive(engine,20u);
	MockResidentClientKill(0u);
	TestDrive(engine,30u);
	MockResidentClientRevive(0u);
	TestDriveUntilTerminal(engine,&state,2u,2000u);
	CHECK( state.completed_events[1] == 1u && state.error_events[1] == 0u,
		"chaos: first concurrent request completed across the rank death");
	CHECK( state.completed_events[2] == 1u && state.error_events[2] == 0u,
		"chaos: second concurrent request completed across the rank death");
	SparkModelBatchEngineDestroy(engine);
}

int main(void)
{
	SparkModelResidentDeployment deployment;
	char path[512];
	char runtime_root[256];

	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	(void)snprintf(path,sizeof(path),"%s/mock-batch-deployment.json",runtime_root);
	TestWriteDeployment(path,runtime_root);
	assert(SparkModelResidentDeploymentLoad(path,&deployment) == SPARK_STATUS_OK);
	deployment.eos_token_count = 1u;
	deployment.eos_token_ids[0] = 154820u;

	if ( getenv("ONLY_BUSY") != 0 )
	{
		TestScenarioRankBusyBackpressure(&deployment,runtime_root);
	}
	else
	{
		TestScenarioHappyPath(&deployment,runtime_root);
		TestScenarioRankDiesMidDecode(&deployment,runtime_root);
		TestScenarioRankKilledAndRevived(&deployment,runtime_root);
		TestScenarioCachedPrefixSessionReset(&deployment,runtime_root);
		TestScenarioRankBusyBackpressure(&deployment,runtime_root);
		TestScenarioEosEarlyStop(&deployment,runtime_root);
		TestScenarioTwoRequestsRankDies(&deployment,runtime_root);
	}

	SparkModelResidentDeploymentReset(&deployment);
	MockResidentClientReset();
	(void)unlink(path);

	fprintf(stderr,"%s: %u checks, %u failures\n",
		"test_model_batch_engine_mock", test_checks, test_failures);
	return( test_failures != 0u ? 1 : 0 );
}
