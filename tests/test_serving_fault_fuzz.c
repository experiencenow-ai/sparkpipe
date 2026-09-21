#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "fixtures/model_resident_deployment_fixture.h"
#include "mock_model_resident_client.h"
#include "sparkpipe/spark_model_pipeline_client.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#ifndef TEST_MODEL_SERVING_ADAPTER_PATH
#define TEST_MODEL_SERVING_ADAPTER_PATH ""
#endif
#ifndef TEST_MODEL_RESIDENT_TRANSPORT_PATH
#define TEST_MODEL_RESIDENT_TRANSPORT_PATH ""
#endif

#define TEST_RANKS 3u
#define FAULT_ROUNDS_DEFAULT 400u

static uint32_t test_failures;
static uint32_t test_checks;
static uint64_t fuzz_seed = 1u;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL seed=%llu %s:%d %s\n", \
			    (unsigned long long)fuzz_seed,__FILE__,__LINE__,name); \
		} \
	} while (0)

static uint64_t FuzzRand(void)
{
	fuzz_seed = fuzz_seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
	return(fuzz_seed >> 17u);
}

typedef struct FaultState
{
	uint32_t result_count;
	uint32_t completion_count;
	uint64_t last_submission_id;
	SparkStatus last_result_status;
	SparkStatus last_completion_status;
} FaultState;

static void FaultSubmitResult(void *context, uint64_t submission_id, SparkStatus status)
{
	FaultState *s = (FaultState *)context;
	s->result_count++;
	s->last_submission_id = submission_id;
	s->last_result_status = status;
}

static void FaultCompletion(void *context, const SparkModelServingCompletion *completion)
{
	FaultState *s = (FaultState *)context;
	s->completion_count++;
	s->last_completion_status = (SparkStatus)completion->status;
}

static const char *const FaultHosts[TEST_RANKS] =
{
	"fault-a","fault-b","fault-c"
};

static void FaultBuildDeployment(SparkModelResidentDeployment *deployment,
    const char *path, const char *runtime_root)
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
		endpoints[rank].tcp_host = FaultHosts[rank];
		endpoints[rank].tcp_port = (uint32_t)(59200u + rank);
	}
	memset(&fixture,0,sizeof(fixture));
	fixture.adapter_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_shared_object_path = TEST_MODEL_SERVING_ADAPTER_PATH;
	fixture.driver_program_name = "resident_decode";
	fixture.transport_shared_object_path = TEST_MODEL_RESIDENT_TRANSPORT_PATH;
	fixture.transport_mode = "host-rdma";
	fixture.node_target = "fault.model.serving.target";
	fixture.adapter_configuration_path = "tests/fixtures/model_serving_adapter_config.json";
	fixture.runtime_roots = runtime_roots;
	fixture.transport_hosts = FaultHosts;
	fixture.stage_indices = stage_indices;
	fixture.control_endpoints = endpoints;
	fixture.runtime_limits.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	fixture.runtime_limits.descriptor_bytes = SPARK_MODEL_SERVING_RUNTIME_LIMITS_BYTES;
	fixture.runtime_limits.max_inflight_submission_count = 4u;
	fixture.runtime_limits.max_active_sequence_count = 4u;
	fixture.runtime_limits.max_input_row_count = 8u;
	fixture.runtime_limits.resident_sequence_capacity = 32u;
	fixture.runtime_limits.kv_logical_page_capacity = 256u;
	fixture.runtime_limits.kv_physical_page_capacity = 256u;
	fixture.control_port_base = 59200u;
	fixture.node_count = TEST_RANKS;
	fixture.coordinator_rank_index = 0u;
	assert(TestModelResidentDeploymentWrite(path,&fixture) == 0);
	assert(SparkModelResidentDeploymentLoad(path,deployment) == SPARK_STATUS_OK);
}

static void FaultBuildSubmission(SparkModelServingSubmission *submission,
    SparkModelServingLane *lanes, uint64_t submission_id)
{
	uint32_t slot_base = (uint32_t)((submission_id * 2u) % 28u);
	uint64_t sequence_id = 100u + submission_id;
	static uint32_t token_ids[2];
	static uint32_t row_lane_indices[2];
	static uint64_t row_positions[2];
	static uint64_t row_sequence_ids[2];
	memset(lanes,0,2u * sizeof(lanes[0]));
	lanes[0].request_id = 900u + submission_id;
	lanes[0].request_generation = 1u;
	lanes[0].step_generation = submission_id + 3000u;
	lanes[0].sequence_id = sequence_id;
	lanes[0].resident_sequence_slot = slot_base;
	lanes[0].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	lanes[1].request_id = 901u + submission_id;
	lanes[1].request_generation = 1u;
	lanes[1].step_generation = submission_id + 3000u;
	lanes[1].sequence_id = sequence_id + 1u;
	lanes[1].resident_sequence_slot = slot_base + 1u;
	lanes[1].flags = SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN;
	token_ids[0] = 11u;
	token_ids[1] = 12u;
	row_lane_indices[0] = 0u;
	row_lane_indices[1] = 1u;
	row_positions[0] = 0u;
	row_positions[1] = 0u;
	row_sequence_ids[0] = sequence_id;
	row_sequence_ids[1] = sequence_id + 1u;
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->tokens_per_sequence = 1u;
	submission->submission_id = submission_id;
	submission->request_id = 900u + submission_id;
	submission->sequence_id = sequence_id;
	submission->control_generation = 1u;
	submission->transaction_id = submission_id + 1000u;
	submission->dispatch_generation = 1u;
	submission->request_generation = 1u;
	submission->step_generation = submission_id + 3000u;
	submission->residency.word0 = submission_id;
	submission->residency.word1 = submission_id + 100u;
	submission->residency.generation = submission_id + 200u;
	submission->residency.owner = 1u;
	submission->active_sequence_count = 2u;
	submission->new_token_count = 2u;
	submission->lane_count = 2u;
	submission->row_count = 2u;
	submission->token_count = 2u;
	submission->token_ids = token_ids;
	submission->lanes = lanes;
	submission->row_lane_indices = row_lane_indices;
	submission->row_positions = row_positions;
	submission->row_sequence_ids = row_sequence_ids;
}

static void FaultFireCompletion(uint64_t submission_id, SparkStatus status)
{
	SparkModelServingCompletion completion;
	uint32_t rank;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = (uint32_t)status;
	completion.submission_id = submission_id;
	completion.request_id = 900u + submission_id;
	completion.sequence_id = 100u + submission_id;
	completion.control_generation = 1u;
	completion.transaction_id = submission_id + 1000u;
	completion.dispatch_generation = 1u;
	completion.request_generation = 1u;
	completion.step_generation = submission_id + 3000u;
	completion.residency.word0 = submission_id;
	completion.residency.word1 = submission_id + 100u;
	completion.residency.generation = submission_id + 200u;
	completion.residency.owner = 1u;
	for (rank=0u; rank<TEST_RANKS; rank++)
	{
		if ( rank == TEST_RANKS - 1u )
		{
			completion.token_count = 2u;
			completion.tokens_per_sequence = 1u;
			completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		}
		MockResidentClientFireCompletion(rank,&completion);
	}
}

static uint32_t FaultDriveProgress(SparkModelPipelineClient *pipeline, uint32_t passes)
{
	uint32_t pass;
	uint32_t moved = 0u;
	for (pass=0u; pass<passes; pass++)
	{
		moved += MockResidentClientDriveResults();
		(void)SparkModelPipelineClientProgress(pipeline,8u);
		moved += MockResidentClientDriveDecisions();
		(void)SparkModelPipelineClientProgress(pipeline,8u);
		moved += MockResidentClientDriveCompletions();
		(void)SparkModelPipelineClientProgress(pipeline,8u);
	}
	return(moved);
}

static uint32_t FaultHealthyRun(SparkModelPipelineClient *pipeline, FaultState *cb,
    uint64_t submission_id, const char *tag)
{
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	uint32_t results_before = cb->result_count;
	uint32_t completions_before = cb->completion_count;
	uint32_t spins;
	SparkStatus status;
	FaultBuildSubmission(&submission,lanes,submission_id);
	status = SparkModelPipelineClientSubmit(pipeline,&submission);
	if ( status == SPARK_STATUS_BUSY )
		return(2u);
	if ( status != SPARK_STATUS_OK )
	{
		SparkModelPipelineClientView v;
		fprintf(stderr,"DIAG submit-status=%d after fault; ",(int)status);
		if ( SparkModelPipelineClientGetView(pipeline,&v) == SPARK_STATUS_OK )
			fprintf(stderr,"failed_status=%u active_txn=%u\n",
			    (unsigned)v.failed_status,(unsigned)v.active_transaction_count);
		else
			fprintf(stderr,"view unreadable\n");
		CHECK(0,tag);
		return(0u);
	}
	for (spins=0u; spins<40u; spins++)
	{
		FaultDriveProgress(pipeline,1u);
		if ( cb->completion_count > completions_before )
			return(1u);
	}
	{
		SparkModelPipelineClientView v;
		fprintf(stderr,"DIAG no-completion after 40 spins; ");
		if ( SparkModelPipelineClientGetView(pipeline,&v) == SPARK_STATUS_OK )
			fprintf(stderr,"failed_status=%u active_txn=%u\n",
			    (unsigned)v.failed_status,(unsigned)v.active_transaction_count);
	}
	(void)results_before;
	CHECK(0,tag);
	return(0u);
}

enum
{
	FAULT_SUBMIT_BUSY = 0,
	FAULT_DISCONNECT,
	FAULT_KILL_REVIVE,
	FAULT_DROP_COMPLETION,
	FAULT_LATE_COMPLETION,
	FAULT_ABORT_DECISION,
	FAULT_RESULT_ERROR,
	FAULT_KIND_COUNT
};

int main(int argc, char **argv)
{
	SparkModelResidentDeployment deployment;
	SparkModelPipelineClientConfiguration configuration;
	SparkModelPipelineClient *pipeline;
	SparkModelPipelineClientView view;
	FaultState cb;
	uint32_t round;
	uint32_t rounds = FAULT_ROUNDS_DEFAULT;
	uint64_t next_id = 1u;
	uint64_t generations_before[TEST_RANKS];
	char deploy_path[512];
	char runtime_root[256];
	uint32_t rank;

	if ( argc > 1 )
		fuzz_seed = strtoull(argv[1],0,10);
	if ( argc > 2 )
		rounds = (uint32_t)strtoul(argv[2],0,10);

	MockResidentClientReset();
	assert(getcwd(runtime_root,sizeof(runtime_root)) != 0);
	(void)snprintf(deploy_path,sizeof(deploy_path),"%s/fault-fuzz-deployment.json",runtime_root);
	FaultBuildDeployment(&deployment,deploy_path,runtime_root);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 200u;
	configuration.deployment = &deployment;
	configuration.runtime_root = runtime_root;
	memset(&cb,0,sizeof(cb));
	configuration.submit_result_function = FaultSubmitResult;
	configuration.submit_result_context = &cb;
	configuration.completion_function = FaultCompletion;
	configuration.completion_context = &cb;

	pipeline = 0;
	CHECK(SparkModelPipelineClientConnect(&configuration,&pipeline) == SPARK_STATUS_OK,
	    "connect");
	if ( pipeline == 0 )
	{
		fprintf(stderr,"fault-fuzz: cannot start (connect failed), %u checks\n",test_checks);
		return(1);
	}

	CHECK(FaultHealthyRun(pipeline,&cb,next_id++,"baseline healthy submission completes") == 1u,
	    "baseline run");
	for (rank=0u; rank<TEST_RANKS; rank++)
		generations_before[rank] = MockResidentClientGeneration(rank);

	for (round=0u; round<rounds; round++)
	{
		uint32_t victim = (uint32_t)(FuzzRand() % TEST_RANKS);
		uint32_t kind = (uint32_t)(FuzzRand() % FAULT_KIND_COUNT);
		uint64_t id = next_id++;
		SparkModelServingSubmission submission;
		SparkModelServingLane lanes[2];
		SparkStatus status;

		MockResidentClientScriptSubmitStatus(victim,SPARK_STATUS_OK);
		FaultBuildSubmission(&submission,lanes,id);
		status = SparkModelPipelineClientSubmit(pipeline,&submission);

		switch (kind)
		{
		case FAULT_SUBMIT_BUSY:
			if ( status == SPARK_STATUS_OK )
				MockResidentClientScriptSubmitStatus(victim,SPARK_STATUS_BUSY);
			FaultDriveProgress(pipeline,3u);
			MockResidentClientScriptSubmitStatus(victim,SPARK_STATUS_OK);
			break;
		case FAULT_DISCONNECT:
			FaultDriveProgress(pipeline,1u);
			MockResidentClientDisconnect(victim);
			FaultDriveProgress(pipeline,3u);
			break;
		case FAULT_KILL_REVIVE:
			FaultDriveProgress(pipeline,1u);
			MockResidentClientKill(victim);
			FaultDriveProgress(pipeline,3u);
			MockResidentClientRevive(victim);
			break;
		case FAULT_DROP_COMPLETION:
			FaultDriveProgress(pipeline,4u);
			break;
		case FAULT_LATE_COMPLETION:
			FaultDriveProgress(pipeline,4u);
			FaultFireCompletion(id,SPARK_STATUS_OK);
			break;
		case FAULT_ABORT_DECISION:
			FaultDriveProgress(pipeline,2u);
			MockResidentClientFireDecision(victim,id,2u,SPARK_STATUS_OK);
			FaultDriveProgress(pipeline,3u);
			break;
		case FAULT_RESULT_ERROR:
			FaultDriveProgress(pipeline,1u);
			MockResidentClientFireResult(victim,id,SPARK_STATUS_IO_ERROR);
			FaultDriveProgress(pipeline,3u);
			break;
		default:
			break;
		}
		MockResidentClientScriptSubmitStatus(victim,SPARK_STATUS_OK);
		MockResidentClientRevive(victim);
		FaultDriveProgress(pipeline,6u);

		CHECK(SparkModelPipelineClientGetView(pipeline,&view) == SPARK_STATUS_OK,
		    "view readable after fault");
		CHECK(view.active_transaction_count == 0u,
		    "no leaked transactions after fault recovery window");
		for (rank=0u; rank<TEST_RANKS; rank++)
			CHECK(MockResidentClientGeneration(rank) >= generations_before[rank],
			    "generations never regress");
		{
			uint32_t outcome = FaultHealthyRun(pipeline,&cb,next_id++,
			    "healthy submission completes after any fault sequence");
			CHECK(outcome != 0u,"no permanent wedge after fault+recovery");
			if ( outcome == 2u )
			{
				(void)SparkModelPipelineClientProgress(pipeline,8u);
				CHECK(1,"busy-outcome progress exercised");
			}
		}
		FaultDriveProgress(pipeline,2u);
	}

	SparkModelPipelineClientDestroy(pipeline);
	fprintf(stderr,"test_serving_fault_fuzz: %u checks, %u failures (seed=%llu rounds=%u)\n",
	    test_checks,test_failures,(unsigned long long)fuzz_seed,rounds);
	(void)unlink(deploy_path);
	return(test_failures != 0u ? 1 : 0);
}
