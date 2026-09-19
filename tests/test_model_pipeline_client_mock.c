#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mock_model_resident_client.h"
#include "sparkpipe/spark_model_pipeline_client.h"
#include "sparkpipe/spark_model_resident_deployment.h"

#define TEST_RANKS 3u
#define TEST_MAX_SEQ 4u
#define TEST_MAX_ROWS 8u

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

typedef struct TestCallbackState
{
	uint32_t result_count;
	uint32_t completion_count;
	uint32_t stage_completion_count;
	SparkStatus last_result_status;
	SparkStatus last_completion_status;
	uint64_t last_submission_id;
	uint32_t last_token_count;
} TestCallbackState;

static void TestSubmitResult(void *context, uint64_t submission_id, SparkStatus status)
{
	TestCallbackState *s = (TestCallbackState *)context;
	s->result_count++;
	s->last_result_status = status;
	s->last_submission_id = submission_id;
}

static void TestCompletion(void *context, const SparkModelServingCompletion *completion)
{
	TestCallbackState *s = (TestCallbackState *)context;
	s->completion_count++;
	s->last_completion_status = (SparkStatus)completion->status;
	s->last_token_count = completion->token_count;
}

static void TestStageCompletion(void *context, const SparkModelPipelineStageCompletion *stage_completion)
{
	TestCallbackState *s = (TestCallbackState *)context;
	(void)stage_completion;
	s->stage_completion_count++;
}

static void TestBuildDeployment(SparkModelResidentDeployment *deployment, SparkModelResidentDeploymentNode *nodes)
{
	uint32_t i;
	memset(deployment,0,sizeof(*deployment));
	memset(nodes,0,sizeof(*nodes) * TEST_RANKS);
	deployment->abi_version = 1u;
	deployment->descriptor_bytes = sizeof(*deployment);
	deployment->schema_version = 2u;
	deployment->node_count = TEST_RANKS;
	deployment->coordinator_rank_index = 0u;
	deployment->runtime_limits.max_inflight_submission_count = 8u;
	deployment->runtime_limits.max_active_sequence_count = TEST_MAX_SEQ;
	deployment->runtime_limits.max_input_row_count = TEST_MAX_ROWS;
	deployment->runtime_limits.resident_sequence_capacity = TEST_MAX_SEQ;
	deployment->runtime_limits.kv_logical_page_capacity = 256u;
	deployment->runtime_limits.kv_physical_page_capacity = 256u;
	deployment->nodes = nodes;
	for (i=0u; i<TEST_RANKS; i++)
	{
		nodes[i].rank_index = i;
		nodes[i].stage_index = i;
	}
}

static void TestBuildSubmission(SparkModelServingSubmission *submission, SparkModelServingLane *lanes, uint64_t submission_id)
{
	memset(submission,0,sizeof(*submission));
	submission->abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submission->descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submission->work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submission->tokens_per_sequence = 1u;
	submission->submission_id = submission_id;
	submission->request_id = 100u + submission_id;
	submission->sequence_id = 200u;
	submission->control_generation = 1u;
	submission->transaction_id = submission_id + 1000u;
	submission->dispatch_generation = 1u;
	submission->request_generation = 1u;
	submission->step_generation = 1u;
	submission->active_sequence_count = 1u;
	submission->new_token_count = 1u;
	submission->lane_count = 1u;
	submission->row_count = 1u;
	submission->token_count = 1u;
	lanes[0].request_id = submission->request_id;
	lanes[0].sequence_id = submission->sequence_id;
	lanes[0].resident_sequence_slot = 0u;
	submission->lanes = lanes;
}

static void TestFireAllRanksResult(uint64_t submission_id, SparkStatus status)
{
	uint32_t rank;
	for (rank=0u; rank<TEST_RANKS; rank++)
		MockResidentClientFireResult(rank, submission_id, status);
}

static void TestFireAllRanksCompletion(uint64_t submission_id, SparkStatus status, const TestCallbackState *cb)
{
	SparkModelServingCompletion completion;
	uint32_t rank;
	uint64_t request_id = 100u + submission_id;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = (uint32_t)status;
	completion.submission_id = submission_id;
	completion.request_id = request_id;
	completion.sequence_id = 200u;
	completion.control_generation = 1u;
	completion.transaction_id = submission_id + 1000u;
	completion.dispatch_generation = 1u;
	completion.request_generation = 1u;
	completion.step_generation = 1u;
	completion.token_count = 1u;
	(void)cb;
	for (rank=0u; rank<TEST_RANKS; rank++)
		MockResidentClientFireCompletion(rank, &completion);
}

int main(void)
{
	SparkModelResidentDeployment deployment;
	SparkModelResidentDeploymentNode nodes[TEST_RANKS];
	SparkModelPipelineClientConfiguration configuration;
	SparkModelPipelineClient *pipeline;
	SparkModelServingSubmission submission;
	SparkModelServingLane lanes[2];
	TestCallbackState cb;
	SparkModelPipelineClientView view;
	SparkStatus status;
	uint64_t fingerprint_a, fingerprint_b;

	MockResidentClientReset();
	TestBuildDeployment(&deployment, nodes);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_MODEL_PIPELINE_CLIENT_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_MODEL_PIPELINE_CLIENT_CONFIGURATION_BYTES;
	configuration.connect_timeout_ms = 1000u;
	configuration.deployment = &deployment;
	memset(&cb,0,sizeof(cb));
	configuration.submit_result_function = TestSubmitResult;
	configuration.submit_result_context = &cb;
	configuration.completion_function = TestCompletion;
	configuration.completion_context = &cb;
	configuration.stage_completion_function = TestStageCompletion;
	configuration.stage_completion_context = &cb;

	pipeline = 0;
	status = SparkModelPipelineClientConnect(&configuration, &pipeline);
	CHECK(status == SPARK_STATUS_OK, "connect");
	if ( status != SPARK_STATUS_OK )
		return(1);

	TestBuildSubmission(&submission, lanes, 1u);
	status = SparkModelPipelineClientSubmit(pipeline, &submission);
	CHECK(status == SPARK_STATUS_OK, "submit 1");
	CHECK( MockResidentClientCalls(0, MOCK_CALL_PREPARE) == 1u, "rank 0 prepared");
	CHECK( MockResidentClientCalls(2, MOCK_CALL_PREPARE) == 1u, "rank 2 prepared");

	TestFireAllRanksResult(1u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);

	fingerprint_a = SparkModelPipelineClientSessionFingerprint(pipeline);

	TestFireAllRanksCompletion(1u, SPARK_STATUS_OK, &cb);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	CHECK( cb.completion_count == 1u, "completion fired once");
	CHECK( cb.last_token_count == 1u, "one token");

	TestBuildSubmission(&submission, lanes, 2u);
	status = SparkModelPipelineClientSubmit(pipeline, &submission);
	CHECK(status == SPARK_STATUS_OK, "submit 2");

	MockResidentClientFireResult(0u, 2u, SPARK_STATUS_OK);
	MockResidentClientFireResult(0u, 2u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
		CHECK( view.failed_status == SPARK_STATUS_OK, "duplicate result is not fatal");

	TestFireAllRanksResult(2u, SPARK_STATUS_OK);
	TestFireAllRanksCompletion(2u, SPARK_STATUS_OK, &cb);
	MockResidentClientFireCompletion(1u, &(const SparkModelServingCompletion){
		.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
		.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES,
		.submission_id = 999u,
	});
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
		CHECK( view.failed_status == SPARK_STATUS_OK, "unknown-submission completion is not fatal");

	SparkModelPipelineClientClearTransactions(pipeline);
	MockResidentClientFireResult(0u, 3u, SPARK_STATUS_OK);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
		CHECK( view.failed_status == SPARK_STATUS_OK, "late result after invalidation is not fatal");

	TestBuildSubmission(&submission, lanes, 3u);
	status = SparkModelPipelineClientSubmit(pipeline, &submission);
	TestFireAllRanksResult(3u, SPARK_STATUS_OK);
	MockResidentClientFireResult(1u, 3u, SPARK_STATUS_BUSY);
	(void)SparkModelPipelineClientProgress(pipeline, 8u);
	if ( SparkModelPipelineClientGetView(pipeline, &view) == SPARK_STATUS_OK )
		CHECK( view.failed_status != SPARK_STATUS_OK || cb.last_result_status == SPARK_STATUS_BUSY,
			"a rank's real error surfaces (not swallowed)");

	fingerprint_b = SparkModelPipelineClientSessionFingerprint(pipeline);
	CHECK( fingerprint_a == fingerprint_b, "fingerprint stable without disconnects");
	MockResidentClientDisconnect(1u);
	{
		uint64_t fp = SparkModelPipelineClientSessionFingerprint(pipeline);
		CHECK( fp != fingerprint_b, "fingerprint changes on a rank disconnect");
	}

	TestBuildSubmission(&submission, lanes, 4u);
	submission.submission_id = 2u;
	status = SparkModelPipelineClientSubmit(pipeline, &submission);
	CHECK( status == SPARK_STATUS_INVALID_ARGUMENT, "non-increasing submission id rejected");

	SparkModelPipelineClientDestroy(pipeline);
	MockResidentClientReset();

	fprintf(stderr,"%s: %u checks, %u failures\n",
		"test_model_pipeline_client_mock", test_checks, test_failures);
	return( test_failures != 0u ? 1 : 0 );
}
