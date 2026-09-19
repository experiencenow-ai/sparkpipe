#include "mock_model_resident_client.h"

#include <stdlib.h>
#include <string.h>

struct SparkModelResidentClient
{
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t connected;
	uint32_t failed;
	uint64_t client_generation;
	SparkModelResidentSubmitResultFunction submit_result_function;
	void *submit_result_context;
	SparkModelResidentDecisionResultFunction decision_result_function;
	void *decision_result_context;
	SparkModelServingCompletionFunction completion_function;
	void *completion_context;
	uint32_t submit_calls;
	uint32_t prepare_calls;
	uint32_t continue_calls;
	uint32_t commit_calls;
	uint32_t abort_calls;
	uint64_t last_submission_id;
	uint32_t is_final_rank;
	SparkModelServingSubmission last_submission;
	SparkStatus scripted_submit_status;
};

static SparkModelResidentClient *mock_registry[MOCK_RESIDENT_MAX_RANKS];
static uint32_t mock_registry_count;

SparkModelResidentClient *MockResidentClientByRank(uint32_t stage_index)
{
	uint32_t i;
	for (i=0u; i<mock_registry_count; i++)
		if ( mock_registry[i] != 0 &&
		     mock_registry[i]->stage_index == stage_index )
			return(mock_registry[i]);
	return(0);
}

uint32_t MockResidentClientCalls(uint32_t stage_index, uint32_t kind)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c == 0 )
		return(0u);
	switch (kind)
	{
		case MOCK_CALL_SUBMIT: return(c->submit_calls);
		case MOCK_CALL_PREPARE: return(c->prepare_calls);
		case MOCK_CALL_CONTINUE: return(c->continue_calls);
		case MOCK_CALL_COMMIT: return(c->commit_calls);
		case MOCK_CALL_ABORT: return(c->abort_calls);
	}
	return(0u);
}

void MockResidentClientScriptSubmitStatus(uint32_t stage_index, SparkStatus status)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->scripted_submit_status = status;
}

void MockResidentClientFireResult(uint32_t stage_index, uint64_t submission_id, SparkStatus status)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 && c->submit_result_function != 0 )
		c->submit_result_function(c->submit_result_context,submission_id,status);
}

void MockResidentClientFireDecision(uint32_t stage_index, uint64_t submission_id, uint32_t decision_kind, SparkStatus status)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 && c->decision_result_function != 0 )
		c->decision_result_function(c->decision_result_context,submission_id,decision_kind,status);
}

void MockResidentClientFireCompletion(uint32_t stage_index, const SparkModelServingCompletion *completion)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 && c->completion_function != 0 )
		c->completion_function(c->completion_context,completion);
}

void MockResidentClientDisconnect(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->connected = 0u;
}

void MockResidentClientReset(void)
{
	uint32_t i;
	for (i=0u; i<mock_registry_count; i++)
		free(mock_registry[i]);
	mock_registry_count = 0u;
	memset(mock_registry,0,sizeof(mock_registry));
}

SparkStatus SparkModelResidentClientConnect(
	const SparkModelResidentClientConfiguration *configuration,
	SparkModelResidentClient **client_out)
{
	SparkModelResidentClient *c;
	if ( configuration == 0 || client_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( mock_registry_count >= MOCK_RESIDENT_MAX_RANKS )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	c = (SparkModelResidentClient *)calloc(1u,sizeof(*c));
	if ( c == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	c->rank_index = configuration->rank_index;
	c->stage_index = configuration->stage_index;
	c->connected = 1u;
	c->client_generation = 1u;
	c->submit_result_function = configuration->submit_result_function;
	c->submit_result_context = configuration->submit_result_context;
	c->decision_result_function = configuration->decision_result_function;
	c->decision_result_context = configuration->decision_result_context;
	c->completion_function = configuration->completion_function;
	c->completion_context = configuration->completion_context;
	c->scripted_submit_status = SPARK_STATUS_OK;
	mock_registry[mock_registry_count++] = c;
	*client_out = c;
	return(SPARK_STATUS_OK);
}

void SparkModelResidentClientDestroy(SparkModelResidentClient *client)
{
	(void)client;
}

void SparkModelResidentClientFailStop(SparkModelResidentClient *client)
{
	if ( client != 0 )
		client->failed = 1u;
}

SparkStatus SparkModelResidentClientSubmit(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->submit_calls++;
	client->last_submission_id = submission->submission_id;
	return(client->scripted_submit_status);
}

SparkStatus SparkModelResidentClientPrepare(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->prepare_calls++;
	client->last_submission_id = submission->submission_id;
	client->last_submission = *submission;
	return(client->scripted_submit_status);
}

SparkStatus SparkModelResidentClientCanQueueContinuation(
	const SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	(void)client;
	(void)submission;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientContinue(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->continue_calls++;
	client->last_submission_id = submission->submission_id;
	client->last_submission = *submission;
	return(client->scripted_submit_status);
}

SparkStatus SparkModelResidentClientCanQueueDecision(
	const SparkModelResidentClient *client,
	uint64_t submission_id,
	uint32_t decision_kind)
{
	(void)client;
	(void)submission_id;
	(void)decision_kind;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientCommit(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->commit_calls++;
	client->last_submission_id = submission_id;
	return(client->scripted_submit_status);
}

SparkStatus SparkModelResidentClientAbort(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->abort_calls++;
	client->last_submission_id = submission_id;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientProgress(
	SparkModelResidentClient *client,
	uint32_t maximum_message_count)
{
	(void)client;
	(void)maximum_message_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientGetPollDescriptor(
	const SparkModelResidentClient *client,
	SparkModelResidentClientPollDescriptor *descriptor)
{
	(void)client;
	(void)descriptor;
	return(SPARK_STATUS_UNSUPPORTED);
}

SparkStatus SparkModelResidentClientGetView(
	const SparkModelResidentClient *client,
	SparkModelResidentClientView *view)
{
	if ( client == 0 || view == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(view,0,sizeof(*view));
	view->connected = client->connected;
	view->client_generation = client->client_generation;
	view->queue_capacity = 64u;
	view->max_active_sequence_count = 4u;
	view->max_input_row_count = 8u;
	view->resident_sequence_capacity = 4u;
	return(SPARK_STATUS_OK);
}

static uint32_t mock_auto_tokens;

void MockResidentClientSetAutoTokens(uint32_t count)
{
	mock_auto_tokens = count;
}

void MockResidentClientSetFinalRank(uint32_t stage_index, uint32_t is_final)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->is_final_rank = is_final;
}

uint32_t MockResidentClientDriveAll(void)
{
	uint32_t i,drove = 0u;
	for (i=0u; i<mock_registry_count; i++)
	{
		SparkModelResidentClient *c = mock_registry[i];
		if ( c == 0 || c->last_submission_id == 0u || c->connected == 0u )
			continue;
		if ( c->submit_result_function != 0 )
			c->submit_result_function(c->submit_result_context,c->last_submission_id,SPARK_STATUS_OK);
		if ( c->decision_result_function != 0 )
			c->decision_result_function(c->decision_result_context,c->last_submission_id,1u,SPARK_STATUS_OK);
		if ( c->completion_function != 0 )
		{
			SparkModelServingCompletion completion;
			memset(&completion,0,sizeof(completion));
			completion.abi_version = 1u;
			completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
			completion.status = SPARK_STATUS_OK;
			completion.submission_id = c->last_submission_id;
			completion.request_id = c->last_submission.request_id;
			completion.sequence_id = c->last_submission.sequence_id;
			completion.sequence_position = c->last_submission.sequence_position;
			completion.control_generation = c->last_submission.control_generation;
			completion.transaction_id = c->last_submission.transaction_id;
			completion.dispatch_generation = c->last_submission.dispatch_generation;
			completion.request_generation = c->last_submission.request_generation;
			completion.step_generation = c->last_submission.step_generation;
			completion.residency = c->last_submission.residency;
			if ( c->is_final_rank != 0u && mock_auto_tokens != 0u )
			{
				uint32_t t;
				completion.token_count = c->last_submission.active_sequence_count * mock_auto_tokens;
				completion.tokens_per_sequence = mock_auto_tokens;
				completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
				completion.accepted_token_count = completion.token_count;
				for (t=0u; t<completion.token_count && t<(uint32_t)(sizeof(completion.token_ids)/sizeof(completion.token_ids[0])); t++)
					completion.token_ids[t] = 11u + t;
			}
			c->completion_function(c->completion_context,&completion);
		}
		drove++;
	}
	return(drove);
}
