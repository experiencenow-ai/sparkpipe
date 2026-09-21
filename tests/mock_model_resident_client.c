#include "mock_model_resident_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Link-time stand-in for the real resident client. Fidelity contract with
 * runtime/model_resident_client.c:
 * - in-flight submissions are a bounded per-rank queue (real: pending ring);
 *   a full queue reports BUSY.
 * - every submit path runs EnsureConnected first: the first activity after
 *   a server-side drop reports the dead socket once (IO_ERROR), the next
 *   attempt reconnects with a fresh client generation; killed ranks keep
 *   refusing until revived.
 * - FailStop / server-side death silently drops every in-flight submission
 *   and pending decision (the real client clears its pending ring without
 *   callbacks; the pipeline learns via the progress error path).
 * - decisions are only answered after the pipeline actually committed or
 *   aborted the submission; continued transactions auto-commit via the
 *   lease and never see a decision. */

#define MOCK_INFLIGHT_CAPACITY 8u

typedef struct MockInflight
{
	uint64_t submission_id;
	uint32_t result_driven;
	uint32_t completion_driven;
	SparkModelServingSubmission submission;
} MockInflight;

typedef struct MockPendingDecision
{
	uint64_t submission_id;
	uint32_t decision_kind;
} MockPendingDecision;

struct SparkModelResidentClient
{
	uint32_t rank_index;
	uint32_t stage_index;
	uint32_t connected;
	uint32_t failed;
	uint32_t detected;
	uint32_t stay_dead;
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
	MockInflight inflight[MOCK_INFLIGHT_CAPACITY];
	uint32_t inflight_count;
	MockPendingDecision pending_decisions[MOCK_INFLIGHT_CAPACITY];
	uint32_t pending_decision_count;
	uint32_t is_final_rank;
	SparkStatus scripted_submit_status;
};

static SparkModelResidentClient *mock_registry[MOCK_RESIDENT_MAX_RANKS];
static uint32_t mock_registry_count;
static uint32_t mock_auto_tokens;
static uint32_t mock_token_start = 11u;

static int MockTraceEnabled(void)
{
	return( getenv("MOCK_TRACE") != 0 );
}

SparkModelResidentClient *MockResidentClientByRank(uint32_t stage_index)
{
	uint32_t i;
	SparkModelResidentClient *found = 0;
	for (i=0u; i<mock_registry_count; i++)
		if ( mock_registry[i] != 0 &&
		     mock_registry[i]->stage_index == stage_index &&
		     (found == 0 || mock_registry[i]->client_generation >
		        found->client_generation) )
			found = mock_registry[i];
	return(found);
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

uint64_t MockResidentClientGeneration(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	return( c != 0 ? c->client_generation : 0u );
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

static void MockResidentClientDropInflight(SparkModelResidentClient *c)
{
	memset(c->inflight,0,sizeof(c->inflight));
	c->inflight_count = 0u;
	memset(c->pending_decisions,0,sizeof(c->pending_decisions));
	c->pending_decision_count = 0u;
}

void MockResidentClientDisconnect(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
	{
		c->connected = 0u;
		c->detected = 0u;
		MockResidentClientDropInflight(c);
	}
}

void MockResidentClientKill(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
	{
		c->connected = 0u;
		c->detected = 0u;
		c->stay_dead = 1u;
		MockResidentClientDropInflight(c);
	}
}

void MockResidentClientRevive(uint32_t stage_index)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->stay_dead = 0u;
}

void MockResidentClientReset(void)
{
	uint32_t i;
	for (i=0u; i<mock_registry_count; i++)
		if ( mock_registry[i] != 0 )
			free(mock_registry[i]);
	mock_registry_count = 0u;
	mock_auto_tokens = 0u;
	mock_token_start = 11u;
	memset(mock_registry,0,sizeof(mock_registry));
}

static SparkStatus MockResidentClientEnsureConnected(SparkModelResidentClient *c)
{
	if ( c->connected != 0u )
		return(SPARK_STATUS_OK);
	if ( c->detected == 0u )
	{
		c->detected = 1u;
		return(SPARK_STATUS_IO_ERROR);
	}
	if ( c->stay_dead != 0u )
		return(SPARK_STATUS_IO_ERROR);
	c->connected = 1u;
	c->detected = 0u;
	c->client_generation++;
	if ( MockTraceEnabled() )
		fprintf(stderr,"MOCK rank=%u reconnected generation=%llu\n",
			c->stage_index,(unsigned long long)c->client_generation);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientConnect(
	const SparkModelResidentClientConfiguration *configuration,
	SparkModelResidentClient **client_out)
{
	SparkModelResidentClient *c;
	uint32_t i;
	if ( configuration == 0 || client_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	c = 0;
	for (i=0u; i<mock_registry_count; i++)
		if ( mock_registry[i] != 0 &&
		     mock_registry[i]->stage_index == configuration->stage_index &&
		     mock_registry[i]->connected == 0u )
		{
			free(mock_registry[i]);
			mock_registry[i] = 0;
			c = (SparkModelResidentClient *)calloc(1u,sizeof(*c));
			mock_registry[i] = c;
			break;
		}
	if ( c == 0 )
	{
		if ( mock_registry_count >= MOCK_RESIDENT_MAX_RANKS )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		c = (SparkModelResidentClient *)calloc(1u,sizeof(*c));
		if ( c == 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		mock_registry[mock_registry_count++] = c;
	}
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
	{
		client->failed = 1u;
		client->connected = 0u;
		client->detected = 1u;
		MockResidentClientDropInflight(client);
	}
}

static SparkStatus MockResidentClientEnqueue(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission,
	const char *op)
{
	MockInflight *slot;
	SparkStatus connect_status;
	connect_status = MockResidentClientEnsureConnected(client);
	if ( connect_status != SPARK_STATUS_OK )
		return(connect_status);
	if ( client->scripted_submit_status != SPARK_STATUS_OK )
		return(client->scripted_submit_status);
	if ( client->inflight_count >= MOCK_INFLIGHT_CAPACITY )
		return(SPARK_STATUS_BUSY);
	slot = &client->inflight[client->inflight_count++];
	memset(slot,0,sizeof(*slot));
	slot->submission_id = submission->submission_id;
	slot->submission = *submission;
	client->last_submission_id = submission->submission_id;
	if ( MockTraceEnabled() )
		fprintf(stderr,"MOCK rank=%u %s id=%llu kind=%u rows=%u seq=%llu inflight=%u\n",
			client->stage_index,op,
			(unsigned long long)submission->submission_id,
			(unsigned)submission->work_kind,(unsigned)submission->row_count,
			(unsigned long long)submission->sequence_id,
			(unsigned)client->inflight_count);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientSubmit(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->submit_calls++;
	return(MockResidentClientEnqueue(client,submission,"submit"));
}

SparkStatus SparkModelResidentClientPrepare(
	SparkModelResidentClient *client,
	const SparkModelServingSubmission *submission)
{
	if ( client == 0 || submission == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->prepare_calls++;
	return(MockResidentClientEnqueue(client,submission,"prepare"));
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
	return(MockResidentClientEnqueue(client,submission,"continue"));
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

static SparkStatus MockResidentClientQueueDecision(
	SparkModelResidentClient *client,
	uint64_t submission_id,
	uint32_t decision_kind)
{
	MockPendingDecision *slot;
	uint32_t k;
	/* an abort settles the submission on the server: the rank's prepared
	 * work is freed and no completion will follow. A COMMIT is different —
	 * the work runs after the commit and the completion still arrives. */
	if ( decision_kind == SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT )
		for (k=0u; k<client->inflight_count; k++)
			if ( client->inflight[k].submission_id == submission_id )
			{
				client->inflight[k] = client->inflight[client->inflight_count - 1u];
				client->inflight_count--;
				break;
			}
	if ( client->pending_decision_count >= MOCK_INFLIGHT_CAPACITY )
		return(SPARK_STATUS_BUSY);
	slot = &client->pending_decisions[client->pending_decision_count++];
	slot->submission_id = submission_id;
	slot->decision_kind = decision_kind;
	client->last_submission_id = submission_id;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkModelResidentClientCommit(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->commit_calls++;
	return(MockResidentClientQueueDecision(client,submission_id,
		SPARK_MODEL_RESIDENT_IPC_DECISION_COMMIT));
}

SparkStatus SparkModelResidentClientAbort(
	SparkModelResidentClient *client,
	uint64_t submission_id)
{
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	client->abort_calls++;
	return(MockResidentClientQueueDecision(client,submission_id,
		SPARK_MODEL_RESIDENT_IPC_DECISION_ABORT));
}

SparkStatus SparkModelResidentClientProgress(
	SparkModelResidentClient *client,
	uint32_t maximum_message_count)
{
	(void)maximum_message_count;
	if ( client == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(MockResidentClientEnsureConnected(client));
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
	view->pending_submission_count = client->inflight_count;
	view->max_active_sequence_count = 4u;
	view->max_input_row_count = 8u;
	view->resident_sequence_capacity = 4u;
	return(SPARK_STATUS_OK);
}

void MockResidentClientSetAutoTokens(uint32_t count)
{
	mock_auto_tokens = count;
}

void MockResidentClientSetTokenStart(uint32_t first_token_id)
{
	mock_token_start = first_token_id;
}

void MockResidentClientSetFinalRank(uint32_t stage_index, uint32_t is_final)
{
	SparkModelResidentClient *c = MockResidentClientByRank(stage_index);
	if ( c != 0 )
		c->is_final_rank = is_final;
}

uint32_t MockResidentClientDriveResults(void)
{
	uint32_t i,k,drove = 0u;
	for (i=0u; i<mock_registry_count; i++)
	{
		SparkModelResidentClient *c = mock_registry[i];
		if ( c == 0 || c->connected == 0u )
			continue;
		for (k=0u; k<c->inflight_count; k++)
		{
			MockInflight *slot = &c->inflight[k];
			if ( slot->result_driven != 0u )
				continue;
			slot->result_driven = 1u;
			if ( c->submit_result_function != 0 )
				c->submit_result_function(c->submit_result_context,slot->submission_id,SPARK_STATUS_OK);
			drove++;
		}
	}
	return(drove);
}

uint32_t MockResidentClientDriveCompletions(void)
{
	uint32_t i,k,drove = 0u;
	for (i=0u; i<mock_registry_count; i++)
	{
		SparkModelResidentClient *c = mock_registry[i];
		if ( c == 0 || c->connected == 0u )
			continue;
		for (k=0u; k<c->inflight_count; k++)
		{
			MockInflight *slot = &c->inflight[k];
			if ( slot->result_driven == 0u || slot->completion_driven != 0u )
				continue;
			slot->completion_driven = 1u;
			if ( c->completion_function != 0 )
			{
				SparkModelServingCompletion completion;memset(&completion,0,sizeof(completion));
				completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
				completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
				completion.status = SPARK_STATUS_OK;
				completion.submission_id = slot->submission_id;
				completion.request_id = slot->submission.request_id;
				completion.sequence_id = slot->submission.sequence_id;
				completion.sequence_position = slot->submission.sequence_position;
				completion.control_generation = slot->submission.control_generation;
				completion.transaction_id = slot->submission.transaction_id;
				completion.dispatch_generation = slot->submission.dispatch_generation;
				completion.request_generation = slot->submission.request_generation;
				completion.step_generation = slot->submission.step_generation;
				completion.residency = slot->submission.residency;
				if ( c->is_final_rank != 0u && mock_auto_tokens != 0u &&
				     slot->submission.work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
				{
					uint32_t t;
					completion.token_count = slot->submission.active_sequence_count * mock_auto_tokens;
					completion.tokens_per_sequence = mock_auto_tokens;
					completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
					completion.accepted_token_count = completion.token_count;
					for (t=0u; t<completion.token_count && t<(uint32_t)(sizeof(completion.token_ids)/sizeof(completion.token_ids[0])); t++)
						completion.token_ids[t] = mock_token_start + t;
				}
				c->completion_function(c->completion_context,&completion);
			}
			/* the delivered completion retires the submission on the
			 * server — free the slot (swap-remove) so long runs with many
			 * submissions don't fill the queue. The callback may retire
			 * it synchronously (pipeline forwarding/abort); re-find by id
			 * so an already-retired or shifted slot is never reused. */
			{
				uint32_t idx;
				uint32_t found = 0u;
				for (idx=0u; idx<c->inflight_count; idx++)
					if ( c->inflight[idx].submission_id == slot->submission_id )
					{
						found = 1u;
						break;
					}
				if ( found != 0u )
				{
					c->inflight[idx] = c->inflight[c->inflight_count - 1u];
					c->inflight_count--;
					if ( idx <= k )
						k--;
				}
			}
			drove++;
		}
	}
	return(drove);
}

uint32_t MockResidentClientDriveDecisions(void)
{
	uint32_t i,k,drove = 0u;
	for (i=0u; i<mock_registry_count; i++)
	{
		SparkModelResidentClient *c = mock_registry[i];
		if ( c == 0 || c->connected == 0u )
			continue;
		for (k=0u; k<c->pending_decision_count; k++)
		{
			MockPendingDecision *d = &c->pending_decisions[k];
			if ( c->decision_result_function != 0 )
				c->decision_result_function(c->decision_result_context,d->submission_id,d->decision_kind,SPARK_STATUS_OK);
			drove++;
		}
		c->pending_decision_count = 0u;
	}
	return(drove);
}

uint32_t MockResidentClientDriveAll(void)
{
	uint32_t a,b,c;
	a = MockResidentClientDriveResults();
	b = MockResidentClientDriveDecisions();
	c = MockResidentClientDriveCompletions();
	return(a + b + c);
}
