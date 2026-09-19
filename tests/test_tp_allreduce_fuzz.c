#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"

#define FUZZ_MAX_RANKS SPARK_TP_DEVICE_COLLECTIVE_MAX_DEGREE
#define FUZZ_HIDDEN 64u
#define FUZZ_ELEMENTS FUZZ_HIDDEN
#define FUZZ_WATCHDOG_NS (8ull * 1000000000ull)
#define FUZZ_ROUND_TIMEOUT_MS 400u

static uint32_t test_failures;
static uint32_t test_checks;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

static uint8_t *g_regions[FUZZ_MAX_RANKS];
static uint32_t g_rank_count;
static uint32_t g_connect_rank_hint;
static volatile uint64_t g_broadcast_count;
static volatile uint32_t g_shipper_stop;

SparkStatus SparkWeightdClientConnect(const char *socket_path, SparkWeightdClient **client, SparkWeightdHelloResult *hello_out)
{
	(void)socket_path; (void)hello_out;
	*client = (SparkWeightdClient *)calloc(1u, 16u);
	if ( *client == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*(uint32_t *)*client = g_connect_rank_hint;
	return(SPARK_STATUS_OK);
}

void SparkWeightdClientClose(SparkWeightdClient *client)
{
	free(client);
}

uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client)
{
	(void)client;
	return(1u);
}

SparkStatus SparkWeightdClientMeshBroadcast(SparkWeightdClient *client, uint32_t peer_mask, uint64_t source_offset, uint64_t remote_offset, uint32_t length, uint64_t seq_value, uint64_t seq_remote_offset, uint64_t timeout_nanoseconds)
{
	uint32_t source_rank = *(uint32_t *)client;
	uint32_t peer;
	(void)seq_value; (void)seq_remote_offset; (void)timeout_nanoseconds;
	__sync_add_and_fetch(&g_broadcast_count, 1u);
	if ( source_rank >= g_rank_count || g_regions[source_rank] == 0 )
		return(SPARK_STATUS_IO_ERROR);
	for ( peer = 0u; peer < SPARK_WEIGHTD_MESH_RANKS_PER_BAND; peer++ )
	{
		if ( (peer_mask & (1u << peer)) == 0u || peer >= g_rank_count ||
		     g_regions[peer] == 0 )
			continue;
		memcpy(g_regions[peer] + remote_offset,
		    g_regions[source_rank] + source_offset, length);
	}
	__sync_synchronize();
	return(SPARK_STATUS_OK);
}

int SparkGlm5NextLaunchMeshCopyDown(void *stream, volatile void *destination, const void *source, uint64_t bytes)
{
	(void)stream;
	memcpy((void *)destination, source, (size_t)bytes);
	__sync_synchronize();
	return(0);
}

int SparkGlm5NextLaunchMeshGuard(void *stream, volatile void *error_word, void *output)
{
	(void)stream; (void)error_word; (void)output;
	return(0);
}

static uint16_t FuzzBf16FromFloat(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, 4u);
	return((uint16_t)(bits >> 16));
}

static float FuzzBf16ToFloat(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16;
	float out;
	memcpy(&out, &bits, 4u);
	return(out);
}

static SparkStatus FuzzCombineBf16(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	uint16_t *destination = (uint16_t *)destination_device;
	const uint16_t *source = (const uint16_t *)source_device;
	uint32_t count = active_sequence_count * hidden_dimension;
	uint32_t index;
	(void)combine_context; (void)cuda_stream;
	for ( index = 0u; index < count; index++ )
		destination[index] = FuzzBf16FromFloat(
		    FuzzBf16ToFloat(destination[index]) +
		    FuzzBf16ToFloat(source[index]));
	return(SPARK_STATUS_OK);
}

typedef struct FuzzRank
{
	SparkTpDeviceCollective collective;
	uint32_t rank;
	uint16_t partial[FUZZ_ELEMENTS];
	uint16_t output[FUZZ_ELEMENTS];
	volatile uint64_t completion_count;
} FuzzRank;

static FuzzRank g_ranks[FUZZ_MAX_RANKS];

static void FuzzComplete(void *context, const SparkTpDeviceCollectiveCompletion *completion)
{
	FuzzRank *rank = (FuzzRank *)context;
	(void)completion;
	__sync_add_and_fetch(&rank->completion_count, 1u);
}

static uint64_t FuzzNowNs(void)
{
	struct timespec now;
	if ( clock_gettime(CLOCK_MONOTONIC,&now) != 0 )
		return(0ull);
	return((uint64_t)now.tv_sec * UINT64_C(1000000000) + (uint64_t)now.tv_nsec);
}

static void *FuzzShipperMain(void *argument)
{
	uint64_t seen[FUZZ_MAX_RANKS] = {0u};
	(void)argument;
	while ( __sync_add_and_fetch(&g_shipper_stop, 0u) == 0u )
	{
		uint32_t rank;
		for ( rank = 0u; rank < g_rank_count; rank++ )
		{
			volatile uint64_t *entry = (volatile uint64_t *)
			    (g_regions[rank] +
			    SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(0u, rank));
			uint64_t tag;
			uint64_t slot_index;
			uint64_t bytes;
			uint64_t slot_base;
			uint64_t key_lo;
			uint64_t first_missed;
			uint32_t resync_mask = 0u;
			uint32_t peer;
			uint32_t ring;
			uint32_t stable = 0u;
			uint32_t tries;
			for ( tries = 0u; tries < 64u && stable == 0u; tries++ )
			{
				tag = entry[0];
				bytes = entry[1];
				slot_index = entry[2];
				__sync_synchronize();
				if ( tag == entry[0] && bytes == entry[1] &&
				     slot_index == entry[2] )
					stable = 1u;
			}
			if ( stable == 0u || tag == 0ull || tag == seen[rank] )
				continue;
			if ( slot_index >= SPARK_WEIGHTD_MESH_SLOTS_PER_BAND ||
			     bytes == 0ull ||
			     bytes + 16ull > SPARK_WEIGHTD_MESH_SLOT_BYTES )
			{
				seen[rank] = tag;
				continue;
			}
			slot_base = slot_index * SPARK_WEIGHTD_MESH_SLOT_BYTES;
			key_lo = (tag & ~((1ull << 16u) - 1ull)) + 1ull;
			first_missed = seen[rank] + 1ull;
			if ( first_missed < key_lo )
				first_missed = key_lo;
			if ( tag > first_missed )
			{
				uint64_t missed;
				if ( tag - first_missed >
				     SPARK_WEIGHTD_MESH_SLOTS_PER_RANK )
					first_missed = tag -
					    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
				for ( missed = first_missed; missed < tag;
				      missed++ )
					resync_mask |= (uint32_t)1u <<
					    (uint32_t)(missed &
					    (uint64_t)(SPARK_WEIGHTD_MESH_SLOTS_PER_RANK - 1u));
			}
			for ( ring = 0u; ring < SPARK_WEIGHTD_MESH_SLOTS_PER_RANK;
			      ring++ )
			{
				uint64_t ring_base;
				if ( (resync_mask & (1u << ring)) == 0u )
					continue;
				ring_base = ((uint64_t)rank *
				    SPARK_WEIGHTD_MESH_SLOTS_PER_RANK + ring) *
				    SPARK_WEIGHTD_MESH_SLOT_BYTES;
				for ( peer = 0u; peer < g_rank_count; peer++ )
				{
					if ( peer == rank )
						continue;
					memcpy(g_regions[peer] + ring_base,
					    g_regions[rank] + ring_base,
					    SPARK_WEIGHTD_MESH_SLOT_BYTES);
				}
			}
			for ( peer = 0u; peer < g_rank_count; peer++ )
			{
				if ( peer == rank )
					continue;
				memcpy(g_regions[peer] + slot_base,
				    g_regions[rank] + slot_base, (size_t)bytes);
			}
			__sync_synchronize();
			for ( peer = 0u; peer < g_rank_count; peer++ )
			{
				if ( peer == rank )
					continue;
				memcpy(g_regions[peer] + slot_base +
				    SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u,
				    g_regions[rank] + slot_base +
				    SPARK_WEIGHTD_MESH_SLOT_BYTES - 8u, 8u);
			}
			__sync_synchronize();
			seen[rank] = tag;
		}
	}
	return(0);
}

static SparkStatus FuzzCreateRank(uint32_t rank)
{
	SparkTpDeviceCollectiveConfig config;
	SparkStatus status;
	uint32_t index;
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	config.backend_kind = SPARK_TP_DEVICE_COLLECTIVE_BACKEND_HIDDEN_TRANSPORT;
	config.tp_degree = g_rank_count;
	config.tp_rank = rank;
	config.local_hidden_dimension = FUZZ_HIDDEN;
	config.max_active_sequence_count = 4u;
	config.connect_timeout_milli = 1000u;
	config.operation_timeout_milli = FUZZ_ROUND_TIMEOUT_MS;
	config.collective_identifier = 0u;
	config.combine_bf16_function = FuzzCombineBf16;
	g_connect_rank_hint = rank;
	status = SparkTpDeviceCollectiveCreate(&config, &g_ranks[rank].collective);
	if ( status != SPARK_STATUS_OK )
		return(status);
	g_ranks[rank].rank = rank;
	status = SparkTpDeviceCollectivePrepareReceiveBf16(
	    &g_ranks[rank].collective, g_regions[rank], 1u, FUZZ_HIDDEN, 0u, 0);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for ( index = 0u; index < FUZZ_ELEMENTS; index++ )
		g_ranks[rank].partial[index] = FuzzBf16FromFloat((float)(rank + 1u));
	return(SPARK_STATUS_OK);
}

static void FuzzResetRank(uint32_t rank)
{
	SparkStatus status;
	SparkTpDeviceCollectiveDestroy(&g_ranks[rank].collective);
	memset(&g_ranks[rank].collective,0,sizeof(SparkTpDeviceCollective));
	status = FuzzCreateRank(rank);
	CHECK( status == SPARK_STATUS_OK, "reset recreates the rank" );
}

typedef struct FuzzTask
{
	FuzzRank *rank;
	uint64_t argument;
	SparkStatus status;
	volatile uint32_t done;
	pthread_t thread;
} FuzzTask;

static FuzzTask g_tasks[FUZZ_MAX_RANKS];

static void *FuzzChainMain(void *data)
{
	FuzzTask *task = (FuzzTask *)data;
	task->status = SparkTpDeviceCollectiveChainKey(&task->rank->collective,
	    task->argument);
	__sync_synchronize();
	task->done = 1u;
	return(0);
}

static void *FuzzRoundMain(void *data)
{
	FuzzTask *task = (FuzzTask *)data;
	SparkTpDeviceCollectiveSubmission submission;
	memset(&submission,0,sizeof(submission));
	submission.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
	submission.descriptor_bytes = sizeof(submission);
	submission.slot_index = 0u;
	submission.active_sequence_count = 1u;
	submission.ordinal = task->argument;
	submission.local_device = task->rank->partial;
	submission.full_device = task->rank->output;
	submission.cuda_stream = (void *)0x1;
	submission.completion_function = FuzzComplete;
	submission.completion_context = task->rank;
	task->status = SparkTpDeviceCollectiveEnqueue(&task->rank->collective,
	    &submission, SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
	__sync_synchronize();
	task->done = 1u;
	return(0);
}

static void FuzzCancelAll(void)
{
	uint32_t rank;
	for ( rank = 0u; rank < g_rank_count; rank++ )
		if ( g_ranks[rank].collective.implementation != 0 )
			SparkTpDeviceCollectiveBroadcastCancel(&g_ranks[rank].collective);
}

static uint32_t FuzzWedge(const char *phase, uint64_t round, uint32_t rank)
{
	test_failures++;
	fprintf(stderr,"WEDGE phase=%s round=%llu rank=%u\n", phase,
	    (unsigned long long)round, rank);
	if ( g_ranks[rank].collective.implementation != 0 )
		SparkTpDeviceCollectiveGraphStuckDump(&g_ranks[rank].collective);
	FuzzCancelAll();
	return(0u);
}

static uint32_t FuzzWaitDone(uint32_t rank, const char *phase, uint64_t round)
{
	uint64_t deadline = FuzzNowNs() + FUZZ_WATCHDOG_NS;
	while ( g_tasks[rank].done == 0u )
	{
		if ( FuzzNowNs() >= deadline )
			return(FuzzWedge(phase, round, rank));
		usleep(1000);
	}
	return(1u);
}

static uint32_t FuzzRunSet(void *(*worker)(void *), uint64_t argument,
    const uint32_t *run, uint32_t run_count, const char *phase,
    uint64_t round, int32_t reset_rank)
{
	uint32_t i;
	uint32_t ok = 1u;
	for ( i = 0u; i < run_count; i++ )
	{
		FuzzTask *task = &g_tasks[run[i]];
		memset(task->rank->output, 0xFF, sizeof(task->rank->output));
		task->argument = argument;
		task->status = SPARK_STATUS_INTERNAL_ERROR;
		task->done = 0u;
		__sync_synchronize();
		if ( pthread_create(&task->thread, 0, worker, task) != 0 )
		{
			task->status = SPARK_STATUS_INTERNAL_ERROR;
			task->done = 1u;
		}
	}
	if ( reset_rank >= 0 )
	{
		uint32_t in_set = 0u;
		for ( i = 0u; i < run_count; i++ )
			if ( run[i] == (uint32_t)reset_rank )
				in_set = 1u;
		if ( in_set == 0u )
			FuzzResetRank((uint32_t)reset_rank);
		else if ( FuzzWaitDone((uint32_t)reset_rank, phase, round) == 0u )
			ok = 0u;
		else
		{
			pthread_join(g_tasks[reset_rank].thread, 0);
			g_tasks[reset_rank].done = 2u;
			FuzzResetRank((uint32_t)reset_rank);
		}
	}
	for ( i = 0u; i < run_count; i++ )
	{
		uint32_t rank = run[i];
		if ( g_tasks[rank].done == 2u )
			continue;
		if ( g_tasks[rank].done == 0u &&
		     FuzzWaitDone(rank, phase, round) == 0u )
			ok = 0u;
		pthread_join(g_tasks[rank].thread, 0);
	}
	return(ok);
}

static uint32_t FuzzSumOk(uint32_t rank)
{
	uint16_t expected = FuzzBf16FromFloat(
	    (float)(g_rank_count * (g_rank_count + 1u) / 2u));
	uint32_t index;
	for ( index = 0u; index < FUZZ_ELEMENTS; index++ )
	{
		if ( g_ranks[rank].output[index] != expected )
		{
			fprintf(stderr,"WRONG-SUM rank=%u elem=%u got=%u want=%u\n",
			    rank, index, (unsigned)g_ranks[rank].output[index],
			    (unsigned)expected);
			return(0u);
		}
	}
	return(1u);
}

static uint32_t FuzzAllRanks(uint32_t *run)
{
	uint32_t rank;
	for ( rank = 0u; rank < g_rank_count; rank++ )
		run[rank] = rank;
	return(g_rank_count);
}

static void FuzzBasic(void)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint32_t i, rank;
	uint64_t request = 1000u;
	for ( i = 0u; i < 8u; i++ )
	{
		uint64_t ordinal = 16ull * (uint64_t)i + 1ull;
		request++;
		CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
		    "chain", i, -1) != 0u, "chain key round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "chain key status ok" );
		CHECK( FuzzRunSet(FuzzRoundMain, ordinal, run, run_count,
		    "round", i, -1) != 0u, "allreduce round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "round status ok" );
			CHECK( FuzzSumOk(rank) != 0u, "round sum correct" );
		}
	}
	for ( rank = 0u; rank < run_count; rank++ )
	{
		uint64_t count = 0u;
		SparkTpDeviceCollectiveRoundStats(&g_ranks[rank].collective,
		    &count, 0, 0u);
		CHECK( count == 8u, "round stats advance (no rounds=0 wedge)" );
	}
	request++;
	CHECK( FuzzRunSet(FuzzChainMain, request, run, run_count,
	    "chain", 8u, -1) != 0u, "multi-round chain key completes" );
	for ( i = 0u; i < 4u; i++ )
	{
		CHECK( FuzzRunSet(FuzzRoundMain, 16ull * 9ull + 1ull + (uint64_t)i, run,
		    run_count, "round", 8u + i, -1) != 0u,
		    "chained round completes" );
		for ( rank = 0u; rank < run_count; rank++ )
		{
			CHECK( g_tasks[rank].status == SPARK_STATUS_OK,
			    "chained round status ok" );
			CHECK( FuzzSumOk(rank) != 0u, "chained round sum correct" );
		}
	}
	{
		uint64_t deadline = FuzzNowNs() + 2ull * 1000000000ull;
		uint32_t drained;
		do {
			drained = 1u;
			for ( rank = 0u; rank < run_count; rank++ )
				if ( g_ranks[rank].completion_count != 12u )
					drained = 0u;
			if ( drained == 0u )
				usleep(1000);
		} while ( drained == 0u && FuzzNowNs() < deadline );
		for ( rank = 0u; rank < run_count; rank++ )
			CHECK( g_ranks[rank].completion_count == 12u,
			    "every round fires its completion" );
	}
}

static uint32_t FuzzRun(uint32_t rounds, uint32_t seed, uint32_t kill_percent)
{
	uint32_t run[FUZZ_MAX_RANKS];
	uint32_t run_count = FuzzAllRanks(run);
	uint32_t unrecovered = 0u;
	uint32_t recovery_pending = 0u;
	uint64_t request = 500000u;
	uint32_t round;
	srand(seed);
	for ( round = 0u; round < rounds; round++ )
	{
		uint32_t kill = 0u;
		uint32_t victim = 0u;
		uint32_t kill_before = 0u;
		uint32_t failures_before = test_failures;
		uint32_t i;
		if ( recovery_pending == 0u &&
		     (uint32_t)rand() % 100u < kill_percent )
		{
			kill = 1u;
			victim = (uint32_t)rand() % g_rank_count;
			kill_before = (uint32_t)rand() & 1u;
		}
		request++;
		if ( FuzzRunSet(FuzzChainMain, request, run, run_count,
		        "chain", round, -1) == 0u )
		{
			fprintf(stderr,"%u rounds, %u unrecovered\n", rounds,
			    unrecovered + 1u);
			return(1u);
		}
		for ( i = 0u; i < run_count; i++ )
			CHECK( g_tasks[i].status == SPARK_STATUS_OK,
			    "fuzz chain key status ok" );
		if ( kill != 0u && kill_before != 0u )
		{
			uint32_t others[FUZZ_MAX_RANKS];
			uint32_t other_count = 0u;
			for ( i = 0u; i < run_count; i++ )
				if ( i != victim )
					others[other_count++] = i;
			if ( FuzzRunSet(FuzzRoundMain, 16ull * (uint64_t)round + 1ull,
			        others, other_count, "round", round,
			        (int32_t)victim) == 0u )
			{
				fprintf(stderr,"%u rounds, %u unrecovered\n",
				    rounds, unrecovered + 1u);
				return(1u);
			}
			for ( i = 0u; i < other_count; i++ )
				CHECK( g_tasks[others[i]].status != SPARK_STATUS_OK,
				    "killed-before-publish rank fails peers fast" );
			recovery_pending = 1u;
		}
		else if ( kill != 0u )
		{
			if ( FuzzRunSet(FuzzRoundMain, 16ull * (uint64_t)round + 1ull,
			        run, run_count, "round", round,
			        (int32_t)victim) == 0u )
			{
				fprintf(stderr,"%u rounds, %u unrecovered\n",
				    rounds, unrecovered + 1u);
				return(1u);
			}
			for ( i = 0u; i < run_count; i++ )
			{
				CHECK( g_tasks[i].status == SPARK_STATUS_OK,
				    "killed-after-publish round still completes" );
				CHECK( FuzzSumOk(i) != 0u,
				    "killed-after-publish sum stays correct" );
			}
			recovery_pending = 1u;
		}
		else
		{
			if ( FuzzRunSet(FuzzRoundMain, 16ull * (uint64_t)round + 1ull,
			        run, run_count, "round", round, -1) == 0u )
			{
				fprintf(stderr,"%u rounds, %u unrecovered\n",
				    rounds, unrecovered + 1u);
				return(1u);
			}
			for ( i = 0u; i < run_count; i++ )
			{
				CHECK( g_tasks[i].status == SPARK_STATUS_OK,
				    "fuzz round status ok" );
				CHECK( FuzzSumOk(i) != 0u, "fuzz round sum correct" );
			}
			if ( recovery_pending != 0u )
			{
				if ( test_failures != failures_before )
					unrecovered++;
				recovery_pending = 0u;
			}
		}
	}
	fprintf(stderr,"%u rounds, %u unrecovered\n", rounds, unrecovered);
	return( unrecovered != 0u ? 1u : 0u );
}

int main(int argc, char **argv)
{
	uint32_t ranks = 4u;
	uint32_t fuzz_rounds = 0u;
	uint32_t seed = 12345u;
	uint32_t kill_percent = 40u;
	uint32_t rank;
	uint32_t wedge;
	pthread_t shipper;
	int arg;
	for ( arg = 1; arg < argc; arg++ )
	{
		if ( strcmp(argv[arg], "--ranks") == 0 && arg + 1 < argc )
			ranks = (uint32_t)strtoul(argv[++arg], 0, 10);
		else if ( strcmp(argv[arg], "--fuzz") == 0 && arg + 1 < argc )
			fuzz_rounds = (uint32_t)strtoul(argv[++arg], 0, 10);
		else if ( strcmp(argv[arg], "--seed") == 0 && arg + 1 < argc )
			seed = (uint32_t)strtoul(argv[++arg], 0, 10);
		else if ( strcmp(argv[arg], "--kill-percent") == 0 && arg + 1 < argc )
			kill_percent = (uint32_t)strtoul(argv[++arg], 0, 10);
		else
		{
			fprintf(stderr,"usage: %s [--ranks N] [--fuzz ROUNDS] [--seed S] [--kill-percent K]\n", argv[0]);
			return(2);
		}
	}
	if ( ranks < 2u || ranks > FUZZ_MAX_RANKS )
	{
		fprintf(stderr,"ranks must be in [2,%u]\n", FUZZ_MAX_RANKS);
		return(2);
	}
	(void)setenv("SPARK_WEIGHTD_SOCKET","/tmp/tp_allreduce_fuzz.sock",1);
	g_rank_count = ranks;
	for ( rank = 0u; rank < ranks; rank++ )
	{
		g_regions[rank] = (uint8_t *)calloc(1u,
		    (size_t)SPARK_WEIGHTD_MESH_REGION_BYTES);
		if ( g_regions[rank] == 0 )
		{
			fprintf(stderr,"mesh region alloc failed rank=%u\n", rank);
			return(1);
		}
	}
	if ( pthread_create(&shipper, 0, FuzzShipperMain, 0) != 0 )
	{
		fprintf(stderr,"shipper start failed\n");
		return(1);
	}
	for ( rank = 0u; rank < ranks; rank++ )
		CHECK( FuzzCreateRank(rank) == SPARK_STATUS_OK, "rank create" );
	for ( rank = 0u; rank < ranks; rank++ )
		g_tasks[rank].rank = &g_ranks[rank];
	wedge = 0u;
	if ( fuzz_rounds == 0u )
		FuzzBasic();
	else
		wedge = FuzzRun(fuzz_rounds, seed, kill_percent);
	g_shipper_stop = 1u;
	pthread_join(shipper, 0);
	for ( rank = 0u; rank < ranks; rank++ )
	{
		if ( g_ranks[rank].collective.implementation != 0 )
			SparkTpDeviceCollectiveDestroy(&g_ranks[rank].collective);
		free(g_regions[rank]);
	}
	fprintf(stderr,"test_tp_allreduce_fuzz: %u checks, %u failures (broadcasts=%llu)\n",
	    test_checks, test_failures, (unsigned long long)g_broadcast_count);
	return( test_failures != 0u || wedge != 0u ? 1 : 0 );
}
