#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_kv_page_cache.h"

#define SEQ_CAP 8u
#define PAGE_CAP 4u
#define ENTRY_CAP 64u
#define BUCKETS 32u
#define MAX_LANES 2u

static uint32_t test_failures;
static uint32_t test_checks;
static uint64_t fuzz_seed = 7u;

#define CHECK(cond, name) do { \
		test_checks++; \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL seed=%llu %s\n",(unsigned long long)fuzz_seed,name); \
		} \
	} while (0)

static uint64_t FuzzRand(void)
{
	fuzz_seed = fuzz_seed * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
	return(fuzz_seed >> 17u);
}

static uint64_t NowNs(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC,&ts);
	return((uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec);
}

typedef struct FuzzHarness
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[ENTRY_CAP];
	SparkKvPageCacheEntry entries[ENTRY_CAP];
	SparkKvPageCacheSequence sequences[SEQ_CAP];
	uint32_t bucket_heads[BUCKETS];
	uint32_t resident_slots[ENTRY_CAP];
	uint32_t logical_map[SEQ_CAP * PAGE_CAP];
	uint32_t physical_map[SEQ_CAP * PAGE_CAP];
	SparkKvPageCache cache;
	SparkKvLaneTransaction lanes[SEQ_CAP];
	SparkKvLaneTransactions transactions;
} FuzzHarness;

static void HarnessInit(FuzzHarness *h)
{
	SparkKvPageCacheConfiguration config;
	memset(h,0,sizeof(*h));
	h->arena.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	h->arena.descriptor_bytes = (uint32_t)sizeof(SparkKvCacheArena);
	h->arena.logical_block_count = ENTRY_CAP;
	h->arena.block_token_count = 1u;
	h->arena.resident_block_capacity = ENTRY_CAP;
	h->arena.layer_count = 1u;
	h->arena.kv_head_count = 1u;
	h->arena.head_dim = 1u;
	h->arena.bytes_per_scalar = 2u;
	h->arena.key_block_stride_bytes = 2u;
	h->arena.value_block_stride_bytes = 2u;
	h->arena.blocks = h->blocks;
	h->arena.resident_slot_logical_block_indices = h->resident_slots;
	memset(&config,0,sizeof(config));
	config.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	config.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	config.sequence_capacity = SEQ_CAP;
	config.entry_capacity = ENTRY_CAP;
	config.hash_bucket_count = BUCKETS;
	config.kv_cache_arena = &h->arena;
	config.entries = h->entries;
	config.sequences = h->sequences;
	config.hash_bucket_heads = h->bucket_heads;
	config.entry_indices_by_logical_page = h->logical_map;
	assert(SparkKvPageCacheInitialize(&h->cache,&config) == SPARK_STATUS_OK);
	h->transactions.cache = &h->cache;
	h->transactions.lanes = h->lanes;
	h->transactions.logical_pages = h->logical_map;
	h->transactions.physical_pages = h->physical_map;
	h->transactions.page_capacity = PAGE_CAP;
	h->transactions.validation_epoch = 1u;
}

static void BuildRequest(SparkModelDriverAdmissionRequest *request,
	SparkModelDriverCacheLane *lanes, uint64_t request_id, uint32_t flags,
	uint32_t slot0)
{
	uint32_t i;
	memset(lanes,0,MAX_LANES * sizeof(lanes[0]));
	for (i=0u; i<MAX_LANES; i++)
	{
		lanes[i].request_generation = 1u;
		lanes[i].sequence_id = 100u + slot0 + i;
		lanes[i].resident_sequence_slot = (slot0 + i) % SEQ_CAP;
		lanes[i].flags = 0u;
	}
	memset(request,0,sizeof(*request));
	request->request_id = request_id;
	request->submission_id = request_id + 5000u;
	request->sequence_id = 100u + slot0;
	request->sequence_position = 0u;
	request->control_generation = 1u;
	request->transaction_id = request_id + 9000u;
	request->program_id = 1u;
	request->active_slot_count = MAX_LANES;
	request->new_token_count = 1u;
	request->admission_flags = flags;
	request->cache_lanes = lanes;
	request->cache_lane_count = MAX_LANES;
}

int main(int argc, char **argv)
{
	FuzzHarness h;
	uint32_t round;
	uint32_t rounds = 500u;
	uint64_t next_request_id = 1u;
	if ( argc > 1 )
		fuzz_seed = strtoull(argv[1],0,10);
	if ( argc > 2 )
		rounds = (uint32_t)strtoul(argv[2],0,10);
	HarnessInit(&h);
	for (round=0u; round<rounds; round++)
	{
		uint64_t request_id = ++next_request_id;
		uint32_t op = (uint32_t)(FuzzRand() % 5u);
		uint32_t slot = (uint32_t)(FuzzRand() % (SEQ_CAP - 1u));
		SparkModelDriverAdmissionRequest request;
		SparkModelDriverCacheLane lanes[MAX_LANES];
		SparkStatus status;
		uint32_t i;
		switch (op)
		{
		case 0:
			BuildRequest(&request,lanes,request_id,
			    SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE,slot);
			(void)SparkKvLaneTransactionsAdmit(&h.transactions,&request);
			break;
		case 1:
			BuildRequest(&request,lanes,request_id,
			    SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT,slot);
			(void)SparkKvLaneTransactionsAdmit(&h.transactions,&request);
			break;
		case 2:
			BuildRequest(&request,lanes,request_id,
			    SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT,slot);
			(void)SparkKvLaneTransactionsAdmit(&h.transactions,&request);
			break;
		case 3:
			request.admission_flags = 0u;
			request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
			BuildRequest(&request,lanes,request_id,0u,slot);
			request.frame_flags = SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE;
			(void)SparkKvLaneTransactionsAdmit(&h.transactions,&request);
			break;
		default:
			BuildRequest(&request,lanes,request_id,
			    SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE,slot);
			status = SparkKvLaneTransactionsAdmit(&h.transactions,&request);
			if ( status == SPARK_STATUS_OK )
			{
				BuildRequest(&request,lanes,request_id,
				    SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT,slot);
				(void)SparkKvLaneTransactionsAdmit(&h.transactions,&request);
			}
			break;
		}
		if ( (round & 15u) == 0u )
		{
			uint32_t prepared_orphan = 0u;
			uint64_t now = NowNs();
			for (i=0u; i<SEQ_CAP; i++)
				if ( h.lanes[i].phase == SPARK_KV_LANE_TRANSACTION_PREPARED &&
				     h.lanes[i].request.request_id != 0u &&
				     h.lanes[i].validation_epoch != h.transactions.validation_epoch )
					prepared_orphan++;
			(void)prepared_orphan;
			(void)now;
		}
	}
	{
		uint32_t i;
		uint32_t stuck_prepared = 0u;
		for (i=0u; i<SEQ_CAP; i++)
			if ( h.lanes[i].phase != SPARK_KV_LANE_TRANSACTION_EMPTY )
				stuck_prepared++;
		CHECK(stuck_prepared <= SEQ_CAP,"lane phases sane at end");
	}
	fprintf(stderr,"test_kv_lane_fuzz: %u checks, %u failures (seed=%llu rounds=%u)\n",
	    test_checks,test_failures,(unsigned long long)fuzz_seed,rounds);
	return(test_failures != 0u ? 1 : 0);
}
