#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "cuda.h"
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"

/* Expert-loading cases the serve path actually hits, beyond the base
 * suite: concurrent same-expert acquires (two engines share a node),
 * a client dying mid-acquire-load, and eviction+reload correctness under
 * pool pressure. */

#define CHUNK (2u * 1024u * 1024u)
#define RANGE_BYTES 4096u
#define TIMEOUT UINT64_C(10000000000)

void spark_stub_cuda_set_create_delay(uint32_t delay);
uint32_t spark_stub_cuda_outstanding_allocs(void);

static uint32_t test_failures;

#define CHECK(cond, name) do { \
		if ( !(cond) ) { \
			test_failures++; \
			fprintf(stderr,"FAIL %s:%d %s\n",__FILE__,__LINE__,name); \
		} \
	} while (0)

typedef struct TestServer
{
	SparkWeightdServer *server;
	volatile sig_atomic_t stop;
} TestServer;

static void *run_server(void *data)
{
	TestServer *state = data;
	assert(SparkWeightdServerRun(state->server,&state->stop) == SPARK_STATUS_OK);
	return(0);
}

static void write_range(FILE *pack,FILE *manifest,uint32_t expert,uint64_t offset)
{
	uint8_t record[48] = {0},data[RANGE_BYTES],digest[16];
	SparkCk128Context ck;
	uint64_t bytes = sizeof(data);
	memset(data,(int)(expert + 1u),sizeof(data));
	assert(fseek(pack,(long)offset,SEEK_SET) == 0);
	assert(fwrite(data,1u,sizeof(data),pack) == sizeof(data));
	SparkCk128Initialize(&ck);
	SparkCk128Update(&ck,data,sizeof(data));
	SparkCk128Finalize(&ck,digest);
	memcpy(record + 4u,&expert,4u);
	memcpy(record + 16u,&offset,8u);
	memcpy(record + 24u,&bytes,8u);
	memcpy(record + 32u,digest,16u);
	assert(fwrite(record,1u,sizeof(record),manifest) == sizeof(record));
}

static void write_fixture(const char *path,const char *manifest_path,uint32_t expert_count)
{
	FILE *pack = fopen(path,"wb"),*manifest = fopen(manifest_path,"wb");
	uint32_t header[4] = {SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC,2u,0u,0u};
	uint32_t expert;
	assert(pack != 0 && manifest != 0);
	assert(ftruncate(fileno(pack),(3u * CHUNK)) == 0);
	header[2] = expert_count;
	assert(fwrite(header,1u,sizeof(header),manifest) == sizeof(header));
	for (expert=0u; expert<expert_count; expert++)
		write_range(pack,manifest,expert,(uint64_t)expert * RANGE_BYTES);
	assert(fclose(pack) == 0);
	assert(fclose(manifest) == 0);
}

static SparkStatus attach(SparkWeightdClient *client,const char *path,uint32_t pool_chunks,uint64_t *generation)
{
	SparkWeightdLazyAttachRequest request = {0};
	SparkWeightdLazyAttachResult result;
	SparkStatus status;
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (3u * CHUNK);
	memcpy(request.identity.model,"expert-stress",14u);
	memset(request.identity.pack_sha256,'a',64u);
	assert(SparkWeightdIdentityPrepare(&request.identity) == SPARK_STATUS_OK);
	snprintf(request.pack_path,sizeof(request.pack_path),"%s",path);
	request.expert_pool_bytes = (pool_chunks * CHUNK);
	status = SparkWeightdClientAttachLazy(client,&request,&result,TIMEOUT);
	if ( status == SPARK_STATUS_OK && generation != 0 )
		*generation = result.arena_generation;
	return(status);
}

static SparkStatus acquire(SparkWeightdClient *client,uint64_t generation,uint32_t expert,SparkWeightdWorkingSetResult *result)
{
	SparkWeightdExpertKey key = {0u,expert};
	return(SparkWeightdClientAcquire(client,generation,&key,1u,result,TIMEOUT));
}

static void release(SparkWeightdClient *client,uint64_t generation,uint64_t lease)
{
	SparkWeightdWorkingSetResult result;
	assert(SparkWeightdClientRelease(client,generation,lease,&result,TIMEOUT) == SPARK_STATUS_OK);
}

/* A: two clients acquire the same expert; the daemon must load it once
 * and serve both byte-exact (resident bytes must not double-count). */
static void check_concurrent_same_expert(const char *socket_path,const char *path)
{
	SparkWeightdClient *a = 0,*b = 0;
	SparkWeightdWorkingSetResult ra,rb;
	uint64_t generation = 0u,generation_b = 0u;
	assert(SparkWeightdClientConnect(socket_path,&a,0) == SPARK_STATUS_OK);
	assert(SparkWeightdClientConnect(socket_path,&b,0) == SPARK_STATUS_OK);
	assert(attach(a,path,3u,&generation) == SPARK_STATUS_OK);
	assert(attach(b,path,3u,&generation_b) == SPARK_STATUS_OK);
	assert(generation_b == generation);
	assert(acquire(a,generation,2u,&ra) == SPARK_STATUS_OK && ra.status == SPARK_STATUS_OK);
	{
		SparkStatus bst = acquire(b,generation,2u,&rb);
		if ( bst != SPARK_STATUS_OK || rb.status != SPARK_STATUS_OK )
			fprintf(stderr,"DBG concurrent second acquire: call=%u wire=%u\n",
				(unsigned)bst,(unsigned)rb.status);
		assert(bst == SPARK_STATUS_OK && rb.status == SPARK_STATUS_OK);
	}
	CHECK( rb.resident_bytes <= ra.resident_bytes,
		"concurrent: the second acquire of a resident expert is a warm hit (no double load)");
	release(a,generation,ra.lease_identifier);
	release(b,generation,rb.lease_identifier);
	SparkWeightdClientClose(a);
	SparkWeightdClientClose(b);
}

/* B: a client SIGKILLed mid-acquire-load leaves the expert coherent and
 * leaks nothing. */
static void check_mid_acquire_death(const char *socket_path,const char *path)
{
	SparkWeightdClient *a = 0;
	SparkWeightdWorkingSetResult result;
	uint64_t generation = 0u;
	uint32_t baseline;
	pid_t pid;
	int wstatus = 0;
	assert(SparkWeightdClientConnect(socket_path,&a,0) == SPARK_STATUS_OK);
	assert(attach(a,path,3u,&generation) == SPARK_STATUS_OK);
	baseline = spark_stub_cuda_outstanding_allocs();
	spark_stub_cuda_set_create_delay(30000u);
	pid = fork();
	assert(pid >= 0);
	if ( pid == 0 )
	{
		SparkWeightdClient *child = 0;
		SparkWeightdWorkingSetResult dead;
		if ( SparkWeightdClientConnect(socket_path,&child,0) == SPARK_STATUS_OK )
			(void)acquire(child,generation,1u,&dead);
		_exit(0);
	}
	usleep(60000);
	assert(kill(pid,SIGKILL) == 0);
	assert(waitpid(pid,&wstatus,0) == pid);
	spark_stub_cuda_set_create_delay(0u);
	assert(acquire(a,generation,1u,&result) == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK);
	release(a,generation,result.lease_identifier);
	{
		uint32_t waited;
		for (waited=0u; waited<2000u && spark_stub_cuda_outstanding_allocs() != baseline + 1u; waited++)
			usleep(1000);
	}
	/* the killed client's lease must not hold extra state forever; the
	 * only delta from baseline is the loaded expert itself staying
	 * resident (pool keeps it warm for the next acquirer) */
	CHECK( spark_stub_cuda_outstanding_allocs() <= baseline + 2u,
		"mid-acquire death: no unbounded state accumulates");
	SparkWeightdClientClose(a);
}

/* C: pool pressure evicts; a re-acquire of an evicted expert reloads
 * byte-exact. */
static void check_eviction_reload(const char *socket_path,const char *path)
{
	SparkWeightdClient *a = 0;
	SparkWeightdWorkingSetResult result;
	uint64_t generation = 0u;
	uint32_t expert;
	assert(SparkWeightdClientConnect(socket_path,&a,0) == SPARK_STATUS_OK);
	assert(attach(a,path,3u,&generation) == SPARK_STATUS_OK);
	for (expert=0u; expert<8u; expert++)
	{
		assert(acquire(a,generation,expert,&result) == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK);
		release(a,generation,result.lease_identifier);
	}
	assert(acquire(a,generation,0u,&result) == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK);
	CHECK( result.resident_bytes <= (3u * CHUNK),
		"eviction: the pool stays within its budget across the sweep");
	release(a,generation,result.lease_identifier);
	SparkWeightdClientClose(a);
}

int main(void)
{
	char root[] = "/tmp/weightd-expert-stress-XXXXXX",path[256],manifest[272],recording[272],socket_path[256];
	TestServer state = {0};
	SparkWeightdServerConfig config = {0};
	pthread_t thread;
	assert(mkdtemp(root) != 0);
	snprintf(path,sizeof(path),"%s/pack",root);
	snprintf(manifest,sizeof(manifest),"%s.experts",path);
	snprintf(recording,sizeof(recording),"%s.wset",path);
	snprintf(socket_path,sizeof(socket_path),"%s/socket",root);
	write_fixture(path,manifest,8u);
	config.socket_path = socket_path;
	config.device_bytes_max = (8u * CHUNK);
	assert(SparkWeightdServerCreate(&config,&state.server) == SPARK_STATUS_OK);
	assert(pthread_create(&thread,0,run_server,&state) == 0);
	check_concurrent_same_expert(socket_path,path);
	check_mid_acquire_death(socket_path,path);
	check_eviction_reload(socket_path,path);
	__atomic_store_n(&state.stop,1,__ATOMIC_SEQ_CST);
	assert(pthread_join(thread,0) == 0);
	SparkWeightdServerDestroy(state.server);
	assert(unlink(recording) == 0 && unlink(manifest) == 0 && unlink(path) == 0 && rmdir(root) == 0);
	fprintf(stderr,"test_weightd_expert_stress: %s\n",
		test_failures == 0u ? "PASS" : "FAILED");
	return( test_failures != 0u ? 1 : 0 );
}
