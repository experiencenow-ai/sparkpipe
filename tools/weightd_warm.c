#include "sparkpipe/spark_weightd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argument_count,char **arguments)
{
	const char *socket,*pack_path,*digest,*revision,*topology_text;
	const char *wset_path = 0;
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyAttachResult attach_result;
	SparkWeightdWorkingSetResult working_set;
	SparkWeightdExpertKey *keys;
	SparkWeightdClient *client;
	struct stat pack_status;
	FILE *wset_in;
	SparkStatus status;
	uint64_t timeout_ns;
	uint32_t layers,experts,layer,expert,key_count;
	if ( argument_count < 6 )
	{
		fprintf(stderr,
			"usage: %s SOCKET PACK SHA256 REVISION TOPOLOGY [LAYERS=45 [EXPERTS=288 [TIMEOUT_S=1800]]]\n"
			"       %s SOCKET PACK SHA256 REVISION TOPOLOGY --wset FILE [TIMEOUT_S=300]\n"
			"       the wset file holds (layer,expert) u32 pairs — one recorded smoke test;\n"
			"       its experts load in ONE batched acquire (snapshot the daemon's live\n"
			"       recording, <PACK>.wset, per test after running it)\n",
			arguments[0],arguments[0]);
		return 2;
	}
	socket = arguments[1];
	pack_path = arguments[2];
	digest = arguments[3];
	revision = arguments[4];
	topology_text = arguments[5];
	if ( argument_count > 6 && strcmp(arguments[6],"--wset") == 0 )
	{
		if ( argument_count < 8 )
		{
			fprintf(stderr,"weightd_warm: --wset needs FILE\n");
			return 2;
		}
		wset_path = arguments[7];
		timeout_ns = (uint64_t)(argument_count > 8 ? strtoul(arguments[8],0,10) : 300u) *
			UINT64_C(1000000000);
		layers = 0u;
		experts = 0u;
	}
	else
	{
		layers = argument_count > 6 ? (uint32_t)strtoul(arguments[6],0,10) : 45u;
		experts = argument_count > 7 ? (uint32_t)strtoul(arguments[7],0,10) : 288u;
		timeout_ns = (uint64_t)(argument_count > 8 ? strtoul(arguments[8],0,10) : 1800u) *
			UINT64_C(1000000000);
	}
	if ( strlen(digest) != 64u || stat(pack_path,&pack_status) != 0 )
	{
		fprintf(stderr,"weightd_warm: bad digest or unreadable pack\n");
		return 2;
	}
	memset(&request,0,sizeof(request));
	memcpy(request.identity.pack_sha256,digest,65u);
	(void)snprintf(request.identity.model,sizeof(request.identity.model),"%s","glm5_next_stage");
	(void)snprintf(request.identity.revision,sizeof(request.identity.revision),"%s",revision);
	request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
	request.identity.arena_bytes = (uint64_t)pack_status.st_size;
	request.identity.topology = (uint32_t)strtoul(topology_text,0,10);
	memcpy(request.pack_path,pack_path,strlen(pack_path) + 1u);
	{
		const char *pool_text = getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES");
		request.expert_pool_bytes = pool_text != 0 && pool_text[0] != '\0' ?
			(uint64_t)strtoull(pool_text,0,10) : UINT64_C(34359738368);
	}
	if ( wset_path != 0 )
	{
		uint32_t capacity = 1024u;
		uint32_t in_layer,in_expert,i;
		keys = (SparkWeightdExpertKey *)calloc(capacity,sizeof(*keys));
		if ( keys == 0 )
			return 1;
		wset_in = fopen(wset_path,"rb");
		if ( wset_in == 0 )
		{
			fprintf(stderr,"weightd_warm: cannot open wset %s\n",wset_path);
			free(keys);
			return 2;
		}
		key_count = 0u;
		while ( fread(&in_layer,sizeof(in_layer),1u,wset_in) == 1u &&
			fread(&in_expert,sizeof(in_expert),1u,wset_in) == 1u )
		{
			int duplicate = 0;
			for (i=0u; i<key_count; i++)
				if ( keys[i].layer == in_layer && keys[i].expert == in_expert )
				{
					duplicate = 1;
					break;
				}
			if ( duplicate != 0 )
				continue;
			if ( key_count == capacity )
			{
				SparkWeightdExpertKey *grown;
				capacity *= 2u;
				grown = (SparkWeightdExpertKey *)realloc(keys,
					(size_t)capacity * sizeof(*grown));
				if ( grown == 0 )
				{
					(void)fclose(wset_in);
					free(keys);
					return 1;
				}
				keys = grown;
			}
			keys[key_count].layer = in_layer;
			keys[key_count].expert = in_expert;
			key_count++;
		}
		(void)fclose(wset_in);
		if ( key_count == 0u )
		{
			fprintf(stderr,"weightd_warm: wset %s is empty\n",wset_path);
			free(keys);
			return 2;
		}
		status = SparkWeightdClientConnect(socket,&client,0);
		if ( status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"weightd_warm: connect failed status=%d\n",(int)status);
			free(keys);
			return 1;
		}
		status = SparkWeightdClientAttachLazy(client,&request,&attach_result,timeout_ns);
		if ( status != SPARK_STATUS_OK || attach_result.status != SPARK_STATUS_OK )
		{
			fprintf(stderr,"weightd_warm: attach-lazy failed status=%d attach_status=%d\n",
				(int)status,(int)attach_result.status);
			free(keys);
			return 1;
		}
		{
			uint64_t began_ns,ended_ns;
			struct timespec ts;
			(void)clock_gettime(CLOCK_MONOTONIC,&ts);
			began_ns = (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
			fprintf(stderr,"weightd_warm: WSET-ONE-SHOT file=%s keys=%u\n",
				wset_path,key_count);
			status = SparkWeightdClientAcquire(client,attach_result.arena_generation,
				keys,key_count,&working_set,timeout_ns);
			if ( status == SPARK_STATUS_OK && working_set.status == SPARK_STATUS_OK )
				status = SparkWeightdClientRelease(client,attach_result.arena_generation,
					working_set.lease_identifier,&working_set,timeout_ns);
			(void)clock_gettime(CLOCK_MONOTONIC,&ts);
			ended_ns = (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
			fprintf(stderr,"weightd_warm: WSET-%s keys=%u elapsed_ms=%llu\n",
				status == SPARK_STATUS_OK && working_set.status == SPARK_STATUS_OK ?
				"WARM" : "FAILED",
				key_count,
				(unsigned long long)((ended_ns - began_ns) / 1000000ull));
		}
		free(keys);
		(void)SparkWeightdClientClose(client);
		return 0;
	}
	keys = (SparkWeightdExpertKey *)calloc(experts,sizeof(*keys));
	if ( keys == 0 )
		return 1;
	status = SparkWeightdClientConnect(socket,&client,0);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"weightd_warm: connect failed status=%d\n",(int)status);
		free(keys);
		return 1;
	}
	status = SparkWeightdClientAttachLazy(client,&request,&attach_result,timeout_ns);
	if ( status != SPARK_STATUS_OK || attach_result.status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"weightd_warm: attach-lazy failed status=%d attach_status=%d\n",
			(int)status,(int)attach_result.status);
		free(keys);
		return 1;
	}
	fprintf(stderr,"weightd_warm: attached generation=%llu chunks=%u — warming %u layers x %u experts\n",
		(unsigned long long)attach_result.arena_generation,
		attach_result.chunk_count,layers,experts);
	for (layer=0u; layer<layers; layer++)
	{
		for (expert=0u; expert<experts; expert++)
		{
			keys[expert].layer = layer;
			keys[expert].expert = expert;
		}
		status = SparkWeightdClientAcquire(client,attach_result.arena_generation,
			keys,experts,&working_set,timeout_ns);
		if ( status == SPARK_STATUS_OK && working_set.status == SPARK_STATUS_OK )
		{
			status = SparkWeightdClientRelease(client,attach_result.arena_generation,
				working_set.lease_identifier,&working_set,timeout_ns);
			fprintf(stderr,"weightd_warm: layer %u WARM\n",layer);
		}
		else
			fprintf(stderr,"weightd_warm: layer %u FAILED status=%d working=%d\n",
				layer,(int)status,(int)working_set.status);
	}
	free(keys);
	(void)SparkWeightdClientClose(client);
	return 0;
}
