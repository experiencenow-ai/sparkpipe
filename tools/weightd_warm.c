#include "sparkpipe/spark_weightd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argument_count,char **arguments)
{
	const char *socket,*pack_path,*digest,*revision,*topology_text;
	SparkWeightdLazyAttachRequest request;
	SparkWeightdLazyAttachResult attach_result;
	SparkWeightdWorkingSetResult working_set;
	SparkWeightdExpertKey *keys;
	SparkWeightdClient *client;
	struct stat pack_status;
	SparkStatus status;
	uint64_t timeout_ns;
	uint32_t layers,experts,layer,expert;
	if ( argument_count < 6 )
	{
		fprintf(stderr,
			"usage: %s SOCKET PACK SHA256 REVISION TOPOLOGY [LAYERS=45 [EXPERTS=288 [TIMEOUT_S=1800]]]\n",
			arguments[0]);
		return 2;
	}
	socket = arguments[1];
	pack_path = arguments[2];
	digest = arguments[3];
	revision = arguments[4];
	topology_text = arguments[5];
	layers = argument_count > 6 ? (uint32_t)strtoul(arguments[6],0,10) : 45u;
	experts = argument_count > 7 ? (uint32_t)strtoul(arguments[7],0,10) : 288u;
	timeout_ns = (uint64_t)(argument_count > 8 ? strtoul(arguments[8],0,10) : 1800u) *
		UINT64_C(1000000000);
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
