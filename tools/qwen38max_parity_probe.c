#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd_lease.h"
#include "sparkpipe/spark_weightd_manifest.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int parity_stat_bytes(const char *path, uint64_t *bytes)
{
	struct stat info;
	if ( stat(path,&info) != 0 || S_ISREG(info.st_mode) == 0 )
		return(-1);
	*bytes = (uint64_t)info.st_size;
	return(0);
}

static int parity_join(char *destination, size_t capacity, const char *source, const char *suffix)
{
	size_t bytes = strlen(source) + strlen(suffix) + 1u;
	if ( bytes > capacity )
		return(-1);
	memcpy(destination,source,strlen(source));
	memcpy(destination + strlen(source),suffix,strlen(suffix) + 1u);
	return(0);
}

static int parity_derive_module_keys(uint32_t layer, uint32_t resident, SparkWeightdExpertKey *keys, uint32_t capacity, uint32_t *count)
{
	uint32_t offsets[SPARK_WEIGHTD_LEASE_GROUPS_MAX + 1u];
	uint32_t i;
	if ( resident == 0u || resident > SPARK_WEIGHTD_LEASE_GROUPS_MAX )
		return(-1);
	for ( i = 0u; i <= resident; i++ )
		offsets[i] = i;
	return(SparkWeightdRouteKeys(layer,offsets,resident,resident,keys,capacity,count) == SPARK_STATUS_OK ? 0 : -1);
}

int main(int argc, char **argv)
{
	SparkWeightdManifest manifest;
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
	const SparkWeightdRangeGroup *module_group,*global_group;
	uint64_t pack_bytes;
	uint32_t layer,rank,resident,first,key_count = 0u,i,module_hit = 0u,module_miss = 0u,global_hit = 0u,global_miss = 0u;
	char manifest_path[4096];
	if ( argc != 6 )
	{
		fprintf(stderr,"usage: %s <pack> <layer> <tp_rank> <routed_expert_count> <tp_degree>\n",argv[0]);
		return(2);
	}
	layer = (uint32_t)strtoul(argv[2],0,10);
	rank = (uint32_t)strtoul(argv[3],0,10);
	resident = (uint32_t)strtoul(argv[4],0,10) / (uint32_t)strtoul(argv[5],0,10);
	first = rank * resident;
	if ( parity_stat_bytes(argv[1],&pack_bytes) != 0 )
	{
		fprintf(stderr,"parity-1 pack stat failed %s\n",argv[1]);
		return(1);
	}
	if ( parity_join(manifest_path,sizeof(manifest_path),argv[1],".experts") != 0 ||
		SparkWeightdManifestLoad(manifest_path,pack_bytes,&manifest) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"parity-2 manifest load failed %s\n",manifest_path);
		return(1);
	}
	if ( parity_derive_module_keys(layer,resident,keys,SPARK_WEIGHTD_LEASE_GROUPS_MAX,&key_count) != 0 )
	{
		fprintf(stderr,"parity-3 route keys failed layer=%u resident=%u\n",layer,resident);
		SparkWeightdManifestDestroy(&manifest);
		return(1);
	}
	printf("parity pack=%s layer=%u rank=%u resident=%u first=%u keys=%u groups=%u\n",
		argv[1],layer,rank,resident,first,key_count,manifest.group_count);
	for ( i = 0u; i < key_count; i++ )
	{
		module_group = SparkWeightdManifestFind(&manifest,keys[i].layer,keys[i].expert);
		global_group = SparkWeightdManifestFind(&manifest,keys[i].layer,keys[i].expert + first);
		if ( module_group != 0 )
			module_hit++;
		else
			module_miss++;
		if ( global_group != 0 )
			global_hit++;
		else
			global_miss++;
		printf("parity key slot=%u module=(layer=%u expert=%u %s) global=(layer=%u expert=%u %s)\n",
			i,keys[i].layer,keys[i].expert,module_group != 0 ? "HIT" : "MISS",
			keys[i].layer,keys[i].expert + first,global_group != 0 ? "HIT" : "MISS");
	}
	printf("parity verdict module_hit=%u module_miss=%u global_hit=%u global_miss=%u basis=%s\n",
		module_hit,module_miss,global_hit,global_miss,
		global_miss == 0u && module_miss != 0u ? "global" : (module_miss == 0u && global_miss != 0u ? "module-local" : "ambiguous"));
	SparkWeightdManifestDestroy(&manifest);
	return(0);
}
