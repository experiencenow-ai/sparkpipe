#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_sha256.h"
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define GEN_HEAD_GAP (4u * 1024u * 1024u)
#define GEN_EXPERT_BYTES (64u * 1024u)

int main(int argc,char **argv)
{
	uint8_t record[48],data[GEN_EXPERT_BYTES],digest[16];
	uint64_t offset = GEN_HEAD_GAP;
	uint32_t resident,base,layers,layer,slot,expert;
	int32_t fd;
	char sha[SPARK_SHA256_HEX_BYTES];
	if ( argc != 6 )
	{
		fprintf(stderr,"usage: qmax_rank_window_gen pack manifest resident base layers\n");
		return(2);
	}
	resident = (uint32_t)strtoul(argv[3],0,10);
	base = (uint32_t)strtoul(argv[4],0,10);
	layers = (uint32_t)strtoul(argv[5],0,10);
	if ( resident == 0u || layers == 0u )
		return(2);
	fd = open(argv[1],O_RDWR | O_CREAT | O_TRUNC,0644);
	if ( fd < 0 )
	{
		fprintf(stderr,"gen: pack open failed\n");
		return(1);
	}
	if ( ftruncate(fd,(off_t)(GEN_HEAD_GAP + (uint64_t)layers * resident * GEN_EXPERT_BYTES)) != 0 )
	{
		fprintf(stderr,"gen: truncate failed\n");
		return(1);
	}
	{
		FILE *man = fopen(argv[2],"wb");
		uint32_t header[4] = {0x58504557u,2u,layers * resident,0u};
		if ( man == 0 )
		{
			fprintf(stderr,"gen: manifest open failed\n");
			return(1);
		}
		fwrite(header,1u,sizeof(header),man);
		for (layer = 0u; layer < layers; layer++)
			for (slot = 0u; slot < resident; slot++)
			{
				SparkCk128Context ck;
				expert = base + slot;
				memset(data,(int32_t)(1u + ((layer * 7u + expert * 3u) % 251u)),sizeof(data));
				SparkCk128Initialize(&ck);
				SparkCk128Update(&ck,data,sizeof(data));
				SparkCk128Finalize(&ck,digest);
				memset(record,0,sizeof(record));
				memcpy(record,&layer,4u);
				memcpy(record + 4u,&expert,4u);
				memcpy(record + 16u,&offset,8u);
				{
					uint64_t bytes = GEN_EXPERT_BYTES;
					memcpy(record + 24u,&bytes,8u);
				}
				memcpy(record + 32u,digest,16u);
				fwrite(record,1u,sizeof(record),man);
				if ( pwrite(fd,data,sizeof(data),(off_t)offset) != (ssize_t)sizeof(data) )
				{
					fprintf(stderr,"gen: pwrite failed\n");
					return(1);
				}
				offset += GEN_EXPERT_BYTES;
			}
		fclose(man);
	}
	close(fd);
	if ( SparkSha256File(argv[1],sha) != SPARK_STATUS_OK )
	{
		fprintf(stderr,"gen: sha failed\n");
		return(1);
	}
	printf("%s\n",sha);
	return(0);
}
