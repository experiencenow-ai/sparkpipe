#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int parse_positive(const char *text,uint64_t maximum,uint64_t *value)
{
    char *end;
    if ( text == 0 || text[0] < '0' || text[0] > '9' )
        return 0;
    errno = 0;
    *value = strtoull(text,&end,10);
    return errno == 0 && *end == '\0' && *value != 0u && *value <= maximum;
}

int main(int argument_count,char **arguments)
{
    SparkWeightdLazyAttachRequest request = {0};
    SparkWeightdLazyAttachResult attached = {0};
    SparkWeightdManifest manifest = {0};
    SparkWeightdWorkingSetResult result;
    SparkWeightdClient *client = 0;
    SparkWeightdExpertKey *keys = 0;
    struct stat pack;
    SparkStatus status;
    char manifest_path[SPARK_WEIGHTD_PATH_BYTES + 16u];
    uint64_t layers = 45u,experts = 288u,seconds = 1800u,topology;
    uint32_t first,index,count;
    int exit_status = 1;
    if ( argument_count < 6 || argument_count > 9 ||
         !parse_positive(arguments[5],UINT32_MAX,&topology) ||
         (argument_count > 6 && !parse_positive(arguments[6],UINT32_MAX,&layers)) ||
         (argument_count > 7 && !parse_positive(arguments[7],SPARK_WEIGHTD_EXPERT_COUNT_MAX,&experts)) ||
         (argument_count > 8 && !parse_positive(arguments[8],UINT64_MAX / UINT64_C(1000000000),&seconds)) ||
         !parse_positive(getenv("SPARK_WEIGHTD_EXPERT_POOL_BYTES"),
             SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT,&request.expert_pool_bytes) )
    {
        fprintf(stderr,"usage: weightd_warm SOCKET PACK SHA256 REVISION TOPOLOGY [LAYERS=45 [EXPERTS=288 [TIMEOUT_S=1800]]]; finite SPARK_WEIGHTD_EXPERT_POOL_BYTES is required\n");
        return 2;
    }
    if ( strlen(arguments[2]) >= sizeof(request.pack_path) ||
         strlen(arguments[3]) != 64u ||
         strspn(arguments[3],"0123456789abcdefABCDEF") != 64u ||
         strlen(arguments[4]) >= sizeof(request.identity.revision) ||
         stat(arguments[2],&pack) != 0 || !S_ISREG(pack.st_mode) || pack.st_size <= 0 )
    {
        fprintf(stderr,"weightd_warm: invalid pack or identity\n");
        return 2;
    }
    memcpy(request.identity.pack_sha256,arguments[3],65u);
    strcpy(request.identity.model,"glm5_next_stage");
    strcpy(request.identity.revision,arguments[4]);
    request.identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    request.identity.arena_bytes = (uint64_t)pack.st_size;
    request.identity.topology = (uint32_t)topology;
    strcpy(request.pack_path,arguments[2]);
    snprintf(manifest_path,sizeof(manifest_path),"%s.experts",arguments[2]);
    status = SparkWeightdManifestLoad(manifest_path,request.identity.arena_bytes,&manifest);
    if ( status != SPARK_STATUS_OK || manifest.group_count == 0u )
    {
        fprintf(stderr,"weightd_warm: expert manifest failed status=%d\n",(int)status);
        goto done;
    }
    for (index=0u; index<manifest.group_count; index++)
        if ( manifest.groups[index].layer >= layers || manifest.groups[index].expert >= experts )
        {
            fprintf(stderr,"weightd_warm: declared geometry excludes manifest expert %u/%u\n",
                manifest.groups[index].layer,manifest.groups[index].expert);
            goto done;
        }
    keys = calloc((size_t)experts,sizeof(*keys));
    if ( keys == 0 )
        goto done;
    status = SparkWeightdClientConnect(arguments[1],&client,0);
    if ( status == SPARK_STATUS_OK )
        status = SparkWeightdClientAttachLazy(client,&request,&attached,seconds * UINT64_C(1000000000));
    if ( status != SPARK_STATUS_OK || attached.status != SPARK_STATUS_OK )
    {
        fprintf(stderr,"weightd_warm: attach failed status=%d daemon=%u\n",(int)status,attached.status);
        goto done;
    }
    for (first=0u; first<manifest.group_count; first=index)
    {
        count = 0u;
        for (index=first; index<manifest.group_count &&
             manifest.groups[index].layer == manifest.groups[first].layer; index++)
        {
            if ( count >= experts )
                goto done;
            keys[count].layer = manifest.groups[index].layer;
            keys[count++].expert = manifest.groups[index].expert;
        }
        memset(&result,0,sizeof(result));
        status = SparkWeightdClientAcquire(client,attached.arena_generation,
            keys,count,&result,seconds * UINT64_C(1000000000));
        if ( status == SPARK_STATUS_OK && result.status == SPARK_STATUS_OK )
            status = SparkWeightdClientRelease(client,attached.arena_generation,
                result.lease_identifier,&result,seconds * UINT64_C(1000000000));
        if ( status != SPARK_STATUS_OK || result.status != SPARK_STATUS_OK )
        {
            fprintf(stderr,"weightd_warm: layer %u FAILED status=%d daemon=%u\n",
                keys[0].layer,(int)status,result.status);
            goto done;
        }
        fprintf(stderr,"weightd_warm: layer %u WARM experts=%u\n",keys[0].layer,count);
    }
    exit_status = 0;
done:
    SparkWeightdClientClose(client);
    SparkWeightdManifestDestroy(&manifest);
    free(keys);
    return exit_status;
}
