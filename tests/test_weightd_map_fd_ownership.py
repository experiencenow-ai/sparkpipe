from pathlib import Path
import argparse
import os
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r'''
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
static int test_close(int fd);
#define close test_close
#include "MAP_SOURCE"
#undef close
static int watched[2],close_count[2],replacement;
static unsigned imports;
static int test_close(int fd)
{
    unsigned index;
    for (index=0u; index<2u && watched[index]!=fd; index++);
    assert(index<2u);
    close_count[index]++;
    int result=close(fd);
    if (close_count[index]==1) assert(dup2(replacement,fd)==fd);
    return result;
}
CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
    void *fd,CUmemAllocationHandleType type)
{
    (void)handle;(void)fd;(void)type;imports++;
    return CUDA_ERROR_INVALID_VALUE;
}
CUresult cuMemMap(CUdeviceptr address,size_t bytes,size_t offset,
    CUmemGenericAllocationHandle handle,unsigned long long flags)
{
    (void)address;(void)bytes;(void)offset;(void)handle;(void)flags;
    abort();
}
CUresult cuMemSetAccess(CUdeviceptr address,size_t bytes,
    const CUmemAccessDesc *descriptors,size_t count)
{
    (void)address;(void)bytes;(void)descriptors;(void)count;
    abort();
}
static void check(unsigned mode)
{
    SparkWeightdMap map={0};
    SparkWeightdExportBatch batch={0};
    CUmemGenericAllocationHandle handles[2]={0};
    uint64_t owners[2]={0};
    uint8_t mapped[2]={1u,1u};
    uint32_t last=0u;
    imports=0u;
    replacement=open("/dev/zero",O_RDONLY);
    assert(replacement>=0);
    for (unsigned i=0u;i<2u;i++)
    {
        watched[i]=open("/dev/null",O_RDONLY);
        assert(watched[i]>=0);
        close_count[i]=0;
        batch.fds[i]=watched[i];
        batch.chunk_indices[i]=i;
    }
    map.chunk_bytes=4096u;map.chunk_count=2u;
    map.mapped=mapped;map.owners=owners;map.handles=handles;
    batch.chunk_bytes=mode==1u?8192u:4096u;
    batch.chunk_count=2u;batch.batch_count=2u;
    if (mode==2u) mapped[0]=0u;
    SparkStatus expected=mode==1u?SPARK_STATUS_SCHEMA_ERROR:
        mode==2u?SPARK_STATUS_IO_ERROR:SPARK_STATUS_OK;
    assert(map_import_batch(&map,0u,&batch,&last,0u,2u,
        map_now()+UINT64_C(1000000000))==expected);
    assert(imports==(mode==2u?1u:0u));
    for (unsigned i=0u;i<2u;i++)
    {
        unsigned char value=255u;
        assert(fcntl(watched[i],F_GETFD)>=0);
        assert(read(watched[i],&value,1u)==1 && value==0u);
        assert(close_count[i]==1);
        assert(close(watched[i])==0);
    }
    assert(close(replacement)==0);
}
int main(void)
{
    for (unsigned mode=0u;mode<3u;mode++) check(mode);
    puts("PASS weightd map descriptor ownership: cached, schema rejection, import failure; six reused descriptors preserved");
    return 0;
}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'runtime/spark_weightd_map.c')
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='weightd-map-fd-') as directory:
        temp = Path(directory)
        source = temp / 'test.c'
        source.write_text(HARNESS.replace('MAP_SOURCE', str(args.source.resolve())))
        command = [os.environ.get('CC', 'cc'), '-std=c11', '-D_POSIX_C_SOURCE=200809L',
                   '-ffunction-sections', '-fdata-sections', '-I' + str(ROOT / 'include'),
                   '-I' + str(ROOT / 'tests/cuda_stub'), str(source), '-pthread',
                   '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                   '-o', str(temp / 'test')]
        if args.sanitize:
            command += ['-fsanitize=address,undefined', '-fno-omit-frame-pointer']
        subprocess.run(command, check=True)
        subprocess.run([str(temp / 'test')], check=True, timeout=10)


if __name__ == '__main__':
    main()
