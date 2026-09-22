#include "tests/fixtures/tp_mesh_hardware_fixture.h"
#include "node/weightd_mesh.c"
#include <assert.h>

void SparkTestMeshWaitInitialize(void *region,uint32_t degree)
{
    assert(degree >= 2u && degree <= SPARK_WEIGHTD_MESH_RANKS_PER_BAND && region != 0);
    memset(&weightd_mesh,0,sizeof(weightd_mesh));
    weightd_mesh.recv_buffer = region;
    weightd_mesh.rank_mask = (1u << degree) - 1u;
    weightd_mesh.activity_owners = 1u;
    weightd_mesh.mesh_ready = 1u;
}

void SparkTestMeshWaitPoll(uint64_t now_ns)
{
    SparkWeightdMeshWaitRequestsPoll(now_ns);
}
