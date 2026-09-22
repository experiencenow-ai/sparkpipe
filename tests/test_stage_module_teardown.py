#!/usr/bin/env python3
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (
    ('qwen4_flash', 'Qwen4Flash', 'tp_collective_initialized', 'ple_prev_context_u32', True),
    ('qwen38_max', 'Qwen38Max', 'tp_collective_initialized', 't1_stage_hidden', True),
    ('gemma4', 'Gemma4', 'tp_collective_initialized', 'slots[0].host_row_lane_indices', False),
    ('muse_glimmer', 'MuseGlimmer', 'tp_collective_initialized', 'kv_logical_to_slot', False),
)
HARNESS = r'''
#include <assert.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "sparkpipe/spark_weightd_lazy_pack.h"
#include "sparkpipe/spark_stage_kv_client.h"
static void TestFree(void *pointer);
#define free TestFree
#include "modules/@FAMILY@_resident_decode_stage/source/spark_@FAMILY@_resident_decode_stage_module.c"
#include "runtime/stage_module_lifecycle.c"
#undef free
static Spark@PREFIX@ModuleState *owner;
static void *sentinel;
static uint32_t fail_collective,fail_lazy,fail_slots;
static uint32_t collectives,lazy_calls,unmaps,host_frees,state_frees,ledger_frees;
static void TestFree(void *pointer)
{
    if (pointer == 0)
        return;
    assert(owner->tp_device_collective.implementation == 0);
    @CHECK_LAZY@
    if (pointer == sentinel)
        host_frees++;
    if (pointer == owner)
    {
        assert(ledger_frees == 1u && host_frees == 1u);
        state_frees++;
    }
    free(pointer);
}
void SparkTpDeviceCollectiveDestroy(SparkTpDeviceCollective *collective)
{
    assert(collective == &owner->tp_device_collective);
    assert(collective->implementation != 0 && host_frees == 0u);
    collectives++;
    if (fail_collective != 0u)
        return;
    collective->implementation = 0;
}
SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack)
{
    assert(owner->tp_device_collective.implementation == 0 && host_frees == 0u);
    @CHECK_PACK@
    lazy_calls++;
    if (fail_lazy != 0u)
        return SPARK_STATUS_IO_ERROR;
    unmaps++;
    free(pack);
    return SPARK_STATUS_OK;
}
SparkStatus SparkStageModuleWaitForSlots(const char *tag,const atomic_uint *states,uint32_t count,uint64_t timeout)
{
    (void)tag;(void)states;(void)count;(void)timeout;
    return fail_slots != 0u ? SPARK_STATUS_BUSY : SPARK_STATUS_OK;
}
void SparkStageModuleLedgerRelease(SparkStageModuleLedger *ledger)
{
    assert(ledger == &owner->ledger && host_frees == 1u);
    assert(owner->tp_device_collective.implementation == 0);
    @CHECK_LAZY@
    ledger_frees++;
}
void SparkStageKvClientClose(SparkStageKvClient *client) { (void)client; }
void SparkStageModuleStageTimingShutdown(SparkStageModuleStageTiming *timing) { (void)timing; }
cudaError_t cudaFree(void *pointer) { assert(pointer == 0);return cudaSuccess; }
int main(void)
{
    const SparkStageModuleLifecycleOps ops = {
        .describe = Spark@PREFIX@ModuleDescribe,
        .state_destroy = Spark@PREFIX@ModuleStateTeardown
    };
    for (uint32_t failures=0u; failures<8u; failures++)
    {
        collectives=lazy_calls=unmaps=host_frees=state_frees=ledger_frees=0u;
        fail_collective=failures&1u;fail_lazy=failures&2u;fail_slots=failures&4u;
        owner=calloc(1u,sizeof(*owner));assert(owner != 0);
        sentinel=malloc(32u);assert(sentinel != 0);
        owner->@SENTINEL@=sentinel;
        owner->@FLAG@=1u;
        owner->tp_device_collective.implementation=(void *)(uintptr_t)1u;
        @SET_LAZY@
        if (fail_slots != 0u)
        {
            assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_BUSY);
            assert(collectives == 0u && host_frees == 0u && ledger_frees == 0u && state_frees == 0u);
            fail_slots=0u;
        }
        if (fail_collective != 0u)
        {
            assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_BUSY);
            assert(lazy_calls == 0u && host_frees == 0u && ledger_frees == 0u && state_frees == 0u);
            assert(owner->tp_device_collective.implementation != 0);
            fail_collective=0u;
        }
        @FAIL_LAZY@
        assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_OK);
        assert(collectives == 1u+(failures&1u) && host_frees == 1u && ledger_frees == 1u && state_frees == 1u);
        @CHECK_UNMAP@
    }
    assert(SparkStageModuleLifecycleDestroy(0,&ops) == SPARK_STATUS_OK);
    puts("PASS @FAMILY@ actual teardown ownership and retry");
    return 0;
}
'''


GLM52 = r'''#include <assert.h>
#include "modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_module.c"
static uint32_t fail_release,records[2],released[2];
SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t id,cudaStream_t stream)
{
    assert(map == (void *)(uintptr_t)1u && stream == (void *)(uintptr_t)2u && id>=10u && id<=11u);
    assert(records[id-10u] == 0u);
    records[id-10u]++;
    return SPARK_STATUS_OK;
}
SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t id,uint64_t timeout)
{
    (void)timeout;
    assert(map == (void *)(uintptr_t)1u && id>=10u && id<=11u && records[id-10u] == 1u);
    if (fail_release != 0u)
        return SPARK_STATUS_IO_ERROR;
    assert(released[id-10u] == 0u);
    released[id-10u]++;
    return SPARK_STATUS_OK;
}
cudaError_t cudaEventSynchronize(cudaEvent_t event) { assert(event == (void *)(uintptr_t)3u);return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t stream) { assert(stream == (void *)(uintptr_t)2u);return cudaSuccess; }
int main(void)
{
    SparkGlm52ModuleState state = {0};
    SparkGlm52ExecutionSlot slot = {0};
    SparkWeightdLazyPack pack = {0};
    SparkGlm52TpChain chain = {0},*recovered;
    state.lazy_pack=&pack;pack.map=(void *)(uintptr_t)1u;
    slot.stream=(void *)(uintptr_t)2u;slot.expert_done_event=(void *)(uintptr_t)3u;
    for (uint32_t mask=1u; mask<4u; mask++)
    {
        memset(&chain,0,sizeof(chain));memset(records,0,sizeof(records));memset(released,0,sizeof(released));
        chain.state=&state;chain.slot=&slot;
        chain.retired_lease=(mask&1u) != 0u ? 10u : 0u;chain.retired_begun=(mask&1u) != 0u;
        chain.expert_lease=(mask&2u) != 0u ? 11u : 0u;chain.expert_lease_begun=(mask&2u) != 0u;
        state.lazy_retained[0]=&chain;fail_release=1u;recovered=0;
        assert(SparkGlm52LazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_IO_ERROR);
        assert(state.lazy_retained[0] == &chain && recovered == 0);
        assert(released[0] == 0u && released[1] == 0u);
        fail_release=0u;
        assert(SparkGlm52LazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_OK);
        assert(state.lazy_retained[0] == 0 && recovered == &chain);
        assert(chain.retired_lease == 0u && chain.expert_lease == 0u);
        assert(records[0] == ((mask&1u) != 0u) && records[1] == ((mask&2u) != 0u));
        assert(released[0] == records[0] && released[1] == records[1]);
        assert(SparkGlm52LazyRecoverLease(&state,0u,&recovered) == SPARK_STATUS_NOT_FOUND && recovered == 0);
    }
    puts("PASS glm52 retained-only, active-only and combined expert lease retry");
    return 0;
}
'''


def main():
    with tempfile.TemporaryDirectory(prefix='spark-teardown-') as directory:
        for family, prefix, flag, sentinel, lazy in (*FAMILIES, ('glm52', '', '', '', False)):
            harness = HARNESS
            replacements = {
                'FAMILY': family, 'PREFIX': prefix, 'FLAG': flag, 'SENTINEL': sentinel,
                'CHECK_LAZY': 'assert(owner->lazy_pack == 0);' if lazy else '',
                'CHECK_PACK': 'assert(pack == owner->lazy_pack);' if lazy else '(void)pack;',
                'SET_LAZY': 'owner->lazy_pack=calloc(1u,sizeof(*owner->lazy_pack));assert(owner->lazy_pack != 0);' if lazy else '',
                'FAIL_LAZY': r'''if (fail_lazy != 0u) {
                    assert(SparkStageModuleLifecycleDestroy(owner,&ops) == SPARK_STATUS_IO_ERROR);
                    assert(owner->tp_device_collective.implementation == 0 && owner->lazy_pack != 0);
                    assert(host_frees == 0u && ledger_frees == 0u && state_frees == 0u);
                    fail_lazy=0u;
                }''' if lazy else '',
                'CHECK_UNMAP': 'assert(unmaps == 1u && lazy_calls == 1u+((failures&2u) != 0u));' if lazy else 'assert(unmaps == 0u && lazy_calls == 0u);',
            }
            for key, value in replacements.items():
                harness = harness.replace('@' + key + '@', value)
            source, binary = Path(directory) / (family + '.c'), Path(directory) / family
            source.write_text(GLM52 if family == 'glm52' else harness)
            includes = ['.', 'include', 'tests/cuda_stub', 'model-families/common/include',
                        f'model-families/{family}/include', f'model-families/{family}/include/sparkpipe', f'modules/{family}_resident_decode_stage/include',
                        f'modules/{family}_resident_decode_stage/source']
            subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE', '-O1',
                            '-ffunction-sections', '-fdata-sections',
                            '-Wl,-dead_strip' if sys.platform == 'darwin' else '-Wl,--gc-sections',
                            *['-I' + path for path in includes], '-DSPARK_LLM_MTP_LAYER_COUNT=0u',
                            f'-D{family.upper()}_MODEL_REVISION="fixture"', '-DQWEN38_MODEL_REVISION="fixture"',
                            f'-D{family.upper()}_CONTRACT_SHA256="fixture"',
                            '-DGLM_EXPERT_WEIGHT_CODEC=5', '-DGLM_EXPERT_CODEC_NAME="fp8"', '-DGLM_CONTRACT_SHA256="fixture"',
                            *shlex.split(os.environ.get('SPARK_TEST_SANITIZER_FLAGS', '')),
                            str(source), '-o', str(binary)], cwd=ROOT, check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    main()
