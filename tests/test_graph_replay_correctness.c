#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <pthread.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>

#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_weightd.h"

#define TEST_DEGREE 4
#define TEST_ROWS 1
#define TEST_HIDDEN 512
#define TEST_ROUNDS 7
#define TEST_REPLAYS 5
#define TEST_PAYLOAD_BYTES (TEST_ROWS * TEST_HIDDEN * 2)
#define TEST_WATCHDOG_NS (45ull * 1000000000ull)
#define TEST_CHAIN_KEY 1000001ull
#define TEST_STEP_F 0.125f

extern int SparkGlm5NextLaunchSeedF32(void *stream,float *destination,
    const void *a,const void *b,uint32_t element_count);
extern int SparkGlm5NextLaunchAddF32(void *stream,float *destination,
    const void *b,uint32_t element_count);
extern int SparkGlm5NextLaunchRoundF32(void *stream,void *destination,
    const float *source,uint32_t element_count);

static int g_failures;
static void *g_mesh;
static pthread_barrier_t g_barrier;

static void Failf(const char *format,...)
{
    va_list args;
    va_start(args,format);
    fprintf(stderr,"FAIL: ");
    vfprintf(stderr,format,args);
    fprintf(stderr,"\n");
    va_end(args);
    __sync_fetch_and_add(&g_failures,1);
}

#define CHECK(cond,name) do { \
    if (!(cond)) { \
        Failf("%s:%d rank-context %s",__FILE__,__LINE__,name); \
    } \
} while (0)

static void CompletionNoop(void *context,SparkStatus status)
{
    (void)context;(void)status;
}

static SparkStatus CombineSeed(void *context,void *destination,
    const void *a,const void *b,uint32_t element_count,void *stream)
{
    (void)context;
    return SparkGlm5NextLaunchSeedF32(stream,(float *)destination,a,b,
        element_count) == cudaSuccess ? SPARK_STATUS_OK :
        SPARK_STATUS_IO_ERROR;
}

static SparkStatus CombineAdd(void *context,void *destination,
    const void *source,uint32_t element_count,void *stream)
{
    (void)context;
    return SparkGlm5NextLaunchAddF32(stream,(float *)destination,source,
        element_count) == cudaSuccess ? SPARK_STATUS_OK :
        SPARK_STATUS_IO_ERROR;
}

static SparkStatus CombineRound(void *context,void *destination,
    const void *source,uint32_t element_count,void *stream)
{
    (void)context;
    return SparkGlm5NextLaunchRoundF32(stream,destination,
        (const float *)source,element_count) == cudaSuccess ?
        SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
}

typedef struct RankThread
{
    uint32_t rank;
    SparkTpDeviceCollective collective;
    cudaStream_t stream;
    void *scratch;
    uint16_t *host_scratch;
    uint64_t rounds_done;
} RankThread;

static RankThread g_ranks[TEST_DEGREE];

static uint64_t NowNs(void)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC,&now);
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

static void WaitStream(cudaStream_t stream,uint32_t rank,const char *phase)
{
    uint64_t stop = NowNs() + TEST_WATCHDOG_NS;
    for ( ;; )
    {
        cudaError_t err = cudaStreamQuery(stream);
        if ( err == cudaSuccess )
            return;
        if ( err != cudaErrorNotReady )
        {
            Failf("rank=%u %s stream error %s",rank,phase,
                cudaGetErrorString(err));
            (void)cudaGetLastError();
            return;
        }
        if ( NowNs() >= stop )
        {
            Failf("rank=%u %s stream hung (watchdog)",rank,phase);
            return;
        }
        usleep(500);
    }
}

static float Bf16ToFloat(uint16_t bits)
{
    uint32_t word = (uint32_t)bits << 16;
    float value;
    memcpy(&value,&word,sizeof(value));
    return value;
}

static uint16_t FloatToBf16(float value)
{
    uint32_t word;
    uint16_t bits;
    memcpy(&word,&value,sizeof(word));
    bits = (uint16_t)(word >> 16);
    return bits;
}

static void EnqueueRound(RankThread *rank_thread,uint64_t ordinal,
    uint32_t capture_mode)
{
    SparkTpDeviceCollectiveSubmission sub;
    SparkStatus status;
    memset(&sub,0,sizeof(sub));
    sub.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    sub.descriptor_bytes = sizeof(sub);
    sub.slot_index = 0u;
    sub.active_sequence_count = TEST_ROWS;
    sub.ordinal = ordinal;
    sub.local_device = rank_thread->scratch;
    sub.full_device = rank_thread->scratch;
    sub.cuda_stream = rank_thread->stream;
    sub.completion_function = capture_mode ? 0 : CompletionNoop;
    sub.completion_context = 0;
    status = SparkTpDeviceCollectiveEnqueue(&rank_thread->collective,&sub,
        SPARK_TP_DEVICE_COLLECTIVE_OPERATION_ALL_REDUCE_SUM_BF16);
    if ( capture_mode == 0u )
    {
        CHECK(status == SPARK_STATUS_OK,"eager enqueue ok");
        WaitStream(rank_thread->stream,rank_thread->rank,"eager round");
        rank_thread->rounds_done++;
    }
    else
        CHECK(status == SPARK_STATUS_OK,"captured enqueue ok");
}

static void CheckValue(RankThread *rank_thread,const char *phase)
{
    uint32_t element;
    float expected = (float)((uint64_t)rank_thread->rounds_done *
        ((uint64_t)TEST_DEGREE * (TEST_DEGREE + 1u) / 2u)) * TEST_STEP_F;
    cudaMemcpy(rank_thread->host_scratch,rank_thread->scratch,
        TEST_PAYLOAD_BYTES,cudaMemcpyDeviceToHost);
    for ( element = 0u; element < TEST_PAYLOAD_BYTES / 2u; element++ )
    {
        float got = Bf16ToFloat(rank_thread->host_scratch[element]);
        float delta = got - expected;
        if ( delta < 0.0f )
            delta = -delta;
        if ( delta > 0.0625f )
        {
            Failf("rank=%u %s element=%u got=%f expected=%f",
                rank_thread->rank,phase,element,got,expected);
            return;
        }
    }
}

static void *RankMain(void *argument)
{
    RankThread *rank_thread = (RankThread *)argument;
    SparkTpDeviceCollectiveConfiguration config;
    cudaGraph_t graph = 0;
    cudaGraphExec_t exec = 0;
    uint32_t round;
    uint32_t replay;
    float initial = (float)(rank_thread->rank + 1u) * TEST_STEP_F;

    memset(&config,0,sizeof(config));
    config.abi_version = SPARK_TP_DEVICE_COLLECTIVE_ABI_VERSION;
    config.descriptor_bytes = sizeof(config);
    config.tp_rank = rank_thread->rank;
    config.tp_degree = TEST_DEGREE;
    config.local_hidden_dimension = TEST_HIDDEN;
    config.collective_identifier = 0;
    config.connect_timeout_milli = 30000;
    config.operation_timeout_milli = 10000;
    config.max_active_sequence_count = 1;
    config.combine_f32_seed_function = CombineSeed;
    config.combine_f32_add_function = CombineAdd;
    config.round_f32_function = CombineRound;
    cudaStreamCreate(&rank_thread->stream);
    cudaMalloc(&rank_thread->scratch,TEST_PAYLOAD_BYTES);
    rank_thread->host_scratch = calloc(TEST_PAYLOAD_BYTES,1u);
    for ( round = 0u; round < TEST_PAYLOAD_BYTES / 2u; round++ )
        rank_thread->host_scratch[round] = FloatToBf16(initial);
    cudaMemcpy(rank_thread->scratch,rank_thread->host_scratch,
        TEST_PAYLOAD_BYTES,cudaMemcpyHostToDevice);
    CHECK(SparkTpDeviceCollectiveCreate(&config,
        &rank_thread->collective) == SPARK_STATUS_OK,"collective create");
    CHECK(SparkTpDeviceCollectivePrepareReceiveBf16(
        &rank_thread->collective,g_mesh,TEST_ROWS,TEST_HIDDEN,0,
        rank_thread->stream) == SPARK_STATUS_OK,"prepare receive");

    CHECK(SparkTpDeviceCollectiveChainKey(&rank_thread->collective,
        TEST_CHAIN_KEY) == SPARK_STATUS_OK,"chain key");

    for ( round = 0u; round < TEST_ROUNDS; round++ )
        EnqueueRound(rank_thread,rank_thread->rank * 3u + round * 5u,0u);
    CheckValue(rank_thread,"eager complete");
    pthread_barrier_wait(&g_barrier);

    CHECK(SparkTpDeviceCollectiveArmCapture(
        &rank_thread->collective) == SPARK_STATUS_OK,"arm capture");
    for ( round = 0u; round < TEST_ROUNDS; round++ )
        EnqueueRound(rank_thread,rank_thread->rank * 7u + round * 3u,1u);
    {
        cudaError_t err;
        err = cudaStreamEndCapture(rank_thread->stream,&graph);
        if ( err != cudaSuccess || graph == 0 )
            Failf("rank=%u end capture %s",rank_thread->rank,
                cudaGetErrorString(cudaGetLastError()));
    }
    pthread_barrier_wait(&g_barrier);
    if ( graph != 0 &&
         cudaGraphInstantiate(&exec,graph,0) != cudaSuccess )
    {
        Failf("rank=%u instantiate %s",rank_thread->rank,
            cudaGetErrorString(cudaGetLastError()));
        exec = 0;
    }
    SparkTpDeviceCollectiveDisarmCapture(&rank_thread->collective);
    pthread_barrier_wait(&g_barrier);

    for ( replay = 0u; replay < TEST_REPLAYS && exec != 0; replay++ )
    {
        uint64_t graph_error;
        char phase[32];
        snprintf(phase,sizeof(phase),"replay %u",replay);
        CHECK(SparkTpDeviceCollectiveGraphCancelSeed(
            &rank_thread->collective,rank_thread->stream) ==
            SPARK_STATUS_OK,"cancel seed");
        CHECK(SparkTpDeviceCollectiveGraphPreLaunch(
            &rank_thread->collective,rank_thread->stream) ==
            SPARK_STATUS_OK,"pre-launch pads");
        if ( cudaGraphLaunch(exec,rank_thread->stream) != cudaSuccess )
        {
            Failf("rank=%u replay %u launch %s",rank_thread->rank,replay,
                cudaGetErrorString(cudaGetLastError()));
            break;
        }
        WaitStream(rank_thread->stream,rank_thread->rank,phase);
        graph_error = SparkTpDeviceCollectiveGraphError(
            &rank_thread->collective);
        SparkTpDeviceCollectiveClearGraphError(&rank_thread->collective);
        if ( graph_error != 0ull )
        {
            Failf("rank=%u replay %u graph error %llu",rank_thread->rank,
                replay,(unsigned long long)graph_error);
            SparkTpDeviceCollectiveBroadcastCancel(
                &rank_thread->collective);
            break;
        }
        rank_thread->rounds_done += TEST_ROUNDS;
        CheckValue(rank_thread,phase);
        if ( replay + 1u < TEST_REPLAYS )
        {
            EnqueueRound(rank_thread,rank_thread->rank * 11u + replay * 2u,
                0u);
            CheckValue(rank_thread,"interleaved eager");
        }
        pthread_barrier_wait(&g_barrier);
    }

    if ( exec != 0 )
        cudaGraphExecDestroy(exec);
    if ( graph != 0 )
        cudaGraphDestroy(graph);
    SparkTpDeviceCollectiveDestroy(&rank_thread->collective);
    cudaStreamDestroy(rank_thread->stream);
    cudaFree(rank_thread->scratch);
    free(rank_thread->host_scratch);
    return 0;
}

int main(void)
{
    pthread_t threads[TEST_DEGREE];
    uint32_t rank;
    const char *socket_path = getenv("SPARK_WEIGHTD_SOCKET");

    if ( socket_path == 0 || socket_path[0] == '\0' )
    {
        fprintf(stderr,
            "FAIL: SPARK_WEIGHTD_SOCKET not set (rig needs a private weightd)\n");
        return 2;
    }
    if ( cudaMalloc(&g_mesh,SPARK_WEIGHTD_MESH_REGION_BYTES) != cudaSuccess )
    {
        fprintf(stderr,"FAIL: mesh region alloc %s\n",
            cudaGetErrorString(cudaGetLastError()));
        return 2;
    }
    cudaMemset(g_mesh,0,SPARK_WEIGHTD_MESH_REGION_BYTES);
    pthread_barrier_init(&g_barrier,0,TEST_DEGREE);
    for ( rank = 0u; rank < TEST_DEGREE; rank++ )
    {
        g_ranks[rank].rank = rank;
        pthread_create(&threads[rank],0,RankMain,&g_ranks[rank]);
    }
    for ( rank = 0u; rank < TEST_DEGREE; rank++ )
        pthread_join(threads[rank],0);
    pthread_barrier_destroy(&g_barrier);
    cudaFree(g_mesh);
    fprintf(stderr,"test_graph_replay_correctness: %d failures\n",
        g_failures);
    return g_failures ? 1 : 0;
}
