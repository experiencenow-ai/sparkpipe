#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unistd.h>

#define RT(call) do { cudaError_t rc = (call); if (rc != cudaSuccess) { std::fprintf(stderr,"FAIL line=%d runtime=%s\n",__LINE__,cudaGetErrorString(rc)); return 1; } } while (0)
#define DR(call) do { CUresult rc = (call); if (rc != CUDA_SUCCESS) { const char *name = nullptr; cuGetErrorName(rc,&name); std::fprintf(stderr,"FAIL line=%d driver=%s\n",__LINE__,name); return 1; } } while (0)
#define REQUIRE(cond) do { if (!(cond)) { std::fprintf(stderr,"FAIL assertion line=%d\n",__LINE__); return 1; } } while (0)

struct Gate {
    unsigned long long ready;
    unsigned long long error;
    unsigned long long payload;
    unsigned long long output;
    unsigned long long consumed;
    unsigned long long status;
};

__global__ void Consume(volatile Gate *gate)
{
    unsigned long long error = gate->error;
    if (error == 0u) {
        gate->output = gate->payload;
        gate->consumed = gate->consumed + 1u;
    }
    gate->status = error;
    __threadfence_system();
}

int main(int argc,char **argv)
{
    if (argc == 1 || (argc == 2 && std::strcmp(argv[1],"--help") == 0)) {
        std::printf("usage: %s --run [--gpu-waits]\nWithout --run no CUDA calls are made. --run measures graph update cost; --gpu-waits adds three GPU launches.\n",argv[0]);
        return 0;
    }
    if (argc < 2 || argc > 3 || std::strcmp(argv[1],"--run") != 0 ||
        (argc == 3 && std::strcmp(argv[2],"--gpu-waits") != 0)) return 2;
    const bool run = argc == 3;
    alarm(30);
    DR(cuInit(0));
    CUdevice device;
    int supported;
    DR(cuDeviceGet(&device,0));
    DR(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,device));
    REQUIRE(supported == 1);
    RT(cudaSetDevice(0));
    RT(cudaFree(nullptr));
    CUcontext context;
    DR(cuCtxGetCurrent(&context));
    Gate *host = nullptr,*mapped = nullptr;
    RT(cudaHostAlloc(reinterpret_cast<void **>(&host),sizeof(*host),cudaHostAllocMapped));
    RT(cudaHostGetDevicePointer(reinterpret_cast<void **>(&mapped),host,0));
    std::memset(host,0,sizeof(*host));
    cudaStream_t stream;
    RT(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    RT(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    CUstreamCaptureStatus capture_status;
    cuuint64_t capture_id;
    CUgraph graph;
    const CUgraphNode *dependencies;
    size_t dependency_count;
    DR(cuStreamGetCaptureInfo(reinterpret_cast<CUstream>(stream),&capture_status,
        &capture_id,&graph,&dependencies,nullptr,&dependency_count));
    REQUIRE(capture_status == CU_STREAM_CAPTURE_STATUS_ACTIVE);
    constexpr unsigned int count = 91u;
    CUgraphNode nodes[count];
    CUstreamBatchMemOpParams operation = {};
    operation.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    operation.waitValue.address = reinterpret_cast<CUdeviceptr>(mapped) + offsetof(Gate,ready);
    operation.waitValue.value64 = 1u;
    operation.waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    CUDA_BATCH_MEM_OP_NODE_PARAMS params = {};
    params.ctx = context;
    params.count = 1u;
    params.paramArray = &operation;
    for (unsigned int i = 0; i < count; ++i) {
        DR(cuGraphAddBatchMemOpNode(&nodes[i],graph,dependencies,dependency_count,&params));
        dependencies = &nodes[i];
        dependency_count = 1u;
    }
    DR(cuStreamUpdateCaptureDependencies(reinterpret_cast<CUstream>(stream),&nodes[count-1u],nullptr,1u,CU_STREAM_SET_CAPTURE_DEPENDENCIES));
    Consume<<<1,1,0,stream>>>(mapped);
    RT(cudaGetLastError());
    cudaGraph_t captured;
    RT(cudaStreamEndCapture(stream,&captured));
    cudaGraphExec_t executable;
    RT(cudaGraphInstantiate(&executable,captured,0));
    auto started = std::chrono::steady_clock::now();
    for (unsigned int replay = 1; replay <= 1000u; ++replay) {
        operation.waitValue.value64 = replay;
        for (unsigned int i = 0; i < count; ++i)
            DR(cuGraphExecBatchMemOpNodeSetParams(reinterpret_cast<CUgraphExec>(executable),nodes[i],&params));
    }
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    std::printf("graph_nodes=%u updates=%u ns_per_replay=%.2f ns_per_node=%.2f gpu_launches=0\n",count,1000u,double(ns)/1000.0,double(ns)/(1000.0*count));
    if (run) {
        for (unsigned int trial = 0; trial < 3u; ++trial) {
            unsigned long long generation = 2000u + trial;
            operation.waitValue.value64 = generation;
            for (unsigned int i = 0; i < count; ++i)
                DR(cuGraphExecBatchMemOpNodeSetParams(reinterpret_cast<CUgraphExec>(executable),nodes[i],&params));
            host->payload = 0x1234567800000000ull + trial;
            host->output = ~0ull;
            host->consumed = 0u;
            host->status = ~0ull;
            host->error = 0u;
            __atomic_store_n(&host->ready,generation - 1u,__ATOMIC_RELEASE);
            RT(cudaGraphLaunch(executable,stream));
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            REQUIRE(cudaStreamQuery(stream) == cudaErrorNotReady);
            REQUIRE(host->output == ~0ull && host->consumed == 0u && host->status == ~0ull);
            const bool cancel = trial == 1u;
            __atomic_store_n(&host->error,cancel ? 1u : 0u,__ATOMIC_RELEASE);
            __atomic_store_n(&host->ready,generation,__ATOMIC_RELEASE);
            RT(cudaStreamSynchronize(stream));
            REQUIRE(host->status == (cancel ? 1u : 0u));
            REQUIRE(host->consumed == (cancel ? 0u : 1u));
            REQUIRE(host->output == (cancel ? ~0ull : host->payload));
        }
        std::printf("PASS delayed-ready stale-generation cancellation-before-release recovery; gpu_launches=3; network_visibility=unqualified\n");
    }
    RT(cudaGraphExecDestroy(executable));
    RT(cudaGraphDestroy(captured));
    RT(cudaStreamDestroy(stream));
    RT(cudaFreeHost(host));
    return 0;
}
