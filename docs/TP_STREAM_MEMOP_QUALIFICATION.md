# CUDA stream memory wait prototype

This is an isolated qualification tool, not an enabled production collective.
The prototype compiles with CUDA 13.0.88 for GB10 `sm_121a`; it has not been run.
The original compile receipt is
`spark0:/tmp/sparkpipe-glm-c29823be/build/qualification/stream-memop-prototype.log`.
Compilation does not measure update latency, memory visibility or GPU wait behavior.

## Safe invocation

The wrapper and compiled binary default to help without CUDA calls. Compilation
requires an explicit toolkit and target. Any CUDA context creation requires
`--run`; the additional `--gpu-waits` flag permits three graph launches.

```sh
python3 tools/tp_stream_memop_probe.py
python3 tools/tp_stream_memop_probe.py --compile --cuda-root /usr/local/cuda --arch sm_121a
python3 tools/tp_stream_memop_probe.py --run
python3 tools/tp_stream_memop_probe.py --run --gpu-waits
```

Run only in an assigned isolated GPU window. `--run` creates a context, captures
91 wait nodes plus a guarded consumer, instantiates the graph, and times 1,000
sets of 91 node-value updates without launching a graph. `--gpu-waits` also
checks stale/delayed readiness, cancellation before release, and recovery. A
30-second process alarm bounds the standalone experiment. The guard asserts
that cancellation never consumes the payload. No network or service is involved.

Read-only device queries on Spark0 returned success for these CUDA 13 attributes:

| Attribute | ID | Value |
| --- | ---: | ---: |
| CAN_USE_64_BIT_STREAM_MEM_OPS | 122 | 1 |
| CAN_USE_STREAM_WAIT_VALUE_NOR | 123 | 1 |
| CAN_FLUSH_REMOTE_WRITES | 98 | 0 |
| CAN_MAP_HOST_MEMORY | 19 | 1 |
| HOST_REGISTER_SUPPORTED | 99 | 1 |

These describe the sampled device, not every deployment. The tool checks 64-bit
wait support and fails explicitly if absent. It obtains the mapped device alias
with `cudaHostGetDevicePointer` and does not request unsupported remote FLUSH.
CPU publication into this mapped gate cannot qualify RDMA payload visibility.

## Integration constraints

The current transport already pushes payloads and tails by RDMA WRITE into each
receiver's mapped local host memory. GPU `ld.global.cv` polling is not an RDMA
read from another host. The proposed change removes SM-resident waiting.

1. Reuse the collective, mesh control region and acknowledged activity interval.
   Add a versioned, fixed current-wait record per rank/band: generation, wait ID,
   condition kind, expected collective tag, peer mask, readiness ID and error.
   Source-credit and received-peer waits use the same record. Do not introduce
   another transport or a silent spinning fallback.
2. A graph batch-memop node publishes the request fields and ID in order, then
   waits on the separate local readiness ID. The existing active weightd worker
   checks SHIPPED or the peer tails. On cancellation it publishes the error
   before releasing readiness, including later requests in that canceled
   activity. It never forges SHIPPED or payload tails. Idle work still sleeps;
   this first design retains bounded CPU polling of the current active request.
3. Each copy/publish/combine phase checks the error before using payloads. A new
   wait record replaces the old record only after its consumer finishes on the
   same stream. Increasing readiness IDs must have a proven bounded, nonwrapping
   comparison domain; stale requests cannot satisfy a later generation.
4. B1 keeps its existing copy/publication/combine arithmetic. B2+ splits the
   fused tree at phase boundaries, reusing `SparkTpMeshTreeRoute` and
   `SparkTpMeshTreeLevels`. Preserve FP32 partials through reduction and broadcast
   and round BF16 once. Sparse phases, logical batch selection and execution-row
   chunking must remain identical.
5. Derive immutable request IDs/tags from chain epoch, phase count, slot parity
   and prior source ownership before replay. Update node values before launch
   and retain the original graph with its executable; current GLM destroys the
   original graph immediately. Graph updates have one owner and visible edges
   order requests, waits and consumers.
6. Host timeout/cancel may release only mapped local error/readiness state. It
   cannot call CUDA from a CUDA host callback. EndChain still requires terminal
   stream ownership; CQ-qualified source ownership must independently drain.

NVIDIA documents mapped-device-pointer requirements and CUDA-visible ordering
for [stream memory operations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html).
[Graph parameter updates](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html)
affect future launches and require the original node to remain in its graph.
CUDA 13 [capture inspection and dependency APIs](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__STREAM.html)
include edge data; the prototype uses those current signatures.

## Qualification gates

First measure 91-node update cost, then run the three local GPU checks. Before
serving integration, qualify actual RDMA-to-GPU visibility on two assigned hosts
and exercise stale CQEs/requests, skew, missing peers, timeout, cancellation,
failed Begin/End and source-slot reuse. Require rounding-sensitive BF16 SUM,
U64 MAX and rank-major gather at TP2/3/4/8/16 for B1 and logical B2+ split across
execution rows and payload chunks. Compare numerical output and whole-chain
latency with identical model/configuration inputs. Local gate success is not
allreduce qualification; no throughput benefit is claimed by this prototype.
