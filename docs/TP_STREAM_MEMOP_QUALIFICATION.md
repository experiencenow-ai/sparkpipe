# CUDA stream memory wait prototype

This is an isolated qualification tool, not an enabled production collective.
The original local gate prototype compiled and ran on Spark0 with CUDA 13.0.88
for GB10 `sm_121a`. Source commit `6db0dd333ebb089336cd8d9362c5e6a6d36ff202`,
receipt `spark0:/tmp/sparkpipe-hw-wait-6db0dd33/run.log`, reported 4,258.53 ns per
91-node update replay (46.80 ns/node) and all three local GPU cases passed.
Source SHA256 was `307b5b21fea6cd4b022794e43eefd435ed4151a988120c4d270c32b9c68e4425`;
binary SHA256 was `e10ad2bc35c8881b2a3a39019529e37816c6379672a59417d8c300355a064dc3`.
The two existing weightd/resident processes and their measured GPU allocations
were unchanged after the run. This was a shared host timing sample, not an
isolated performance benchmark or GLM inference result.

The current extension adds RDMA modes and expands the scalar payload to 4 KiB
on cache lines separate from readiness, error and GPU output. Those changes
require new local and remote qualification; the earlier receipt does not
qualify NIC visibility or the extended probe. The extension compiled for
`sm_121a` in `spark0:/tmp/sparkpipe-rdma-memop-20260922`; receipts are
`build/qualification/compile.log`, `help.log` and `SHA256SUMS`. Only compile and
help were executed for this extension before this checkpoint.

## Safe invocation

The wrapper and compiled binary default to help without CUDA or RDMA calls. Compilation
requires an explicit toolkit and target plus libibverbs headers/library. Any CUDA context creation requires
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
that cancellation never consumes the payload. The local mode involves no network or service.

## Two-host NIC visibility check

Both processes require `--run` and explicit device, port, GID, IPv4 address and
unique TCP port. The receiver binds that address; the sender connects to it.
`--iterations` repeats all three cases 1..128 times (default 8), within the same
30-second process alarm. The sender creates no CUDA context. Each process owns
one RC QP, one 8-entry CQ and one MR; at most three WRs are outstanding. Every
WR is signaled and checked before source reuse. Normal teardown destroys the
QP before deregistering its MR; failed/partial posts terminate the isolated
probe rather than reusing memory. The OS releases only this process's resources
on the bounded alarm. No daemon, persistent config or existing QP is modified.

For every trial the sender first RDMA-writes a stale generation. The receiver
requires the launched graph to remain incomplete with untouched output after
that write. The sender then posts payload, error and current readiness in that
order on the same RC QP. The GPU consumes all 512 words and produces a checksum;
the receiver computes its expectation from the trial number, without reading
the received payload on CPU before GPU completion. Cancellation must consume
nothing, and recovery must observe the subsequent NIC-written error clear.
The receiver never writes a successful readiness value or transport ACK into
the MR. TCP messages coordinate phases only; NIC writes release the graph.

Read-only sysfs inventory on 2026-09-22 found `rocep1s0f0`, port 1, GID index 3
active on both hosts, RoCE v2, IPv4-mapped GIDs `10.10.200.0` (Spark0) and
`10.10.200.1` (Spark1). Selection is explicit, with no interface/GID fallback;
active MTU below 4096 is rejected. An assigned run can use these commands after
checking that the chosen TCP port is unused:

```sh
python3 tools/tp_stream_memop_probe.py --run --rdma-receive --ib-device rocep1s0f0 --ib-port 1 --gid-index 3 --address 10.10.200.0 --tcp-port 49387 --iterations 8
python3 tools/tp_stream_memop_probe.py --run --rdma-send --ib-device rocep1s0f0 --ib-port 1 --gid-index 3 --address 10.10.200.0 --tcp-port 49387 --iterations 8
```

This tests a CUDA-mapped pinned host MR on one link. A pass does not qualify the
production memfd/shared registration path, cross-process ownership, allreduce,
missing-peer recovery, or model throughput. No unsupported remote FLUSH is
requested; failure of ordering or visibility is a qualification failure.

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
for [stream memory operations](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEMOP.html).
[Graph parameter updates](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__GRAPH.html)
affect future launches and require the original node to remain in its graph.
CUDA 13 [capture inspection and dependency APIs](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-driver-api/group__CUDA__STREAM.html)
include edge data; the prototype uses those current signatures.

## Qualification gates

The original local mode passed; rerun it for the extended payload layout. Before
serving integration, qualify actual RDMA-to-GPU visibility on two assigned hosts
and exercise stale CQEs/requests, skew, missing peers, timeout, cancellation,
failed Begin/End and source-slot reuse. Require rounding-sensitive BF16 SUM,
U64 MAX and rank-major gather at TP2/3/4/8/16 for B1 and logical B2+ split across
execution rows and payload chunks. Compare numerical output and whole-chain
latency with identical model/configuration inputs. Local gate success is not
allreduce qualification; no throughput benefit is claimed by this prototype.
