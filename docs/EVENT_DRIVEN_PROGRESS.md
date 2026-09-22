# Completion events and polling

Payload movement already uses RDMA writes. Replacing a repeated readiness read
with a pushed notification is a separate change from changing payload direction.
CPU polling, GPU wait kernels and network transfer time need separate measurements.

The API worker now sleeps on resident sockets and a nonblocking request-queue
pipe. Enqueue and cancellation wake that pipe; only the worker mutates the batch
engine. The batch CLI flushes events immediately after progress and checks
completion before sleeping. There is no fixed 5/10 ms polling cadence in either
consumer. Submission checkpoint persistence retains its 60-second deadline.

`SparkModelResidentClientNextProgressNs`, the pipeline aggregator and
`SparkModelBatchEngineNextProgressNs` expose absolute monotonic deadlines:
zero means no timer, one means immediately runnable. Reconnect, rejected-work
backoff, circuit delay and actual inflight timeout remain timed operations.
Readiness inspection never extends a deadline. Successful dispatch resets the
per-submission timeout and clears its rejected-work retry deadline.

Validation: real API text/token-ID and system-loopback tests pass. The added
API test waits for an idle worker, queues a request and asserts both exact fixture
tokens and a response within two seconds. Removing its enqueue wake fails the
test. Batch tests verify idle has no timer, new work is immediate, inflight work
has its real timeout, BUSY has an exact retry deadline, and completed work returns
to event-only waiting. These are host lifecycle tests, not GPU throughput evidence.

## Device and transport boundary

The developer's PR1077 commit 07333264 reports idle engines at 96% SM and 0%
memory utilization. Its success-time cancellation broadcasts are unscoped:
a faster rank can cancel another rank's valid final round. Cancellation must
preserve request ownership and drain before rearming shared cells. A same-stream
CUDA completion callback cannot prove that another stream or request has drained.

GB10 reports 64-bit stream memory operations, NOR waits and mapped host memory
support, but no remote-write flush capability. The read-only attribute probe
created no CUDA context and launched no GPU work. This does not qualify replacing
mapped-host wait kernels with hardware waits. Graph replay needs correct wait
values, explicit timeout/cancellation wakeups and payload visibility ordering.

CUDA documents [stream memory operations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html).
RDMA completion notifications are [one-shot](https://man7.org/linux/man-pages/man3/ibv_req_notify_cq.3.html)
and require rearming and draining without a lost-wakeup race. A plain incoming
RDMA write does not produce the receiver completion event needed for that design;
GPU writes to a shared host doorbell do not themselves wake a CPU file descriptor.
