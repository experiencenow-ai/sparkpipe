# Partial TP16 hardware-wait profile

The isolated GLM5.3 Flash run from source
`606cc825ca77d6ca9c3f20828b8396871e0c0f54` produced 17 token events, then remained
in admission BUSY without a completed event. It is a partial reliability result,
not a successful request or a sustained throughput qualification. The flushed
rank-zero CUPTI trace has all 121 returned buffers, zero dropped activity
records and no malformed/incomplete GPU intervals.

The measured decode window starts at output token 0 and ends at output token
16: 16 intervals in 1.480802838 seconds, or 10.80495 tokens/s for that partial,
profiled window. Startup and prefill are excluded. Same-host monotonic bounds
are `576116547247054` and `576118028049892` ns. Profiling perturbs timing; a
matching trace-disabled run remains necessary.

| Interval measurement | Median ms | Mean ms |
| --- | ---: | ---: |
| Output-to-output wall time | 82.957 | 92.550 |
| GPU activity union | 56.078 | 56.384 |
| Compute kernels | 54.647 | 54.754 |
| Mesh kernels | 1.354 | 1.367 |
| Device memory copies | 0.053 | 0.242 |
| Device memory sets | 0.020 | 0.021 |
| Uncovered GPU timeline | 26.657 | 36.167 |

Coverage merges overlapping intervals, excludes host-to-host copies, and clips
activity to each token window. Category coverage and independently calculated
medians are not additive. Gaps are not automatically network time. CUPTI records
show nonzero graph IDs on all 28,379 kernels in the independently checked first
13 decode intervals, confirming that captured kernels actually executed.

The separate rank-zero device counters over its last 16 completed chains average
1.125 ms source wait, 27.474 ms peer wait, 0.233 ms copy and 0.230 ms combine.
Average chain wall time is 87.723 ms; output-to-output time is 92.550 ms because
the observation boundaries differ. Wait counters include scheduling and daemon
progress. Copy/combine counters describe work also present in the CUPTI mesh
category; adding both would double-count it.

## Compute finding

Across 16 intervals, the leading kernel totals are 223.203 ms for one FP8 GEMM
specialization, 205.079 ms for BF16 GEMM, 194.212 ms for `Glm5NextHcMixKernel`,
and 69.142 ms for `Glm5NextHcSplitSinkhornKernel`. The two HC kernels alone average
16.460 ms per token, from 90 calls each per token.

The trace records HcMix grid `(1,1,1)`, block `(256,1,1)` and 16 KiB dynamic shared
memory. In `modules/glm5_next_resident_decode_stage/source/cuda/layer.cuh`, one
block processes the complete 16,384-element input for B1. Eight warps each compute
three of 24 mixing outputs, serially across four 4,096-element tiles. The observed
mix kernels take roughly 135 microseconds. Sinkhorn launches one 64-thread block
but maps a complete row to one thread; only thread zero performs B1's twenty
iterations. Its observed kernel duration is roughly 48 microseconds.

A bounded HcMix experiment can partition independent output mixes across blocks,
repeat the unchanged RMS reduction per block, and preserve the original warp
reduction and tile accumulation order. This needs no new scratch or extra kernel
launch. Three-block and 24-block variants require bitwise comparison with the
unchanged kernel and both repeated-weight and rotating-weight timing before a
production change. Sinkhorn is a separate optimization; it is not changed by
that experiment. DSV4 already has split-K HC machinery, but it reassociates the
reductions and is not an exact-order substitute.

## Reproduction and receipts

Remote root: `spark0:/tmp/sparkpipe-perf-606cc825ca77-graph-r2/`.
Raw `rank0.cupti.log` SHA256:
`a13f25a9ad5f76af7a04fe5086cd6abb58def929ab1f2189a53d3be74b89de23`.
`batch.events.jsonl` SHA256:
`1591546c3c3ba4a639102b4a7398a16cd78c493d2fec2d7d08a7ab8726bd43aa`.
The local receipt directory is
`/private/tmp/sparkpipe-pr1082-receipts/606cc825-r2-cupti/`; it contains the raw
files, `token0-to16.json` and `per-token.json`. Separate all-rank device counters
are in `/private/tmp/sparkpipe-pr1082-receipts/fleet-performance/sparkpipe-perf-606cc825ca77-graph-r2/collective-partial.json`.

```sh
python3 tools/tp_cupti_trace_report.py rank0.cupti.log --clock monotonic --start-ns 576116547247054 --end-ns 576118028049892 --output token0-to16.json
```

The profiler and timing tool are described in [TP_CUPTI_TRACE.md](TP_CUPTI_TRACE.md).
