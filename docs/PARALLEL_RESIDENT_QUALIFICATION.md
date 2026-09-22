# Parallel resident qualification

The release gate is concurrent inference with correct tokens and bounded shared
residency, followed by verified cleanup. Starting processes or printing readiness
is insufficient. Different model families also need their own execution evidence;
a repeated GLM deployment does not establish their numerical correctness.

## First shared-daemon fleet campaign

Source `2b24873d89c0da9744391d104f9d61369fb750f0` passed a fresh 164-check host
campaign with zero failures, setup failures or timeouts, plus CUDA compilation,
GPU module publication, driver linking/loading and exact artifact verification.
CI `compile-sm121a` also passed. The campaign selected 114 of 115 registered C
targets; the Qwen GPU target and 165 unselected Python files remain separately
accounted for. Host fixtures do not establish every model's GPU behavior.

The authoritative controller is `mac@mac-studio`, using its existing
`/Users/mac/.sparkpipe/queue` ledger. Three stale dispatchers were stopped and
replaced by one dispatcher from the pinned source. Existing notes/invalid jobs
and a ledger backup were preserved. The old fleet-agent services were stopped
on all sixteen Sparks; orphaned Spark3 test residents were drained as well.
Every new participant ran inside its queue-owned systemd cgroup.

Each Spark ran one shared weightd plus independent GLM5.3 Flash FP8 TP16
resident processes. Every resident had B1, one in-flight submission, one active
sequence, context capacity 512, 128 logical/physical pages and a 2 GiB backing
limit. Ports, runtime/cache roots and explicitly reserved collective lanes were
separate. CUDA function/data loading were LAZY, connection count 32, hardware
waits enabled and graph/expert pinning explicitly enabled.

The pinned `.wset` contains 336 expert keys and was warmed once per daemon.
Graph execution subsequently pins every expert; this campaign accounts for the
whole 21.7 GB pack, not a small working set. The declared daemon budget was
28 GiB plus 512 MiB overhead; each resident had a 4 GiB device reservation and
4 GiB host reservation. Measured device allocations were 20,874 MiB per daemon
and 3,434 MiB per resident.

| Residents per Spark | Result | Complete outputs | Decode tok/s per resident | Aggregate common-window tok/s | Device MiB per Spark |
| --- | --- | ---: | --- | ---: | ---: |
| 2 | PASS | 64 | 6.545, 6.301 | 12.940 | 27,742 |
| 3 | PASS | 96 | 4.322, 4.326, 4.330 | 12.958 | 31,176 |
| 4 | FAIL | Three requests completed; one stalled | Not qualified | Not qualified | Recorded in retained snapshot |

All successful requests produced the exact same 32 tokens as the isolated
baseline. The common decode windows overlapped for 4.714 and 7.023 seconds.
All 48 and 64 owned processes respectively were absent after successful
shutdown; the queue independently confirmed stopped control groups.
Independent engines share device capacity but do not combine requests into one
batched GEMM. These results therefore establish concurrent developer execution,
not continuous-batch throughput scaling or independent model-quality validation.

## Four-resident failure and causal regression

The four-resident run exposed a completion boundary at rank zero, lane two,
submission 172. The graph completed successfully, then `EndChain` returned
`BUSY` and the module retained ownership without completing the request. The
other three residents finished their 32-token requests. Logs were retained and
the queue cancelled the failed run; all participant control groups stopped.

The common CUDA wait helper published callback completion before the callback
returned, allowing a successful wait while the stream still reported not-ready.
A direct GPU reproduction using the unchanged helper observed one such success
in 200,000 waits across four independent CUDA processes. The preceding 50,000
single-process waits observed none. The deterministic host regression controls
the notification/retirement gap instead of relying on this rare schedule.

Commit `3f888927` retains the completion receipt until the stream is terminal,
within the original deadline. It waits on the condition during GPU work and
uses bounded retirement checks after notification. Timeout preserves ownership
and reuses the pending receipt rather than appending callbacks. The retained real-CUDA regression then passed 200,000 waits across four
concurrent processes: zero false successes, all four exited zero and all PIDs
were absent. Its binary SHA256 is
`2139eff5fc2458dae58b6bd698c8ef7f7c3e70b60cc0e283ccea163941bd8076`,
built from `bd25d2c7`. This is a common completion-helper gate; a new fleet run
is still required before counting four or eight residents as passed.

## Evidence

The compact [fleet receipt](receipts/parallel-residents-2b24873d.json) pins source,
bundle, attempts, output counts, timing boundaries and per-rank receipt hashes.
Full evidence is retained at
`/private/tmp/sparkpipe-pr1082-receipts/shared-2b24873d/` on the review workstation.
The unchanged-helper GPU reproduction is
`/private/tmp/sparkpipe-cuda-receipt-probe-2b24873d-four.log`.
The fresh host receipt is
`/private/tmp/sparkpipe-pr1082-receipts/host-2b24873d/reliability/results.json`.

Partial-pool eviction, common lane assignment across different model families,
and nonidentity physical rank maps require their own subsequent qualification.
The first two passing rows above do not imply those cases passed.
