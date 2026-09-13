# D-1 weightd module-execute stability — 2026-09-13

Lane: `lane/d1-weightd-stability`. Clone: `/Users/mac/d1-weightd` (Mac, canonical),
`spark0:~/d1-weightd` (main build) and `spark0:~/d1-hill1` (hill1 build).
Dispatch scope: prove whether `origin/glm53flash-hill1` fixes the recorded
module-execute failure class, honestly, on spark0 only.

## 1. Failure signatures (journals, spark2 + spark3, 09-12 00:00–02:00 KST)

Sources: `fleet_node_agent.sh` user journal, `~/sparkdata/glm53flash.fp8.tp16/residentd.log*`,
`/home/spark3/q38max-*` client logs, `/home/spark3/weightd.log`. Current residentd
logs were reborn 09-14 04:19 (redeploy); the .prev rotation is startup-only, so the
window was reconstructed from the journal plus append-only client logs.

- S1 lease/map IO_ERROR under module-execute (spark3, `q38max_exec_timing_noblock.log`
  01:42): `ERRSITE runtime/spark_weightd.c:2484/2571/3352 status=4` then
  `spark_weightd_map.c:340 status=4`. status 4 = `SPARK_STATUS_IO_ERROR`, raised in
  `SparkWeightdClientWriteAll` when `send()` fails with a non-EAGAIN errno — the
  unix-socket peer was gone. Immediately after: `cuda_error site=moe
  error=invalid argument`, `stage_module_common.c:164 status=17`,
  `failure=module_execute rows=8`.
- S2 attach BUSY latch (spark3 `q38max_negA/negB.log` 00:49):
  `spark_weightd_attach.c:46 status=15` (= `SPARK_STATUS_BUSY`) repeating, module
  initialize fails; later `pack_geometry_mismatch status=-5`.
- S3 daemon restart churn both hosts: journal shows residentd/weightd SIGKILLed by
  `fleet_node_agent.sh` at lines 53/106/221/239/293/330/385 through the window;
  at 00:26:06 `core: weightd d0288f41 -> 59548787; deliberate restart` (binary swap)
  kills BOTH weightd and residentd. The spark2 "double-restart" is the agent's
  recycling loop racing itself: 00:41:38 `running residentd none != disk; recycling`
  kill+start, again 00:42:58, again 00:43:08 weightd start.
- S4 phase-2 fixed-slot ring wedge / admission latch: no in-daemon log; the mechanism
  is code-level (see section 3) and now has a repro receipt (section 4).

Root-cause chain for the window: fleet-agent binary-swap/recycle SIGKILLs the
daemons mid-serve (S3) → every connected client's IPC dies with real socket errors
(S1) and attach is refused while the replacement warms (S2) → the q38max stage
module then touches lease VAs it never mapped and fails CUDA execute (S1 tail).
No evidence of a residentd or weightd daemon wedging on its own in this window.

## 2. Client-vs-daemon verdict

- S1/S2/S3 are supervision artifacts plus client-side aggravation, NOT daemon bugs:
  the daemon died because the fleet agent killed it (SIGKILL, deliberate or recycle);
  the stage proceeding to CUDA execute after a failed lease acquire is the same
  client-bug class as the on-record memcpy-on-unmapped-lease-VAs precedent.
  The moe_lease timing rows in the same log (4.57 s for 4 calls) show the client
  path was alive and slow, not wedged, before the daemon swap.
- One genuine daemon-side bug of the era is already fixed on main: #963
  (152cea7, 09-12 20:54 UTC) "residentd: continuation and submission failures stay
  request-scoped" — its message names the old fatal-latch as "the restart loop
  driver". The deployed 09-12 trees predate it.
- S4 (engine-level latch) is real on main and is the class hill1 targets.

## 3. hill1 diff vs main (tip 09d2ebe0, one commit past the dispatched 1808d08)

- a6f1d77 `runtime/model_resident_client.c` +52: `EnsureConnected` — submit/progress
  re-open the endpoint and re-handshake when `connected==0` instead of returning
  `INVALID_ARGUMENT` forever; main fail-stops permanently on first socket error
  (`Progress` → `FailStop` → every later call `INVALID_ARGUMENT`).
- a6f1d77 `runtime/model_batch_engine.c` +34: 240 s inflight deadline fails stuck
  PREFILL/DECODE inflight requests with BUSY instead of wedging the engine.
- 19a6462 eviction-epoch validation + `ChainRetire` epoch bump (seq-reuse class).
- 1808d08 FP32 allreduce through the weightd-mesh transport + deterministic
  zero-init combine order (numerics).
- hill1 does NOT touch `runtime/spark_weightd*.c` — the weightd lease/map IPC and
  the attach path are unchanged. hill1 cannot and does not fix S1/S2/S3.

## 4. Contrast repro (spark0, real daemon + real client, no serving touched)

Rig: new regression test `tests/test_model_resident_reconnect.c` (+ Makefile
target `build/test_model_resident_reconnect`), modeled on
`tests/test_model_resident_end_to_end.c`. It connects to a live `build/sparkpipe_model_residentd`,
runs one full decode execute (prepare → result → commit → completion), SIGKILLs the
daemon, verifies the client fail-stops, starts a fresh daemon on the SAME socket,
then requires the client to recover and complete a second execute (ids continue
monotonically). Public client API only; compiles clean on both branches.

Receipts (spark0, gcc 13.3.0, aarch64, `sudo -n systemd-run --scope -q
-p MemoryMax=4096M -p MemoryHigh=2900M --uid=1000`):

- MAIN (d1c3822): `test_model_resident_reconnect` ABORTS —
  `Assertion 'status == SPARK_STATUS_OK' failed` at the bounded reconnect loop
  (3/3 runs: first run plus 2 sparkcap'd). After daemon replacement the client
  returns INVALID_ARGUMENT forever. WEDGE PROVEN.
- HILL1 (09d2ebe0): RC=0 (3/3 sparkcap'd runs). Client reconnects transparently
  (`view.connected == 1`), submission 502 completes end-to-end on the new daemon,
  daemon exits 0.

VERDICT: hill1's a6f1d77 fixes the admission-latch/wedge class exactly as
claimed, proven by contrast. 19a6462/1808d08 were not independently re-proven
here (see section 5).

### New bug found on BOTH branches (not fixed by hill1)

`tests/test_model_resident_end_to_end` rank-6 case: residentd dies by SIGSEGV at
teardown. gdb on the core: heap corruption inside `cuMemFreeHost` →
`SparkModelResidentdFreeBoundary` (node/model_residentd.c:588) →
`SparkModelResidentdDestroy` ← `main`. Reproduces in isolation (only the rank-6
case), on main AND hill1 identically. Rank 0/12 cases pass. This is a live heap
corruption in the middle-rank boundary path; receipts: `/tmp/d1core.*` on spark0
(backtrace above). Needs its own lane; block-merge impact unknown.

### Build break on main AND hill1 (fixed in this lane)

`make build/sparkpipe_model_residentd` fails `-Werror` on gcc 13.3 (all spark
hosts): `node/model_residentd.c:2304 unused parameter 'runtime'` — #963 deleted
the `SparkModelResidentdFailLocked` call but kept the parameter. CI never noticed:
`cuda13-sm121a-compile.yml` compiles CUDA targets with nvcc and never builds the
residentd host binary. Fix here: drop the parameter, update 2 call sites.
Also relevant: `weightd.log.smoke-prev` shows the weightd-required startup gate
working as designed (status=-16 without supervised daemon + pack sidecar).

## 5. Per-op collective latency

UNMEASURED — out of authorized scope, honestly unproven. hill1's FP32 allreduce
runs through the weightd mesh bands (16-host QP mesh). spark0 alone cannot form
the collective; the only live mesh is the serving deployment whose daemons are
untouchable. No single-host degree-1 number is meaningful against the ~100 µs/op
claim. Recommendation: measure on the next authorized GLM mesh window with
`OPPATH`/timing stderr from hill1's transport before quoting any latency.

## 6. Soak

Final numbers (hill1 build, spark0, every run sparkcap'd
`MemoryMax=4096M MemoryHigh=2900M --uid=1000`):

- Phase 1: 25 consecutive reconnect-rig runs = 50/50 execute cycles green in 24 s.
- 10-minute soak: loop continued to a 600 s wall — 1700 total execute cycles
  green, 0 failed runs, 0 daemon crashes.

## 7. Recommendation

- ADOPT hill1's client changes (a6f1d77 is proven; 19a6462/1808d08 ride along on
  coredev's receipts) — merge call is mgr2's with the operator. This lane's PR
  carries the independent regression test; it is red on main by design and green
  on hill1, so it gates the adoption.
- Do NOT credit hill1 with fixing the 09-12 fleet incident classes S1/S2/S3;
  those are fleet-agent kill/recycle churn plus client-side proceed-after-failed
  lease. The client proceed-on-failed-lease bug (CUDA invalid argument after
  failed map) is still open in main and hill1 and should be a follow-up lane:
  fail the execute fast when lease acquire fails.
- Land the build fix (unused parameter) — every gcc-13 host build of residentd
  is currently broken on main.
- The fleet-agent restart churn itself (binary-swap SIGKILL without draining
  clients) remains the largest real-world availability factor; worth a
  supervision-side lane (graceful stop, or client-visible restart sentinel).

## 8. Serve-gate meaning for qwen38max

The 09-12 q38max exec failures on spark3 were daemon-kill churn, not model-side
instability: with a stable daemon the same logs show the full stage pipeline
initializing and executing (moe_lease slow path aside — 1.14 s per lease call is
its own performance item). A qwen38max serve-gate should require: (a) hill1's
client reconnect or equivalent, (b) no fleet-agent recycling during the gate
window, (c) the soak receipts from this lane's rig as the stability bar.

## 9. Blockers / unproven items

- Per-op collective latency: unmeasured (section 5).
- 19a6462 epoch semantics: code-reviewed only, no independent repro.
- rank-6 boundary SIGSEGV: open on both branches, core + backtrace captured.
- Client proceed-on-failed-lease: open on both branches (S1 tail), follow-up lane.
