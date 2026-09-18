# GLM 5.3 Flash TP16 — graph/mesh handoff (2026-09-18)

Branch under test: `lane/glm53-graph-replay` (tip 35d4e8e). Resilience agent
branch: `lane/fleet-resilience` (tip 92d857b, PR #1036 open). Graph PR #1030.

## What was fixed today (all pushed to lane/glm53-graph-replay)

1. **seq_cell NULL (the illegal-memory-access root cause)** — 35d4e8e.
   `SparkGlm5NextMeshPublishKernel` did `atomicAdd(seq_cell)` with
   seq_cell == NULL: the device sequence cells were only allocated in
   `ArmCapture` (graph mode) but the branch's eager round also uses the
   kernel publish. First eager round atomicked address 0, poisoned the
   CUDA context, and every later failure was a sticky-error ghost.
   Found with compute-sanitizer on spark0 (Invalid __global__ atomic,
   access to 0x0, spark_tp_mesh_kernels.cuh:35). Fix: `EnsureCells` at
   round entry and ArmCapture; seq_cell seeded from round_seq (not 0).
2. **Whole-region mesh registration** — e1a8e76. `PrepareReceiveBf16`
   registered only one 2GB lane of the 16GB mesh region for CUDA and a
   static pointer guard prevented registering the rest; kernel access to
   other lanes was illegal. Now registers `SPARK_WEIGHTD_MESH_REGION_BYTES`.
3. **SumRanksF32 grid-stride** — a34a54f. The fused allreduce combine
   kernel ignored blockIdx.x: every block recomputed the entire output
   (8x redundant at 8KiB payloads, 64x at 64KiB, 1024x at 1MiB).
   External review finding, confirmed and fixed; per-pair accumulation
   order (numerics) unchanged.
4. Diagnostics now in the driver: MESH-SPIN-TIMEOUT names missing peer
   ranks; chainfail prints the CUDA error string; eager round failure
   sites are tagged (MESH-COPYDOWN/PUBLISH/READBACK-FAIL);
   GRAPH-LAUNCH-ERR splits sticky (pre=) vs launch (rc=) errors.
5. **Agent logs unbuffered** (lane/fleet-resilience 92d857b): residentd
   and model_api launch under `stdbuf -o0 -e0`. Before this, engine
   stdout was block-buffered to the log — "engine idle" log reads were
   LIES and sent diagnosis wrong for hours. Trust only unbuffered logs.

## Current state of the fleet

- Eager mode (G5_GRAPH_PATH=0 via drop-in `zzeager.conf` — the agent maps
  G5_GRAPH_PATH to SPARK_GLM5_NEXT_GRAPH_PATH; the unit's own
  `Environment=SPARK_GLM5_NEXT_GRAPH_PATH=1` and `graph.conf` set G5_GRAPH_PATH=1).
- Transport VERIFIED WORKING end-to-end at the byte level: engine kernel
  publishes land in the memfd doorbell (idx = band*16+rank), weightd
  doorbell loop ships them (WD-SHIP), RDMA writes deliver payload+tail to
  peers' memfds (verified rank0→spark1: payload bytes + tail present).
  Wiring records (.rec) fresh and consistent after a clean weightd bounce.
- The first chain's rounds COMPLETE when peers are warm: a chain reached
  stage 4 (REDUCE_MLP) — attention reduce (round 1) passed.
- **Failure that remains**: mesh round times out (30s, MESH-SPIN-TIMEOUT)
  under any peer skew (cold expert loads take up to ~120s; a single slow
  rank kills the round), then the engine EXITS (see below) → crash loop.
- Slot-tail/sequence contract needs redesign: tail = kernel's global
  seq_cell counter; slot parity = per-chain ordinal. Peers compare
  `tail >= my_published` across counters that skew per rank. After churn
  the values interleave wrongly (observed: doorbell seq=3073 but slot tail
  =3072/5120 from other generations). Design fix: tail should be
  `(boot_epoch << 32) | chain_ordinal`, where boot_epoch comes from the
  attached weightd's record boot_ns — a restarted rank is always newer
  than anything in the surviving memfd, so monotonicity is free and no
  fencing/high-water machinery is needed (stale state is recognizable as
  stale, never adoptable). The wait then compares the same quantity on
  all ranks. This is the next real code task.

## The two amplifier bugs (fix these, they turn any hiccup into a fleet outage)

1. **Engine exits on chain IO_ERROR.** Chain timeout → chainfail →
   `route_failed reason=9 (CLIENT_LEASE_DISCONNECT)` → `run=io_error` →
   process exit. The completion handler already fails the *request*
   non-fatally when a route exists (model_residentd.c ~1064); the fatal
   path fires for route==0. The lease disconnect itself follows from the
   teardown. Policy needed: round timeout = fail request, keep process.
2. **weightd restart kills every engine on the node** (lease socket drop
   is fatal per above). WEIGHTSD_BIN exe-sha recycling therefore caused a
   rolling fleet engine massacre today. WEIGHTSD_BIN is PINNED to the
   running build (40067f05 = marker-ship weightd). Do not republish
   WEIGHTSD_BIN unless the weightd source actually changed.

## Operational playbook (verified today)

- Publish: worktree /Users/mac/lane-g53graph; `./tools_local/m` (ships
  curated sources to sparkf:~/sparkpipe-build-958, builds aarch64 driver +
  weightd, publishes to ~/release). Then on sparkf: touch
  ~/release/glm53flash.fp8.tp16/UPDATE; rsync both roots to
  spec@10.10.250.2:~/release/. Agents pull from http://100.123.97.61:8802.
- The 5090 is reached from the Mac via `ssh spec@100.123.97.61`
  (10.10.250.2 only works FROM sparkf). API = g53-api.service on the 5090,
  port 8433. Probe: POST /v1/completions {"prompt_token_ids":[1..8],
  "max_tokens":16,"temperature":0}.
- Engine recycle wedge: agents pull a new driver but a running engine
  stays; sweep with `pkill -9 -f "sparkpipe_model_resident[d]"` on all 16
  (the bracket avoids pkill self-match killing your ssh).
- weightd bounce: `pkill -TERM -f "sparkdata/weight[d]/sparkpipe_weightd"`
  per node; agent restarts it with correct args. This KILLS ENGINES
  (lease drop) — sweep engines too.
- Ground truth past log buffering: the mesh memfd is directly readable:
  `ls /proc/$(pgrep -f weightd)/fd` for the spark-mesh fd; doorbell region
  at 0x400000000 (24-byte entries, idx = band*16+rank: seq, bytes, slot);
  band N base = N * 0x40000000; slot tail = band_base + (rank*16+slot)*4MB
  + 4MB - 8.
- The doorbell loop can be proven alive by planting a fake entry
  (band 15 rank 15 = idx 255) and watching for WD-SHIP.

## Evening update — TAIL CONTRACT WORKS, chains complete end-to-end

Deployed (branch tip 417aa09):
- Tail contract `(chain_epoch<<32)|chain_round` live: publishes fleet-wide
  carry identical tags (verified across 8 nodes: all `(12,1)`).
- The marker-ship weightd was NEVER actually running until tonight — the
  announced WEIGHTSD_BIN predated the marker-ship build, and two "fake
  entry" probe tests only passed because the full-slot recovery path
  masked the missing tail ship (band-15 fake shipped tail via recovery;
  a band-1 fake with no seq gap exposed payload-lands-but-tail-doesn't).
  With weightd f922f476 fleet-wide, tails land and rounds complete.
- residentd is now non-fatal on route-less/late completions (FailLocked
  removed): engines survive round timeouts and weightd lease resets
  ("client_lease_disconnect live_leases reset", process stays up).
- publish pipeline hardened: m rsyncs the full source set with -R and
  --no-times (was: curated scp list that silently drifted + preserved
  mtimes that made make link stale archives; residentd in releases was
  a Sep-12 binary until tonight). publish_local honors SPARK_TREE.
- 5090 api + adapter must be rebuilt from the same commit (x86):
  ~/sparkpipe-build reset to the tip, make build/sparkpipe_model_api,
  make -C modules/glm5_next_resident_decode_stage adapter ... sm_90a,
  cp into ~/glm53flash.fp8.tp16/{bin,lib}, restart g53-api.

Measured: a full 61-layer chain completes — CHAIN-TIME status=0
rounds=91, allreduce 116.6s of 379s; the 379s is first-token cold expert
loading (stage MLP dominates), not the mesh. A warm fleet should be
orders faster.

Remaining blocker: slot/lane lifecycle across failed requests — after a
client gives up mid-chain, later submits hit `submit -> 15` (BUSY):
stale chains hold engine slots. Need chain reclamation on client
disconnect / request retirement.

Then: warm probes (expect fast tokens), re-arm graph, fixture, merges.

## Late-night addendum

- A WARM chain completes a full token in 313ms eager (91 rounds,
  allreduce 63ms -> ~695us/round, 7-10x above the 50-100us target).
- COLD chains cost 283-452s/token: per-layer lazy expert loads dominate
  (stage_ms[3] = 99% of chain time). The expert pool churns per token.
- CKEY base-cell broadcasts from rank 0 do NOT land on peers
  (sparkc waited on cell (10,...) while rank 0 had written (13,...));
  cancel cells appear locally because each rank writes its own on
  chainfail. New diag: CKEY-BCAST-FAIL prints if the MeshBroadcast IPC
  itself fails (deployed in driver 965a... era = tip ba60539).
- The API retry cap (10000 restores at <=200ms) keeps dead requests
  storming the engines for tens of minutes; combined with
  slot_index = request_id % 4 dispatch, four unlucky colliding requests
  pin all four pipeline slots. Engines self-heal in ~30-60s per stale
  chain ONLY if probes stop arriving.
- Engines that outlive a weightd restart keep an ORPHANED mesh mapping
  (old memfd): they publish into a buffer nobody reads. The engine must
  re-attach its mesh mapping on lease reset (or restart). Until then:
  after any weightd restart, restart the engines too.
- Probe discipline that works: full clean bounce (kill all residentd +
  weightd, let agents converge, all 16 listeners), ONE probe with a
  600s+ budget, no parallel probes.

## Open work, in order

1. Tail contract `(epoch<<32)|ordinal` in the publish kernel + wait
   (kills the cross-rank counter-skew deadlock class for good).
2. Non-fatal chain failure (request-level error, engine survives).
3. Re-arm graph (G5_GRAPH_PATH=1 drop-in removal of zzeager.conf),
   3+ sequential probes, then the 176-token fixture (case 0, answer "B").
4. Merge #1036 (fleet-resilience) and #1030 (graph branch).
5. Mesh region right-size: 16GB pinned for 128KB of live B1 data is
   absurd (slot = 128 max rows x 32KB max row = 4MB, x16-deep ring, x16
   ranks, x16 bands). Size from real hidden x batch cap; coordinated
   wire-format change.
6. Fuzz harness: random weightd/residentd kills, measure time-to-serve,
   must never wedge.
