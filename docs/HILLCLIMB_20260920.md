# ALLREDUCE HILLCLIMB — 2026-09-20

## GOAL (set 09-20, operator)
**Allreduce ≤ 100µs/round MEASURED on the 16-spark fleet** (91-round chain, status=0, checksums green), starting from 1.79ms/round (the healed fleet's cold-class eager number). Every intermediate number graded MEASURED (wire-math floor check before grading); every step committed + PR'd.

**Ladder:** 1790µs → <1000 → <500 → <200 → ≤100µs. Current: **1334µs/round** (S1, MEASURED: 121.41ms/91r; second chain 153.7ms/91r=1690µs — steady ~1.3-1.7ms). Chain totals now 432-518ms WARM-CLASS (weightd's expert cache survives engine recycles — fresh engines serve warm immediately; the cold cost is once per weightd boot). In-process harness floor: 1530µs.

## Step log (append every step: what / number / verdict)

### S1 — DONE (09-20 ~13:5x): per-op cudaStreamSynchronize → pinned-cell poll (660b6b1)
MEASURED: allreduce 121.4ms/91r = **1334µs/round** (from 1910-2700µs; −30-50%). Chains total 432-518ms warm-class = ~2-2.3 tok/s through the full engine. Zero MESH-CELL-TIMEOUTs (the poll's guards work). NOTE: chains are warm by default now (weightd caches experts across engine recycles — only the FIRST engine per weightd boot pays cold).
NEXT S2 candidates (measured against the 1.33ms round): the host peer-wait spin (the remaining serialization — each op waits 15 tails sequentially-polled), the publish-ack wait (waits the PREVIOUS op's ship before publishing — adds a full ship latency per op), and the per-op kernel launches (3 launches + 1 D2H per op). Then S3 (graph).

### PR-BY-LAW DEPLOY (09-20 ~19:00): PR #1064 = THE deployable line
Merged lane/fleet-resilience (23 agent commits) into the PR + all wedge/climb fixes; head 4f476aa; DEPLOYED SHAS (in the PR body, audit-verified 16/16): weightd eec4938201602fe4, agent 600ed95c9777eb1f, driver/adapter via the glm53 MANIFEST, x86 api+adapter from the same head. Deploy ONLY from this PR now.

### CURRENT OPEN (as of the PR deploy verification): a NEW wedge signature
rounds=0 again, but DIFFERENT from the healed one: zero CKEY-CELL-TIMEOUTS (key exchange fine); each rank STABLY misses ONE DIFFERENT peer across retries (r0→12, r2→13,14, r4→15, r6→14, r7→15, r8→14, r9→12, rc→14, re→15, rf→1; ranks 1,5,a,b,d never spin). Deterministic per-poll single-peer miss — NOT the stale-record black hole (records 16/16, wiring 30 entries, cellTO=0). Next instruments: (a) slotdump a missing pair during the spin — is the peer's tail landing in a DIFFERENT slot than polled (parity disagreement for specific pairs)? (b) check whether the missing-peer ranks are even publishing (their engines' chain state); (c) the ordinal parity: pub=34359738369=8<<32|1 → ordinal bits — the merged branch changed SparkTpChainOrdinal/parity; verify publisher parity == poller parity per pair.

### WEDGE HUNT (09-20 ~20:00): mesh exchange FULLY EXONERATED — the ships exist at the right key and still don't land
Post-settle state (weightds stable, one generation): records self-healed + current (spark0 rec mtime == weightd start), wiring boot-current BOTH directions (sparka->peer0 wire boot == spark0's own record boot), CKEY epochs CONVERGED (all ranks adopt epoch 29, all spin at key 28 — the divergence theory dead), cells fine. THE FACT: sparka's relay SHIPS key-28 publishes (WD-SHIP idx=26 seq=28<<32|1...) but rank0's receive slots for peers 10/11/14 (slots 160/161, 176/177, 224/225) are ALL-ZERO — the RDMA writes never land, silently (no CQ errors, no wire fails). Only SOME pairs fail (rank0 misses 10,11,14; other ranks miss their own subsets).
NEXT INSTRUMENT (first move next tick): a one-line diagnostic in the weightd ship loop printing per-peer (peer, addr, rkey) for the FIRST ship of each pass — compare against the peer's CURRENT record recv_addr (the qp_snapshot may hold a stale addr/rkey for specific pairs even after WD-WIRED: the transition logs the wire but maybe a failed RTU path leaves the old qp_info, or the snapshot copy races the rewire). Also check ibv post errors on those specific QPs.

### WEDGE CONVICTED (09-20 ~21:15): the chain keys DRIFT across ranks on independent retries — kimi's partial-cohort verdict was RIGHT
Complete evidence chain: (1) ship addrs verified byte-exact (sparka->peer0 addr=ffde13ffe000 rkey=1836032 == spark0's current record — the qp_snapshot suspicion DEAD); (2) the mesh delivers EVERYTHING: a 12-peer slot census on spark0 shows tails LANDED at every peer's parity slot, all = key 13 (55834574849); (3) but rank0 spins waiting at key 11/12 missing peers 1,3,15 — the peers' keys and rank0's key DRIFTED because each rank's 30s timeout advances ITS OWN chain key independently (CKEY-ADOPT converges epochs only at chain START; retries self-increment in between); (4) CORRECTNESS HAZARD exposed: a peer AHEAD of the waiter passes the >= check with WRONG-CHAIN data — the spin survivors are silent wrong-data accepts waiting for laggards.
THE FIX (next code step, kimi's gate + the deeper invariant): (a) the dispatch gate — no chain start unless all 16 ranks admitted (kills partial cohorts); (b) retries must RE-ADOPT rank0's current broadcast epoch (not self-increment) so keys re-converge after any partial attempt; (c) the wait compare should be == exact-key (parity-slot freshness) not >=, making cross-key data a loud rejection instead of a silent accept.

### FIX-C DEPLOYED + NEW PROXIMATE CAUSE (09-20 ~21:4x)
Fix-C (1ce99fb, marker V6-EXACTKEY) deployed fleet-wide (driver 670e41a3): the wait now demands tail_key == expected_key — VERIFIED WORKING (rank0's peer-11 slot holds key-9 data and is now correctly REJECTED while waiting key 10; the wrong-chain silent accept is dead).
THE WEDGE'S PROXIMATE CAUSE FOUND: the missing-peer ranks' ENGINES HAVE NO MESH MAPPING — spark7's engine: 0 spark-mesh fds, zero mesh lines in its log, engine started 6s after its weightd, yet its chain machine runs (stage prints) while its relay sees NOTHING (23 WD-SEENs total). An engine that booted while its weightd was mid-init landed in a no-mesh state and chains still 'run' — publishing nowhere. The boot race: engine start (weightd+6s) < weightd mesh-ready (weightd+76s wiring!) — the engine's lazy attach hit a weightd whose mesh wasn't wired, failed, and never retried.
NEXT (first move): the engine's lazy-attach must RETRY until the weightd's mesh is wired (the .ready marker exists as the gate) — or the agent must not start engines until /tmp/weightd-mesh-fleet/.ready exists (it gates on the marker already but the marker is written by the weightd at wire time — verify the ordering; the engine at weightd+6s vs wiring at +76s means the marker appeared AFTER engine boot → the engine missed it and its retry must poll the marker or re-attach on EOPNOTSUPP/attach failure).

### FIX-D DEPLOYED (e0266e0, driver cfd67273): attach-without-mesh-fd = BUSY
The no-mesh-engine class is dead: an attach reporting mesh wired but carrying no fd now fails BUSY and the engine's 600x1s retry loop re-attaches (all engines verified holding the memfd VMA — note: the fd is CLOSED after mmap; check /proc/E/maps for 'spark-mesh', NOT fds — my earlier '0 fds' diagnosis on spark7 was an instrument error, though sparkf's engine genuinely lacked the map at first sight).
STATE AT TICK END (recorded for next tick): engines 16/16 ready with mesh maps, weightds stable, records fresh — but the API is in connection churn again: set-failure status=4 stage=0/1 cycling, engine sees 797 peer-eof reconnects, ZERO submissions reach the module (no ADMIT prints), RANK-RESULT-FAIL rank=0/6/13 status=15 — the request dies before chains. This is the S0-era churn SHAPE but with a new gate: the first prepare rejection comes as BUSY from ranks 0/6/13 BEFORE any module admission print — next tick starts at the engine's submission entry path (what rejects BUSY pre-module: the resident-slot claim at boot state? the route reserve?) — instrument the residentd submission handler's early returns.

### TICK 09-20 ~21:15: churn cleared on fresh API; the wedge narrowed to the TWIN-WEIGHTD generation fight
Fresh API connect + probe: submissions FLOW again (DECISION prints, chains run) — the churn was the stale API process. The surviving wedge signature: rank0 spins missing {11,12,13,15}; rank11 (sparkb) shows CKEY-CELL-TIMEOUT cell=774726319 (its base cell reads 0 — rank0's broadcast doesn't reach it). ROOT SHAPE FOUND ON SPARKB: TWO weightd generations fought — the running one started 20:34 (boot ...060, recv_addr e7399fffe000, its record is CURRENT), but rank0 WIRED peer11 at 20:36 (boot ...205, addr ea2abbffe000 — a generation that no longer exists; ships to its dead MR vanish silently = the black-hole class). rank0's local mesh-b.rec (fetched 21:02) carries the 20:34 gen — TryWire SHOULD rewire back (record.boot != wired_boot) — VERIFY the newest WD-WIRED peer=11 line and whether TryWire rewires DOWNWARD (it may refuse: most code only moves forward). NEXT: (1) check TryWire's rewire-down path (stale-record vs newer wired boot — likely the missing case); (2) the twin-weightd fight itself: why did a 20:36 weightd start while the 20:34 one lived (agent watchdog vs my manual restart racing — the janitor then killed the 20:36 one); (3) after both fixed: full-cohort chain + the S2 measurement.

### FIX-E DEPLOYED (e9b64c8): the janitor's singleton path was WRONG — the 30-min churn machine dead
The janitor checked /tmp/spark_weightd.singleton but the weightd holds <socket>.singleton = /tmp/spark_weightd.SOCK.singleton (verified live: the running weightd's fd 3). The holder lookup ALWAYS returned empty → EVERY weightd older than 30min was killed as 'not the singleton holder' → agent restart → engine recycle → mixed generations → wedge → repeat. This explains sparkb's twin generations (20:34 live, 20:36 ghost) and possibly days of 'stable-then-shifting' missing sets. Deployed fleet-wide (direct agent cp — NOTE: must also flow through the core release next publish).
POST-FIX-E STATE: no more weightd churn, but the wedge PERSISTS with shifting per-rank missing sets ({6,10,11,15} on r0, {6,11,15} r1, {4,7,8} r5...) and ZERO cell timeouts — the shape is now PURE key-drift/partial-cohort (fix-C made all cross-key encounters loud). THE REMAINING FIX = kimi's dispatch gate (no chain start unless all 16 ranks admitted the same submission) + the retry re-adopt. THAT is the next tick's code step — everything else is exonerated.

### TICK 09-20 ~22:30 (latch-weightd fleet): the wedge COLLAPSED to remaining=1
First probe on the latch fleet: chains run, and rank0's spin now shows `remaining=1 missing=2` — from 3-4 missing peers to ONE. The stabilizations (fix-E janitor fix + the latch weightds ending a week of generation churn) let almost the whole cohort join each chain. The last missing peer is the key-drift/partial-cohort tail — THE DISPATCH GATE (no chain start unless all 16 admitted) is the single remaining named fix; everything else is exonerated or fixed. Ladder: dispatch gate → S2.5 device-resident slots → S3 graph.

### DISPATCH-GATE / KEY-REAdopt IMPLEMENTATION PLAN (ready to execute — written for mechanical next-tick execution)
WHERE (from the code already read): ring/transport/tp_device_collective.c, SparkTpDeviceCollectiveChainKey (lines ~425-510).
FACT from the read: the pipeline commit path ALREADY gates on all-16 (ResolveAdmission commits only when transaction->status==OK, and any rank failure poisons it) — the partial cohort forms BELOW the pipeline, at the CHAIN-KEY layer: rank0 self-increments epoch per retry (line ~438: `epoch = cell_epoch + 1` after `if (chain_epoch > cell_epoch) cell_epoch = chain_epoch`), and peers adopt whatever rank0 LAST broadcast — a peer retrying between rank0's broadcasts reuses a STALE consumed_cell (`cell == consumed_cell` blocks re-adopt at line ~486) and its own key drifts.
THE TWO CHANGES:
1. KEY RE-ADOPT ON RETRY (the root): a non-rank0 rank whose chain FAILED must clear `implementation->consumed_cell` and re-read the base cell (waiting for rank0's CURRENT epoch) BEFORE its next publish — i.e., in the chain-fail path (or at ChainKey entry when a failure marker is set), set consumed_cell=0 so the adopt loop actually re-adopts. One variable reset + one call-site.
2. WAIT-KEY EQUALITY (defense): the peer-pass compare (already exact-key since fix-C) plus the missing-peer diagnostic should print the PEER's landed tail key (slotdump inline) so a key mismatch is named in one log line.
VERIFY AFTER: full-cohort chain (91 rounds, status=0), then re-measure allreduce_ms/rounds vs S1=1334µs.

### TICK 09-20 ~23:1x: API connection-churn shape returned (engine gen=1241, zero chains) — next tick starts HERE
The engine recycles connections (gen=1241) with no submissions landing — the same churn shape as the stale-API era (S0-fix note: a fresh API restart cleared it then). NEXT TICK ORDER: (1) restart g53-api, (2) confirm submissions flow (DECISION prints), (3) execute the DISPATCH-GATE/RE-ADOPT PLAN below (it is written for mechanical execution), (4) measure. S1=1334µs/round stands.

### TICK 09-20 ~23:4x: FULL CHAIN ON THE LATCH FLEET — status=0, 91/91 ROUNDS
API restart cleared the churn (as predicted from the S0 era). First full chain on the latch-weightd fleet: `CHAIN-TIME status=0 total_ms=457683 rounds=91 allreduce_ms=183005` = 2011µs/round COLD-interleaved (fresh weightd era, expert loads). The serving path WORKS end-to-end; next chains run warm (~1334µs/round S1 class or better). REMAINING LADDER: (1) the re-adopt/dispatch-gate plan (written above) for the last-peer reliability, (2) S2.5 device-resident slots, (3) S3 graph.

### TICK 09-21 ~00:0x: TWO CONSECUTIVE FULL CHAINS, zero spins — serving is STABLE
Chain 1: status=0, 91/91r, 2011µs/round. Chain 2: status=0, 91/91r, 1924µs/round (175s allreduce). Both complete, ZERO MESH-SPINs — the last-peer drift did NOT bite these two. Both are cold-interleaved class because each decode token routes to new experts (the per-token-miss law, 09-17) — the allreduce host-protocol cost is S1's 1334µs class; the rest is expert loading. The canary's 900s client timeout vs 2×~450s chains = the TimeoutError (client patience, not a failure).
STATE: the fleet reliably serves full chains. LADDER unchanged: re-adopt fix (reliability insurance) → S2.5 device-resident slots → S3 graph (the ≤100µs route; the expert-load cost is the ORTHOGONAL lever = the bulk pool import, queued from the 09-17 items).

### FIX-F DEPLOYED+VERIFIED (486887f, driver feaa1ab1): key re-adopt on retry
ChainRetire now clears consumed_cell — a failed chain's retry re-adopts rank0's CURRENT epoch instead of reusing a stale key (the last-peer drift root). VERIFIED: full chain on the fix-F fleet (status=0, 91/91r, zero spins, 2713µs/round cold-interleaved). The wedge-class fixes are COMPLETE: B (stale records), C (exact-key waits), D (no-mesh attach), E (janitor), latch (idempotent weightd), F (key re-adopt).
CLIMB RESUMES NEXT TICK: S2.5 device-resident mesh slots (the flags are host memfd — the poll cost) → S3 graph (≤100µs). Orthogonal lever: bulk pool import for the expert-load cost.

### Step 0 — BASELINE TAKEN (09-20 ~13:00)
Eager fleet rounds: 1.91-2.70ms/round (cold-interleaved; every decode token routes to new experts → chains stay cold-class — the per-token-miss law). The in-process harness says the host protocol itself is ~1.53ms — so S1/S2 (host ceremony) can at best reach ~1.5ms-class; **the ≤100µs target requires S3 (in-graph device waits)**. Two blockers found+fixed en route: (a) published_host_cell NULL in eager-only mode (lazy alloc, 6c1c760); (b) the janitor kills manually-started weightds ("not the singleton holder") → engines orphan on the corpse socket → WEIGHTD-DEAD mid-wait — LAW: NEVER manually start weightd; agent-managed only (deploy the binary, let the agent's watchdog restart it).
#### The chain-3 mid-chain IO (status=4 at round 47): likely the same janitor/weightd churn class — verify the weightd etime vs the engine's attach before debugging deeper.

### OLD step-0 note (superseded)
The probe requests never reach chains: the API cycles connections (`pipeline set-failure status=4 stage=14` repeating, session fingerprint +67/+136 per cycle). ROOT (named, one look): rank 14 (sparke) fails some submission with IO; the pipeline's CONTINUATION path still calls SetFailure on IO (`model_pipeline_client.c` RankResult: `if (transaction->continued != 0u) SparkModelPipelineClientSetFailure(...)`) — the ONLY mass-close path left for IO. FIX = same treatment as Prepare: IO on a continuation = backpressure, no mass-close (one-line guard). THEN step 0 (warm baseline) proceeds.
The 1.79ms came from the cold chain (expert loads interleaved). Warm chains (experts cached) ran 323-455ms/91r ≈ 3.5-5ms... no — the earlier era's warm chains measured 323-455ms TOTAL with allreduce_ms ~61-155ms → 0.7-1.7ms/round. Measure the current warm number cleanly: keep ONE request flowing after the cold chain, read the next CHAIN-TIME's allreduce_ms/rounds.

## POLL/MEM-SATURATION PLAN (operator-supplied analysis, reconciled to our code 09-20)
The analysis's two failure modes map EXACTLY onto our measurements:
- **Mode 1 CONFIRMED — our flags are host memory.** weightd's mesh buffer is a memfd (mmap host pages, ibv_reg_mr on it — node/weightd_mesh.c:445-477); the engine mmaps the same fd into GPU VA. Every MeshWaitKernel poll of a slot tail and every host spin poll crosses the coherent fabric to system memory — THIS is why unpaced polling regressed 4x (the __nanosleep(200) lesson) and why the host spins needed yield pacing: the polled line can't stay L2-resident, and polling + peer RDMA writes fight over the same coherence path.
- **Mode 2 CONFIRMED — our waits are already single-thread** (<<<1,32>>, thread 0 only) — the pacing, not the thread count, was the issue; with the flag host-resident the pacing is load-bearing.
- **The fix for mode 1 = move the receive slots to DEVICE memory and point peers' RDMA at it** (GPUDirect): allocate the mesh region with cuMemCreate/device alloc in weightd, register THAT with ibv, export/import to engines via VMM fd (the machinery already exists — the expert-chunk imports use exactly this). Then the poll line is L2-resident: single-thread acquire-load polling becomes ~free (no pacing needed — consistent with the no-timing-choreography law), and the payload lands L2-hot next to the flag (flag = the slot tail = last 8 bytes of the same RDMA write — our layout ALREADY does this).
- **cp.async.bulk / 128-bit LDG ingest**: our combine kernels already read the slots via 128-bit loads; the TMA global→shared hop is an optimization for the combine path once the buffer is device-resident.
- **Host interrupt path**: rejected 10-30µs+jitter — matches our fast-fail law; interrupts only for teardown/errors. No action.
LADDER INSERT (after the dispatch gate, before/with S3): **S2.5 = device-resident mesh slots** (the single biggest poll-cost fix; also removes the C2C fight between polls and RDMA writes). Order: dispatch gate (correctness) → S2.5 (this) → S3 graph (persistent in-graph waits become trivially cheap once slots are L2-resident).

## LATCH WEIGHTD (2f46cc0, deployed fleet-wide 09-20 ~22:00): idempotent by PORT, not files (operator design)
The operator's ruling: files can be deleted out from under a live holder (the agent's own `rm -f` was the demonstrated twin mechanism); a port bind cannot. weightd now binds loopback:61900 (override --latch-port / SPARK_WEIGHTD_LATCH_PORT) BEFORE touching files: bind ok -> own the node, unlink stale socket, serve; EADDRINUSE+connect-ok -> 'already serving', exit 0 (start is IDEMPOTENT — verified live: duplicate start on spark0 printed 'an instance already serves latch port 61900; nothing to do' and exited 0); EADDRINUSE+refused -> find holder via /proc/net/tcp inode->fd scan, SIGKILL, retry 12s, serve. The flock singleton DELETED; the janitor's weightd-kill rule DELETED (operator: causing more problems than it fixes); the agent no longer rms the unix socket. CAUTION: duplicate foreground starts TRUNCATE ~/weightd.log (they redirect before exiting) — cosmetic; the serving instance's 'already serves' line in a log tail means exactly one instance is running (pgrep self-match inflated earlier counts — use `pgrep -af 'sparkpipe_weightd --socket'`).

## The known cost structure (from all prior analysis — where the time goes)
1. **Per-op host ceremony (eager)**: CopyDown kernel + publish kernel + the pinned-cell readback + `cudaStreamSynchronize` (a FULL stream sync per op — added for correctness; ~50-200µs each) + host spin over 15 peers' tails + the combine launch. The stream sync is the first thing to overlap (readback can poll the pinned cell instead of syncing).
2. **The serialized 15-relay propagation**: each rank's publish ships via ITS weightd relay scanning doorbells; peers' tails land after scan+RDMA (~10-15µs posts + wire). Lock-free doorbell scan is live (tight loop).
3. **The in-graph path = the 100µs class**: the CUDA graph replays publish/wait/combine with DEVICE-side waits — no host per op. BLOCKED on the spark3 illegal-access bisection (09-18 handoff; 1-op graph clean, full graph faults). The graph path is the structural route to ≤100µs; the eager path's floor is the host ceremony (~0.7-1ms).

## Ladder plan (revised as evidence lands)
- S1: kill the per-op cudaStreamSynchronize (poll the pinned cell; overlap the readback with the ship latency) → eager −0.1-0.3ms?
- S2: batch the host ceremony (publish N ops ahead? not possible in eager single-chain semantics) → limited; skip if S1 lands.
- S3: fix the graph path (spark3 bisection ladder N=2,8,24,45,91) → replays complete → in-graph rounds ~5-50µs/round class + layer compute.
- S4: in-graph round tuning (the PoC's parity-slot + zero-seed structure is live; the waits' spin + the relay's scan cadence).

## Wedge playbook (if rounds=0 / spins)
1. Check every weightd has `--mesh-dir /tmp/weightd-mesh-fleet` (manual restarts drop it — the #1 regression source).
2. Check /tmp/weightd-mesh-fleet/*.rec all fresh (the self-heal fix 2086fbd rewrites within 1s; if stale → weightd predates the fix or crashed).
3. Check WD-WIRED boots in ~/weightd.log vs peers' actual weightd starts (two eras = stale wiring).
4. Engines: recycle via the agent ONLY (kill -9 residentd + systemctl --user restart fleet-agent.service); never kill weightsd.
5. The probe: single-session API — ONE request at a time, 900s timeout (cold chain ~520s).

## Fleet/deploy facts (current)
- weightd 2086fbd (=PR #1064 self-heal) on all 16, --mesh-dir correct, records fresh.
- Engines: whatever the agent runs (the graph env is OFF by default; SPARK_GLM5_NEXT_GRAPH_PATH=1 per-unit enables it — currently check before graph steps).
- Deploy: build on sparkf ~/sparkpipe-build-main (rsync from the fix/hillclimb worktree, full source set incl. inference/), module publish + model_compile + release rsync + MANIFEST regen; the CORE release flow has a delivery bug (nodes installed a stale binary) — for weightd use direct scp+restart (documented argv), for the DRIVER use the glm53flash release path (that one works).
- The probe loop: /tmp/canary2.py on rtx5090 (127.0.0.1:8433).

## 2026-09-21 night session — S-bulk LIVE, rulings landed, serving GREEN

### The S-bulk root cause (the missing producer)
`WD-MAP-POOL-BULK` never printed because `SparkWeightdPremapPool` was a SKIP
stub — `pool_export_handle` was READ in three places but WRITTEN nowhere in the
tree. Same class as c152edd: the consumer shipped, the producer never existed.
d14818f implements the real producer + 5 companion fixes (EvictGroup BUSY on
pooled arenas so pooled chunks never unmap/epoch-move; whole-span teardown with
the pool handle released once; pool fd staged on BOTH create and re-attach;
flag-driven fd interpretation on the client — the positional read mmap'd the
pool fd as the mesh when mesh wasn't ready, caught by check_pooled_attach;
lease-export short-circuit to an empty valid batch). Verified 16/16: `pool
single-alloc chunks=10351 span=21707620352` + `WD-MAP-POOL-BULK` on every node.

### The mesh QP wedge class (fixed + fuzzed)
Mass weightd recycles strangle QPs: RC-retry exhaustion sends a QP to ERROR,
and TryWire only re-transitioned peers whose RECORD changed — a QP that died
after its last rewire stayed dead forever. Measured live: +1368 flush errors
per +2048 completions, chains wedging missing={4,7,9,11,14} across reporters.
28b5a59: TryWire queries every send/recv QP state each pass and force-repairs
anything not in RTS (WD-QP-REPAIR). Deployed: spark9 repaired all 15 peers,
error counters collapsed to single digits.

### The ordinal wall (the "180 million retries" bug)
The submission counter crossed the chain-id capacity of the ordinal math
(TP_CHAIN_OPERATIONS = 106×65536 with the compile-max row count; wall at
~162.07M ids) → every submit capacity_exceeded forever. THREE consumption
paths fixed: (1) dispatch failures roll back next_submission_id (retry storms
burn nothing); (2) the API seed file is session-conditional with a watermark
(same engine session → continue at watermark+10001, new session → rebase to
1M; the naive fixed rebase tripped the pipeline's monotonic gate and killed
in-flight requests status=4 — 3420575); (3) SparkTpChainIdCapacity in the
fuzzer pins the wall and both policy branches.

### Operator rulings (13c1113)
- POOL: 8GiB magic default deleted; attach declares the full pack; weightd
  pools whenever the span fits the device budget (90GB packs pool whole on the
  110GB budget); oversized packs degrade LOUDLY (WD-POOL-CLAMP) to per-chunk.
- MESH REGION: 16GiB → 128MB per daemon (SLOTS_PER_RANK 16→2 — the parity/
  credit contract; SLOT_ROWS=8; geometry printed at MR registration).
- FUZZER FIRST: the fleet bug classes above are replicated and pinned in
  test_tp_allreduce_fuzz.c (FuzzEdgeOrdinal / IdPolicy / RewirePolicy /
  Geometry). 237-check basic + 2328-check 200-round fuzz green.

### Verdict (all MEASURED, r3 generation, 2026-09-21 18:32 UTC)
- Canary http=200, 8/8 tokens, DETERMINISTIC output ×4 runs
  ([3764,10,4999,1725,15,98886,18,100461]).
- Warm steady state: 17.1-17.3s per 8-token request (~2.1s/token wall clock).
- Warm full chain: status=0, 91 rounds, 120-140ms total, allreduce 51-64ms
  (0.56-0.70ms/round; the in-chain pure-mesh component measured 136µs/round).
- Cold chain (first touch, experts loading from pack): 247-277s ONE TIME per
  engine boot; pooled arenas never evict so warm persists.
- S1 was 1334µs/round → warm serving now 561-700µs/round with the pure-mesh
  component at ~136µs. The 100µs goal needs S3 (in-graph path).

### Deploy facts (r3 = 3420575)
- weightd sha 6f0ea32528dc on 16/16 (mesh 128MB print verified), engines on
  the r3 driver, API = r3 x86 build on rtx5090 (session-conditional seeding).
- GOTCHA: publish_local.sh's adapter path is stale
  (libglm5_next_resident_decode_stage_serving_adapter_fp8.so vs actual
  libglm5_next_serving_adapter_fp8.so) — publish manually or fix the script.
- GOTCHA: the release bin/sparkpipe_model_api must be the x86 build (build on
  the rtx host ~/sparkpipe-build-main); a sparkf aarch64 api binary = 203/EXEC.
- GOTCHA: engines re-cold after every weightd bounce (pool arena dies with the
  process); the first canary after any bounce pays 2×~270s (two slots' loads).

### Next
1. The same-request-twice cache test (KV skip + expert reuse) — now unblocked.
2. S2.5 device-resident mesh slots (the flags are host memfd; poll cost).
3. S3 the graph path (spark3 illegal-access bisection) — the 100µs class.
4. Warm throughput ladder (batch rows > 1: prefill chains at 8 rows/slot).
