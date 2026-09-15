# T1-QMAX run receipt — 2026-09-15/16 night wave (re-stamp + engine repair)

Lane lane/t1-qmax @ 3217be3 (+ tooling commits), worktree /Users/mac/t1qmaxn.
Mesh lease ACTIVE T1-QMAX window 16:21Z-~19:00Z; CEPH lease held from the
prior session (tail reads only; patch pass + release notes appended).

## VERDICT: T1 FAIL — execute step 0, routed-expert lease NOT_FOUND

The load path is REPAIRED: every placed pack now loads, validates,
attaches to the shared fleet weightsd, and the module initializes and
opens the TP16 collective on all 16 ranks. The decode still cannot run:
ranks whose resident expert window intersects the step-0 routing fail the
weightsd lease acquire (daemon NOT_FOUND), while ranks with an empty
window skip cleanly and time out on the collective. No tokens produced.
No tolerance touched. Phase 2 (measured B1 tok/s) correctly not attempted
- it is gated on T1.

## What this wave delivered (all MEASURED, receipts in this directory)

1. Re-stamp pipeline completed across the fleet. 16/16 placed packs now
   carry the codec-contract layout (weight_format 8 NVFP4_PACKED, group
   16, payload elements/2, scale elements/16 + resident*8 f32 tails):
   - ranks 0,1,2,3,4,6: warm rebuilds from the prior session
     (tool=qwen38_stagepack.py in each receipt)
   - ranks 5,8,9,7,a-f: in-place codec-contract patch by
     tools/qwen38max_patch_pack.py (tool=qwen38max_patch_pack.py,
     patched_from_output_sha256 records the superseded digest)
   - rank0 byte-identity re-proven by this agent:
     rank0-patch-identity-proof.json - sha256(patched_rank0.sp) ==
     placed warm-rebuild receipt sha c911156b...
   - uniform fleet verify 16/16 PASS with recomputed file digests
     (verify-rank{0..15}.log), packs relocked chattr +i after swap.
2. Pack-name hex bug fixed (ranka..rankf for ranks 10-15): wave 5 had
   silently lost ranks 10-15 to FileNotFoundError at the receipt read.
3. Seven engine defects fixed and committed (each commit cites evidence):
   - 027dff7 ValidateEntry accepts the stamped NVFP4_PACKED code on
     routed-expert entries
   - 3c99285 ValidateEntry expects the rank-sharded slab shape (the true
     wave-5 blocker: full 512-expert shape vs the 32-expert rank shard,
     ERBSITE module.c:410 in wave 5)
   - 59d7a3b + 484e4a7 sharded-shape expectations for every TP-sharded
     kind via kind masks (rows: routed slabs, shared gate/up, attn
     query/key/value, GDN qkv/gate; cols: shared down, GDN output, attn
     output; beta/decay/norms/router/conv replicated)
   - d9b3c17 grouped-scalar expert launch consumes the bound rank-slab
     view as-is (the copied tp_rank re-slice pushed ranks 1-15 past
     their slab)
   - 2b7ada8 shared expert consumes the rank shard with a TP allreduce
   - 510d6ab MTP completeness tracks the compiled MTP count (archive
     builds with MTP=0 against MTP-stripped packs)
   - 980afb1 acquire routed experts from the resident window only
     (rebased local prefix; zeroed expert output buffer; the hidden
     allreduce sums partials)
   - 3217be3 synchronize the slot stream before reading the routing
     prefix (the D2H used the default stream; keys were derived from
     stale offsets)
4. Attach evidence (single-rank, rank0): load ok (1682/1682 entries
   validated), lazy attach to the shared weightsd ok, tp_collective_open
   ok, MoE stage launches ok, run ends ONLY at the collective awaiting
   absent peers (attach3.log via attach runs).
5. Daemon path proven independently: weightd_execute_probe against the
   shared daemons on spark0 and spark5 acquires 8-key leases on the
   patched packs (EXECUTE-PROBE lines; run on rank0 and rank5 packs with
   correct per-rank shas).

## The remaining blocker (exact)

Wave 2/3: ranks with a nonempty resident-window intersection fail
SparkWeightdClientAcquire with daemon NOT_FOUND (ERRSITE
spark_weightd_map.c:393 status=3); rank0 (empty window at step 0) skips
the lease and times out on the collective (status=19). The probe
acquires identical-class keys (layer 0, resident experts) on the same
daemons with the same identity, through the same
SparkWeightdLazyPackCreate/CreateChecked code path. Un-tested
difference: the probe derives its keys from the loaded manifest groups;
the module derives them from routing. Next test: an instrumented wave
printing (layer, count, keys, raw status) per rank - the debug harness
for this exists (tree-patched build, not committed) - plus a
client/daemon ManifestFind parity check on the arena the module's
connection actually attached to. Wave 3 also hit MESH-REGISTER-FAIL
(tp_device_collective.c:960 status=4, NIC memory-band registration) on
relaunch - infrastructure state needs the night's leftover registrations
to drain before further waves.

## Evidence files

- patch-rank{1..15}.log (+ .fail1/.fail2 preserved first attempts):
  per-rank precondition sha, patch json, audit PASS, verify PASS, swap,
  chattr relock
- verify-rank{0..15}.log: uniform fleet verify, 16/16 verdict=PASS
- rank0-patch-identity-proof.json: byte-identity re-proof
- rank{0..15}.log: wave logs (workstation side)
- per-host /tmp/t1qmax/harness.log: the wave ERBSITE chains (wave 2:
  map.c:393 status=3 / collective 19; wave 3: MESH-REGISTER-FAIL)
- attach{,2,3,4}.log on spark0: single-rank attach progression
- fix commits 027dff7..3217be3 (8 commits) on lane/t1-qmax

## Honesty notes

- No tolerance was loosened; the compare stage was never reached.
- No fabricated tok/s: the only performance number tonight is the
  analytic ceiling 22.0 tok/s with NO measured companion (gated on T1).
- The fixture, its manifest sha, and the checkpoint identity chain are
  untouched from the prior wave's audit.
