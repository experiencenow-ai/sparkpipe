# WAVE RECEIPT — Q3F-T1 resume night (2026-09-17, dispatch: fix the packs + T1)

## Done (MEASURED)

- Scale-plane re-emission tooling landed (emit/apply + payload-identity
  probes + lock-set discipline; commits b8a74717..da537dc4). tp8 scale
  planes re-emitted on ALL 16 hosts; every rank's probes 432/432 PASS
  against warm (payload identity = the placed fp8 payload bytes are the
  official-fp8 release bytes; asset identity holds).
- spark1 + spark2: FULL chain complete - sidecar regenerated
  (146,883 digest bytes differ vs pre-repair on rank01), receipt/sha
  updated, `qwen4_flash_pack_verify --source-layout fp8-official` PASS
  (byte-exact full scale planes + payload rows vs the split release
  tensors), stamp-check clean, chattr lock-set restored exactly.
- Verifier completed for the fp8-official arm (--source-layout; streamed
  receipt digest replacing a 30.5 GB read_bytes()).
- Fixture window executed (deterministic run1==run2; negative control
  convicted; MANIFEST verified; fixtures committed under
  qualification/t1_reference/qwen4_flash/).
- Reference-engine first exercise: deterministic but garbage generations,
  head top-1 3.4-6.2, near-uniform route weights - reference validity
  unproven; decoder capture bug (final streams in every anchor slot)
  found and fixed; fixture regen queued.
- Ledger correction: the 09-13 build emitted the corrupt scales (build
  digest 59872174 = the placed digest), not a post-verify repair.
- weightsd fleet: units restarted spark1-7; spark3/spark6 binaries
  reinstalled (sha e5a60a6e); spark0 unit exits cleanly post-ready with
  no signal (strace clean) - manual setsid instance stable; 8/8 rank
  hosts served.

## Blocked (MEASURED, evidence in CEPH_LEASE)

- Fleet ceph metadata-wedge incident since ~18:00Z: D-state
  folio_wait_bit_common for every sustained header/stat/small-read
  pattern (five Q3F pythons across spark0/1/2/3/4/5 + the emit pass on
  spark7 + HY4-T1's decoder 2h+ wedged with zero output), while
  single-stream dd keeps passing. Escalated 23:30Z for
  sysadmin/operator; per-client kceph/MDS stale-cap class.
- Consequently: sidecar/sha/receipt/verify/relock on the remaining 14
  tp8 hosts, tp4pp4 stage3 12-15, fixture regen, the T1 decode wave,
  and measured B1 tok/s did NOT run. T1: NOT REACHED. B1: correctly
  not attempted (gated on T1). No loosening; every parked state is
  fail-closed (stale sidecar digests make weightd refuse the packs).
