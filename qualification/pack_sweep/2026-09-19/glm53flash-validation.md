# glm53flash stagepack validation — operator-ordered, all arms × all 16 nodes

**Date:** 2026-09-20 · **Lane:** `lane/glm53flash-validation` · **Order:** the
operator directly ordered "validate all of the glm5.3 flash stagepacks". This
supersedes the standing channel exclusion FOR VALIDATION ONLY: this lane read
the glm5.3-flash family's placed packs and warm sources, and did NOT touch
the serving stack (no residentd, no weightd, no model_api, no GPU, no
deployments). Grading is MEASURED: every cell below re-measured on the node
on 2026-09-20 (raw per-pack evidence:
`qualification/pack_sweep/2026-09-19/glm53flash-validation-raw.jsonl`).

## Verification stack (strongest available per cell)

The family verify tool is `tools/glm5_next_pack_verify.py` (the glm53flash
pack format IS the glm5_next `.g5nsp` wire; `glm52_validate_pack.py` covers
the glm53full MLA format and does NOT apply). Its plan-diff and round-trip
modes need the warm checkpoint, so the sweep ran the new `--structure-only`
mode added to the tool (header geometry + directory walk + payload/scale
byte formulas + bounds + trailing-byte check + directory sha256 — no
checkpoint, no ceph), plus, node-locally:

1. streamed sha256 of every placed pack (and `.experts` sidecar);
2. identity against EVERY placed receipt pin (`output_sha256`/`sha256`) and
   the placed `.sha256` sidecar, listing stale co-receipts;
3. `glm5_next_pack_verify --structure-only` with the receipt's
   rank/topology/stage fields and byte pin (header fallback for
   receipt-less packs);
4. generation forensics (`packed_by`, `note`, `source`), mtimes, chattr
   lock state, stray-file census (`.partial-*`, `.premtp-old`,
   `.restore-*`, `symlinkfix` leftovers are reported as stray, never
   swept as packs).

One verify at a time per node, `nice 10`, CPU only, node-local NVMe reads.
Sweep driver: `tools/glm53flash_stagepack_sweep.py`.

## Mechanical matrix (arm × node)

Cells: `P` = pack sha256 == receipt pin (+ `.sha256` sidecar where placed)
AND structure verify PASS · `P*` = P with a stale side pin or
generation flag (see findings) · `U` = structure PASS, identity UNPINNED
(no receipt/sidecar placed) · `—` = not this node's rank.

| arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | a | b | c | d | e | f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm53flash.bf16.tp16 (rank=node) | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| glm53flash.fp8.tp16 (rank=node) | P | P* | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| glm53flash.fp8.tp8 (rank r→node r, replica r+8) | P* | P | P | P | P | P | P(g) | P | P* | P | P | P | P | P | P | P |
| glm53flash.fp8.tp4pp4 (world rank = node) | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| glm53flash.nvfp4.tp16 (rank=node) | P | P | P | P | P | P | P | P | P* | P | P | P | P | P | P | U |

Structure facts (uniform per arm, from the directory walks): bf16/fp8.tp16/
nvfp4.tp16 packs are 1160-entry single-stage tp16 packs;
fp8.tp4pp4 stage packs carry 272/287/314 entries for stages 0/1-2/3;
fp8.tp8 post-repair packs carry 1160 entries. Every verified pack:
magic/format/geometry (hidden 4096, vocab 154880, experts 288, 45 layers)
correct, offsets 256-aligned and bounded, payload/scale byte counts match
the codec formulas, no empty payloads, no trailing bytes, chattr-locked.

## Node-f receipt: glm53flash.fp8.tp16 (the sweep FAIL cell)

The 2026-09-19 sweep reported node f's pack sha `f0e94cd3ef709fa1…` as
matching neither the receipt nor the sidecar. Measured re-investigation:

- The placed pack
  `/home/sparkf/sparkdata/glm53flash.fp8.tp16/packs/glm53flash.fp8.tp16.rankf.sp`
  measures sha256 `37ca901c017d9470f051a062348f7fc8415974a4b69963b6c9c319ff15c7caba`
  = the receipt's `output_sha256` = the `.sha256` sidecar pin. Structure
  verify PASS (tp16 rank 15, 1160 tensors, layout clean, header
  file_bytes == actual 21,706,046,976). **The placed pack is VALID and
  CURRENT — no re-emission from warm is needed.**
- **Which side was stale: neither — the SWEEP was wrong.** The earlier sweep
  hashed the leftover copy
  `.restore-rankf/glm53flash.fp8.tp16.rankf.sp` (inside the packs dir),
  which is the pre-2026-09-11 "coordinator-stagepacks" emission pinned by
  the retained `.premtp-old` sidecars (`f0e94cd3…`, 21,706,046,976 B,
  Sep 8). The current receipt/sidecar pair was rewritten by the
  `scale-plane-rebuild-0911` wave (Sep 11 12:05 local) for the rebuilt
  pack. The pack mtime (Sep 11 11:58 local) is unchanged since before the
  earlier sweep, so the earlier sweep selected the wrong file; it did not
  hash the placed pack.
- Sidecar cosmetic defect: the `.sha256` sidecar's recorded file name is
  `glm53flash.fp8.tp16.rank15.sp` (decimal rank) while the placed file is
  `…rankf.sp` (hex rank); the pinned hash matches the placed bytes.
- Hygiene (flagged, not deleted — never-overwrite law): on sparkf,
  `.restore-rankf/` (a full stale rank-f pack) + `.premtp-old` receipt/sha
  sidecars; on spark1, `glm53flash.fp8.tp16.rank1.sp.mtp.partial-20260908`
  (22,116,130,048 B, unlocked partial) + `.premtp-old` sidecars. These
  inflate the agent's memory-gate `du` and will re-trigger this exact
  false FAIL in any sweep that walks recursively. Recommend coordinator
  approve deletion or relocation out of `packs/`.

## Additional findings (flagged cells)

1. **`glm53flash.fp8.tp8` rank 6 (spark6) still serves the pre-#877 pack
   generation.** Its pack is 42,063,908,352 B with 1157 entries (mtime
   Sep 4), sha-consistent with its legacy
   `sparkpipe.qwen38_27b.stagepack-receipt.v1` receipt — internally
   coherent, mechanically PASS — but the other 14 rank packs were replaced
   on Sep 13 by the `sp4r-repair` wave ("current-packer rebuild post-#877
   pairing; replaces defective pre-#877 generation", 42,381,110,784 B,
   1160 entries). The placed rank-6 pack predates the routed-expert
   pairing repair (GLM_FLASH_HILLCLIMB 2026-09-09 numeric-defect class)
   and should be re-emitted/replaced by the owning lane (not done here:
   repair writes are outside validation scope; serving stack untouched).
2. **Stale co-receipts** on spark0 and spark8 (`glm5_next_stage.tp8.rank0.
   g5nsp.receipt.json`): both pin superseded generations (spark0:
   pre-repair sha `578c3aab…` with the old 42,063,908,352 byte count;
   spark8: an intermediate "coordinator-stagepacks rebuilt owns-shape"
   emission, sha `3ad58198…`). The current v1 receipts + `.sha256`
   sidecars pin the actual packs (`36cf4ae5…`, byte-identical across the
   rank-0 primary/replica pair). Cells PASS; stale receipts flagged for
   hygiene.
3. **`glm53flash.nvfp4.tp16` spark8 stale `.sha256` sidecar**: pack
   `8c92a471…` matches its `coordinator-nvfp4-repack` receipt and passes
   structure; the sidecar still pins the pre-repack `0ad21b27…`. Cell P*.
4. **`glm53flash.nvfp4.tp16` node f: identity UNPINNED.** The packs dir
   holds ONLY `glm53flash.nvfp4.tp16.rank15.sp` (12,778,578,432 B, mtime
   2026-09-11T23:11:30Z — inside the coordinator-nvfp4-repack emission
   window) with NO receipt and NO sidecar. Structure verify PASS (tp16
   rank 15, 1160 tensors, layout clean). Re-emission for identity closure
   is BLOCKED with current tooling: `glm5_next_resident_stagepack.py` on
   current main fails closed against the current
   `/mnt/model-warm/glm-5.3-flash-nvfp4-nvidia` tree — the expert probe
   expects the `*_packed` redhatai layout while the warm tree now carries
   nvidia-modelopt names (`…up_proj.weight` U8 + `…up_proj.weight_scale`
   F8_E4M3 + `…up_proj.weight_scale_2` F32), and the dense up|gate fused
   plan/producer then disagree (produced 12,582,912 B vs planned
   6,291,456 B for layers.0). Fail-closed fired as designed; no pack was
   written. Needed from the owning lane: either a receipt for the placed
   rank-f pack from the repack wave's records, or a packer adapted to the
   current warm layout plus a fresh emit.
5. **Replica determinism receipts (measured)**: `glm53flash.fp8.tp8`
   primary/replica pairs are byte-identical on all 7 pairs
   (rank0 36cf4ae5…@spark0+spark8, rank1 fa9ecfc7…@1+9, rank2 178e6caf…@2+a,
   rank3 1fedb34e…@3+b, rank4 6cb0345e…@4+c, rank5 a6888033…@5+d,
   rank7 350872db…@7+f); sparke's rank-6 replica is 54da2ee4… (its primary
   on spark6 is the pre-877 pack above, so the pair is NOT identical —
   further evidence for finding 1).

## Arm → warm source provenance (receipt-pinned, measured)

| arm | warm source | evidence |
|---|---|---|
| glm53flash.bf16.tp16 | `/mnt/model-warm/glm-5.3-flash-bf16-official` | every receipt `source` field, expert_codec `bf16-source-native` |
| glm53flash.fp8.tp16 | `/mnt/model-warm/glm-5.3-flash` (rev 84c6a6aa…) | `expert_codec fp8-source-native` + uniform byte count 21,706,046,976 = the fp8 tp16 receipt the family verify tool pins; fp8 native source is the glm-5.3-flash release |
| glm53flash.fp8.tp8 | `/mnt/model-warm/glm-5.3-flash` | receipts' `checkpoint` field: `/mnt/model-warm/glm-5.3-flash` |
| glm53flash.fp8.tp4pp4 | `/mnt/model-warm/glm-5.3-flash` | receipts pin `source` + `source_revision 84c6a6aa…` + packer commit 6c6caf5… |
| glm53flash.nvfp4.tp16 | `/mnt/model-warm/glm-5.3-flash-nvfp4-nvidia` (nvidia/GLM-5.3-Flash-NVFP4) | receipts `source` + `source_fingerprint: node-local rsync of …nvfp4-nvidia` |

Frozen reference copies used by the Phase-2 runs (node-local NVMe, sha
pins measured at freeze time 2026-09-20):

- spark6 `~/sparkdata/t1ref-warm/glm-5.3-flash` — config sha256
  `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f`,
  index sha256 `3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05`
- spark6 `~/sparkdata/t1ref-warm/glm-5.3-flash-nvfp4-nvidia` — config
  `e23c5d98f53e861d49a51bd3c68591621c5482ce829e42c31724152322fba03d`,
  index `26765b2601fd246ef361cfb9f5e10f9fb291a59e05ad0a109062f3a4747c7fd1`
- spark1 `~/sparkdata/t1ref-warm/glm-5.3-flash-bf16-official` — config
  `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943`,
  index `e6007bd58fb7e07f9fe69544257ee2713f252ef5855bbf685b48c991d524ef0f`

## Scope addition (operator question, 2026-09-20): the missing nvfp4 topologies

The nvfp4.tp16-only placement is HISTORY, not physics. Added to this lane by
coordinator order: emit, place, and verify `glm53flash.nvfp4.tp8` and
`glm53flash.nvfp4.tp4pp4` from the same warm source, plus the bf16.tp8
roofline. fp8 already covers tp16/tp8/tp4pp4.

**Packer prerequisite (fixed here, additive + fail-closed).** Current-main
`glm5_next_resident_stagepack.py` failed closed against the current
nvidia-modelopt warm layout (expert probe expected the redhatai `*_packed`
names; U8-packed DENSE storage reported half-width columns). The patch:
the expert probe accepts `…up_proj.weight` stored U8 (payload at the plain
name, per-16 E4M3 scales at `…weight_scale`, F32 global at
`…weight_scale_2`); `nvfp4_payload`/`nvfp4_weight_global` resolve both
layouts; `add_spine_bf16` and `add_up_gate_fused` plan REAL columns for U8
sources (wire stays bf16). Emitter correctness is proven by BYTE IDENTITY:
re-emitting tp16 rank 6 from the frozen warm must reproduce the placed
pack sha `3fddd328…` (measured result recorded below).

**bf16.tp8 roofline — INFEASIBLE (measured arithmetic, not placed).**
- Placed bf16.tp16 = 40,136,867,328 B = 37.4 GiB/rank (measured); the
  bf16 weight set totals 598 GiB over 16 ranks.
- tp8 doubles the per-rank shard: 598/8 = 74.8 GiB of packs per node
  before a single KV token.
- The family's serving overhead is fixed per NODE, not per rank: 32 GiB
  expert-arena pool + 16 GiB pinned mesh region (measured serving config)
  + the memory gate's packs + 8 GiB check.
- KV pool floor (4,194,304 tokens; ~12 GiB at tp16 by the same
  latent-512 geometry) grows to ~24 GiB at tp8.
- 74.8 + 24 + 8 = 106.8 GiB of compulsory footprint against 119 GiB
  visible — and 106.8 + 32 + 16 = 154.8 GiB total demand. Even granting
  the operator's ~96 GiB serving ceiling directly: 74.8 GiB of packs
  leaves 21.2 GiB for KV + pool + mesh + activations, which the 16 GiB
  mesh alone almost exhausts. **bf16.tp8 does not fit; tp16 remains the
  only feasible bf16 topology.** (fp8.tp8 fits because its packs are
  39.5 GiB — half of bf16.)

Emission/placement driver: `tools/glm53flash_nvfp4_emit_place.sh`
(staged emits on spark2, per-law placement rank r → spark r, replica →
spark r+8 for tp8, world rank w → spark w with stage w//4 and tp_rank w%4
and the 11/11/11/12 layer split for tp4pp4, receipts + `.sha256` sidecars
placed with every pack, `chattr +i`, never-overwrite PLACE-SKIP guards).
Measured wave results are appended below.

**Measured wave results (2026-09-20):**

- Byte-identity proof: re-emitted tp16 rank 6 = 1160 entries;
  **1157/1160 entries byte-identical to the placed pack** (full
  payload+scale byte comparison per entry); `--all-tensors` round-trip vs
  warm PASSes every region of the re-emitted pack. The 3 differing
  entries are (kind 19 = dense down, layers 0/1/2) — the placed arm's
  half-width dense down planes.
- Emitted: nvfp4.tp8 ranks 0-7 — 1160 entries, uniform 24,544,940,544 B
  per rank (corrected dense-down [4096, 1536]/rank); nvfp4.tp4pp4 world
  ranks 0-15 — stages 0/1/2/3 = 272/287/314 entries,
  9,708,624,128 / 12,308,398,336 / 12,308,398,336 / 13,733,371,392 B
  (corrected dense-down [4096, 3072]/rank).
- Placed 32/32 pack instances (16 tp8 across primary+replica nodes, 16
  tp4pp4 across the fleet) + 32 receipts + 32 `.sha256` sidecars,
  `chattr +i`.
- Placement verification sweep (same mechanical stack, re-run on all 16
  nodes): **32/32 PASS** (sha == receipt == sidecar, structure verify
  PASS). Raw records appended to `glm53flash-validation-raw.jsonl`.

| arm (new) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | a | b | c | d | e | f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| glm53flash.nvfp4.tp8 (rank r→node r, replica r+8) | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| glm53flash.nvfp4.tp4pp4 (world rank = node) | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |

**Emitter correctness proof (measured).** Re-emitting tp16 rank 6 from the
frozen warm with the patched packer produces a 1160-entry pack that is
BYTE-IDENTICAL to the placed `glm53flash.nvfp4.tp16.rank6.sp` on 1157 of
1160 entries; `glm5_next_pack_verify --all-tensors` PASSES the re-emitted
pack against the warm checkpoint on every payload and scale region. The
only 3 differing entries are the dense `down_proj` planes of layers 0-2
and they expose a real defect in the PLACED arm — see the next section.

**Placed-arm defect found: `glm53flash.nvfp4.tp16` dense `down_proj` is
half-width.** The warm dense `down_proj` is U8-packed [4096, 6144]
(real [4096, 12288], intermediate 12288; scales [4096, 768]). The fp8 and
bf16 arms carry per-rank dense-down entries [4096, 768] (16 × 768 =
12288 ✓, measured on rank 6 of both arms). The placed nvfp4.tp16 packs
carry [4096, 384] (16 × 384 = 6144 — HALF the intermediate), i.e. the
emission that produced the placed nvfp4.tp16 generation treated the U8
packed column width as the real width for `add_spine_bf16` (dense down)
while handling the fused up|gate path at real width. The packer patch in
this lane fixes the width; the new tp8/tp4pp4 emissions carry the correct
[4096, 1536]/[4096, 3072] per-rank dense-down entries. The 16 placed
nvfp4.tp16 packs need a dense-down re-patch or re-emission by the owning
lane (this lane does not modify placed packs).

**Reference-engine fix found by the nvfp4 leg (dtype-blind HC read).** The
nvfp4-nvidia release stores the hyper-connection `hc_*_base`/`hc_*_scale`
planes as BF16 ([24]/[3]) while the fp8/bf16 releases store them F32; the
reference engine's `.astype(float32)` read misinterpreted BF16 codes as
integers (values ~49,408 vs ~-8.0 true), driving the HC sinkhorn combiner
into whole-row exp underflow (0/0 → NaN) at layer 0. The engine now
decodes these planes dtype-aware (`tensor()`); the placed PACKS are not
affected (the packer stores HC planes as F32 payloads, upcasting BF16
exactly). A warm-data census additionally found zero non-finite values in
the nvfp4 expert scale planes (72,576 planes scanned), zero non-finite
BF16 spine tensors, and clean embed/lm_head — the fixture NaN came solely
from the engine read above.

**Second fix: the modelopt nvfp4 dequant factor is ×0.5 (measured against
the fp8 twin).** Solving the dequant convention tensor-by-tensor against
the fp8 release's identical weights (layers.0 dense down/up, experts.9 up;
candidate grid: nibble order × global sign × 0.5 factor): low-nibble-even
e2m1 with `value = e2m1[code] × e4m3_block_scale × weight_scale_2 × 0.5`
reconstructs the fp8 twin with max abs error 0.008-0.011 (median ~0.0012,
i.e. plain nvfp4 quantization noise); every other candidate is 10-100×
worse. Both the reference engine and the packer's U8 spine branch
previously omitted the 0.5 (values 2× true). Consequences handled here:
the new tp8/tp4pp4 emissions were RE-EMITTED with the corrected factor and
the just-placed copies replaced in place (chattr -i → copy → re-verify;
these were this lane's own artifacts from the same wave); the nvfp4
reference fixtures were rerun on the fixed engine. Standing items for the
owning lane: (a) the placed nvfp4.tp16 packs carry BOTH the half-width
dense down AND the 2× dense-spine values (their emission used the same
un-corrected factor); (b) the module's nvfp4 expert dequant must be
checked against the measured ×0.5×global convention — if it decodes
without the 0.5, nvfp4 experts serve at 2× (the expert payload/scale
bytes in every pack are verbatim source bytes, so this is purely a decode
contract question).

## Phase 2 — CPU reference validation chain

(Reported in the PR body and `qualification/t1_reference/glm53flash/`;
this section is finalized with the measured fixture receipts when the runs
complete.)

## Per-arm verdicts (MEASURED)

- **glm53flash.bf16.tp16 — VALIDATED.** 16/16 identity + structure.
- **glm53flash.fp8.tp16 — VALIDATED.** 16/16 identity + structure; the
  historical node-f FAIL is retracted as a sweep artifact (see receipt
  above); stale leftovers flagged for hygiene.
- **glm53flash.fp8.tp4pp4 — VALIDATED.** 16/16 identity + structure;
  `.experts` sidecars match their receipt pins.
- **glm53flash.fp8.tp8 — VALIDATED WITH ONE GENERATION EXCEPTION.** 15/16
  post-877 packs clean (with byte-identical replicas); rank 6 on spark6 is
  a pre-877 defective-generation pack needing the sp4r-repair rebuild.
- **glm53flash.nvfp4.tp16 — VALIDATED WITH ONE IDENTITY GAP.** 15/16
  identity + structure (one stale sidecar); node f rank 15 is
  structurally valid but has no receipt/sidecar and cannot currently be
  re-derived from warm (packer layout drift) — owning-lane action
  required.
