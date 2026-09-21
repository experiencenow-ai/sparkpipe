# Pack-emission wave 2 — closing the six warm-quant gaps + the MTP-strip repair — 2026-09-20

Wave 1 (#1046) emitted 4 of the 9 unplaced warm quants and left six measured
gaps plus the coordinator-ruled repair of the qwen27b.tp4 rank != 0 packs.
This wave closes them. CPU only; every emission read warm `/mnt/model-warm`
on sparkc (the sanctioned final warm read — the OUTPUT is NVMe packs).
Placement law: rank r -> spark r (hex), replica -> spark r + 8; no existing
pack was overwritten except the corrupt `qwen3flash.fp8.tp8` arm, whose
overwrite the coordinator sanctioned explicitly. Every heavy write node ran
`sync` + `drop_caches` after placement; packs are chattr-locked.

Every quantization convention below was pinned NUMERICALLY against a warm
twin before any payload was produced; the per-arm sections carry the numbers.

## Verdicts (grading is MEASURED)

| # | arm | verdict |
|---|---|---|
| 1 | minimax.text.bf16.tp4 | EMITTED+PLACED+VERIFIED |
| 2 | ling.fp8.tp16 | EMITTED+PLACED+VERIFIED |
| 3 | laguna fp8 / nvfp4 | FAILED (mixed routed-expert dtypes; census + codec paths shipped, fail-closed demonstrated) |
| 4 | gemma4_31b.nvfp4.tp16 | EMITTED+PLACED+VERIFIED |
| 5 | dsv41flash.nvfp4.tp8 | BLOCKED-MODULE (nvidia reader proven; module has no nvfp4 expert codec) |
| 6 | qwen3flash.fp8.tp8 re-emission | EMITTED+PLACED+VERIFIED (corrupt arm overwritten per sanction) |
| + | qwen27b.tp4 MTP-strip repair | REPAIRED (12/12 packs carrying the signature) |

## Arm 1 — minimax.text.bf16.tp4: EMITTED+PLACED+VERIFIED

The warm tree is a MiniMaxH3ModularPipeline monorepo; the serving target per
the merged #1044 scope is the TEXT TOWER `text_encoder/` (vision, audio and
the DiT are out of scope and never referenced). The tower is a Qwen3VL text
submodel: 64 dense layers, hidden 5120, 64 query heads x 128, 8 KV heads
(GQA), FFN 25600, vocab 151936, untied lm_head, RMSNorm without the +1 fold,
all BF16 — the same dense-GQA family shape as the qwen38_27b packer, whose
TPmax is TP4.

- New packer: `tools/minimax_stagepack.py`, modeled on
  `tools/qwen38_27b_stagepack.py` (120-byte 26I2Q header, 56-byte 6I4Q
  entries, 256-byte payload alignment, magic 'MNTX'). Config pin fails
  closed on any other release; every consumed tensor is pinned by name,
  dtype and shape; a `--verify` mode re-parses a pack and re-derives every
  directory entry from the inventory (the gemma4-style receipt).
- Topology: TP4. 5120/64L is the 27b geometry whose fleet topology is TP4;
  every shard dimension divides (q heads 64/4, kv heads 8/4 so GQA shards
  without replication, FFN 25600/4, vocab 151936/4). The embedding
  replicates (no collective broadcast yet); lm_head shards by vocab rows.
- Per rank: 707 tensors, 17,548,985,344 B (16.34 GiB); rank0 sha256
  `79c5dba912119508…`. Two-pass tail verify PASS on all 4 ranks at build.
- Placement: rank r -> spark r + replica spark r+8 (4 ranks x 2 = 8 nodes),
  `~/sparkdata/minimax.text.bf16.tp4/packs/minimax.text.bf16.tp4.rank<h>.mntx`
  + `.sha256` + `.receipt.json`, sha-verified at placement, chattr-locked.
- No module consumes this wire yet (no minimax module exists on main); the
  pack is the serving-target artifact for the minimax lane. Disk: 8 x 16.34
  GiB = 130.7 GiB.

## Arm 2 — ling.fp8.tp16: EMITTED+PLACED+VERIFIED

Wave 1 verdict was FAILED ("packer is bf16-only"). The module was never the
blocker: `spark_ling_stagepack_format.h` marks the routed-expert kinds
`codec_from_arg`, and `SparkLingStagePackExpectedShape` accepts any expert
codec in [INT6..MXFP4] — including FP8_E4M3 — with
`SparkWeightCodecScaleBytes` capping the fp8 plane at
`groups x rows x ceil(cols/128) x 4` F32 bytes. The packer was.

- `tools/ling_stagepack.py` gains the `--expert-codec fp8` arm:
  - routed experts pass through VERBATIM on the FP8 wire (codec 5,
    PAYLOAD_PACKED_WEIGHT, scale encoding F32): per-expert up|gate fused
    rows and down columns, unconverted F8_E4M3 codes;
  - the [rows/128, cols/128] `weight_scale_inv` planes are regathered to
    the module's per-output-row layout (payload row r carries block row
    r//128 across all column blocks);
  - the expert manifest now carries BOTH planes (kind = tensor_kind*2 and
    *2+1, the glm52 fp8 convention) with per-expert ck128 slabs;
  - every non-routed-expert tensor (attention projections, dense MLP,
    shared experts — 63332 - 61440 = 1892 fp8 spine tensors + planes)
    dequantizes to BF16 through `weight_scale_inv` with ONE
    round-to-nearest-even bf16 rounding, because the module pins
    `linear_weight_codec == BF16`; the packer still never quantizes;
  - `model-families/ling/name_map_fp8.json` pins the fp8 census: 127115
    tensors at the measured index sha `c5b0212d…` / config sha `621969cc…`.
- Convention pinned numerically against the warm bf16 twin
  (`ling-3.0-flash`): W = code x scale_inv per-128-block. Measured
  rel-mean 2.25%, rel-p99 5.6% on four spot tensors spanning attention,
  dense, shared and routed experts (pure fp8 quantization noise); the
  divide reading is off by ~8 orders of magnitude.
- Census closes: 127115 = 124020 packed + 3095 MTP (layer 42 omitted).
- Per rank (TP16): 730 pack tensors, 8,542,324,224 B — spine bf16
  625,519,384 B + routed-expert fp8 payload 7,549,747,200 B + per-row
  F32 scale planes; 81,920 manifest ranges (40 layers x 2 kinds x 512
  experts x 2 planes). 16/16 ranks emitted; the family verifier
  (`tools/ling_verify_pack.py`, extended this wave to accept the fp8
  expert arm and check the per-row scale-plane geometry) PASSes all 16
  ranks — 11,680 pack tensors, boundary-rank proof (identical counts,
  complementary head/vocab/expert shards between ranks 0 and 15) — and
  the receipt census closes on every rank. Placement rank r -> spark r.
- A serving module must be built with `LING_EXPERT_WEIGHT_CODEC` = fp8
  (Makefile `EXPERT_CODEC=fp8`); the pack header carries codec 5 and fails
  header validation on any other build, per the module's own check.

## Arms 3 — laguna fp8 + nvfp4: FAILED (mixed routed-expert dtypes)

Wave 1 failed both arms at the census lock. This wave ships the census and
the codec paths, and the arms then fail closed on a deeper, measured
property of the releases themselves.

- `model-families/laguna/tensor_patterns.json` gains measured census
  variants (the 23-pattern/36769 bf16 lock is untouched):
  - fp8: 26 patterns / 69793 tensors (3745 BF16 + 33024 F8_E4M3 + 33024
    F32 scale_inv) — only the ROUTED EXPERTS are quantized, layers 1..43
    (11008 = 43 x 256 per projection); layers 44..47 keep BF16 experts;
  - nvfp4: 35 patterns / 126625 tensors (6817 BF16 + 29952 U8 packed +
    29952 F8_E4M3 per-16 planes + 59904 F32 globals) — routed experts
    quantized on layers 1..39 (`weight_packed` + `weight_scale` +
    `weight_global_scale` + `input_global_scale`); layers 40..47 carry
    BF16-only `weight`.
- `tools/laguna_stagepack.py` gains `--expert-codec {fp8,nvfp4}` with the
  module-wire expert layouts (fp8: per-row per-128-block F32 planes;
  nvfp4: per-16 e4m3 plane + the F32 weight global as the 4-byte tail of
  each expert's scale slab, matching the laguna module's per-expert
  manifest slicing), plus the variant census selection.
- Why FAILED, measured: the laguna module (like every stage module) pins
  ONE expert codec per pack — `LAGUNA_EXPERT_WEIGHT_CODEC` at build time,
  `SparkLagunaStagePackExpectedShape` validates every routed-expert entry
  against it. The releases quantize a SUBSET of the routed layers, so an
  all-fp8 (or all-nvfp4) pack would have to quantize the BF16 layers'
  experts — the packers never quantize — and a mixed pack cannot load.
  The fail-closed is demonstrated, not asserted:
  - fp8 arm: `FAIL layer 44 routed experts are not native to the 5 arm:
    model.layers.44.mlp.experts.0.gate_proj.weight is BF16 …`
  - nvfp4 arm: `FAIL layer 40 routed experts are not native to the nvfp4
    arm: …weight_packed is absent …`
  NOT PLACED (123 GiB + 93 GiB warm sources untouched).

## Arm 4 — gemma4_31b.nvfp4.tp16: EMITTED+PLACED+VERIFIED

Wave 1 failed on `mlp.gate_proj.weight: dtype U8, expected BF16 (never
quantize)`. The measured census (1728 tensors = 1008 BF16 + 360 F32 +
180 F8_E4M3 + 180 U8) shows the release quantizes exactly the per-layer
MLP gate/up/down projections to nvfp4; the attention geometry is
bit-for-bit the bf16 release's.

- `tools/gemma4_stagepack.py` gains the nvfp4 MLP reader: the packed U8
  payload is decoded to its DEFINED bf16 values —
  W = e2m1(U8, low nibble first) x e4m3(weight_scale per-16) x
      weight_scale_2 (F32 scalar), one RNE bf16 rounding —
  with row/column windows decoded in place for the TP slices. This is the
  U8->BF16 upcast of the quantized set only; the never-quantize set
  (attention, norms, embed, layer_scalar) passes through byte-identical
  to before, and nothing is ever quantized in the other direction.
- Convention pinned elementwise against the warm bf16 twin
  (`gemma-4-31b-it`): decoded values reproduce the twin within e2m1
  quantization noise (e.g. row 5 of layer-0 gate_proj: twin 0.005188 /
  1.0 x 4.284087e-03; -0.026123 / -6.0 x 4.284087e-03 …). `input_scale`
  is the activation-side scale and never enters the weight decode.
- Per rank (TP16): 723 tensors — the same census as the bf16 arm —
  3,881,009,408 B; `mlp_source: nvfp4-decoded-bf16` in the receipt;
  two-pass placement proof PASS at build. 16/16 ranks emitted; the
  source index sha `aff5569bed013db0…` matches the wave-1 pin.
- Placement: rank r -> spark r (tp16 covers the fleet, no replicas),
  `~/sparkdata/gemma4_31b.nvfp4.tp16/packs/`; `--verify-existing` on a
  placed pack: `tensors=723 proof=True`. Disk: 16 x 3.88 GB = 62.1 GB.

## Arm 5 — dsv41flash.nvfp4.tp8: BLOCKED-MODULE

Wave 1 probed the wrong packer for this family (`tools/dsv4_stagepack.py`
is the dsv4 flash/pro packer); the dsv4.1-flash family packer is
`tools/dsv41_flash_stagepack.py`. This wave:

- adds the nvidia-convention reader: per routed-expert projection, U8
  packed e2m1 [rows, cols/2] (the same byte layout as the DSpark I8
  plane), F8_E4M3 per-16 scale plane [rows, cols/16], and two F32 scalars
  (`input_scale`, `weight_scale_2`) — the same convention #1049's T1
  engine measured;
- proves it with `validate-nvidia`: all 40 layers x 384 experts x 3
  projections x 4 tensors = **184320 tensors verified** against the
  convention (dtypes U8 46080 + F8_E4M3 46080 + F32 92160), the full
  expert inventory of the release accounted;
- does NOT emit. The dsv41_flash module accepts expert codecs
  MXFP4_E2M1 and FP8_E4M3 ONLY
  (`SparkDsv41FlashStagePackExpectedShape`: `if mxfp4 … else if fp8 …
  else return 0`) — the same module-lacks-the-codec law the wave orders
  define for ling. A nvfp4-wire pack would fail the entry check at load;
  transcoding e4m3-g16 to E8M0-g32 would be a requantization. The wire
  needs the module format bump first; the reader is ready when it lands.
  NOT PLACED (492 GiB warm source).

## Arm 6 — qwen3flash.fp8.tp8 re-emission: EMITTED+PLACED+VERIFIED

The placed arm was corrupt (receipt mismatches on 13 of 16 nodes, a
partial re-pack and multi-era sidecars in its past). The coordinator
sanctioned overwriting THIS arm.

- Re-emitted all 16 rank packs from `/mnt/model-warm/qwen3.8-flash-next-fp8`
  with the main-tree packer:
  `tools/qwen4_flash_stagepack.py --expert-format fp8-official --no-mtp
  --tp-degree 8` (48 layers, 1215 tensors, 30,518,612,480 B per rank).
- THE DEEPER FINDING — the arm was not merely mis-placed, the bytes were
  bad: the family verifier
  (`tools/qwen4_flash_pack_verify.py --source-layout fp8-official`)
  fails the first rebuild with `fp8-official scale plane mismatch (all
  64 experts, byte-exact)` on every routed-expert kind. Root cause,
  measured in the main-tree packer:
  `copy_fp8_official_experts` opened each per-expert
  `weight_scale_inv` shard but NEVER SOUGHT to the tensor offset — the
  pack's expert scale segments contained the shards' safetensors HEADER
  bytes (the loop tracked `offset = s_off` and never used it). The
  2026-09-14 "wave-pr" build carried the identical bug with a bogus
  `PASS` receipt: its verify ran sampled payload byte-traces, which pass,
  and never exercised the scale planes byte-exactly. This wave fixes the
  packer (`sf.seek(s_off)`), re-emits, and the family verify now passes
  the full entry walk including the byte-exact expert scale planes.
- Determinism receipt: the corrupt-era rank0 sha `598721746c105c85…`
  reproduced byte-identically from the (buggy) main-tree packer — the
  packer is deterministic; the shipped bytes were corrupt because the
  packer was. The FIXED rebuild's rank0 sha `b6f780982db8f7e7…` equals
  the sha of the file that was originally placed on spark0 on 09-14 —
  i.e. the wave-pr era DID produce correct bytes for at least rank0 and
  then corrupted the fleet with mixed partial copies and lying receipts
  (13 of 16 placed files did not match their receipts). The fixed
  per-rank shas are the placed sidecars:
  `b6f78098…` (r0), `a2334408…` (r1), `1add7b83…` (r2), `3da5bdcf…` (r3),
  `1a82c5cc…` (r4), `353eb9a5…` (r5), `659ad3b4…` (r6), `16a36127…` (r7).
- Family verify on placed packs (byte-exact expert scale planes
  included): rank0 on spark0 `PASS … header geometry, 1215 directory
  entries (tp 8/0), 8 byte-traced samples receipt=verified`, zero scale
  plane mismatches; rank5 on spark5 likewise.
- Placement: rank r -> spark r + replica spark r+8, sha-verified per
  node against the staged sha before lock; the corrupt-era artifacts
  (2-digit-named `rank01..07` packs, `.patch.json`,
  `.g5nsp.receipt.json`, `.experts.pre-repair`, stale receipts) were
  purged; fresh `.receipt.json` (with `placed_on`/`placed_at`/`wave`)
  and `.sha256` sidecars written; packs chattr-locked; sync +
  drop_caches per node. Disk: 16 x 30.5 GB = 488.3 GB.

## The MTP-strip repair (coordinator ruling executed)

The 2026-09-05 16-wide strip ran with the old qwen36sp map
(`mtp_index` 25 — actually tp_rank) and zeroed `tp_rank` on every
rank != 0 pack while leaving `mtp_layer_count` = 1. The fix and a
regression test existed in the unmerged wave-1 WIP commit and are now on
main via this PR: `FAMILIES["qwen36sp"]["mtp_index"] = 23` with explicit
`tp_degree_index` 24 / `tp_rank_index` 25
(`tests/test_stagepack_mtp_strip_qwen36sp.py`, 5 checks PASS), plus the
new `--repair-header` mode.

- Repair mode: `stagepack_mtp_strip.py --repair-header --expect-tp-degree
  4 --expect-tp-rank <r>` fails closed unless the exact corruption
  signature holds (mtp field nonzero, tp_rank field == the expected
  rank, tp_degree field 4, no MTP entries left in the directory), swaps
  the fields back (for rank 0 that is mtp->0 only, the field that is
  actually wrong there), re-shas, and writes a `header_repair` receipt
  with the before/after words.
- Fleet state MEASURED 2026-09-20: the arm was re-shaped by other
  operators between the waves — wave 1 measured 48 rank != 0 packs
  (ranks 1..3 x 16 nodes); today the arm holds ONE pack per node on all
  16 nodes (12 rank != 0 packs + 4 rank0 packs). All 16 carried the
  signature (rank != 0: mtp=1 tpdeg=4 tprank=0; rank0: mtp=1 with
  tprank=0 already correct) and ALL 16 are repaired. The root-owned
  `.backup` originals are untouched.
- Receipt (spark1, rank1 — representative; all four rank-1 nodes
  byte-identical post-repair):
  - before: u32[23]=1 u32[24]=4 u32[25]=0, header bytes 92:104
    `01000000 04000000 00000000`
  - after: u32[23]=0 u32[24]=4 u32[25]=1, header bytes 92:104
    `00000000 04000000 01000000`
  - sha256 `bc1a983b72f60d54…` -> `d014e833544046f6…`, receipt carries
    `header_repair.before/after`, relocked.
- Determinism cross-check: repaired rank1 sha `d014e833…` identical on
  spark1/spark5/spark9/sparkd; rank2 `a1491177517eb085…` identical on
  spark2/spark6/sparka/sparke; rank3 `f0bbaf48…` identical on
  spark3/spark7/sparkb/sparkf; rank0 `986fa147…` identical on
  spark0/spark4/spark8/sparkc.

## Placement matrix (measured after placement)

`S` = pack present, sha256 == the placed `.sha256` sidecar, chattr-locked.
`r<n>` in each cell names the rank the node holds.

| arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | a | b | c | d | e | f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minimax.text.bf16.tp4 | S r0 | S r1 | S r2 | S r3 | — | — | — | — | S r0 | S r1 | S r2 | S r3 | — | — | — | — |
| ling.fp8.tp16 | S r0 | S r1 | S r2 | S r3 | S r4 | S r5 | S r6 | S r7 | S r8 | S r9 | S ra | S rb | S rc | S rd | S re | S rf |
| gemma4_31b.nvfp4.tp16 | S r0 | S r1 | S r2 | S r3 | S r4 | S r5 | S r6 | S r7 | S r8 | S r9 | S ra | S rb | S rc | S rd | S re | S rf |
| qwen3flash.fp8.tp8 | S r0 | S r1 | S r2 | S r3 | S r4 | S r5 | S r6 | S r7 | S r0 | S r1 | S r2 | S r3 | S r4 | S r5 | S r6 | S r7 |

Determinism cross-checks (byte-identical packs on multiple nodes):
minimax rank0 `79c5dba9…` on spark0+spark8, rank1 `0cf438e6…` on
spark1+spark9, rank2 `a6c41cc4…` on spark2+sparka, rank3 `9e4a52f3…` on
spark3+sparkb. qwen3flash rank0 `b6f78098…` on spark0+spark8,
rank1 `a2334408…` on spark1+spark9, rank2 `1add7b83…` on spark2+sparka,
rank3 `3da5bdcf…` on spark3+sparkb, rank4 `1a82c5cc…` on spark4+sparkc,
rank5 `353eb9a5…` on spark5+sparkd, rank6 `659ad3b4…` on spark6+sparke,
rank7 `16a36127…` on spark7+sparkf.

## Disk impact

| arm | per-pack bytes | packs | total |
|---|---|---|---|
| minimax.text.bf16.tp4 | 17,548,985,344 | 8 (4 ranks + replicas) | 140.4 GB |
| ling.fp8.tp16 | 8,542,324,224 | 16 | 136.7 GB |
| gemma4_31b.nvfp4.tp16 | 3,881,009,408 | 16 | 62.1 GB |
| qwen3flash.fp8.tp8 | 30,518,612,480 | 16 | 488.3 GB |
| qwen27b.tp4 repair | 0 (in-place header swap) | 12 | 0 |

## Wave-2 tool changes

1. `tools/minimax_stagepack.py` — new (arm 1).
2. `tools/ling_stagepack.py` + `model-families/ling/name_map_fp8.json` —
   the fp8 expert arm (arm 2).
3. `model-families/laguna/tensor_patterns.json` +
   `tools/laguna_stagepack.py` — census variants + fp8/nvfp4 expert
   paths with the mixed-source fail-closed (arms 3).
4. `tools/gemma4_stagepack.py` — the nvfp4 MLP decode reader (arm 4).
5. `tools/dsv41_flash_stagepack.py` — the nvidia-convention reader +
   `validate-nvidia` (arm 5).
6. `tools/stagepack_mtp_strip.py` +
   `tests/test_stagepack_mtp_strip_qwen36sp.py` — the main-tree landing
   of the wave-1 qwen36sp fix, the tp field map, the `--repair-header`
   mode, and the extended regression test.
