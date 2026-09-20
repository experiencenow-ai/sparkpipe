# Topology coverage survey — skipped viable topologies across all models×quants — 2026-09-20

Operator question: "how many viable topologies have been skipped across all
the models?"

Surveyed revision: `origin/main` `7ffb6f2` (includes the 2026-09-20 sweep
addendum, PR #1066). Fleet measurement: `spark0` node-local NVMe
(`du -sb ~/sparkdata/<arm>/packs`), 2026-09-20; arms are replicated across
the 16 nodes, one node is representative for pack-set bytes. CPU only,
read-only.

## Method and rules

Candidate set per model×quant, from the operator placement ruling
(2026-09-11 — restated in the survey orders; on main it is exercised in
`docs/AGENT_LANE_BRIEFS/dsv41_flash.md` "TP8xPP2 … per operator ruling
2026-09-11" and the dsv41 M7 rulings "TP8 confirmed, TP16 packs dropped"):

- TP16-geometry models (every sharded head/expert count divides 16):
  candidates = **TP16 + TP4×PP4**.
- TP8-geometry models (tp16 blocked by geometry, tp8 divides):
  candidates = **TP8 + TP8×PP2 + TP4×PP4**.
- Standalone sub-16 (TP8/TP4) added only when per-node pack bytes at that
  rank count ≤ ~30 GB **and** the family driver already implements that TP
  width. Size rule as applied: per-node = 16-rank total × (16/ranks).
- dsv41 is governed by its explicit 09-11 ruling (mesh TP8×PP2 over the 16
  sparks; TP16 packs dropped by ruling), which overrides the geometry-class
  matrix.

Viability rule per candidate (graded DERIVED; every input is cited):

1. geometry — sharded head/expert counts divide the TP width (KV
   replication allowed where the family packer implements it); layer count
   divides the PP count, or the family tooling already implements the exact
   uneven split;
2. driver — the family has implemented topology support: a compiled TP pin,
   gen_deployment, or a placed arm proves the TP width; PP stage machinery
   proven somewhere in the family (placed pp>1 arm or stage-plan tooling);
3. codec — the family packer accepts the quant on main (measured verdicts
   in `emission-wave.md`);
4. memory — per-node bytes + ~15 GB runtime ≤ 96 GB.

SKIPPED = viable ∧ ¬placed. PLACED = arm attested on main (`matrix.md`
including its 2026-09-20 addendum, plus the four arms emitted+placed in
`emission-wave.md`: `gemma4_31b.bf16.tp16`, `lingfin.bf16.tp16` + MTP
sidecar, `qwen27b.fp8.tp4`, `qwen27b.fp8.tp4pp4`).

## Geometry facts (MEASURED — family headers, origin/main)

| family | scope model | q heads | kv heads | experts | layers | hidden | tp16 | tp8 | tp4 | pp notes |
|---|---|---|---|---|---|---|---|---|---|---|
| ling | ling / lingfin | 32 MLA (+32 KDA) | latent | 512 | 42 | 2560 | Y | Y | Y | 42 % 4 = 2 (uneven) |
| gemma4 (MoE) | gemma4-26b | 16 / 8 slide, 2 full | replicate | 128 | 30 | 2816 | Y | Y | Y | uneven 8/8/7/7 implemented |
| gemma4 (dense) | gemma4-31b | 32 / 16 slide, 4 full | replicate | — | 60 | 5376 | Y | Y | Y | 60 % 4 = 0 |
| glm52 | glm53full | 64 MLA | latent | 256 | 78 | 6144 | Y | Y | Y | 78 % 4 = 2 (uneven implemented, placed) |
| hy4 | hy4 | 64 MLA | latent | 256 | 78 | 6144 | Y | Y | Y | 78 % 4 = 2 (uneven) |
| k3 | k3 | 96 MLA (+96 KDA) | latent | 896 | 93 | 7168 | Y | Y | Y | uneven implemented, placed |
| laguna | laguna-s-2.1 | 48 full + 72 slide / 8 kv | shard | 256 | 48 | 3072 | N (72 % 16 = 8) | Y | Y | 48 % 2 = 0, 48 % 4 = 0 |
| muse_glimmer | muse | 32 / 2 | replicate | — | 52 | 6656 | Y | Y | Y | 52 % 4 = 0 |
| minimax | minimax text tower | 64 / 8 | replicate | — | 64 | 5120 | Y | Y | Y | 64 % 4 = 0 |
| qwen4_flash | qwen3flash | 24 / 2 + GDN 16k/48v | replicate | 512 | 48+1 MTP | 2560 | N (24 % 16) | Y | Y | 48 % 2 = 0, 48 % 4 = 0 |
| qwen38_27b | qwen38-27b | 24 / 4 + GDN 16k/48v | replicate | — (dense FFN) | 64+1 MTP | 5120 | N (24 % 16) | Y (geom) | Y | 64 % 4 = 0 |
| qwen38_max | qwen38max | 64 / 4 + GDN 16k/128v | replicate | 512 | 92+1 MTP | 8192 | Y | Y | Y | 92 % 4 = 0 |
| dsv41_flash | dsv41 | 64 | shard | 384 | 40 | 5120 | Y (geom) | Y | Y | 40 % 2 = 0, 40 % 4 = 0 |

Compiled TP pins (MEASURED): `hy4 llm_defines.h` `SPARK_LLM_TP_DEGREE 16u`
with static asserts; `laguna llm_defines.h`
`SPARK_LLM_TENSOR_PARALLEL_DEGREE 8u` + `SPARK_LLM_PIPELINE_STAGE_COUNT 2u`
with static asserts; `qwen38_27b serving_constants.h`
`SPARK_QWEN38_27B_SERVING_TP_DEGREE 4u`; ling header asserts "ling assumes
TP16". No `tp4pp2` tooling exists anywhere on main.

## Placed set and per-node pack bytes (MEASURED — spark0 `du -sb`, 2026-09-20)

| arm (scope) | per-node bytes | matrix verdicts |
|---|---|---|
| ling.bf16.tp16 | 15,727,036,580 | P×16 |
| lingfin.bf16.tp16 (+6,144,582,792 MTP sidecar) | 15,727,036,865 | wave1 placed, S×15 + relay tail |
| ling.bf16.tp16.t1ling | 23 (fixture) | excluded (T1 fixture) |
| k3.mxfp4.tp16 | 99,581,352,352 | P×16 |
| k3.mxfp4.tp4pp4 | 98,768,787,438 | P×16 |
| glm53full.bf16.tp16 / .fp8 / .nvfp4 | 98,021,298,298 / 54,140,235,891 / 32,903,038,976 | P×16 (nvfp4 node 5 M) |
| glm53full.{bf16,fp8,nvfp4}.tp4pp4 | 86,513,071,264 / 46,725,879,800 / 27,474,047,776 | P×16 (addendum) |
| hy4.fp8.tp16 | 56,132,013,159 | S×16 |
| hy4.ud-iq1m.tp16 | 18,724,766,306 | S×16 |
| laguna-s-2.1.bf16.tp8pp2 | 14,438,100,686 | S×16 |
| muse.tp16.bf16 | 3,639,552,874 | S×16 |
| gemma4_26b.tp4pp4.t1 | 3,638,724,586 | P×16 (addendum) |
| gemma4_31b.bf16.tp16 | 3,881,013,057 | wave1 placed, S×15 + relay tail |
| qwen27b.fp8.tp4 | 10,524,422,760 | wave1 placed (8 ranks + replicas) |
| qwen27b.fp8.tp4pp4 / qwen38_27b.tp4pp4 | 4,359,936,294 / 4,359,934,976 | wave1 placed / S×16 |
| qwen38-27b.nvfp4a16.tp4 | 11,800,626,664 | S×16 |
| qwenmax.nvfp4.tp16 (dir `qwenmax.pp16-stripped`) | 96,918,911,434 | S×16 |
| qwen38_max.tp4pp4 | 93,243,510,740 | S×16 |
| qwen3flash.bf16.tp4 (`packs_v4`) | 15,834,725,521 | S×16 |
| qwen3flash.bf16.tp4pp4 / .bf16.tp8 | 41,818,392,805 / 46,333,556,056 | P×16 / P×16 (addendum) |
| qwen3flash.fp8.tp4pp4 / .fp8.tp8 | 42,557,300,889 / 30,518,614,556 | S×16 / **F×14, S×2** |
| qwen3flash.nvfp4.tp4pp4 / .nvfp4.tp8 | 30,965,644,767 / 47,817,944,295 | S×16 / S×16 |
| dsv41flash.mxfp4.tp8 | 39,159,428,195 | P×16 (addendum) |

## Viability matrix (per model×quant; DERIVED per the rules above)

Legend: **P** placed · **S** skipped (viable, not placed) · NV = not viable
(reason: geom / driver / codec / size).

| model×quant | tp16 | tp8 | tp8pp2 | tp4pp4 | tp4 (standalone) |
|---|---|---|---|---|---|
| ling bf16 | P | NV size (63 GB/node > 30) | NV size | NV driver+geom (no pp machinery; 42%4 uneven) | NV size |
| ling fp8 | NV codec (packer bf16-only, wave1 #5) | NV codec | NV codec | NV codec+driver | NV codec |
| ling fin | P | NV size | NV size | NV driver+geom | NV size |
| gemma4-26b bf16 | **S** | NV driver (family tp8 never compiled) | NV driver | P | **S** (14.6 GB/node ≤ 30; family tp4 proven) |
| gemma4-31b bf16 | P | NV driver | NV driver | **S** | **S** (15.5 GB/node; family tp4 proven) |
| gemma4-31b nvfp4 | NV codec (wave1 #3) | NV codec | NV codec | NV codec | NV codec |
| glm53full bf16 / fp8 / nvfp4 | P×3 | NV driver (family serves tp16/tp4pp4 only) | NV driver | P×3 | NV size |
| hy4 fp8 / ud-iq1m | P×2 | NV driver (TP_DEGREE 16u pin) | NV driver | NV driver (no pp machinery) | NV driver |
| k3 mxfp4 | P | NV size (199 GB/node) | NV size | P | NV size |
| laguna bf16 | NV geom (72 % 16) | **S** (28.9 GB/node; tp8+stage machinery compiled) | P | NV driver (TP pin 8u — laguna tp4 never compiled) | NV driver |
| laguna fp8 | NV geom | NV codec (wave1 #7) | NV codec | NV codec | NV codec |
| laguna nvfp4 | NV geom | NV codec (wave1 #8) | NV codec | NV codec | NV codec |
| muse bf16 | P | NV size (29 GB/node — rule edge, family tp8 unproven) | NV driver (no pp machinery) | NV driver (no pp machinery) | NV driver (family tp4 unproven) |
| minimax text bf16 | NV driver (NO-PACKER, wave1 #1) | NV driver | NV driver | NV driver | NV driver |
| qwen3flash bf16 | NV geom | P | **S** (23.2 GB/node) | P | P |
| qwen3flash fp8 | NV geom | P but **defective** (F×14) | **S** (15.3 GB/node) | P | NV size (61 GB/node) |
| qwen3flash nvfp4 | NV geom | P | **S** (23.9 GB/node) | P | NV size (95.7 GB/node) |
| qwen38-27b fp8 | NV geom+driver | NV driver (SERVING_TP_DEGREE 4u pin) | NV driver | P | P |
| qwen38-27b nvfp4a16 | NV geom+driver | NV driver | NV driver | **S** (~3.0 GB/node; family (4,4) placed at fp8) | P |
| qwen38max nvfp4 | P (96.9 GB/node — over nominal 96 GB budget, see defects) | NV size | NV size | P | NV size (373 GB/node) |
| dsv41 mxfp4 | excluded by 09-11 ruling (TP16 packs dropped) | P | **S** (19.6 GB/node — the ruled mesh) | excluded by ruling | NV size (78 GB/node) |
| dsv41 nvfp4 | NV codec (wave1 #4) | NV codec | NV codec | NV codec | NV codec |

Assumption: gemma4-26b is scoped bf16-only — no 26b nvfp4 source is pinned
anywhere on main and the gemma4 packer codec is bf16 (the only nvfp4 warm
source pinned is 31b, wave1 #3).

## SKIPPED — the exact table (10 items)

| # | model×quant | skipped topology | viable-by | est. per-node GB | build effort (packer exists?) |
|---|---|---|---|---|---|
| 1 | dsv41 mxfp4 | tp8×pp2 (16-rank mesh) | geom (40%2) + driver (TP8 module placed; (8,2) is the ruled serving mesh — engram row-shard contract `model_contracts/dsv41_flash_authoritative.json` + `tools/dsv41_flash_engram_shard.py` are on main) + mem | 19.6 (= 8×39.16 GB / 16) | packer exists (`dsv4_stagepack.py` stage spans; `dsv4_tp4_pp4_stagepacks.py` pattern) — cut 16 stage packs from the r2 set and place; engram shards already cut per mesh rank |
| 2 | gemma4-31b bf16 | tp4×pp4 | geom (60%4=15) + driver (family pp4+tp4 proven by placed 26b tp4pp4) + codec + mem | 3.9 | packer exists (tp4pp4 stage machinery in `tools/gemma4_stagepack.py`, 26b plan) — add the 31b stage plan + pp4 module compile |
| 3 | gemma4-26b bf16 | tp16 | geom (q16, 128 experts; kv 8/2 replicate via packer's replication) + driver (family tp16 proven by placed 31b tp16) + codec + mem | 3.6 | packer exists (tp16pp1 topology form in the packer, 31b plan) — add the 26b tp16 plan + tp16 module compile |
| 4 | qwen3flash nvfp4 | tp8×pp2 | geom (48%2) + driver (family tp8 placed; pp stage packs placed at (4,4)) + codec + mem | 23.9 | packer exists (`qwen4_flash_stagepack.py --tp-degree` + stage slicing proven placed) — (8,2) plan + stage compiles |
| 5 | qwen3flash bf16 | tp8×pp2 | geom + driver + mem | 23.2 | same as #4 |
| 6 | qwen3flash fp8 | tp8×pp2 | geom + driver + mem | 15.3 | same as #4 (note: fp8.tp8's 14 corrupt ranks should be repaired first) |
| 7 | qwen38-27b nvfp4a16 | tp4×pp4 | geom (64%4) + driver (family (4,4) placed at fp8) + codec (nvfp4a16 placed at tp4) + mem | 3.0 | packs only — combine the placed nvfp4a16 codec pass with the placed `qwen38_27b_tp4pp4_stagepacks.py` plan |
| 8 | laguna bf16 | tp8 (pp1) | geom (72%8, 48%1) + driver (tp8 + stage machinery compiled; pp1 is a stage-count variant) + codec + mem | 28.9 | packer exists (`laguna_stagepack.py --tp-degree 8 --stage-count 1` runs today) — stage-count-1 module compile + gen_deployment variant (current tool pins tp8×pp2, 16 hosts) |
| 9 | gemma4-26b bf16 | tp4 (standalone, 4 nodes) | size rule (14.6 GB/node ≤ 30) + driver (family tp4 proven) + geom + codec + mem | 14.6 | packer exists — tp4pp1 plan + module compile |
| 10 | gemma4-31b bf16 | tp4 (standalone, 4 nodes) | size rule (15.5 GB/node ≤ 30) + driver + geom + codec + mem | 15.5 | packer exists — tp4pp1 plan + module compile |

**Total skipped: 10** — 8 fleet-wide 16-rank topologies (#1–#8) plus 2
conservative sub-16 standalone additions (#9–#10). Excluding the standalone
additions, the count is **8**.

## Top skipped items by serving value

1. **dsv41 mxfp4 tp8×pp2** — this is not an extra topology, it is the
   operator-ruled serving shape (09-11) that the engram row-shard contract
   is already written against; the fleet only holds the 8 full-model tp8
   rank packs (39.2 GB/node). The mesh halves per-node weights to 19.6 GB
   and completes the ruled deployment.
2. **gemma4-31b bf16 tp4×pp4** — the 31b currently eats all 16 nodes at
   tp16 for 3.9 GB/node; the 26b sibling already serves tp4pp4, so this is
   the cheapest co-residency win in the catalog.
3. **gemma4-26b bf16 tp16** — standard-matrix completion: maximum TP width
   for the MoE decode path; the family already compiles and serves tp16
   (31b).
4. **qwen3flash tp8×pp2 (nvfp4, bf16, fp8)** — qwen3flash is the active
   onboarding serving model; every quant has tp8 and tp4pp4 but the
   standard TP8-geometry middle topology (tp8×pp2) is missing for all
   three; fp8 should wait for the tp8 repair below.
5. **qwen38-27b nvfp4a16 tp4×pp4** — packs-only build (both halves of the
   combination are already placed in other arms).
6. **laguna bf16 tp8** — completes the TP8-geometry standard matrix;
   low-effort (packer flag combination), needs a stage-count-1 compile.

## Placed-with-defects (coverage debt, not skipped)

- `qwen3flash.fp8.tp8`: 14/16 nodes FAIL — placed pack sha256 mismatches
  the placed receipt; re-emit/repair needed before the fp8 tp8 topology
  counts as servable.
- `glm53full.nvfp4.tp16`: node 5 pack MISSING (15/16).
- `qwen27b.tp4` (legacy dir): rank≠0 packs carry the mtp-strip header
  corruption documented in wave1 (`mtp=1 tpdeg=4 tprank=0`); superseded by
  the clean `qwen27b.fp8.tp4`.
- `qwen27b.tp4pp4`: packs on nodes 0–3 only; `qwen38max.tp4`: phantom arm
  (matrix row, zero bytes on all 16 nodes) — and tp4 is not a viable
  qwen38max topology in any case (373 GB/node).
- `qwenmax.nvfp4.tp16` at 96.9 GB/node + ~15 GB runtime exceeds the nominal
  96 GB budget — it is placed and verified, so the de facto budget
  accommodates it, but no larger per-node arm is admissible.

## Excluded from the survey (per scope)

- **glm53flash coredev family** (glm5_next): bf16/fp8/nvfp4 tp16, fp8 tp8,
  fp8 tp4pp4 arms.
- **Drafter/dspark variants**: `qwen38-dflash2-drafter`, `dspark` trees,
  dspark draft arms.
- **T1 reference fixtures**: `ling.bf16.tp16.t1ling` (23 bytes).
- **Wave2 in-flight arms** (present on the fleet, unattested on main,
  single-rank staging seen on spark0): `ling.fp8.tp16`,
  `minimax.text.bf16.tp4`, `gemma4_31b.nvfp4.tp16`. The wave2 lane closes
  the codec gaps that make ling fp8, minimax, gemma4-31b nvfp4, laguna
  fp8/nvfp4 and dsv41 nvfp4 "not viable" in this main-based survey; until
  it lands, those cells are NV(codec), not skipped.
- **dsv4 (v4 flash/pro) family arms** (`dsv4flash.tp16`, `dsv4flash.tp4pp4`,
  `dsv4_pro.tp16`, `dsv4_pro.tp4pp4`) — out of scope (scope is dsv41).
- Legacy/ambiguous dirs: `qwenmax.pp16` (empty), `qwen4_flash.tp4` naming
  (the v4 deployment spans 16 nodes; packs under `packs_v4`),
  `qwen27b.tp4` (see defects), mimo25.

## Ledger regen

Run on the PR branch after adding this file:
`generate_package_manifest.py` && `generate_sha256sums.py` &&
`verify_package_manifest.py` — receipt below.
