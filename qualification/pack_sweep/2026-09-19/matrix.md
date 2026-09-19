# Fleet-wide placed-stagepack sweep — 2026-09-19

Every placed stagepack arm under `/home/spark*/sparkdata/` was checked on
each of the 16 nodes against local NVMe pack content only: the family's
verify tool where one runs without a warm checkpoint, else the placed
pack's sha256 against the placed receipt/sidecar. No ceph/warm reads,
no GPU. One verify at a time per node, `nice 10`.

## Verdict counts

- PASS — tool (`P`): 126
- PASS — sha vs placed receipt (`S`): 337
- FAIL (`F`): 30
- NO-TOOL (`NT`): 114
- MISSING (`M`): 33

## Matrix (arm × node)

`P` PASS via family tool · `S` PASS via sha256 vs placed receipt ·
`F` FAIL · `NT` NO-TOOL · `M` MISSING · `—` no report

| arm | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | a | b | c | d | e | f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ling.bf16.tp16 | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| ling.bf16.tp16.t1ling | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| k3.mxfp4.tp16 | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| k3.mxfp4.tp4pp4 | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| glm53full.bf16.tp16 | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| glm53full.fp8.tp16 | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P | P |
| glm53full.nvfp4.tp16 | P | P | P | P | P | M | P | P | P | P | P | P | P | P | P | P |
| glm53full.bf16.tp4pp4 | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT |
| glm53full.fp8.tp4pp4 | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT |
| glm53full.nvfp4.tp4pp4 | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT |
| glm53flash.bf16.tp16 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| glm53flash.fp8.tp16 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | F |
| glm53flash.nvfp4.tp16 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| glm53flash.fp8.tp8 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| glm53flash.fp8.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen3flash.bf16.tp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen3flash.bf16.tp4pp4 | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT |
| qwen3flash.bf16.tp8 | NT | NT | NT | NT | S | S | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT |
| qwen3flash.fp8.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen3flash.fp8.tp8 | F | S | S | F | S | F | F | F | F | F | F | F | F | F | F | F |
| qwen3flash.nvfp4.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen3flash.nvfp4.tp8 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen4_flash.tp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwenmax.nvfp4.tp16 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| gemma4_26b.tp4pp4.t1 | P | S | S | S | P | S | S | S | S | S | S | S | P | S | S | S |
| hy4.fp8.tp16 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| hy4.ud-iq1m.tp16 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| laguna-s-2.1.bf16.tp8pp2 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| muse.tp16.bf16 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen27b.tp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen27b.tp4pp4 | NT | NT | NT | NT | M | M | M | M | M | M | M | M | M | M | M | M |
| qwen38_27b.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen38-27b.nvfp4a16.tp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen38_max.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| qwen38max.tp4 | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M | M |
| dsv4flash.tp16 | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT |
| dsv4flash.tp4pp4 | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S | S |
| dsv4_pro.tp16 | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F | F |
| dsv4_pro.tp4pp4 | P | M | M | M | P | P | M | P | P | P | P | P | P | P | P | P |
| dsv41flash.mxfp4.tp8 | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT | NT |

## FAIL cells

- `glm53flash.fp8.tp16` node f: sha256 f0e94cd3ef709fa1... matches none of glm53flash.fp8.tp16.rankf.sp.receipt.json,glm53flash.fp8.tp16.rankf.sp.sha256
- `qwen3flash.fp8.tp8` node 0: sha256 b6f780982db8f7e7... matches none of qwenflash.tp8.fp8.rank0.spstage.receipt.json,qwenflash.tp8.fp8.rank0.spstage.sha256
- `qwen3flash.fp8.tp8` node 3: sha256 3da5bdcf24203b79... matches none of qwenflash.tp8.fp8.rank03.spstage.receipt.json,qwenflash.tp8.fp8.rank03.spstage.sha256
- `qwen3flash.fp8.tp8` node 5: sha256 353eb9a5a61a4d39... matches none of qwenflash.tp8.fp8.rank05.spstage.receipt.json,qwenflash.tp8.fp8.rank05.spstage.sha256
- `qwen3flash.fp8.tp8` node 6: sha256 659ad3b42069770b... matches none of qwenflash.tp8.fp8.rank06.spstage.receipt.json,qwenflash.tp8.fp8.rank06.spstage.sha256
- `qwen3flash.fp8.tp8` node 7: sha256 16a36127301f169e... matches none of qwenflash.tp8.fp8.rank07.spstage.receipt.json,qwenflash.tp8.fp8.rank07.spstage.sha256
- `qwen3flash.fp8.tp8` node 8: sha256 b6f780982db8f7e7... matches none of qwenflash.tp8.fp8.rank0.spstage.receipt.json,qwenflash.tp8.fp8.rank0.spstage.sha256
- `qwen3flash.fp8.tp8` node 9: sha256 a23344080b0f0fed... matches none of qwenflash.tp8.fp8.rank01.spstage.receipt.json,qwenflash.tp8.fp8.rank01.spstage.sha256
- `qwen3flash.fp8.tp8` node a: sha256 1add7b836b2d2925... matches none of qwenflash.tp8.fp8.rank02.spstage.receipt.json,qwenflash.tp8.fp8.rank02.spstage.sha256
- `qwen3flash.fp8.tp8` node b: sha256 3da5bdcf24203b79... matches none of qwenflash.tp8.fp8.rank03.spstage.receipt.json,qwenflash.tp8.fp8.rank03.spstage.sha256
- `qwen3flash.fp8.tp8` node c: sha256 1a82c5cc46137882... matches none of qwenflash.tp8.fp8.rank04.spstage.receipt.json,qwenflash.tp8.fp8.rank04.spstage.sha256
- `qwen3flash.fp8.tp8` node d: sha256 353eb9a5a61a4d39... matches none of qwenflash.tp8.fp8.rank05.spstage.receipt.json,qwenflash.tp8.fp8.rank05.spstage.sha256
- `qwen3flash.fp8.tp8` node e: sha256 659ad3b42069770b... matches none of qwenflash.tp8.fp8.rank06.spstage.receipt.json,qwenflash.tp8.fp8.rank06.spstage.sha256
- `qwen3flash.fp8.tp8` node f: sha256 16a36127301f169e... matches none of qwenflash.tp8.fp8.rank07.spstage.receipt.json,qwenflash.tp8.fp8.rank07.spstage.sha256
- `dsv4_pro.tp16` node 0: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 1: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 2: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 3: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 4: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 5: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 6: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 7: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 8: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node 9: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node a: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node b: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node c: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node d: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node e: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)
- `dsv4_pro.tp16` node f: rc=2 FAIL: header geometry fields disagree: (61, 7168, 129280, 384, 0)

## NO-TOOL / MISSING gap list

- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes 0,1,2,3,4,5,6,7: pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes 8: single local pack rank0.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes 9: single local pack rank1.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes a: single local pack rank2.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes b: single local pack rank3.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes c: single local pack rank4.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes d: single local pack rank5.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes e: single local pack rank6.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv41flash.mxfp4.tp8` [NO-TOOL] nodes f: single local pack rank7.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4_pro.tp4pp4` [MISSING] nodes 1,2,3,6: local rank pack absent on this node
- `dsv4flash.tp16` [NO-TOOL] nodes 0: pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 2: single placed pack dsv4flash.tp16.rank1.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes b: single placed pack dsv4flash.tp16.rank10.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes c: single placed pack dsv4flash.tp16.rank11.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes d: single placed pack dsv4flash.tp16.rank12.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes e: single placed pack dsv4flash.tp16.rank13.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes f: single placed pack dsv4flash.tp16.rank14.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 1: single placed pack dsv4flash.tp16.rank15.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 3: single placed pack dsv4flash.tp16.rank2.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 4: single placed pack dsv4flash.tp16.rank3.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 5: single placed pack dsv4flash.tp16.rank4.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 6: single placed pack dsv4flash.tp16.rank5.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 7: single placed pack dsv4flash.tp16.rank6.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 8: single placed pack dsv4flash.tp16.rank7.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes 9: single placed pack dsv4flash.tp16.rank8.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `dsv4flash.tp16` [NO-TOOL] nodes a: single placed pack dsv4flash.tp16.rank9.spstage; rank label placement-mapped, not node-indexed; pack present; no family tool usable locally and no sha receipt placed
- `gemma4_26b.tp4pp4.t1` [NO-TOOL] nodes 8: sha verified; pack present; stage span receipt unreadable
- `gemma4_26b.tp4pp4.t1` [NO-TOOL] nodes 1,2,3,5,6,7,9,a,b,d,e,f: sha verified; pack present; verify-existing plans tp rank 0 only
- `glm53full.bf16.tp4pp4` [NO-TOOL] nodes 0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f: validator inventories the full 78-layer span; placed rank packs are per-stage spans
- `glm53full.fp8.tp4pp4` [NO-TOOL] nodes 0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f: validator inventories the full 78-layer span; placed rank packs are per-stage spans
- `glm53full.nvfp4.tp16` [MISSING] nodes 5: local rank pack absent on this node
- `glm53full.nvfp4.tp4pp4` [NO-TOOL] nodes 0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f: validator inventories the full 78-layer span; placed rank packs are per-stage spans
- `qwen27b.tp4pp4` [MISSING] nodes 4,5,6,7,8,9,a,b,c,d,e,f: local rank pack absent on this node
- `qwen27b.tp4pp4` [NO-TOOL] nodes 0,1,2,3: pack present; no family tool usable locally and no sha receipt placed
- `qwen38max.tp4` [MISSING] nodes 0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f: local rank pack absent on this node
- `qwen3flash.bf16.tp4pp4` [NO-TOOL] nodes 0,1,2,3,4,5,6,7,8,9,a,b,c,d,e,f: pack present; no family tool usable locally and no sha receipt placed
- `qwen3flash.bf16.tp8` [NO-TOOL] nodes 0,1,2,3,6,7,8,9,a,b,c,d,e,f: pack present; no family tool usable locally and no sha receipt placed
