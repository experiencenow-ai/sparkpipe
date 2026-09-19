# HY4-T1 slot-4 window receipt — 2026-09-17 16:25-16:34Z

Claim GPU_ALLOCATION.md slot 4, MESH_LEASE ACTIVE/RELEASED same window.
Stack built from main 4bffe5d (PR #1033 mesh delivery fix) via lane merge
33df819; harnesses compiled on sparkf from lane/hy4-t1 @ 2e598c3
(tools/hy4_window.sh build).

## Deliverable: attach-evidence + stub boundary (NOT a T1 verdict)

The hy4 module Execute is a fail-closed stub; this window proves the v2
chain end-to-end into the shared weightd and pins the stub boundary.

## v2 lazy-attach (SparkWeightdLazyPackCreateChecked, shared weightd)

- 12/16 ranks PASS (11 first-pass + rank-14 on immediate retry; its
  io_error at spine.c:228 did not reproduce = transient read, no defect).
- Per-rank receipt: pack path, arena bytes 56131321268/9, experts sidecar
  236560 B, pack sha = the placed emission sha, CreateChecked status=0.
- The daemon parsed the WEPX version-2 sidecars (4928 ranges, contract
  SPARK_WEIGHTD_RANGE_MANIFEST_VERSION=2) and mapped the pack arenas.

## Capacity fails (4/16, next-window retry, no code change implied)

spark5 / spark7 / sparka / sparkf: CAPACITY_EXCEEDED at
spark_weightd_lazy_pack.c:62 = cudaMalloc of the ~9 GB spine inside the
daemon context failed. Memory polled right after: 73-75 GB of 121 in use,
sparkf at 5 GB free - co-tenant pressure (production glm53 residentd +
lane residue), not a hy4 defect. Next window: retry the 4 ranks after a
co-tenant check.

## Execute stub boundary 16/16

- initialize status=0 OK (adapter init + tp collective stub) on all 16.
- execute status=19 SPARK_STATUS_UNSUPPORTED fail-closed on all 16
  (11 in the wave logs, 5 in rank{5,7,10,14,15}-stub.log after their
  attach failed set -e before the stub ran).
- destroy done; harnesses exited; zero residual processes; the shared
  weightd daemons were never restarted and own all GPU state.

## Logs

runs/hy4-window/rank{0..15}.log (wave), rank{5,7,10,14,15}-stub.log
(makeup stubs), this file. Hy4 window tooling: tools/hy4_window.sh,
tools/hy4_stub_harness.c, tools/hy4_attach_harness.c.
