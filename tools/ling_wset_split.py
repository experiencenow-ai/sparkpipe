#!/usr/bin/env python3
"""Split a ling smoke working set into weightd_warm-legal shards.

The weightd client's lease-group ceiling is 512 keys per acquire
(SPARK_WEIGHTD_LEASE_GROUPS_MAX, enforced by tools/weightd_warm.c's
wset reader), and ling's chunk-union-16 geometry col-shards the expert
inter dimension across every rank: each rank pack manifests ALL routed
layers, so unlike k3 (per-PP-stage subsets fit under the cap) the
lane-9 preload is the FULL model-families/ling/smoke_experts.json head
on every node - 1,374 keys at the >= 2-of-21 cut. The sanctioned shape
is sequential shard warms through the same socket: N invocations of
weightd_warm, each with a sorted shard of at most 512 deduplicated
u32 pairs; the arena identity is identical across shards so every
acquire lands in the same resident arena.

Usage:
  ling_wset_split.py WSET SHARD_PREFIX   -> writes SHARD_PREFIX.000..NNN
  (prints one line per shard: path keys)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

SHARD_KEYS_MAX = 512


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    prefix = Path(sys.argv[2])
    raw = source.read_bytes()
    if len(raw) == 0 or len(raw) % 8 != 0:
        raise SystemExit("wset must be nonempty complete key pairs")
    pairs = sorted({struct.unpack_from("<II", raw, index)
                    for index in range(0, len(raw), 8)})
    if not pairs:
        raise SystemExit("wset has no keys")
    shards = [pairs[start:start + SHARD_KEYS_MAX]
              for start in range(0, len(pairs), SHARD_KEYS_MAX)]
    for index, shard in enumerate(shards):
        output = Path(f"{prefix}.{index:03d}")
        with output.open("wb") as handle:
            for layer, expert in shard:
                handle.write(layer.to_bytes(4, "little"))
                handle.write(expert.to_bytes(4, "little"))
        print(f"{output} {len(shard)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
