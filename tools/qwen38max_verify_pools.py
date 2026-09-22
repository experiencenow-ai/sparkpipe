#!/usr/bin/env python3
"""Verify and measure this node's smoke-expert pool from its placed pack.

Reads the .experts sidecar of one operator-placed qwenmax.nvfp4.tp16 rank
pack (WEPX format: magic, u32 count at offset 8, 48-byte records of
layer/expert/kind u32 + pad + arena offset/bytes u64), intersects it with
the smoke working set (model-families/qwen38_max/smoke_experts.json), and
emits BOTH byte bases the M3 pool math needs:

  raw      sum of span bytes of every smoke expert on this rank
  chunked  size of the 2 MiB chunk union covering those spans — the same
           arithmetic the weightd lazy tier materializes at (VmmReserve:
           GRANULARITY_MINIMUM with a 2 MiB floor, runtime/spark_weightd.c)

Pin-all (GRAPH_PATH + PIN_EXPERTS) pays no chunking; lazy/partial attach
budgets the chunked number. The ratio is family-specific (qwen38_max
~1.40-1.44x; do not borrow another lane's factor).

Runs bare inside a per-node CPU queue job; rank defaults to this host's
suffix (one pack per node: rank i on spark{hex(i)}). Output: one JSON
line on stdout (the job log is the receipt) and, with --out, the same
object written for collection.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

CHUNK_BYTES = 2 * 1024 * 1024
RECORD_BYTES = 48
PACK_TEMPLATE = ("/home/{host}/sparkdata/qwenmax.nvfp4.tp16/packs/"
                 "qwenmax.nvfp4.tp16.rank{rank}.sp")
DEFAULT_WSET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model-families", "qwen38_max", "smoke_experts.json")


def node_rank(host: str) -> int:
    suffix = host.removeprefix("spark")
    if not suffix or len(suffix) > 2 or any(c not in "0123456789abcdef" for c in suffix):
        raise SystemExit("cannot derive rank from hostname %r" % host)
    return int(suffix, 16)


def measure(sidecar: bytes, wanted: set) -> dict:
    if sidecar[:4] != b"WEPX":
        raise SystemExit("bad sidecar magic (expected WEPX)")
    count = struct.unpack_from("<I", sidecar, 8)[0]
    if len(sidecar) < 16 + count * RECORD_BYTES:
        raise SystemExit("truncated sidecar: %d records declared, %d bytes present"
                         % (count, len(sidecar)))
    chunks: set = set()
    raw = 0
    spans = 0
    experts: set = set()
    for i in range(count):
        layer, expert, _kind = struct.unpack_from("<III", sidecar, 16 + i * RECORD_BYTES)
        off, by = struct.unpack_from("<QQ", sidecar, 16 + i * RECORD_BYTES + 16)
        if (layer, expert) not in wanted:
            continue
        raw += by
        spans += 1
        experts.add((layer, expert))
        chunks.update(range(off // CHUNK_BYTES, (off + by - 1) // CHUNK_BYTES + 1))
    return {
        "experts": len(experts),
        "spans": spans,
        "raw": raw,
        "chunked": len(chunks) * CHUNK_BYTES,
        "factor": round(len(chunks) * CHUNK_BYTES / raw, 4) if raw else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rank", type=int, default=None,
                        help="TP rank (default: derived from this host)")
    parser.add_argument("--pack", default=None,
                        help="rank pack path (default: this node's placed pack)")
    parser.add_argument("--wset", default=DEFAULT_WSET,
                        help="smoke working-set manifest (default: family copy)")
    parser.add_argument("--out", default=None,
                        help="also write the result JSON here")
    arguments = parser.parse_args()

    host = os.uname().nodename
    rank = arguments.rank if arguments.rank is not None else node_rank(host)
    if not 0 <= rank < 16:
        raise SystemExit("rank out of range: %d" % rank)
    pack = arguments.pack or PACK_TEMPLATE.format(host=host, rank=rank)
    sidecar_path = pack + ".experts"
    if not os.path.isfile(sidecar_path):
        raise SystemExit("sidecar not found: %s" % sidecar_path)

    working_set = json.load(open(arguments.wset, encoding="utf-8"))
    wanted = {(int(e["layer"]), int(e["expert"])) for e in working_set["experts"]}

    result = measure(open(sidecar_path, "rb").read(), wanted)
    if result["experts"] == 0:
        raise SystemExit("no smoke experts found on rank %d sidecar" % rank)
    result.update({
        "rank": rank,
        "host": host,
        "pack": pack,
        "chunk_bytes": CHUNK_BYTES,
        "measured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    line = json.dumps(result, sort_keys=True)
    print(line)
    if arguments.out:
        with open(arguments.out, "w", encoding="utf-8") as handle:
            handle.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
