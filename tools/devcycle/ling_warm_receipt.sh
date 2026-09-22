#!/bin/bash
# ling timed warm receipt (lane 9 M3): the fleet-standard cold-launch
# numbers (dsv4_pro M3 shape, PR #1122 RUN-WALL convention; k3 receipt
# PR #1131's tool) on the shared weightd, for the committed
# repetition-head preload set:
#
#   1. daemon cold-warm    - first warm after a daemon restart: lazy-arena
#                            create + pool-chunk materialization dominate
#                            (the once-per-daemon cost)
#   2. warm-daemon preload - a cold-launching instance's batch preload when
#                            the daemon already holds the set resident:
#                            WSET-WARM elapsed_ms. THE < 5 s claim.
#   3. steady-state        - repeated preloads against the warm daemon
#
# ling specifics vs the k3 receipt:
#   - no per-rank wset split: ling TP16 col-shards the expert inter
#     dimension, so every rank pack holds EVERY (layer, expert) pair's
#     shard (chunk-union 16, PR #1144) - the committed full-model wset
#     applies as-is on every node.
#   - the warm presents the ling module arena identity via --family ling
#     (model tag "ling_stage", PR #1146): revision = the stage config
#     model_revision, topology = tp_degree 16, arena bytes = the pack
#     size per the daemon's size-mismatch contract (WDATTACH).
#
# Bases: smoke_set_raw_bytes_per_node (exact expert span bytes of the
# keys) and smoke_set_chunked_bytes_per_node (ceil-per-span at the pool
# chunk size, the fleet receipt basis), with the exact mapped-union
# figure alongside (pool chunk = 2 MiB measured on sm_121a).
#
# Queue invocation (bare repo-resident path): --cmd 'bash
# tools/devcycle/ling_warm_receipt.sh' --per-node --nodes spark0,...,sparkf
#
# Env:
#   SPARK_WEIGHTD_SOCKET   the shared daemon socket (required)
#   LING_WSET              working set (default: the committed
#                          model-families/ling/smoke-ling-v1.wset)
#   LING_POOL_BYTES        expert pool bound (default 1073741824: the
#                          preload set is 966.1 MiB raw per node)
#   LING_WARM_RUNS         timed runs (default 5: 1 cold + 1 claim + 3
#                          steady)
#   LING_POOL_CHUNK_BYTES  chunk basis (default 2097152; measured sm_121a)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOCKET="${SPARK_WEIGHTD_SOCKET:?shared weightd socket required}"
WSET="${LING_WSET:-$REPO/model-families/ling/smoke-ling-v1.wset}"
POOL="${LING_POOL_BYTES:-1073741824}"
RUNS="${LING_WARM_RUNS:-5}"
CHUNK="${LING_POOL_CHUNK_BYTES:-2097152}"
REVISION="e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"

HOST="$(hostname)"
case "$HOST" in
spark*) RANK=$((16#${HOST#spark})) ;;
*) echo "unexpected host: $HOST" >&2; exit 2 ;;
esac
ROOT="/home/$HOST/sparkdata/ling.bf16.tp16"
PACK="$ROOT/packs/ling.bf16.tp16.rank$RANK.sp"
SHA="$(cut -d' ' -f1 "$PACK.sha256")"
MANIFEST="$PACK.experts"
for required in "$PACK" "$PACK.sha256" "$MANIFEST" "$WSET"; do
  [ -f "$required" ] || { echo "missing: $required" >&2; exit 2; }
done

cd "$REPO"
make -s build/weightd_warm

echo "== ling warm receipt (rank $RANK, $PACK)"
echo "== pool=$POOL chunk=$CHUNK runs=$RUNS wset=$WSET"
SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
  build/weightd_warm "$SOCKET" "$PACK" "$SHA" "$REVISION" 16 \
    --family ling --identity-print

python3 - "$MANIFEST" "$WSET" "$CHUNK" <<'PYBASE'
import struct, sys

manifest_path, wset_path, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
if chunk <= 0:
    raise SystemExit("chunk basis must be positive")
data = open(manifest_path, "rb").read()
if len(data) < 16:
    raise SystemExit("manifest shorter than header")
magic, version, count, zero = struct.unpack_from("=IIII", data, 0)
if zero != 0 or count == 0 or len(data) != 16 + 48 * count:
    raise SystemExit(f"malformed manifest: magic=0x{magic:08x} version={version} "
                     f"count={count} size={len(data)}")
ranges = {}
offset = 16
for _ in range(count):
    layer, expert, kind, reserved, range_offset, range_bytes = \
        struct.unpack_from("=IIIIQQ", data, offset)
    offset += 48
    if reserved != 0:
        raise SystemExit("nonzero reserved field in range record")
    ranges.setdefault((layer, expert), []).append((range_offset, range_bytes))
keys = set()
wset = open(wset_path, "rb").read()
if len(wset) == 0 or len(wset) % 8 != 0:
    raise SystemExit("wset must be nonempty complete key pairs")
for index in range(0, len(wset), 8):
    keys.add(struct.unpack_from("<II", wset, index))
missing = sorted(key for key in keys if key not in ranges)
if missing:
    raise SystemExit(f"wset keys absent from manifest: {missing[:4]}")
raw = ceil_span = 0
chunks = set()
for key in keys:
    for range_offset, range_bytes in ranges[key]:
        raw += range_bytes
        ceil_span += -(-range_bytes // chunk) * chunk
        chunks.update(range(range_offset // chunk,
                            (range_offset + range_bytes - 1) // chunk + 1))
union = len(chunks) * chunk
mib = 1024 * 1024
print(f"WARM-BASES keys={len(keys)} chunk_bytes={chunk}")
print(f"smoke_set_raw_bytes_per_node={raw} ({raw / mib:.1f} MiB)")
print(f"smoke_set_chunked_bytes_per_node={ceil_span} "
      f"({ceil_span / mib:.1f} MiB, ceil-per-span basis)")
print(f"smoke_set_chunked_union_bytes_per_node={union} "
      f"({union / mib:.1f} MiB, exact mapped union)")
print(f"chunk_inflation={ceil_span / raw:.2f}x "
      "(lazy/partial pool tier only; pin-all whole-arena pays no chunking)")
PYBASE

echo "== timed warm (run 1 = daemon-cold-warm, run 2 = warm-daemon preload"
echo "   [the < 5 s claim], runs 3+ = steady-state)"
echo "   ling's full head exceeds the client's 512-key lease-group cap on"
echo "   every rank (chunk-union 16: all layers per rank pack), so each run"
echo "   warms the shard sequence through the same socket/arena; RUN-WALL"
echo "   covers the WHOLE sequence (connect + attach/arena-create +"
echo "   acquire + release + close per shard, shell monotonic) and"
echo "   WSET-SUM is the sum of the tool's internal WSET-WARM elapsed_ms."
SHARD_PREFIX="$(mktemp -u /tmp/ling-warm-rank$$-XXXXXX)"
python3 "$REPO/tools/ling_wset_split.py" "$WSET" "$SHARD_PREFIX" > "$SHARD_PREFIX.list"
SHARDS="$(wc -l < "$SHARD_PREFIX.list")"
echo "WSET-SHARDS count=$SHARDS"
index=1
while [ "$index" -le "$RUNS" ]; do
  echo "-- warm run $index/$RUNS"
  began_ns=$(date +%s%N)
  wset_sum_ms=0
  while read -r shard keys; do
    SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
      build/weightd_warm "$SOCKET" "$PACK" "$SHA" "$REVISION" 16 \
      --family ling --wset "$shard" 300 2>&1 | tee "$SHARD_PREFIX.warm.$$"
    shard_ms="$(sed -n 's/.*WSET-WARM keys=[0-9]* elapsed_ms=\([0-9]*\).*/\1/p' "$SHARD_PREFIX.warm.$$")"
    wset_sum_ms=$(( wset_sum_ms + ${shard_ms:-0} ))
    rm -f "$SHARD_PREFIX.warm.$$"
  done < "$SHARD_PREFIX.list"
  ended_ns=$(date +%s%N)
  printf 'RUN-WALL run=%s elapsed_ms=%s wset_sum_ms=%s\n' "$index" \
    "$(( (ended_ns - began_ns) / 1000000 ))" "$wset_sum_ms"
  index=$((index + 1))
done
rm -f "$SHARD_PREFIX".[0-9]* "$SHARD_PREFIX.list"
echo "WARM-RECEIPT-DONE rank=$RANK pool=$POOL chunk=$CHUNK runs=$RUNS shards=$SHARDS"
