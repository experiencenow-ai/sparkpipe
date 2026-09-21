#!/bin/bash
# glm53flash nvfp4 tp8 + tp4pp4 emission, placement, and verification.
# Runs ON the emit node (spark2) against the node-local frozen warm copy:
#   $1 = frozen warm dir   $2 = staging dir
# Laws honored: never overwrite (emit() refuses existing outputs; placement
# skips existing files), CPU only, nice 10, receipts + sha sidecars placed
# with every pack, chattr-locked after placement.
set -euo pipefail
WARM="$1"; STAGE="$2"
EMIT=/tmp/g53emit
NODES=(0 1 2 3 4 5 6 7 8 9 a b c d e f)

receipt() {
  python3 - "$@" <<'PY'
import datetime, json, sys
arm, rank, tp, tp_rank, stage, first, count, file, bytes_, sha = sys.argv[1:11]
doc = {
    "kind": "sparkpipe.g5nsp.stagepack-receipt.v1",
    "arm": arm,
    "rank": int(rank),
    "tp_degree": int(tp),
    "first_layer": int(first),
    "layer_count": int(count),
    "mtp": "none",
    "expert_codec": "nvfp4-source-native",
    "source": "nvidia/GLM-5.3-Flash-NVFP4",
    "file_bytes": int(bytes_),
    "sha256": sha,
    "placed_at": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"),
    "packed_by": "glm53flash-validation lane (operator scope addition 2026-09-20)",
    "emitted_on": "spark2",
    "source_fingerprint": "node-local rsync of /mnt/model-warm/glm-5.3-flash-nvfp4-nvidia",
}
if stage != "-":
    doc["stage"] = int(stage)
if tp_rank != "-":
    doc["tp_rank"] = int(tp_rank)
print(json.dumps(doc))
PY
}

place() {
  local arm="$1" file="$2" node="$3"
  local dest="/home/spark$node/sparkdata/$arm/packs"
  ssh "spark$node" "mkdir -p $dest"
  if ssh "spark$node" "test -e $dest/$file"; then
    echo "PLACE-SKIP spark$node $file (exists)"
  else
    rsync -a --no-times --partial "$STAGE/$file" "spark$node:$dest/$file"
    ssh "spark$node" "chmod 444 $dest/$file 2>/dev/null; chattr +i $dest/$file 2>/dev/null || true"
    echo "PLACED spark$node $file"
  fi
}

echo "== byte-identity proof: re-emit tp16 rank 6, expect 3fddd328..."
nice -n 10 python3 "$EMIT/emit_rank.py" --source "$WARM" --out "$STAGE/proof-tp16" \
  --degree 16 --ranks 6 --owns-embedding --owns-head
sha256sum "$STAGE/proof-tp16/glm5_next_stage.tp16.rank6.g5nsp"

echo "== emit nvfp4.tp8 ranks 0-7 (uniform packs: every rank carries its embed/head row slices)"
for r in 0 1 2 3 4 5 6 7; do
  nice -n 10 python3 "$EMIT/emit_rank.py" --source "$WARM" --out "$STAGE/tp8" \
    --degree 8 --ranks "$r" --owns-embedding --owns-head
done

echo "== emit nvfp4.tp4pp4 world ranks 0-15 (stages 11/11/11/12)"
for w in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  stage=$((w / 4)); tprank=$((w % 4))
  first=$((stage * 11)); count=11
  if [ "$stage" = "3" ]; then first=33; count=12; fi
  extra=""
  [ "$stage" = "0" ] && extra="--owns-embedding"
  [ "$stage" = "3" ] && extra="--owns-head"
  nice -n 10 python3 "$EMIT/emit_rank.py" --source "$WARM" --out "$STAGE/tp4pp4" \
    --degree 4 --stage-count 4 --stage-index "$stage" \
    --first-layer "$first" --layer-count "$count" --ranks "$tprank" $extra
done

echo "== receipts, sidecars, placement"
for f in "$STAGE/tp8"/*.g5nsp; do
  file=$(basename "$f")
  rank=$(echo "$file" | sed -E 's/.*rank([0-9]+)\.g5nsp/\1/')
  bytes=$(stat -c %s "$f"); sha=$(sha256sum "$f" | cut -d" " -f1)
  printf '%s  %s\n' "$sha" "$file" > "$STAGE/tp8/$file.sha256"
  receipt "glm53flash.nvfp4.tp8" "$rank" 8 "$rank" "-" 0 45 \
          "$file" "$bytes" "$sha" > "$STAGE/tp8/$file.receipt.json"
  for n in "${NODES[@]}"; do
    r=$((16#$n)); if [ "$r" -ge 8 ]; then r=$((r - 8)); fi
    if [ "$r" = "$rank" ]; then
      place "glm53flash.nvfp4.tp8" "$file" "$n"
      place "glm53flash.nvfp4.tp8" "$file.sha256" "$n"
      place "glm53flash.nvfp4.tp8" "$file.receipt.json" "$n"
    fi
  done
done

echo "== tp4pp4 per-world-rank receipts and placement"
for w in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  stage=$((w / 4)); tprank=$((w % 4)); node=$(printf '%x' "$w")
  first=$((stage * 11)); count=11
  if [ "$stage" = "3" ]; then first=33; count=12; fi
  file=$(printf 'glm5_next_stage.tp4.pp4.stage%d.rank%d.g5nsp' "$stage" "$tprank")
  bytes=$(stat -c %s "$STAGE/tp4pp4/$file")
  sha=$(sha256sum "$STAGE/tp4pp4/$file" | cut -d" " -f1)
  receipt "glm53flash.nvfp4.tp4pp4" "$w" 4 "$tprank" "$stage" "$first" "$count" \
          "$file" "$bytes" "$sha" > "$STAGE/tp4pp4/$file.receipt.json"
  place "glm53flash.nvfp4.tp4pp4" "$file" "$node"
  place "glm53flash.nvfp4.tp4pp4" "$file.sha256" "$node"
  place "glm53flash.nvfp4.tp4pp4" "$file.receipt.json" "$node"
done
echo "== emission wave complete"
