#!/bin/bash
# ling_decode_receipt.sh — lane-9 first-decode + T1R exactness receipt
# (M3/M4). Runs INSIDE one queue job on the COORDINATOR NODE (spark0)
# after the 16-node attach job is up: starts the lane API against the
# private deployment the wrapper staged, decodes the committed smoke
# prompts, and compares the emitted token ids against the committed T1R
# fixtures BYTE FOR BYTE (the qualification/t1_reference/ling set, PR
# #1021) - the exactness gate for the b128 bucketed-variant driver.
#
# Sequence: residentd must already be running (the attach job holds the
# 16 control endpoints); this script only touches spark0's endpoint and
# the lane session block's API port 23776 (reserved by the attach job's
# --ports).
#
# Env:
#   SPARK_QUEUE_RUNTIME_ROOT  the ATTACH job's runtime root on spark0
#                             (deployment.json + config/ live there)
#   LING_FIRMWARE_ROOT        firmware dir with sparkpipe_model_api
#                             (default /home/spark0/sparkdata/
#                             ling-lane9-firmware)
#   LING_API_PORT             default 23776 (lane-9 session block tail)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${SPARK_QUEUE_RUNTIME_ROOT:?the attach job runtime root is required}"
FIRMWARE="${LING_FIRMWARE_ROOT:-/home/$(hostname)/sparkdata/ling-lane9-firmware}"
API_PORT="${LING_API_PORT:-23776}"

[ "$(hostname)" = "spark0" ] || { echo "runs on the coordinator rank only" >&2; exit 2; }
for artifact in "$FIRMWARE/sparkpipe_model_api" "$ROOT/deployment.json" \
    "$ROOT/config/adapter.json"; do
  [ -e "$artifact" ] || { echo "missing: $artifact" >&2; exit 2; }
done

API_ROOT="$(mktemp -d /tmp/ling-api-XXXXXX)"
cleanup() { [ -n "${API_PID:-}" ] && kill -TERM "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT
LD_LIBRARY_PATH="$FIRMWARE" "$FIRMWARE/sparkpipe_model_api" \
  --deployment "$ROOT/deployment.json" \
  --runtime-root "$ROOT" \
  --port "$API_PORT" > "$API_ROOT/api.log" 2>&1 &
API_PID=$!
for _ in $(seq 1 60); do
  grep -q "model_api ready" "$API_ROOT/api.log" 2>/dev/null && break
  kill -0 "$API_PID" 2>/dev/null || { echo "api exited early" >&2; exit 2; }
  sleep 1
done
grep -q "model_api ready" "$API_ROOT/api.log" || { echo "api not ready in 60s" >&2; exit 2; }
echo "api ready on $API_PORT"

python3 - "$REPO" "$API_PORT" <<'PYDEC'
import json, sys, time, urllib.request
from pathlib import Path

repo, port = Path(sys.argv[1]), int(sys.argv[2])
fixture_dir = repo / "qualification/t1_reference/ling"
prompts = json.loads((fixture_dir / "prompts.json").read_text())["prompts"]
failures = 0
for prompt in prompts:
    name = prompt["name"]
    fixture = fixture_dir / f"{name}.t1r"
    data = fixture.read_bytes()
    # T1R1: u64 meta length, json meta, then per-array u64 size + payload.
    import struct
    length = struct.unpack("<Q", data[4:12])[0]
    meta = json.loads(data[12:12 + length])
    offset = 12 + length
    expected = None
    import zlib
    for entry in meta["arrays"]:
        size = struct.unpack("<Q", data[offset:offset + 8])[0]
        raw = zlib.decompress(data[offset + 8:offset + 8 + size])
        offset += 8 + size
        if entry["name"] == "generated_token_ids":
            count = len(raw) // 4
            expected = list(struct.unpack(f"<{count}i", raw))
    if expected is None:
        print(f"DECODE {name}: FAIL no reference tokens in fixture")
        failures += 1
        continue
    began = time.monotonic()
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/completions",
        data=json.dumps({
            "prompt_token_ids": prompt["prompt_token_ids"],
            "max_tokens": prompt["new_tokens"],
        }).encode(),
        headers={"Content-Type": "application/json"})
    response = json.loads(urllib.request.urlopen(
        request, timeout=300).read())
    elapsed = (time.monotonic() - began) * 1000.0
    emitted = response.get("token_ids") or response.get("tokens") or []
    status = "EXACT" if list(emitted) == list(expected) else "MISMATCH"
    if status != "EXACT":
        failures += 1
    print(f"DECODE {name}: {status} emitted={list(emitted)[:8]} "
          f"expected={list(expected)[:8]} elapsed_ms={elapsed:.0f} "
          f"new_tokens={prompt['new_tokens']}")
if failures:
    print(f"DECODE-RECEIPT-FAIL count={failures}")
    raise SystemExit(1)
print("DECODE-RECEIPT-PASS exact=both fixtures")
PYDEC
