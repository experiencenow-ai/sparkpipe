#!/usr/bin/env bash
# qwen38max_multidev_first_decode.sh — the cold-launch receipt client for
# the qwen38_max TP16 family (lane 2). Drives ONE residentd (default: this
# host's rank-0 control endpoint) through sparkpipe_model_batch with a
# small deterministic batch, wraps it in tools/model_stream_decode_benchmark.py
# (the dsv4-pro first_decode pattern) and prints the receipt: TTFT, decode
# rate, token hash — the family's serving numbers.
#
# Queue cmd stays BARE; the runtime root is discovered from the LIVE
# family job's private namespace (newest /tmp/sparkqueue-*/deployment.json
# whose stage names this family), or overridden with QMAX_RECEIPT_ROOT.
set -euo pipefail

FAMILY_TAG="qwen38_max_resident_decode_stage"
BENCH_ID="${QMAX_RECEIPT_ID:-qmax-first-decode-$(date -u +%Y%m%dT%H%M%SZ)}"

fail() { echo "qwen38max-first-decode: $*" >&2; exit 1; }

HOST="$(hostname)"
case "$HOST" in spark[0-9a-f]) ;; *) fail "unexpected hostname '$HOST'" ;; esac
PREBUILT="/home/$HOST/sparkdata/qwen38max.tp16/build-latest"
[ -x "$PREBUILT/sparkpipe_model_batch" ] ||
  fail "model_batch missing in $PREBUILT (run the r20+ build arm)"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"

# Discover the family runtime root: newest private namespace carrying our
# deployment (the attach job's SPARK_QUEUE_RUNTIME_ROOT).
ROOT="${QMAX_RECEIPT_ROOT:-}"
if [ -z "$ROOT" ]; then
  for candidate in $(ls -dt /tmp/sparkqueue-*/ 2>/dev/null); do
    if [ -f "$candidate/deployment.json" ] &&
       grep -q "$FAMILY_TAG" "$candidate/deployment.json" 2>/dev/null; then
      ROOT="${candidate%/}"
      break
    fi
  done
fi
[ -n "$ROOT" ] && [ -f "$ROOT/deployment.json" ] ||
  fail "no live qwen38_max runtime root found (set QMAX_RECEIPT_ROOT)"

# Deterministic receipt batch: two short prompts, bounded outputs. The
# smoke census prompt distribution is the reference; any in-vocab ids
# exercise the full stack (embedding through head).
BATCH="$(mktemp /tmp/qmax-batch-XXXXXX.json)"
trap 'rm -f "$BATCH"' EXIT
python3 - "$BATCH" <<'PY'
import json, sys
prompts = [
    [1000 + (i * 371) % 200000 for i in range(16)],
    [2000 + (i * 517) % 200000 for i in range(24)],
]
batch = {
    "schema_version": 1,
    "connect_timeout_ms": 20000,
    "request_capacity": 4,
    "max_context_tokens": 512,
    "max_prefill_rows_per_submission": 4,
    "maximum_messages_per_rank_per_progress": 8,
    "maximum_new_submissions_per_progress": 4,
    "stop_token_ids": [],
    "requests": [
        {
            "request_id": rid,
            "sequence_id": rid,
            "priority": 0,
            "output_token_budget": 32,
            "prompt_token_ids": prompt,
        }
        for rid, prompt in enumerate(prompts, start=1)
    ],
}
with open(sys.argv[1], "w") as handle:
    json.dump(batch, handle)
PY

OUT="/tmp/${BENCH_ID}.json"
echo "qwen38max-first-decode: root=$ROOT batch=$BATCH"
python3 "$CHECKOUT/tools/model_stream_decode_benchmark.py" --output "$OUT" \
  "$PREBUILT/sparkpipe_model_batch" \
    --deployment "$ROOT/deployment.json" \
    --runtime-root "$ROOT" \
    --batch "$BATCH"

echo "== receipt =="
python3 - "$OUT" <<'PY'
import hashlib, json, sys
result = json.load(open(sys.argv[1]))
tokens = [e["event"]["token_id"] for e in result.get("events", [])
          if e.get("event", {}).get("event") == "token"]
print(json.dumps({k: result.get(k) for k in
      ("decode_tokens_per_second", "token_count", "ttft_seconds",
       "total_seconds", "inter_token_p95_seconds")}, indent=1))
print("token_hash:", hashlib.sha256(
    ",".join(str(t) for t in tokens).encode()).hexdigest())
print("receipt:", sys.argv[1])
PY
