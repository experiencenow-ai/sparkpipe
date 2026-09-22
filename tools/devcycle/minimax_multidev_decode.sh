#!/usr/bin/env bash
# minimax lane-10 decode driver: the common sparkpipe_model_api against a
# LIVE serve attempt's PRIVATE deployment (read-only; the residentds own
# the control endpoints), driving the canonical T1 prompt set
# (qualification/t1_reference/minimax/prompts.json) at B1.
#
# Ported from tools/laguna_multidev_decode.sh (the lane-8 precedent).
#
# The API listener binds one port inside the lane-10 session block's spare
# tail (23870; reserve it with --ports on THIS job).
#
# Environment
#   MINIMAX_DECODE_ATTEMPT   the SERVE job's 32-hex attempt id (whose
#                            /tmp/sparkqueue-<attempt> namespace holds the
#                            private deployment + residentd logs)
#   MINIMAX_DECODE_API_PORT  default 23870
#   MINIMAX_DECODE_FIRMWARE  firmware tarball holding bin/sparkpipe_model_api
#                            (default ~/sparkdata/firmware/minimax-text-tp4-fw.tar.gz)
# Receipt lines: DECODE-REQ and DECODE-DONE with wall ms.
set -euo pipefail

ATTEMPT="${MINIMAX_DECODE_ATTEMPT:?set MINIMAX_DECODE_ATTEMPT to the serve job attempt id}"
API_PORT="${MINIMAX_DECODE_API_PORT:-23870}"
FIRMWARE="${MINIMAX_DECODE_FIRMWARE:-$HOME/sparkdata/firmware/minimax-text-tp4-fw.tar.gz}"
ROOT="/tmp/sparkqueue-$ATTEMPT"
CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="$(hostname)"
[ "$HOST" = "spark8" ] || { echo "minimax decode driver runs on spark8 (the lane-10 coordinator)" >&2; exit 2; }
[ -d "$ROOT" ] || { echo "serve attempt root missing: $ROOT" >&2; exit 2; }
[ -f "$ROOT/deployment.json" ] || { echo "serve deployment missing" >&2; exit 2; }
[ -s "$FIRMWARE" ] || { echo "firmware tarball missing: $FIRMWARE" >&2; exit 2; }

fail() { echo "minimax-decode: $*" >&2; exit 1; }

# 1. The release-built common API against the serve attempt's private
#    deployment (this job's own cgroup; TERM-stopped on exit). The API
#    waits for the engine ranks itself and serves GET /health once up.
FW_DIR="$CHECKOUT/build/minimax-decode-fw"
mkdir -p "$FW_DIR"
tar -xzf "$FIRMWARE" -C "$FW_DIR"
API_LOG="$ROOT/api.log"
"$FW_DIR/bin/sparkpipe_model_api" \
  --deployment "$ROOT/deployment.json" \
  --runtime-root "$ROOT" \
  --port "$API_PORT" >"$API_LOG" 2>&1 &
API_PID=$!
trap 'kill -TERM "$API_PID" 2>/dev/null || true' EXIT
HEALTHY=""
for _ in $(seq 1 240); do
  if curl -sf --max-time 5 "http://127.0.0.1:$API_PORT/health" >"$ROOT/health.json" 2>/dev/null; then
    HEALTHY=1
    break
  fi
  kill -0 "$API_PID" 2>/dev/null || { tail -5 "$API_LOG" >&2; fail "api exited early"; }
  sleep 5
done
[ -n "$HEALTHY" ] || { tail -5 "$API_LOG" >&2; fail "api not healthy in 20 min (engine ranks not ready?)"; }
echo "DECODE-API health=$(cat "$ROOT/health.json")"

# 2. Canonical prompts at B1 (single-sequence completions, the exact T1
#    fixture prompts and token budgets).
decode() {
  local name="$1" ids="$2" new_tokens="$3" began ended
  began=$(date +%s%N)
  body="$(curl -sf --max-time 120 -X POST "http://127.0.0.1:$API_PORT/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt_token_ids\": [$ids], \"max_tokens\": $new_tokens}")" || {
    echo "DECODE-REQ name=$name status=ERROR" >&2; return 1; }
  ended=$(date +%s%N)
  echo "DECODE-REQ name=$name wall_ms=$(( (ended - began) / 1000000 )) body=${body:0:400}"
}

python3 - "$CHECKOUT" <<'PYIDS' > /tmp/minimax-decode-ids.$$
import json, sys
prompts = json.load(open(sys.argv[1] + "/qualification/t1_reference/minimax/prompts.json"))["prompts"]
for prompt in prompts:
    print(prompt["name"] + "\t" + ",".join(str(t) for t in prompt["prompt_token_ids"]) + "\t" + str(prompt["new_tokens"]))
PYIDS
while IFS=$'\t' read -r name ids new_tokens; do
  [ -n "$name" ] || continue
  decode "$name" "$ids" "$new_tokens"
done < "/tmp/minimax-decode-ids.$$"
rm -f "/tmp/minimax-decode-ids.$$"
echo "DECODE-DONE attempt=$ATTEMPT api_port=$API_PORT"
