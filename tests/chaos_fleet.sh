#!/usr/bin/env bash
# chaos_fleet.sh — randomized fault injection against the glm53flash fleet.
#
# Kills residentd/weightd processes on random nodes at random intervals,
# and asserts the fleet CONVERGES back to serving within a bound and never
# wedges. This is the harness that catches the class of bugs unit tests
# cannot: cross-process state races, restart storms, deploy drift.
#
# usage: chaos_fleet.sh <duration_s> [seed]
# exits 0 if the fleet served a request after every fault window, 1 on wedge.
set -uo pipefail

DURATION=${1:-600}
SEED=${2:-$$}
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
API="http://100.123.97.61:8433"

rand_node() { echo $NODES | tr ' ' '\n' | awk -v s=$((SEED + $1)) 'BEGIN{srand(s)} {l[NR]=$0} END{print l[int(rand()*NR)+1]}'; }
rand_kind() { awk -v s=$((SEED + $1)) 'BEGIN{srand(s); print (rand()<0.5 ? "residentd" : "weightd")}'; }

fleet_up() {
  ssh -o ConnectTimeout=6 spark0 'python3 -c "
import socket
up=0
for i in range(16):
    s=socket.socket(); s.settimeout(1)
    try: s.connect((\"10.10.100.%d\"%(10+i),19560+i)); up+=1
    except Exception: pass
    s.close()
print(up)"' 2>/dev/null
}

probe_serves() {
  ssh -o ConnectTimeout=10 spec@100.123.97.61 'timeout 120 python3 -c "
import http.client, json
body = json.dumps({\"prompt_token_ids\": [1,2,3,4,5,6,7,8], \"max_tokens\": 2, \"temperature\": 0})
conn = http.client.HTTPConnection(\"127.0.0.1\", 8433, timeout=110)
conn.request(\"POST\", \"/v1/completions\", body=body)
d = json.loads(conn.getresponse().read())
print(len(d.get(\"tokens\", [])))
"' 2>/dev/null
}

echo "chaos_fleet: duration=${DURATION}s seed=${SEED}"
start=$(date +%s)
round=0
fail=0
while [ $(( $(date +%s) - start )) -lt "$DURATION" ]; do
  round=$((round + 1))
  node=$(rand_node $round)
  kind=$(rand_kind $round)
  echo "[round $round] kill $kind on $node"
  if [ "$kind" = "residentd" ]; then
    ssh -o ConnectTimeout=6 "$node" 'pkill -9 -f "sparkpipe_model_resident[d]"' 2>/dev/null
  else
    ssh -o ConnectTimeout=6 "$node" 'pkill -TERM -f "sparkdata/weight[d]/sparkpipe_weightd"' 2>/dev/null
  fi
  sleep 20
  # convergence bound: fleet must be 16/16 within 300s of the kill
  t0=$(date +%s)
  while :; do
    up=$(fleet_up)
    [ "${up:-0}" = "16" ] && break
    [ $(( $(date +%s) - t0 )) -gt 300 ] && { echo "WEDGE: fleet at $up/16 after 300s (round $round, $kind on $node)"; fail=1; break; }
    sleep 10
  done
  [ "$fail" = 1 ] && break
  # serving proof: a request must return tokens (warm fleet = fast)
  toks=$(probe_serves)
  if [ "${toks:-0}" -lt 1 ]; then
    echo "WEDGE: fleet converged but request returned no tokens (round $round, $kind on $node)"
    fail=1
    break
  fi
  echo "[round $round] converged and served ($toks tokens) in $(( $(date +%s) - t0 ))s"
done
if [ "$fail" = 0 ]; then echo "chaos_fleet: PASS ($round rounds, no wedge)"; else echo "chaos_fleet: FAIL"; fi
exit $fail
