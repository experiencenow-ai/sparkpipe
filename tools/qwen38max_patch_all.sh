#!/bin/sh
# Patch all placed qwen38max TP16 packs to the explicit codec contract.
# Rank 0 is already swapped (prior wave, byte-identity proven vs warm
# rebuild); this pass covers ranks 1-15 in parallel, one host each.
set -eu
LEASE=/Users/mac/sparkpipe-coord/CEPH_LEASE
HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
EXPORT_ROOT=/Users/mac/t1qmaxn/runs/t1qmax
REPO=/Users/mac/t1qmaxn
RANKS="${RANKS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15}"

say() {
	printf '[q38patch %s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

rank_host() {
	echo $HOSTS | cut -d' ' -f$(($1 + 1))
}

grep -qi "holder: T1-QMAX" "$LEASE" || { say "ceph lease not held by T1-QMAX"; exit 1; }
echo "continued_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ) (T1-QMAX codec-contract patch pass, ranks 1-15, tail reads only)" >> "$LEASE"

mkdir -p "$EXPORT_ROOT"
for rank in $RANKS; do
	host=$(rank_host "$rank")
	ssh -o BatchMode=yes "$host" "mkdir -p /tmp/t1qmax_stage/tools /tmp/t1qmax_rebuild && test \$(df -k /tmp | awk 'NR==2{print \$4}') -gt 209715200 || { echo \$host low scratch; exit 1; }"
	scp -q "$REPO/tools/qwen38max_patch_pack.py" "$REPO/tools/qwen38max_tp16_rank_verify.py" \
		"$REPO/tools/qwen38_stagepack.py" "$REPO/tools/spark_pack_common.py" \
		"$REPO/tools/qwen38max_patch_rank.sh" "$host:/tmp/t1qmax_stage/tools/"
	ssh -o BatchMode=yes "$host" "test -x /tmp/t1qmax_stage/tools/qwen38max_experts_manifest" \
		|| scp -q spark0:/tmp/t1qmax_stage/tools/qwen38max_experts_manifest "$host:/tmp/t1qmax_stage/tools/"
	ssh -o BatchMode=yes "$host" "chmod +x /tmp/t1qmax_stage/tools/qwen38max_experts_manifest /tmp/t1qmax_stage/tools/qwen38max_patch_rank.sh"
	say "staged rank $rank on $host"
done

pids=""
for rank in $RANKS; do
	host=$(rank_host "$rank")
	say "patch+swap rank $rank on $host"
	ssh -o BatchMode=yes "$host" "sh /tmp/t1qmax_stage/tools/qwen38max_patch_rank.sh $rank" \
		> "$EXPORT_ROOT/patch-rank$rank.log" 2>&1 &
	pids="$pids $!"
done
fail=0
for pid in $pids; do
	wait "$pid" || fail=1
done
if [ "$fail" != 0 ]; then
	say "PATCH WAVE INCOMPLETE - inspect $EXPORT_ROOT/patch-rank*.log"
	exit 1
fi

for rank in $RANKS; do
	host=$(rank_host "$rank")
	verdict=$(grep -c "rank $rank patched swapped relocked verified" "$EXPORT_ROOT/patch-rank$rank.log" || true)
	[ "$verdict" = "1" ] || { say "rank $rank missing success marker"; fail=1; }
done
[ "$fail" = 0 ] || exit 1
say "15/15 patched, swapped, relocked, verified; rank0 already in place"
