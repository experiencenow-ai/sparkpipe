#!/bin/sh
# HY4-T1 slot-4 GPU window orchestrator: build the module archive + the two
# harnesses on sparkf, ship to 16 hosts, run the v2 lazy-attach receipt and
# the Execute stub boundary per rank, harvest logs. CEPH untouched: the
# weightd reads each rank pack from node-local placed storage.
set -eu

REPO=/Users/mac/hy4tr
BUILD_NODE=sparkf
STAGE=/tmp/hy4win_stage
RUNTIME=/tmp/hy4win
EXPORT_ROOT=/Users/mac/hy4tr/runs/hy4-window
BASE=sparkdata/hy4.fp8.tp16
HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"

say() {
	printf '[hy4win %s] %s\n' "$(date -u +%H:%M:%S)" "$*"
}

rank_host() {
	echo $HOSTS | cut -d' ' -f$(($1 + 1))
}

rank_hex() {
	printf "%02d" $1
}

stage_build() {
	say "bundling lane/hy4-t1 for $BUILD_NODE"
	rm -rf "$STAGE"
	mkdir -p "$STAGE"
	ssh "$BUILD_NODE" "rm -rf $STAGE && mkdir -p $STAGE"
	git -C "$REPO" bundle create "$STAGE/hy4.bundle" lane/hy4-t1
	scp -q "$STAGE/hy4.bundle" "$BUILD_NODE:$STAGE/"
	ssh "$BUILD_NODE" "rm -rf $STAGE/tree && git clone -q -b lane/hy4-t1 $STAGE/hy4.bundle $STAGE/tree"
	say "building on $BUILD_NODE"
	ssh "$BUILD_NODE" "set -e
		cd $STAGE/tree
		export PATH=/usr/local/cuda/bin:\$PATH
		REPOSITORY_ROOT=\$PWD make -C modules/hy4_resident_decode_stage archive > $STAGE/make_module.log 2>&1
		make build/libsparkpipe_core.a build/libsparkpipe_model_common.a > $STAGE/make_libs.log 2>&1
		nvcc -std=c++17 -O3 -arch=sm_121a \
			-I. -Iinclude -Imodel-families/common/include -Imodel-families/hy4/include \
			-Imodules/hy4_resident_decode_stage/include \
			-Imodules/hy4_resident_decode_stage/source \
			-DSPARK_HY4_MODULE_BUILD=1 \
			tools/hy4_stub_harness.c \
			build/modules/hy4_resident_decode_stage/libhy4_resident_decode_stage.a \
			build/libsparkpipe_model_common.a build/libsparkpipe_core.a \
			-L/usr/local/cuda/lib64 -lcudart -lcuda -o $STAGE/hy4_stub 2> $STAGE/nvcc_stub.log
		cc -std=c11 -O3 -D_GNU_SOURCE -I. -Iinclude \
			-I/usr/local/cuda/include \
			tools/hy4_attach_harness.c \
			build/libsparkpipe_model_common.a build/libsparkpipe_core.a \
			-L/usr/local/cuda/targets/sbsa-linux/lib -lcudart -lcuda -lpthread \
			-o $STAGE/hy4_attach 2> $STAGE/cc_attach.log
		ls -la $STAGE/hy4_stub $STAGE/hy4_attach"
	say "build green on $BUILD_NODE"
}

stage_ship() {
	for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
		host=$(rank_host "$rank")
		ssh "$host" "rm -rf $RUNTIME && mkdir -p $RUNTIME"
		scp -q "$BUILD_NODE:$STAGE/hy4_stub" "$BUILD_NODE:$STAGE/hy4_attach" "$host:$RUNTIME/"
	done
	say "staged harnesses on 16 hosts"
}

one_rank() {
	rank=$1
	host=$(rank_host "$rank")
	xx=$(rank_hex "$rank")
	pack="\$HOME/$BASE/packs/rank-$xx/model-fp8-tp16-rank-$xx.safetensors"
	ssh -o BatchMode=yes "$host" "set -e
		SHA=\$(cut -d' ' -f1 $pack.sha256)
		export SPARK_WEIGHTD_ATTACH_SHA256=\$SHA
		$RUNTIME/hy4_attach $pack
		$RUNTIME/hy4_stub" > "$EXPORT_ROOT/rank$rank.log" 2>&1
}

run_wave() {
	mkdir -p "$EXPORT_ROOT"
	fail=0
	pids=""
	for rank in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
		one_rank "$rank" &
		pids="$pids $!"
	done
	for pid in $pids; do
		wait "$pid" || fail=1
	done
	attach_ok=$(grep -l "CreateChecked status=0" "$EXPORT_ROOT"/rank*.log | wc -l | tr -d ' ')
	stub_ok=$(grep -l "STUB-BOUNDARY execute status=19" "$EXPORT_ROOT"/rank*.log | wc -l | tr -d ' ')
	say "waves done: attach_ok=$attach_ok/16 stub_ok=$stub_ok/16 fail=$fail"
	[ "$attach_ok" = 16 ] && [ "$stub_ok" = 16 ] && [ "$fail" = 0 ]
}

case "${1:-}" in
build) stage_build ;;
ship) stage_ship ;;
run) run_wave ;;
all)
	stage_build
	stage_ship
	run_wave
	;;
*)
	echo "usage: $0 build|ship|run|all" >&2
	exit 2
	;;
esac
