#!/bin/sh
set -eu
arm=${1:?usage: t1_q3f_patch_fleet.sh tp8|tp4pp4}
tools_dir="$HOME/q3ft1r/tools"
runs_dir="$HOME/q3ft1r/runs/q3f-t1/patch"
remote_tools='$HOME/q3ft1_rt_patch'
binary_host=spark7
mkdir -p "$runs_dir"
case $arm in
	tp8)
		hosts="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
		tp_degree=8
		pack_dir='sparkdata/qwen3flash.fp8.tp8/packs'
		prefix='qwenflash.tp8.fp8.rank'
		;;
	tp4pp4)
		hosts="sparkc sparkd sparke sparkf"
		tp_degree=4
		pack_dir='sparkdata/qwen3flash.fp8.tp4pp4/packs'
		prefix='qwenflash.tp4_pp4_fp8.rank'
		;;
	*)
		echo "unknown arm: $arm" >&2
		exit 2
		;;
esac
scp -q "$tools_dir/qwen4_flash_scale_plane_patch.py" \
	"$tools_dir/qwen4_flash_pack_stamp_check.py" \
	"$tools_dir/qwen4_flash_pack_verify.py" \
	"$tools_dir/spark_pack_common.py" \
	"$tools_dir/qwen4_flash_stagepack.py" \
	"$binary_host:q3ft1_rt_stage/"
rc_total=0
for host in $hosts; do
	case $host in
		sparka) digit=10 ;;
		sparkb) digit=11 ;;
		sparkc) digit=12 ;;
		sparkd) digit=13 ;;
		sparke) digit=14 ;;
		sparkf) digit=15 ;;
		*) digit=${host#spark} ;;
	esac
	if [ "$arm" = tp8 ]; then
		rank=$((digit % 8))
	else
		rank=$digit
	fi
	tp_rank=$((rank % tp_degree))
	pack=$(ssh -o BatchMode=yes "$host" "
		for candidate in \"\$HOME/$pack_dir/${prefix}$(printf '%02d' $rank).spstage\" \
			\"\$HOME/$pack_dir/${prefix}$rank.spstage\"; do
			[ -f \"\$candidate\" ] && { printf '%s' \"\$candidate\"; exit 0; }
		done
		echo MISSING")
	if [ "$pack" = MISSING ]; then
		echo "[$host] no pack found - FAILING LOUD" >&2
		rc_total=1
		break
	fi
	echo "[$(date -u +%H:%M:%S)] $host patching rank=$rank tp_rank=$tp_rank"
	ssh -o BatchMode=yes "$host" "rm -rf $remote_tools && mkdir -p $remote_tools"
	scp -q "$binary_host:q3ft1_rt_stage/build-qwen4flash-experts-manifest" \
		"$binary_host:q3ft1_rt_stage/qwen4_flash_scale_plane_patch.py" \
		"$binary_host:q3ft1_rt_stage/qwen4_flash_pack_stamp_check.py" \
		"$binary_host:q3ft1_rt_stage/qwen4_flash_pack_verify.py" \
		"$binary_host:q3ft1_rt_stage/spark_pack_common.py" \
		"$binary_host:q3ft1_rt_stage/qwen4_flash_stagepack.py" \
		"$host:q3ft1_rt_patch/"
	scp -q "$tools_dir/t1_q3f_patch_pack.sh" "$host:q3ft1_rt_patch/"
	if ssh -o BatchMode=yes "$host" "sh $remote_tools/t1_q3f_patch_pack.sh '$pack' $tp_degree $tp_rank $remote_tools" \
		> "$runs_dir/$host.log" 2>&1; then
		echo "[$(date -u +%H:%M:%S)] $host OK"
		scp -q "$host:$pack.patch.json" "$runs_dir/$host.patch.json" || true
		scp -q "$host:$pack.stampcheck.json" "$runs_dir/$host.stampcheck.json" || true
	else
		rc=$?
		echo "[$(date -u +%H:%M:%S)] $host FAILED rc=$rc - see $runs_dir/$host.log" >&2
		rc_total=1
		break
	fi
done
exit "$rc_total"
