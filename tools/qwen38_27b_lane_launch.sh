#!/usr/bin/env bash
# Shared-socket family wrapper for the qwen38_27b developer lane (lane 1,
# TP4 on spark0-spark3). Runs INSIDE an admitted spark_queue job: prepare a
# private deployment under $SPARK_QUEUE_RUNTIME_ROOT, then exec the verified
# sparkpipe_model_residentd against the SHARED weightd. Never starts a
# private weightd (multidev quickstart law).
#
# Required environment
#   SPARK_QUEUE_RUNTIME_ROOT / SPARK_QUEUE_RANK / SPARK_QUEUE_SIZE /
#   SPARK_QUEUE_ATTEMPT        (supplied by the queue)
#   QWEN38_27B_LANE_FIRMWARE_ROOT  verified module_build_release.sh output
#
# Optional environment (defaults: the tracked shared weightd socket
# /run/sparkpipe-weightd-shared/weightd.sock, lane-1 hosts/ports/mesh,
# weightd mesh lane 1, node-local nvfp4a16 TP4 rank packs).
#
# Lane defaults (override through QWEN38_27B_LANE_*):
#   hosts spark0-spark3, control 23016, collective 53016, transport 64016,
#   weightd mesh lane 1, physical mesh ranks 0,1,2,3,
#   packs /home/<host>/sparkdata/qwen38-27b.nvfp4a16.tp4/packs.
#
# The controller's run-family-job.sh calls this script as the job command;
# every participant stays inside the queue cgroup because this script only
# ever execs the resident daemon.
set -euo pipefail

: "${SPARK_QUEUE_RUNTIME_ROOT:?run inside a spark_queue job}"
: "${SPARK_QUEUE_RANK:?run inside a spark_queue job}"
: "${SPARK_QUEUE_SIZE:?run inside a spark_queue job}"
: "${SPARK_QUEUE_ATTEMPT:?run inside a spark_queue job}"
: "${QWEN38_27B_LANE_FIRMWARE_ROOT:?point at a verified module build}"
: "${QWEN38_27B_LANE_SHARED_SOCKET:=/run/sparkpipe-weightd-shared/weightd.sock}"
export QWEN38_27B_LANE_SHARED_SOCKET

if [ "${SPARK_QUEUE_SIZE}" -ne 4 ]; then
    echo "qwen38_27b lane: TP4 lane expects SPARK_QUEUE_SIZE=4 (got ${SPARK_QUEUE_SIZE})" >&2
    exit 2
fi

here=$(cd -- "$(dirname -- "$0")" && pwd)
staged=$(python3 "${here}/qwen38_27b_lane_deployment.py")
runtime_root=$(printf '%s' "${staged}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])')

export SPARK_WEIGHTD_SOCKET="${QWEN38_27B_LANE_SHARED_SOCKET}"
export SPARK_WEIGHTD_LANE="${QWEN38_27B_LANE_WEIGHTD_LANE:-1}"
export SPARK_TP_MESH_RANKS="${QWEN38_27B_LANE_MESH_RANKS:-0,1,2,3}"
export LD_LIBRARY_PATH="${runtime_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

echo "qwen38_27b lane: rank=${SPARK_QUEUE_RANK} socket=${SPARK_WEIGHTD_SOCKET}" \
     "lane=${SPARK_WEIGHTD_LANE} mesh=${SPARK_TP_MESH_RANKS} runtime=${runtime_root}"
exec "${QWEN38_27B_LANE_FIRMWARE_ROOT}/bin/sparkpipe_model_residentd" \
    --deployment "${SPARK_QUEUE_RUNTIME_ROOT}/deployment.json" \
    --rank-index "${SPARK_QUEUE_RANK}"
