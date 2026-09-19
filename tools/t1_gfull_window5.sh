#!/usr/bin/env bash
set -u
R=${1:?rank}
ROOT=$HOME/sparkdata/glm53full.fp8.tp16
export SPARK_GLM52_SERVING_FLAT_RANKS=16
export SPARK_WEIGHTD_SOCKET=/tmp/spark_weightd.sock
export SPARK_WEIGHTD_ATTACH=1
export SPARK_WEIGHTD_ATTACH_LAZY=1
export SPARK_WEIGHTD_EXPERT_POOL_BYTES=2147483648
export SPARK_WEIGHTD_SPINE_BUDGET_BYTES=8589934592
export SPARK_GLM52_T1=1
export LD_LIBRARY_PATH=$ROOT/lib
cd "$ROOT"
DIG=$(cat packs/glm53full.fp8.tp16-rank$R.glm52sp.sha256 | cut -d' ' -f1)
export SPARK_WEIGHTD_PACK_SHA256=$DIG
echo "t1launch rank=$R node=$(hostname) memlock=$(ulimit -l) socket=$SPARK_WEIGHTD_SOCKET sha=$DIG"
nohup ./bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $R > residentd_t1.log 2>&1 < /dev/null &
PID=$!
STATE=timeout
i=0
while [ $i -lt 60 ]; do
    if ! kill -0 $PID 2>/dev/null; then STATE=exited; break; fi
    if grep -q "model_residentd ready" residentd_t1.log 2>/dev/null; then STATE=ready; break; fi
    i=$((i+1)); sleep 5
done
echo "t1launch rank=$R state=$STATE pid=$PID"
tail -3 residentd_t1.log
