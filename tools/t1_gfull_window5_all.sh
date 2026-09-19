#!/usr/bin/env bash
set -u
MODE=${1:-wave}
HUB=$HOME/sparkdata/glm53full.fp8.tp16
case $MODE in
wave)
    : > /tmp/t1_w5_launch.log
    for r in $(seq 0 15); do
        h="spark$(printf '%x' "$r")"
        scp -q "$(dirname "$0")/t1_gfull_window5.sh" "$h:/home/$h/t1_gfull_window5.sh"
        if [ "$r" -eq 15 ]; then
            ssh -o BatchMode=yes "$h" "sudo -n su - sparkf -c 'bash /home/sparkf/t1_gfull_window5.sh 15'" >> /tmp/t1_w5_launch.log 2>&1 &
        elif [ "$h" = "$(hostname -s)" ]; then
            bash "$HOME/t1_gfull_window5.sh" "$r" >> /tmp/t1_w5_launch.log 2>&1 &
        else
            ssh -o BatchMode=yes "$h" "bash /home/$h/t1_gfull_window5.sh $r" >> /tmp/t1_w5_launch.log 2>&1 &
        fi
    done
    wait
    grep -c "state=ready" /tmp/t1_w5_launch.log
    grep -E "state=(exited|timeout)" /tmp/t1_w5_launch.log || true
    ;;
api)
    cd "$HUB"
    export SPARK_GLM52_SERVING_FLAT_RANKS=16
    export SPARK_GLM52_T1=1
    export LD_LIBRARY_PATH=$HUB/lib
    nohup ./bin/sparkpipe_model_api --deployment config/model_resident.json --runtime-root "$HUB" --port 8477 > model_api_t1.log 2>&1 < /dev/null &
    echo "api pid $!"
    sleep 3
    tail -2 model_api_t1.log
    ;;
stop)
    for r in $(seq 0 15); do
        h="spark$(printf '%x' "$r")"
        ssh -o BatchMode=yes "$h" "for pid in \$(pgrep -f sparkpipe_model_residentd); do case \$(readlink /proc/\$pid/cwd) in */glm53full.fp8.tp16) kill -TERM \$pid;; esac; done; true"
    done
    echo "stop issued (glm53full.fp8.tp16 cwds only)"
    ;;
*)
    echo "usage: $0 wave|api|stop" >&2
    exit 2
    ;;
esac
