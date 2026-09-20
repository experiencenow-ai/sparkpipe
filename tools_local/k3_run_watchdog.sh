#!/usr/bin/env bash
# K3 lane fixture watchdog: ensures run A completes, chains run B, compares.
set -u
cd /home/spark7/lane-k3-t1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export K3_LAYER_TIMING=1

note() { echo "WATCHDOG: $1" >> watchdog_status.log; }

: > watchdog_status.log

# run A: relaunch whenever nothing is running and the manifest is missing
while [ ! -f out_a/k3/MANIFEST.json ]; do
  if ! pgrep -f "tools/t1_reference_decode[r]" >/dev/null; then
    note "launching run A"
    rm -rf out_a run_a.log
    setsid nohup python3 tools/t1_reference_decoder.py \
      --family k3 --checkpoint /home/spark7/lane-k3-t1/kimi-k3-local \
      --header model-families/k3/include/sparkpipe/llm_defines.h \
      --prompts prompts.json --output out_a > run_a.log 2>&1 < /dev/null &
  fi
  sleep 60
done
note "run A complete"

if [ ! -f out_b/k3/MANIFEST.json ]; then
  note "launching run B"
  rm -rf out_b run_b.log
  setsid nohup python3 tools/t1_reference_decoder.py \
    --family k3 --checkpoint /home/spark7/lane-k3-t1/kimi-k3-local \
    --header model-families/k3/include/sparkpipe/llm_defines.h \
    --prompts prompts.json --output out_b > run_b.log 2>&1 < /dev/null &
fi

while [ ! -f out_b/k3/MANIFEST.json ]; do
  if ! pgrep -f "tools/t1_reference_decode[r]" >/dev/null; then
    note "run B died; relaunching"
    setsid nohup python3 tools/t1_reference_decoder.py \
      --family k3 --checkpoint /home/spark7/lane-k3-t1/kimi-k3-local \
      --header model-families/k3/include/sparkpipe/llm_defines.h \
      --prompts prompts.json --output out_b > run_b.log 2>&1 < /dev/null &
  fi
  sleep 60
done
note "run B complete"

if cmp -s out_a/k3/capital_of_france.t1r out_b/k3/capital_of_france.t1r && \
   cmp -s out_a/k3/count_up.t1r out_b/k3/count_up.t1r; then
  note "fixtures BYTE-IDENTICAL across runs"
else
  note "DIVERGENCE between run A and run B fixtures"
fi
