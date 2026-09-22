#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -d build/obj ]; then
  echo "Use a fresh checkout to keep host CUDA stubs separate from production objects" >&2
  exit 2
fi
mkdir -p build/reliability
tests=(
  test_memory_buffer test_model_batch_engine_mock
  test_model_pipeline_client_mock test_model_pipeline_client
  test_model_resident_session test_model_resident_reconnect
  test_model_resident_deadline test_model_resident_ipc
  test_tp_allreduce_fuzz test_glm5_next_lazy_dispatch
  test_weightd_fd_frames test_weightd_working_set test_weightd_expert
  test_weightd_churn test_weightd_expert_stress
  test_weightd_attach test_weightd_worker
)
targets=("${tests[@]/#/build/}")
make -j4 CUDA_HOME=/nonexistent "${targets[@]}" > build/reliability/build.log 2>&1
for test in "${tests[@]}"; do
  timeout 180 "build/$test" > "build/reliability/$test.log" 2>&1
done
for test in test_weightd_supervision test_glm5_next_graph_failure test_glm5_next_stage_context test_glm5_next_embedding_collective; do
  timeout 180 python3 "tests/$test.py" > "build/reliability/$test.log" 2>&1
done
