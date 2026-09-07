#!/usr/bin/env bash
# Original Optimus-1 baseline: wooden then stone, dual GPU3/4, 15 episodes per GPU per task.
set -euo pipefail

ROOT="/home/ubuntu/data/xym/Optimus-1-origin"
cd "$ROOT"

GPU_A=3
GPU_B=4
PORT_A=9003
PORT_B=9004
TIMES=15

LOG_ROOT="logs/baseline/wooden_stone_gpu34"
mkdir -p "$LOG_ROOT"

export PATH="/data/conda/envs/optimus-1/bin:/usr/bin:/bin:$PATH"
export JAVA_HOME="/data/conda/envs/optimus-1"
export HF_HOME="${HF_HOME:-/data/hdd3/hf_cache}"
export PYTHONPATH="${ROOT}/minerl:${ROOT}/src"
export PYTHONNOUSERSITE=1
export OPTIMUS_PLAN_MODEL="${OPTIMUS_PLAN_MODEL:-qwen-vl}"
export OPTIMUS_VLM_MODEL="${OPTIMUS_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

WOODEN_EVAL="[0,1,2,3,4,5,6,7,8,9,10,11]"
STONE_EVAL="[]"

ensure_server() {
  local gpu="$1"
  local port="$2"
  local pidfile="${LOG_ROOT}/server_gpu${gpu}.pid"
  if curl -s -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/docs" 2>/dev/null | grep -q 200; then
    echo "VLM port ${port} ready (GPU${gpu})"
    return 0
  fi
  echo "Starting VLM on GPU${gpu} port ${port} ..."
  CUDA_VISIBLE_DEVICES="${gpu}" OPTIMUS_VLM_DEVICE=0 OPTIMUS_PORT="${port}" \
    nohup uvicorn app:app --host 127.0.0.1 --port "${port}" \
    >"${LOG_ROOT}/server_gpu${gpu}.log" 2>&1 &
  echo $! >"${pidfile}"
  for _ in $(seq 1 60); do
    if curl -s -m 10 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/docs" 2>/dev/null | grep -q 200; then
      echo "VLM ready: GPU${gpu} port ${port}"
      return 0
    fi
    sleep 10
  done
  echo "VLM failed GPU${gpu} port ${port}" >&2
  exit 1
}

prepare_memories() {
  echo "Resetting baseline memory from official tar (fresh extract) ..."
  FORCE_EXTRACT=1 bash scripts/reset_baseline_memory.sh "baseline_initial"
  for name in "wooden_gpu${GPU_A}" "wooden_gpu${GPU_B}" "stone_gpu${GPU_A}" "stone_gpu${GPU_B}"; do
    bash scripts/reset_baseline_memory.sh "${name}" &
  done
  wait
}

write_manifest() {
  cat >"${LOG_ROOT}/manifest.json" <<EOF
{
  "started_at": "$(date -Iseconds)",
  "project": "Optimus-1-origin",
  "protocol": "original_baseline_no_wm",
  "split": "15_per_gpu_30_total",
  "gpus": [${GPU_A}, ${GPU_B}],
  "wooden_tasks": 12,
  "stone_tasks": 10,
  "times_per_gpu": ${TIMES},
  "times_total_per_task": $((TIMES * 2)),
  "total_episodes": 660,
  "memory_baseline": "src/optimus1/memories/baseline_initial",
  "phases": ["wooden_all_tasks", "stone_all_tasks"]
}
EOF
}

launch_worker() {
  local bench="$1"
  local gpu="$2"
  local port="$3"
  local eval_list="$4"
  local pidfile="$5"
  nohup scripts/_baseline_worker.sh "${bench}" "${gpu}" "${port}" \
    "${LOG_ROOT}/${bench}_gpu${gpu}.log" \
    "src/optimus1/memories/ablation/${bench}_gpu${gpu}" \
    "${LOG_ROOT}/run_gpu${gpu}_${bench}" \
    "${eval_list}" \
    "${TIMES}" \
    >"${LOG_ROOT}/${bench}_gpu${gpu}.wrapper.log" 2>&1 &
  echo $! >"${pidfile}"
  echo "Launched ${bench} GPU${gpu} x${TIMES} pid $(cat "${pidfile}")"
}

wait_worker() {
  local pidfile="$1"
  local label="$2"
  local pid
  pid=$(cat "${pidfile}")
  echo "Waiting ${label} pid ${pid} ..."
  wait "${pid}" || true
}

chmod +x scripts/reset_baseline_memory.sh scripts/_baseline_worker.sh

prepare_memories
ensure_server "${GPU_A}" "${PORT_A}"
ensure_server "${GPU_B}" "${PORT_B}"
write_manifest

echo "=== Phase 1: wooden 12 tasks x${TIMES} on GPU${GPU_A} + GPU${GPU_B} ==="
launch_worker wooden "${GPU_A}" "${PORT_A}" "${WOODEN_EVAL}" "${LOG_ROOT}/worker_wooden_gpu${GPU_A}.pid"
launch_worker wooden "${GPU_B}" "${PORT_B}" "${WOODEN_EVAL}" "${LOG_ROOT}/worker_wooden_gpu${GPU_B}.pid"
wait_worker "${LOG_ROOT}/worker_wooden_gpu${GPU_A}.pid" "wooden GPU${GPU_A}"
wait_worker "${LOG_ROOT}/worker_wooden_gpu${GPU_B}.pid" "wooden GPU${GPU_B}"

echo "=== Phase 2: stone 10 tasks x${TIMES} on GPU${GPU_A} + GPU${GPU_B} ==="
launch_worker stone "${GPU_A}" "${PORT_A}" "${STONE_EVAL}" "${LOG_ROOT}/worker_stone_gpu${GPU_A}.pid"
launch_worker stone "${GPU_B}" "${PORT_B}" "${STONE_EVAL}" "${LOG_ROOT}/worker_stone_gpu${GPU_B}.pid"
wait_worker "${LOG_ROOT}/worker_stone_gpu${GPU_A}.pid" "stone GPU${GPU_A}"
wait_worker "${LOG_ROOT}/worker_stone_gpu${GPU_B}.pid" "stone GPU${GPU_B}"

echo "All baseline jobs finished $(date)"
