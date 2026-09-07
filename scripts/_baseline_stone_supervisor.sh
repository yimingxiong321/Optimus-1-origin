#!/usr/bin/env bash
# Wait for wooden workers, then launch stone phase on GPU3/4.
set -euo pipefail

ROOT="/home/ubuntu/data/xym/Optimus-1-origin"
LOG_ROOT="${ROOT}/logs/baseline/wooden_stone_gpu34"
cd "$ROOT"

wait_worker() {
  local pidfile="$1"
  local label="$2"
  local pid
  pid=$(cat "${pidfile}")
  echo "Waiting ${label} pid ${pid} ..."
  wait "${pid}" || true
}

launch_stone() {
  local gpu="$1"
  local port="$2"
  local pidfile="${LOG_ROOT}/worker_stone_gpu${gpu}.pid"
  nohup scripts/_baseline_worker.sh stone "${gpu}" "${port}" \
    "${LOG_ROOT}/stone_gpu${gpu}.log" \
    "src/optimus1/memories/ablation/stone_gpu${gpu}" \
    "${LOG_ROOT}/run_gpu${gpu}_stone" \
    "[]" 15 \
    >"${LOG_ROOT}/stone_gpu${gpu}.wrapper.log" 2>&1 &
  echo $! >"${pidfile}"
  echo "Launched stone GPU${gpu} pid $(cat "${pidfile}")"
}

wait_worker "${LOG_ROOT}/worker_wooden_gpu3.pid" "wooden GPU3"
wait_worker "${LOG_ROOT}/worker_wooden_gpu4.pid" "wooden GPU4"

echo "=== Phase 2: stone $(date) ==="
launch_stone 3 9003
launch_stone 4 9004
wait_worker "${LOG_ROOT}/worker_stone_gpu3.pid" "stone GPU3"
wait_worker "${LOG_ROOT}/worker_stone_gpu4.pid" "stone GPU4"
echo "All baseline jobs finished $(date)"
