#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ubuntu/data/xym/Optimus-1-origin"
cd "$ROOT"

export PATH="/data/conda/envs/optimus-1/bin:/usr/bin:/bin:$PATH"
export JAVA_HOME="/data/conda/envs/optimus-1"
export HF_HOME="${HF_HOME:-/data/hdd3/hf_cache}"
export PYTHONPATH="${ROOT}/minerl:${ROOT}/src"
export PYTHONNOUSERSITE=1
export OPTIMUS_PLAN_MODEL="${OPTIMUS_PLAN_MODEL:-qwen-vl}"
export OPTIMUS_VLM_MODEL="${OPTIMUS_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"

BENCH="${1:?benchmark wooden|stone}"
GPU="${2:?gpu id}"
PORT="${3:?vlm port}"
LOG="${4:?log file}"
MEM="${5:?memory path}"
RUN_DIR="${6:?hydra run dir}"
EVAL="${7:-[]}"
TIMES="${8:-15}"

RESILIENT="${BASELINE_RESILIENT:-1}"
MAX_CRASH_RETRIES="${BASELINE_MAX_CRASH_RETRIES:-100}"
CRASH_DELAY_SEC="${BASELINE_CRASH_DELAY_SEC:-45}"

run_job() {
  local eval_list="$1"
  local run_times="$2"
  {
    echo "=== ${BENCH} GPU${GPU} x${run_times} start $(date) port=${PORT} eval=${eval_list} ==="
    xvfb-run -a python -m optimus1.main \
      "server.port=${PORT}" \
      "benchmark=${BENCH}" \
      "evaluate=${eval_list}" \
      "env.times=${run_times}" \
      "memory.path=${MEM}" \
      "hydra.run.dir=${RUN_DIR}"
    echo "=== ${BENCH} GPU${GPU} done $(date) eval=${eval_list} ==="
  } >>"${LOG}" 2>&1
}

compute_resume() {
  python "${ROOT}/scripts/ablation_compute_resume.py" "${BENCH}" "${LOG}" "${TIMES}"
}

if [[ "${RESILIENT}" != "1" ]]; then
  run_job "${EVAL}" "${TIMES}"
  exit 0
fi

crash_retries=0
while (( crash_retries < MAX_CRASH_RETRIES )); do
  if [[ -f "${LOG}" ]]; then
    if ! resume_out="$(compute_resume 2>/dev/null)"; then
      CURRENT_EVAL="${EVAL}"
      CURRENT_TIMES="${TIMES}"
    elif [[ "${resume_out}" == "DONE" ]]; then
      echo "=== ${BENCH} GPU${GPU} all tasks complete $(date) ===" >>"${LOG}"
      exit 0
    else
      CURRENT_EVAL="$(echo "${resume_out}" | awk '{print $1}')"
      CURRENT_TIMES="$(echo "${resume_out}" | awk '{print $2}')"
    fi
  else
    CURRENT_EVAL="${EVAL}"
    CURRENT_TIMES="${TIMES}"
  fi

  echo "=== ${BENCH} GPU${GPU} resilient launch $(date) eval=${CURRENT_EVAL} times=${CURRENT_TIMES} ===" >>"${LOG}"

  if run_job "${CURRENT_EVAL}" "${CURRENT_TIMES}"; then
    if ! resume_out="$(compute_resume 2>/dev/null)"; then
      continue
    fi
    if [[ "${resume_out}" == "DONE" ]]; then
      echo "=== ${BENCH} GPU${GPU} all tasks complete $(date) ===" >>"${LOG}"
      exit 0
    fi
    continue
  fi

  crash_retries=$((crash_retries + 1))
  {
    echo "=== ${BENCH} GPU${GPU} crash retry ${crash_retries}/${MAX_CRASH_RETRIES} $(date) ==="
  } >>"${LOG}"
  sleep "${CRASH_DELAY_SEC}"
done

echo "=== ${BENCH} GPU${GPU} exceeded max crash retries $(date) ===" >>"${LOG}"
exit 1
