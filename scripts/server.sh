#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/data/hdd3/hf_cache}"
export OPTIMUS_PLAN_MODEL="${OPTIMUS_PLAN_MODEL:-qwen-vl}"
export OPTIMUS_VLM_MODEL="${OPTIMUS_VLM_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export OPTIMUS_VLM_DEVICE="${OPTIMUS_VLM_DEVICE:-0}"
export OPTIMUS_PORT="${OPTIMUS_PORT:-9000}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uvicorn app:app --host 127.0.0.1 --port "${OPTIMUS_PORT}"