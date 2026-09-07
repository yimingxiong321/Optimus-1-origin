#!/usr/bin/env bash
# Restore pristine Optimus-1 initial memory for baseline runs.
set -euo pipefail

ROOT="/home/ubuntu/data/xym/Optimus-1-origin"
cd "$ROOT"

CKPT_ZIP="${CKPT_ZIP:-/home/ubuntu/data/xym/optimus1_steve1_ckpt.zip}"
MEMORY_TAR="src/optimus1/memories/v1_hf_download/memory.tar.gz"
BASELINE="src/optimus1/memories/baseline_initial"

if [[ ! -f "$CKPT_ZIP" ]]; then
  echo "Missing checkpoint zip: $CKPT_ZIP" >&2
  exit 1
fi
if [[ ! -f "$MEMORY_TAR" ]]; then
  echo "Missing memory archive: $MEMORY_TAR" >&2
  exit 1
fi

if [[ ! -f checkpoints/steve1/steve1.weights ]]; then
  echo "Extracting STEVE-1 checkpoints from $CKPT_ZIP ..."
  unzip -o "$CKPT_ZIP" -d "$ROOT"
fi

if [[ "${FORCE_EXTRACT:-0}" == "1" || ! -d "$BASELINE/plan" ]]; then
  echo "Extracting initial memory to $BASELINE from $MEMORY_TAR ..."
  rm -rf "$BASELINE"
  mkdir -p "$BASELINE"
  tar -xzf "$MEMORY_TAR" -C "$BASELINE"
  echo "Baseline memory ready: $BASELINE"
else
  echo "Baseline memory already exists: $BASELINE"
fi

copy_snapshot() {
  local name="$1"
  local dst="src/optimus1/memories/ablation/${name}"
  mkdir -p "src/optimus1/memories/ablation"
  rm -rf "$dst"
  cp -a "$BASELINE" "$dst"
  echo "Memory snapshot: $dst"
}

copy_snapshot "${1:-default}"
