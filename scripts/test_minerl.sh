#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export OPTIMUS_PORT="${OPTIMUS_PORT:-9100}"

xvfb-run -a python -m optimus1.test_optimus1 "server.port=${OPTIMUS_PORT}" evaluate=[0]
