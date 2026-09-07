#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."

export OPTIMUS_PORT="${OPTIMUS_PORT:-9100}"

xvfb-run -a python -m optimus1.main "server.port=${OPTIMUS_PORT}" benchmark=diamond evaluate=[0,1,2,3,4,5,6] env.times=30