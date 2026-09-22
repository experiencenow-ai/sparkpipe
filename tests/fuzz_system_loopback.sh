#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
rounds="${1:-200}"
seed="${2:-1}"
if [ ! -x build/test_system_loopback ]; then
    make build/test_system_loopback
fi
build/test_system_loopback --fuzz "$rounds" "$seed"
