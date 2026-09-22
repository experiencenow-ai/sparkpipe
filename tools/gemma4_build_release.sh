#!/usr/bin/env bash
# Build the coherent gemma4-31b TP16 serving release inside a queue-synced
# checkout (GPU-owned queue job, sm_121a): host services, residentd, adapter,
# validated module, compiled driver. Output: build/gemma4_31b_tp16 with
# SOURCE_COMMIT + SHA256SUMS (see tools/module_build_release.sh).
set -euo pipefail
if [ "$#" -ne 0 ]; then
    printf '%s\n' 'gemma4_build_release.sh takes no arguments' >&2
    exit 2
fi
exec bash "$(dirname "$0")/module_build_release.sh" gemma4_resident_decode_stage bf16 gemma4_31b_tp16 842da3794eaa0b77d5f08bae87a17459d91ff475 model_contracts/gemma4_31b_authoritative.json
