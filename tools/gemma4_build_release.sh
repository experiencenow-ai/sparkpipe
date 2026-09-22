#!/usr/bin/env bash
# Build the coherent gemma4-31b TP16 serving release inside a queue-synced
# checkout (GPU-owned queue job, sm_121a): host services, residentd, adapter,
# validated module, compiled driver. Output: build/gemma4_31b_tp16 with
# SOURCE_COMMIT + SHA256SUMS (see tools/module_build_release.sh).
#
# The gemma4 GPU validator consumes no stage pack, but the module publish
# gate requires a readable STAGE_PACK_PATH: under a queue job this resolves
# to this node's landed rank pack (sparkX holds rank X, hex-suffixed). The
# queue's systemd embedding expands $-syntax in the job command itself, so
# queue cmd files must stay dollar-free - keep that resolution HERE.
set -euo pipefail
if [ "$#" -ne 0 ]; then
    printf '%s\n' 'gemma4_build_release.sh takes no arguments' >&2
    exit 2
fi
if [ -z "${STAGE_PACK_PATH:-}" ]; then
    if [ -n "${SPARK_QUEUE_RANK:-}" ]; then
        RANK_HEX="$(printf %x "$SPARK_QUEUE_RANK")"
        STAGE_PACK_PATH="$HOME/sparkdata/gemma4_31b.bf16.tp16/packs/gemma4_31b_tp16_rank${RANK_HEX}_stage0.gemma4sp"
        export STAGE_PACK_PATH
    else
        printf '%s\n' 'STAGE_PACK_PATH is unset and SPARK_QUEUE_RANK is absent: point STAGE_PACK_PATH at a landed rank pack' >&2
        exit 2
    fi
fi
if [ ! -r "$STAGE_PACK_PATH" ]; then
    printf 'gemma4_build_release.sh: FAIL: STAGE_PACK_PATH unreadable: %s\n' "$STAGE_PACK_PATH" >&2
    exit 2
fi
exec bash "$(dirname "$0")/module_build_release.sh" gemma4_resident_decode_stage bf16 gemma4_31b_tp16 842da3794eaa0b77d5f08bae87a17459d91ff475 model_contracts/gemma4_31b_authoritative.json
