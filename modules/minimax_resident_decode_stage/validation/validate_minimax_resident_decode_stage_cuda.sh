#!/usr/bin/env bash
set -euo pipefail

# minimax text-tower resident decode stage, retained-receipt GPU validation.
#
# STATUS: NOT YET IMPLEMENTED - fail-closed placeholder. The adopted PR #1080
# driver reached lane 10 with its GPU qualification explicitly execution-
# gated (kernel-level decode/prefill vs the CPU oracle, TP4 all-reduce
# end-to-end, multi-position token match vs the t1 fixtures, sliding-window
# semantics, graph replay). Until the real validator lands, `make validate`
# must fail loudly rather than report an unearned PASS, and every module
# publication carries this placeholder's name so the audit trail shows the
# module is unqualified (the serving adapter additionally requires
# SPARK_MINIMAX_ALLOW_UNQUALIFIED_EXECUTION=1 at initialize).
#
# Contract for the future real validator (shared validation-driver
# skeleton, gemma4/dsv4 precedent):
#   argument 1: validation configuration sha256 (retained receipt)
#   environment: RUNTIME_CONFIGURATION from resident_decode_stage_rules.mk
#   exit 0 only on measured GPU evidence recorded under
#   qualification/serving-receipts/.

echo "minimax GPU validator: NOT IMPLEMENTED - module is not GPU-qualified" >&2
echo "  (lane 10 bring-up interim; the module's InitializeGate requires"      >&2
echo "   SPARK_MINIMAX_ALLOW_UNQUALIFIED_EXECUTION=1 and the publication"     >&2
echo "   records this placeholder. Do not ship production traffic.)"          >&2
exit 2
