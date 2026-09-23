#!/usr/bin/env python3
"""Lock the layer_scalar pack contract across emitter and loader.

tools/gemma4_stagepack.py asserts every checkpoint layer carries a
layer_scalar of exactly 1.0 and emits one KIND_LAYER_SCALAR entry per layer
in the common (dense and MoE) per-layer path; the module binds it
(TENSOR_LAYER_SCALAR -> layer_scalar_by_layer) and the decode launch feeds
it to SparkGemma4LaunchLayerScale. The residentd coverage gate
(SparkGemma4ModuleExpectedLayerBits) must therefore expect the
LAYER_SCALAR bit on every layer of both arms - missing it fails
pack_layer_incomplete at launch (gemma4-tp16-launch-12/-13, 16/16 nodes,
seen=0x040307fc expected=0x0307fc on layer 0).
Run: python3 tests/test_gemma4_layer_scalar_coverage.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

MODULE_SOURCE = REPOSITORY / "modules/gemma4_resident_decode_stage/source/spark_gemma4_resident_decode_stage_module.c"
FORMAT_HEADER = REPOSITORY / "modules/gemma4_resident_decode_stage/source/spark_gemma4_stagepack_format.h"
EMITTER = REPOSITORY / "tools/gemma4_stagepack.py"


def main() -> int:
    failures: list[str] = []
    module = MODULE_SOURCE.read_text(encoding="utf-8")
    header = FORMAT_HEADER.read_text(encoding="utf-8")
    emitter = EMITTER.read_text(encoding="utf-8")

    # The enum still defines the kind (regression guard against renumbering).
    if not re.search(r"SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_SCALAR\s*=\s*26\b", header):
        failures.append("stagepack format header no longer defines TENSOR_LAYER_SCALAR = 26")

    # The loader binds the kind into the per-layer array the decode launch reads.
    if "case SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_SCALAR:" not in module:
        failures.append("module lost the TENSOR_LAYER_SCALAR bind case")
    if "layer_scalar_by_layer[layer]" not in module:
        failures.append("module no longer consumes layer_scalar_by_layer in the decode launch")

    # ExpectedLayerBits must include the LAYER_SCALAR bit unconditionally.
    match = re.search(
        r"SparkGemma4ModuleExpectedLayerBits\(.*?\)\s*\{(.*?)\n\}",
        module,
        re.DOTALL,
    )
    if match is None:
        failures.append("SparkGemma4ModuleExpectedLayerBits not found")
    else:
        body = match.group(1)
        base = body.split("#if", 1)[0]
        if "SPARK_GEMMA4_STAGEPACK_TENSOR_LAYER_SCALAR" not in base:
            failures.append(
                "ExpectedLayerBits base mask lacks TENSOR_LAYER_SCALAR "
                "(pack_layer_incomplete at residentd launch)"
            )

    # The emitter must add exactly one LAYER_SCALAR entry per layer in the
    # common per-layer path (before any dense/MoE branch), for both arms.
    emit = re.search(
        r"for layer in range\(first_layer, first_layer \+ layer_count\):(.*?)(?=\n    if geometry\[\"experts\"\])",
        emitter,
        re.DOTALL,
    )
    if emit is None:
        failures.append("stagepack per-layer plan loop not found")
    else:
        common = emit.group(1)
        if not re.search(r"add\(KIND_LAYER_SCALAR, layer, 1, 1\)", common):
            failures.append(
                "stagepack no longer emits KIND_LAYER_SCALAR in the common per-layer path"
            )

    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        print(f"FAILED {len(failures)} check(s)")
        return 1
    print("PASS gemma4 layer_scalar pack contract: emitter, binder and expected-bits agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
