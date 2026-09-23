#!/usr/bin/env python3
"""The gemma4 serving adapter .so must dlopen in any host process.

The minimax lane-10 connect failure (71fb5d97): an adapter linked from the
runtime static archives without -lcudart leaves cudaFree/cudaMalloc/
cudaMemcpyAsync/cudaStreamSynchronize undefined - the residentd provides
them, but sparkpipe_model_api does not and its RTLD_NOW load fails closed
as NOT_FOUND (engine_connect status=3). module_build_release.sh now fails
the release on adapter undefined symbols (adapter-dependencies.log), so the
gemma4 adapter rule must link the CUDA runtime library.
Run: python3 tests/test_gemma4_adapter_selfcontained.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

MAKEFILE = REPOSITORY / "modules/gemma4_resident_decode_stage/Makefile"


def main() -> int:
    failures: list[str] = []
    text = MAKEFILE.read_text(encoding="utf-8")

    match = re.search(r"^adapter:(.*)\Z", text, re.DOTALL | re.MULTILINE)
    if match is None:
        failures.append("adapter rule not found in the gemma4 module Makefile")
    else:
        rule = match.group(1)
        if "-lcudart" not in rule:
            failures.append("adapter rule does not link -lcudart (undefined CUDA runtime symbols)")
        if "-L\"$(CUDA_HOME)/lib64\"" not in rule:
            failures.append("adapter rule does not search $(CUDA_HOME)/lib64 for libcudart")

    if failures:
        for failure in failures:
            print(f"MISMATCH {failure}")
        print(f"FAILED {len(failures)} check(s)")
        return 1
    print("PASS gemma4 adapter links the CUDA runtime (self-contained .so)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
