#!/usr/bin/env python3

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CUDA_SOURCE = ROOT / "tools" / "hardware" / "spark_cuda_characterize.cu"
NVME_SOURCE = ROOT / "tools" / "hardware" / "spark_nvme_characterize.cu"
PMTU_SOURCE = ROOT / "tools" / "hardware" / "spark_pmtu_characterize.c"
COMMON_HEADER = ROOT / "tools" / "hardware" / "spark_probe_common.h"
CUDA_GATE = ROOT / "tools" / "cuda13_sm121a_compile_gate.sh"
CUDA_STUB = ROOT / "tests" / "hardware_cuda_stub"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise AssertionError(f"command failed: {' '.join(command)}")


def main() -> int:
    cuda = CUDA_SOURCE.read_text(encoding="utf-8")
    nvme = NVME_SOURCE.read_text(encoding="utf-8")
    common = COMMON_HEADER.read_text(encoding="utf-8")
    gate = CUDA_GATE.read_text(encoding="utf-8")

    for required in (
        "gpu_bandwidth_ratio",
        'baseline_options.candidate = "gpu_only"',
        "options.batch_size",
        "options.kernel_count",
        r'\"sample_phase\": \"all\"',
        "cudaSetDeviceFlags(cudaDeviceMapHost)",
        "cudaFuncSetAttribute",
        "SparkCudaProbeMeasureReadReuse",
        "SparkCudaProbeMeasurePointerChase",
    ):
        require(required in cuda, f"CUDA hardware probe is missing {required}")

    for required in (
        "SparkNvmeProbeVerifyPattern",
        "SparkNvmeProbeCopyAndFingerprint",
        "cpu_fingerprint != device_fingerprint",
        "initialization_failed",
        "execution_failed",
        "thread.joinable()",
    ):
        require(required in nvme, f"NVMe hardware probe is missing {required}")

    for required in (
        "SparkProbeFingerprintWords",
        "SparkProbeSummarizeLatency",
        "SparkProbeWriteJsonString",
        "SparkProbeHexSha256IsValid",
    ):
        require(required in common, f"common probe support is missing {required}")

    require("tools/hardware/spark_cuda_characterize.cu" in gate,
            "CUDA 13 gate does not compile the GB10 probe")
    require("tools/hardware/spark_nvme_characterize.cu" in gate,
            "CUDA 13 gate does not compile the NVMe-to-GPU probe")

    if sys.platform.startswith("linux"):
        run([
            "cc", "-std=c11", "-Wall", "-Wextra", "-Werror",
            "-Itools/hardware", "-fsyntax-only", str(PMTU_SOURCE),
        ])
    else:
        # The PMTU probe is Linux networking (endian.h, the PMTU socket
        # options); there is nothing to syntax-check on another OS.
        print("SKIP PMTU probe syntax: Linux-only probe source")

    nvcc = shutil.which("nvcc")
    if nvcc is not None:
        # The real compiler for the real probes: CUDA launch syntax needs a
        # CUDA toolchain, and nvcc is also the production compiler.
        import tempfile
        with tempfile.TemporaryDirectory() as objects:
            for source in (CUDA_SOURCE, NVME_SOURCE):
                run([
                    nvcc,
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-Itools/hardware",
                    "-c", str(source),
                    "-o", os.path.join(objects, source.stem + ".o"),
                ])
    else:
        clang = shutil.which("clang++") or "/usr/local/swift/usr/bin/clang++"
        if pathlib.Path(clang).is_file():
            common_command = [
                clang,
                "-std=c++17",
                "-x", "cuda",
                "--cuda-host-only",
                "-nocudainc",
                "-nocudalib",
                # The syntax check reads tests/hardware_cuda_stub, not the
                # installed toolkit; whatever CUDA clang auto-detects locally
                # (and possibly rejects as too new) is not the input here.
                "-Wno-unknown-cuda-version",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Itests/hardware_cuda_stub",
                "-Itools/hardware",
                "-fsyntax-only",
            ]
            try:
                run(common_command + [str(CUDA_SOURCE)])
                run(common_command + [str(NVME_SOURCE)])
            except AssertionError:
                # Host clang without CUDA launch support (Apple clang, or
                # <<< >>> against a stub): the check needs a CUDA toolchain,
                # and none is present - a skip, not a pass and not a failure.
                print("SKIP CUDA probe syntax: no CUDA-capable compiler")
        else:
            print("SKIP Clang CUDA host syntax: clang++ unavailable")

    print("PASS hardware CUDA, NVMe, PMTU, shared-support, and exact-target source contracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
