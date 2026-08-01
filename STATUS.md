# SparkPipe Status — Phase 7 Foundational Safety

This source tree is an implementation candidate for the following mandatory targets:

- Kimi K3: MXFP4 routed-expert weights, BF16 expert activations, BF16 non-expert tensors, FP32 accumulation.
- GLM 5.2: FP8 E4M3 routed-expert weights, BF16 activations and non-expert tensors, FP32 accumulation.
- Qwen 3.6 27B: BF16 weights and activations, with FP32 accumulation where required.
- DeepSeek V4 Flash and DeepSeek V4 Pro: separate checkpoint-derived contracts and separate execution packages.

Phase 7 closes the audited NVMe ownership/layout defects, replaces raw arena pointers with generation-carrying allocation handles, makes required KV-cache access fail closed, restores an explicit sliding-window position producer, removes global CUDA fast-math policy, derives the GLM estimator from the model contract, and replaces Git-dependent/circular package receipts with a deterministic source-package identity.

The source tree does not certify itself. Build, test, gate, archive, and extraction results are valid only when tied to the exact released archive SHA-256 by an external verification receipt.

```text
SOURCE_PACKAGE_KIND=sparkpipe_source
HOST_BUILD_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT
HOST_TEST_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT
ARCHITECTURE_GATE_STATUS=SEE_EXTERNAL_VERIFICATION_RECEIPT
CUDA13_SM121A_COMPILE_NOT_RUN=true
BLACKWELL_EXECUTION_NOT_MEASURED=true
PHYSICAL_NETWORK_EXECUTION_NOT_MEASURED=true
PRODUCTION_READY=false
```

The remaining blockers are tracked in `docs/PHASE7_REMAINING_WORK.md`.
