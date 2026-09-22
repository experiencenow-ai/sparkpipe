# Batch, cache and parallel inference implementation status

This draft stacks on PR1081 and exposes the implementation while qualification
continues. It does not claim a reliable GPU server or a throughput recovery.

Implemented and host-tested:

- GLM honors separate logical/physical cache capacities and attaches the
  existing dirty-page writeback callback. Bounded diagnostics capture complete
  KV/index/recurrent/window state, final hidden state and selected head score
  while execution ownership is held. B1/B3/B5 probes include mixed-position
  joins, eviction, nonidentity restore, state/token comparison and reset.
- Common cache copy-on-write preserves immutable partial prefixes across
  divergent branches, copy failures, eviction pressure, abort and failed Finish.
- Mesh source ownership lasts through every posted transfer completion, with
  finite send credits, full epoch/sequence tags, stale completion rejection and
  source-copy gating. Explicit participant masks isolate TP groups; readiness
  requires all configured peers with matching ABI/group identity.
- The existing queue admits explicitly budgeted shared GPU jobs, accounts for
  persistent service identity and ports, rejects unknown consumers, and keeps
  ownership until verified cleanup. Legacy synthetic smoke receipts no longer
  accept teardown/failure as success.

Host evidence from the contributing commits: 46 queue and 3 legacy-receipt tests;
common cache deterministic tests plus seed73/2000 (647261 checks), sanitized
seed1337/2000 (649948 checks), and two killed mutations; mesh2889 checks;
collective TP2/TP3/TP4/TP16 respectively715/985/1392/4495 checks; sanitized mesh
and TP4; eight failures from restoring premature mesh acknowledgement. GLM
module/probe host harnesses and sanitizer checks pass. Host probes validate
control flow and byte ownership, not model numerical correctness.

Still required:

- Propagate physical cache capacity into the CUDA view's bounds.
- Reach arbitrary partial-prefix reuse through common serving validation and
  batch publication, not only the driver probe.
- Execute temporal prefill rows together through each layer with causal
  recurrent/index attention semantics; current execution repeats token waves.
- Implement the required B2+ tree policy. Its test currently fails: TP4 sends
  twelve payloads where a tree permits six.
- Finish and test the real-inference smoke runner, including every listener,
  loaded artifact, configuration/environment and shutdown result.
- Enforce/verify complete device allocation budgets before shared GPU rollout.
  GB10 MemoryMax does not contain all CUDA allocations; declarations and a
  process census alone do not establish a hard device memory bound.
- Real GPU numerical parity, graph replay/failure, TP4/TP16/TP4xPP4 serving,
  concurrent different-model inference and sustained performance qualification.

The existing fleet services have not been changed or restarted. The current
unbounded fleet-agent GPU processes correctly block shared admission. Report
functional, setup, unrun hardware and performance results separately.
