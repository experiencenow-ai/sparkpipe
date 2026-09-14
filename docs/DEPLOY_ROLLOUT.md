# Fleet rollout: multi-dev deploy protocol

Companion to `docs/DEPLOY_PROTOCOL_MAP.md` (as-deployed evidence) and
`docs/DEPLOY_MULTI_DEV_PROPOSAL.md` (approved items 0-6, now implemented
in `tools/fleet_node_agent.sh`, `tools/publish_local.sh`,
`tools/publish_core.sh`, `tools/module_build_release.sh`,
`tools/weightsd_announce.sh`, `tools/fleet_release_hygiene.sh`). This
runbook is NOT executed by the PR that lands the code: the hub steps
below touch live fleet state and need coredev informed beforehand (the
operator owns that notification, including the retirement of the
out-of-band `tools_local/{p,b}` deploy session).

Scope of the landed code, for reference during rollout:

- Node agent (item 0.3/0.4, 1, 2, 3, 4): content-hash diff + fetch +
  scope before any restart; TERM-drain with `kill -9` deadline fallback
  (residentd/api only, never weightd); heartbeat carries
  epoch/load/mem_avail_gb + per-root pid/rss_mb/log_age_s and emits
  valid JSON (the repo agent's `roots:` key was unquoted — fleet_view
  silently skipped repo-agent heartbeats); `install_core` no longer
  kills or starts weightd — it only installs a core-channel weightd
  whose sha16 equals the published `WEIGHTSD_BIN` (absent = inert).
  The deployed-agent `$home` unbound-variable crash (map 0.3) is gone
  because the weightd start path no longer exists in the agent.
- Publishers (items 4, 6): `module_build_release.sh` no longer builds
  weightd at all and ends in `publish_core.sh agent`;
  `publish_core.sh` requires an explicit mode (`agent` or `weightd` —
  weightd mode is the weights lane only); `publish_local.sh` stages
  into `~/release/<root>/.staging/`, moves files into place, writes
  MANIFEST last (commit bit); per-root flock in `publish_local.sh`,
  global publish flock in `module_build_release.sh` (shared build
  tree) and `publish_core.sh`.
- weightd lifecycle rule (item 4, exact): weightd restarts ONLY on a
  published `WEIGHTSD_BIN` sha16. coredev builds and publishes the
  candidate (`publish_core.sh weightd`), coredev publishes the sha
  (`tools/weightsd_announce.sh`), mgr2 syncs weightsd deliberately.
  NOTHING auto-builds or auto-restarts weightd from any module release,
  and the node agent never kills or starts weightd; `ensure_weightd`
  remains a down-state watchdog only.

## Step 0 — Preconditions

1. PR `lane/deploy-protocol-multi-dev` merged to main; hub build tree
   updated (`git -C ~/sparkpipe-build fetch && git reset --hard main`).
2. `bash tests/test_deploy_restart_scope.sh` green on the hub checkout
   (52 checks; needs no GPU or fleet access).
3. coredev informed: W58/`tools_local` retirement (this step 1) and the
   weightsd handoff ceremony (step 4).
4. Snapshot rollback material (see Rollback): the deployed agent
   `71ec91d332f2f076` still exists on sparkf at
   `~/sparkpipe-build-958/tools_local/fleet_node_agent.sh` — copy it to
   `~/release/core/bin/fleet_node_agent.sh.rollback-71ec91d3` BEFORE
   the first `publish_core.sh` run overwrites `~/release/core/bin/`.

## Step 1 — Hub reconcile (retire out-of-band path)

Coordinate with coredev (operator informs him). Then on sparkf:

1. Retire the out-of-band deploy tools:
   `mv ~/sparkpipe-build-958/tools_local/{p,b,fullbuild_once.sh,module_build_release.sh} \
   ~/sparkpipe-build-958/tools_local/retired-$(date +%Y%m%d)/`
   From this point the only deploy path is the repo publish tooling
   (in-band, per-channel). Nothing streams tarballs or kills by
   exe/cwd pattern anymore.
2. Reconcile the core channel with installed reality (map 0.2): the
   hub `~/release/core/bin/sparkpipe_weightd` (mtime Sep 14, pushed by
   the retired `p` without a MANIFEST regen) is a CANDIDATE. Leave it;
   do NOT announce it (`WEIGHTSD_BIN` must be absent — verify:
   `ssh sparkf 'cat ~/release/core/WEIGHTSD_BIN'` must fail). The new
   agent installs core-channel weightd only when the sha equals the
   announced value, so the stale candidate is inert.

## Step 2 — Staged agent rollout (spark0 first)

The core MANIFEST is fleet-wide, so staging is done node-locally: the
agent self-updates from `~/sparkdata/core/bin/` regardless of the hub
manifest.

1. Publish the new agent into the hub core channel:
   on sparkf, `tools/publish_core.sh agent` (repo checkout). Expected
   first-run side effect: the regenerated core MANIFEST now also lists
   the candidate weightd — nodes will fetch it once and NOT install
   (no WEIGHTSD_BIN), zero restarts. Watch one node's journal to
   confirm: `journalctl --user -u fleet-agent | tail` shows
   "manifest changed; syncing diff" with no restart lines.
2. Stage on spark0 only:
   `scp sparkf:~/release/core/bin/fleet_node_agent.sh spark0:~/sparkdata/core/bin/`
   The running agent execs the new file within ~1s (self_update).
3. Verify spark0 (rank 0, api host):
   - `curl -s sparkf:8801/host/spark0` now shows `epoch`, `load`,
     `mem_avail_gb`, and per-root `pid`, `rss_mb`, `log_age_s`;
     `/summary` `age_s` becomes meaningful (staleness detection).
   - `systemctl --user status fleet-agent` active; residentd state
     `ready`; api healthy on :8433.
   - Live weightd-no-restart proof: touch a stage config into the glm
     release root (`publish_local.sh` with only a config change, or
     edit `~/release/glm53flash.fp8.tp16/config/stage_05.json`), then
     within ~15s check spark0's heartbeat: residentd `pid` unchanged,
     root state stays `ready`, `weightd` sha unchanged. Receipts via
     D-1's execute rig (spark_queue dispatch, per-command exit + log
     receipt) if operator wants the paper trail.

## Step 3 — Fleet agent rollout

1. `publish_core.sh agent` already put the agent in the core channel
   (step 2.1). All remaining nodes converge via core MANIFEST within
   ~1-2s each; `self_update` execs the new agent.
2. Watch for a clean wave: on the hub,
   `for h in spark*; do echo "== $h"; ssh $h 'journalctl --user -u fleet-agent -n 5 --no-pager'; done`
   Expect "self-updating" lines, no restart lines (no publish is in
   flight), no weightd churn anywhere.
3. Confirm all 16 heartbeats show the new `agent` sha and the new
   schema fields.

## Step 4 — Per-root layout cut-over (items 1+2 acceptance)

Per-driver release roots already exist for the active driver. For each
additional driver (k3, laguna, ...):

1. Create its hub root: driver's lane publishes with its own root name
   (`tools/module_build_release.sh FAMILY CODEC <driver-root> ...`).
   The per-root flock makes concurrent publishes by different devs on
   different roots safe; the same root serializes.
2. Point each node's unit at its driver's root only:
   edit `~/.config/systemd/user/fleet-agent.service` ExecStart arg 1
   (CSV for multi-root/coredev-style nodes), `daemon-reload`, restart
   the unit. Default = the node's own driver root.
3. Isolation proof (the acceptance gate): publish to the
   glm53flash.fp8.tp16 root, then verify k3- and laguna-subscribed
   nodes did NOTHING: their heartbeats' residentd pids, residentd shas,
   driver shas and weightd shas byte-identical before/after, and their
   journals show no "manifest changed" line (their roots' MANIFESTs
   were untouched — per-channel by construction, since a publish only
   ever writes `~/release/<root>/`).
4. Restart-scope proof on a subscribed node: config-only publish ->
   zero daemon restarts (pids unchanged); driver .so publish -> one
   residentd drain/restart (TERM first — journal shows the drain, not
   a SIGKILL), zero weightd restarts; api-only publish -> api drain on
   rank 0 only.

## Step 5 — weightsd handoff (item 4 ceremony)

1. Bootstrap weightsd as a system unit on each node (one-time,
   coredev-owned daemon, already running by hand on some nodes):
   `ExecStart=$HOME/weightsd1/build/sparkpipe_weightsd --socket /run/sparkpipe-weightsd/weightsd.sock`,
   `Restart=always`. mgr2 executes the bootstrap; coredev confirms the
   daemon build to use.
2. From here the ONLY weightd change path is: weights lane builds
   weightd -> `tools/publish_core.sh weightd` (candidate, inert) ->
   coredev announces `tools/weightsd_announce.sh` (writes
   `~/release/core/WEIGHTSD_BIN`) -> mgr2 syncs weightsd deliberately
   (weightsd installs + drain-restarts weightd onto the announced sha)
   -> convergence check before declaring done: every
   `current/spark*.json` `weightd` field equals the announced sha16
   (same retire-guard discipline as `fleet_sync.sh retire-update`).
3. `ensure_weightd` stays as the down-state watchdog: a dead weightd is
   started from `~/sparkdata/weightd/sparkpipe_weightd`; a RUNNING
   weightd is never touched by the agent.

## Step 6 — NVMe retention (item 5)

1. Dry run from the hub: `tools/fleet_release_hygiene.sh` (default
   age gate 14d). Reviews the 40+ stale roots (k3.*, qwen*, glm53full.*,
   dsv4_* — ~1.6 TB on spark1) and packs debris (`.old`, `.experts.old`,
   `.partial-*`, `.premtp-old`), gated on the fleet-wide ROOTS union.
2. Operator reviews the list, then runs
   `tools/fleet_release_hygiene.sh --apply` explicitly (interactive
   DELETE confirmation; pack debris is deleted only after the
   immutable-set check passes per root).
3. Convention going forward: packs live only on node NVMe
   (`~/sparkdata/<root>/packs`, `chattr +i`); sidecar debris that would
   break the digest scan is removed by this tool, not by hand. Ceph is
   the warm SOURCE for packs (content-addressed, sha256-verified into
   NVMe) — the pipeline integration is a separate, coredev-coordinated
   change; until it lands, host-to-node pack placement continues as
   today.

## Rollback

- Node-level (agent): `scp sparkf:~/release/core/bin/fleet_node_agent.sh.rollback-71ec91d3 \
  <node>:~/sparkdata/core/bin/fleet_node_agent.sh` — the running agent
  self-updates to the old binary within ~1s. Restores the kill-based
  agent; a fleet-wide bounce after rollback is expected and acceptable.
- Hub-level (agent channel): copy the rollback binary into
  `~/release/core/bin/fleet_node_agent.sh` and re-run
  `tools/publish_core.sh agent` (regenerates the core MANIFEST so
  nodes re-converge to the old agent and stay there).
- Publishers: `git -C ~/sparkpipe-build reset --hard` to the pre-merge
  main; old `publish_local.sh` semantics (direct install, MANIFEST
  last) are wire-compatible with the new agent.
- `WEIGHTSD_BIN`: removing the file makes `install_core` inert again;
  weightsd keeps running the last installed weightd (the binary file is
  never removed by rollout steps).

## Final verification checklist

1. `/summary` on :8801 shows real, increasing-staleness `age_s` for all
   16 hosts (epoch flowing; valid JSON — no skipped heartbeats).
2. Config-only publish: zero daemon restarts fleet-wide (all pids
   unchanged in heartbeats).
3. Driver .so publish: exactly 16 residentd restarts, 0 weightd
   restarts, 0 api restarts off rank 0.
4. Per-channel isolation: a glm53flash.fp8.tp16 publish does not
   restart k3's or laguna's residentd or weightd on any node.
5. Core publish with no WEIGHTSD_BIN: weightd pids/shas unchanged on
   all 16 (install_core inert).
6. Two devs publish different roots concurrently: both succeed (per-
   root locks independent), both MANIFESTs well-formed, no interleaved
   trees (`.staging` never visible in a MANIFEST).
7. Hygiene dry run reviewed by the operator; nothing deleted without
   an explicit `--apply` + DELETE confirmation.
