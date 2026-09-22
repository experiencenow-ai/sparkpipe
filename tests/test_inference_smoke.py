#!/usr/bin/env python3
import copy
import importlib.util
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
spec = importlib.util.spec_from_file_location("inference_smoke", Path(__file__).resolve().parents[1] / "tools/inference_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeReceipts(unittest.TestCase):
    def setUp(self):
        self.batch = {"requests": [dict(request_id=11, sequence_id=21, output_token_budget=2),
                                   dict(request_id=12, sequence_id=22, output_token_budget=1)], "stop_token_ids": []}
        self.reference = dict(model_id="fixture", model_revision="r1", vocabulary_size=100,
                              eos_token_ids=[9], tokens={"11": [7, 8], "12": [3]})
        self.events = [dict(schema_version=1, event="ready", model_id="fixture", model_revision="r1")]
        for request_id, sequence_id, handle in ((11, 21, 101), (12, 22, 102)):
            self.events.append(dict(schema_version=1, event="accepted", status=0, request_id=request_id,
                                    sequence_id=sequence_id, request_handle=handle))
        for request_id, sequence_id, handle, index, token in ((11, 21, 101, 0, 7), (12, 22, 102, 0, 3), (11, 21, 101, 1, 8)):
            self.events.append(dict(schema_version=1, event="token", status=0, request_id=request_id,
                                    sequence_id=sequence_id, request_handle=handle, token_index=index,
                                    generated_token_count=index + 1, token_id=token, stop_token=False))
        for request_id, sequence_id, handle, count in ((12, 22, 102, 1), (11, 21, 101, 2)):
            self.events.append(dict(schema_version=1, event="completed", status=0, request_id=request_id,
                                    sequence_id=sequence_id, request_handle=handle, generated_token_count=count,
                                    stop_token=False))

    def verify(self, events=None):
        return smoke.verify_events(map(json.dumps, self.events if events is None else events), self.batch, self.reference)

    def test_interleaved_real_event_schema(self):
        self.assertEqual(self.verify(), 3)

    def test_identity_counts_and_tokens_are_independent_oracles(self):
        for index, key, value in ((0, "model_revision", "wrong"), (2, "request_handle", 101),
                                  (3, "request_id", 999), (3, "sequence_id", 22),
                                  (3, "request_handle", 102), (3, "status", 7),
                                  (3, "token_index", 1), (3, "generated_token_count", 8),
                                  (3, "token_id", 99), (3, "token_id", 100),
                                  (7, "generated_token_count", 1), (7, "event", "cancelled")):
            with self.subTest(index=index, key=key, value=value):
                events = copy.deepcopy(self.events)
                events[index][key] = value
                with self.assertRaises(ValueError):
                    self.verify(events)

    def test_missing_duplicate_and_reordered_terminal(self):
        for events in (self.events[:-1], self.events + self.events[-1:],
                       self.events[:3] + [self.events[-1]] + self.events[3:], []):
            with self.assertRaises(ValueError):
                self.verify(events)

    def test_early_eos_is_on_token_event_not_terminal_event(self):
        self.events[3].update(token_id=9, stop_token=True)
        del self.events[5]
        self.events[-1]["generated_token_count"] = 1
        self.reference["tokens"]["11"] = [9]
        self.assertEqual(self.verify(), 2)
        self.events[3]["stop_token"] = False
        with self.assertRaisesRegex(ValueError, "short completion"):
            self.verify()

    def test_full_namespace_includes_secondary_sessions_and_draft(self):
        config = {"draft_bridge_port": 7793, "tp_collective": {
            "listen_port": 63640, "peer_ports": [63640, 63641],
            "session_ports": [[0, 63001], [63002, 0]],
            "session_ports_hc": [[0, 64001], [64002, 0]]}}
        mapping = {str(port): 30000 + index for index, port in enumerate((7793, 63640, 63641, 63001, 63002, 64001, 64002))}
        result = smoke.private_config(config, mapping, [(30000, 30010)], {31000}, "a" * 32)
        self.assertEqual(result["draft_bridge_port"], 30000)
        self.assertEqual(result["tp_collective"]["session_ports_hc"], [[0, 30005], [30006, 0]])
        self.assertEqual(config["draft_bridge_port"], 7793)
        del mapping["64002"]
        with self.assertRaisesRegex(ValueError, "missing private mapping"):
            smoke.private_config(config, mapping, [(30000, 30010)], set(), "a" * 32)

    def test_unhandled_conflicting_and_unreserved_listener_rejected(self):
        for config, mapping, protected in (({"other_port": 7793}, {"7793": 30000}, set()),
                                           ({"listen_port": 7793}, {"7793": 30000}, {30000}),
                                           ({"listen_port": 7793}, {"7793": 40000}, set()),
                                           ({"peer_ports": [1, 2]}, {"1": 30000, "2": 30000}, set())):
            with self.assertRaises(ValueError):
                smoke.private_config(config, mapping, [(30000, 30010)], protected, "a" * 32)

    def test_shutdown_rejects_crash_and_timeout(self):
        for script, expected in (("raise SystemExit(3)", "FAIL"),
                                 ("import signal,time; signal.signal(signal.SIGTERM,lambda *x:exit(0)); print('ready',flush=True); time.sleep(60)", "PASS"),
                                 ("import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)", "FAIL")):
            child = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, start_new_session=True)
            try:
                child.stdout.readline()
                if "SystemExit" in script:
                    child.wait()
                receipt = {"status": "PASS"}
                smoke.stop_owned([child], receipt, timeout=0.1)
                self.assertEqual(receipt["status"], expected)
                self.assertIsNotNone(child.poll())
            finally:
                child.stdout.close()
                if child.poll() is None:
                    child.kill()
                    child.wait()


class SmokePreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="inference-profile-")
        self.addCleanup(self.temporary.cleanup)
        self.source = Path(self.temporary.name)
        self.attempt = uuid.uuid4().hex
        self.root = Path("/tmp/sparkqueue-" + self.attempt)
        self.addCleanup(lambda: shutil.rmtree(self.root, ignore_errors=True))
        config = {"stage_pack_path": "packs/model.pack", "tp_collective": {
            "backend_module_path": "lib/collective.so", "peer_hosts": ["spark0"],
            "peer_ports": [19000], "listen_port": 19000}}
        contents = {"lib/driver.so": "driver", "lib/adapter.so": "adapter", "lib/transport.so": "transport",
                    "lib/collective.so": "collective", "config/model.json": json.dumps(config),
                    "packs/model.pack": "fixture pack", "packs/model.pack.experts": "fixture manifest",
                    "packs/model.pack.sha256": hashlib.sha256(b"fixture pack").hexdigest()}
        for name, text in contents.items():
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        self.deployment = {"schema_version": 2, "eos_token_ids": [9], "coordinator_rank_index": 0,
                           "driver": {"shared_object_path": "lib/driver.so", "program_name": "resident_decode"},
                           "adapter": {"shared_object_path": "lib/adapter.so"},
                           "transport": {"shared_object_path": "lib/transport.so", "control_port_base": 19000},
                           "nodes": [{"rank_index": 0, "runtime_root": str(self.source),
                                      "adapter_configuration_path": "config/model.json", "kv_backing_directory": "/old/cache",
                                      "kv_backing_maximum_bytes": 4096}]}
        self.deployment_path = self.source / "deployment.json"
        self.deployment_path.write_text(json.dumps(self.deployment))
        self.batch_path = self.source / "batch.json"
        self.batch_path.write_text('{"requests":[],"stop_token_ids":[]}')
        self.reference = dict(deployment_sha256=smoke.digest(self.deployment_path), eos_token_ids=[9],
                              environment={}, source_commit="a" * 40, batch_sha256=smoke.digest(self.batch_path),
                              executables={"build/" + name: "b" * 64 for name in
                                           ("sparkpipe_weightd", "sparkpipe_model_residentd", "sparkpipe_model_batch")},
                              ranks=[{"model_device_bytes": 1024,
                                      "assets": {name: smoke.digest(self.source / name) for name in contents if not name.endswith(".pack")}}])
        self.reference_path = self.source / "reference.json"
        self.spec = dict(hosts=["spark0"], port_base=30000, port_map={"19000": 30002}, environment={},
                         deployment=str(self.deployment_path), batch=str(self.batch_path), reference=str(self.reference_path),
                         budgets=dict(weightd_device_bytes=4096, model_device_bytes=1024, expert_pool_bytes=2048, spine_bytes=2048))
        self.environment = dict(SPARK_QUEUE_ATTEMPT=self.attempt, SPARK_QUEUE_RUNTIME_ROOT=str(self.root),
                                SPARK_QUEUE_RANK="0", SPARK_QUEUE_SIZE="1", SPARK_QUEUE_PORTS="30000:30010",
                                SPARK_QUEUE_DEVICE_MEMORY_MIB="1")

    def prepare(self):
        self.reference_path.write_text(json.dumps(self.reference))
        original = smoke.digest
        with patch.object(smoke, "digest", side_effect=lambda p: "b" * 64 if str(p).startswith("build/") else original(p)), \
             patch.object(smoke.subprocess, "check_output", return_value="a" * 40), \
             patch.object(smoke.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            return smoke.prepare(self.spec, self.environment)

    def test_private_runtime_preserves_pack_and_changes_all_owned_paths(self):
        root, rank, hosts, batch, reference, deployment = self.prepare()
        self.assertEqual(rank, 0)
        self.assertTrue((root / "runtime/packs/model.pack").is_symlink())
        self.assertFalse((root / "runtime/config/model.json").is_symlink())
        self.assertEqual(deployment["weightd"]["socket_path"], str(root / "weightd.sock"))
        self.assertEqual(deployment["nodes"][0]["kv_backing_directory"], str(root / "kv"))
        private = json.loads((root / "runtime/config/model.json").read_text())
        self.assertEqual(private["tp_collective"]["listen_port"], 30002)
        self.assertEqual(json.loads((self.source / "config/model.json").read_text())["tp_collective"]["listen_port"], 19000)
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_changed_deployment_invalidates_memory_plan(self):
        self.deployment["runtime_limits"] = {"resident_sequence_capacity": 1000}
        self.deployment_path.write_text(json.dumps(self.deployment))
        with self.assertRaisesRegex(ValueError, "deployment differs"):
            self.prepare()

    def test_unpinned_collective_module_rejected(self):
        del self.reference["ranks"][0]["assets"]["lib/collective.so"]
        with self.assertRaisesRegex(ValueError, "collective module is not pinned"):
            self.prepare()

    def test_changed_asset_rejected(self):
        (self.source / "lib/driver.so").write_text("changed driver")
        with self.assertRaisesRegex(ValueError, "asset differs"):
            self.prepare()

    def test_memory_and_port_reservations_enforced(self):
        self.spec["budgets"]["model_device_bytes"] = 2 * 1024 * 1024
        with self.assertRaisesRegex(ValueError, "exceeds queue reservation"):
            self.prepare()
        self.spec["budgets"]["model_device_bytes"] = 1024
        self.environment["SPARK_QUEUE_PORTS"] = "30000:30001"
        with self.assertRaisesRegex(ValueError, "reserve every listener"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
