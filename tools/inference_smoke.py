#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time
from glm5_next_bench_wrap import measure


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_events(lines, batch, reference):
    requests = {row["request_id"]: row for row in batch["requests"]}
    require(len(requests) == len(batch["requests"]), "duplicate request IDs")
    require(set(map(str, requests)) == set(reference["tokens"]), "reference request set differs")
    observed, handles, completed = {}, set(), set()
    ready = 0
    for line in lines:
        event = json.loads(line)
        require(event.get("schema_version") == 1, "invalid event schema")
        kind = event.get("event")
        if kind == "ready":
            ready += 1
            require(ready == 1 and not observed, "duplicate or late readiness")
            require(all(event.get(key) == reference[key] for key in ("model_id", "model_revision")), "model identity differs from reference")
            continue
        request_id = event.get("request_id")
        require(ready == 1 and request_id in requests, "event has unknown request identity")
        request = requests[request_id]
        require(event.get("status") == 0 and event.get("sequence_id") == request["sequence_id"], "failed event or wrong sequence")
        require(request_id not in completed, "event after completion")
        handle = event.get("request_handle")
        if kind == "accepted":
            require(request_id not in observed and isinstance(handle, int) and handle > 0 and handle not in handles, "duplicate acceptance or handle")
            observed[request_id] = {"handle": handle, "tokens": [], "stopped": False}
            handles.add(handle)
            continue
        require(request_id in observed and handle == observed[request_id]["handle"], "event ownership differs")
        tokens = observed[request_id]["tokens"]
        if kind == "token":
            require(not observed[request_id]["stopped"], "token after stop token")
            require(event.get("token_index") == len(tokens) and event.get("generated_token_count") == len(tokens) + 1, "noncontiguous token event")
            token = event.get("token_id")
            require(isinstance(token, int) and 0 <= token < reference["vocabulary_size"], "invalid token ID")
            tokens.append(token)
            observed[request_id]["stopped"] = event.get("stop_token") is True
        elif kind == "completed":
            require(event.get("generated_token_count") == len(tokens), "completion token count differs")
            require(tokens == reference["tokens"][str(request_id)], "generated tokens differ from pinned reference")
            require(0 < len(tokens) <= request["output_token_budget"], "invalid output length")
            if len(tokens) < request["output_token_budget"]:
                require(observed[request_id]["stopped"] and tokens[-1] in reference["eos_token_ids"] + batch["stop_token_ids"], "unexplained short completion")
            completed.add(request_id)
        else:
            raise ValueError("unexpected event: " + str(kind))
    require(completed == set(requests), "missing terminal requests")
    return sum(len(row["tokens"]) for row in observed.values())


def private_deployment(deployment, root, base, hosts):
    value = json.loads(json.dumps(deployment))
    count = len(hosts)
    require(len(value["nodes"]) == count and 1 <= count <= 16, "queue and deployment topology differ")
    value["weightd"] = {"socket_path": str(root / "weightd.sock")}
    value["transport"]["control_port_base"] = base
    for rank, node in enumerate(value["nodes"]):
        require(node["rank_index"] == rank, "deployment ranks must be ordered")
        node["runtime_root"] = str(root / "runtime")
        node["transport_host"] = hosts[rank]
        node["control_endpoint"] = {"kind": "tcp", "host": hosts[rank], "port": base + count + rank}
        if node.get("kv_backing_directory") is not None:
            require(node["kv_backing_maximum_bytes"] > 0, "unbounded backing store")
            node["kv_backing_directory"] = str(root / "kv")
    return value


def private_config(config, port_map, reserved, protected, attempt):
    port_names = {"listen_port", "peer_ports", "session_ports", "session_ports_hc", "draft_bridge_port"}
    destinations = list(port_map.values())
    require(len(set(destinations)) == len(destinations), "private ports alias distinct original listeners")
    require(all(isinstance(port, int) and port not in protected and any(first <= port <= last for first, last in reserved) for port in destinations), "private configuration uses unreserved or conflicting ports")

    def ports(value):
        if isinstance(value, list):
            return [ports(item) for item in value]
        require(isinstance(value, int), "invalid port value")
        if value == 0:
            return 0
        require(str(value) in port_map, "missing private mapping for port " + str(value))
        return port_map[str(value)]

    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {}
        for key, item in value.items():
            require(not (key.endswith("_port") or key.endswith("_ports") or key.startswith("session_ports")) or key in port_names, "unhandled listener configuration: " + key)
            result[key] = ports(item) if key in port_names else rewrite(item)
        return result

    result = rewrite(config)
    if result.get("tp_collective"):
        result["tp_collective"]["collective_identifier"] = int(attempt[:15], 16) + 1
    return result


def prepare(spec, environment):
    attempt = environment.get("SPARK_QUEUE_ATTEMPT", "")
    require(re.fullmatch(r"[0-9a-f]{32}", attempt), "run through the authoritative spark queue")
    root = Path(environment["SPARK_QUEUE_RUNTIME_ROOT"])
    require(str(root) == "/tmp/sparkqueue-" + attempt, "unexpected job namespace")
    rank, size = int(environment["SPARK_QUEUE_RANK"]), int(environment["SPARK_QUEUE_SIZE"])
    hosts = spec["hosts"]
    require(size == len(hosts) and 0 <= rank < size and len(set(hosts)) == size and all(re.fullmatch(r"spark[0-9a-f]", host) for host in hosts), "invalid participant list")
    base = spec["port_base"]
    require(isinstance(base, int) and 1024 <= base <= 65535 - 3 * size, "invalid private port range")
    reserved = [tuple(map(int, value.split(":"))) for value in environment.get("SPARK_QUEUE_PORTS", "").split(",") if value]
    require(all(any(first <= port <= last for first, last in reserved) for port in range(base, base + 3 * size + 1)), "queue does not reserve every listener")
    device = int(environment["SPARK_QUEUE_DEVICE_MEMORY_MIB"]) * 1024 * 1024
    limits = spec["budgets"]
    require(all(isinstance(limits[key], int) and limits[key] > 0 for key in ("weightd_device_bytes", "model_device_bytes", "expert_pool_bytes", "spine_bytes")), "all device budgets must be finite and positive")
    require(limits["weightd_device_bytes"] + limits["model_device_bytes"] <= device, "device plan exceeds queue reservation")
    require(limits["expert_pool_bytes"] + limits["spine_bytes"] <= limits["weightd_device_bytes"], "weightd working set exceeds ceiling")
    deployment = json.loads(Path(spec["deployment"]).read_text())
    batch = json.loads(Path(spec["batch"]).read_text())
    reference = json.loads(Path(spec["reference"]).read_text())
    require(reference["deployment_sha256"] == digest(spec["deployment"]), "deployment differs from pinned allocation plan")
    require(reference["eos_token_ids"] == deployment["eos_token_ids"], "EOS metadata differs from reference")
    require(reference["environment"] == spec["environment"], "model environment differs from reference")
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    require(source_commit == reference["source_commit"], "source differs from reference")
    require(subprocess.run(["git", "diff", "--quiet", "HEAD"]).returncode == 0, "source checkout is dirty")
    require(set(reference["executables"]) == {"build/sparkpipe_weightd", "build/sparkpipe_model_residentd", "build/sparkpipe_model_batch"}, "pin every smoke executable")
    for name, sha in reference["executables"].items():
        require(digest(name) == sha, "executable differs from reference: " + name)
    require(reference["batch_sha256"] == digest(spec["batch"]), "batch differs from pinned reference")
    plan = reference["ranks"][rank]
    require(plan["model_device_bytes"] == limits["model_device_bytes"], "model allocation plan differs from pinned budget")
    source = Path(deployment["nodes"][rank]["runtime_root"])
    assets = plan["assets"]
    config_name = deployment["nodes"][rank]["adapter_configuration_path"]
    required = {deployment[key]["shared_object_path"] for key in ("driver", "adapter", "transport")} | {config_name}
    require(required <= assets.keys(), "reference does not pin every loaded binary/configuration")
    for name, sha in assets.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "asset escapes runtime root")
        require(digest(source / name) == sha, "asset differs from reference: " + name)
    config = json.loads((source / config_name).read_text())
    backend_module = config.get("tp_collective", {}).get("backend_module_path")
    if backend_module:
        require(backend_module in assets, "collective module is not pinned")
    pack = config["stage_pack_path"]
    require(not Path(pack).is_absolute() and ".." not in Path(pack).parts, "pack escapes runtime root")
    require(pack + ".sha256" in assets and pack + ".experts" in assets, "lazy pack identity and experts manifest must be pinned")
    require((source / pack).is_file(), "stage pack is missing")
    root.mkdir(mode=0o700)
    runtime = root / "runtime"
    runtime.mkdir()
    for name in set(assets) | {pack}:
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if name != config_name:
            target.symlink_to((source / name).resolve())
    peers = config.get("tp_collective", {}).get("peer_hosts", [])
    require(all(host in hosts for host in peers), "collective references unreserved peers")
    protected = set(range(base, base + 2 * size)) | {base + 3 * size}
    config = private_config(config, spec["port_map"], reserved, protected, attempt)
    (runtime / config_name).write_text(json.dumps(config) + "\n")
    (root / "kv").mkdir()
    (root / "mesh").mkdir()
    value = private_deployment(deployment, root, base, hosts)
    atomic_json(root / "deployment.json", value)
    atomic_json(root / "batch.json", batch)
    return root, rank, hosts, batch, reference, value


def remote_bytes(host, path):
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", host,
                             shlex.join(["cat", str(path)])], capture_output=True, timeout=6)
    return result.stdout if result.returncode == 0 else None


def stop_owned(children, receipt, timeout=10):
    for child in reversed(children):
        if child.poll() is not None:
            receipt.update(status="FAIL", error="owned daemon exited before shutdown: " + str(child.returncode))
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if child.poll() is None:
            try:
                child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                receipt.update(status="FAIL", error="owned daemon did not drain within shutdown deadline")
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            if child.returncode != 0:
                receipt.update(status="FAIL", error="owned daemon shutdown failed: " + str(child.returncode))
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            pass
        else:
            receipt.update(status="FAIL", error="owned process group outlived daemon")
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def run(spec):
    root, rank, hosts, batch, reference, deployment = prepare(spec, os.environ)
    deadline = time.monotonic() + spec["timeout_seconds"]
    require(0 < spec["timeout_seconds"] <= 840, "invalid smoke timeout")
    children, logs = [], []
    receipt = {"schema_version": 1, "rank": rank, "attempt": os.environ["SPARK_QUEUE_ATTEMPT"],
               "reference_sha256": digest(spec["reference"]), "status": "FAIL"}
    env = {key: value for key, value in os.environ.items() if not key.startswith("SPARK_")}
    limits = spec["budgets"]
    env.update(SPARK_WEIGHTD_ATTACH="1", SPARK_WEIGHTD_SOCKET=str(root / "weightd.sock"),
               SPARK_WEIGHTD_DEVICE_BYTES_MAX=str(limits["weightd_device_bytes"]),
               SPARK_WEIGHTD_EXPERT_POOL_BYTES=str(limits["expert_pool_bytes"]),
               SPARK_WEIGHTD_SPINE_BUDGET_BYTES=str(limits["spine_bytes"]),
               SPARK_WEIGHTD_KV_RESERVE_BYTES="0", SPARK_WEIGHTD_MESH_DIR=str(root / "mesh"),
               SPARK_WEIGHTD_LATCH_PORT=str(spec["port_base"] + 3 * len(hosts)))
    owned = {
        "SPARK_WEIGHTD_ATTACH", "SPARK_WEIGHTD_SOCKET", "SPARK_WEIGHTD_DEVICE_BYTES_MAX",
        "SPARK_WEIGHTD_EXPERT_POOL_BYTES", "SPARK_WEIGHTD_SPINE_BUDGET_BYTES",
        "SPARK_WEIGHTD_KV_RESERVE_BYTES", "SPARK_WEIGHTD_MESH_DIR", "SPARK_WEIGHTD_LATCH_PORT"}
    require(all(key.startswith("SPARK_") and not key.startswith("SPARK_QUEUE_") and key not in owned and isinstance(value, str) for key, value in spec["environment"].items()), "model environment overrides owned job configuration")
    env.update(spec["environment"])
    receipt.update(effective_deployment_sha256=digest(root / "deployment.json"),
                   effective_config_sha256=digest(root / "runtime" / deployment["nodes"][rank]["adapter_configuration_path"]),
                   source_commit=reference["source_commit"], budgets=limits, environment=spec["environment"])

    def start(name, command):
        log = (root / (name + ".log")).open("wb")
        logs.append(log)
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        children.append(child)
        return child

    def wait(check):
        while time.monotonic() < deadline:
            require(all(child.poll() is None for child in children), "owned daemon exited; inspect job logs")
            result = check()
            if result:
                return result
            time.sleep(0.1)
        raise TimeoutError("inference smoke deadline expired")

    try:
        command = ["build/sparkpipe_weightd", "--socket", str(root / "weightd.sock"), "--device-bytes-max", str(limits["weightd_device_bytes"])]
        if len(hosts) > 1:
            mesh = spec["mesh"]
            command += ["--mesh-rank", str(rank), "--mesh-rank-mask", str((1 << len(hosts)) - 1), "--mesh-interface", mesh["interface"], "--mesh-sgid-index", str(mesh["sgid_index"]), "--mesh-dir", str(root / "mesh")]
        start("weightd", command)
        wait(lambda: "spark_weightd ready " in (root / "weightd.log").read_text(errors="replace"))
        if len(hosts) > 1:
            for peer, host in enumerate(hosts):
                if peer == rank:
                    continue
                path = root / "mesh" / f"mesh-{peer:x}.rec"
                payload = wait(lambda: remote_bytes(host, path))
                require(0 < len(payload) <= 4096, "invalid peer mesh record")
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(payload)
                temporary.replace(path)
        start("residentd", ["build/sparkpipe_model_residentd", "--deployment", str(root / "deployment.json"), "--rank-index", str(rank)])
        wait(lambda: f"model_residentd ready rank={rank} " in (root / "residentd.log").read_text(errors="replace"))
        atomic_json(root / "ready.json", {"attempt": receipt["attempt"], "rank": rank, "reference_sha256": receipt["reference_sha256"]})
        coordinator = deployment["coordinator_rank_index"]
        if rank == coordinator:
            for peer, host in enumerate(hosts):
                peer_ready = json.loads(wait(lambda: remote_bytes(host, root / "ready.json"))) if peer != rank else json.loads((root / "ready.json").read_text())
                require(peer_ready == {"attempt": receipt["attempt"], "rank": peer, "reference_sha256": receipt["reference_sha256"]}, "peer runs a different smoke job")
            with (root / "batch.events.jsonl").open("wb") as output:
                result = measure(["build/sparkpipe_model_batch", "--deployment", str(root / "deployment.json"), "--runtime-root", str(root / "runtime"), "--batch", str(root / "batch.json"), "--profile-stages"], max(1, deadline - time.monotonic()), env=env, event_sink=output, stderr_path=root / "batch.stderr.log")
            require(result["valid"], "model_batch failed: " + str(result["errors"]))
            receipt["tokens"] = verify_events((root / "batch.events.jsonl").read_text().splitlines(), batch, reference)
            receipt["timing"] = result
            atomic_json(root / "inference.json", {"attempt": receipt["attempt"], "reference_sha256": receipt["reference_sha256"], "tokens": receipt["tokens"]})
        else:
            outcome = json.loads(wait(lambda: remote_bytes(hosts[coordinator], root / "inference.json")))
            require(outcome["attempt"] == receipt["attempt"] and outcome["reference_sha256"] == receipt["reference_sha256"], "coordinator result identity differs")
            receipt["tokens"] = outcome["tokens"]
        receipt["status"] = "PASS"
    except (OSError, ValueError, KeyError, TimeoutError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
    finally:
        stop_owned(children, receipt)
        for log in logs:
            log.close()
        atomic_json(root / "receipt.json", receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return int(receipt["status"] != "PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(run(json.loads(args.spec.read_text())))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
