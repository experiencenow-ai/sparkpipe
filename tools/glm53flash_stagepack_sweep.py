#!/usr/bin/env python3
"""Placed-stagepack mechanical sweep for the glm53flash arms, node-local.

Runs ON a spark node against ~/sparkdata/<arm>/packs (no ceph, no warm, no
GPU). Per placed pack it measures, in order:

  1. sha256 of the pack bytes (and the .experts sidecar when one sits next
     to it) streamed in 8 MiB chunks;
  2. identity against EVERY placed receipt (output_sha256/sha256) and the
     placed .sha256 sidecar, listing stale co-receipts when a pack matches
     only some of its receipts;
  3. format structure via tools/glm5_next_pack_verify.py --structure-only
     (header geometry, directory walk, payload/scale byte formulas, bounds,
     directory sha256), invoked with the receipt's rank/topology/stage
     fields — or the pack header's own fields when no receipt exists;
  4. generation + forensic metadata: packed_by/note/source receipt fields,
     mtimes, chattr lock state, and any unaccounted sidecar files
     (.partial-*, .premtp-old, .restore-* leftovers are reported as stray,
     never swept as packs).

Emits one JSON object per pack on stdout plus one per arm with unaccounted
stray files. Exit 0 if every pack passed both identity and structure, 1
otherwise. Verify tools run one at a time, nice 10.

Usage:
  python3 glm53flash_stagepack_sweep.py \
      --verify-tool /tmp/g53sweep/glm5_next_pack_verify.py \
      [--arms a.b.c ...] [--root ~/sparkdata]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import struct
import subprocess
import sys

ARMS = [
    "glm53flash.bf16.tp16",
    "glm53flash.fp8.tp16",
    "glm53flash.fp8.tp8",
    "glm53flash.fp8.tp4pp4",
    "glm53flash.nvfp4.tp16",
]
SIDECAR_SUFFIXES = (".sha256", ".experts", ".mtp-tail.bin")
SHA_KEYS = ("output_sha256", "sha256")
BYTES_KEYS = ("file_bytes", "bytes")
META_KEYS = ("kind", "arm", "rank", "tp_degree", "tp_rank", "stage",
             "first_layer", "layer_count", "mtp", "expert_codec", "source",
             "checkpoint", "source_revision", "packed_by", "note", "verify",
             "packer", "emitted_on")


def sha256_stream(path, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb", buffering=0) as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def receipt_pick(receipt, keys):
    for key in keys:
        if key in receipt:
            return receipt[key]
    return None


def sidecar_fields(path):
    raw = open(path).read().split()
    return raw[0], os.path.basename(raw[1]) if len(raw) > 1 else None


def chattr_locked(path):
    try:
        out = subprocess.run(["lsattr", "-d", path], capture_output=True,
                             text=True, timeout=30).stdout.split()
        return bool(out and "i" in out[0])
    except Exception:
        return None


def mtime(path):
    return datetime.datetime.fromtimestamp(
        os.stat(path).st_mtime, datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def structure_verify(verify_tool, pack, tp_degree, tp_rank, stage,
                     first_layer, layer_count, expected_bytes):
    command = [sys.executable, verify_tool, "--pack", pack,
               "--structure-only", "--tp-rank", str(tp_rank),
               "--tp-degree", str(tp_degree),
               "--first-layer", str(first_layer),
               "--layer-count", str(layer_count)]
    if stage is not None:
        command += ["--stage-count", "4", "--stage-index", str(stage)]
    if expected_bytes is not None:
        command += ["--expected-bytes", str(expected_bytes)]
    run = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    checks = [line for line in run.stdout.splitlines()
              if line.startswith(("PASS", "FAIL", "directory sha256"))]
    return run.returncode, checks, (run.stdout + run.stderr).strip()[-2000:]


def header_tp_fields(pack):
    with open(pack, "rb") as fh:
        head = fh.read(96)
    fields = struct.unpack_from("<20I", head, 0)
    file_bytes = struct.unpack_from("<Q", head, 88)[0]
    return {"stage_count": fields[7], "stage_index": fields[8],
            "first_layer": fields[9], "layer_count": fields[10],
            "tp_degree": fields[18], "tp_rank": fields[19],
            "file_bytes": file_bytes}


def sweep_pack(arm, pack, merged, receipt_records, sidecars, verify_tool):
    record = {"arm": arm, "pack": os.path.basename(pack),
              "bytes": os.path.getsize(pack), "pack_mtime": mtime(pack),
              "locked": chattr_locked(pack)}
    record["sha256"] = sha256_stream(pack)
    record["receipt_meta"] = {key: merged.get(key) for key in META_KEYS}
    record["receipts"] = []
    for name, receipt in receipt_records:
        claimed = receipt_pick(receipt, SHA_KEYS)
        record["receipts"].append({
            "name": name, "sha256": claimed,
            "bytes": receipt_pick(receipt, BYTES_KEYS),
            "matches_pack": claimed == record["sha256"]
            and receipt_pick(receipt, BYTES_KEYS) in (None, record["bytes"]),
            "packed_by": receipt.get("packed_by"),
            "note": receipt.get("note")})
    matching = [entry for entry in record["receipts"] if entry["matches_pack"]]
    record["stale_receipts"] = [entry["name"] for entry
                                in record["receipts"]
                                if not entry["matches_pack"]]
    record["sha_matches_receipt"] = bool(matching)
    authority = {}
    if matching:
        authority = next(receipt for name, receipt in receipt_records
                         if name == matching[0]["name"])
    record["rank_source"] = "receipt" if matching else "header"
    header = header_tp_fields(pack)
    stage = authority.get("stage", None if header["stage_count"] == 1
                          else header["stage_index"])
    record["struct_rc"], record["struct_checks"], record["struct_tail"] = \
        structure_verify(
            verify_tool, pack,
            authority.get("tp_degree", header["tp_degree"]),
            authority.get("tp_rank",
                          authority.get("rank", header["tp_rank"])),
            stage,
            authority.get("first_layer", header["first_layer"]),
            authority.get("layer_count", header["layer_count"]),
            receipt_pick(authority, BYTES_KEYS)
            or (header["file_bytes"] if header["file_bytes"]
                == record["bytes"] else None))
    side = sidecars.get(os.path.basename(pack) + ".sha256")
    if side:
        value, name = sidecar_fields(side)
        record["sidecar_sha256"] = value
        record["sidecar_name"] = name
        record["sha_matches_sidecar"] = value == record["sha256"]
        if not record["sha_matches_sidecar"]:
            record["stale_sidecar"] = True
    experts = pack + ".experts"
    if os.path.exists(experts):
        entry = {"bytes": os.path.getsize(experts),
                 "sha256": sha256_stream(experts)}
        pinned = next((receipt["experts_sha256"] for _, receipt in
                       receipt_records if receipt.get("experts_sha256")),
                      None)
        if pinned:
            entry["matches_receipt"] = entry["sha256"] == pinned
        record["experts"] = entry
    passed = (record["sha_matches_receipt"]
              and record["struct_rc"] == 0)
    record["verdict"] = "PASS" if passed else "FAIL"
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=os.path.expanduser("~/sparkdata"))
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--verify-tool", required=True)
    arguments = ap.parse_args()
    fail = False
    for arm in arguments.arms:
        packs_dir = os.path.join(arguments.root, arm, "packs")
        if not os.path.isdir(packs_dir):
            print(json.dumps({"arm": arm, "verdict": "MISSING",
                              "packs_dir": packs_dir}))
            fail = True
            continue
        files = {name: os.path.join(packs_dir, name)
                 for name in os.listdir(packs_dir)
                 if os.path.isfile(os.path.join(packs_dir, name))}
        sidecars = {name: path for name, path in files.items()
                    if name.endswith(".sha256")}
        receipt_records = [(name, json.load(open(path)))
                           for name, path in sorted(files.items())
                           if name.endswith(".receipt.json")]
        merged = {}
        for _, receipt in receipt_records:
            for key, value in receipt.items():
                merged.setdefault(key, value)
        packs = [path for name, path in sorted(files.items())
                 if not name.endswith(SIDECAR_SUFFIXES)
                 and not name.endswith((".json",))
                 and ".partial-" not in name
                 and os.path.getsize(path) >= (1 << 27)]
        for pack in packs:
            try:
                record = sweep_pack(arm, pack, merged, receipt_records,
                                    sidecars, arguments.verify_tool)
            except Exception as error:
                record = {"arm": arm, "pack": os.path.basename(pack),
                          "verdict": "ERROR", "error": repr(error)[:500]}
            print(json.dumps(record), flush=True)
            if record["verdict"] != "PASS":
                fail = True
        accounted = {os.path.basename(path) for path in packs}
        accounted |= {name for name, _ in receipt_records}
        accounted |= {name + ".sha256" for name in accounted}
        accounted |= {name + ".experts" for name in accounted}
        stray = [{"name": name, "bytes": os.path.getsize(path),
                  "mtime": mtime(path)}
                 for name, path in sorted(files.items())
                 if name not in accounted
                 and not name.endswith(SIDECAR_SUFFIXES)]
        if stray:
            print(json.dumps({"arm": arm, "stray": stray}), flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
