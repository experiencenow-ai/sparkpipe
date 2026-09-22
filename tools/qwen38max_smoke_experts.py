#!/usr/bin/env python3
"""Generate model-families/qwen38_max/smoke_experts.json (machine-generated, PR'd).

The routed-expert working set is derived from the T1 reference fixtures in
qualification/t1_reference/qwen38_max/ (gate-qualified in PR #1072): each
fixture records the per-position, per-layer top-k route ids the recorded
smoke prompt set actually touched under the merged streamed decoder. The
deduplicated (layer, expert) union is exactly the batch-preload working set
for the < 5 s cold-launch goal.

Per-expert and spine byte counts are MEASURED from the checkpoint
safetensors headers (payload + weight_scale + weight_scale_2 extents for
each routed expert; every non-".mlp.experts." tensor counts toward the
full-resolution spine, quality law). Header reads only - no weight data
is touched, so the scan is safe next to the shared warm mount.

Output follows the lane-0 smoke-expert manifest convention consumed by
tools/devcycle/lane_budget_calc.py. Never hand-edit the output.

Modes:
  --checkpoint DIR   measure bytes from the checkpoint index now
  --measured FILE    reuse a --scan-only table instead of the checkpoint
  --scan-only        with --checkpoint: print the measured table as JSON
                     (run this on the node that holds the checkpoint)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(ROOT, "qualification", "t1_reference", "qwen38_max")
DEFAULT_OUTPUT = os.path.join(ROOT, "model-families", "qwen38_max", "smoke_experts.json")
EXPERT_MARK = ".mlp.experts."
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
T1R_MAGIC = b"T1R1"


def read_i32_arrays(path: str) -> dict[str, list[int]]:
    """Minimal stdlib T1R1 reader for the I32 arrays this tool consumes."""
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != T1R_MAGIC:
        raise SystemExit(f"{path}: not a T1R1 fixture")
    meta_len = struct.unpack("<Q", data[4:12])[0]
    meta = json.loads(data[12:12 + meta_len])
    offset = 12 + meta_len
    arrays: dict[str, list[int]] = {}
    for entry in meta["arrays"]:
        comp_len = struct.unpack("<Q", data[offset:offset + 8])[0]
        offset += 8
        blob = zlib.decompress(data[offset:offset + comp_len])
        offset += comp_len
        if entry["dtype"] != "I32" or len(blob) != entry["bytes"]:
            continue
        arrays[entry["name"]] = list(struct.unpack(f"<{len(blob) // 4}i", blob))
    return arrays


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
# Full-attention KV per token per rank at TP16: 23 layers x 4 kv heads
# (replicated x4 over 16 ranks) x 256 head dim x (K+V) x BF16.
KV_BYTES_PER_TOKEN_PER_RANK = 23 * 4 * 256 * 2 * 2
# Decode-slack multiplier over the recorded smoke token counts: covers
# regeneration runs and small-batch replays of the same prompt set.
KV_FLOOR_HEADROOM = 64
WORKSPACE_BYTES = 512 * 1024 * 1024  # allowance until the wrapper lands


def scan_checkpoint(checkpoint: str) -> dict:
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise SystemExit(f"missing index: {index_path}")
    weight_map = json.load(open(index_path))["weight_map"]
    headers: dict[str, dict] = {}

    def file_header(fname: str) -> dict:
        if fname not in headers:
            path = os.path.join(checkpoint, fname)
            with open(path, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                headers[fname] = json.loads(fh.read(size))
        return headers[fname]

    def extent(name: str) -> int:
        entry = file_header(weight_map.get(name, "model.safetensors")).get(name)
        if entry is None:
            raise SystemExit(f"tensor {name} absent from checkpoint index")
        return int(entry["data_offsets"][1]) - int(entry["data_offsets"][0])

    spine_bytes = 0
    per_expert: dict[str, int] = {}
    sizes_seen: dict[int, int] = {}
    for name in sorted(weight_map):
        size = extent(name)
        if EXPERT_MARK not in name:
            spine_bytes += size
            continue
        # model.layers.L.mlp.experts.E.<proj>.<suffix>
        tail = name.split(EXPERT_MARK, 1)[1]
        pieces = tail.split(".")
        if len(pieces) < 4 or pieces[2] not in PROJECTIONS or pieces[3] not in (
                "weight", "weight_scale", "weight_scale_2"):
            raise SystemExit(f"unexpected expert tensor name: {name}")
        key = f"{pieces[0]}/{pieces[1]}"
        per_expert[key] = per_expert.get(key, 0) + size
        sizes_seen.setdefault(len(per_expert), 0)
    uniform = len(set(per_expert.values())) == 1
    return {"per_expert": per_expert, "spine_bytes": spine_bytes,
            "expert_sizes_uniform": uniform,
            "tensor_count": len(weight_map)}


def touched_experts(fixtures_dir: str) -> tuple[dict, dict]:
    manifest = json.load(open(os.path.join(fixtures_dir, "MANIFEST.json")))
    touched: set[tuple[int, int]] = set()
    max_tokens = 0
    for fixture_name in sorted(manifest["fixtures"]):
        path = os.path.join(fixtures_dir, fixture_name)
        arrays = read_i32_arrays(path)
        prompt_ids = arrays.get("prompt_token_ids")
        generated = arrays.get("generated_token_ids")
        if prompt_ids is None or generated is None:
            raise SystemExit(f"{fixture_name}: missing token id arrays")
        max_tokens = max(max_tokens, len(prompt_ids) + len(generated))
        for name, values in arrays.items():
            if not name.endswith("_route_ids"):
                continue
            layer = int(name.split("_layer", 1)[1].split("_", 1)[0])
            for expert in values:
                touched.add((layer, int(expert)))
    return touched, {"max_total_tokens": max_tokens,
                     "fixture_shas": {name: manifest["fixtures"][name]["sha256"]
                                      for name in sorted(manifest["fixtures"])}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures-dir", default=FIXTURES_DIR)
    parser.add_argument("--checkpoint", help="checkpoint dir with model.safetensors.index.json")
    parser.add_argument("--measured", help="JSON table produced by --scan-only")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    if arguments.scan_only:
        if not arguments.checkpoint:
            raise SystemExit("--scan-only requires --checkpoint")
        print(json.dumps(scan_checkpoint(arguments.checkpoint), sort_keys=True))
        return 0
    if arguments.checkpoint:
        measured = scan_checkpoint(arguments.checkpoint)
    elif arguments.measured:
        measured = json.load(open(arguments.measured))
    else:
        raise SystemExit("need --checkpoint or --measured")

    touched, prompt_info = touched_experts(arguments.fixtures_dir)
    experts = []
    for layer, expert in sorted(touched):
        key = f"{layer}/{expert}"
        if key not in measured["per_expert"]:
            raise SystemExit(f"expert {key} touched by fixtures but absent from checkpoint")
        experts.append({"layer": layer, "expert": expert,
                        "codec": "nvfp4_e4m1_e4m3g16",
                        "bytes": measured["per_expert"][key]})

    kv_floor_bytes = KV_BYTES_PER_TOKEN_PER_RANK * prompt_info["max_total_tokens"] \
        * KV_FLOOR_HEADROOM
    document = {
        "schema_version": 1,
        "family": "qwen38_max",
        "prompt_set": "t1-smoke-v1",
        "topology": "TP16",
        "nodes": 16,
        "expert_shard": "tp",
        "spine_bytes": measured["spine_bytes"],
        "kv_floor_bytes": kv_floor_bytes,
        "workspace_bytes": WORKSPACE_BYTES,
        "experts": experts,
        "provenance": {
            "generator": "tools/qwen38max_smoke_experts.py",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "routing_source": "T1R1 fixtures (PR #1072 gate-qualified streamed decoder)",
            "fixture_dir": os.path.relpath(arguments.fixtures_dir, ROOT),
            "fixture_sha256": prompt_info["fixture_shas"],
            "max_total_tokens": prompt_info["max_total_tokens"],
            "kv_floor_formula": ("23 full-attn layers x 4 kv heads x 256 head dim "
                                 f"x (K+V) x BF16 x max_total_tokens x {KV_FLOOR_HEADROOM}"),
            "workspace_note": "allowance pending wrapper measurements",
            "expert_sizes_uniform": measured.get("expert_sizes_uniform"),
            "checkpoint_tensor_count": measured.get("tensor_count"),
            "measured_from": arguments.checkpoint or arguments.measured,
        },
    }
    with open(arguments.output, "w", encoding="utf-8") as fh:
        json.dump(document, fh, indent=1, sort_keys=False)
        fh.write("\n")
    expert_bytes = sum(entry["bytes"] for entry in experts)
    print(json.dumps({
        "output": os.path.relpath(arguments.output, ROOT),
        "experts": len(experts),
        "expert_bytes_mib_full_model": round(expert_bytes / (1024 * 1024), 1),
        "spine_bytes_gib": round(measured["spine_bytes"] / (1024 ** 3), 2),
        "kv_floor_bytes_mib": round(kv_floor_bytes / (1024 * 1024), 2),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
