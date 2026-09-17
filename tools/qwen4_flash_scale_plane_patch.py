#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
	sys.path.insert(0, _TOOLS_DIR)
from spark_pack_common import SafetensorsSource  # noqa: E402
from qwen4_flash_pack_stamp_check import (  # noqa: E402
	FMT_FP8_F32B128, read_dir)

PROJ = {6: "gate_proj", 7: "up_proj", 8: "down_proj"}
EXPERTS_TOTAL = 512
LAYER_PREFIX = "model.language_model.layers."
PROBE_BYTES = 65536


def widen_bf16(raw):
	return (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).view(np.float32).tobytes()


def fatal(problems, message):
	problems.append(message)
	print(json.dumps({"problems": problems}, indent=1))
	raise SystemExit(1)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--pack", required=True)
	parser.add_argument("--warm", default="/mnt/model-warm/qwen3.8-flash-next-fp8")
	parser.add_argument("--tp-degree", type=int, default=8)
	parser.add_argument("--tp-rank", type=int, required=True)
	parser.add_argument("--write", action="store_true")
	parser.add_argument("--json-out", default=None)
	arguments = parser.parse_args()
	expert_count = EXPERTS_TOTAL // arguments.tp_degree
	expert_start = arguments.tp_rank * expert_count
	pack_path = Path(arguments.pack)
	size = pack_path.stat().st_size
	header, entries = read_dir(str(pack_path))
	source = SafetensorsSource(Path(arguments.warm))
	fp8_entries = [e for e in entries if e["fmt"] == FMT_FP8_F32B128]
	if not fp8_entries:
		raise SystemExit("no fp8 entries in pack - nothing the repair targets")
	problems = []
	with pack_path.open("rb") as pack:
		for entry in fp8_entries:
			kind = entry["kind"]
			layer = entry["layer"]
			proj = PROJ[kind]
			expert_rows = entry["rows"] // expert_count
			row_bytes = expert_rows * entry["cols"]
			if entry["payload_bytes"] != row_bytes * expert_count:
				fatal(problems, f"kind={kind} layer={layer} payload_bytes {entry['payload_bytes']} != {row_bytes * expert_count}")
			probe_head = min(PROBE_BYTES, row_bytes)
			for probe in (0, expert_count // 2, expert_count - 1):
				global_expert = expert_start + probe
				name = f"{LAYER_PREFIX}{layer}.mlp.experts.{global_expert}.{proj}.weight"
				shard, meta, data_offset = source.resolve(name)
				if meta["dtype"] != "F8_E4M3":
					fatal(problems, f"{name}: dtype {meta['dtype']}, expected F8_E4M3")
				if meta["shape"] != [expert_rows, entry["cols"]]:
					fatal(problems, f"{name}: shape {meta['shape']}, expected [{expert_rows},{entry['cols']}]")
				with (source.root / shard).open("rb") as shard_file:
					shard_file.seek(data_offset)
					want_head = shard_file.read(probe_head)
					shard_file.seek(data_offset + row_bytes - probe_head)
					want_tail = shard_file.read(probe_head)
				payload_base = entry["payload_offset"] + probe * row_bytes
				pack.seek(payload_base)
				got_head = pack.read(probe_head)
				pack.seek(payload_base + row_bytes - probe_head)
				got_tail = pack.read(probe_head)
				if got_head != want_head or got_tail != want_tail:
					problems.append(f"kind={kind} layer={layer} expert {global_expert} payload probe mismatch")
	if problems:
		print(json.dumps({"problems": problems[:32], "payload_probe_failures": len(problems),
			"verdict": "payload identity FAILED - pack not patched"}, indent=1))
		if arguments.json_out:
			with open(arguments.json_out, "w") as out:
				json.dump({"payload_probe_failures": len(problems), "problems": problems[:32],
					"verdict": "payload identity FAILED - pack not patched"}, out, indent=1)
		raise SystemExit(1)
	patched_planes = 0
	clean_planes = 0
	patched_bytes = 0
	plane_problems = []
	with pack_path.open("r+b" if arguments.write else "rb") as pack:
		for entry in fp8_entries:
			kind = entry["kind"]
			layer = entry["layer"]
			proj = PROJ[kind]
			expert_rows = entry["rows"] // expert_count
			s_rows = expert_rows // 128
			s_cols = entry["cols"] // 128
			per_expert_scale = s_rows * s_cols * 4
			if entry["scale_bytes"] != per_expert_scale * expert_count:
				fatal(plane_problems, f"kind={kind} layer={layer} scale_bytes {entry['scale_bytes']} != {per_expert_scale * expert_count}")
			expected_plane = bytearray()
			for offset in range(expert_count):
				global_expert = expert_start + offset
				name = f"{LAYER_PREFIX}{layer}.mlp.experts.{global_expert}.{proj}.weight_scale_inv"
				shard, meta, data_offset = source.resolve(name)
				if meta["dtype"] != "BF16":
					fatal(plane_problems, f"{name}: dtype {meta['dtype']}, expected BF16")
				if meta["shape"] != [s_rows, s_cols]:
					fatal(plane_problems, f"{name}: shape {meta['shape']}, expected [{s_rows},{s_cols}]")
				with (source.root / shard).open("rb") as shard_file:
					shard_file.seek(data_offset)
					raw = shard_file.read(s_rows * s_cols * 2)
				if len(raw) != s_rows * s_cols * 2:
					fatal(plane_problems, f"{name}: short read {len(raw)}")
				expected_plane += widen_bf16(raw)
			pack.seek(entry["scale_offset"])
			current = pack.read(entry["scale_bytes"])
			if current == bytes(expected_plane):
				clean_planes += 1
				continue
			patched_planes += 1
			patched_bytes += entry["scale_bytes"]
			if arguments.write:
				pack.seek(entry["scale_offset"])
				pack.write(bytes(expected_plane))
	result = {
		"pack": str(pack_path),
		"warm": arguments.warm,
		"tp_degree": arguments.tp_degree,
		"tp_rank": arguments.tp_rank,
		"write": arguments.write,
		"fp8_entries": len(fp8_entries),
		"payload_probes": len(fp8_entries) * 3,
		"payload_probe_failures": 0,
		"planes_clean": clean_planes,
		"planes_patched": patched_planes,
		"scale_bytes_patched": patched_bytes,
		"problems": plane_problems[:32],
	}
	print(json.dumps(result, indent=1))
	if arguments.json_out:
		with open(arguments.json_out, "w") as out:
			json.dump(result, out, indent=1)
	if plane_problems:
		raise SystemExit(1)
	if not arguments.write and patched_planes:
		print(f"CHECK: {patched_planes} of {len(fp8_entries)} planes differ from warm (run --write)", file=sys.stderr)
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
