#!/usr/bin/env python3
"""Verify a qwen38_max TP rank pack against the v2 wire format.

Covers both placed generations of the max family:

  * current-packer packs: MTP-stripped (mtp_layer_count 0), routed experts
    stamped NVFP4_PACKED (8) with per-expert [input_scale][weight_scale_2]
    f32 tails, directory rows/columns authoritative;
  * placed tp4pp4 stage packs (tools/qwen38_max_tp4pp4_stagepacks.py,
    mtp_layer_count 1): routed experts carry the pre-split nvfp4 code (4)
    with the tail-less per-16 scale plane, and the directory rows/columns
    ride the packer's late-binding defect (every entry repeats the LAST
    inventory tensor's shape; commit 4ca697a). On those packs the entry
    byte math is proven from the tp plan instead of the on-wire shape
    fields, and the stale pair must equal the last inventory ref's packed
    shape exactly.

Header identity (v2 128-byte header carrying tp_degree/tp_rank), inventory
closure, payload/scale byte math, 256-alignment, bounds, extent and the
receipt are checked on both. The content pass (--checkpoint) re-emits each
rank's bytes through the packer's own copy path; it requires the
current-packer codec-8 form.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

_spec = importlib.util.spec_from_file_location(
    "qwen38_max_stagepack_tables", str(Path(_TOOLS_DIR) / "qwen38_stagepack.py"))
_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tables)
_tables.EXPERT_CODEC = "nvfp4"

HASH_CHUNK = 16 * 1024 * 1024
MAGIC = _tables.MAGIC
HEADER_STRUCT = struct.Struct("<28I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")
PAYLOAD_ALIGNMENT = _tables.PAYLOAD_ALIGNMENT
BF16_BYTES = 2
F32_BYTES = 4
NVFP4_TAIL_BYTES = 8


def expert_kinds(tables):
    return (tables.KIND_MOE_W1, tables.KIND_MOE_W3, tables.KIND_MOE_DOWN)


class _HashingSink:
    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        self.digest.update(data)
        return len(data)


def entry_bytes_for(tables, weight_format: int, kind: int, rows: int, cols: int,
                    resident_experts: int) -> tuple[int, int]:
    if weight_format == tables.WEIGHT_NVFP4_PACKED:
        return (rows * (cols // 2),
                rows * (cols // 16) + resident_experts * NVFP4_TAIL_BYTES)
    if weight_format == tables.WEIGHT_FP8_F32B128 and kind in expert_kinds(tables):
        return rows * (cols // 2), rows * (cols // 16)
    if weight_format == tables.WEIGHT_FP8_F32B128:
        return rows * cols, (rows // 128) * (cols // 128) * F32_BYTES
    element = BF16_BYTES if weight_format == tables.WEIGHT_BF16 else F32_BYTES
    return rows * cols * element, 0


def want_group(tables, weight_format: int, kind: int) -> int:
    if weight_format == tables.WEIGHT_NVFP4_PACKED:
        return 16
    if weight_format == tables.WEIGHT_FP8_F32B128:
        return 128
    return 0


def verify(pack: Path, tp_degree: int, tp_rank: int, checkpoint: Path | None,
           receipt_path: Path | None, recompute_file_hash: bool,
           tables=None) -> tuple[bool, dict]:
    tables = tables if tables is not None else _tables
    findings: list[str] = []

    def fail(message: str) -> None:
        findings.append(message)

    file_bytes_actual = pack.stat().st_size
    first_layer = layer_count = -1
    tensor_count = 0
    entries: list[tuple] = []
    file_sha = None
    source = None

    with pack.open("rb") as f:
        raw_header = f.read(128)
        if len(raw_header) != 128:
            return False, {"verdict": "FAIL", "pack": str(pack),
                           "errors": [f"header truncated: {len(raw_header)} bytes"]}
        header = HEADER_STRUCT.unpack(raw_header)
        (magic, version, header_bytes, entry_bytes, tensor_count, hidden,
         layer_count, first_layer, total_layers, period, full_phase,
         gdn_kh, gdn_vh, gdkd, gdvd, conv_k, qh, kvh, hd, rope_d,
         expert_count, experts_per_token, moe_int, vocab, mxfp4_group,
         mtp_count, header_tp_degree, header_tp_rank,
         directory_offset, file_bytes) = header

        def want(field: str, got, expected) -> None:
            if got != expected:
                fail(f"header {field}={got}, expected {expected}")

        want("magic", magic, MAGIC)
        want("format_version", version, tables.FORMAT2_VERSION)
        want("header_bytes", header_bytes, tables.HEADER2_BYTES)
        want("directory_entry_bytes", entry_bytes, tables.ENTRY_BYTES)
        want("hidden_dimension", hidden, tables.HIDDEN)
        want("total_layer_count", total_layers, tables.LAYER_COUNT)
        want("attention_period", period, tables.ATTENTION_PERIOD)
        want("full_attention_phase", full_phase, tables.FULL_PHASE)
        want("gdn_key_head_count", gdn_kh, tables.GDN_KEY_HEADS)
        want("gdn_value_head_count", gdn_vh, tables.GDN_VALUE_HEADS)
        want("gdn_head_key_dimension", gdkd, tables.GDN_HEAD_KEY_DIM)
        want("gdn_head_value_dimension", gdvd, tables.GDN_HEAD_VALUE_DIM)
        want("gdn_conv_kernel", conv_k, tables.GDN_CONV_KERNEL)
        want("attn_query_head_count", qh, tables.ATTN_QUERY_HEADS)
        want("attn_kv_head_count", kvh, tables.ATTN_KV_HEADS)
        want("attn_head_dimension", hd, tables.ATTN_HEAD_DIM)
        want("attn_rope_dimension", rope_d, tables.ATTN_ROPE_DIM)
        want("routed_expert_count", expert_count, tables.EXPERT_COUNT)
        want("experts_per_token", experts_per_token, tables.EXPERTS_PER_TOKEN)
        want("expert_intermediate_dimension", moe_int, tables.EXPERT_INTERMEDIATE)
        want("output_vocab_count", vocab, tables.VOCAB)
        want("mxfp4_group_size", mxfp4_group, tables.MXFP4_GROUP)
        if mtp_count not in (0, tables.MTP_LAYERS):
            fail(f"header mtp_layer_count={mtp_count}, expected 0 "
                 f"(MTP-stripped) or {tables.MTP_LAYERS} (placed tp4pp4 form)")
        want("header tp_degree", header_tp_degree, tp_degree)
        want("header tp_rank", header_tp_rank, tp_rank)
        want("directory_offset", directory_offset, tables.HEADER2_BYTES)
        want("file_bytes", file_bytes, file_bytes_actual)
        if layer_count <= 0 or first_layer < 0 or first_layer + layer_count > tables.LAYER_COUNT:
            fail(f"invalid slice {first_layer}+{layer_count} of {tables.LAYER_COUNT}")
        if tp_degree <= 1 or not 0 <= tp_rank < tp_degree:
            fail(f"invalid tp placement degree={tp_degree} rank={tp_rank}")

        tables.STRIP_MTP = mtp_count == 0
        expected_refs: dict[tuple[int, int], object] = {}
        try:
            for ref in tables.build_inventory(first_layer, layer_count):
                expected_refs[(ref.kind, ref.layer)] = ref
        except tables.PackFailure as error:
            fail(f"inventory build failed: {error}")
        want("tensor_count", tensor_count, len(expected_refs))
        if not expected_refs or tensor_count != len(expected_refs):
            return False, {"verdict": "FAIL", "pack": str(pack),
                           "errors": findings + [
                               "tensor_count does not match the format "
                               "inventory; directory not read"]}

        raw_dir = f.read(tensor_count * tables.ENTRY_BYTES)
        if len(raw_dir) != tensor_count * tables.ENTRY_BYTES:
            return False, {"verdict": "FAIL", "pack": str(pack),
                           "errors": findings + ["directory truncated"]}

        planned: dict[tuple[int, int], tuple] = {}
        for key, ref in expected_refs.items():
            plan = tables.build_tp_plan(ref, tp_degree, tp_rank)
            srows, scols = tables.packed_tp_shape(ref, plan)
            resident = 0
            if ref.kind in expert_kinds(tables):
                resident = srows // (ref.rows // tables.EXPERT_COUNT)
            ladder_fmt = tables.ref_weight_format(ref)
            variants = {ladder_fmt: "packer"}
            if ladder_fmt == tables.WEIGHT_NVFP4_PACKED:
                variants[tables.WEIGHT_FP8_F32B128] = "placed-legacy"
            accepted = {}
            for fmt_variant in variants:
                payload, scale = entry_bytes_for(
                    tables, fmt_variant, ref.kind, srows, scols, resident)
                accepted[fmt_variant] = (payload, scale,
                                         want_group(tables, fmt_variant, ref.kind))
            planned[key] = (plan, ladder_fmt, sorted(accepted), accepted,
                            srows, scols)

        decoded = []
        wire_pairs: set[tuple[int, int]] = set()
        plan_matches: set[bool] = set()
        for index in range(tensor_count):
            entry = ENTRY_STRUCT.unpack_from(raw_dir, index * tables.ENTRY_BYTES)
            (kind, layer, fmt, rows, cols, scale_group, p_off, p_bytes,
             s_off, s_bytes) = entry
            tag = f"entry[{index}] kind={kind} layer={hex(layer)}"
            if not 0 <= kind < 32:
                fail(f"{tag}: kind out of range")
                continue
            key = (kind, layer)
            if key not in planned:
                fail(f"{tag}: not in the inventory of slice "
                     f"{first_layer}+{layer_count}")
                continue
            plan, ladder_fmt, fmts, accepted, srows, scols = planned[key]
            if fmt not in accepted:
                fail(f"{tag}: weight_format={fmt}, accepted formats for this "
                     f"kind are {fmts} (packer ladder {ladder_fmt})")
                continue
            want_payload, want_scale, w_group = accepted[fmt]
            if scale_group != w_group:
                fail(f"{tag}: scale_group_size={scale_group}, expected {w_group}")
            matches_plan = (rows, cols) == (srows, scols)
            if p_bytes != want_payload or s_bytes != want_scale:
                detail = (f"payload_bytes={p_bytes} (expected {want_payload}), "
                          f"scale_bytes={s_bytes} (expected {want_scale})")
                if matches_plan:
                    fail(f"{tag}: {detail}")
                else:
                    fail(f"{tag}: shape {rows}x{cols}, expected rank shard "
                         f"{srows}x{scols}; {detail}")
            if p_off % PAYLOAD_ALIGNMENT != 0:
                fail(f"{tag}: payload_offset {p_off} not {PAYLOAD_ALIGNMENT}-aligned")
            if s_bytes and s_off % PAYLOAD_ALIGNMENT != 0:
                fail(f"{tag}: scale_offset {s_off} not {PAYLOAD_ALIGNMENT}-aligned")
            if p_off + p_bytes > file_bytes_actual:
                fail(f"{tag}: payload region overruns file")
            if s_bytes and s_off + s_bytes > file_bytes_actual:
                fail(f"{tag}: scale region overruns file")
            decoded.append((key, ref, plan, fmt, p_off, p_bytes, s_off, s_bytes))
            wire_pairs.add((rows, cols))
            plan_matches.add(matches_plan)

        for key in sorted(set(planned) - {d[0] for d in decoded}):
            fail(f"missing tensor kind={key[0]} layer={hex(key[1])}")

        if decoded and len(wire_pairs) == 1 and plan_matches != {True}:
            last_key = max(planned)
            last_shape = (planned[last_key][3], planned[last_key][4])
            wire_pair = next(iter(wire_pairs))
            if wire_pair != last_shape:
                fail(f"directory carries a uniform non-plan shape "
                     f"{wire_pair[0]}x{wire_pair[1]}; the late-binding stale "
                     f"signature must equal the last inventory ref's packed "
                     f"shape {last_shape[0]}x{last_shape[1]}")

    stale_any = bool(decoded) and len(wire_pairs) == 1 and plan_matches != {True}
    expert_fmt4 = any(d[3] == tables.WEIGHT_FP8_F32B128 and
                      d[1].kind in expert_kinds(tables) for d in decoded)

    verdict = {
        "verdict": "FAIL",
        "pack": str(pack),
        "file_bytes": file_bytes_actual,
        "tp_degree": tp_degree,
        "tp_rank": tp_rank,
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": tensor_count,
        "mtp_layer_count": mtp_count,
        "placed_stale_shape_directory": bool(stale_any),
        "errors": findings,
    }
    if findings:
        return False, verdict

    if checkpoint is not None:
        if stale_any or expert_fmt4:
            fail("content pass requires the current-packer form "
                 "(plan-shaped directory, codec-8 experts); this pack is "
                 "the placed legacy form and is structure+receipt verified "
                 "only")
        else:
            source = tables.SafetensorsSource(checkpoint)
            source.check_config()
            content_failures = 0
            compared = 0
            with pack.open("rb") as f:
                for key, ref, plan, fmt, p_off, p_bytes, s_off, s_bytes, _ in decoded:
                    pack_digest = hashlib.sha256()

                    def stream_region(offset: int, length: int) -> bool:
                        f.seek(offset)
                        remaining = length
                        while remaining > 0:
                            step = min(remaining, HASH_CHUNK)
                            chunk = f.read(step)
                            if len(chunk) != step:
                                return False
                            pack_digest.update(chunk)
                            remaining -= step
                        return True

                    ok = stream_region(p_off, p_bytes)
                    if s_bytes:
                        ok = stream_region(s_off, s_bytes) and ok
                    if not ok:
                        content_failures += 1
                        fail(f"kind={ref.kind} layer={hex(ref.layer)}: pack region short read")
                        continue
                    try:
                        _, _, source_offset = source.check_shape(ref)
                        sink = _HashingSink()
                        if plan is not None:
                            tables.copy_tp_plan(source, ref, plan, sink)
                        elif ref.weight_format == tables.WEIGHT_FP8_F32B128:
                            tables.copy_nvfp4_experts(source, ref, sink)
                        else:
                            tables.copy_bf16_tensor(source, ref, source_offset, sink)
                    except (RuntimeError, tables.PackFailure) as error:
                        content_failures += 1
                        fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                             f"source re-emit failed: {error}")
                        continue
                    compared += 1
                    if pack_digest.hexdigest() != sink.digest.hexdigest():
                        content_failures += 1
                        fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                             f"pack bytes != source bytes for this rank")
                    if compared % 50 == 0:
                        print(f"  content {compared}/{len(decoded)} tensors", flush=True)
            verdict["tensors_compared"] = compared
            verdict["content_errors"] = content_failures
            if content_failures:
                fail(f"{content_failures} content mismatches")

    if recompute_file_hash:
        file_sha = sha256_file(pack)
        verdict["file_sha256_recomputed"] = file_sha

    if receipt_path is not None and Path(receipt_path).is_file():
        receipt = json.loads(Path(receipt_path).read_text())
        checks = {
            "tensor_count": (receipt.get("tensor_count"), tensor_count),
            "bytes": (receipt.get("bytes"), file_bytes_actual),
            "file_bytes": (receipt.get("file_bytes"), file_bytes_actual),
            "first_layer_index": (receipt.get("first_layer_index"), first_layer),
            "layer_count": (receipt.get("layer_count"), layer_count),
            "tp_degree": (receipt.get("tp_degree"), tp_degree),
            "tp_rank": (receipt.get("tp_rank"), tp_rank),
        }
        if file_sha is not None:
            checks["output_sha256"] = (receipt.get("output_sha256"), file_sha)
        for name, (recorded, recomputed) in checks.items():
            if recorded is not None and recorded != recomputed:
                fail(f"receipt {name}={recorded!r}, verifier recomputed {recomputed!r}")
        if source is not None and receipt.get("source_index_sha256") not in (None, source.index_sha256):
            fail("receipt source_index_sha256 does not match the live checkpoint index")

    verdict["errors"] = findings
    verdict["verdict"] = "PASS" if not findings else "FAIL"
    return not findings, verdict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="verify one qwen38_max TP rank "
                                     "pack against the v2 wire format (both the "
                                     "MTP-stripped codec-8 form and the placed "
                                     "tp4pp4 legacy form) and, optionally, the "
                                     "live checkpoint")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--tp-degree", type=int, default=16)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--structure-only", action="store_true",
        help="skip the per-entry content pass against the checkpoint")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--recompute-file-hash", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.structure_only:
        args.checkpoint = None

    ok, verdict = verify(args.pack, args.tp_degree, args.tp_rank, args.checkpoint,
                         args.receipt, args.recompute_file_hash)
    if args.json_out:
        args.json_out.write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps({k: v for k, v in verdict.items() if k != "errors"},
                     indent=2))
    if verdict["errors"]:
        for message in verdict["errors"][:20]:
            print(f"  {message}", file=sys.stderr)
    print(f"verdict={verdict['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
