#!/usr/bin/env python3
"""Verify a qwen38 27B stage pack against the wire format AND the live checkpoint.

This is the pack gate for the qwen38 27B lane (the module ships no pack
validation harness of its own; per docs/AGENT_LANE_BRIEFS/pack_agent_rules.md
"run the pack verifier. Exit must be PASS before deploy").

The layout contract is the v3 120-byte header of tools/qwen38_27b_stagepack.py
(magic 'Q6SP', format 3, dense-FFN kind table, tp_degree/tp_rank on the wire).
The tables load from the 27B packer itself so a table drift and a verifier
drift cannot pass silently.

Three layers of checking, all from raw bytes:

  1. STRUCTURE - header magic/version/geometry vs the packer's constants,
     directory entry count vs the format's computed inventory (optionally
     minus the 11-entry MTP pseudo-layer for the qwen36sp stripped form),
     per-entry shape/format/scale rules, payload alignment and bounds, no
     duplicate or missing (kind, layer) pair. Both placed wire forms are
     accepted and each is enforced strictly: the packer form (256-aligned,
     bf16-natural kinds optionally fp8 f32b128 or nvfp4 with their scale
     planes) and the qwen36sp compact-strip form (64-aligned, bf16-natural
     kinds riding scale-less fp8, scale_group 0).
  2. CONTENT - every entry's payload+scale bytes are hashed out of the pack
     file and compared against the byte stream the packer's own copy path
     produces from the source checkpoint. Requires --checkpoint.
  3. RECEIPT - tensor/byte counts, slice identity, the qwen36sp strip fields
     and the packer's output_sha256 are cross-checked. --recompute-file-hash
     re-reads the whole file to recompute that digest independently.

Exit 0 only on PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from spark_pack_common import sha256_file  # noqa: E402

DEFAULT_TABLES = "qwen38_27b_stagepack.py"

HASH_CHUNK = 16 * 1024 * 1024

HEADER_BYTES = 120
ENTRY_BYTES = 56
PAYLOAD_ALIGNMENT = 256
HEADER_STRUCT = struct.Struct("<26I2Q")
ENTRY_STRUCT = struct.Struct("<6I4Q")

WEIGHT_BF16 = 0
WEIGHT_F32 = 1
WEIGHT_FP8_E4M3_F32B128 = 5
WEIGHT_NVFP4_PACKED = 8

GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE

BF16_BYTES = 2
F32_BYTES = 4
FP8_SCALE_GROUP = 128
NVFP4_GROUP = 16
NVFP4_GLOBAL_TAIL_BYTES = 4


def load_tables(path: str | None = None):
    spec = importlib.util.spec_from_file_location(
        "qwen38_27b_tables", str(Path(_TOOLS_DIR) / (path or DEFAULT_TABLES)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload_bytes_for(weight_format: int, rows: int, columns: int) -> int:
    elements = rows * columns
    if weight_format == WEIGHT_NVFP4_PACKED:
        return rows * (columns // 2)
    if weight_format == WEIGHT_FP8_E4M3_F32B128:
        return elements
    return elements * (BF16_BYTES if weight_format == WEIGHT_BF16 else F32_BYTES)


def scale_bytes_for(weight_format: int, rows: int, columns: int,
                    scale_group: int = 0) -> int:
    if weight_format == WEIGHT_FP8_E4M3_F32B128:
        if scale_group == 0:
            return 0
        return (rows // FP8_SCALE_GROUP) * (columns // FP8_SCALE_GROUP) * F32_BYTES
    if weight_format == WEIGHT_NVFP4_PACKED:
        return rows * (columns // NVFP4_GROUP) + NVFP4_GLOBAL_TAIL_BYTES
    return 0


def natural_format(tables, kind: int) -> int:
    if kind in (tables.KIND_GDN_A_LOG, tables.KIND_GDN_DT_BIAS):
        return WEIGHT_F32
    return WEIGHT_BF16


class _HashingSink:
    def __init__(self):
        self.digest = hashlib.sha256()

    def write(self, data) -> int:
        self.digest.update(data)
        return len(data)


def hash_source_entry(tables, source, ref, plan) -> str:
    """sha256 of the byte stream the packer's own copy path yields."""
    sink = _HashingSink()
    _, _, offset = source.check_shape(ref)
    tables.copy_tensor(source, ref, offset, plan, sink)
    if ref.weight_format in (WEIGHT_FP8_E4M3_F32B128, WEIGHT_NVFP4_PACKED):
        tables.copy_scale(source, ref, plan, sink)
    return sink.digest.hexdigest()


def verify(pack: Path, checkpoint: Path | None, receipt_path: Path | None,
           recompute_file_hash: bool, tp_degree: int, strip_mtp: bool,
           ffn_format: str, tp_rank: int | None = None,
           tables=None) -> tuple[bool, dict]:
    tables = tables if tables is not None else load_tables()
    findings: list[str] = []

    def fail(message: str) -> None:
        findings.append(message)

    file_bytes_actual = pack.stat().st_size
    tensor_count = 0
    first_layer = layer_count = -1
    entries: list[tuple] = []
    file_sha = None
    source = None

    with pack.open("rb") as f:
        raw_header = f.read(HEADER_BYTES)
        if len(raw_header) != HEADER_BYTES:
            return False, {"verdict": "FAIL", "pack": str(pack),
                           "errors": [f"header truncated: {len(raw_header)} bytes"]}
        header = HEADER_STRUCT.unpack(raw_header)
        (magic, version, header_bytes, entry_bytes, tensor_count, hidden,
         layer_count, first_layer, total_layers, period, full_phase,
         gdn_kh, gdn_vh, gdkd, gdvd, conv_k, qh, kvh, hd, rope_d,
         ffn_int, vocab, mxfp4_group, mtp_count, header_tp_degree,
         header_tp_rank, directory_offset, file_bytes) = header

        def want(field: str, got, expected) -> None:
            if got != expected:
                fail(f"header {field}={got}, expected {expected}")

        want("magic", magic, tables.MAGIC)
        want("format_version", version, tables.FORMAT_VERSION)
        want("header_bytes", header_bytes, HEADER_BYTES)
        want("directory_entry_bytes", entry_bytes, ENTRY_BYTES)
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
        want("ffn_intermediate_dimension", ffn_int, tables.FFN_INTERMEDIATE)
        want("output_vocab_count", vocab, tables.VOCAB)
        want("mxfp4_group_size", mxfp4_group, tables.MXFP4_GROUP)
        want("mtp_layer_count", mtp_count, tables.MTP_LAYERS)
        want("header tp_degree", header_tp_degree, tp_degree)
        want("directory_offset", directory_offset, HEADER_BYTES)
        want("file_bytes", file_bytes, file_bytes_actual)
        plan_rank = header_tp_rank if tp_rank is None else tp_rank
        if tp_degree > 1 and not 0 <= plan_rank < tp_degree:
            fail(f"tp rank {plan_rank} outside 0..{tp_degree - 1}")
        if layer_count <= 0 or first_layer < 0 or first_layer + layer_count > tables.LAYER_COUNT:
            fail(f"invalid slice {first_layer}+{layer_count} of {tables.LAYER_COUNT}")

        expected_refs: dict[tuple[int, int], object] = {}
        try:
            for ref in tables.build_inventory(first_layer, layer_count):
                if strip_mtp and ref.layer == MTP_LAYER:
                    continue
                if tp_degree > 1:
                    ref.plan = tables.build_tp_plan(ref, tp_degree, plan_rank)
                    ref.packed = tables.packed_shape(ref, ref.plan)
                else:
                    ref.plan = None
                    ref.packed = (ref.rows, ref.columns)
                expected_refs[(ref.kind, ref.layer)] = ref
        except tables.PackFailure as error:
            fail(f"inventory build failed: {error}")
        if expected_refs and tensor_count != len(expected_refs):
            fail(f"tensor_count={tensor_count}, format inventory expects "
                 f"{len(expected_refs)}"
                 + (" (MTP pseudo-layer stripped)" if strip_mtp else ""))

        raw_dir = f.read(tensor_count * ENTRY_BYTES)
        if len(raw_dir) != tensor_count * ENTRY_BYTES:
            fail("directory truncated")

        decoded = [ENTRY_STRUCT.unpack_from(raw_dir, index * ENTRY_BYTES)
                   for index in range(tensor_count)]
        offsets_256 = all(entry[6] % PAYLOAD_ALIGNMENT == 0 for entry in decoded)
        offsets_64 = all(entry[6] % 64 == 0 for entry in decoded)
        if offsets_256:
            alignment = PAYLOAD_ALIGNMENT
        elif offsets_64:
            alignment = 64
        else:
            alignment = 0
            fail("directory payload offsets are neither 256-aligned (the "
                 "packer form) nor uniformly 64-aligned (the qwen36sp "
                 "compact-strip form)")

        seen: set[tuple[int, int]] = set()
        for index, entry in enumerate(decoded):
            (kind, layer, fmt, rows, cols, scale_group, p_off, p_bytes,
             s_off, s_bytes) = entry
            tag = f"entry[{index}] kind={kind} layer={hex(layer)}"
            if not 0 <= kind < 32:
                fail(f"{tag}: kind out of range")
                continue
            key = (kind, layer)
            if key in seen:
                fail(f"{tag}: duplicate (kind, layer)")
            seen.add(key)
            ref = expected_refs.get(key)
            if ref is None:
                fail(f"{tag}: not in the format inventory of slice "
                     f"{first_layer}+{layer_count}"
                     + (" (MTP pseudo-layer stripped)" if strip_mtp else ""))
                continue
            packed_rows, packed_cols = ref.packed
            if (rows, cols) != (packed_rows, packed_cols):
                fail(f"{tag}: shape {rows}x{cols}, expected {packed_rows}x{packed_cols}")
            natural = natural_format(tables, kind)
            if fmt == natural:
                want_group = 0
            elif fmt == WEIGHT_FP8_E4M3_F32B128 and natural == WEIGHT_BF16 \
                    and scale_group in (0, FP8_SCALE_GROUP):
                want_group = scale_group
            elif fmt == WEIGHT_NVFP4_PACKED and natural == WEIGHT_BF16:
                want_group = NVFP4_GROUP
            else:
                want_group = -1
                fail(f"{tag}: weight_format={fmt} scale_group={scale_group} "
                     f"outside the packer ladder (natural {natural}, bf16 "
                     f"kinds may ride fp8 f32b128, fp8 compact-strip, or "
                     f"nvfp4)")
            if scale_group != want_group:
                fail(f"{tag}: scale_group_size={scale_group}, expected {want_group}")
            want_payload = payload_bytes_for(fmt, packed_rows, packed_cols)
            want_scale = scale_bytes_for(fmt, packed_rows, packed_cols, scale_group)
            if p_bytes != want_payload:
                fail(f"{tag}: payload_bytes={p_bytes}, format math says {want_payload}")
            if s_bytes != want_scale:
                fail(f"{tag}: scale_bytes={s_bytes}, format math says {want_scale}")
            if alignment and p_off % alignment != 0:
                fail(f"{tag}: payload_offset {p_off} not {alignment}-aligned")
            if s_bytes and s_off % alignment != 0:
                fail(f"{tag}: scale_offset {s_off} not {alignment}-aligned")
            if p_off + p_bytes > file_bytes_actual:
                fail(f"{tag}: payload region overruns file")
            if s_bytes and s_off + s_bytes > file_bytes_actual:
                fail(f"{tag}: scale region overruns file")
            entries.append((ref, p_off, p_bytes, s_off, s_bytes))

        for key in sorted(set(expected_refs) - seen):
            fail(f"missing tensor kind={key[0]} layer={hex(key[1])}")

    verdict = {
        "verdict": "FAIL",
        "pack": str(pack),
        "file_bytes": file_bytes_actual,
        "first_layer": first_layer,
        "layer_count": layer_count,
        "tensor_count": tensor_count,
        "mtp_pseudo_layer_stripped": bool(strip_mtp),
        "errors": findings,
    }
    if findings:
        return False, verdict

    if checkpoint is not None:
        source_class = getattr(tables, "Nvfp4A16Source", None) \
            if ffn_format == "nvfp4a16" else None
        source = (source_class or tables.SafetensorsSource)(checkpoint)
        source.check_config()
        content_failures = 0
        compared = 0
        with pack.open("rb") as f:
            for ref, p_off, p_bytes, s_off, s_bytes in entries:
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

                if not stream_region(p_off, p_bytes):
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)}: pack payload short read")
                    continue
                if s_bytes and not stream_region(s_off, s_bytes):
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)}: pack scale short read")
                    continue
                try:
                    source_digest = hash_source_entry(tables, source, ref, ref.plan)
                except (RuntimeError, tables.PackFailure) as error:
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                         f"source re-emit failed: {error}")
                    continue
                compared += 1
                if pack_digest.hexdigest() != source_digest:
                    content_failures += 1
                    fail(f"kind={ref.kind} layer={hex(ref.layer)} name={ref.name}: "
                         f"pack bytes != source bytes")
                if compared % 20 == 0:
                    print(f"  content {compared}/{len(entries)} tensors", flush=True)
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
            "file_bytes": (receipt.get("file_bytes"), file_bytes_actual),
            "bytes": (receipt.get("bytes"), file_bytes_actual),
            "first_layer_index": (receipt.get("first_layer_index"), first_layer),
            "layer_count": (receipt.get("layer_count"), layer_count),
            "tp_degree": (receipt.get("tp_degree"), tp_degree),
        }
        if file_sha is not None:
            checks["output_sha256"] = (receipt.get("output_sha256"), file_sha)
        if strip_mtp and receipt.get("mtp") not in (None, "stripped"):
            fail(f"receipt mtp={receipt.get('mtp')!r}, expected 'stripped'")
        if source is not None and receipt.get("source_index_sha256") not in (None, source.index_sha256):
            fail("receipt source_index_sha256 does not match the live checkpoint index")
        for name, (recorded, recomputed) in checks.items():
            if recorded is not None and recorded != recomputed:
                fail(f"receipt {name}={recorded!r}, verifier recomputed {recomputed!r}")

    verdict["errors"] = findings
    verdict["verdict"] = "PASS" if not findings else "FAIL"
    return not findings, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tp-degree", type=int, default=1,
        help="the pack's TP degree; entry shapes are compared per-rank")
    parser.add_argument("--tp-rank", type=int, default=None,
        help="this pack's TP rank for the sharded plan (default: the "
             "header's tp_rank field)")
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        help="checkpoint dir; omit for structure-only pass")
    parser.add_argument("--strip-mtp", action="store_true",
        help="the pack omits the 11-entry MTP pseudo-layer (the qwen36sp "
             "stripped form: MTP globals present, mtp_entries_dropped 11)")
    parser.add_argument("--ffn-format", choices=("bf16", "nvfp4a16"),
                        default="bf16",
                        help="checkpoint source class for the content pass")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--recompute-file-hash", action="store_true",
        help="re-read the pack to recompute the whole-file sha256")
    parser.add_argument("--json-out", type=Path, help="verdict JSON path")
    args = parser.parse_args()

    ok, verdict = verify(args.pack, args.checkpoint, args.receipt,
                         args.recompute_file_hash, args.tp_degree,
                         args.strip_mtp, args.ffn_format, args.tp_rank)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
    print(f"qwen38_pack_verify pack={args.pack} verdict={verdict['verdict']} "
          f"file_gib={verdict['file_bytes'] / 2**30:.2f} errors={len(verdict['errors'])}")
    for finding in verdict["errors"][:40]:
        print(f"  {finding}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
