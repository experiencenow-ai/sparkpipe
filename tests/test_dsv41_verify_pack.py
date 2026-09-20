"""tools/dsv41_verify_pack.py: no-warm mechanical verification for the two
DeepSeek-V4 NO-TOOL sweep families.

  dsv41flash: the 257-byte header + 64-byte records of
  tools/dsv41_flash_stagepack.py, verified against the kind map
  (entry_shape/entry_layout) and the exact cmd_plan cursor layout.

  dsv4flash: the 80-byte header + 40-byte records of
  tools/dsv4_tp16_stagepack.py, rebuilt from the contract records through
  plan_entry's sharding rules.

Receipt emission is exercised on a tiny summary (the placed packs are tens
of GiB; the digest pass dominates the real runs and is already covered
there).
"""

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

TOOL = ROOT / "tools" / "dsv41_verify_pack.py"
CONTRACT = ROOT / "model_contracts" / "dsv4_flash.json"


def load(name: str, filename: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, str(ROOT / "tools" / filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_dsv41_pack(path: Path, tp: int, rank: int) -> None:
    d = load("dsv41_stagepack_tables", "dsv41_flash_stagepack.py")
    order = [(kind, d.GLOBAL_LAYER) for kind in range(d.KIND_COUNT)
             if d.is_global(kind)]
    for layer in range(d.LAYER_COUNT):
        for kind in range(d.K_ATTN_NORM, d.KIND_COUNT):
            if d.kind_in_layer(kind, layer):
                order.append((kind, layer))
    entries = []
    cursor = 0
    for kind, layer in order:
        rows, cols, groups = d.entry_shape(kind, tp)
        payload_type, codec, scale_enc, pbytes, sbytes = d.entry_layout(kind, tp)
        cursor = (cursor + d.ALIGN - 1) & ~(d.ALIGN - 1)
        entry = [kind, layer, payload_type, codec, scale_enc, groups, rows,
                 cols, cursor, pbytes, 0, sbytes]
        cursor += pbytes
        if sbytes:
            cursor = (cursor + d.ALIGN - 1) & ~(d.ALIGN - 1)
            entry[10] = cursor
            cursor += sbytes
        entries.append(entry)
    directory_offset = (d.HEADER_BYTES + d.ALIGN - 1) & ~(d.ALIGN - 1)
    payload_base = (directory_offset + len(entries) * d.ENTRY_BYTES
                    + d.ALIGN - 1) & ~(d.ALIGN - 1)
    file_bytes = cursor + payload_base
    header = bytearray(d.HEADER_BYTES)
    struct.pack_into("<20I2Q", header, 0, d.MAGIC, 1, d.HEADER_BYTES,
                     d.ENTRY_BYTES, 1, 0, len(entries), 1, 0, 0, d.LAYER_COUNT,
                     d.LAYER_COUNT, d.HIDDEN, d.VOCAB, d.ROUTED_EXPERTS,
                     d.CODEC_FP8, d.CODEC_MXFP4, tp, rank, 0,
                     directory_offset, file_bytes)
    header[96:96 + 40] = b"dba1be0a40aa45a94ad051997016db3960a90277"
    header[161:193] = bytes(range(32))
    header[193:225] = bytes(range(32, 64))
    header[225:257] = bytes(range(64, 96))
    with path.open("wb") as handle:
        handle.truncate(file_bytes)
        handle.write(bytes(header))
        handle.seek(directory_offset)
        for index, entry in enumerate(entries):
            entry[8] += payload_base
            if entry[10]:
                entry[10] += payload_base
            handle.write(struct.pack("<8I4Q", *entry))


def build_dsv4flash_pack(path: Path, rank: int) -> None:
    stagepack = load("dsv4_stagepack_tables", "dsv4_stagepack.py")
    tp16 = load("dsv4_tp16_tables", "dsv4_tp16_stagepack.py")
    contract = json.loads(CONTRACT.read_text())
    records = stagepack.build_records(contract, 0, tp16.LAYERS)
    expected = []
    for record in records:
        entry = (record.kind, record.layer, record.weight_format,
                 record.rows, record.columns, 0, 0, 0)
        try:
            planned = tp16.plan_entry(entry, rank, 1, 0)
        except stagepack.PackFailure as error:
            if str(error) == "filtered":
                continue
            raise
        _k, _l, weight, rows, cols = planned[0][:5]
        pbytes = tp16.payload_bytes(weight, rows, cols)
        sbytes = tp16.scale_bytes(weight, rows, cols)
        expected.append([_k, _l, weight, rows, cols, pbytes, sbytes])
    header_size = tp16.HEADER.size
    entry_size = tp16.ENTRY.size
    cursor = header_size + entry_size * len(expected)
    directory = b""
    for kind, layer, weight, rows, cols, pbytes, sbytes in expected:
        poff = cursor
        cursor += pbytes
        soff = 0
        if sbytes:
            soff = cursor
            cursor += sbytes
        directory += tp16.ENTRY.pack(kind, layer, weight, rows, cols, 0,
                                     poff, soff)
    header = struct.pack(
        "<16I2Q", tp16.MAGIC, tp16.VERSION, header_size, entry_size, 1,
        5, 7, 1, len(expected), 0, tp16.LAYERS, tp16.LAYERS, tp16.HIDDEN,
        tp16.VOCAB, tp16.EXPERTS, stagepack.MTP_LAYER_COUNT_MAX,
        header_size, cursor)
    with path.open("wb") as handle:
        handle.truncate(cursor)
        handle.write(header)
        handle.seek(header_size)
        handle.write(directory)


def run_tool(pack: Path, *extra: str):
    done = subprocess.run(
        [sys.executable, str(TOOL), "--pack", str(pack), *extra],
        capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def test_dsv41flash_pack_passes():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank3.spstage"
        build_dsv41_pack(pack, 8, 3)
        code, output = run_tool(pack)
        assert code == 0, output
        assert "1038 directory entries" in output


def test_dsv41flash_corrupted_entry_fails():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank3.spstage"
        build_dsv41_pack(pack, 8, 3)
        raw = bytearray(pack.read_bytes())
        struct.pack_into("<I", raw, 512 + 5 * 64 + 24, 12345)
        pack.write_bytes(bytes(raw))
        code, output = run_tool(pack)
        assert code == 1, output


def test_dsv41flash_wrong_tp_fails():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank3.spstage"
        build_dsv41_pack(pack, 8, 3)
        raw = bytearray(pack.read_bytes())
        struct.pack_into("<I", raw, 68, 4)
        pack.write_bytes(bytes(raw))
        code, output = run_tool(pack)
        assert code == 1, output


def test_dsv4flash_pack_passes():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "dsv4flash.tp16.rank3.spstage"
        build_dsv4flash_pack(pack, 3)
        code, output = run_tool(pack, "--family", "dsv4flash")
        assert code == 0, output


def test_dsv4flash_rank_mismatch_fails():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "dsv4flash.tp16.rank3.spstage"
        build_dsv4flash_pack(pack, 0)
        code, output = run_tool(pack, "--family", "dsv4flash", "--rank", "3")
        assert code == 1, output


def test_receipt_emission_pair():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "rank0.spstage"
        pack.write_bytes(b"tiny")
        summary = {"family": "dsv41flash", "tp_degree": 8, "tp_rank": 0,
                   "tensor_count": 1038, "file_bytes": 4,
                   "sha256": hashlib.sha256(b"tiny").hexdigest()}
        sys.path.insert(0, str(TOOL.parent))
        import dsv41_verify_pack as V
        V.emit(summary, pack)
        receipt = json.loads(Path(str(pack) + ".receipt.json").read_text())
        assert receipt["output_sha256"] == receipt["sha256"]
        assert receipt["kind"] == "sparkpipe.dsv4.pack-verify-receipt.v1"
        sidecar = Path(str(pack) + ".sha256").read_text()
        assert sidecar.endswith("  rank0.spstage\n")
        V.emit(summary, pack)
        assert len(list(Path(tmp).iterdir())) == 3


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
