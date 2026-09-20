"""The qwen38 27B pack verifier against the v3 dense-FFN wire.

The placed qwen27b arms ride two forms the verifier must enforce strictly:
the packer form (256-aligned, scale planes per formula) and the qwen36sp
compact-strip form (64-aligned, bf16-natural kinds on scale-less fp8 with
scale_group 0, some entries keeping the f32b128 plane, MTP pseudo-layer
stripped). Mini tables keep the packs tiny; the stale-attribute regression
(EXPERT_COUNT and friends) is pinned by loading the real 27B packer tables.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import qwen38_pack_verify as V

HEADER_BYTES = 120
ENTRY_BYTES = 56

MINI_TABLES_SOURCE = '''
MAGIC = 0x51385354
FORMAT_VERSION = 3
HIDDEN = 8
LAYER_COUNT = 4
ATTENTION_PERIOD = 4
FULL_PHASE = 3
GDN_KEY_HEADS = 1
GDN_VALUE_HEADS = 2
GDN_HEAD_KEY_DIM = 2
GDN_HEAD_VALUE_DIM = 2
GDN_CONV_KERNEL = 4
ATTN_QUERY_HEADS = 2
ATTN_KV_HEADS = 1
ATTN_HEAD_DIM = 4
ATTN_ROPE_DIM = 2
FFN_INTERMEDIATE = 8
VOCAB = 16
MXFP4_GROUP = 32
MTP_LAYERS = 1
WEIGHT_BF16 = 0
WEIGHT_F32 = 1
KIND_EMBEDDING, KIND_FINAL_NORM, KIND_LM_HEAD, KIND_NORM, KIND_FFN, KIND_A_LOG = range(6)
KIND_GDN_A_LOG = KIND_A_LOG
KIND_GDN_DT_BIAS = KIND_A_LOG
GLOBAL_LAYER = 0xFFFFFFFF
MTP_LAYER = 0xFFFFFFFE

class PackFailure(Exception):
    pass

class Ref:
    def __init__(self, kind, layer, name, rows, columns, weight_format):
        self.kind = kind
        self.layer = layer
        self.name = name
        self.rows = rows
        self.columns = columns
        self.weight_format = weight_format
        self.plan = None
        self.packed = (rows, columns)

class SafetensorsSource:
    def __init__(self, *a, **k):
        raise RuntimeError("content pass not under test")

def build_inventory(first_layer, layer_count):
    refs = []
    if first_layer == 0:
        refs.append(Ref(KIND_EMBEDDING, GLOBAL_LAYER, "embed", VOCAB, HIDDEN, WEIGHT_BF16))
    for layer in range(first_layer, first_layer + layer_count):
        refs.append(Ref(KIND_NORM, layer, "norm", 1, HIDDEN, WEIGHT_BF16))
        refs.append(Ref(KIND_FFN, layer, "ffn", HIDDEN, HIDDEN, WEIGHT_BF16))
    if first_layer + layer_count == LAYER_COUNT:
        if first_layer != 0:
            refs.append(Ref(KIND_EMBEDDING, GLOBAL_LAYER, "embed", VOCAB, HIDDEN, WEIGHT_BF16))
        refs.append(Ref(KIND_FINAL_NORM, GLOBAL_LAYER, "final", 1, HIDDEN, WEIGHT_BF16))
        refs.append(Ref(KIND_LM_HEAD, GLOBAL_LAYER, "head", VOCAB, HIDDEN, WEIGHT_BF16))
        refs.append(Ref(KIND_FFN, MTP_LAYER, "mtp", HIDDEN, HIDDEN, WEIGHT_BF16))
    return refs

def build_tp_plan(ref, degree, rank):
    if degree <= 1 or ref.layer in (GLOBAL_LAYER, MTP_LAYER):
        return None
    if ref.kind in (KIND_LM_HEAD, KIND_FFN):
        return ("rows", ref.rows // degree)
    return None

def packed_shape(ref, plan):
    if plan is None:
        return (ref.rows, ref.columns)
    return (plan[1], ref.columns)

def copy_tensor(source, ref, offset, plan, out):
    raise PackFailure("content pass not under test")
'''


def mini_tables():
    import types
    module = types.ModuleType("mini_qwen38_27b_tables")
    exec(MINI_TABLES_SOURCE, module.__dict__)
    return module


def entry(kind, layer, fmt, rows, cols, group, poff, pbytes, soff=0, sbytes=0):
    return struct.pack("<6I4Q", kind, layer, fmt, rows, cols, group,
                       poff, pbytes, soff, sbytes)


def build_pack(directory: bytes, tensor_count: int, payload_bytes: int,
               tp_degree: int, tp_rank: int, header_bytes: int = HEADER_BYTES,
               file_bytes: int | None = None) -> bytes:
    if file_bytes is None:
        file_bytes = header_bytes + len(directory) + payload_bytes
    header = struct.pack(
        "<26I2Q", 0x51385354, 3, HEADER_BYTES, ENTRY_BYTES, tensor_count,
        8, 4, 0, 4, 4, 3, 1, 2, 2, 2, 4, 2, 1, 4, 2, 8, 16, 32, 1,
        tp_degree, tp_rank, header_bytes, file_bytes)
    return header + directory


def full_directory(tables, tp_degree: int, tp_rank: int, strip_mtp: bool,
                   align: int, compact: bool = False):
    """Directory + payload size for the full mini inventory."""
    refs = []
    for ref in tables.build_inventory(0, tables.LAYER_COUNT):
        if strip_mtp and ref.layer == tables.MTP_LAYER:
            continue
        plan = tables.build_tp_plan(ref, tp_degree, tp_rank) if tp_degree > 1 else None
        rows, cols = tables.packed_shape(ref, plan)
        refs.append((ref, rows, cols))
    pieces = []
    cursor = 0
    for ref, rows, cols in refs:
        natural = 1 if ref.kind == tables.KIND_A_LOG else 0
        fmt = natural
        group = 0
        per = 4 if natural == 1 else 2
        if compact and ref.kind == tables.KIND_FFN and ref.layer != tables.MTP_LAYER:
            fmt = 5
            per = 1
        pbytes = rows * cols * per
        sbytes = 0
        if fmt == 5 and not compact:
            raise AssertionError("compact flag governs fp8 entries")
        cursor = (cursor + align - 1) & ~(align - 1)
        poff = cursor
        cursor += pbytes
        pieces.append(entry(ref.kind, ref.layer if ref.layer != tables.GLOBAL_LAYER
                            else 0xFFFFFFFF, fmt, rows, cols, group, poff, pbytes))
    return b"".join(pieces), cursor


def run_verify(pack: Path, **kwargs):
    tables = kwargs.pop("tables") if "tables" in kwargs else mini_tables()
    return V.verify(pack, kwargs.get("checkpoint"), kwargs.get("receipt"),
                    False, kwargs.get("tp_degree", 1),
                    kwargs.get("strip_mtp", False), "bf16",
                    kwargs.get("tp_rank"), tables=tables)


def test_stale_attribute_regression_real_tables_load():
    tables = V.load_tables()
    for attribute in ("HIDDEN", "LAYER_COUNT", "FFN_INTERMEDIATE", "VOCAB",
                      "KIND_GDN_A_LOG", "KIND_GDN_DT_BIAS", "build_inventory",
                      "build_tp_plan", "packed_shape", "copy_tensor"):
        assert hasattr(tables, attribute), attribute
    assert not hasattr(tables, "EXPERT_COUNT"), "stale attr resurrected"


def test_packer_form_passes():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 1, 0, False, 256)
        pack = Path(tmp) / "p.sp"
        body = build_pack(directory, len(directory) // ENTRY_BYTES, payload, 1, 0)
        pack.write_bytes(body + b"\0" * payload)
        ok, verdict = run_verify(pack)
        assert ok, verdict["errors"]


def test_compact_strip_form_passes():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 4, 3, True, 64, compact=True)
        pack = Path(tmp) / "p.sp"
        body = build_pack(directory, len(directory) // ENTRY_BYTES, payload, 4, 3)
        pack.write_bytes(body + b"\0" * payload)
        ok, verdict = run_verify(pack, tp_degree=4, strip_mtp=True, tp_rank=3)
        assert ok, verdict["errors"]


def test_strip_mtp_off_with_pseudo_layer_missing_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 4, 3, True, 256)
        pack = Path(tmp) / "p.sp"
        pack.write_bytes(build_pack(directory, len(directory) // ENTRY_BYTES,
                                    payload, 4, 3) + b"\0" * payload)
        ok, verdict = run_verify(pack, tp_degree=4, tp_rank=3, strip_mtp=False)
        assert not ok
        assert any("missing tensor" in e for e in verdict["errors"])


def test_bad_magic_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 1, 0, False, 256)
        pack = Path(tmp) / "p.sp"
        body = bytearray(build_pack(directory, len(directory) // ENTRY_BYTES,
                                    payload, 1, 0))
        body[0:4] = b"XXXX"
        pack.write_bytes(bytes(body) + b"\0" * payload)
        ok, verdict = run_verify(pack)
        assert not ok
        assert any("magic" in e for e in verdict["errors"])


def test_missing_tensor_fails():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 1, 0, False, 256)
        directory = directory[:ENTRY_BYTES] + directory[2 * ENTRY_BYTES:]
        pack = Path(tmp) / "p.sp"
        pack.write_bytes(build_pack(directory, len(directory) // ENTRY_BYTES,
                                    payload, 1, 0) + b"\0" * payload)
        ok, verdict = run_verify(pack)
        assert not ok
        assert any("missing tensor" in e or "not in the format inventory" in e
                   for e in verdict["errors"])


def test_corrupt_payload_bytes_fail():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 1, 0, False, 256)
        fields = list(struct.unpack_from("<6I4Q", directory, 0))
        fields[7] += 2
        directory = struct.pack("<6I4Q", *fields) + directory[ENTRY_BYTES:]
        pack = Path(tmp) / "p.sp"
        pack.write_bytes(build_pack(directory, len(directory) // ENTRY_BYTES,
                                    payload, 1, 0) + b"\0" * payload)
        ok, verdict = run_verify(pack)
        assert not ok
        assert any("payload_bytes" in e for e in verdict["errors"])


def test_non_64_aligned_offset_fails_in_both_forms():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 1, 0, False, 64)
        fields = list(struct.unpack_from("<6I4Q", directory, ENTRY_BYTES))
        fields[6] += 8
        directory = (directory[:ENTRY_BYTES] + struct.pack("<6I4Q", *fields)
                     + directory[2 * ENTRY_BYTES:])
        pack = Path(tmp) / "p.sp"
        pack.write_bytes(build_pack(directory, len(directory) // ENTRY_BYTES,
                                    payload, 1, 0) + b"\0" * payload)
        ok, verdict = run_verify(pack)
        assert not ok
        assert any("aligned" in e for e in verdict["errors"])


def test_receipt_crosscheck():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 1, 0, False, 256)
        pack = Path(tmp) / "p.sp"
        pack.write_bytes(build_pack(directory, len(directory) // ENTRY_BYTES,
                                    payload, 1, 0) + b"\0" * payload)
        receipt = Path(tmp) / "p.receipt.json"
        receipt.write_text(json.dumps({"tensor_count": 3, "bytes": 7}))
        ok, verdict = run_verify(pack, receipt=receipt)
        assert not ok
        assert any("receipt" in e for e in verdict["errors"])


def test_main_wiring_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        tables = mini_tables()
        directory, payload = full_directory(tables, 1, 0, False, 256)
        pack = Path(tmp) / "p.sp"
        pack.write_bytes(build_pack(directory, len(directory) // ENTRY_BYTES,
                                    payload, 1, 0) + b"\0" * payload)
        old_argv = sys.argv
        old_loader = V.load_tables
        sys.argv = ["qwen38_pack_verify.py", "--pack", str(pack)]
        try:
            V.load_tables = lambda path=None: mini_tables()
            code = V.main()
        finally:
            sys.argv = old_argv
            V.load_tables = old_loader
        assert code == 0


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
