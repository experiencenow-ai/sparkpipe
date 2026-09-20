"""qwen4_flash_pack_verify: the receipt pair and the structure-only form.

The fleet receipt convention is <pack>.receipt.json + <pack>.sha256; the
verifier emits it on PASS (--emit-receipt) but never overwrites a
packer-written receipt, and --checkpoint is now optional so placed packs
can be verified structure-only on nodes without the warm source.
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

import qwen4_flash_pack_verify as V

TOOL = ROOT / "tools" / "qwen4_flash_pack_verify.py"


def run_tool(pack: Path, *extra: str):
    done = subprocess.run(
        [sys.executable, str(TOOL), "--pack", str(pack), *extra],
        capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def test_emit_receipt_writes_the_fleet_pair():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "qwenflash.tp8.rank3.pack"
        pack.write_bytes(b"x" * 4096)
        header = {"first_layer_index": 0, "layer_count": 48,
                  "tensor_count": 1246}
        written = V.emit_receipt(pack, header, 8, 3, True)
        assert written is not None
        receipt_path, sidecar = written
        receipt = json.loads(receipt_path.read_text())
        digest = hashlib.sha256(pack.read_bytes()).hexdigest()
        assert receipt["output_sha256"] == digest == receipt["sha256"]
        assert receipt["verify_mode"].startswith("structure-only")
        assert receipt["tp_degree"] == 8 and receipt["tp_rank"] == 3
        assert sidecar.read_text() == f"{digest}  {pack.name}\n"


def test_emit_receipt_never_overwrites_a_packer_receipt():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "qwenflash.tp8.rank3.pack"
        pack.write_bytes(b"y" * 128)
        packer_receipt = Path(str(pack) + ".receipt.json")
        packer_receipt.write_text(json.dumps({"kind": "packer", "output_sha256": "0"}))
        header = {"first_layer_index": 0, "layer_count": 48, "tensor_count": 1}
        assert V.emit_receipt(pack, header, 8, 3, True) is None
        assert json.loads(packer_receipt.read_text())["kind"] == "packer"


def test_structure_only_mode_runs_without_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "short.pack"
        pack.write_bytes(b"\0" * 8)
        code, output = run_tool(pack)
        assert code == 1, output
        assert "truncated" in output
        assert "--checkpoint" not in output
        assert "required" not in output


def test_header_geometry_gate_still_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        pack = Path(tmp) / "qwenflash.tp8.rank3.pack"
        from qwen4_flash_stagepack import ENTRY_BYTES, FORMAT_VERSION, HEADER_BYTES
        header = struct.pack(
            "<26I2Q", 0x50533451, FORMAT_VERSION + 1, HEADER_BYTES, ENTRY_BYTES,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            HEADER_BYTES, HEADER_BYTES)
        pack.write_bytes(header)
        code, output = run_tool(pack)
        assert code == 1, output


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
