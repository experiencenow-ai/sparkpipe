"""Per-rank receipt contract for the Ling pack verifier.

The packer writes receipts/rankN.json next to the rankN pack and placement
ships each node only its own rank's receipt, so the verifier must derive the
arm from the verified rank's own receipt (or the placed pack filename) and
must not require receipts/rank0.json off-node. These checks exercise main()
end-to-end on synthetic single-entry packs: the nonzero-rank regression,
the rank-zero path, the boundary proof gated on both end receipts, and the
fail-loud modes, no checkpoint or large pack needed.
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

from ling_stagepack import (  # noqa: E402
    ALIGNMENT, CODEC_BF16, ENTRY_BYTES, EXPERTS, GLOBAL_LAYER, HIDDEN,
    LAYERS, MAGIC, PAYLOAD_BF16, VOCAB,
)

TOOL = ROOT / "tools" / "ling_verify_pack.py"
ARM = "sweep.bf16.tp16"
HEADER_BYTES = 264
DIRECTORY_OFFSET = 512


def pack_bytes(rank: int) -> bytes:
    payload_bytes = HIDDEN * 2
    payload_offset = DIRECTORY_OFFSET + ENTRY_BYTES
    payload_offset = (payload_offset + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    file_bytes = payload_offset + payload_bytes
    header = struct.pack(
        "<20I", MAGIC, 1, HEADER_BYTES, ENTRY_BYTES, 1, 0, 1, 1, 0, 0,
        LAYERS, LAYERS, HIDDEN, VOCAB, EXPERTS, CODEC_BF16, CODEC_BF16,
        CODEC_BF16, 16, rank)
    header += struct.pack("<QQ", DIRECTORY_OFFSET, file_bytes)
    header += b"sweep-test".ljust(HEADER_BYTES - len(header), b"\0")
    entry = struct.pack(
        "<IIIIIIIIQQQQ", 1, GLOBAL_LAYER, PAYLOAD_BF16, CODEC_BF16, 0,
        1, 1, HIDDEN, payload_offset, payload_bytes, 0, 0)
    body = b"".join([
        header,
        b"\0" * (DIRECTORY_OFFSET - HEADER_BYTES),
        entry,
        b"\0" * (payload_offset - DIRECTORY_OFFSET - ENTRY_BYTES),
        bytes((rank + index) % 251 for index in range(payload_bytes)),
    ])
    assert len(body) == file_bytes
    return body


def write_rank(pack_dir: Path, rank: int, receipt: bool = True,
               sha256: str = None) -> Path:
    pack_dir.mkdir(parents=True, exist_ok=True)
    pack = pack_dir / f"{ARM}.rank{rank:x}.sp"
    body = pack_bytes(rank)
    pack.write_bytes(body)
    if receipt:
        receipt_path = pack_dir / "receipts" / f"rank{rank}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps({
            "pack": pack.name, "arm": ARM,
            "sha256": sha256 or hashlib.sha256(body).hexdigest(),
            "file_bytes": len(body), "tensors": 1,
            "tp_degree": 16, "tp_rank": rank,
            "census": {"checkpoint_tensors": 1, "packed": 1,
                       "omitted_mtp": 0},
        }))
    return pack_dir


def run_verify(pack_dir: Path, *extra: str):
    done = subprocess.run(
        [sys.executable, str(TOOL), "--pack-dir", str(pack_dir), *extra],
        capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


def test_nonzero_rank_verifies_from_its_own_receipt():
    with tempfile.TemporaryDirectory() as tmp:
        code, output = run_verify(write_rank(Path(tmp), 7),
                                  "--tp-degree", "16", "--ranks", "7")
        assert code == 0, output
        assert "PASS sweep.bf16.tp16.rank7.sp" in output
        assert "receipt rank7.json" in output
        assert "verified 1 rank packs" in output


def test_rank_zero_dir_still_verifies():
    with tempfile.TemporaryDirectory() as tmp:
        code, output = run_verify(write_rank(Path(tmp), 0),
                                  "--tp-degree", "16", "--ranks", "0")
        assert code == 0, output
        assert "receipt rank0.json" in output


def test_boundary_proof_runs_when_both_end_receipts_exist():
    with tempfile.TemporaryDirectory() as tmp:
        pack_dir = Path(tmp)
        write_rank(pack_dir, 0)
        write_rank(pack_dir, 15)
        code, output = run_verify(pack_dir, "--tp-degree", "16",
                                  "--ranks", "0,15")
        assert code == 0, output
        assert "PASS boundary ranks 0 and 15" in output


def test_missing_own_rank_receipt_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        code, output = run_verify(write_rank(Path(tmp), 7, receipt=False),
                                  "--tp-degree", "16", "--ranks", "7")
        assert code == 1, output
        assert "FAIL receipt" in output
        assert "rank7.json" in output


def test_receipt_sha_mismatch_fails_loud():
    with tempfile.TemporaryDirectory() as tmp:
        wrong = "0" * 64
        code, output = run_verify(write_rank(Path(tmp), 7, sha256=wrong),
                                  "--tp-degree", "16", "--ranks", "7")
        assert code == 1, output
        assert "FAIL receipt" in output
        assert "sha256" in output


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print(f"PASS {name}")
