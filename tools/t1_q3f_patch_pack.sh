#!/bin/sh
set -eu
pack=$1
tp_degree=$2
tp_rank=$3
tools=$4
warm=${WARM:-/mnt/model-warm/qwen3.8-flash-next-fp8}
[ -f "$pack" ] || { echo "no pack: $pack" >&2; exit 1; }
lock_state=$(lsattr -d "$pack" 2>/dev/null || sudo -n lsattr -d "$pack")
case $lock_state in
	*i*) ;;
	*)
		if [ ! -f "$pack.patch.json" ]; then
			echo "pack not chattr +i locked and no prior patch evidence: $pack ($lock_state)" >&2
			exit 1
		fi
		echo "resuming unlocked pack with prior patch evidence: $pack"
		;;
esac
if command -v lsof >/dev/null 2>&1; then
	holders=$(lsof -t "$pack" 2>/dev/null || true)
elif command -v fuser >/dev/null 2>&1; then
	holders=$(fuser "$pack" 2>/dev/null || true)
else
	echo "neither lsof nor fuser available for the hold pre-check" >&2
	exit 1
fi
[ -z "$holders" ] || { echo "pack held by: $holders" >&2; exit 1; }
old_sha=$(cut -d' ' -f1 "$pack.sha256")
sudo -n chattr -i "$pack"
rc=0
python3 "$tools/qwen4_flash_scale_plane_patch.py" --pack "$pack" --warm "$warm" \
	--tp-degree "$tp_degree" --tp-rank "$tp_rank" --write \
	--json-out "$pack.patch.json" || rc=$?
if [ "$rc" != 0 ]; then
	echo "PATCH FAILED rc=$rc - $pack left UNLOCKED, evidence: $pack.patch.json" >&2
	exit "$rc"
fi
"$tools/build-qwen4flash-experts-manifest" "$pack" "$tp_degree" "$tp_rank"
python3 - "$pack" "$old_sha" <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path
import time
pack = Path(sys.argv[1])
old_sha = sys.argv[2]
digest = hashlib.sha256()
with pack.open("rb") as fh:
	while True:
		chunk = fh.read(1 << 24)
		if not chunk:
			break
		digest.update(chunk)
new_sha = digest.hexdigest()
patch = json.loads(Path(str(pack) + ".patch.json").read_text())
for suffix in (".receipt.json", ".g5nsp.receipt.json"):
	path = Path(str(pack) + suffix)
	if not path.is_file():
		continue
	receipt = json.loads(path.read_text())
	receipt["replaced_sha256"] = receipt["output_sha256"]
	receipt["output_sha256"] = new_sha
	receipt["verdict"] = (
		"scale-plane re-emission applied: fp8 planes widened from warm bf16 "
		"weight_scale_inv; payload probes byte-exact; verify pending")
	receipt["repair_tool"] = "tools/qwen4_flash_scale_plane_patch.py"
	receipt["repair"] = {key: patch[key] for key in (
		"fp8_entries", "planes_clean", "planes_patched", "scale_bytes_patched")}
	receipt["repair"]["prior_sha256"] = old_sha
	receipt["repaired_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
	path.write_text(json.dumps(receipt, indent=1) + "\n")
Path(str(pack) + ".sha256").write_text(f"{new_sha}  {pack.name}\n")
print(json.dumps({"pack": pack.name, "prior_sha256": old_sha, "output_sha256": new_sha}))
PYEOF
rc=0
python3 "$tools/qwen4_flash_pack_verify.py" --pack "$pack" --checkpoint "$warm" \
	--tp-degree "$tp_degree" --tp-rank "$tp_rank" --no-mtp \
	--source-layout fp8-official || rc=$?
python3 "$tools/qwen4_flash_pack_stamp_check.py" --pack "$pack" \
	--tp-degree "$tp_degree" > "$pack.stampcheck.json" || rc=$?
if [ "$rc" != 0 ]; then
	python3 - "$pack" <<'PYEOF'
import json
import sys
from pathlib import Path
import time
pack = Path(sys.argv[1])
for suffix in (".receipt.json", ".g5nsp.receipt.json"):
	path = Path(str(pack) + suffix)
	if not path.is_file():
		continue
	receipt = json.loads(path.read_text())
	receipt["verdict"] = "POST-PATCH VERIFY FAILED - pack NOT validated; left unlocked for operator"
	receipt["failed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
	path.write_text(json.dumps(receipt, indent=1) + "\n")
PYEOF
	echo "VERIFY FAILED rc=$rc - $pack left UNLOCKED with failure receipts" >&2
	exit "$rc"
fi
python3 - "$pack" <<'PYEOF'
import json
import sys
from pathlib import Path
import time
pack = Path(sys.argv[1])
for suffix in (".receipt.json", ".g5nsp.receipt.json"):
	path = Path(str(pack) + suffix)
	if not path.is_file():
		continue
	receipt = json.loads(path.read_text())
	receipt["verdict"] = (
		"scale-plane re-emission VERIFIED: qwen4_flash_pack_verify "
		"--source-layout fp8-official byte-exact PASS + stamp-check clean; "
		"streaming sha == receipt output_sha256")
	receipt["verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
	path.write_text(json.dumps(receipt, indent=1) + "\n")
PYEOF
sudo -n chattr +i "$pack"
lsattr -d "$pack" "$pack.experts" "$pack.sha256" "$pack.receipt.json"
echo "PATCH+VERIFY+RELOCK OK: $pack"
