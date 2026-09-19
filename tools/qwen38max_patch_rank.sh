#!/bin/sh
# Patch-swap ONE placed qwen38max TP16 nvfp4 rank pack in place:
# drift-check against the old receipt, codec-contract patch (weight_format
# 4 -> 8 + per-expert f32 tails), entry audit, verify, chattr unlock/swap/
# relock, regenerate the experts manifest, verify the placed result.
set -eu
rank=$1
checkpoint=/mnt/model-warm/qwen3.8-max-nvfp4-radixark-bf16-spine
tools=/tmp/t1qmax_stage/tools
pack_dir=$HOME/sparkdata/qwenmax.nvfp4.tp16/packs
name=$(printf 'qwenmax.nvfp4.tp16.rank%x' "$rank")
placed=$pack_dir/$name.sp
scratch=/tmp/t1qmax_rebuild
patched=$scratch/$name.patched.sp
receipt=$placed.receipt.json
mkdir -p "$scratch"
cd "$tools"

precondition=$(sudo -n /usr/local/sbin/sparkcap --mem 65536 python3 - "$placed" "$receipt" <<'PY'
import hashlib, json, os, sys
pack, receipt_path = sys.argv[1], sys.argv[2]
if os.stat(pack).st_dev != os.stat("/tmp").st_dev:
    sys.exit("placed pack and /tmp are on different filesystems; swap would not be atomic")
receipt = json.load(open(receipt_path))
digest = hashlib.sha256()
with open(pack, "rb") as handle:
    for chunk in iter(lambda: handle.read(1 << 24), b""):
        digest.update(chunk)
actual = digest.hexdigest()
if actual != receipt["output_sha256"]:
    sys.exit(f"placed pack drifted from its receipt: {actual} != {receipt['output_sha256']}")
print(actual)
PY
)
echo "precondition_ok old_sha256=$precondition"

patch_json=$(sudo -n /usr/local/sbin/sparkcap --mem 65536 python3 "$tools/qwen38max_patch_pack.py" \
	--old-pack "$placed" \
	--checkpoint "$checkpoint" \
	--tp-degree 16 --tp-rank "$rank" \
	--out "$patched")
echo "patch $patch_json"
new_sha=$(printf '%s\n' "$patch_json" | python3 -c "import json,sys;print(json.load(sys.stdin)[\"sha256\"])")

sudo -n /usr/local/sbin/sparkcap --mem 65536 python3 - "$placed" "$patched" <<'PY'
import json, struct, sys
HEADER, ENTRY = 128, 56
HS, ES = struct.Struct("<28I2Q"), struct.Struct("<6I4Q")
ALIGN = 256
EXPERT_KINDS = (6, 7, 8)
EXPERT_ROWS = {6: 2048, 7: 2048, 8: 8192}

def entries_of(path):
    with open(path, "rb") as handle:
        header = HS.unpack(handle.read(HEADER))
        count = header[4]
        entries = [ES.unpack(handle.read(ENTRY)) for _ in range(count)]
    return header, entries

def audit(old_pack, new_pack):
    old_header, old = entries_of(old_pack)
    new_header, new = entries_of(new_pack)
    if old_header[:29] != new_header[:29]:
        sys.exit("header fields 0..28 changed")
    if len(old) != len(new):
        sys.exit("tensor_count changed")
    payload_base = -(-((HEADER + len(new) * ENTRY)) // ALIGN) * ALIGN
    cursor = payload_base
    report = {"expert_entries": 0, "payload_bytes_total": 0}
    spans = []
    for index, (o, n) in enumerate(zip(old, new)):
        okind, olayer, ofmt, orows, ocols, ogroup, opoff, opb, osoff, osb = o
        nkind, nlayer, nfmt, nrows, ncols, ngroup, npoff, npb, nsoff, nsb = n
        if (okind, olayer, orows, ocols) != (nkind, nlayer, nrows, ncols):
            sys.exit(f"entry {index}: kind/layer/shape changed")
        if opb != npb:
            sys.exit(f"entry {index}: payload_bytes {opb} -> {npb}")
        report["payload_bytes_total"] += npb
        if okind in EXPERT_KINDS:
            resident = orows // EXPERT_ROWS[okind]
            if nfmt != 8 or ngroup != 16:
                sys.exit(f"entry {index}: expert entry not format 8 group 16")
            if ofmt == 8:
                if nsb != osb:
                    sys.exit(f"entry {index}: already-patched scale_bytes changed")
                spans.append(("expert_scale", [(osoff + k * (nsb // resident), nsoff + k * (nsb // resident)) for k in range(resident)]))
            elif nsb != osb + resident * 8:
                sys.exit(f"entry {index}: scale_bytes delta != resident*8")
            else:
                plane = (orows // resident) * (ocols // 16)
                spans.append(("expert_scale", [(osoff + k * plane, nsoff + k * (plane + 8)) for k in range(resident)]))
            report["expert_entries"] += 1
        else:
            if (ofmt, ogroup, osb) != (nfmt, ngroup, nsb):
                sys.exit(f"entry {index}: non-expert entry fields changed")
            if osb:
                spans.append(("scale", [(osoff, nsoff)]))
        if npoff != -(-cursor // ALIGN) * ALIGN:
            sys.exit(f"entry {index}: payload offset not the walked aligned cursor")
        if nsb and nsoff != npoff + npb:
            sys.exit(f"entry {index}: scale offset not payload-adjacent")
        spans.append(("payload", [(opoff, npoff)]))
        cursor = npoff + npb + nsb
    if cursor != new_header[29]:
        sys.exit(f"walked end {cursor} != file_bytes {new_header[29]}")
    with open(old_pack, "rb") as old_handle, open(new_pack, "rb") as new_handle:
        for kind_name, pairs in spans:
            step = 64 if kind_name != "expert_scale" else 8
            for old_off, new_off in pairs:
                old_handle.seek(old_off)
                new_handle.seek(new_off)
                if old_handle.read(step) != new_handle.read(step):
                    sys.exit(f"{kind_name} bytes differ at old {old_off} new {new_off}")
    report["verdict"] = "PASS"
    print(json.dumps(report, sort_keys=True))

audit(sys.argv[1], sys.argv[2])
PY

python3 - "$receipt" "$patched" "$new_sha" "$placed" <<'PY'
import json, sys
old_receipt, patched, new_sha, placed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
receipt = json.load(open(old_receipt))
receipt["bytes"] = __import__("os").path.getsize(patched)
receipt["output_sha256"] = new_sha
receipt["tool"] = "tools/qwen38max_patch_pack.py"
receipt["file"] = placed
receipt["patched_from_output_sha256"] = json.load(open(old_receipt))["output_sha256"]
receipt.setdefault("weight_formats", {})["routed_experts"] = "nvfp4_e2m1_e4m3b16_f32tail"
with open(patched + ".receipt.json", "w") as handle:
    json.dump(receipt, handle, indent=2, sort_keys=True)
    handle.write("\n")
print("receipt_written", patched + ".receipt.json")
PY

sudo -n /usr/local/sbin/sparkcap --mem 65536 python3 "$tools/qwen38max_tp16_rank_verify.py" \
	--pack "$patched" --tp-degree 16 --tp-rank "$rank" \
	--receipt "$patched.receipt.json" --recompute-file-hash 2>&1 | tail -2

sudo -n /usr/local/sbin/sparkcap "$tools/qwen38max_experts_manifest" "$patched"

sudo -n chattr -i "$placed" "$receipt" "$placed.experts" 2>/dev/null || true
mv "$placed" "$scratch/$name.sp.replaced"
mv "$placed.experts" "$scratch/$name.sp.experts.replaced" 2>/dev/null || true
mv "$receipt" "$scratch/$name.sp.receipt.json.replaced"
mv "$patched" "$placed"
mv "$patched.experts" "$placed.experts"
mv "$patched.receipt.json" "$receipt"
sudo -n chattr +i "$placed" "$receipt" "$placed.experts"
lsattr "$placed" "$receipt" "$placed.experts" | cut -d' ' -f1

sudo -n /usr/local/sbin/sparkcap --mem 65536 python3 "$tools/qwen38max_tp16_rank_verify.py" \
	--pack "$placed" --tp-degree 16 --tp-rank "$rank" \
	--receipt "$receipt" --recompute-file-hash 2>&1 | tail -2
echo "rank $rank patched swapped relocked verified"
