#!/usr/bin/env bash
# Fleet census for the placed hy4 v2 packs: per node verify the 4 rank files,
# the sha gate, the manifest digest, the v2 experts sidecar, and hold the
# pack immutable (chattr +i). Run from the emit node (~/hy4-v2-emit).
set -u
BASE="sparkdata/hy4.fp8.tp16"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
PASS=0
FAIL=0
rank_node() {
	local r=$1
	if [ "$r" -lt 10 ]; then echo "spark$r"; else printf "spark%x" "$r"; fi
}
one() {
	local r=$1
	local xx
	xx=$(printf "%02d" $r)
	local node
	node=$(rank_node $r)
	local out
	out=$($SSH "$node" "cd ~/$BASE/packs/rank-$xx 2>/dev/null || { echo CENSUS-FAIL $xx no-dir; exit 1; }
		n=0
		for f in model-fp8-tp16-rank-$xx.safetensors \
			model-fp8-tp16-rank-$xx.safetensors.sha256 \
			model-fp8-tp16-rank-$xx.safetensors.experts \
			manifest-rank-$xx.json; do
			[ -f \$f ] || { echo CENSUS-FAIL $xx missing \$f; exit 1; }
		done
		sha256sum -c model-fp8-tp16-rank-$xx.safetensors.sha256 > /dev/null || { echo CENSUS-FAIL $xx sha; exit 1; }
		WANT=\$(python3 -c \"import json;print(json.load(open('manifest-rank-$xx.json'))['file_sha256'])\")
		HAVE=\$(cut -d' ' -f1 model-fp8-tp16-rank-$xx.safetensors.sha256)
		[ \"\$WANT\" = \"\$HAVE\" ] || { echo CENSUS-FAIL $xx manifest-digest; exit 1; }
		python3 - <<'EOF' || exit 1
import struct, sys
f = open('model-fp8-tp16-rank-$xx.safetensors.experts', 'rb')
h = f.read(16)
magic, version, ranges = struct.unpack('<4sII', h[:12])
assert magic == b'WEPX' and version == 2 and ranges == 4928, (magic, version, ranges)
EOF
		IMM=\$(lsattr model-fp8-tp16-rank-$xx.safetensors 2>/dev/null | cut -d' ' -f1)
		case \"\$IMM\" in *i*) ;; *) sudo -n chattr +i model-fp8-tp16-rank-$xx.safetensors 2>/dev/null || echo CENSUS-WARN $xx chattr ;; esac
		IMM=\$(lsattr model-fp8-tp16-rank-$xx.safetensors 2>/dev/null | cut -d' ' -f1)
		case \"\$IMM\" in *i*) echo CENSUS-OK $xx immutable ;; *) echo CENSUS-OK $xx mutable ;; esac
	") || { echo "$out" | tail -1; FAIL=$((FAIL+1)); return 1; }
	echo "$out" | tail -1
	case "$(echo "$out" | tail -1)" in
		CENSUS-OK*) PASS=$((PASS+1)) ;;
		*) FAIL=$((FAIL+1)) ;;
	esac
}
for r in $(seq 0 15); do
	one $r
done
echo "CENSUS pass=$PASS fail=$FAIL $(date -u +%H:%M:%SZ)"
