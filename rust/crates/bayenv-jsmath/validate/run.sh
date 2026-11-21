#!/usr/bin/env bash
# Full bit-exactness validation of bayenv-jsmath against the local node.
#   N=10000000 ./validate/run.sh [outdir]
# Env: N (records per function, default 10M), KEEP=1 keeps the .bin files,
#      JITCHECK=0 skips the --jitless / --no-turbofan cross-checks.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${N:-10000000}"
OUT="${1:-${TMPDIR:-/tmp}/bayenv-jsmath-vectors}"
FNS="sin cos tan atan asin acos exp expm1 log log2 tanh pow powop atan2 hypot2 hypot3 round sign fround sqrt floor ceil toint32 touint32 imul max2 min2 max3 min3"
echo "node $(node -v) / V8 $(node -p process.versions.v8) / $(uname -sm); N=$N; out=$OUT"
cargo build -q --release --example validate
BIN="$(cargo metadata --format-version 1 --no-deps | node -e 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>console.log(JSON.parse(s).target_directory))')/release/examples/validate"
fail=0
for f in $FNS; do
  node validate/gen.mjs gen "$OUT" "$N" "$f" >/dev/null
  "$BIN" "$OUT" "$f" | tail -n +2 || fail=1
  if [ "${JITCHECK:-1}" != 0 ]; then
    node --jitless validate/gen.mjs check "$OUT" "$f" | sed 's/^/    /' || fail=1
    node --no-turbofan validate/gen.mjs check "$OUT" "$f" | sed 's/^/    /' || fail=1
  fi
  [ "${KEEP:-0}" = 1 ] || rm -f "$OUT/$f.bin"
done
[ $fail = 0 ] && echo "ALL BIT-EXACT" || { echo "MISMATCHES FOUND"; exit 1; }
