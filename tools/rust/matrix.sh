#!/bin/bash
# Differential matrix: JS trace + Rust check for every scenario world x policy (+ option variants).
#   bash tools/rust/matrix.sh <outdir> [steps] [every]
set -u
OUT=${1:-/tmp/bayenv-matrix}; STEPS=${2:-4000}; EVERY=${3:-10}
ROOT=$(cd "$(dirname "$0")/../.." && pwd); BIN=$ROOT/rust/target/release/trace_check
mkdir -p "$OUT"; cd "$ROOT"
pass=0; fail=0
run() { # name args...
  local name=$1; shift
  node tools/rust/trace.mjs --out "$OUT/$name" --steps $STEPS --every $EVERY "$@" > "$OUT/$name.js.log" 2>&1 || { echo "JSFAIL $name"; fail=$((fail+1)); return; }
  if $BIN "$OUT/$name" 5 > "$OUT/$name.rs.log" 2>&1; then pass=$((pass+1)); echo "ok   $name $(tail -1 "$OUT/$name.rs.log" | grep -o '[0-9]* steps/s')"; else fail=$((fail+1)); echo "FAIL $name"; head -4 "$OUT/$name.rs.log"; fi
  rm -f "$OUT/$name/trace.pack"
}
for w in $(ls data/scenarios | grep -v index.json); do
  for p in cruise cruisenoise random; do run "$w-$p" --world scenario:$w --policy $p; done
done
run synth-cruise --world synthetic --policy cruise --n 300
run synth-noise --world synthetic --policy cruisenoise --n 300
run synth-sparse --world synthetic --policy cruise --n 300 --mode sparse
run synth-empty --world synthetic --policy cruisenoise --n 300 --mode empty
run junc-opts --world scenario:junction-s1 --policy cruisenoise --opts '{"pathSight":1,"junctionSight":1,"deadlockBreak":1,"spawnLaneRoute":1}'
run grid-opts --world scenario:grid-s1 --policy cruisenoise --opts '{"pathSight":1,"junctionSight":1,"deadlockBreak":1,"spawnLaneRoute":1,"passWrecks":true}'
run urban-sparse --world scenario:urban-s1 --policy cruise --mode sparse
echo "pass $pass fail $fail"
