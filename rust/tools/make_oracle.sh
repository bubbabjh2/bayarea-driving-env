#!/usr/bin/env bash
# Generate every JS-oracle dump the Rust equivalence tests read (rust/oracle/ is gitignored), from the JS sources of
# this checkout, with node. Then `cargo test --release` in rust/ runs the JS-equivalence tests instead of skipping
# them; set BAYENV_REQUIRE_ORACLE=1 to make a missing oracle fail the test run.
#
#   bash rust/tools/make_oracle.sh [--quick] [--only <part>[,<part>...]] [--brain <bayarea-driving-sim checkout>]
#     parts: math world junctions nav gpu traces v2 pop   (default: all)
#     --quick  smaller runs (fewer commuters / steps); the full set takes ~30-60 min and ~6 GB of RAM per node process
#     --brain  the private training repo (v2 / TeacherObs oracle needs its src/brain); default ../bayarea-driving-sim
#              next to this repo. Without it the v2 part is skipped (with a message).
# Needs: node >= 20, the data/ directory (npm run unpack etc., see README "Rebuilding the world"). Builds a missing
# data/junctions.bin first (npm run unpack if data/controls.json is missing, then npm run build:junctions, ~90 s).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
O="$ROOT/rust/oracle"
NODE="node --max-old-space-size=14000"
QUICK=0; ONLY=""; BRAIN="$(cd "$ROOT/.." && pwd)/bayarea-driving-sim"
while [ $# -gt 0 ]; do case "$1" in
  --quick) QUICK=1;; --only) ONLY="$2"; shift;; --brain) BRAIN="$2"; shift;;
  *) echo "unknown argument $1"; exit 2;; esac; shift; done
want() { [ -z "$ONLY" ] || [[ ",$ONLY," == *",$1,"* ]]; }
say() { printf '\n== %s\n' "$*"; }
cd "$ROOT"
[ -f data/roads.meta.json ] || { echo "data/roads.meta.json missing: build or unpack the world first (README)"; exit 1; }
# The unpacked JSON (data/controls.json etc., gitignored) is an input of build:junctions: without controls.json it
# silently builds the junctions without the control crosswalk legs, a different junctions.bin.
if [ ! -f data/controls.json ]; then printf '\n== data/controls.json missing: npm run unpack\n'; node tools/osm/unpack.mjs; fi
# data/junctions.bin is a gitignored build output. Without it the JS loaders build the junctions in memory, and the JS
# GPUSim then packs a different world blob than the Rust host (which always has the full junctions): host_oracle fails.
# So build it first, as `npm run build:junctions` (it also rewrites controls.*, identical to the committed files).
if [ ! -f data/junctions.bin ] || [ ! -f data/junctions.meta.json ]; then
  printf '\n== data/junctions.bin missing: npm run build:junctions\n'
  $NODE tools/build_junctions.mjs data/
  git diff --quiet -- 'data/controls.*' || echo "WARNING: build:junctions changed the committed data/controls.* (git diff data/)"
fi
mkdir -p "$O"
if [ $QUICK = 1 ]; then N2K=1000; STEPS=1000; MATHN=100000; else N2K=2000; STEPS=3000; MATHN=400000; fi

if want math; then say "jsmath: V8 Math.* on generated inputs -> rust/oracle/math.bin (bayenv-world jsmath_oracle)"
  $NODE rust/tools/oracle/gen_math.mjs "$O/math.bin" --n=$MATHN; fi

if want world; then say "world: graph, controls, lane geometry -> rust/oracle/{graph,controls,lg}.* (bayenv-world geom_exact, bayenv-world check)"
  $NODE rust/tools/oracle/dump_world.mjs data "$O" lg graph controls; fi

if want junctions; then say "junctions: npm run build:junctions on a scratch copy -> rust/oracle/ref/{junctions,controls}.* (geom_exact::junctions)"
  # build_junctions.mjs writes junctions.* and rewrites controls.* in its directory: run it on a scratch directory of
  # symlinks (copies for the files it writes) so data/ is not modified
  JT="$(mktemp -d)"
  for f in data/*; do b="$(basename "$f")"; case "$b" in junctions.*|controls.*) [ -f "$f" ] && cp "$f" "$JT/";; *) ln -s "$ROOT/$f" "$JT/$b";; esac; done
  $NODE tools/build_junctions.mjs "$JT/"
  mkdir -p "$O/ref"; cp "$JT/junctions.bin" "$JT/junctions.meta.json" "$JT"/controls.* "$O/ref/"; rm -rf "$JT"; fi

if want nav; then say "nav: route.js routes + packRoutes -> rust/oracle/nav/nav.json (bayenv-world nav_oracle)"
  $NODE rust/tools/oracle/nav_dump.mjs data "$O/nav"; fi

if want gpu; then say "gpu: GPUSim on a recording WebGPU mock -> rust/oracle/gpu10k (bayenv-gpu host_oracle, pack_oracle)"
  $NODE rust/tools/oracle/gpu_mock.mjs data "$O/gpu10k" --n=10000 --seed=11; fi

if want traces; then say "sim traces (state + obs every step) -> rust/oracle/sim/traces (bayenv-sim obs_trace, events, world_hops)"
  T="$O/sim/traces"; mkdir -p "$T"
  $NODE tools/rust/trace.mjs --world scenario:junction-s1 --policy cruisenoise --steps $STEPS --every 10 --obs 1 --out "$T/junction-s1"
  $NODE tools/rust/trace.mjs --world synthetic --policy cruisenoise --n 300 --steps $STEPS --every 10 --obs 1 --out "$T/synthetic"
  $NODE tools/rust/trace.mjs --world real --n $N2K --policy cruisenoise --steps $STEPS --every 20 --obs 1 --opts '{"startTime":25200}' --out "$T/real2k"
  echo "(the full scenario x policy matrix: bash tools/rust/matrix.sh <outdir>, checked with rust/target/release/trace_check)"; fi

if want v2; then say "v2 / eyes.near / TeacherObs encoder -> rust/oracle/sim/v2 (bayenv-sim obs_v2_trace)"
  if [ -f "$BRAIN/src/brain/driver.js" ]; then
    V="$O/sim/v2"; mkdir -p "$V"
    $NODE tools/rust/v2_oracle.mjs --brain "$BRAIN" --world scenario:junction-s1 --policy cruisenoise --steps $STEPS --every 20 --out "$V/junction-s1"
    $NODE tools/rust/v2_oracle.mjs --brain "$BRAIN" --world scenario:urban-s1 --policy cruisenoise --steps $STEPS --every 10 --out "$V/urban-s1"
    $NODE tools/rust/v2_oracle.mjs --brain "$BRAIN" --world real --n $N2K --policy cruisenoise --steps $STEPS --every 20 --opts '{"startTime":25200}' --out "$V/real2k"
  else echo "SKIPPED v2: no $BRAIN/src/brain/driver.js (pass --brain <path of the bayarea-driving-sim checkout>)"; fi; fi

if want pop; then say "populations -> rust/oracle/sim/pop (bayenv-sim population_gen)"
  P="$O/sim/pop"; mkdir -p "$P"
  $NODE tools/rust/export_pop.mjs --world synthetic --n 300 --seed 1 --out "$P/synth300-s1"
  $NODE tools/rust/export_pop.mjs --world scenario:junction-s1 --seed 1 --out "$P/scn-junction-s1"
  $NODE tools/rust/export_pop.mjs --world real --n $N2K --seed 1 --out "$P/real-s1"
  $NODE tools/rust/export_pop.mjs --world real --n $N2K --seed 1 --opts '{"returnTrips":true}' --out "$P/real-s1-ret"
  $NODE tools/rust/export_pop.mjs --world real --n $N2K --seed 1 --parallel --nostations --opts '{"returnTrips":true,"region":{"preset":"sf"}}' --out "$P/sf-s1"; fi

say "done: rust/oracle. Run: cd rust && BAYENV_REQUIRE_ORACLE=1 cargo test --release --workspace --exclude bayenv-py"
