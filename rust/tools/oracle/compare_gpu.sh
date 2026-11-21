#!/usr/bin/env bash
# JS GPU sim (headless Chrome, test/gpu/run.mjs) vs the Rust wgpu host: same world, seed, constant actions
# (gas 0.3), same runSteps batches; per-car state compared after STEPS steps.
#   rust/tools/oracle/compare_gpu.sh [n=10000] [steps=200] [batch=10] [seed=11]
set -euo pipefail
N=${1:-10000}; STEPS=${2:-200}; BATCH=${3:-10}; SEED=${4:-11}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
OUT=rust/oracle/cmp-$N-$SEED; mkdir -p "$OUT"
export GPU_SCRATCH=${GPU_SCRATCH:-$(mktemp -d)}
node --max-old-space-size=14000 rust/tools/oracle/gpu_mock.mjs data "$OUT" --n=$N --seed=$SEED --k=2 >/dev/null
node test/gpu/run.mjs rust/tools/oracle/gpu_run.html --query="n=$N&seed=$SEED&steps=$STEPS&batch=$BATCH&name=js" --out="$OUT" | grep -E '"(driving|crashes|crashScenes|avgSpeed|msPerStep)"' || true
cd rust
cargo run -q --release -p bayenv-gpu --example oracle_run -- ../data "../$OUT" $STEPS $BATCH "../$OUT/rs.bin" full 2>/dev/null | grep -E "mismatch|dispatches|steps in"
cargo run -q --release -p bayenv-gpu --example compare_state -- "../$OUT/js.bin" "../$OUT/rs.bin" 2>/dev/null
