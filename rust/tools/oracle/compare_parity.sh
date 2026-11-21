#!/usr/bin/env bash
# Rust GPU host vs the JS GPUSim for populations, streaming, refuel plans / pumps, obstacles, snapshots, resetDay.
#  1. mock runs (node, no GPU): gpu_parity_mock.mjs --scenario=pop|stream|obstacles|snapshot -> parity_check (bit-identical)
#  2. Chrome run (test/gpu/run.mjs + gpu_parity_run.html) vs parity_run: commuter population (+ obstacles), gas 0.3,
#     per-car state after STEPS steps (compare_state) and host counters / slot tables.
#   rust/tools/oracle/compare_parity.sh [data=data] [steps=1000]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
DATA=${1:-data}; STEPS=${2:-1000}
export GPU_SCRATCH=${GPU_SCRATCH:-$(mktemp -d)}
cd rust && cargo build -q --release -p bayenv-gpu --examples && cd ..
for s in pop stream obstacles snapshot; do
  O=rust/oracle/parity-$s; mkdir -p $O
  node --max-old-space-size=14000 rust/tools/oracle/gpu_parity_mock.mjs "$DATA" $O --scenario=$s >/dev/null
  echo "== $s: $(rust/target/release/examples/parity_check "$DATA" $O $([ $s = pop ] && echo native-pop) | tail -1)"
done
O=rust/oracle/parity-pop
node test/gpu/run.mjs rust/tools/oracle/gpu_parity_run.html --query="steps=$STEPS&batch=10&obstacles=40&name=js" --in=$O/pop.json,$O/pop.bin --out=$O >/dev/null
rust/target/release/examples/parity_run "$DATA" $O $STEPS 10 rs 32768 25200 1 40 3 | tail -1 | cut -c1-120
rust/target/release/examples/compare_state $O/js.bin $O/rs.bin | head -1
python3 - "$O" <<'PY'
import json, sys, numpy as np
O = sys.argv[1]; a = json.load(open(O + '/js.json')); b = json.load(open(O + '/rs.json')); cap = len(a['slotTrip'])
print('counters (js, rs):', {k: (a['stats'][k], b['stats'][k]) for k in a['stats']})
print('slotTrip differ', sum(x != y for x, y in zip(a['slotTrip'], b['slotTrip'])), ' outcome differ', sum(x != y for x, y in zip(a['outcome'], b['outcome'])))
A = np.fromfile(O + '/js.bin', np.uint32); B = np.fromfile(O + '/rs.bin', np.uint32)
va, vb = A[:cap * 40].reshape(cap, 40), B[:cap * 40].reshape(cap, 40); pa, pb = A[cap * 40:].view(np.float32).reshape(cap, 4), B[cap * 40:].view(np.float32).reshape(cap, 4)
act = (va[:, 7] != 0) | (vb[:, 7] != 0); dp = np.hypot(pa[:, 0] - pb[:, 0], pa[:, 2] - pb[:, 2])[act]; dv = np.abs(va[act, 2].view(np.float32) - vb[act, 2].view(np.float32))
print('cars %d: same state + edge %.2f%%, within 0.5 m and 0.1 m/s %.2f%%' % (act.sum(), 100 * ((va[act, 7] == vb[act, 7]) & (va[act, 0] == vb[act, 0])).mean(), 100 * ((dp < 0.5) & (dv < 0.1)).mean()))
PY
