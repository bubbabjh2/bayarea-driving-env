# bayenv-sim benchmarks

Apple M4 Max (10 performance + 4 efficiency cores, 36 GB), macOS, node v22.22.0, rustc 1.100.

Workload: `tools/rust/bench_js.mjs` (JS) and `bin/bench` / `bin/bench_vec` (Rust) run the same env and the same
deterministic policy (`cruise`: lane-keeping P controller + speed / gap pedals, computed from the state). Start at
07:00 (`startTime 25200`), 600 warm-up steps, then timed steps at dt = 0.1 s. Policy time is included on both
sides. "agents" = mean driving cars per step. The mean agent counts are identical in JS and Rust (the runs are
the same simulation). steps/s = env steps per second, agent-steps/s = driving cars x steps per second.

* synthetic = `buildGraph(makeSynthGraph({blocks 12, cross 6, bridges 2}))` with the README population options
  (few cars are on the road at 07:00, so these rows mostly measure fixed per-step cost).
* real = the Bay Area world from `data/` (two tiers, lane geometry, junctions, controls, stations, surfaces).

## Single env

| world | n | agents | JS steps/s | Rust 1 thread | speedup | Rust parallel phases (14 threads) | Rust `fast` + native |
|---|---|---|---|---|---|---|---|
| synthetic | 300 | 0.3 | 3,416 | 535,882 | 157x | | |
| synthetic | 2,000 | 1.4 | 3,399 | 78,540 | 23x | | |
| synthetic | 10,000 | 33.7 | 1,557 | 11,886 | 7.6x | | |
| real | 2,000 | 229 | 821 | 2,619-3,208 | 3.2-3.9x | | 3,474 (4.2x) |
| real | 10,000 | 1,059 | 148 | 372-489 | 2.5-3.3x | 780 (5.3x) | 569 (3.8x) |
| real | 100,000 | 11,630 | 7.4 | 26.4 | 3.6x | 44.7 (6.0x) | 28.0 / 45.4 par (6.1x) |

agent-steps/s at real 100k: JS 86k, Rust 307k (1 thread), 528k (parallel phases).

Real 100k after the parallel-phase work (2026-11-20, under the bench lock at load ~6, 3 runs, BAYENV_PAR=1, 200 steps
from 07:00, 15.5k driving): **117 +- 1 steps/s, 1.82 M agent-steps/s** (15.8x JS); with the lookahead refuel-plan
prefetch 121-125 steps/s. Per step (ms): integrate 2.46 (parallel physics 2.0), leaders 1.97 (phase-1 base 1.13,
P2 speculation ~0.7, ordered commit 0.12), spawn 1.23 (refuel plans ~1.1, routing), contacts 0.55 (parallel pre-scan
0.36), lanes 0.55, controls 0.47, scenes 0.29, drive 0.19, hash 0.15, traffic + reroute 0.15.
Ranges = repeated runs on a machine shared with other jobs (load average up to ~15 during some rows).

## Many envs per process (VecEnv, 14 threads, independent envs, each single-threaded)

| world | n per env | envs | agents per env | env-steps/s | agent-steps/s | vs one JS process |
|---|---|---|---|---|---|---|
| real | 2,000 | 28 | 304 | 25,748 | 7.82 M | 42x (JS 188k) |
| real (fast + native) | 2,000 | 28 | 304 | 26,516 | 8.05 M | 43x |
| real | 10,000 | 14 | 1,486 | 2,912 | 4.33 M | 28x (JS 157k) |
| synthetic | 10,000 | 28 | 50 | 89,860 | 4.49 M | 85x (JS 53k) |

Thread scaling, real 2k x 28 envs: 1 thread 0.58 M, 4 threads 3.28 M, 8 threads 5.74 M, 10 threads 7.93 M,
14 threads 9.29 M agent-steps/s. Before removing Arc clones of the shared world from the hot paths, 14 threads
were *slower* than 4 (atomic refcount on one cache line).

## VecEnv, single-threaded envs (2026-11-20, under the bench lock, load ~6-12, cruise policy, 14 rayon threads)

| world | envs | agents per env | agent-steps/s before | after (memory work) | max RSS before | after | per-env memory before | after |
|---|---|---|---|---|---|---|---|---|
| real 2k | 1 | 242 | 0.80 M | 0.81 M | 1,226 MB | 1,222 MB | | |
| real 2k | 2 | 242 | 1.57 M | 1.59 M | 1,324 MB | 1,204 MB | | |
| real 2k | 4 | 242 | 3.07 M | 3.08 M | 1,505 MB | 1,218 MB | | |
| real 2k | 8 | 242 | 6.22 M | 6.29 M | 1,886 MB | 1,259 MB | | |
| real 2k | 14 | 242 | 7.94 M | 8.23 M | 2,462 MB | 1,350 MB | 95 MB | 9.8 MB |
| real 20k | 1 | 2,919 | 0.58 M | 0.58 M | 1,234 MB | 1,213 MB | | |
| real 20k | 14 | 2,919 | 4.59 M | 4.49 M (noise) | 2,634 MB | 1,533 MB | 108 MB | 24.6 MB |

After the single-thread work (interleaved dynamics, dense phase A, interleaved vertices), same protocol, back to back
with the pre-memory-work binary (load 7-11): **14 x real 2k 7.80 -> 9.05 M agent-steps/s (+16 %), 14 x real 20k
4.58 -> 5.20 M (+14 %)**; per-env memory 95 -> 9.8 MB (2k) and 108 -> 24.6 MB (20k).

Scaling (real 2k): 1 / 2 / 4 / 8 / 14 envs = 1.0 / 2.0 / 3.8 / 7.8 / 10.2x: linear up to 8 envs, then the 4 efficiency
cores. Real 20k reaches 7.7x at 14 envs (larger per-env working set). Per-env memory = (RSS(14) - RSS(1)) / 13; the
shared world + population is ~1.2 GB. Live heap of one more env (examples/env_mem): 90 -> 5.3 MB at creation.

Single-thread profile (bench, no BAYENV_PAR): real 2k 3,367 steps/s (215 cars): integrate 45 %, spawn 22 %, leaders
17 %, contacts 6 %; real 20k 187 steps/s (1,917 cars): integrate 48 %, leaders 26 %, contacts 6 %, controls 6 %.
Top self time (sampling): physics::step_dynamics (10 substeps of 0.01 s per 0.1 s step; atan x2, tanh x2, tan, cos /
sin memoised per substep, all V8-exact fdlibm) 23 % at 2k, 11 % at 20k; then fdlibm atan / atan2 / sin / cos /
rem_pio2, spatial-hash queries, lead_base, lane geometry (offset_point, y_along, lane_width), router searches
(reroutes / refuel plans, ~5 %).

| single-env / VecEnv change | result |
|---|---|
| LaneIndex per-key arrays in per-edge blocks (key layout and order unchanged) | -25.6 MB per env |
| junction moves_into map once per world (World::mv_into) | -24 MB per env |
| router: one search graph per world (World::nav_base) + search state per worker thread (ThreadNavRouter) | -36 MB per env |
| interleaved vehicle dynamics in single-threaded phase A: 2 or 4 consecutive cars' substeps alternate (step_dynamics2 / 4; each car's ops unchanged) | real 2k single thread ~3,900 -> 4,300 (2-wide) / 4,350 (4-wide) steps/s (+11-12 %); VecEnv 14 x real 2k 6.6 -> 7.5 / 7.6 M agent-steps/s (+15 %), 14 x real 20k 4.27 -> 4.61 / 4.62 M (+8 %) (load 6-16, lock) |
| phase-A results dense in slot order in a reused buffer (no high-sized Option vec per step; pose as 5 fields) | real 2k single thread 4,520 -> 4,790 steps/s (+6 %), real 20k 228 -> 240 (+5 %); real 100k parallel within noise |
| contacts: candidates j <= i dropped inside the spatial-hash walk; LaneIndex rebuild without its counting pass | no measurable change (kept: same results, less work) |
| rejected: jsmath::sincos (one argument reduction for sin + cos, bit-identical) in step_dynamics | -4 % (4,770 -> 4,585 steps/s at real 2k); hr / steer are mostly below pi/4 (no reduction to share) |
| rejected: cache of avoid / penalty reroute queries per (closed set, penalty) epoch | 512 queries in 3,000 steps at real 20k, 0 repeats |
| lane geometry vertices interleaved ([x, y, z, mx, mz] per vertex; offset_point reads vertices i, i + 1 from one or two cache lines) | real 2k 4,765 -> 4,845 steps/s (+1.7 %), real 20k within noise |
| not pursued: contact queries reading a position snapshot in the spatial hash (contiguous reads) | the JS reads positions live (write_back moves cars during the pass); a snapshot needs exact moved-since-rebuild tracking |
| measured for stage 2 (vector fdlibm): scalar throughput per call atan 2.7 ns, tanh 4.4 ns, tan 3.0 ns, sin 1.6 ns, cos 2.0 ns (latency 9.5 / 22.8 / 11.5 / 7.3 / 9.5 ns); per car-step ~180 ns of transcendentals in a ~400 ns interleaved step_dynamics | a perfect 2-lane port would save at most ~90 ns per car-step: <= ~8 % end to end at real 2k, ~4 % at real 20k. **Stage 2 deferred, realistic ceiling ~5 %** (lane blending of fdlibm's range branches; the interleaved scalar code already keeps the 4 FP pipes busy; tanh = expm1 with argument reduction and 5 ranges would be first, a multi-day bit-exact port) |
| rejected: fixed-point skip in step_dynamics (reuse a substep when (u, v, r, steer, sl, ax) repeat bitwise) | only 0.8 % (2k) / 1.4 % (20k) of substeps repeat |
| rejected: periodic re-ordering of cars by edge / lane for locality | slot order is semantic (slot-order commits, contact pass order, event order) |
| not pursued: allocation-free step | 0.35 allocations per car-step (73 per step at 2k, 701 at 20k): < 1 % of step time |
| not pursued: SIMD fdlibm across cars | possible bit-exactly (lane-wise IEEE ops + V8's FMA pattern) but a large port for at most ~2x on the transcendental part (2 f64 lanes) |

## Optimizations (each re-checked with the differential traces; all are exact)

| change | effect |
|---|---|
| port as is (dense 2^18 spatial-hash CSR rebuilt every step, as JS) | real 2k: 1,150 steps/s (JS 950); junction-s1 1,581 (JS 4,070) |
| SpatialHash rebuild over touched cells only (same per-cell slot order) | real 2k 5,040, junction-s1 71,000 steps/s |
| FxHash for internal maps, ListMap for per-step slot lists | real 100k 22.4 -> 26.1 steps/s |
| per-phase profile (BAYENV_PROFILE=1) at real 100k, 1 thread | leaders 29 %, integrate 34 %, traffic re-routing 13 %, spawn 9 %, contacts 6 %, controls 5 % |
| integrate split into gather / pure physics / scatter, physics + pose in parallel (`SimOpts.parallel`) | integrate 14.1 -> 4.6 ms/step at 100k |
| leaders phase 1 (lane / zipper leaders, lane ends, off-lane bodies, scenes) and control obs in parallel | real 100k 35 -> 45 steps/s |
| leaders phase 2 speculative in parallel (reads of earlier cars recorded, re-run on mismatch), contact / spawn pre-scans with exact invalidation, batched reroutes / refuel plans on the shared router | real 100k 45 -> 117 steps/s (load ~6) |
| refuel plans prefetched 60 s ahead in one router batch, station candidate scoring in parallel (plans are pure in (from, goal)) | spawn-time plans 1.5 -> 1.1 ms/step |
| rejected: dropping the lane-key sort in LaneIndex::rebuild (range layout is observed: obs / v2 traces diverge) | - |
| no `Arc<World>` clones in hot paths (`Sim::gw()`) | VecEnv 14 threads: 2.0 M -> 9.3 M agent-steps/s |
| physics: reuse cos/sin(steer), cos/sin(headingRel) while bitwise unchanged; skip tanh when the tyre-force blend is 0 | ~5-8 % of the physics |
| `--profile fast` (fat LTO, panic=abort) + `-C target-cpu=native` | +6-30 % single env |

Observations (`obs.rs`, fill_obs + v3 + signals): Rust 0.5-1.7 us per agent vs JS env.observe 6-9 us.
Population generation (real n=10000): JS 10-18 s, Rust 6.3 s (1 thread), 0.7 s (14 threads).
Env construction without node (`Env::create_real`, n=2000): 0.7 s (JS createTrainingEnv ~8 s).

Not exact / not done: no approximate fast paths (every result is bit-identical to the JS); PGO not applied;
re-routing time is the router (bayenv_world::nav).

Reproduce:
```
node --max-old-space-size=14000 tools/rust/bench_js.mjs --world real --n 10000 --policy cruise --steps 1000 --warmup 600 --opts '{"startTime":25200}' --out /tmp/b/real-10000
cargo run --release -p bayenv-sim --bin bench -- /tmp/b/real-10000 1000 600            # BAYENV_PAR=1 for the parallel phases
cargo run --release -p bayenv-sim --bin bench_vec -- /tmp/b/real-2000 28 500 300 [threads]
RUSTFLAGS="-C target-cpu=native" cargo build --profile fast -p bayenv-sim
```
