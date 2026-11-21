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

## Optimizations (each re-checked with the differential traces; all are exact)

| change | effect |
|---|---|
| port as is (dense 2^18 spatial-hash CSR rebuilt every step, as JS) | real 2k: 1,150 steps/s (JS 950); junction-s1 1,581 (JS 4,070) |
| SpatialHash rebuild over touched cells only (same per-cell slot order) | real 2k 5,040, junction-s1 71,000 steps/s |
| FxHash for internal maps, ListMap for per-step slot lists | real 100k 22.4 -> 26.1 steps/s |
| per-phase profile (BAYENV_PROFILE=1) at real 100k, 1 thread | leaders 29 %, integrate 34 %, traffic re-routing 13 %, spawn 9 %, contacts 6 %, controls 5 % |
| integrate split into gather / pure physics / scatter, physics + pose in parallel (`SimOpts.parallel`) | integrate 14.1 -> 4.6 ms/step at 100k |
| leaders phase 1 (lane / zipper leaders, lane ends, off-lane bodies, scenes) and control obs in parallel | real 100k 35 -> 45 steps/s |
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
