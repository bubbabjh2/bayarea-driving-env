# Rust port (`rust/`)

Native Rust implementation of the environment's host side. The WebGPU sim runs from Python or Rust without a browser:
the WGSL kernels of `src/gpu/kernels/` run unchanged through [wgpu](https://wgpu.rs) 30.0.2 (Metal on macOS,
Vulkan / DX12 elsewhere). The world loader and junction builder are ported bit-exactly. The JS implementation is the
oracle, and every port is checked against dumps of it (`tools/oracle/`). Layout and API: `ARCHITECTURE.md`. Timings:
`BENCHMARKS.md`.

| crate | contents |
|---|---|
| `bayenv-world` | Road packs (mmap'd), SimGraph, lane geometry, junctions + builder, traffic-control plans, surfaces, grade separations, stations, DEM, node index, A* router, route guidance, V8-exact `jsmath`, bake format |
| `bayenv-gpu` | wgpu host: world blob, buffers, kernels, `gpuSim.js` step loop (spawn / contacts / lists / junctions / sense / act / integrate), scatter population, events, crash scenes, closures and reroutes, live-traffic sampling, actions, observations, state readback |
| `bayenv-sim` | CPU sim (port of `src/sim`), in progress on branch `rust-cpu` |
| `bayenv-py` | Python module `bayenv` (PyO3 0.30 + maturin, abi3 ≥ 3.9) |

## Build

```
curl https://sh.rustup.rs -sSf | sh           # Rust (stable; tested with 1.100)
cd rust
cargo build --release                         # all crates
cargo run --release -p bayenv-world --bin build-junctions -- ../data --out /tmp/j   # Rust build:junctions
python3 -m venv .venv && .venv/bin/pip install maturin numpy gymnasium
cd crates/bayenv-py && ../../.venv/bin/maturin develop --release                      # installs `bayenv`
```

## Python

```python
import numpy as np, bayenv
env = bayenv.Env("data", n_cars=10000, seed=11)      # loads the world, builds the GPU sim, scatters 10k cars
A = np.zeros((env.n_slots, 3), np.float32); A[:, 1] = 0.3
env.act(A)                     # [steer, gas, brake] per car (slot), held until changed
env.step(20)                   # 20 steps (dt 0.05 s) in one GPU submit, GIL released
obs = env.observe()            # float32 (n_slots, 70): v2 observation layout (src/gpu/README.md)
st = env.state()               # dict of numpy arrays: edge, s, speed, lat, x, y, z, heading, state, crashed, disabled, ...
crashed = env.crash_flags()    # bool (n_slots,)
sig = env.signals()            # per stop line: edge, stop_at, control, type, state, time_in, time_left
env.stats(), env.time, env.raw("veh")
```

* Constructor options: `backend="gpu"`, `capacity` (default `n_cars` rounded up to 1024; results are identical for any
  capacity ≥ `n_cars`), `obs="v2" | "v3" | "none"`, `dt`, `start_time`, `accidents`, `collisions`, `fuel`, `controls`,
  `junctions`, `steer="position" | "rate"`, `decide_every`, `keep_events`.
* `step_observe(actions, k=1, out=arr)`: actions upload, k steps and the observation copy in one submit and one wait
  (the fused RL step). `step(k)` returns right after the submit; the readback and events are processed by the next
  call (same results, the Python side overlaps with the GPU). `sync()` waits explicitly.
* `bayenv.VecEnv(data_dir, n_envs, n_cars, seed)`: several independent envs on one device sharing one world (env i
  uses seed + i); `act((E, N, 3))`, `step(k)`, `step_observe(actions, k, out=(E, N, obs))`, `crash_flags() -> (E, N)`.
* `observe(out=arr)` fills a preallocated array (no allocation). Readbacks go through persistent staging buffers and copy
  only the rows in use.
* `backend="cpu"` raises `NotImplementedError` until `bayenv-sim` is merged.
* Gymnasium-style wrapper: `crates/bayenv-py/examples/gym_wrapper.py` (one observation and action row per car; the
  environment defines no reward). Benchmark: `crates/bayenv-py/examples/bench.py`.

## Equivalence with the JS implementation

| component | check | result |
|---|---|---|
| SimGraph, controls (plans + `stateOf` samples), stations, surfaces | `bayenv-world check` vs `tools/oracle/dump_world.mjs` | bit-identical |
| lane geometry (16 arrays), hop map | `tests/geom_exact.rs` | bit-identical |
| junctions (`build-junctions`, and `World::load` without `data/junctions.bin`) | vs `npm run build:junctions` | `junctions.bin` byte-identical; `controls.*` patch byte-identical |
| `Math.*` (19 functions) | `tests/jsmath_oracle.rs`, 40M inputs | bit-identical to node 22 on arm64 (V8 fdlibm with FMA contraction) |
| A* router, guidance codes, route words | `tests/nav_oracle.rs` (5,250 queries, 1.03M words) | bit-identical |
| GPU world blob (344 MB), params, kernel sources + defines, scatter state, dispatch lists | `crates/bayenv-gpu/tests/host_oracle.rs` vs the JS `GPUSim` (mock WebGPU run, `tools/oracle/gpu_mock.mjs`) | bit-identical |
| per-car state after N steps, JS GPU sim in Chrome vs wgpu | `tools/oracle/compare_gpu.sh` (10k cars, seed 11, gas 0.3, batches of 10) | see below |

After 200 steps (10 s): all 9,983 cars are on the same edge in the same state with the same flags; 68.5 % are
bit-identical in all 40 fields + pose; max |Δs| 0.029 m, |Δspeed| 0.032 m/s, |Δlat| 0.010 m, pose 0.030 m. Same
counts: 9,960 driving, 32 contacts, 14 crash scenes, 23 crashed vehicles.

After 1 step, 23 of 10,000 cars differ, all inside junction connectors, by about 1 ulp of world coordinates (≤ 1e-3 m).
The cause: Chrome compiles the kernels with Tint, wgpu with naga, and Metal fuses / reorders the two generated MSL
programs differently. Both use Metal's default fast math. Turning fast math off in wgpu made the match worse (40
bit-identical cars instead of 6,838). The differences then grow through contacts. After 1,000 steps (50 s): 99.45 % of
cars have the same state and edge, 96.9 % are within 0.5 m and 0.1 m/s, and aggregates agree (7,565 vs 7,560 driving,
4,712 vs 4,710 contacts, 940 vs 941 crash scenes). JS CPU vs JS GPU reaches 92.5 % within 0.5 m after 1,000 steps
(`src/gpu/README.md`).

## Performance (Apple M4 Max, Metal; real world, constant gas 0.3, obs v2, bench.html protocol)

| | 10k cars | 100k cars |
|---|---|---|
| JS GPU sim, Chrome (`test/gpu/bench.html`, pipelined un-awaited submits, capacity 131072) | 1.49 ms/step | 4.62 ms/step |
| JS GPU sim, Chrome, awaited `runSteps(20)` (events processed every batch) | 1.65 ms/step | 4.97 ms/step |
| Rust, capacity 131072, before the dispatch optimisation | 1.60 ms/step | 4.91 ms/step |
| Rust, capacity 131072 (indirect contact rounds) | 1.41 ms/step | 4.81 ms/step |
| Rust, capacity = cars (default) | 1.12 ms/step (892 steps/s) | 4.66 ms/step (215 steps/s) |
| Python `env.step(20)` | 1.11 ms/step (901 steps/s) | 4.63 ms/step (216 steps/s) |
| Python `act` + `step(1)` + `observe(out=)` every step | 1.63 ms/step | 5.78 ms/step |

The step is GPU-bound. Host time per batch is the encode plus one 260 KB io readback, and batch size barely matters (1.78
ms/step at batch 1 vs 1.58 at batch 20, 10k cars, capacity 131072). Startup takes 0.5-0.8 s: world load 60 ms, then
kernel compile, the world blob (159 ms) and the scatter.

## WGSL / backend patches

None. All 38 kernels (39 with obs v3) compile unchanged through naga. The only GPU code added is
`crates/bayenv-gpu/kernels/contact_args.wgsl`, a one-thread helper. It writes the indirect dispatch size of the
contact-resolve rounds from the contact count. Those kernels already return for `id >= count`, so results are
bit-identical, and the JS behaviour is available with `GpuOptions.indirect_resolve = false`.

## Not ported yet (GPU host)

* `setPopulation` / `setStreaming` (commuter populations, spawning from the trip schedule). Only `scatter` populations
  are available, as in `bench.html`.
* Refuel planning (`StationNetwork.plan`, pumps). As in `gpuSim.js` without `setStations`, every low-fuel request gets
  `FUEL_PLAN_FAILED`. This needs `src/sim/fuel.js` from `bayenv-sim`.
* `placeObstacles` / `placeWrecks`, `importSnapshot`, `resetDay`, the timestamp profiler, crash-detail records (type,
  maneuver, contact point) in the event log, `readAgents` / reward sums.
* Scatter routes carry no `slotRoute`, as in the JS, so missed-exit / closure reroutes are no-ops for scatter cars. The
  reroute path (A* + SET_ROUTE) is implemented for cars that have one.
* Vulkan / DX12 are untested (developed on Metal). Obs v3 kernels compile but were not compared with the JS.
