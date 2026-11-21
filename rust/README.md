# Rust port (`rust/`)

Native Rust implementation of the environment's host side. The WebGPU sim runs from Python or Rust without a browser:
the WGSL kernels of `src/gpu/kernels/` run unchanged through [wgpu](https://wgpu.rs) 30.0.2 (Metal on macOS,
Vulkan / DX12 elsewhere). The world loader and junction builder are ported bit-exactly. The JS implementation is the
oracle, and every port is checked against dumps of it (`tools/oracle/`). Layout and API: `ARCHITECTURE.md`. Timings:
`BENCHMARKS.md`.

| crate | contents |
|---|---|
| `bayenv-world` | Road packs (mmap'd), SimGraph, lane geometry, junctions + builder, traffic-control plans, surfaces, grade separations, stations, DEM, node index, A* router, route guidance, V8-exact `jsmath`, bake format |
| `bayenv-gpu` | wgpu host: world blob, buffers, kernels, `gpuSim.js` step loop (spawn / contacts / lists / junctions / sense / act / integrate), scatter population, commuter populations (`setPopulation` / `setStreaming`), refuel plans and pumps, obstacles / wrecks, snapshot import, `resetDay`, events, crash scenes, closures and reroutes, live-traffic sampling, actions, observations, state readback |
| `bayenv-sim` | CPU sim (port of `src/sim`) |
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

## Data verification

`World::load` checks every input (roads, controls, stations, surfaces, crossings, crosswalks) against the sha256 values
in `tools/osm/world-manifest.json`, which is embedded at compile time; hashes are cached per size and mtime.
- A missing unpacked `controls.json` is restored from the committed `.json.gz`.
- A missing `data/junctions.bin` is built from the verified inputs. The result must equal the manifest's
  junctions.bin byte for byte, and it is cached in `data/build/rust/junctions.bake`, keyed by the input hashes.
- A missing, stale or modified file is an error naming the fix (`git checkout -- data/<file>`, `npm run unpack`,
  delete a stale junctions.bin). The loader never silently builds a different world.
- `LoadOptions { verify: false, .. }` is for custom data dirs.
- Test: `tests/fresh_clone.rs` copies only the git-tracked data into a temp dir and must reach the same
  `World::content_hash()` as the full checkout. A tampered input must fail.

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

pop = bayenv.Env("data", population=20000, pop_seed=42, start_time=7 * 3600.0)   # commuters on their trip schedule
pop.step(20); pop.slot_trip()  # slot -> 2 * commuter + leg (-1 unused, -2 freed, -3 obstacle)
pop.outcomes(), pop.agents()   # per commuter outcome; readAgents (reward sums, outcomes, trip time, crashes) + fuel_left
pop.place_obstacles(count=40, seed=3); pop.place_wrecks(10); pop.reset_day()
```

* Populations (gpu): `population=N, pop_seed=42` replaces the scatter with N commuters of `src/sim/population.js`
  (generated natively, bit-identical with the JS, 10x faster than popParallel.js) spawning on their trip schedule, outbound
  and return legs (`return_trips=True`), with refuel detours through the world's gas stations (`stations=True`) and
  station pumps. `streaming=True` uses a CommuterTable whose routes are computed at departure (`GPUSim.setStreaming`), prefetched by `route_threads` workers.
  `n_slots` = `capacity` (default min(N, 131072) rounded up to 1024): slots are recycled, `slot_trip()` maps them.
  `refuel_budget_ms` (default none = every plan in the frame it is requested, deterministic; JS default 4 ms).
  Also `reset_day(start_time=None)`, `place_obstacles(count, kinds, seed, slow_kmh, brake_every)` or
  `place_obstacles(obstacles=[{edge, at, lane | lat, type}])`, `place_wrecks(n)`, `outcomes()`, `agents()`,
  `import_snapshot(path)` (a CPU-sim `exportState` snapshot).
* Streaming route prefetch (`route_threads`, `GpuOptions.route_threads` / `route_ahead`): the `RouteWorkers` design of
  `src/gpu/streamPack.js`. Every frame, departures up to 300 sim s ahead are queued (batches of 16) on a rayon pool of
  `route_threads` workers (default: the machine's cores), one `nav::Router` each. A finished route waits in a per-OD map
  until its departures take it (the `RouteWorkers` done map; departures with the same OD inside the window share one
  A*). The JS `RouteProvider` OD LRU cache is `GpuOptions.route_cache` (entries) and is off by default: on the full
  morning only 0.2 % of departures repeat an OD, so a 200k-entry cache saves 0.15 % of A* runs for ~80 MB and more lock
  traffic. Reroutes don't use it (missed exits go through the host router's own cache; congestion reroutes depend on
  the penalty set; in a full-population peak window 0.2 % / 0.02 % of reroute ODs repeat). Prefetch pauses at 65,536
  outstanding ODs: a full slot pool keeps departures pending while the window moves on. A departure whose route is still being computed blocks the sim thread
  until it is ready instead of being delayed (the JS `waitingRoute` retry, which makes spawn steps depend on wall-clock
  time), and a departure that was not prefetched is routed on the sim thread. So every `route_threads` value, 0 (serial A*
  at departure, the previous behaviour) included, gives the same route and the same spawn step for every commuter
  (`stats()["spawn_hash"]` hashes the placed departures; `examples/stream_prefetch.rs` checks it). Timings:
  `BENCHMARKS.md`.
* Host work per frame: congestion / missed-exit reroutes (JS budget 200 per frame) and, without a refuel wall-clock
  budget, refuel plans run as parallel router batches and are applied in queue order (same ops and routes as one at a
  time; `BAYENV_REROUTE_CHECK=1` verifies reroutes). At the full-population morning peak the step went from 17.5 to
  5.2 ms (GPU 3.8 ms; `BENCHMARKS.md`).
* Constructor options: `backend="gpu"`, `capacity` (default `n_cars` rounded up to 1024; results are identical for any
  capacity ≥ `n_cars`), `obs="v2" | "v3" | "none"`, `dt`, `start_time`, `accidents`, `collisions`, `fuel`, `controls`,
  `junctions`, `steer="position" | "rate"`, `decide_every`, `keep_events`, `cpu_connector_join` (default `True`), `contact_tangent` (default `True`), `replan_at_missed_event` (CPU, default `True`).
* **GPU model fix (default since the kerb investigation):** `cpu_connector_join=True` makes junction connector entry
  match the CPU sim. `False` is the legacy mode, the JS GPU kernels bit-for-bit, including their connector-entry bug.
  **Legacy GPU runs and all earlier GPU checkpoints were trained with the bug:** steer-0 cars drift into the kerb after
  junction boxes (`docs/kerb-investigation.md`). `contact_tangent=True` (default since the crash-divergence
  investigation) gives the contact impulse an f32-robust friction tangent. With `False`, the JS kernels' tangent is
  rounding noise in an aligned rear-end, and the wrecks spin to the kerb (`docs/crash-divergence.md`). The JS-oracle checks (`host_oracle` test, `oracle_run`,
  `parity_run`, `parity_check`, `opt_check`, hence `compare_gpu.sh` / `compare_parity.sh`) run with both `false`.
* **CPU model fix (default since the crash-divergence investigation):** `replan_at_missed_event=True`
  (`SimOpts.replan_at_missed_event`, backend="cpu") re-plans a missed turn at the event, while the car is still on the
  in-edge, as the GPU does. `False` is the JS CPU sim's model: the re-plan comes one step late, the next junction box is
  entered late and re-projected (`headingRel += dh`), and steer-0 cars drift into the kerb (`docs/crash-divergence.md`
  cause A). **Earlier checkpoints trained on the JS CPU sim (and on bayenv-sim before this default) saw that behaviour.**
  The JS-oracle checks (`trace_check`, `v2_check`, the obs / events / population traces and the JS-port tests, all via
  `SimOpts::js()` / `harness::sim_opts_from`) run with it `false`.
* **CPU model fix (default since ENV-REQUESTS #15):** `SimOpts.lane_index_snapshot = true`: lane-index `ahead` /
  `behind` read a car as it was indexed (its edge and lanes at the index rebuild). The JS reads the car's current edge
  / lane_lo; in the observation (after integrate, on the index built before it) a car that crossed onto another edge
  can make `behind` return the car itself, and the junction-yield scan (`_yieldJ`) never returns: a hang the JS has
  too. With the fix those walks stay inside one lane range. The sim state is unchanged by it (it only reaches
  observations: 20 of 3,632 v2 rows of the real-2k oracle run differ); `SimOpts::js()` (all JS-oracle checks) has it
  `false`. In both modes the lane-index walks also stop after `high * MAX_OCC + 1` steps (`Sim::walk_limit`,
  counted in `junction::WALK_CYCLES`, state dumped to `$BAYENV_WALK_DUMP`), so no state can hang a run.
* `step_observe(actions, k=1, out=arr)`: actions upload, k steps and the observation copy in one submit and one wait
  (the fused RL step). `step(k)` returns right after the submit; the readback and events are processed by the next
  call (same results, the Python side overlaps with the GPU). `sync()` waits explicitly.
* `bayenv.VecEnv(data_dir, n_envs, n_cars, seed)`: several independent envs on one device sharing one world (env i
  uses seed + i); `act((E, N, 3))`, `step(k)`, `step_observe(actions, k, out=(E, N, obs))`, `observe()`,
  `step_async`, `observe_wait`, `crash_flags() -> (E, N)`. Every observation call takes the options below; all envs
  read back into one shared buffer, so `copy=False` is a single (E, N, obs) view.
* `events()`: per-slot event records since the last call, as a dict of equal-length numpy arrays (one row per event
  and car). Columns: `time`, `kind` (1 arrived, 2 crashed = this car disabled, 3 contact, 4 stranded, 5 cleared,
  6 spawned, 7 missed exit, 8 signal / stop violation), `slot`, `agent` (trip id), `other` (partner slot, -1 barrier),
  `dv_kmh`, `other_dv_kmh`, `share` (fault share from the crash kernel, with `crash_share=True`, 1/255 steps), `late_by`
  (arrivals with a population: arrival − workStart, s), `progress_m` (route metres), `trip_m` / `trip_s` (arrivals).
  Arrivals carry their exact arrival time; other events carry the end of the step batch. `VecEnv.events()` adds `env`.
  Shape type: `bayenv_world::events::EventLog`, shared with the CPU sim. `state()["progress_m"]` gives dense progress.
  The old scene / outcome log is `event_log()`.
* Observation readback (`observe`, `step_observe`, `step_async` / `observe_wait`; `crates/bayenv-gpu/src/obs.rs`):
  the observation copy goes into one of two mapped readback buffers (a ring), so the host reads step t while the GPU
  may already run step t+1 into the other one. Options:
  * `out=arr`: fill a preallocated C-contiguous array (no allocation). Default (`copy=True`, float32): a copy,
    bit-identical with the previous implementation (`crates/bayenv-py/examples/obs_check.py`).
  * `copy=False`: a read-only zero-copy numpy view of the mapped buffer (no host memcpy; 28 MB at 100k cars). Valid
    until your next `observe` / `step*` call on that env (the ring reuses its buffer after that), then its contents
    change (the memory itself stays alive: the view's `base` holds the buffer). `np.array(v)` it to keep it. Zero-copy
    needs unified memory (Metal, `env.zero_copy == True`); on other backends `copy=False` returns a copy.
  * `dtype="float16"`: converted on the GPU (`crates/bayenv-gpu/kernels/obs_f16.wgsl`, round to nearest even,
    saturating at ±65504; identical to numpy's `float32 -> float16`), returned as numpy float16: half the bytes. Every
    v2 field is in [-1.8, 1.8], so the error is ≤ 2^-12 per unit (max abs error 4.9e-4, mean 1.5e-5 over all fields;
    per field in rust/BENCHMARKS.md); values below 6e-8 flush to 0. Off by default.
  * `lag=1` (`step_observe`): pipelined stepping. Returns the observation as of the *start* of the call (after the
    previous call's steps, before this call's `actions` act) and leaves this call's k steps running on the GPU while
    Python computes the next action; the next call (or any call that needs the sim) waits for them. The obs returned
    by call t therefore does not yet reflect `actions_t`: the policy acts on a one-step-stale observation (the
    trajectory for a given action sequence is identical to lag=0, only what the policy sees shifts by one call). The
    first call returns the current observation.
  * `step_async(actions, k)` + `observe_wait()`: the same split by hand: submit and return; `observe_wait()` blocks
    and returns that batch's observations (no batch in flight: the current ones). The previous readback (and a
    `copy=False` view of it) stays intact while the batch runs.
  * `backend="cpu"` takes the same arguments: `copy=False` returns a copy, float16 is converted on the host, `lag=1`
    has the same staleness semantics (no overlap: the CPU steps synchronously), `step_async` steps at once.
  Python per-step cost at 100k cars (M4 Max, best of 3): `step_observe(out=)` 4.27 ms -> `copy=False` 3.89 ms,
  `copy=False, lag=1` 3.41 ms; with a policy reading the full observation every step 4.99 ms -> 3.86 ms (lag=1); a bare
  `step(1)` is 3.6-3.7 ms. Details in rust/BENCHMARKS.md.
* `backend="cpu"`: the CPU reference sim (`bayenv-sim`) behind the same calls and shapes: `act`, `step`,
  `step_observe`, `observe` (`obs="v2"` 70 floats via the v2 encoder, sensed after every step like the JS driver loop;
  `"v3"` 358; `"flat"` 77, CPU only), `state()`, `events()`, `crash_flags`, `signals`, `stats`. It uses a commuter
  population of `n_cars` (population.js, `seed`), and cars spawn on their trip schedule, so `n_slots` grows; rows are slots.
  `bayenv.VecEnv(..., backend="cpu")` steps its envs in parallel (rayon). It pads rows to the largest env, and `env_slots()`
  gives the per-env counts.
* CPU vs GPU events: `rust/tools/compare_backends.py` gives both backends the same population, start time and dt,
  uses k = 1 and constant actions, and matches rows by agent. Both backends report a barrier (kerb) scrape once, when
  it starts: the GPU's `barrier` kernel tracks F_WALL / F_WALL_PREV, and the CPU tracks `_barrierTouch`. With the
  default GPU model the aggregates agree: crashed 28 / 27, contacts 17 / 12, missed exits 120 / 123 and violations
  1,627 / 1,627 with `--spawn-default`, which gives spawning GPU cars the CPU's first-step default action.
  `crates/bayenv-gpu/tests/cpu_vs_gpu.rs` asserts this in `check.sh`. `--legacy-gpu` runs the JS GPU model, which
  has 1,394 contact rows from the kerb-strike bug. See `docs/cpu-vs-gpu.md` and `docs/kerb-investigation.md`.
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
| population (generator + `setPopulation`), spawn queue entries, trip schedule, params, slot tables, refuel plans (`StationNetwork.plan` + `SET_ROUTE_REFUEL` blocks), station / pump ops (`_enterStation`, `_pumps`), REENTER / ARRIVE (fuel to the return leg, outcomes), `resetDay` | `tools/oracle/compare_parity.sh` part 1: `gpu_parity_mock.mjs --scenario=pop` vs `examples/parity_check.rs` (20k commuters, seed 42, 07:00, 40 refuel plans) | bit-identical |
| `setStreaming`: CommuterTable (9 arrays), routes at departure, size-class route arena (incl. reuse after arrivals) | `--scenario=stream` | bit-identical |
| `placeObstacles` (spec, 40 of 8 kinds; explicit list), `placeWrecks`: veh / par / ctl records, scene blocks + edge heads, routes, scene records | `--scenario=obstacles` | bit-identical |
| `importSnapshot` (CPU sim `exportState`, 95 cars): veh / par / ring / pose / routes / clock | `--scenario=snapshot` | bit-identical |
| commuter population + 40 obstacles in Chrome vs wgpu | `compare_parity.sh` part 2 (`gpu_parity_run.html` vs `examples/parity_run.rs`, 20k commuters, gas 0.3) | see below |

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

Population runs (07:00, 3,217 departures queued in the first frame, gas 0.3): spawns, arrivals, trip pointer, slot
assignment (`slotTrip`), refuel plans, station lots and outcomes match the JS exactly. Per-car state matches like the
scatter runs: after 200 steps every car has the same state and edge and 99.8 % are within 0.5 m and 0.1 m/s; after
1,000 steps 99.5 % (99.2 % with 40 obstacles) have the same state and edge and 96.9 % (96.4 %) are within 0.5 m and
0.1 m/s; counters stay within 1 % (crash scenes 222 vs 224, refuel plans 110 vs 110).

## Verifying against the JS

The JS-equivalence tests compare against dumps of the JS implementation in `rust/oracle/`, which is gitignored (several
GB). In a fresh clone they cannot run: each prints a `SKIPPED <test>: <reason>` line (shown without `--nocapture`)
and passes, so a plain `cargo test` / `tools/check.sh` there checks the Rust code but **not** equivalence with the JS.
Generate the oracle from this checkout's JS sources, then run the tests with skipping turned into failure:

```
bash rust/tools/make_oracle.sh            # all parts, ~30-60 min; --quick for smaller runs; --only math,nav,...
cd rust && BAYENV_REQUIRE_ORACLE=1 cargo test --release --workspace --exclude bayenv-py
```

| part | generator | written to | tests |
|---|---|---|---|
| math | `tools/oracle/gen_math.mjs` | `oracle/math.bin` | bayenv-world `jsmath_oracle` |
| world | `tools/oracle/dump_world.mjs` | `oracle/{graph,controls,lg}.*` | bayenv-world `geom_exact`, `bayenv-world check` |
| junctions | `tools/build_junctions.mjs` (on a scratch copy of `data/`) | `oracle/ref/junctions.*` | `geom_exact::junctions` |
| nav | `tools/oracle/nav_dump.mjs` | `oracle/nav/nav.json` | bayenv-world `nav_oracle` |
| gpu | `tools/oracle/gpu_mock.mjs` (mock WebGPU, no GPU needed) | `oracle/gpu10k` | bayenv-gpu `host_oracle`, `pack_oracle` |
| traces | `tools/rust/trace.mjs --obs 1` (junction-s1, synthetic, real 2k) | `oracle/sim/traces` | bayenv-sim `obs_trace`, `events`, `world_hops` |
| v2 | `tools/rust/v2_oracle.mjs` (needs the bayarea-driving-sim checkout, `--brain`) | `oracle/sim/v2` | bayenv-sim `obs_v2_trace` (inputs, eyes.near, TeacherObs) |
| pop | `tools/rust/export_pop.mjs` (synthetic, scenario, real, region) | `oracle/sim/pop` | bayenv-sim `population_gen` |

The env vars `BAYENV_OBS_TRACE`, `BAYENV_V2_TRACE` (':'-separated dirs), `BAYENV_EVENTS_TRACE`, `BAYENV_POP_DIR`,
`BAYENV_TRACE_DIR` and `JSMATH_ORACLE` point the tests at other dumps. The JS-port tests (`tests/js_port_*`) build their
fixtures with node on first use and skip the same way without node. Larger differential runs:
`bash tools/rust/matrix.sh <outdir>` (every scenario x policy, checked with `target/release/trace_check`), and
`target/release/v2_check <dir>` for any `v2_oracle.mjs` run. Bit-exactness is against `SimOpts::js()` (the model fixes
off), which every oracle path uses.

## Performance (Apple M4 Max, Metal; real world, constant gas 0.3, obs v2, bench.html protocol)

| | 10k cars | 100k cars |
|---|---|---|
| JS GPU sim, Chrome (`test/gpu/bench.html`, pipelined un-awaited submits, capacity 131072) | 1.49 ms/step | 4.62 ms/step |
| JS GPU sim, Chrome, awaited `runSteps(20)` (events processed every batch) | 1.65 ms/step | 4.97 ms/step |
| Rust, capacity 131072, before the dispatch optimisation | 1.60 ms/step | 4.91 ms/step |
| Rust, capacity 131072 (indirect contact rounds) | 1.41 ms/step | 4.81 ms/step |
| Rust, capacity = cars, before the GPU-step optimisations | 1.12 ms/step (892 steps/s) | 4.66 ms/step (215 steps/s) |
| Rust, capacity = cars, GPU-step optimisations (default, bit-identical; see below) | 0.64 ms/step (1560 steps/s) | 3.50 ms/step (286 steps/s) |
| Python `env.step(20)` (measured before the GPU-step optimisations) | 1.11 ms/step (901 steps/s) | 4.63 ms/step (216 steps/s) |
| Python `act` + `step(1)` + `observe(out=)` every step (same) | 1.63 ms/step | 5.78 ms/step |

GPU time per stage (`bench --profile`, timestamps per dispatch, ms per step), before -> after:

| stage | 10k | 100k |
|---|---|---|
| contacts hash (clear, count, scan, scatter) | 0.058 -> 0.045 | 0.175 -> 0.22 |
| barrier + detect | 0.047 -> 0.039 | 0.26 -> 0.30 |
| resolve (contact rounds) | 0.30 -> 0.046 | 0.48 -> 0.19 |
| lists + jlists | 0.094 -> 0.084 | 0.32 -> 0.38 |
| leaders | 0.15 -> 0.13 | 1.00 -> 0.70 |
| junctions (perceive, courtesy, yield) | 0.20 -> 0.175 | 1.23 -> 0.99 |
| sense | 0.22 -> 0.090 | 0.80 -> 0.50 |
| act + integrate | 0.11 -> 0.069 | 0.54 -> 0.47 |
| whole step, one pass | 1.02 -> 0.63 | 4.47 -> 3.40 |

(Machine shared with other GPU jobs: ±10-20 % between runs; rows are typical, `rust/BENCHMARKS.md` has the steps.)
At 2,500 cars the step is 0.84 -> 0.46 ms. Small envs are latency-bound (each per-car kernel's longest thread), a dispatch
itself costs ~1.2 µs (30 extra empty dispatches: +0.035 ms), so the remaining fixed cost is mostly kernel latency.

The step is GPU-bound. Host time per batch is the encode plus one 260 KB io readback, and batch size barely matters (1.78
ms/step at batch 1 vs 1.58 at batch 20, 10k cars, capacity 131072). Startup takes 0.5-0.8 s: world load 60 ms, then
kernel compile, the world blob (159 ms) and the scatter.

## WGSL / backend patches

The JS kernels of `src/gpu/kernels/` stay the oracle and compile unchanged through naga. GPU code of this crate lives
in `crates/bayenv-gpu/kernels/` (each file's header says what it changes). Every option below is bit-identical to the
JS dispatch structure (`--example opt_check`: veh / par / pose / obs inputs / stats after 200 steps at 10k and 400
steps at 2k cars; `oracle_run` + `compare_state` against the unmodified host) and on by default; `GpuOptions` flags
switch each back to the JS behaviour (env `BAYENV_GPU_OPTS=flag=0,...` in the examples):

| flag | file | change |
|---|---|---|
| `indirect_resolve` | `contact_args.wgsl` | contact-resolve rounds dispatched by the contact count instead of contactCap |
| `fused_resolve` | `resolve_fused.wgsl` | the 12 rounds x 3 phases + serial tail in one workgroup (barriers instead of 37 dispatches), stopping once every contact is resolved; parallel order-keeping compaction |
| `sorted_order` | `sorted.wgsl`, `list_args.wgsl` | leaders / sense / junction.yield threads in lane-list order (entry renamed to a function, wrapper entry added by the host; indirect over lane entries + cap; every slot handled once) |
| `fast_scan` | `scan4k.wgsl` | lane / cell starts at 4096 per workgroup, counts into accumulators the scan zeroes (no clear kernels; 12 -> 6 dispatches) |
| `fast_search` | `patches.wgsl` | function patches applied to the resolved sources (`kernels::patch`, original text checked): binary instead of linear polyline segment search (`yAlongEdge`, `arcPointJ`, `connPointJ`; precondition: non-decreasing cumulative distances, checked on the world at creation), one-entry `hopMap` memo |
| `cpu_connector_join` (**on**; a model fix, so not bit-identical with the JS) | `cpu_connector_join.wgsl` | junction connector entry as in the CPU sim (`sArrive`, not the end-of-step `s`); fixes the GPU kerb strikes (`docs/kerb-investigation.md`). `false` = legacy JS GPU model, used by every JS-oracle check |
| `contact_tangent` (**on**; a model fix) | `contact_tangent.wgsl` | contact impulse friction tangent re-orthogonalised to the normal, slip below `1e-5 |rv|` = none; without it an aligned rear-end gets a full Coulomb impulse in an f32-noise direction (`docs/crash-divergence.md`). `false` = legacy JS GPU model, used by every JS-oracle check |
| `deterministic_contacts`, `deterministic_spawn`, `deterministic_events`, `deterministic_leaders` (**on**; same-seed runs bit-identical) | `contact_order.wgsl`, `spawn_order.wgsl`, host `canonical_event_order`, `leaders_order` / `courtesy_order` / `junction_order` / `jyield_order` / `hash_order.wgsl` | contact priorities from the car pair, lane claims in bid / commit passes, events in slot order per step, perception from snapshots and slot-ordered lists (`docs/determinism.md`; +8 % GPU time at 100 k cars). `false` = the JS kernels / host order (run-to-run differences), used by every JS-oracle check. `refuel_budget_ms` now defaults to `inf` (a finite budget is wall-clock dependent) |
| `zero_init_workgroup` (off) | - | no workgroup-memory zero fill (every kernel writes its workgroup variables first) |

`GpuOptions.profile` (`bench --profile`) records GPU timestamps per dispatch for the first step of every batch
(inside-pass timestamps when the adapter has them, else one pass per dispatch) plus the one-pass time of the second
step (`GpuSim.profiler`, `crate::profile`). Diagnostics: `BAYENV_SORTED=a,b` (sorted kernels), `BAYENV_WG=k=64`,
`BAYENV_NOPATCH=name`, `BAYENV_KDIR=dir` (replace a kernel source for experiments).

## Not ported yet (GPU host)

* Web-Worker paths of `gpuSim.js` for refuel plans / reroutes: the Rust host runs them on the main thread like the JS
  without workers (`fuelWorkers: 0`), which gives the same results. Streaming routes do use workers (below).
* Crash-detail records (type, maneuver, contact point) in the event log, the `crashPenLog`, `getStats()` as one object
  (the Python `stats()` exposes the counters).
* Scatter routes carry no `slotRoute`, as in the JS, so missed-exit / closure reroutes are no-ops for scatter cars. The
  reroute path (A* + SET_ROUTE) is implemented for cars that have one.
* Vulkan / DX12 are untested (developed on Metal). Obs v3 kernels compile but were not compared with the JS.
