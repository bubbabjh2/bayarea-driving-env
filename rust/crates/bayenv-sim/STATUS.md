# bayenv-sim status (for a successor)

Rust port of the JS CPU sim (`src/sim`), bit-exact with it. Branch `rust-cpu`, merged into `rust` at checkpoints.

## What is ported (all verified bit-exact against the JS oracle)

| JS | Rust | notes |
|---|---|---|
| sim.js step (spawn, contacts, lanes, leaders, junctions, controls, drive, integrate, scenes, missed exits, traffic sample, re-routing, density) | `sim.rs`, `leaders.rs`, `junction.rs`, `contacts.rs`, `extra.rs` | every SIM_DEFAULTS option incl. pathSight, junctionSight, deadlockBreak, spawnLaneRoute, passWrecks, trafficMode empty/sparse, placeWrecks / placeObstacles / placeStaticObstacles / scenario obstacles |
| physics/ (dynamics, collide, params) | `physics.rs` | same expression order; Math via `jm` -> `bayenv_world::jsmath` (V8 fdlibm + arm64 FMA) |
| vehicle.js VehicleStore | `vehicle.rs` | SoA with the JS typed-array element types (f32 where Float32Array). `integrate` = gather / `integrate_car` / scatter |
| spatial.js | `spatial.rs` | SpatialHash: per-touched-cell counts (same per-cell order as the dense JS CSR) |
| surface.js, laneEnds.js, crashModel.js, rng.js, routeStore.js offsets | `surface.rs`, `leaders.rs`, `contacts.rs`, `rng.rs`, `fuel.rs` | |
| fuel.js StationNetwork + nav/stations.js routeViaStation | `fuel.rs` | `StationNetwork::plan(&World, &mut dyn Router, from, goal)` (stable API, used by the GPU host) |
| controls.js stateOf / movementOf + sim `_controlObs` | `extra.rs` | plans come from bayenv_world (TrafficControls) |
| obs.js, perceive.js, obsLayout.js, gradeObs.js, junctionObs / oncomingObs | `obs.rs` | flat rows: 77 scalars (+358 v3) per slot, layout in the module doc |
| population.js generatePopulation (+ popParallel), scenarios/load.js scenarioPopulation | `population_gen.rs` | rayon, thread-count independent |
| env.js TrainingEnv | `env.rs`, `vec_env.rs` | act / step(k) / reset; Python-shaped helpers (act_rows (N,3), observe_into, state_columns, crash_flags, stats), `Env::create_real` without node |
| route.js makeRouter | **not here**: `bayenv_world::nav::Router` (lead) via `route_stub::NavRouter`; `route_stub::AStar` is the old local port, kept for A/B (BAYENV_ROUTER=stub) |
| roadGraph.js / src/geom queries | `world.rs` (`World::from_bayenv` from `bayenv_world::World`, or `World::load` of a JS-exported world pack) | hop table checked against the JS for every (in, out) pair |

Not ported: events / tripLog (no effect on state), getStats string fields, the legacy `{throttle}` driver path, streaming (`opts.table`), SpatialHash.raycast / WorldQuery (brain-only).

## Equivalence testing

* `tools/rust/trace.mjs` runs a JS env (world: `synthetic`, `real`, `scenario:<name>`) under a deterministic policy
  (`idle|const|random|cruise|cruisenoise`, mirrored in `harness.rs Policy`) and records every VehicleStore field
  every k steps (`--obs 1` also every agent's observation). `cargo run --release -p bayenv-sim --bin trace_check -- <dir>`
  replays it in Rust and compares bit for bit (NaN == NaN).
* `bash tools/rust/matrix.sh <out> [steps] [every]`: all 17 scenario worlds x 3 policies, synthetic (full / sparse /
  empty / options), real world 2k (noise, sparse, all options, obstacles). Last run: 62/62 OK (see BENCHMARKS.md).
* `bin/world_check <data> <world.pack>`: bayenv_world-loaded world == JS-loaded world (every array).
* `BAYENV_PAR=1` runs the parallel phases; traces are bit-exact either way.
* Tests: `cargo test --release -p bayenv-sim` (JS test ports in `tests/js_port_*.rs`; trace / population / obs tests
  run when `BAYENV_OBS_TRACE` / `BAYENV_POP_DIR` point at fixtures).

## Determinism / parallelism rules

* No result may depend on HashMap iteration order (FxHash maps everywhere; JS Maps whose order matters are Vecs).
* Phases that run in parallel (`SimOpts.parallel`): integrate physics + pose (`integrate_all` phase A), leaders
  phase 1 (`lead_base`), control obs (`ctl_car`). Each reads only state that no other car changes during that phase
  and is written back in slot order. Everything else (contacts, junction perception, spawning, scenes) is order
  dependent and stays sequential.
* VecEnv: independent envs on the rayon pool, sharing `Arc<World>` / `Arc<Population>` read-only. Never clone the
  world Arc in hot paths (`Sim::gw()`): the atomic refcount made 14 threads slower than 4.

## API for other crates (stable)

* `Env` (env.rs): `new` / `create_real` / `reset`, `act(slot, Action)`, `act_rows` (N,3) / `act_batch(slots, n*3)`,
  `step(k)`, `step_events(k)`, `events()` (per-slot `sim::Event` log, kinds 1..8, sorted by (slot, kind) per step),
  `observe_batch(ObsKind::{V3, Flat, FlatV3}, out, slots)`, `observe_into` (rows = slots), `state_columns`,
  `progress_m`, `crash_flags`, `stats`, `outcomes`. `VecEnv` for many envs.
* `Sim::batch_policy`: batched external policy called once per step inside drive (JS batchDrive semantics;
  checked with `trace.mjs --hook 1`). Driver queries for external drivers: `yield_required`, `box_busy`,
  `rubberneck`, `speed_cap`, `lane_open`, `blocked_ahead`, `lane_gap_ok`; lane ends are in `vs.lane_end_*`.
* `fuel::StationNetwork::plan<G: PlanGraph>` and `population_gen::*<G: PopGraph>` work on `bayenv_world`'s SimGraph
  directly (no copy). `sim::Router` takes std HashSet / HashMap (bayenv-gpu implements it).
* `fuel::Plan.detour_time` = via-station time - direct time (routeViaStation detourTime).

## Open items / ideas

* Stationary-car fast path in `step_dynamics` (exact fixed point incl. signed zeros) — not done.
* Spawn: the pending queue is retried every step (JS behaviour); at 100k / 07:00 this is ~10 % of a step.
* Re-routing cost is the router (lead's crate).
* PGO not yet applied (`cargo build --profile fast`, RUSTFLAGS="-C target-cpu=native" measured in BENCHMARKS.md).
