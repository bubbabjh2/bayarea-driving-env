# Design decisions of the Rust port

The environment was first written in JavaScript: a CPU simulator (`src/sim`) and a WebGPU simulator (`src/gpu`).
This file records the decisions behind the Rust port in `rust/`, with the reasons, the alternatives that were tried
or considered, and the measurements behind them. Measurements are on an Apple M4 Max (10 performance + 4 efficiency
cores) unless stated, mostly on a machine shared with other jobs, so throughput figures are ranges or best-of runs
under the bench lock. Details live in `crates/bayenv-sim/BENCHMARKS.md`, `README.md`, `ARCHITECTURE.md` and
`docs/*.md`.

Each entry: date, decision, why, alternatives, numbers, and one line to say it on camera.

---

## 1. Port the simulator to Rust

- **Date:** 2026-11-20
- **Decision:** Re-implement the JS CPU sim and the host side of the WebGPU sim in Rust (crates `bayenv-world`,
  `bayenv-sim`, `bayenv-gpu`). The GPU kernels stay the same WGSL, run through `wgpu` (Metal on macOS).
- **Why:** Reinforcement learning needs millions of simulated car-steps per second, deterministic runs, and many
  environments in one process. The JS sim runs in one thread per process (Node or a browser) and the GPU sim needs a
  browser tab. Rust gives native speed, real threads with no garbage collector, and precise control of floating-point
  operations, which the bit-exactness goal (entry 3) depends on.
- **Alternatives:** keep JS and run many Node processes (each process loads its own copy of the world, about 1 GB of
  memory, and pays the single-thread cost); C++ (same speed, no memory safety and a harder build for Python users).
- **Numbers:** real world, 100,000 commuters: JS 7.4 steps/s; Rust 26.4 steps/s on one thread, 127 steps/s with the
  parallel phases. Real world, 2,000 commuters, one thread: JS 821 steps/s, Rust 4,920 steps/s.
- **On camera:** "We rewrote the simulator in Rust so it runs natively on every core, about seventeen times faster on
  the big map."

## 2. Python bindings with PyO3 and maturin

- **Date:** 2026-11-20
- **Decision:** Expose the environments to Python as the `bayenv` module (`crates/bayenv-py`), built with PyO3 and
  maturin, with numpy arrays in and out. One `Env` / `VecEnv` API covers both backends (`backend="cpu"` or `"gpu"`).
- **Why:** RL tooling is in Python. PyO3 releases the GIL while the Rust code steps, and maturin produces a normal
  wheel (`pip install`). Actions and observations cross the boundary as contiguous float arrays, no per-car Python
  objects.
- **Alternatives:** a socket or shared-memory server (an extra process and serialisation on every step); ctypes or
  cffi over a C ABI (hand-written glue, no safety on the Rust side).
- **Numbers:** Python `VecEnv(cpu)` with 14 envs x 2,000 commuters: `observe` about 0.05 ms per call; the
  act + step + observe round trip adds about 0.1-0.2 ms on top of the simulation (1.4-2.0 ms per step for all 14 envs).
- **On camera:** "Python calls straight into Rust; the data never gets copied into Python objects."

## 3. The JS simulator is the oracle: bit-exact, not "close enough"

- **Date:** 2026-11-20
- **Decision:** The Rust CPU sim must reproduce the JS sim bit for bit: every per-car field after every step, every
  observation float, every population, route and event. JS `Float32Array` fields are stored as f32 and widened on
  read, JS `Map` iteration order is kept where it is observable, signed zeros and NaN handling follow the JS, and the
  operation order of every expression is the JS one.
- **Why:** It makes the port testable without judgement calls: any difference is a bug, found at the first step and
  field where it appears, instead of a slow statistical drift. It also means behaviour learned on one implementation
  transfers to the other.
- **How it is checked:** `tools/rust/trace.mjs` records a JS run (state of every car every step, observations, actions);
  `trace_check` replays it in Rust and compares every field. `v2_check` does the same for the 70-float observation
  encoder, `tests/population_gen.rs` for generated populations, `jsmath_oracle` for the maths library.
  `rust/tools/make_oracle.sh` regenerates every dump from the JS; with `BAYENV_REQUIRE_ORACLE=1` a missing dump fails
  the tests instead of skipping them.
- **Alternatives:** statistical agreement (mean speeds, crash counts); rejected because it hides real bugs and gives
  no place to start debugging.
- **Numbers:** 61 of 62 scenario x policy runs of the differential matrix bit-exact, plus every saved real-world
  trace (2,000 / 10,000 / 100,000 commuters) and all population runs, serial and parallel. (The 62nd, a random policy
  on the urban scenario with observations, never finished on the JS side.) Every optimisation in this file was
  re-checked against these.
- **On camera:** "The Rust version doesn't just behave like the JavaScript one. It produces the exact same numbers,
  down to the last bit, and we test that on every change."

## 4. Port V8's maths library, including where the CPU fuses multiply and add

- **Date:** 2026-11-20
- **Decision:** `bayenv_world::jsmath` is a statement-by-statement port of V8's fdlibm (`ieee754.cc`): sin, cos,
  tan, atan, atan2, exp, log, tanh, expm1, pow and the rest. It uses a fused multiply-add exactly where clang
  contracts one in V8's arm64 build, and plain multiply then add everywhere else.
- **Why:** `Math.sin` and friends are not exactly specified by JavaScript, and Rust's `f64::sin` calls the platform
  library, which differs in the last bit. One wrong bit in a heading compounds into different trajectories within
  seconds of simulated time.
- **Alternatives:** the system libm (fails bit-exactness); a correctly rounded library such as CORE-MATH (also not
  what V8 computes).
- **Numbers:** 40 million inputs across 19 functions (sim-typical ranges, random bit patterns, special values,
  argument-reduction edges) bit-identical to node 22 on arm64.
- **On camera:** "Even sine and cosine had to be re-implemented exactly the way Chrome's engine computes them,
  including which multiplications the chip fuses."

## 5. Model fixes are flags: on by default, legacy JS behaviour kept for the oracle

- **Date:** 2026-11-20 / 2026-11-21
- **Decision:** When an investigation finds a bug in the JS model, the Rust fix goes behind a flag that defaults to
  on. The flag set to `false` reproduces the JS exactly, and every JS-oracle check runs that way (`SimOpts::js()`,
  `GpuOptions` legacy values). Current fixes:
  - `cpu_connector_join` (GPU): junction connector entry as in the CPU sim (entry 6).
  - `contact_tangent` (GPU): an f32-robust friction tangent in collisions (entry 7).
  - `replan_at_missed_event` (CPU): re-plan a missed turn at the moment it happens (entry 7).
  - `lane_index_snapshot` (CPU): lane-index walks read a car as indexed; fixes an infinite loop (entry 8).
- **Why:** Bit-exactness against the JS (entry 3) and fixing real bugs conflict. Flags keep both: the default
  environment is the corrected model, and the oracle tests still prove the rest of the port exact.
- **Alternatives:** fix the JS too (then all earlier results and the original browser demo change, and there is no
  longer an independent reference); leave the bugs in (the bugs make cars crash into kerbs for no reason).
- **On camera:** "When we found bugs in the original, we fixed them behind a switch, so we can still prove everything
  else matches the original exactly."

## 6. Finding: the GPU sim's cars drifted into kerbs (connector-entry bug)

- **Date:** 2026-11-20
- **Decision:** Make the GPU's junction connector entry use the start-of-step position, as the CPU sim does
  (`cpu_connector_join`, default on).
- **Why:** The WGSL kernel tested the end-of-step position, so above 6 m/s almost every junction entry took a
  re-projection branch meant only for back-to-back junctions. That branch adds a small heading error (up to about
  0.09 rad) that nothing in the dynamics removes, so a car with a straight steering wheel drifts sideways until it
  hits the kerb.
- **Numbers:** 5,000 commuters, 1,200 steps, every car driving straight with constant gas: GPU 1,394 contact rows
  (1,388 of them kerb strikes) against 17 on the CPU. With the fix the two backends agree on crashes and contacts
  (entry 7). Report: `docs/kerb-investigation.md`.
- **On camera:** "One inequality checked the wrong moment in time, and the GPU cars slowly steered themselves into
  the curb."

## 7. Finding: why some cars crashed on only one backend

- **Date:** 2026-11-20
- **Decision:** Two fixes, both flags (entry 5): `contact_tangent` on the GPU and `replan_at_missed_event` on the CPU.
- **Why:** After the kerb fix, 9 agents still crashed on the CPU but not on the GPU. Tracing the first three gave
  two causes, both deterministic:
  - A. The CPU re-planned a missed turn one step late. At the next junction the stale route made the car enter late,
    get re-projected with a 0.2-0.26 rad heading error, and drift into the kerb. The GPU re-plans at the event.
  - B. In a perfectly straight rear-end collision, an f32 subtraction on the GPU produced rounding noise that passed
    as a sideways direction, adding a full friction impulse that spun both wrecks to the kerb. The JS CPU (f64) sees
    no sideways motion.
- **Alternatives:** make the GPU copy the CPU's late re-plan (rejected: it reproduces a bug that causes crashes).
- **Numbers:** 5,000 commuters x 3,000 steps: crashes CPU 76 (6 CPU-only) -> 70 (0 CPU-only, all 70 also on the GPU,
  which has 71); arrivals agree to within 0.00 s of lateness (was 1.75 s). Report: `docs/crash-divergence.md`.
- **On camera:** "The two simulators disagreed about who crashes; each disagreement turned out to be a small bug, and
  after fixing them they agree crash for crash."

## 8. Finding: an infinite loop the JS has too (lane-index snapshot)

- **Date:** 2026-11-21
- **Decision:** `lane_index_snapshot` (default on): the lane-index queries `ahead` / `behind` read a car's edge and
  lanes as they were when the index was built. In addition every lane walk stops after a bounded number of steps.
- **Why:** The observation is computed after the cars have moved but uses the lane index built before. A car that
  had just crossed onto a new edge looked itself up under the new edge, found its own entry, and the junction-yield
  scan asked "who is behind me?" forever. The JS code is the same and would hang on the same state.
- **Numbers:** found by a 28-environment run that hung in one environment at step 460; with the fix it completes.
  The fix changes no simulation state (all traces still bit-exact) and 20 of 3,632 observation rows of the real-map
  reference run.
- **On camera:** "One car on the edge between two roads asked who was behind it, got itself as the answer, and kept
  asking forever. That was in the original code too."

## 9. GPU runs are deterministic by default

- **Date:** 2026-11-20
- **Decision:** `deterministic_contacts`, `deterministic_spawn`, `deterministic_events` and `deterministic_leaders`
  are on by default in the GPU sim: collision priorities come from the pair of cars, lane claims for spawning go
  through bid / commit passes, events are sorted per step, perception reads snapshots. Refuel planning no longer has
  a wall-clock budget by default.
- **Why:** With the original kernels, two runs with the same seed diverged: atomics resolve in whatever order the GPU
  threads arrive. RL experiments need to be repeatable.
- **Numbers:** without the fixes a 10,000-car run first differs at step 333 (one car's speed, last bit) and a
  20,000-commuter run within 20 steps; with them, same-seed runs are bit-identical. Cost: about +8 % GPU time at
  100,000 cars. Report: `docs/determinism.md`.
- **On camera:** "On a graphics card thousands of threads race each other, so we made every race end the same way
  each time."

## 10. Memory-mapped world files

- **Date:** 2026-11-20
- **Decision:** The road network and derived data are stored as binary packs (64-byte aligned sections, xxh3
  checksum) that are memory-mapped and used in place.
- **Why:** The world is large (lane geometry, junctions, a 344 MB GPU world blob). Parsing JSON took seconds, and
  every environment or process would hold its own copy. With mmap the operating system loads pages on demand and
  shares them between processes.
- **Numbers:** `World::load` about 0.06-0.08 s warm; converting it for the CPU sim 0.13 s.
- **On camera:** "The map loads in under a tenth of a second because we never parse it; the file is the data."

## 11. Struct-of-arrays car storage, and a pure physics phase

- **Date:** 2026-11-20
- **Decision:** Cars are stored as one array per field (`VehicleStore`: speed[], lat[], heading[], ...), slots in a
  fixed order. Each step's vehicle physics is a pure function of one car's own fields ("phase A"); the results are
  written back in slot order. Perception that depends on earlier cars runs speculatively in parallel and is re-run
  for a car whose inputs changed.
- **Why:** The JS sim updates cars one after another, and later cars see earlier cars' updates. Running that in
  parallel naively changes results. Splitting out the part that only reads its own car, and committing in the JS
  order, gives the same results on any number of threads.
- **Numbers:** real world, 100,000 commuters: 26.4 steps/s on one thread -> 117-127 steps/s with the parallel phases
  on 14 threads, still bit-exact.
- **On camera:** "Each car's physics only looks at that car, so we can do all of them at once and still apply the
  results in the original order."

## 12. Four cars' physics interleaved on one core

- **Date:** 2026-11-20
- **Decision:** On a single thread the vehicle dynamics of four consecutive cars run interleaved: substep 1 of car A,
  B, C, D, then substep 2 of each, and so on. Each car's arithmetic is unchanged.
- **Why:** Each car's physics is a long chain of dependent operations (10 substeps per step, each with atan, tanh
  and trig calls), so one car leaves most of the core idle waiting for results. Four independent chains keep the
  core busy.
- **Alternatives:** two cars (about the same gain); vectorised maths (entry 16, rejected).
- **Numbers:** per car-step 520 ns -> 400 ns; real 2k single thread about +12 %; 14 envs x 2k about +15 %.
- **On camera:** "One car's physics keeps the processor waiting, so we do four cars at once on the same core."

## 13. Memory per environment: share what the world determines

- **Date:** 2026-11-20
- **Decision:** Everything derived only from the map is built once per world and shared (junction movement tables,
  the router's search graph); the router's scratch state is per thread, not per environment; the lane index only
  allocates storage for roads that have had cars on them.
- **Why:** Many environments per process is how RL throughput scales on a CPU (entry 14), and each environment used
  to carry about 100 MB of copies of map-derived data.
- **Numbers:** memory per environment 95 -> 9.8 MB (2,000 commuters) and 108 -> 24.6 MB (20,000); 14 environments
  2.46 GB -> 1.35 GB in total, of which about 1.2 GB is the shared map.
- **On camera:** "Fourteen copies of the city went down to one copy and fourteen small sets of cars."

## 14. Many independent environments, and how many to run

- **Date:** 2026-11-20 / 2026-11-21
- **Decision:** `VecEnv` runs many independent single-threaded environments on the thread pool. Guidance: run at
  least twice as many environments as threads.
- **Why:** Parallelising inside one environment needs synchronisation many times per step; independent environments
  need none. But an RL loop steps every environment once per call and waits for the slowest one (an efficiency core,
  or a thread the OS paused), so extra environments let fast cores take more of the work.
- **Numbers:** free-running (each environment does all its steps in one task): 14 x 2k 9.2 M car-steps per second,
  28 x 2k 10.0 M, 14 x 20k 5.3 M. Stepped together once per call: 14 x 2k 5.5 M (-40 %), 28 x 2k 8.8 M (-12 %).
  One JS process: about 0.19 M.
- **On camera:** "Stepping fourteen cities in lockstep means always waiting for the slowest; with twenty-eight the
  fast cores just do more of them."

## 15. Population generation: exact, but fast

- **Date:** 2026-11-20
- **Decision:** Generated populations (homes, workplaces, routes, departure times) match the JS generator exactly.
  Workplace snapping uses a k-d tree that returns exactly the node the JS ring search finds (same distance formula,
  same tie-break), and all worker threads share one router graph.
- **Why:** Regional populations spent 85 % of their time in a nearest-node search that scanned the whole bay.
- **Numbers:** San Francisco, 20,000 commuters: JS 377 s; Rust 79 s -> 12 s on one thread, 1.2 s on 14 threads.
  0 mismatches on 320,000 nearest-node test queries; all 38 population checks exact.
- **On camera:** "Building twenty thousand commuters went from six minutes to about a second."

## 16. Rejected: vectorised maths (two-lane NEON fdlibm)

- **Date:** 2026-11-21
- **Decision:** Not used. Bit-exact two-lane NEON versions of atan, tanh and expm1 exist (`bayenv_world::jsmath_x2`,
  verified on 3 million pairs and the V8 oracle), but the simulator keeps the scalar versions.
- **Why:** In isolation the vector atan is 2.1x faster per value (tanh 1.1x). Inside the interleaved physics (entry
  12), which already keeps the core's floating-point units busy, it made things slower or no faster.
- **Numbers:** per car-step 400 ns -> 500 ns (atan and tanh vectorised), 425 ns (atan only), 397 ns (atan with an
  all-vector fast path). The estimated best case had been about 5 % end to end.
- **On camera:** "We built a faster sine for two numbers at a time, and it made the simulator slower, because the
  processor was already busy."

## 17. Rejected: balancing one environment's work across cores

- **Date:** 2026-11-21
- **Decision:** Not used. Each step was split into three pool passes (every environment's first half, the physics of
  all environments as one shared list of 64-car chunks, every environment's second half) so idle cores could help
  slow ones. Results stayed bit-exact.
- **Why rejected:** Each extra barrier waits for its own slowest environment, and the physics results had to move
  between passes.
- **Numbers:** 14 x 2k: 7.8-9.0 M -> 2.0-3.4 M car-steps per second (2-3x slower).
- **On camera:** "We tried letting fast cores help slow ones mid-step; the coordination cost three times more than it
  saved."

## 18. Rejected: other ideas, with their numbers

- **Date:** 2026-11-21
- **Pooled observation encoding** (encode all environments' observations on the shared pool after all have stepped):
  about 8 % slower (1.46 -> 1.60 ms per step for 14 environments).
- **Core pinning** (keep environments on performance cores): macOS has no thread-to-core affinity, only quality-of-
  service classes, and 14 busy threads on 10 performance cores put 4 on efficiency cores regardless. Entry 14 is the
  workaround.
- **Shared sine / cosine argument reduction** (`sincos`, bit-identical): 4 % slower; the angles are mostly small
  enough that there is no reduction to share.
- **Skipping repeated physics substeps** (when a car's state repeats exactly): only 0.8-1.4 % of substeps repeat.
- **Re-ordering cars in memory by road** for cache locality: the slot order is part of the model (the order cars are
  committed and collide in), so results would change.
- **Dropping the lane-key sort** in the lane index: the observation and encoder checks diverged; the layout is
  observable.
- **Caching re-routes** around blocked roads: 512 queries in 3,000 steps of a 20,000-commuter run, none repeated.
- **Allocation-free steps:** 0.35 allocations per car-step, under 1 % of the step time.
- **On camera:** "Most ideas didn't survive measurement, and we kept the numbers to prove it."

## 19. Exactness bugs found by the optimisations themselves

- **Date:** 2026-11-21
- **Decision:** When the parallel or interleaved physics ran ahead of the JS order, a step that might label a car as
  "stuck" falls back to the original order.
- **Why:** The label looks at the cars ahead, which the fast path had already moved; the JS looks at them before
  they move. Only a statistics counter could differ, and only after a car had waited 30 minutes, so none of the saved
  traces hit it. A trace made with a 15-second stuck timeout (5 stuck labels) confirmed the fix in serial, batched
  and parallel modes.
- **On camera:** "Making it faster almost changed a statistic in a case that happens once in thirty minutes, and the
  exact-match tests are what caught it."

## 20. Tests say when they could not check against the JS

- **Date:** 2026-11-21
- **Decision:** The oracle dumps are not in the repository (several GB). Every test that needs them prints
  `SKIPPED <test>: <reason>` and passes, or fails when `BAYENV_REQUIRE_ORACLE=1`. `rust/tools/make_oracle.sh`
  regenerates all of them from the JS with node.
- **Why:** On a fresh clone the equivalence tests used to pass silently without checking anything.
- **On camera:** "A test that can't run now says so out loud, instead of quietly passing."

## 21. Physics model fixes from the audit (one flag each, default on)

- **Date:** 2026-11-21
- **Decision:** The findings of `docs/physics-audit.md` that change outcomes are fixed in both backends. Each fix has
  its own flag: a `SimOpts` field on the CPU, a `GpuOptions` field (a kernel text patch) on the GPU, and a keyword of
  `bayenv.Env`. All are on by default, except `curvature_dynamics` (entry 23). `SimOpts::js()` and the legacy GPU flags turn them off, which gives the JS
  models bit for bit, and every JS-oracle check runs that way. Entries 22–30 cover the fixes one by one;
  `tests/physics_fixes.rs` checks each fix against the JS model and against the physics it restores.
- **Why:** The audit compared the sim with real vehicles and found loopholes an RL agent can exploit (free cornering,
  immovable wrecks, contacts that never count twice). Flags keep the JS model reproducible and let a training run
  switch over in one step.
- **Alternatives:** change the JS too (then the oracle has to be regenerated, and the old runs could no longer be
  reproduced); one master flag (it hides which fix caused a change in a training curve).
- **On camera:** "We audited our physics against real cars, then fixed every bug we found, each behind its own
  switch."
- **Driver check:** with every car driven by a heuristic driver (IDM car following, steering on the heading
  error; 2,000 commuters, 3,000 steps of 0.1 s), crashed vehicles were 4 with the JS model and 7 with these defaults, and 52 cars arrived in both runs.

## 22. Wrecks have mass (`wreck_real_mass`)

- **Date:** 2026-11-21
- **Decision:** In pair contacts a stopped wreck or parked obstacle uses its real mass and moment of inertia. Before,
  the JS gave it infinite mass. Barriers stay immovable.
- **Numbers:** sedan at 50 km/h into a parked sedan: the striker's Δv goes from 49 km/h to 30 km/h (the
  equal-mass value with restitution 0.2 is 30), the wreck moves off at 29 km/h, and momentum along the road is
  conserved within 2 %. A bus at 49 km/h into a parked sedan: the JS stopped it dead; now it carries on at 43 km/h
  and the sedan is thrown at 53 km/h.
- **On camera:** "Every parked car in our city was secretly an immovable wall. Now a bus hitting a sedan pushes it
  out of the way, like it would in real life."

## 23. The road no longer turns the car (`curvature_dynamics`, default off for now)

- **Date:** 2026-11-21
- **Decision:** (Implemented on both backends, **default off** until the lane geometry is smoothed; see
  Consequence.) The car state stays in road coordinates (s, lat, heading relative to the lane), but over a step its
  world heading changes by its own yaw only. After the step, the relative heading is recomputed from the frame at the
  new position: `heading_rel = wrap(frame − (heading_before − yaw))`. This covers curved edges, edge-to-edge hops and
  junction connectors, so the heading jumps at the 2,973 kinked joins (audit finding 8) go away too.
- **Why:** In the JS, the Frenet frame absorbed road curvature: a car with the wheel straight followed every curve.
- **Numbers:** 948 cars held at 108 km/h with steer 0 on empty roads: median peak world lateral acceleration 14.6 g
  → 0. On the 90-degree corner of the `road3` test road at 108 km/h with steer 0, the JS took the turn at a peak of
  95 g; now the car goes straight on. A heading-and-offset tracker takes a 1.7 rad junction turn at 30 km/h with a
  peak of 0.96 g. Rollout scan: 78 heading jumps → 0.
- **Consequence:** drivers must now steer through curves before they reach them. The road polylines have short
  kinks (for example 0.7 rad within 2.5 m), so a driver that only reacts to heading error hits the kerb. The
  heuristic driver used for imitation learning is one such driver. With every car on it (2,000 commuters, 3,000
  steps), crashed vehicles went from 4 (JS model) and 7 (all other fixes) to 706 of 720, and arrivals from 52 to 1.
  The typical crash: a car at 5 m/s meets a lane polyline that turns 0.7 rad within 2.5 m (a radius of about 3.5 m,
  not a real road). It reacts after its reaction time, by which point it has already hit the kerb. Adding a yaw-rate
  feed-forward from the lane-heading change to that driver did not help (698 crashed). So the flag ships off.
  Next: smooth the lane and connector centrelines to realistic minimum radii by road class (done, entry 33), add a
  lane-heading preview to the observation, then turn the flag on once a look-ahead driver drives at JS-level crash
  and arrival rates.
- **Alternatives:** smooth the polylines first (changes the world build and every position); bound the absorbed
  curvature by tyre grip (still free steering below the bound).
- **On camera:** "Our cars were taking freeway-speed right turns at 14 g. The road was secretly turning them. When
  we made the tyres do the work, we found the roads had 3-metre hairpins hiding in them."

## 24. A sliding wreck stops at the end of its road (`wreck_end_stop`)

- **Date:** 2026-11-21
- **Decision:** A crashed car that slides into the end of its edge or connector (where it cannot hop on) now stops
  there with speed 0. Before, it froze in place but kept its speed, which decayed only by tyre friction.
- **Numbers:** a wreck sliding at 25 m/s, 15 m before an edge end: 46 steps pinned at the end with speed in the JS
  (19.5 → 9 m/s at a fixed position), 0 with the fix.
- **On camera:** "A wreck could hit the end of a road and keep 'moving' at 70 km/h without going anywhere."

## 25. Every impact counts, not just the first touch (`contact_episodes`)

- **Date:** 2026-11-21
- **Decision:** A contact episode lasts while a pair (or a car and a barrier) keeps touching. A step is an impact
  if it is the first touch, or if its own Δv is at least `touch_kmh` (1 km/h). Each impact adds its Δv to the
  episode total and reports an event carrying that total. So the disable threshold (10 km/h), the severity and the
  reward all see the true accumulated Δv. Steps with smaller impulses (a resting push) neither add nor report.
- **Why:** The JS reported only the first step of a contact. A car leaning on the kerb could take a 12 km/h strike
  with no event and keep driving.
- **Numbers:** a car leaning on the barrier, then struck at 6 m/s lateral: no event in the JS on the strike step;
  with the fix the strike reports, and the episode total (10.5 km/h) disables the car.
- **Alternatives:** an event on every contact step (floods the log during resting contact); per-step Δv only (a
  long hard scrape never reaches the disable threshold).
- **On camera:** "We found a loophole: if you were already touching the wall, hitting it harder didn't count."

## 26. Junction boxes have walls (`connector_barrier`)

- **Date:** 2026-11-21
- **Decision:** A car on a junction connector meets a wall at its in-edge's kerb lines. The limits are carried in
  the in-edge lateral coordinate that the connector keeps, and the wall normal comes from the connector's tangent at
  the car. The wall uses the same impulse, events and position correction as the edge barrier. It is on by default
  already. It matters most with `curvature_dynamics`: once the frame stops turning the car, a car that misses a turn
  would otherwise wander through the junction box.
- **Numbers:** rollout scan (3,000 commuters, 1,200 steps): teleports of more than 1.5 × speed × dt + 0.5 m go from
  39 (worst 16 m, a car 15 m off its connector snapped back at the exit) to 0.
- **On camera:** "Inside an intersection a car could wander anywhere, then teleport back onto the road."

## 27. Brakes hold a stopped car (`brake_hold`)

- **Date:** 2026-11-21
- **Decision:** Below 0.05 m/s, when the brake force plus the rolling resistance is at least the drive force minus
  the grade force, the car is held at exactly 0 (static friction).
- **Why:** In the JS, below 0.02 m/s the brakes only cancelled the drive, and rolling resistance switched off below
  0.05 m/s. A car could creep under a held brake for ever.
- **Numbers:** on the brake from 0.015 m/s for 60 s: 0.9 m crept in the JS, 0 with the fix. The audit found creep
  in 169–261 of 400 stops per vehicle type, worst 1.2 m per minute.
- **On camera:** "Our parked cars were slowly creeping forward with the brakes on, about a metre a minute."

## 28. Truck and bus brakes calibrated (`heavy_brakes`)

- **Date:** 2026-11-21
- **Decision:** Trucks and buses (mass over 5 t) brake at 4.7 m/s² instead of 6.5 m/s².
- **Numbers:** 100–0 km/h: 59 m → 80 m. Loaded heavy vehicles need about 80 m (FMVSS 121). Sedans and SUVs are
  unchanged (100–0 in 41 m).
- **On camera:** "Our trucks stopped like sports cars. Now a loaded truck needs eighty metres from 100 km/h."

## 29. Barrier contact pushes, it doesn't teleport (`barrier_soft_push`)

- **Date:** 2026-11-21
- **Decision:** The barrier's position correction is at most 0.25 m per step. The JS removed the full penetration
  depth in one step. The wall impulse is unchanged.
- **Numbers:** a car 1.2 m into the barrier: lateral jump in one step 2.2 m in the JS, 0.25 m with the fix.
- **On camera:** "Touching a wall used to teleport you up to two metres sideways."

## 30. Knocked backwards stays backwards (`impact_reverse`)

- **Date:** 2026-11-21
- **Decision:** A crashed car may move backwards after an impact: its speed can go negative, and it slides on locked
  wheels until friction stops it. A driving car still has no reverse gear, so it keeps a backward speed only when the
  same impact disables it. Every "is this wreck still moving" test uses |speed|. A sliding wreck's body-frame
  rotation terms (v·r, −u·r) are applied as an exact rotation. Explicit Euler grew |velocity| by √(1 + (r·h)²) per
  sub-step, so a wreck spinning at 15 rad/s gained speed up to 85 m/s. The JS's clamp of u at ≥ 0 had hidden this.
- **Numbers:** a bus at 50 km/h into a stationary sedan facing it: the sedan slides back at up to 14 m/s with the
  fix; the JS clamped its speed to 0, so it stopped dead on impact.
- **On camera:** "A sedan hit head-on by a bus used to stop dead on the spot. Now it gets shoved backwards."

## 31. Input checks: NaN actions and dt

- **Date:** 2026-11-21
- **Decision:** On the GPU host, NaN action components become 0 before upload, as on the CPU. WGSL `clamp(NaN)` is
  undefined, and Metal returns the lower bound, so a NaN steer became full left lock. `Sim::new`, `GpuSim::new` and
  `bayenv.Env` reject a dt that is not positive and finite.
- **On camera:** "One bad number from the AI used to yank the wheel full left on the GPU."

## 32. Known simplifications we keep

- **Date:** 2026-11-21
- **Decision:** These audit findings are documented but not changed:
  - Trucks do not roll over. Lateral acceleration is limited only by tyre grip, while real trucks tip at about
    0.3–0.4 g.
  - Junction boxes are flat on hills.
  - Kerbs are rigid walls. Real 15 cm kerbs are mountable.
  - Cars have no reverse control: the action space has no reverse gear, so a boxed-in car stays boxed in.
- **Why:** Each one needs either new world data or a change to the action space. None of them let an agent exploit
  the sim the way the fixed findings did.
- **On camera:** "There are still things our sim gets wrong, and we're telling you which ones."

## 33. Smoothed frame geometry for `curvature_dynamics` (`smooth_geometry`, default off)

- **Date:** 2026-11-21
- **Decision:** A second, smoothed copy of the geometry the dynamics follow is derived deterministically at load time
  (`bayenv_world::smooth`, `LoadOptions::smooth_geometry` / `World::smooth_geometry`). It replaces the lane-geometry
  reference polylines (`LaneGeometry` vx / vy / vz / vs / mx / mz) and the junction connector points
  (`JunctionSet.pts`, with `mInfo` firstPt / ptCount). Everything reads them from there: on the CPU `lat_point`,
  `edge_point_dir`, bounds and barriers, `map_lat`, `connector_point`, the connector conflict paths; on the GPU the
  world blob, which is packed from the smoothed world. So positions and frames always agree. The flag is
  `SimOpts::smooth_geometry` / `GpuOptions::smooth_geometry` / `Env(smooth_geometry=True)`. It must match the world
  (`Sim::new` panics, `GpuSim` returns an error) and it needs `curvature_dynamics`. Default off, and `js()` turns it
  off.
  - **Fillets:** every vertex becomes a circular arc tangent to its two segments, sampled every 0.04 rad on edges
    (the lane frame heading is constant along a segment) and 0.1 rad on connectors (`connector_point` blends it).
    Neighbouring fillets split a shared segment max-min fairly. Where the segments are too short, the largest radius
    that fits is used. Runs of short kinks merge into one fillet at the intersection of the outer segments when that
    gives a larger radius. Merging runs in symmetric rounds, so a polyline and its reverse smooth alike.
  - **Joins:** consecutive edges at a pass-through node (no junction box, no other edge bar the twins; 19,763 joins
    in 14,895 chains) are smoothed as one polyline. Both edge ends move to the same arc point and take its normal.
  - **Connectors:** each is re-fitted to the smoothed lanes. Its ends are the smoothed lane points at s_in / s_out,
    their tangents the smoothed lane headings (a 0.5 m lead segment when they are off), and the interior is filleted.
  - **s is preserved:** kept vertices keep their s exactly, and an arc spans the s-range of the polyline it replaces.
    So lengths, s_exit / s_stop, gore points, controls, connector s_in / s_out / lengths / conflict positions and the
    hop maps are unchanged. Inside a fillet the world speed is (arc / replaced polyline) times the s-rate, at most
    about 0.8 on a 90-degree corner, the same kind of distortion the JS lanes already have off the reference line.
- **Radius table** (the target is for the innermost lane centre; the reference line gets the inner lane offset
  added): R = clamp(v² / (g (e + f)), floor, cap), with v the edge's speed limit, f the AASHTO side friction
  (0.35 at 20 km/h, 0.23 at 40, 0.17 at 60, 0.14 at 80, 0.12 at 100, 0.09 at 120) and per class:

  | class | e | floor | cap |
  |---|---|---|---|
  | motorway, trunk | 0.06 | 150 m | 900 m |
  | motorway_link, trunk_link | 0.06 | 30 m | 250 m |
  | primary | 0.04 | 20 m | 150 m |
  | primary_link | 0.04 | 15 m | 80 m |
  | secondary | 0.04 | 15 m | 120 m |
  | secondary_link | 0.04 | 12 m | 60 m |
  | tertiary | 0.02 | 12 m | 60 m |
  | tertiary_link | 0.02 | 12 m | 40 m |
  | unclassified, residential | 0 | 10 m | 15 m |
  | living_street | 0 | 10 m | 12 m |
  | junction connectors | | 12 m | 30 m (min of the in- and out-edge targets) |

  So a 65 mph motorway gets about 480 m, a 35 mph primary about 120 m, and residential streets 15 m. Two caps keep
  the lanes where the roads are. A fillet's apex may move at most 1 m from its vertex (`dev_max`), unless that would
  take the radius below the class floor. It never moves more than 5 m (`dev_hard`), so a hairpin keeps a tight
  radius rather than cutting off its corner. Turns over 2.6 rad (dead-end stubs that turn back) are not filleted.
- **Why (b), load time, and not (a), the world build:** (b) leaves the data files, the manifest, the JS sim and the
  JS oracle untouched. Every JS-oracle check keeps passing as it was, and with the flag off nothing changes, bit for
  bit. It is rebuildable by construction: under a second on the real world, multi-threaded, no cache. (a) would
  change every position in the JS sim and in earlier training runs, needs the 30–60 min oracle regeneration and a
  JS reader for the new data. It also makes "with and without smoothing" a different world rather than a switch.
- **Alternatives:**
  - Clothoid blending: continuous curvature, but the lane frame heading is piecewise constant per segment anyway, and
    an arc gives exact radius guarantees.
  - Laplacian or Chaikin smoothing: moves straight roads and guarantees no radius.
  - Moving the connector mouths (s_in / s_out) to lift junction-turn radii: changes junction data, stop lines and
    conflicts. Rejected, see Consequence.
- **Numbers** (`examples/geom_radius -- ../data [--smooth]`, real world):
  - **Geometry size:** lane vertices 1.93M → 4.59M, connector points 8.68M → 11.75M. 1,051,795 edge fillets, of
    which 34,290 (3.3 %) are fit-limited (largest radius that fits) and 1,383 are not filleted. 2,736,721 connector
    fillets, of which 720,604 (26 %) are fit-limited. Largest apex move on straight-ish roads (fillets turning
    < 0.1 rad): 0.63 m. Largest overall: 5.0 m.
  - **Share of path length tighter than R** (heading change over a 2 m window, every metre; innermost lane centre;
    raw → smoothed):

    | | < 5 m | < 10 m |
    |---|---|---|
    | residential lanes | 0.09 % → 0.02 % | 0.39 % → 0.22 % |
    | motorway_link lanes | 0.05 % → 0.01 % | 0.48 % → 0.04 % |
    | joins, residential | 78.7 % → 0.9 % | 82.8 % → 10.9 % |
    | joins, primary | 5.9 % → 0 | 13.1 % → 0.2 % |
    | connectors, residential | 0.75 % → 0.78 % | 64.9 % → 65.2 % |

    Under 15 m: residential lanes 2.2 % → 1.5 %, motorway 0.27 % → 0. Smoothing trades a metre of 3 m kink for tens of
    metres at the target radius, so the shares between 15 and 60 m grow (residential < 30 m: 3.8 % → 4.3 %).
  - **Tightest point per turning connector** (> 0.3 rad), under 10 m: residential 92.5 % → 92.9 %, primary 82.6 % →
    84.1 %. The vertex census of entry 23, now on the frame geometry: residential lane vertices under 10 m 16.1 % →
    25.2 %. This is not comparable, because the arcs add vertices, and a 15 m arc sampled per vertex counts as
    15 cos(0.02) m; the path-length census above is the measure.
  - **The kink of entry 23** (edge 212045, 1.7 rad within 5.4 m, about 3.2 m): it becomes an arc of about 6 m. It is
    fit-limited by the edge's 5 m dead-end start segment.
  - **Continuity** (`tests/smooth_world.rs`): 42,149 strict lane hops at nodes without a junction box. None has a
    larger lane-point jump than raw; smoothed joins jump at most 7e-7 m. Frame heading steps above 0.1 rad at those
    hops: 16,148 → 3,886 (the rest are merges and diverges, which are not pass-through). Connector entry / exit: at
    most 2.8 mm, and heading 0.094 / 0.056 rad (raw 0.157 / 0.083). s: every edge runs 0 .. length strictly
    increasing, and every connector 0 .. its length.
  - **Rollouts with curvature_dynamics** (`physics_audit sim ../data 3000 1200`, 504 cars spawned), crashed
    vehicles, raw geometry → smoothed:
    - keep policy (curvature off: 30): 419 → 419.
    - pure-pursuit look-ahead driver (`pursuit`, aims 2 m + 0.8 s plus the reaction delay ahead): 313 → 311.
    - Heading jumps over 0.3 rad: 0 in both. Teleports: 0 → 1 (keep, a wreck sliding backwards 7 m off a connector
      centre, 0.6 m) and 2 → 3 (pursuit).
    - S1 (108 km/h, steer 0): 0 g with and without.
  - **cpu_vs_gpu** (curvature on, smoothed): crashed 733 / 733, contacts 1,909 / 1,903, barrier 1,906 / 1,900,
    missed exits 13 / 15, violations 342 / 343. On raw geometry it was 751 / 751.
- **Consequence:** smoothing removes the lane and join kinks: no frame heading steps at joins, and the hairpins hidden
  in the lanes are gone. The crash counts barely move, though, because those kinks were not where these drivers
  crash. About half the crashes are on junction connectors and half in the first metres after one (keep: 208 / 211).
  They happen just after cars spawn at rest 1–5 m before a junction turn and act with their 0.5–1.2 s reaction delay.
  Junction-turn radii (4–9 m) are set by the junction box, not by polyline kinks. For fixed lane points and headings,
  no curve has a smaller maximum curvature than the shorter tangent leg allows (R ≤ leg / tan(turn / 2)). The JS
  builder already fits that arc, so connectors stay fit-limited, and 26 % of their fillets get less than the target.
  Larger turning radii would need the mouths (s_in / s_out) moved back: new junction data. The keep policy's
  ±0.15 rad steering clamp limits it to radii above about 18 m anyway. `curvature_dynamics` still ships off. Turning it
  on needs a driver that slows for junction turns and anticipates its reaction delay.
- **On camera:** "The roads had 3-metre hairpins hiding in them. We rounded every corner to the radius a real road of
  that class would have, and the cars still crashed, at the junctions. Turns that tight are real, and the drivers
  just weren't looking ahead."

## 34. Junction turns a vehicle can make (`junction_radii`), and spawn clearance (`spawn_clearance`), default off

- **Date:** 2026-11-21
- **Decision:** Two companion flags for `curvature_dynamics`, both default off. `SimOpts::js()` leaves them off, so
  every JS-oracle check is unaffected.
  - **`junction_radii`:** `LoadOptions` / `SimOpts` / `GpuOptions` / `Env(junction_radii=True)`; needs
    `smooth_geometry`, must match the world. At load, after the smoothing (`bayenv_world::junction_radii`), every
    turning connector (0.3–2.6 rad) whose arc is below the target is rebuilt as one circular arc with tangent entry
    and exit. Its s_in moves back on the in-edge and its s_out forward on the out-edge until the arc fits.
  - **`spawn_clearance`:** `SimOpts` / `GpuOptions` / `Env(spawn_clearance=True)` (both backends). A car whose first route edge
    leaves less than 18 m (`spawn_clearance_m`) between its spawn point (s_exit) and the junction entry its route
    takes next starts on the next route edge instead, as if it had driven the skipped ones (route position and
    distance set to match). It is its own flag because it changes where cars start on the raw world too.
- **Targets:** by the lower class of the two edges, never below 6 m where it fits.
  - Local streets (residential, unclassified, living_street): 8 m, real corner kerb-return paths of 6–9 m.
  - Arterials (primary to tertiary and their links): 12 m. That is also above a truck's or bus's 11.5 m full-lock
    radius (`examples/turn_feasibility.rs`: sedan 4.3 m, SUV 4.7 m, truck 11.3 m, bus 11.5 m).
  - Freeway system: 15 m.
  - Heavy vehicles get 12 m where they mostly drive. 12 m on every residential corner would move the entries
    another 4–10 m on short blocks and cut far past the kerbs, so local corners keep 8 m (trucks there take them with
    a wider swing, as in reality).
- **Limits on the move:**
  - At most 30 m.
  - Never past another movement's s_in on the same in-lane, or s_out on the same out-lane.
  - Never past the edge's other mouth (s_exit + 1 m / s_stop − 1 m) or its ends.
  - When a limit stops the move, the connector gets the largest radius that fits.
- **Kept consistent:**
  - Rebuilt from the arrays: the movement's s_in / s_out / length (`mF`), its points (s = true metres), and the
    conflict positions. These are recomputed on the new paths with the builder's own `pair_conflict` (754,474), or
    scaled by the length ratio where the paths no longer meet (12,499).
  - Everything that reads those arrays follows: lane index, leaders, yield, controls lookahead, CPU conflict paths,
    the GPU blob.
  - Unchanged: mouths and gores (so edge s_exit / s_stop, stop lines and controls), turn angles, the conflict list,
    hop maps.
  - s_in before the stop line is a state the JS builder already produces: 225,877 movements raw, 324,790 after. The
    sim uses min(s_stop, s_in) in its junction logic, and the controls keep measuring the in-edge s on the connector.
- **Why:** entry 33 showed the junction box bounds a connector's radius at min(tangent leg) / tan(turn / 2), so only
  moving its ends widens the turn. Doing it at load time keeps the data files, the JS and the oracle untouched.
- **Alternatives:**
  - Move the mouths (s_stop / s_exit) instead: shifts stop lines, the controls and every movement of the edge.
  - Compound curves or clothoids within the old ends: they cannot beat the shorter-leg bound.
- **Numbers** (real world, `examples/geom_radius -- ../data --junction-radii`):
  - **Movements:** 394,804 turning connectors.
    - 46,693 already at target.
    - 260,843 rebuilt at target.
    - 69,244 fit-limited: 47,152 by another movement on the same lane, 20,914 by the edge, 1,178 by the 30 m cap.
    - 18,024 not improvable.
    - Largest moves: 30 m back and 30 m forward.
  - **Arc radius (leg fit) per turning connector:** below 6 m 57,960 → 21,852. Below 11.5 m (heavy vehicles)
    365,124 → 280,653, mostly residential corners at their 8 m target.
  - **Tightest point per turning connector** (2 m heading window; < 6 m and < 11.5 m, before → after):

    | class | < 6 m | < 11.5 m |
    |---|---|---|
    | residential | 18.1 % → 3.6 % | 94.2 % → 84.4 % |
    | tertiary | 21.2 % → 8.0 % | 89.8 % → 36.4 % |
    | secondary | 24.6 % → 13.4 % | 87.0 % → 42.9 % |
    | primary | 29.4 % → 16.7 % | 88.0 % → 49.7 % |
    | motorway_link | 24.9 % → 14.9 % | 85.0 % → 38.1 % |

  - **Continuity** (`tests/smooth_world.rs`): 330,087 movements moved, with entry / exit on the lanes within 2.7 mm,
    tangent within 0.017 / 0.015 rad, s 0 .. length. Mouths, gores, turns and the conflict list are unchanged.
  - **Spawn clearance:** 73 of 504 spawns in the rollout start one or more edges further along.
  - **Rollouts** (`physics_audit sim ../data 3000 1200`, curvature on; crashed vehicles of 504):

    | | smoothed | + junction_radii | + spawn_clearance |
    |---|---|---|---|
    | keep policy | 419 | 420 | 415 |
    | pursuit | 311 | 295 | 287 |

    - Pursuit on raw geometry with curvature on: 317. Keep policy, smoothed + spawn_clearance only: 413.
    - Heading jumps: 0. Teleports: 1 (keep) and 6 (pursuit, edge hops after connector exits).
    - Crash sites (pursuit) are now mostly later and faster, on edges after a curve. The car overshoots the end of
      the curve after its 0.5–1.2 s reaction delay and crosses into the far kerb. That is a driver limitation, not
      geometry.
  - **cpu_vs_gpu** (curvature + smooth + junction_radii): crashed 734 / 734, contacts 2,023 / 2,015, barrier
    2,019 / 2,011, missed exits 13 / 15, violations 337 / 337.
- **Heavy vehicles on residential corners:** local corners stay at 8 m, below the truck and bus minimum turning radius
  of about 11.5 m. This is deliberate: real buses and trucks swing wide or crawl around tight corners. In the sim they
  may strike the kerb there.
- **GPU spawn clearance:** `GpuOptions::spawn_clearance` (default off; off = the JS kernels and host unchanged).
  - The host picks the start index while it writes the trip's spawn-queue entry (`host.rs spawn_start`, the CPU's
    `spawn_edge_index` on the same world arrays: f32 s_in / s_exit / s_stop, movements by in-edge in ascending order).
  - It stores the index in the entry's spare word 26 and the skipped route distance (f64 sum, as f32) in word 27.
    `kernels/spawn_clearance.wgsl` makes `spawn.wgsl` (and the `spawn_order.wgsl` claim / commit passes) spawn on that
    route edge at its s_exit, with that route position and distance; the entry lane is chosen against the next route
    edge, as on the CPU.
  - One difference: the CPU runs its synchronous refuel check first and keeps the car on edge 0 when a refuel plan
    replaced the route. GPU refuel plans are asynchronous (made later from the car's current edge), so a GPU car
    always starts on the clearance edge.
  - `cpu_vs_gpu` (curvature + smooth + junction_radii + spawn_clearance, with static_hold / idle_creep / steer_delay):
    all 834 spawns start on the same route position and edge, within 2 m in s; 127 start past route edge 0 on both.
    Crashed 745 / 745, contacts 1,759 / 1,771, missed exits 14 / 18, violations 375 / 376. `Env(backend="gpu", spawn_clearance=True)` works.
- **On camera:** "Our junction corners were tighter than a car can turn. So we moved where each turn starts, until a
  car could follow the curve and a truck could take the big roads."

## 35. A stopped car can pull away; automatics creep (`static_hold`, `idle_creep`)

- **Date:** 2026-11-21
- **Decision:** The first hold (`brake_hold`, entry 27) is replaced and now defaults off. It counted rolling resistance
  as holding force: a car with gas under about 3 % (2 % gas is 112 N of drive against 162 N of rolling resistance for
  a sedan), or with a light gas-plus-brake mix, stayed parked for ever.
  - `static_hold` holds a car below 0.05 m/s only while the brake force alone beats the drive force minus the grade
    force. Rolling resistance holds it only when there is no drive and no brake at all.
  - `idle_creep` gives the drive force a floor at an automatic's idle creep: 4 × the rolling resistance at rest,
    fading to 0 at 2 m/s. With no pedals on the flat a car settles at about 1.5 m/s, and about 7 % brake holds it.
  - Both are on by default on both backends; `SimOpts::js()` and `GpuOptions::js_model()` turn them off. GPU:
    `GpuOptions::static_hold` / `idle_creep` (`kernels/model_integrate.wgsl`, same sub-step as the CPU, so power and
    fuel see the creep force). `tests/pedals.rs` (300 scattered cars, medians): 2 % gas 20.8 m in 20 s, creep
    1.42 m/s, 10 % brake 0.000 m, JS model 0.00 m.
- **Why:** reported by the training side (ENV-REQUESTS #17). With 1,000 commuters, `brake_hold` alone cut arrivals
  from 81.9 % to 69.1 %; 226 of the 264 unfinished cars were still on the road at the end.
- **Numbers:** 20 s from rest with 2 % gas: 0.00 m with the first hold, 23 m now. No pedals: 1.48 m/s creep. 10 %
  brake against the creep: 1 mm. On the brake from 0.015 m/s: still 0 creep (entry 27's test).
- **On camera:** "Our parking brake was so good that cars couldn't drive away. A real automatic creeps forward the
  moment you lift off the brake, so now ours does too."

## 36. Steering has its own, short delay (`steer_delay`)

- **Date:** 2026-11-21
- **Decision:** Commands go through the control ring with each driver's reaction time (0.6–1.8 s). Steering is now
  read from the ring after `steer_delay` (0.15 s, rounded to steps, capped at the reaction delay). The pedals and
  hazard responses keep the full reaction time. A negative `steer_delay` (and `SimOpts::js()`) restores the JS.
- **Why:** following a path is closed-loop control with a neuromuscular lag of about 0.1–0.2 s, not a reaction to
  a surprise. A whole-second steering delay made every curve a crash once the road stopped turning the car
  (`curvature_dynamics`, entry 23). Reported by the training side (ENV-REQUESTS #16).
- **Numbers:** driver with a 1 s reaction time: the first steering response after 0.20 s (dt 0.05; JS 1.05 s), the
  first brake response still after 1.05 s.
- **GPU:** `GpuOptions::steer_delay` (seconds, default 0.15, negative = the JS): the integrate ring read takes the steer
  word round(steer_delay / dt) steps back when that is below the car's delay steps (`kernels/model_integrate.wgsl`).
  `tests/pedals.rs` (median over scattered cars): steer 0.20 s, brake 1.05 s; JS model steer 0.95 s.
- **On camera:** "Our drivers needed a full second to turn the wheel, as if every bend in the road was a surprise."

- **Observation delay compensation follows the steer delay** (bug found with a learned policy, 2026-11-22). obs v2
  inputs 4 / 5 / 41 (lane offset, heading error, steer angle) are predicted by replaying the steering commands not
  applied yet (adjust.js). Both backends replayed the full reaction window, which with `steer_delay` counted the
  already-applied commands twice; the CPU (`obs_v2.rs push_command`, `obs.rs sense_commands`) and the GPU
  (`kernels/model_sense.wgsl`) now replay only the last min(reaction, sds) commands, and v3 input 87 reads the steer at
  the steer delay. The GPU has one more step: its sense runs at the start of a step, before act writes the command
  uploaded for that step, and the action computed from the observation applies a step later; with `steer_delay` the
  replay therefore also appends that in-flight command (`actions[i]`, or the held ring entry), which the CPU encoder
  already has in its window. `steer_delay` −1 keeps both JS-exact. Closed loop through the obs (`tests/closed_loop.rs`,
  a lane keeper on inputs 4 / 5, 1,000 commuters, 150 s), crashed cpu / gpu before → after: 0 s 27 / 156 → 26 / 44,
  0.15 s 55 / 172 → 53 / 50, 0.3 s 23 / 124 → 22 / 32; barrier rows now 50 / 49, 107 / 117, 163 / 157.

## 37. The lane-heading preview on the GPU (`heading_preview`, default off)

- **Date:** 2026-11-21
- **Decision:** `GpuOptions::heading_preview` / `Env(heading_preview=True)` (GPU backend) adds
  `kernels/heading_preview.wgsl`, dispatched after `integrate` in every step. It writes a cap × 4 f32 buffer
  (`hpreview`). Row i is slot i: the road frame's world heading change (rad, wrapped to (−π, π]) between the car's
  position and 5, 10, 20, 40 m further along its planned path, as `bayenv_sim::obs::heading_preview`. Rows of slots
  that are not driving are 0. `GpuSim::heading_preview()` and `Env.heading_preview()` read it back; the GPU returns
  (capacity, 4) float64 and the CPU (n_slots, 4).
- **How:** the kernel walks the same pieces as the CPU, step for step. It follows the current edge or connector, then
  the route: the connector the car's lane takes into the next route edge (any lane as a fallback), or a plain hop.
  The route-position bookkeeping and the 16-hop limit are the same as on the CPU.
  - Connector heading: `connPointJ`. The connector points are f32 in the world data on both backends, so they match.
  - Edge heading: a per-vertex chord angle buffer (`hpdir`), computed on the host in f64 from the same lane-geometry
    vertices the CPU uses. The blob's f32 vertex coordinates (ulp 4–8 mm at 30–60 km from the origin) gave up to
    7e-3 rad of error on short segments; with `hpdir` the error is 2e-6 rad.
- **Layout:** four f32 per slot, distances fixed at `HEADING_PREVIEW_D` = [5, 10, 20, 40] m, kept simple so it can
  become part of the v2 / v3 observation once the feature list is fixed.
- **Numbers** (`tests/heading_preview.rs`: GPU population, then every driving car's edge, connector, s, lane, route
  position and route words copied into a CPU vehicle store and the CPU function evaluated): raw world, 495 cars, max
  |GPU − CPU| 1.3e-6 rad; smoothed + `junction_radii`, 267 cars, 1.4e-6 rad (all under 2.1e-6 in earlier runs).
- **Alternatives:** computing it on the host from a state readback (a cap-sized readback every step); folding it into
  `sense.wgsl` now (it would change the obs layout before the feature list is settled).
## 38. Kerbs can be mounted at walking pace (`mountable_kerbs`)

- **Date:** 2026-11-21
- **Decision:** Barrier contacts were rigid walls everywhere (audit finding 10). Now a contact on a road that is not a
  rigid barrier, with a closing speed under 2 m/s, is a kerb scrape:
  - no rebound, and kerb friction of 0.6;
  - the car may ride up to 0.3 m past the kerb line;
  - one minor contact event per touch, with a Δv capped below the disable threshold, and nothing added to the
    contact episode.
- **What stays rigid:** faster strikes, and every strike on a rigid barrier, behave as before. "Rigid" (`World::rigid_edge`)
  means the freeway system (motorway, trunk and their links, which have barriers or guard rails) and bridges (the
  graph's bridge flag). The world data has no barrier, wall or kerb tags, so road class and the bridge flag stand in
  for them; medians on arterials and walls on city streets are not distinguished. Default on, CPU and GPU;
  `SimOpts::js()` turns it off.
- **Why:** with `curvature_dynamics`, the look-ahead teacher's remaining crashes were almost all low-speed junction kerb
  touches: median 2.5–3.5 m/s with a 0.4–0.5 rad heading error, so about 1–1.5 m/s into the kerb. Each push into a
  rigid kerb is a new impact of 1–2 km/h. With `contact_episodes` the episode total adds them up, so a car steering
  into the kerb crossed the 10 km/h disable threshold after a few steps. These were CRASHED events with an
  accumulated Δv of just over 10 km/h; no single impact came near it.
- **Numbers:** a sedan at 3.5 m/s, 0.45 rad into a residential kerb, steering into it:
  - rigid: 7 contact events with an accumulated Δv rising 2.6 → 11.0 km/h, disabled at step 10;
  - mountable: one minor contact of 2.4 km/h, still driving;
  - JS model: one contact of 2.6 km/h.
- **GPU:** `GpuOptions::mountable_kerbs` (default on; `js_model()` off), in `kernels/model_contacts.wgsl` for edge and
  connector barriers alike. "Rigid" is the edge flags already in the blob: freeway system (bit 64) or bridge (bit 2).
  The scrape uses the same pre-impulse closing speed, restitution / friction, correction, carried episode total and
  capped first-touch Δv. `tests/kerbs.rs` runs one car on a real residential edge on both backends, and they agree
  exactly. At 0.45 rad, rigid gives 2 minor contacts and no disable on both (this edge's geometry stops the grind
  sooner than the road3 fixture); at 0.7 rad, rigid disables after 3 minor contacts on both. Mountable gives 1 minor
  contact, still driving, on both.
- **Follow-up (grinding):** the scrape test first used the body-centre velocity, and every kerb step that wasn't
  a scrape added its Δv to the episode. A car swinging in and out of a kerb could then sum steps of 1–2 km/h, each
  with a centre speed just over 2 m/s, past 10 km/h. Now the closing speed is taken at the contact point (velocity
  plus yaw rate × lever arm, as the impulse solver sees it). On a kerb, a step adds to the episode only when its own
  Δv is at least 2 m/s, and an event fires only on the first touch or on such a strike. In a rollout with a
  deliberately over-steering controller (1,000 commuters, 2,400 steps): 31 barrier crashes before, 3 of them sums of
  small steps; 30 after, none of them sums. All 30 are single impacts at more than 2.5 m/s, 5 of them on rigid roads.
  Contact events dropped from 145 to 58.
- **On camera:** "Touching a kerb at walking pace used to write the car off. Now it just scrapes, like it does in real
  life, unless you hit a freeway barrier."

## 39. Phantom walls: walls on the junction-box parts of edges (`box_barrier`), dropped lanes at diverges (`lane_drop_hop`)

- **Date:** 2026-11-21
- **Finding:** the video's "wall" wrecks in west SF (four cars at 127–147 km/h, delta-v about equal to the speed,
  in-lane on city streets, the chase cam under the terrain 1–2 s before) are not a geometry or elevation fault at
  the crash site. They are the end of an excursion off the road through a junction box. The JS has no lateral limit on
  junction connectors nor on the box parts of edges (s < sExit + 1, s > sStop). A car crossing a box at a large
  heading error (an untrained policy; or after a spin, which the JS's first-touch-only barrier events let a car survive
  without being disabled) drives sideways in the road frame without limit; on a connector with heading_rel past
  ±90° its `s` is pinned at the start and `lat` runs to hundreds or thousands of metres, while its y stays the road's
  (so the camera is under the hills). The first edge-barrier check past the box then meets it at that depth: the wall
  impulse at a near-perpendicular heading removes about the whole speed, and the JS's full-depth correction teleports
  the car back to the kerb line, in its lane. Reproduced in the Rust env (`SimOpts::js()`, 600 SF commuters, seed
  777, 07:00, a noisy constant-steer policy at full gas): 9–10 such contacts in 40 min, depths 24 m to 16 km, e.g. a
  primary at (−27195, −24149) inside the reported box: lat −726 m, delta-v 190 km/h at 195 km/h, the car 75–110 m
  below the 20 m ground grid for the last 20 s. The default model (connector walls since entry 26) still had them on
  the box parts of edges (a car 165 m off, delta-v 92 km/h at 100 km/h).
- **Decision:** `box_barrier` (default on, both backends; `SimOpts::js()` / `GpuOptions::js_model()` off): on the box
  parts of an edge, the edge barrier acts too, with its lines widened by `BOX_MARGIN` = 3.5 m for the corners and the
  cross street's mouth. Same impulse, kerb scrape, soft push and events as the edge barrier.
- **Second class (hops):** at a strict diverge where a lane continues onto no out-edge (129 (edge, lane, out-edge)
  hops map-wide, e.g. a 6-lane motorway into a 4-lane mainline + a 1-lane ramp), the JS keeps the planned edge and
  the strict lat: a car in the far-left lane routed to the right-hand ramp lands 16.6 m past the ramp's barrier line
  (a 57 km/h barrier crash in one rollout). `lane_drop_hop` (default on, both backends) sends it to the strict out-edge
  its lane overshoots least (a missed exit), about one lane out, where the barrier squeezes it in.
- **Numbers:** noisy-policy rollout (600 commuters in and around the box, 40 min, default model): box-part excursions
  past the kerb line by > 20 m 72 → 0, worst 162 m → 6.8 m, deep (> 2 m) high-delta-v contacts 1 → 0. Lane-keeping
  driver (2,000 commuters, 10 min): barrier crashes 56 → 57, arrivals 112 → 111 (no effect on normal driving).
- **Not changed:** the JS oracle (`SimOpts::js()` is bit-exact as before). The world data: road heights are within
  3.3 m of the ground grid everywhere in the box; the only larger misfits map-wide are tunnels and cuttings.
- **On camera:** "The invisible walls weren't in the city. Our cars were leaving the road inside intersections, where
  nothing stopped them, driving off into the hills, and the first wall they met afterwards was back on the road."

## 40. Cars roll back on hills (`hill_rollback`)

- **Date:** 2026-11-22
- **Decision:** a driving car's speed was clamped at ≥ 0 (there is no reverse gear), so a car that couldn't hold a
  grade simply stopped dead. Now, when the grade force beats drive + brake + rolling resistance, a driving car rolls
  backwards under gravity:
  - brakes and rolling resistance oppose the backward motion;
  - the brake (with `static_hold`) can still hold the car;
  - a backward-rolling car that the forces bring to a stop stays stopped, with no reverse drive;
  - a car rolling back into the one behind is an ordinary contact, and contacts keep a driving car's backward speed.

  Default on, CPU and GPU; `SimOpts::js()` turns it off.
- **Why:** the training side's GPU stranding trace. Cars held 0.44–0.55 gas on 20–23 % streets and sat at 0 m/s for
  hours: drive 0.546 × 5.6 kN = 3.06 kN against 3.17 kN of grade plus rolling resistance. That's a real stall, but
  with the clamp the policy never saw the consequence (lost ground), so it had no reason to press more gas. The
  project rule is "physics, not rules".
- **Numbers:** on trip 818's hill (edge 124811, 17–23 %), from rest:
  - gas 0.45 at s 20 (20 %): rolls back 3.4 m, peaking at 0.45 m/s, to where the grade eases. The JS stops dead.
  - gas 0.546: stalls at the balance point, held by rolling resistance.
  - gas 0.65: climbs.
  - stopped at s 25, gas 0: brake 0.6 holds it (3 mm); with no brake it rolls back to 6.3 m/s.
  - full gas from rest climbs the steepest sustained 30 m on the map, 35.0 % (edge 78271). SF's steepest streets
    (Filbert, 22nd St) are about 31 %, so the drive-force model (f0 = 5.6 kN for a sedan, about 0.38 g) is
    sufficient.
- **On camera:** "Our cars used to freeze on San Francisco's hills like they had a handbrake made of magic. Now if
  you don't give it enough gas, it rolls back down."
- **GPU:** `GpuOptions::hill_rollback` (default on; `js_model()` off), in `kernels/model_integrate.wgsl` (brakes forwards
  while rolling back and the backward hold at rest, rolling resistance against backward rolling, the rollback clamp, the
  static_hold window |u|, |un| <= 0.05, speed 0 at s < 0) and `kernels/model_contacts.wgsl` (write_back keeps a driving
  car's backward speed). `tests/hill.rs` runs trip 818's hill on both backends (one sedan, one step of reaction delay):
  gas 0.546 from s 14.72 stalls at s 26.19 on both (peak 27.48); gas 0.45 from s 20 rolls back to 16.65 (CPU) / 16.75
  (GPU), min speed −0.45 m/s on both (JS clamp: stops dead at 20.00 on both); gas 0.65 climbs to 168 m in 40 s; at
  s 25 brake 0.6 holds (25.00), no pedals rolls back to 7.22 on both.
