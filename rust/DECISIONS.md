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
