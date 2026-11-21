# Benchmarks

JS vs Rust timings, before / after each optimization. Apple Silicon, node 22 (V8) vs `cargo --release`; real Bay Area world
(291,155 edges, 495,228 nodes). Each agent appends its own section.

## Router, route words, world blob (bayenv_world::nav, bayenv_gpu::pack)

Inputs: `node rust/tools/oracle/nav_dump.mjs data rust/oracle/nav` (3000 random OD pairs in the largest SCC, seed 5) and
`rust/oracle/gpu10k` (scatter n=10000, seed 11). Tests: `cargo test --release -p bayenv-world --test nav_oracle -- --nocapture`,
`cargo test --release -p bayenv-gpu --test pack_oracle -- --nocapture`. All outputs are bit-identical to the JS.

| operation | JS | Rust | notes |
|---|---|---|---|
| A* route, weight 1.25 (14.3k heap pops avg) | 2.19 ms | 1.09 ms | `makeRouter(g, {weight: 1.25, cacheMax: 0})` vs `Router::route_into` |
| A* route, weight 1 (27.3k pops) | 4.22 ms | 2.08 ms | |
| A* with avoid set (every 3rd route edge + 20 random) | 8.05 ms | 3.55 ms | stamped per-edge array, no Set |
| A* with penalty map (route edges + 50 random) | 6.00 ms | 2.66 ms | stamped dense per-edge array, no Map |
| packRoutes, one route (avg 129 edges) | 20.1 us | 0.84 us | per-hop maneuver table, O(n) pass |
| packRoutes, pool of 7987 routes (1.03M words) | 317 ms | 6.8 ms | |
| maneuver table (all 807,732 hops) | - | 3.6 ms | `RoutePacker::new`, once per world (rayon) |
| world blob (344 MB incl. hop table, extra conflicts) | - | 159 ms | `build_world_blob` |

Router optimizations (weight 1.25): first straight port with split arrays 1.54 ms -> interleaved node state / CSR-ordered
out-edges + single-division hypot (bit-identical to V8 `Math.hypot`, tested) 1.47 ms -> Morton-ordered internal node
numbering 1.09-1.12 ms (results are independent of the numbering: heap order depends only on keys and push order).

Router v2 (shared graph, batch, OD cache, exact fast A*; `cargo run --release -p bayenv-world --example router_bench -- ../data oracle/nav/nav.json 3`).
Before = the previous router (kept as `nav::router_ref`), after = `nav::Router`, alternating per query in one process, best
of 3 rounds. Taken under the bench lock on a heavily shared machine (load average 30-90 during the run), so the absolute numbers are
higher than the table above. The ratios are the reliable part.

| query kind (oracle queries) | before | after | speed-up |
|---|---|---|---|
| plain w 1.25 (3000) | 1.177 ms | 0.817 ms | 1.44x |
| plain w 1 (750) | 2.364 ms | 1.628 ms | 1.45x |
| avoid w 1.25 (750) | 4.167 ms | 2.780 ms | 1.50x |
| penalty w 1.25 (750) | 3.083 ms | 2.055 ms | 1.50x |
| `route_batch`, 4500 w 1.25 queries (plain + avoid + penalty), 14 threads | 502 q/s (1 thread) | 4312 q/s (740 q/s on 1 thread) | 8.6x vs before |
| OD cache hit (`route`, plain) | 0.9 ms (miss) | 0.12 us | |
| `route_batch` all cache hits | | 0.26 us wall per query | |

Gains come from these changes: a 4-ary heap with goal-bound push skipping and a tie check (an exact fallback to the JS heap
fired 21 times in 102k random queries), 16-byte node state (`gen << 1 | closed`), out-edges that carry the head node's
coordinates and out-edge span (heap entries carry the span, and the pushed node's out-edges are prefetched), and the
junction penalty from a per-rank table. A "no path" between strongly connected components that cannot reach each
other costs 0 pops, where it was a full search before (11k of the 102k random equality queries). The batch scales
by 5.8x on 14 threads under this load. Search state is 16 B per node per worker, and the pool holds at most one state per rayon thread.

## Lane geometry + junction builder (bayenv_world::lanes_build, junctions_build, `build-junctions`)

Full corridor (291,155 edges): `buildLaneGeometry` (opposing-carriageway + taper-limit passes) and `tools/build_junctions.mjs`
(crosswalk legs, 7 junction builds = 6 crossingFix rounds, cross-junction conflicts, controls stop-line patch). Peak memory =
macOS `/usr/bin/time -l` "peak memory footprint" (max RSS in brackets). Rust: 14 cores, rayon.

| step | JS (node 22) | Rust | notes |
|---|---|---|---|
| `buildLaneGeometry` | 3.2 s | 0.85-0.95 s | all 16 arrays bit-identical (`tests/geom_exact.rs`) |
| `build_junctions.mjs` end to end | 94.0 s, 4.16 GB (4.14 GB) | 6.3 s, 1.83 GB (2.8 GB) | `build-junctions data --out <dir>`; junctions.bin, controls.{bin,json,meta.json} byte-identical, junctions.meta.json equal except `buildMs` |
| `buildJunctions` alone (7 builds) | ~88 s | 5.1 s | per-connector / per-junction / per-edge work parallel, merged in JS order |

Checks: committed data (0 stop lines patched) and a perturbed controls copy (every approachStopAt +1.25 m, 1/7 of the
`stopAt` keys removed, `stopLinesFrom` removed: 64,241 patched), both JS vs Rust byte-identical.
Memory: per-junction connector points converted to f32 inside the parallel stage, the previous round's set dropped before
each rebuild, JSON trees dropped during the build (2.34 GB -> 1.83 GB peak).

## GPU host (bayenv_gpu) and world load

Real world, scatter n cars (seed 11), constant gas 0.3, obs v2; 100 warm-up steps, 400 timed steps in batches of 20
(`cargo run --release -p bayenv-gpu --example bench -- ../data 10000,100000 400 v2 20`; `BAYENV_CAP=auto` = capacity n).
Apple M4 Max, Metal. JS: `node test/gpu/run.mjs test/gpu/bench.html` (Chrome, same machine, same day).

| step | 10k cars | 100k cars | notes |
|---|---|---|---|
| JS bench.html (pipelined, capacity 131072) | 1.49 ms | 4.62 ms | GPU timestamps: 1.57 / 4.61 ms per step |
| JS awaited runSteps(20) | 1.65 ms | 4.97 ms | `rust/tools/oracle/gpu_run.html` |
| Rust first version (capacity 131072) | 1.60 ms | 4.91 ms | GPU-bound; host overhead ~0.2 ms/batch |
| + indirect contact-resolve rounds (36 dispatches sized by the contact count instead of contactCap) | 1.41 ms | 4.81 ms | bit-identical state (compared after 200 steps) |
| + capacity = n cars (Python default) | 1.12 ms | 4.66 ms | bit-identical to capacity 131072 for the first n slots |
| Python step(20) | 1.11 ms | 4.63 ms | GIL released during the submit |
| Python act + step(1) + observe(out=) | 1.63 ms | 5.78 ms | persistent staging buffers, only used rows copied |

World load (`World::load`, warm page cache): 60 ms (graph 43 ms, controls 8 ms, lane geometry from the mmap'd bake 2 ms).
Without `data/junctions.bin`: 6.5 s cold (lane geometry + junctions built in Rust and baked to `data/build/rust/`), then
60 ms; JS loader ~75-90 s.

## GPU step optimisations (bayenv_gpu host, rust-gpuopt)

Protocol as above (`BAYENV_CAP=auto ... --example bench -- <data> 10000,100000 400 v2 20`, capacity = cars), min of
3 runs (the machine is shared, ±10-20 % noise). "GPU" = timestamped GPU time of one step encoded as one pass
(`bench --profile`). Equivalence: `--example opt_check` (two sims side by side, 10k cars 200 steps and 2k cars
400 steps: veh / par / pose / obs inputs / ctl words and stats compared against the JS dispatch structure), and
`oracle_run` + `compare_state` against the unmodified host after 200 steps (rust/oracle/gpuopt-base.bin).
Flags: `GpuOptions` (`set_flag`, env `BAYENV_GPU_OPTS` in the examples).

| optimisation | 10k ms/step (GPU) | 100k ms/step (GPU) | equivalence | flag |
|---|---|---|---|---|
| baseline (origin/rust 9d2dc15, indirect resolve) | 1.10 (1.02) | 4.59 (4.47) | - | - |
| + profiler (`GpuOptions.profile`, `bench --profile`; off) | 1.10 | 4.59 | bit-identical (no change when off) | `profile`, off |
| + contact resolve in one dispatch (`kernels/resolve_fused.wgsl`: 12 rounds x 3 phases with workgroup barriers in one workgroup, parallel order-keeping compaction; 37 -> 1 dispatches) | 0.88 (0.83) | 4.40 (4.21) | bit-identical | `fused_resolve`, on |
| + lane-list thread order for leaders / sense / junction.yield (`kernels/sorted.wgsl`, indirect over lane entries + cap) | 0.88 (0.83) | 4.00 (3.81) | bit-identical | `sorted_order`, on |
| + scans at 4096 per workgroup with the count clears folded in (`kernels/scan4k.wgsl`; counts into accumulators zeroed by the scan; 12 -> 6 dispatches) + no workgroup-memory zero fill (`zero_init_workgroup` off: naga's fill of a 16 KB array cost 0.3 ms per workgroup; every kernel writes its workgroup variables first) | 0.85 (0.815) | 3.92 (3.70) | bit-identical | `fast_scan` on, `zero_init_workgroup` off |
| + binary polyline segment search (`kernels/patches.wgsl`: `yAlongEdge`, `arcPointJ`, `connPointJ` linear scans -> binary search; used by the grade obs, physics grade, connector poses; precondition non-decreasing cumulative distances, checked on the world at creation) | 0.68 (0.66) | 3.57 (3.41) | bit-identical | `fast_search`, on |
| + contact rounds stop once every contact is resolved (resolve_fused: resolve 0.058 -> 0.044 ms at 10k, 0.23 -> 0.13 ms at 100k) | 0.66 (0.63) | 3.55 (3.41) | bit-identical | part of `fused_resolve` |
| + one-entry `hopMap` memo (`patches.wgsl`; mapLane is called per lane with the same hop) | 0.64 (0.62) | 3.50 (3.40) | bit-identical | part of `fast_search` |
| final, same session: all off (JS structure) vs defaults, min of 3 | 1.12 -> 0.64 | 4.60 -> 3.50 | bit-identical | 2,500 cars: 0.84 -> 0.46 |

Tried and left out: workgroup sizes 32 / 64 / 256 for the sorted kernels (128 is best or equal); sorted order for
junction.perceive / integrate after the search patches (neutral); fusing more small dispatches (an empty dispatch
costs ~1.2 µs: 30 extra = +0.035 ms per step at 10k, so the 27 dispatches are ~5 % of the step).
Equivalence vs the JS GPU (Chrome, `rust/oracle/js/js_state.bin`, 10k cars, 200 steps) is unchanged: 6,838 cars
bit-identical, same edge / state / flags for all 9,983, max |Δs| 0.029 m, |Δspeed| 0.032 m/s, pose 0.030 m.

Contacts per step: ~6 at 10k, ~520 at 100k, none left to the serial pass. Run-to-run the unmodified host is
itself not bit-reproducible beyond ~200 steps (atomic orders: hash ranks, pair table, event order): after 1,000
steps two runs of the baseline differ in 93 of 8,766 cars, baseline vs fused resolve in 91, same aggregates
(7,559-7,561 driving, 940-941 crash scenes).
Sorted order on the other per-car kernels (junction.perceive, courtesy, act, integrate, contacts.barrier) is
bit-identical too but slower or neutral (the extra cap + lane-entry threads), so those entries are off by default
(`SORTED_KERNELS`, override `BAYENV_SORTED=...`).

## GPU host: populations, spawning, refuel plans (bayenv_gpu::sim::host)

20,000 commuters (seed 42) on the real world, capacity 32768, start 07:00. JS = `gpuSim.js` in node 22 against the
recording mock (`rust/tools/oracle/gpu_parity_mock.mjs`, same host code as in Chrome, no GPU work); Rust =
`examples/parity_check.rs` (14 cores where noted). All outputs bit-identical.

| operation | JS | Rust | notes |
|---|---|---|---|
| population generation, 20k commuters (40k routed legs) | 23.7 s (popParallel.js, worker threads) | 2.4 s (rayon, bayenv-sim generator) | |
| `setPopulation` (packRoutes of 3.21M route words, trip sort) | 1,039 ms | 47 ms | per-hop maneuver table |
| `_enqueueSpawns`, first frame (3,217 queue entries) | 20 ms | 10 ms | one queue write per entry |
| refuel plan (`StationNetwork.plan`: direct A* + 6 candidate stations x 2 A*) | 7.3 ms | 3.1 ms | 40 plans: 291 vs 123 ms |
| CommuterTable, 20k commuters | 5.9 s (worker threads) | 1.7 s (rayon) | |
| streaming enqueue, 3,217 departures (A* at departure + arena) | 4,365 ms | 2,239 ms | serial in both; the browser uses a route-worker pool |
| step with a population (3k cars driving, gas 0.3, batches of 10) | 2.48 ms/step (Chrome) | 1.43 ms/step | GPU-bound; host frame < 0.1 ms without refuel plans |

## Python per-step overhead and multi-env (after the kernel optimisations, ae5ab49+)

`rust/crates/bayenv-py/examples/bench_loop.py` (scatter, seed 11, gas 0.3, capacity = cars). The machine was shared with
other sessions (load average 12-21), so times are noisy (repeat-run spread ~±15 %); best of 3 below.

| call | 10k cars | 100k cars |
|---|---|---|
| Rust `run_steps(20)` per step | 0.66 ms | 3.58 ms |
| Rust `run_steps(1)` | 1.07 ms | 3.94 ms |
| Python `step(20)` per step | 0.75 ms | 3.96 ms |
| Python `act` + `step(1)` + `observe(out=)` (before `step_observe`: 1.63 / 5.78 ms with the old kernels) | 1.73 ms | 7.35 ms |
| Python `step_observe(A, 1, out=)` (one submit, one wait) | 1.05-1.20 ms | 7.0 ms |
| Python `VecEnv(4 x 10k).step_observe` | 3.78 ms per call = 1,059 env-steps/s, 10.6 M car-steps/s | |
| Python `VecEnv(16 x 2.5k).step_observe` | 10.3 ms per call = 1,548 env-steps/s | |

Per-call cost of a synchronous step at 10k cars: ~0.4 ms on top of the GPU time (host prep, submit, completion latency).
At 100k cars the 28 MB observation copy (GPU -> staging -> numpy) dominates `step_observe` (~3 ms); `step(k)` without
observations returns after the submit.

## Observation readback (rust-obs: zero-copy views, lag=1 / async, float16)

`crates/bayenv-py/examples/bench_loop.py` (scatter, seed 11, gas 0.3, capacity = cars, M4 Max). The observation
variants are timed round robin (blocks of 10 calls in turn, min over 30 blocks; the step cost drifts as traffic
evolves, so sequential timings of different variants are not comparable), best of 3 runs; "before" is the previous
build (c409813) run alternately with the same script. The machine was shared (load average 6-12).

| call | 10k before | 10k after | 100k before | 100k after |
|---|---|---|---|---|
| `act` + `step(1)` + `sync()` (no observation) | 0.82 ms | 0.85 ms | 3.61 ms | 3.69 ms |
| `observe(out=)` | 0.175 ms | 0.187 ms | 0.675 ms | 0.690 ms |
| `observe(copy=False)` (zero-copy view) | | 0.146 ms | | 0.304 ms |
| `step_observe(A, 1, out=)` (default path, bit-identical) | 0.90 ms | 0.93 ms | 4.27 ms | 4.29 ms |
| `step_observe(A, 1, copy=False)` | | 0.89 ms | | 3.89 ms |
| `step_observe(A, 1, out=, lag=1)` | | 0.89 ms | | 3.78 ms |
| `step_observe(A, 1, copy=False, lag=1)` | | 0.85 ms | | 3.41 ms |
| `step_async(A, 1)` + `observe_wait(copy=False)` | | 0.88 ms | | 3.52 ms |
| `step_observe(A, 1, out=f16, dtype="float16")` | | 0.93 ms | | 4.02 ms |
| `step_observe(A, 1, copy=False, dtype="float16")` | | 0.90 ms | | 3.97 ms |
| policy stand-in (`obs.sum()`, ~0.7 ms at 100k) + `step_observe(A, 1, out=)` | 0.98 ms | 1.03 ms | 4.99 ms | 5.13 ms |
| policy stand-in + `step_observe(A, 1, copy=False)` | | 0.96 ms | | 4.72 ms |
| policy stand-in + `step_observe(A, 1, copy=False, lag=1)` | | 0.84 ms | | 3.86 ms |

VecEnv `step_observe(A, 1)` per call (bench_loop.py VecEnv section, sequential, best of 3):

| | before (`out=`) | after `out=` | after `copy=False` | after `out=, lag=1` |
|---|---|---|---|---|
| 1 x 10k | 1.70 ms | 1.10 ms | 1.05 ms | 0.91 ms |
| 4 x 10k | 3.62 ms | 3.24 ms | 2.95 ms | 2.96 ms |
| 8 x 10k | 6.26 ms | 6.22 ms | 5.71 ms | 5.79 ms |
| 16 x 2.5k | 8.54 ms | 8.55 ms | 8.24 ms | 8.29 ms |
| 4 x 25k | 5.44 ms | 5.38 ms | 4.88 ms | 4.77 ms |

Where the time went at 100k (`observe` microbenchmarks): submit + wait round trip ~0.25 ms, the 28 MB host memcpy into
numpy ~0.4-0.5 ms, the 28 MB GPU copy `inputs` -> MAP_READ buffer < 0.05 ms (`step_observe(copy=False)` is within
~0.05-0.2 ms of a bare step). So:

* The zero-copy view removes the memcpy (-0.4 ms at 100k). On Metal a mapped wgpu buffer is `MTLBuffer.contents`
  (shared storage) and `unmap` is a no-op, so the view stays valid memory for the lifetime of the buffer (held by the
  view's `base`); its contents change once the 2-slot ring reuses that slot (documented: until the next call).
* Making `inputs` itself mappable (`Features::MAPPABLE_PRIMARY_BUFFERS`) would only save the GPU copy (< 0.05 ms) while
  forcing an unmap before every submit and invalidating views during the next step (the obs kernels write `inputs` in
  place; four kernels write parts of it, so a ring of `inputs` buffers would also change results), so it was not
  adopted.
* `lag=1` / `step_async` hide the remaining round trip and any Python work behind the GPU step: the per-call cost
  becomes max(GPU step, host work) instead of the sum (100k with a policy pass: 4.99 -> 3.86 ms, 1.29x).
* float16 halves the bytes (14 MB at 100k) but costs a conversion pass (~0.1 ms at 100k), so it only pays where the
  bytes are copied or moved on (copy path 0.69 vs 0.78 ms in `observe`; host -> accelerator transfers, replay buffers).

float16 error per observation field (`obs_check.py`: 10k cars, 100 x `step_observe(A, 10)` with random steering, 1M
rows; f16 = numpy's round-to-nearest-even `astype(float16)`, checked bit for bit). Fields not listed are exact (flags,
one-hot and values on the f16 grid): 6, 7, 21-32, 38, 40, 45-55, 58-64, 67, 68.

| fields | max \|x\| | max abs error | mean abs error |
|---|---|---|---|
| 0, 1, 3 | 0.81, 0.73, 1.17 | 2.4e-4, 2.4e-4, 3.3e-4 | 4.7e-5, 7.5e-5, 6.2e-5 |
| 2, 4, 5 | 1 | 2.4e-4 | 3.4e-6, 3.3e-5, 5.0e-5 |
| 8-20 | 1 | 2.4e-4 | 4e-7 - 3.6e-5 (20: 3.6e-5, 8: 9.8e-6) |
| 33, 34, 35, 37, 39 | ≤ 1 | 2.4e-4 | 8.1e-5 - 1.2e-4 |
| 36 | 1.8 | 4.9e-4 | 1.6e-4 |
| 41-44 | ≤ 1 | 2.4e-4 | 2.4e-6 - 8.4e-6 |
| 56, 57, 65, 66, 69 | 1 | 2.4e-4 | 5e-8 - 2.6e-5 |
| all 70 | 1.8 | 4.9e-4 (field 36) | 1.5e-5 |

The relative error is ≤ 4.9e-4 for |x| ≥ 6.1e-5 (normal f16); smaller magnitudes lose relative precision (subnormals) and
values below 6e-8 flush to 0 (max relative error 1 on fields 0, 4, 5, 9, 11, ...).

## Streaming route prefetch (RouteProvider workers, `route_threads`)

Full real-world morning: CommuterTable of 1,380,000 commuters (seed 42, the `packTable.mjs` default). All of them
depart between 04:00 and 10:00, the window a 05:00 `setStreaming` start routes by 10:00.
`cargo run --release -p bayenv-gpu --example stream_prefetch -- ../data route 1380000 14` (serial loop, then the sim's
pattern without the GPU: 1 s frames, prefetch 300 s ahead, each departure takes its route when the clock reaches it).
Apple M4 Max, 14 cores. The machine was shared, and all numbers were taken under the shared bench lock (`lockf`, `nice -n 10`)
at load averages of 60-90, so the per-route times are 1.3-2.8x the 0.75 ms of a lightly loaded machine.

| full morning, 1,376,825 A* | wall | routes/s | notes |
|---|---|---|---|
| serial at departure (`route_threads=0`) | 1,469 s | 937 | 1.06 ms per A* at load 60 |
| 14 prefetch workers | 220 s (6.7x) | 6,251 | 2.1 ms per A* per worker at load 90; the sim thread only waits for routes |

Memory (full-morning process, `/usr/bin/time -l`): 0.78 GB peak footprint with world, table, 15 routers and a 200k-OD cache (the default is now no cache). A `nav::Router` holds per-node search state (495k nodes) plus its CSR copy of the graph. The cost is linear in `route_threads`.
| route hash over all 1.38M commuters | identical | | |

OD cache: 0.2 % of departures share an OD with an earlier one inside the cache's reach (1,376,825 of 1,380,000
departures need their own A*). The prefetch hand-over (`done` map per OD) already shares one A* between departures
with the same OD inside the 300 s window, so the LRU only adds ODs further apart:

| 07:00-08:00 departures (432,169), 14 workers | wall | A* | cache hits | sim-thread locks + bookkeeping | memory |
|---|---|---|---|---|---|
| no cache (`route_cache=0`, default) | 40.6 s | 432,009 | - | 0.30 s | - |
| LRU 200,000 ODs (the JS `RouteProvider` size) | 31.3 s | 431,376 (-0.15 %) | 634 | 0.52 s | 61 MB of route edges + ~20 MB map |

The wall-time difference is load noise: the per-worker A* time was 1.24 ms vs 0.96 ms, and the cache saves 0.15 %
of the A* runs. Reroutes do not go through this provider: missed-exit reroutes use the GPU host's own router, which has
its own OD cache. Congestion / closure reroutes depend on the current penalty and avoid sets, so an OD cache cannot serve
them. Their OD repeat rates are in the sim table below.

GPU sim at the morning peak, full population streaming
(`stream_prefetch -- ../data sim 1380000 14 7.25 0.75 20 131072 0 0.25`: 07:15 start, capacity 131072, batches of 20
steps, stations on, measured 07:15-08:00 in 15-min chunks, under the bench lock with `nice -n 10`, load 12-52):

| 14 route threads | wall | cars on road | departures placed | sim-thread A* / wait | worker A* |
|---|---|---|---|---|---|
| 07:15-07:30 (first frame places the 06:15-07:15 backlog up to the pool size) | 530 s | 111,682 | 114,783 | 0 / 4.0 s | 417,739 |
| 07:30-07:45 | 557 s | 102,567 | 4,320 | 0 / 0 s | 105,668 |
| 07:45-08:00 | 471 s | 100,820 | 483 | 0 / 0 s | 121,011 |

That is ~2,080 s of wall time per simulated hour at ~100k cars. Departure routing is none of it: the slot pool is full
the whole time (departures queue as `poolFull`). The time is host-side reroute A*, not GPU; see "Streaming step
profile" below, which takes this window from 17.5 to 5.2 ms/step. Serial (`route_threads=0`) was not measured in
this window (skipped to free the bench lock). Its routing cost is the A* of the placed departures on the sim thread:
119,586 x ~1.06 ms ~ 127 s over the 45 min (+8 %), most of it in the first frame.
This run found a memory problem: with the pool full the prefetch window kept moving, and 644k routes for pending
departures were held (2.43 GB peak footprint). Prefetch now pauses at `RouteProvider::MAX_OUTSTANDING` = 65,536 ODs in
flight or finished, and departures beyond that are routed when they are placed. The spawn hashes are unchanged, and the
capped version was not re-timed.
Reroutes in the same window: 623,360 queued (255,210 applied). 858 are plain (missed exit or no penalties), with 856
distinct ODs (0.2 % repeats). 622,502 use traffic penalties, with 622,378 distinct (OD, 60 s penalty epoch) keys (0.02 %
repeats). An OD cache would not help reroutes either.

Determinism (`stream_prefetch spawn`, host-only: frames of 20 steps, SPAWN at once, ARRIVE after length / 12 m/s; hash
of commuter, step, slot, queue entry, arena offset and route words of every placed departure): bit-identical for
`route_threads` 0 / 2 / 3 / 14, for 200k commuters over 05:00-10:00 (pool 16384), and for a pool of 1024 that is always
full (73,618 placed, the pending queue retried every frame). The GPU runs (50k commuters, 06:00-07:00, two runs each
at 0 and 14 threads) give the same (commuter, step, route) hash. Slot numbers in GPU runs follow the order of the
arrival events, which is not repeatable over an hour even serially, so slots are not part of that hash.
`parity_check` vs the JS mock (`--scenario=stream`) is identical with the workers on: the first frame (3,217 departures)
takes 438 ms vs 2,239 ms serial.

## Streaming step profile (full population, morning peak)

`stream_prefetch -- ../data prof 1380000 14 7.5 2 5 20` (CommuterTable cached with `BAYENV_TABLE_CACHE`, 07:30 start, 2 sim
min warm-up, then 6,000 steps = 5 sim min timed, batches of 20, capacity 131072, ~87k -> 105k cars, stations on, record
events on). Split per step: host submit (enqueue, scenes, pumps, refuels, reroutes, encode), GPU wait (poll), host
finish. Under the bench lock, `nice -n 10`, load 4-16.

| | ms/step | host submit | GPU wait | warm-up (2 sim min incl. the 115k-departure first frame) |
|---|---|---|---|---|
| before | 17.54 | 13.49 | 4.02 | 46.9 s |
| after | 5.23 | 1.42 | 3.79 | 15.1 s |

`sample` of the main thread showed the cause. The GPU part (3.8 ms at ~100k cars) matches the 100k scatter bench
(3.5 ms). Event records cost ~1 ms/step (`record_events=0`: 22.15 -> 21.06 ms/step in a 2-min window). The rest was host
work:

* Congestion reroutes, ~60 % of the main thread: up to 200 per frame (the JS budget, ~160 per 20-step call here),
  each a penalised A* run serially, plus the traffic-penalty list rebuilt from the map for every query (~25 %) and
  a `HashSet` of the closed edges built for every applied reroute. Now the filter / budget pass stays in queue order,
  the frame's queries run as one `nav::Router::route_batch` (parallel; result i = `route_with(query i)`), and the
  penalty list and closed set are built once per frame (they change only outside `process_reroutes`). The results are
  applied in queue order, and a slot queued twice flushes the batch first.
* Departures routed on the sim thread: after a full first frame, the prefetch window skipped the departures waiting
  for a slot (`pending_stream`). They are now prefetched first, with one request per commuter.
* Refuel plans: with no wall-clock budget (`refuel_budget_ms=None`, the Rust default) the frame's plans run as one
  `StationNetwork::plan_many` (parallel router batches). Results and failures are applied in queue order. A finite
  budget keeps the one-at-a-time loop.

Bit-identity:
* `BAYENV_REROUTE_CHECK=1` recomputes every batched reroute one at a time (goal from the current route, per-query
  penalty list, `route` / `route_with`) and asserts equality. 32,306 peak-window reroutes, all equal.
* `parity_check` vs the JS mock: `stream` and `pop` (40 refuel plans, refuel route blocks) all identical.
* Spawn-table hashes unchanged (30k: f17c5c94f01dba69; 200k, full pool of 1024: d19a400a4f8f4847).
* Whole-run state can't be compared between builds: two runs of the same binary already differ after 1 sim min of the
  peak (order of contacts / arrivals on the GPU, also with collisions off).

Remaining host time (~1.4 ms/step) is mostly waiting for the reroute batch.
