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

Contacts per step: ~6 at 10k, ~520 at 100k, none left to the serial pass. Run-to-run the unmodified host is
itself not bit-reproducible beyond ~200 steps (atomic orders: hash ranks, pair table, event order): after 1,000
steps two runs of the baseline differ in 93 of 8,766 cars, baseline vs fused resolve in 91, same aggregates
(7,559-7,561 driving, 940-941 crash scenes).
Sorted order on the other per-car kernels (junction.perceive, courtesy, act, integrate, contacts.barrier) is
bit-identical too but slower or neutral (the extra cap + lane-entry threads), so those entries are off by default
(`SORTED_KERNELS`, override `BAYENV_SORTED=...`).
