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
