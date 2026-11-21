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
