# Rust workspace architecture

Port of the JS environment (`src/`) to Rust. The JS implementation is the oracle: every port is checked
against dumps of the JS reference (`tools/oracle/`), bit-exact where the JS is deterministic.

```
rust/
  Cargo.toml                 workspace (pinned dependency versions in [workspace.dependencies])
  crates/
    bayenv-world/            world data: road packs, SimGraph, lane geometry, junctions (+ builder), traffic-control
                             plans, surfaces, grade separations, stations, DEM, node index, bake format, jsmath
    bayenv-sim/              CPU reference sim (port of src/sim: physics, vehicle, collisions, crash model,
                             controls runtime, lane ends, route/nav A*, fuel, population, obs, env API)  [branch rust-cpu]
    bayenv-gpu/              wgpu host for the unchanged WGSL kernels (src/gpu/kernels): world pack, buffers,
                             step loop, scatter / population, actions, obs, state readback
    bayenv-py/               PyO3 module `bayenv`: CPU and GPU sims behind one API, numpy zero-copy, batched steps
  tools/oracle/              node scripts that dump JS reference arrays for comparison
  oracle/                    (gitignored) dumps written by tools/oracle
```

Dependency graph: `bayenv-world` <- `bayenv-sim`, `bayenv-gpu` <- `bayenv-py`. External (private) crates depend
on these via `git = "https://github.com/bubbabjh2/bayarea-driving-env", branch = "rust"`.

## bayenv-world public API (stable)

```rust
use bayenv_world::{World, LoadOptions};
let w = World::load("data")?;                 // ~60 ms warm (mmap'd packs + bake)
w.graph            // SimGraph (SoA): nodes nx/nz/node_y, edges from/to/reverse/lanes/class/rank/nav_rank/eflags/
                   // speed_limit/length/heading_start/heading_end/y, polyline CSR pt_start + px/pz/cum/ys,
                   // lane CSR lane_start + lane_off, right_edge, out_*/in_* CSR, banned_hops, surf/gnd_r/gnd_l,
                   // s_exit/s_stop/gore (junction mouths), bounds, max_speed; edge_point(e, s, offset, hint)
w.lane_graph       // LaneGraph (src/geom/graph.js numbering)
w.lanes            // LaneGeometry: right_edge, lane_width, bounds, offset_point, lat_point, lane_point, ...
w.junctions        // Option<JunctionSet>: mouth_s, gore_s, j_*, m_info (7/movement), m_f, pts, conf, conf_s, hull;
                   // describe(m), arc_point(m, t)
w.controls         // Option<TrafficControls>: plans (cycle, offset, phases, ap_*), approach_ahead(e, s), state_of(a, move, t)
w.stations         // Vec<Station>;  w.contact_pairs: Option<ContactPairs>;  w.dem: Option<Dem> (LoadOptions.dem)
w.node_grid        // NodeGrid::nearest(&graph, x, z, filter)
bayenv_world::jsmath   // V8-exact Math.* (hypot, atan2, sin, cos, exp, log, round, min, max, ...): use for bit-exact ports
bayenv_world::hops::hop_map, nav::maneuvers   // (ports in progress)
```

Index conventions = JS: main-tier edges `0..main_edge_count`, local edges after; nodes per `merged_node_maps`.
Numeric types = JS: f64 where JS holds a double, f32 where it stores into a Float32Array.

## Data formats

* Inputs: the committed `data/*.meta.json + *.bin` packs, mmap'd zero-copy (`blob::Pack`, `blob::Slice<T>`).
* Bake: `data/build/rust/world.bake` (gitignored), `bake::Bake` container: magic + version + xxh3 source key +
  section table + xxh3 body checksum, 64-byte aligned sections, mmap'd. Holds derived arrays that are expensive in
  JS (lane geometry; junctions when `data/junctions.bin` is absent). Rebuilt when the key (inputs) changes.

## Equivalence testing

* `cargo run --release -p bayenv-world --bin bayenv-world -- check ../data oracle` compares the loaded world with
  `node rust/tools/oracle/dump_world.mjs data rust/oracle` (graph, controls, lane geometry; element for element).
* GPU sim: same seed + constant actions in the JS GPU sim (headless Chrome, `test/gpu/run.mjs`) and the wgpu host,
  per-car state compared after N steps (`crates/bayenv-gpu/tests`, `tools/oracle/gpu_*.mjs`).
* Benchmarks: `BENCHMARKS.md` (before / after for every optimization).

## Floating point vs JS

V8 on arm64 is compiled with FMA contraction, and its `Math.sin/cos` come from glibc, the rest from fdlibm; the
`libm` crate and the system libm both differ in the last ulp on ~0.1-10 % of inputs. `jsmath` reproduces V8 per
platform (verified on a generated test set). Bit-exact ports must route all transcendental calls through `jsmath`.
