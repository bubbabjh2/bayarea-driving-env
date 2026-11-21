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
  README.md                  build, Python usage, equivalence results, benchmarks
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
bayenv_world::hops::hop_map, nav::{maneuvers, Router, ManeuverTable}   // geom hops, guidance, A* (route.js)
```

Index conventions = JS: main-tier edges `0..main_edge_count`, local edges after; nodes per `merged_node_maps`.
Numeric types = JS: f64 where JS holds a double, f32 where it stores into a Float32Array.

## Router (bayenv_world::nav::router, route.js makeRouter)

```rust
use bayenv_world::nav::{Router, RouterOpts, RouteQuery};
let r = Router::new(&w.graph, &RouterOpts { weight: 1.25, ..Default::default() }); // cache_max 50000 (0 = off)
// batch (rayon, all cores): one result per query, in order; equal to route_with(query) one by one
let qs: Vec<(usize, usize, RouteQuery)> = vec![(start, goal, RouteQuery::default()),
                                              (start, goal, RouteQuery { avoid: &closed, penalty: &pen })];
let paths: Vec<Option<Vec<u32>>> = r.route_batch(&qs);       // &self: callable from a shared reference
let mut r2 = r.fork();                                         // same graph + OD cache, own search state (a thread)
r2.route(start, goal); r2.route_with(start, goal, &q); r2.route_into(start, goal, &q, &mut buf);
```

* `RouterGraph` (immutable, `Arc`, ~25 MB for the full corridor) holds the search graph; `SearchState` (16 B per
  node, + 16 B per edge once a query uses avoid / penalty) the per-query arrays. `Router` = graph + own state +
  pool of states for `route_batch` + optional `Arc<OdCache>`.
* `OdCache`: (start, goal) -> path for plain queries (empty avoid and penalty), 64 mutex shards, a shard holding
  `cache_max / 64` entries is cleared before the next insert (the JS clears the whole Map). Routes are a pure
  function of (graph, opts, start, goal), so a cached route equals a fresh one. Shared by `fork`ed routers;
  `route_batch` also searches each distinct plain (start, goal) of a batch once.
* Exactness (all results bit-identical to the JS, including tie-breaking and "no path"):
  * the JS pops a binary `MinHeap` with lazy deletion; with weighted A* (w = 1.25, heuristic `w * dist / vmax`,
    inconsistent) a node can be popped, closed and reopened, and the path depends on the whole pop sequence. Among
    equal keys the pop order depends on the heap's array layout.
  * fast path: 4-ary heap; pushes with key > the smallest goal key in the heap are skipped (node state is still
    updated): while the goal entry is in the heap every popped key is <= it, and popping the goal ends the search,
    so such an entry is never popped. At every pop the fast path checks that the popped key is strictly below the
    new heap top. If every pop of a search is a unique minimum, the sequence of popped entries is determined by
    the key multiset alone and equals the JS heap's sequence (the JS heap holds the same entries plus the skipped
    ones, which are all above every key it pops before the goal), so states, pushes and the result are equal by
    induction. On the first tie the query is rerun with the exact route.js heap (same sift order, no skipping).
    Ties are rare on the real graph (0 in 100k random queries; the synthetic equal-cost grid test forces them).
    Keys must be NaN-free for the argument: queries with non-finite / huge penalties use the exact heap directly.
  * "no path" without a search when the goal's strongly connected component is not reachable from the start's
    in the condensation order (Tarjan numbering: an edge a -> b implies comp[b] <= comp[a]); bans and avoid sets only
    remove edges, so plain unreachability implies unreachability for every query.
  * layout only (cannot change results): Morton node numbering, CSR out-edges carrying the head node's coordinates
    and out-edge span (heap entries carry the span, the pushed node's out-edges are prefetched), `gen << 1 | closed`
    stamps, junction penalty by rank from a 256-entry table (same expression as the JS).
* Not used, because they change the result: bidirectional search, contraction hierarchies, ALT / landmark
  heuristics, decrease-key with a different tie rule. The JS route is not "the" shortest path but the output of
  one weighted-A* expansion order (w > 1: suboptimal, reopenings, lazy deletion); any search that expands other nodes or in
  another order returns other paths, and even at w = 1 equal-cost alternatives and f64 summation order (a
  bidirectional / CH path sums costs in another order) differ in the last bits. Landmark lower bounds cannot prune
  exactly either: whether a node is popped before the goal depends only on its key (known exactly), and the
  goal-key skip above already drops every entry that provably is not popped. Landmark distances could prove "no
  path" only where plain-graph reachability already does (handled by the SCC test).
* Tests: `tests/nav_oracle.rs` (JS dumps: 5250 queries), `tests/router_equiv.rs` (vs the pre-optimisation router
  `nav::router_ref`, feature `router-ref`: random OD pairs x 6 variants incl. no-path, avoid, penalty, both,
  NaN / inf / negative penalties, single + batch + cached, exact-only chunks; `ROUTER_EQ_N=17000` = 102k queries;
  plus a tie-heavy synthetic grid with banned hops). Bench: `cargo run --release -p bayenv-world --example router_bench -- ../data oracle/nav/nav.json`.

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

V8 (node 22.22.0 = V8 12.4) implements every `Math` transcendental with its fdlibm port `src/base/ieee754.cc`
(node's gyp build never sets `V8_USE_LIBM_TRIG_FUNCTIONS`, so sin/cos are fdlibm too; `Math.pow` is
`ieee754::pow`). Clang compiles it with `-ffp-contract=on`, so on arm64 each `a*b + c` inside one C expression
is an FMA. `jsmath` ports that file statement by statement with explicit `mul_add` at exactly those spots
(plain mul + add on non-aarch64 targets: unverified). The `libm` crate and the system libm differ from node in
the last ulp on ~0.1-30 % of inputs. Bit-exact ports must route all transcendental calls through `jsmath`.
Check: `node rust/tools/oracle/gen_math.mjs` (writes `rust/oracle/math.bin`), then
`cargo test -p bayenv-world --release --test jsmath_oracle` (0 mismatches required).
