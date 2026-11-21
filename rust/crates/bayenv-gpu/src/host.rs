//! Host logic of `src/gpu/gpuSim.js` beyond scatter (child module of `sim`, so it shares `GpuSim`'s private state):
//!
//! * `setPopulation(pop, {stations})` + `_enqueueSpawns`: trip schedule (outbound + return legs sorted by departure),
//!   spawn queue entries, fuel carried between legs (`fuelLeft`), outcomes, agents buffer sizing.
//! * `setStreaming(table)` + `_enqueueStream`: CommuterTable (src/sim/stream.js), routes at departure from a
//!   synchronous RouteProvider (A*, OD cache) into the size-class route arena.
//! * Refuel plans (`_processRefuels` / `_applyRefuel`, the main-thread path with `StationNetwork.plan` of
//!   bayenv-sim), station lots (`_enterStation`, REENTER) and pumps (`_pumps`).
//! * `placeObstacles` (spec or explicit list), `placeWrecks`, `importSnapshot` (src/sim/state.js exportState),
//!   `resetDay`, `readAgents`.
//!
//! Populations come from bayenv-sim (`population_gen`, bit-exact port of src/sim/population.js); this module only
//! converts them to the Float32 arrays the JS GPU path uses (`pop.arrays`).
use super::{f32bits, GpuSim, Wreck, NONE};
use crate::defines::{default_traits as dt, VEHICLE_PARAMS};
use crate::layout::{self as L, flag, op, state};
use bayenv_sim::population::Population;
use bayenv_sim::population_gen::{self as pg, PopConfig, DELTA_SIZE};
use bayenv_world::{jsmath, World};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

// ------------------------------------------------------------------ routers
/// bayenv_world's A* (route.js makeRouter) behind bayenv-sim's `Router` trait (avoid / penalty supported).
pub struct NavRouter(pub bayenv_world::nav::Router);
impl NavRouter {
    pub fn new(w: &World, weight: f64) -> NavRouter { NavRouter(bayenv_world::nav::Router::new(&w.graph, &bayenv_world::nav::RouterOpts { weight, cache_max: 0, ..Default::default() })) }
}
fn nav_route(r: &mut bayenv_world::nav::Router, start: i32, goal: i32, avoid: Option<&HashSet<i32>>, penalty: Option<&HashMap<i32, f64>>) -> Option<Vec<i32>> {
    let p = if avoid.is_none() && penalty.is_none() { r.route(start as usize, goal as usize) } else {
        let a: Vec<u32> = avoid.map(|s| s.iter().map(|&x| x as u32).collect()).unwrap_or_default();
        let p: Vec<(u32, f64)> = penalty.map(|m| m.iter().map(|(&e, &v)| (e as u32, v)).collect()).unwrap_or_default();
        r.route_with(start as usize, goal as usize, &bayenv_world::nav::RouteQuery { avoid: &a, penalty: &p })
    };
    p.map(|v| v.into_iter().map(|x| x as i32).collect())
}
impl bayenv_sim::sim::Router for NavRouter {
    fn route(&mut self, s: i32, g: i32, a: Option<&HashSet<i32>>, p: Option<&HashMap<i32, f64>>) -> Option<Vec<i32>> { nav_route(&mut self.0, s, g, a, p) }
}
struct NavRef<'a>(&'a mut bayenv_world::nav::Router);
impl bayenv_sim::sim::Router for NavRef<'_> {
    fn route(&mut self, s: i32, g: i32, a: Option<&HashSet<i32>>, p: Option<&HashMap<i32, f64>>) -> Option<Vec<i32>> { nav_route(self.0, s, g, a, p) }
}

// ------------------------------------------------------------------ populations
/// What `GPUSim.setPopulation` reads from a population (src/sim/population.js `pop.arrays` are Float32Arrays).
#[derive(Clone, Debug, Default)]
pub struct GpuPopulation {
    pub n: usize,
    /// route of trip k = 2 * id + leg: route_flat[route_offset[k] .. route_offset[k + 1]]
    pub route_flat: Vec<i32>, pub route_offset: Vec<u32>,
    pub vehicle_type: Vec<u8>, pub work_start: Vec<f32>, pub depart: Vec<f32>, pub return_depart: Vec<f32>,
    pub fuel_fill: Option<Vec<f32>>,
    /// 64 floats per commuter (traits in 0..6)
    pub deltas: Option<Vec<f32>>,
}
impl From<&Population> for GpuPopulation {
    fn from(p: &Population) -> Self {
        let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        GpuPopulation { n: p.n, route_flat: p.route_flat.clone(), route_offset: p.route_offset.clone(), vehicle_type: p.vehicle_type.clone(),
            work_start: f(&p.work_start), depart: f(&p.depart), return_depart: f(&p.return_depart),
            fuel_fill: Some(f(&p.fuel_fill)), deltas: if p.deltas.is_empty() { None } else { Some(p.deltas.clone()) } }
    }
}
impl GpuPopulation {
    /// A population pack (`src/gpu/tools/packPopulation.mjs` / popPack.js format: `<base>.json` sections + `<base>.bin`).
    pub fn load_pack(base: &std::path::Path) -> Result<GpuPopulation, String> {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(base.with_extension("json")).map_err(|e| format!("{}: {e}", base.display()))?).map_err(|e| e.to_string())?;
        let bin = std::fs::read(base.with_extension("bin")).map_err(|e| e.to_string())?;
        let sec = |k: &str| -> Option<&[u8]> { let s = &meta["sections"][k]; let (o, n) = (s["byteOffset"].as_u64()? as usize, s["length"].as_u64()? as usize);
            let w = match s["type"].as_str()? { "Uint8Array" | "Int8Array" => 1, "Float64Array" => 8, _ => 4 }; Some(&bin[o..o + n * w]) };
        let f32s = |k: &str| sec(k).map(|b| bytemuck::pod_collect_to_vec::<u8, f32>(b));
        let n = meta["n"].as_u64().ok_or("pack has no n")? as usize;
        Ok(GpuPopulation { n, route_flat: bytemuck::pod_collect_to_vec(sec("routesFlat").ok_or("routesFlat")?), route_offset: bytemuck::pod_collect_to_vec(sec("routesOffset").ok_or("routesOffset")?),
            vehicle_type: sec("vehicleType").ok_or("vehicleType")?.to_vec(), work_start: f32s("workStart").ok_or("workStart")?, depart: f32s("depart").ok_or("depart")?,
            return_depart: f32s("returnDepart").ok_or("returnDepart")?, fuel_fill: f32s("fuelFill"), deltas: f32s("deltas") })
    }
    /// `generatePopulation(graph, { n, seed, returnTrips })` (bayenv-sim, bit-exact with the JS; rayon, same result for
    /// any thread count). The population is generated on a graph-only copy of the world (`routing_from_bayenv`).
    pub fn generate(world: &World, n: usize, seed: f64, return_trips: bool) -> Result<GpuPopulation, String> {
        let cfg = PopConfig { n, seed, return_trips, ..Default::default() };
        let rw = bayenv_sim::world::World::routing_from_bayenv(world);
        let g = pg::generate_population_with(&rw, &cfg, || NavRouter::new(world, cfg.astar_weight), None)?;
        Ok(GpuPopulation::from(&g.pop))
    }
}

/// `src/sim/stream.js CommuterTable` (no routes, no deltas kept; ~40 B per commuter).
#[derive(Clone, Debug)]
pub struct CommuterTable {
    pub n: usize,
    pub home: Vec<i32>, pub work: Vec<i32>, pub depart: Vec<f32>, pub work_start: Vec<f32>, pub return_depart: Vec<f32>,
    pub reaction_time: Vec<f32>, pub fuel_fill: Vec<f32>, pub vehicle_type: Vec<u8>,
    /// ids sorted by departure (then id); sorted_depart[k] = depart[order[k]]
    pub order: Vec<u32>, pub sorted_depart: Vec<f32>,
    cfg: PopConfig, base: [f32; DELTA_SIZE], seed_str: String,
}
impl CommuterTable {
    /// `buildCommuterTable(graph, { n, seed })` (returnTrips false); commuters generated in parallel, routes dropped.
    pub fn generate(world: &World, n: usize, seed: f64) -> Result<CommuterTable, String> {
        let cfg = PopConfig { n, seed, return_trips: false, ..Default::default() };
        let rw = bayenv_sim::world::World::routing_from_bayenv(world);
        let ctx = pg::Context::new(&rw, &cfg)?;
        let chunk = (n / (rayon::current_num_threads() * 8)).clamp(16, 2048);
        let chunks: Vec<(usize, usize)> = (0..n).step_by(chunk.max(1)).map(|s| (s, (s + chunk).min(n))).collect();
        let pool: std::sync::Mutex<Vec<NavRouter>> = std::sync::Mutex::new(Vec::new());
        type Row = (i32, i32, f32, f32, f32, f32, f32, u8);
        let parts: Vec<Result<Vec<Row>, String>> = chunks.par_iter().map(|&(a, b)| {
            let popped = pool.lock().unwrap().pop();
            let mut r = popped.unwrap_or_else(|| NavRouter::new(world, cfg.astar_weight));
            let out = (a..b).map(|id| pg::generate_commuter(&rw, &cfg, &ctx, &mut r, id).map(|c|
                (c.home, c.work, c.depart as f32, c.work_start as f32, c.return_depart as f32, c.reaction_time as f32, c.fuel_fill as f32, c.vehicle_type))).collect();
            pool.lock().unwrap().push(r);
            out
        }).collect();
        let mut t = CommuterTable { n, home: Vec::with_capacity(n), work: Vec::with_capacity(n), depart: Vec::with_capacity(n), work_start: Vec::with_capacity(n),
            return_depart: Vec::with_capacity(n), reaction_time: Vec::with_capacity(n), fuel_fill: Vec::with_capacity(n), vehicle_type: Vec::with_capacity(n),
            order: Vec::new(), sorted_depart: Vec::new(), base: pg::make_genome(None, cfg.delta_norm_cap), seed_str: pg::js_num_str(cfg.seed), cfg };
        for p in parts { for r in p? { t.home.push(r.0); t.work.push(r.1); t.depart.push(r.2); t.work_start.push(r.3); t.return_depart.push(r.4); t.reaction_time.push(r.5); t.fuel_fill.push(r.6); t.vehicle_type.push(r.7); } }
        Ok(t.finish())
    }
    /// `finish()`: ids sorted by departure (f32), ties by id.
    pub fn finish(mut self) -> CommuterTable {
        let mut ids: Vec<u32> = (0..self.n as u32).collect();
        let d = &self.depart;
        ids.sort_by(|&a, &b| d[a as usize].partial_cmp(&d[b as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        self.sorted_depart = ids.iter().map(|&i| d[i as usize]).collect(); self.order = ids;
        self
    }
    /// `indexAt(t)`: first k with sorted_depart[k] >= t.
    pub fn index_at(&self, t: f64) -> usize { let a = &self.sorted_depart; let (mut lo, mut hi) = (0, a.len()); while lo < hi { let m = (lo + hi) >> 1; if (a[m] as f64) < t { lo = m + 1; } else { hi = m; } } lo }
    /// `deltaFor(id)`: the commuter's 64-float delta, rebuilt from (seed, id).
    pub fn delta_for(&self, id: usize) -> [f32; DELTA_SIZE] { pg::delta_for(&self.seed_str, &self.cfg, &self.base, id, self.reaction_time[id] as f64) }
}

/// `RouteProvider` (stream.js): A* (weight 1.25) with an OD cache (the cache never changes a route).
pub struct RouteProvider { router: bayenv_world::nav::Router, cache: HashMap<u64, Option<Arc<Vec<i32>>>>, cache_size: usize, pub hits: u64, pub misses: u64 }
impl RouteProvider {
    pub fn new(w: &World) -> RouteProvider { RouteProvider { router: bayenv_world::nav::Router::new(&w.graph, &bayenv_world::nav::RouterOpts { weight: 1.25, cache_max: 0, ..Default::default() }), cache: HashMap::new(), cache_size: 200000, hits: 0, misses: 0 } }
    pub fn get(&mut self, from: i32, to: i32) -> Option<Arc<Vec<i32>>> {
        let key = (from as u32 as u64) << 32 | to as u32 as u64;
        if let Some(r) = self.cache.get(&key) { self.hits += 1; return r.clone(); }
        self.misses += 1;
        let r = self.router.route(from as usize, to as usize).map(|v| Arc::new(v.into_iter().map(|x| x as i32).collect::<Vec<i32>>()));
        if self.cache.len() >= self.cache_size { self.cache.clear(); }
        self.cache.insert(key, r.clone());
        r
    }
}

#[derive(Clone, Debug, Default)]
struct Arena { free: HashMap<u32, Vec<u32>>, slot: Vec<i32>, cls: Vec<u8> }

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats { pub routed: u64, pub waiting_route: u64, pub pool_full: u64 }

/// Stations: refuel planner (bayenv-sim `StationNetwork`) + the pump queues of `gpuSim.js` (FIFO per station).
pub struct Stations {
    pub net: bayenv_sim::fuel::StationNetwork,
    pub busy: Vec<i32>, pub served: Vec<i32>, queue: Vec<VecDeque<(u32, f64)>>,
    pub max_queue: usize, pub wait_sum: f64, pub wait_n: u64,
}
#[derive(Clone, Copy, Debug)]
struct Pump { slot: u32, station: usize, until: f64 }

/// One placed obstacle (`placeObstacles` result).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed { pub slot: u32, pub kind: ObstacleKind, pub edge: u32, pub s: f64 }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObstacleKind { Static, Midblock, AfterCurve, AfterMerge, Junction, RampEnd, Partial, Slow, Braking }
impl ObstacleKind {
    pub const ALL: [ObstacleKind; 8] = [ObstacleKind::Midblock, ObstacleKind::AfterCurve, ObstacleKind::AfterMerge, ObstacleKind::Junction, ObstacleKind::RampEnd, ObstacleKind::Partial, ObstacleKind::Slow, ObstacleKind::Braking];
    pub fn name(self) -> &'static str { match self { ObstacleKind::Static => "static", ObstacleKind::Midblock => "midblock", ObstacleKind::AfterCurve => "afterCurve", ObstacleKind::AfterMerge => "afterMerge",
        ObstacleKind::Junction => "junction", ObstacleKind::RampEnd => "rampEnd", ObstacleKind::Partial => "partial", ObstacleKind::Slow => "slow", ObstacleKind::Braking => "braking" } }
    pub fn parse(s: &str) -> Option<ObstacleKind> { ObstacleKind::ALL.iter().copied().find(|k| k.name() == s) }
}
/// `placeObstacles(spec)` options (JS defaults).
#[derive(Clone, Debug)]
pub struct ObstacleSpec { pub count: usize, pub kinds: Vec<ObstacleKind>, pub seed: f64, pub slow_kmh: [f64; 2], pub brake_every: [f64; 2] }
impl Default for ObstacleSpec { fn default() -> Self { ObstacleSpec { count: 40, kinds: ObstacleKind::ALL.to_vec(), seed: 1.0, slow_kmh: [10.0, 20.0], brake_every: [15.0, 40.0] } } }
/// Explicit obstacle `{ edge, at, lane | lat, type }`.
#[derive(Clone, Copy, Debug)]
pub struct StaticObstacle { pub edge: i64, pub at: f64, pub lane: Option<f64>, pub lat: Option<f64>, pub ty: u32 }

/// Host state of populations / streaming / stations / obstacles.
#[derive(Default)]
pub struct Ext {
    pub pop: Option<Arc<GpuPopulation>>,
    trip_route_base: usize, trip_route_end: Option<usize>,
    /// trip ids (2 * commuter + leg) in departure order, their departures (f32), next to enqueue
    pub trips: Vec<i32>, pub trip_depart: Vec<f32>, pub trip_ptr: usize,
    /// per commuter: fuel left after the last leg (-1 none), outcome of the outbound leg (1 arrived, 2 crash, 3 stranded, 4 stuck, 5 no route)
    pub fuel_left: Vec<f32>, pub outcome: Vec<u8>,
    pub table: Option<Arc<CommuterTable>>, table_ptr: usize, pending_stream: Vec<u32>, provider: Option<RouteProvider>, arena: Option<Arena>,
    pub stream_deltas: Option<Arc<Vec<f32>>>, pub stream_stats: StreamStats,
    pub stations: Option<Stations>, pumping: Vec<Pump>, pub(super) in_lot: HashSet<u32>,
    pub obstacles: Vec<Placed>,
    /// refuel plans per frame: wall-clock budget (JS refuelBudgetMs; f64::INFINITY = all queued plans every frame)
    pub refuel_budget_ms: Option<f64>,
}
impl Ext {
    pub fn has_pop(&self) -> bool { self.pop.is_some() || self.table.is_some() }
    pub fn vehicle_type(&self, id: usize) -> usize {
        if let Some(t) = &self.table { t.vehicle_type[id] as usize } else if let Some(p) = &self.pop { p.vehicle_type[id] as usize } else { 0 }
    }
    pub fn commuters(&self) -> usize { if let Some(t) = &self.table { t.n } else if let Some(p) = &self.pop { p.n } else { 0 } }
}

/// Per-agent day results (`readAgents`).
#[derive(Clone, Debug, Default)]
pub struct Agents { pub reward_sum: Vec<f32>, pub outcome: Vec<u8>, pub outcome_back: Vec<u8>, pub progress: Vec<f32>, pub trip_time: Vec<f32>, pub crashes: Vec<u32> }

/// A CPU-sim snapshot (`src/sim/state.js exportState`): per-vehicle arrays by name (values as f64; `ctrlRing` f32),
/// the route buffer and the clock. Load with [`Snapshot::load`] (written by rust/tools/oracle/gpu_parity_mock.mjs).
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub version: u32, pub time: f64, pub step: u64, pub dt: f64, pub high: usize,
    pub arrays: HashMap<String, Vec<f64>>, pub ctrl_ring: Vec<f32>, pub route_flat: Vec<i32>,
    /// commuters[id].workStart (importSnapshot's `commuters` argument), if given
    pub work_start: Option<Vec<f64>>,
}
impl Snapshot {
    /// `<base>.json` (meta, sections {name: {type, byteOffset, length}}) + `<base>.bin`.
    pub fn load(base: &std::path::Path) -> Result<Snapshot, String> {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(base.with_extension("json")).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let bin = std::fs::read(base.with_extension("bin")).map_err(|e| e.to_string())?;
        let mut s = Snapshot { version: meta["version"].as_u64().unwrap_or(0) as u32, time: meta["time"].as_f64().unwrap_or(0.0), step: meta["stepCount"].as_u64().unwrap_or(0),
            dt: meta["dt"].as_f64().unwrap_or(0.05), high: meta["high"].as_u64().unwrap_or(0) as usize, ..Default::default() };
        for (name, sec) in meta["sections"].as_object().ok_or("no sections")? {
            let (ty, off, len) = (sec["type"].as_str().unwrap_or(""), sec["byteOffset"].as_u64().unwrap() as usize, sec["length"].as_u64().unwrap() as usize);
            let b = &bin[off..];
            let rd = |k: usize, w: usize| -> [u8; 8] { let mut x = [0u8; 8]; x[..w].copy_from_slice(&b[k * w..k * w + w]); x };
            let v: Vec<f64> = (0..len).map(|k| match ty {
                "Float32Array" => f32::from_le_bytes(rd(k, 4)[..4].try_into().unwrap()) as f64, "Float64Array" => f64::from_le_bytes(rd(k, 8)),
                "Int32Array" => i32::from_le_bytes(rd(k, 4)[..4].try_into().unwrap()) as f64, "Uint32Array" => u32::from_le_bytes(rd(k, 4)[..4].try_into().unwrap()) as f64,
                "Int8Array" => b[k] as i8 as f64, "Uint8Array" => b[k] as f64, "Int16Array" => i16::from_le_bytes(rd(k, 2)[..2].try_into().unwrap()) as f64,
                "Uint16Array" => u16::from_le_bytes(rd(k, 2)[..2].try_into().unwrap()) as f64, _ => f64::NAN }).collect();
            match name.as_str() {
                "ctrlRing" => s.ctrl_ring = v.iter().map(|&x| x as f32).collect(),
                "routesFlat" => s.route_flat = v.iter().map(|&x| x as i32).collect(),
                "commuterWorkStart" => s.work_start = Some(v),
                _ => { s.arrays.insert(name.clone(), v); }
            }
        }
        Ok(s)
    }
    fn a(&self, k: &str) -> &[f64] { self.arrays.get(k).map(|v| v.as_slice()).unwrap_or(&[]) }
}

#[inline] fn fw(x: f64) -> u32 { (x as f32).to_bits() }
#[inline] fn iw(x: f64) -> u32 { jsmath::to_u32(x) }

impl GpuSim {
    fn packer_arc(&mut self) -> Arc<crate::pack::RoutePacker> { self.packer.get_or_insert_with(|| Arc::new(crate::pack::RoutePacker::new(&self.world.graph))).clone() }
    fn pack_one(&mut self, route: &[i32]) -> Vec<u32> { let p = self.packer_arc(); p.pack_routes(&self.world.graph, route, &[0, route.len() as u32]) }
    fn ensure_router(&mut self) {
        if self.router.is_none() { self.router = Some(bayenv_world::nav::Router::new(&self.world.graph, &bayenv_world::nav::RouterOpts { weight: 1.25, ..Default::default() })); }
    }
    fn resize_agents(&mut self, n: usize) {
        if n > self.n_agents { self.n_agents = n; self.gpu.storage("agents", (n * 16) as u64, None, wgpu::BufferUsages::empty()); self.make_bind_groups(); }
    }
    fn tank(ty: usize) -> f64 { VEHICLE_PARAMS[ty.min(3)].tank_l }

    // ------------------------------------------------------------------ population & spawning
    /// `setPopulation(pop, { stations })`: trip schedule, packed population routes, agents buffer.
    /// `stations`: plan refuels with the world's gas stations (else every low-fuel request fails, as without setStations).
    pub fn set_population(&mut self, pop: Arc<GpuPopulation>, stations: bool) {
        let packer = self.packer_arc();
        let packed = packer.pack_routes(&self.world.graph, &pop.route_flat, &pop.route_offset);
        self.route_words = 0; self.route_mirror.clear();
        self.ensure_routes(packed.len() + self.opts.route_slack);
        self.x.trip_route_base = self.append_route(&packed);
        self.x.trip_route_end = Some(self.route_words);
        let (n, off) = (pop.n, &pop.route_offset);
        self.resize_agents(n);
        let mut trips: Vec<i32> = Vec::new();
        for i in 0..n {
            if off[2 * i + 1] > off[2 * i] { trips.push(2 * i as i32); } else { self.counters.no_route += 1; }
            if self.opts.return_trips && pop.return_depart[i].is_finite() && off[2 * i + 2] > off[2 * i + 1] { trips.push(2 * i as i32 + 1); }
        }
        let dep = |k: i32| if k & 1 == 1 { pop.return_depart[(k >> 1) as usize] } else { pop.depart[(k >> 1) as usize] };
        trips.sort_by(|&a, &b| (dep(a) as f64 - dep(b) as f64).partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        self.x.trip_depart = trips.iter().map(|&k| dep(k)).collect();
        self.x.trips = trips; self.x.trip_ptr = 0;
        self.x.fuel_left = vec![-1.0; n]; self.x.outcome = vec![0; n];
        self.x.table = None; self.x.arena = None;
        self.x.pop = Some(pop);
        if stations { self.set_stations(); }
    }

    /// `setStations(world stations)`: StationNetwork (bayenv-sim) for refuel plans + the pump queues.
    pub fn set_stations(&mut self) {
        let w = self.world.clone(); let st = &w.stations;
        if st.is_empty() || !self.opts.features.fuel { return; }
        self.ensure_router();
        let n = st.len();
        let net = bayenv_sim::fuel::StationNetwork { cfg: Default::default(), node: st.iter().map(|s| s.node as i32).collect(), pumps: st.iter().map(|s| s.pumps as i32).collect(),
            x: st.iter().map(|s| s.x).collect(), z: st.iter().map(|s| s.z).collect(), busy: vec![0; n], served: vec![0; n], queue: vec![Vec::new(); n], max_queue: 0, planned: 0, plan_failed: 0 };
        self.x.stations = Some(Stations { net, busy: vec![0; n], served: vec![0; n], queue: vec![VecDeque::new(); n], max_queue: 0, wait_sum: 0.0, wait_n: 0 });
    }

    /// `setStreaming(table, { stations, deltas })`: commuters stream in at departure, routes from a RouteProvider.
    pub fn set_streaming(&mut self, table: Arc<CommuterTable>, stations: bool, deltas: Option<Arc<Vec<f32>>>) {
        self.x.stream_deltas = deltas;
        if self.x.provider.is_none() { self.x.provider = Some(RouteProvider::new(&self.world)); }
        let n = table.n;
        self.resize_agents(n);
        self.x.trips.clear(); self.x.trip_ptr = 0;
        self.x.table_ptr = table.index_at(self.time - 3600.0); self.x.pending_stream.clear();
        self.x.fuel_left = vec![-1.0; n]; self.x.outcome = vec![0; n];
        self.route_words = self.route_words.max(1); self.route_mirror.clear();
        self.x.arena = Some(Arena { free: HashMap::new(), slot: vec![-1; self.cap], cls: vec![0; self.cap] });
        self.x.stream_stats = StreamStats::default();
        self.x.pop = None; self.x.table = Some(table);
        if stations { self.set_stations(); }
    }

    fn arena_alloc(&mut self, words: &[u32]) -> (usize, u8) {
        let mut c = 6u32; while (1usize << c) < words.len() { c += 1; }
        let off = match self.x.arena.as_mut().unwrap().free.get_mut(&c).and_then(|f| f.pop()) {
            Some(o) => o as usize,
            None => { let o = self.route_words; self.ensure_routes(o + (1 << c)); self.route_words = o + (1 << c); o }
        };
        self.gpu.write("routes", bytemuck::cast_slice(words), (off * 4) as u64);
        if self.route_mirror.len() < self.route_words { self.route_mirror.resize(self.route_words + (1 << 16), 0); }
        self.route_mirror[off..off + words.len()].copy_from_slice(words);
        (off, c as u8)
    }
    pub(super) fn arena_free(&mut self, slot: u32) {
        let Some(a) = self.x.arena.as_mut() else { return };
        let s = slot as usize; if a.slot[s] < 0 { return; }
        let c = a.cls[s] as u32; a.free.entry(c).or_default().push(a.slot[s] as u32); a.slot[s] = -1;
    }

    /// Spawn-queue entry: VPar words + depart, flags, fuel, tank, slot.
    #[allow(clippy::too_many_arguments)]
    fn write_entry(&mut self, q: u32, slot: u32, ty: usize, tr: [f64; 6], route_off: u32, route_len: u32, work_start: f64, len: f64, trip: u32, depart: f64, fuel: f64, tank: f64) {
        let mut e = [0u32; L::SPAWN_WORDS];
        self.par_words(&mut e[..L::PAR_WORDS], ty as i64, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], route_off, route_len, work_start, len, trip);
        e[20] = fw(depart); e[21] = 1; e[22] = fw(fuel); e[23] = fw(tank);
        e[24] = slot; e[25] = if self.opts.features.policy { flag::POLICY } else { 0 };
        self.gpu.write("queue", bytemuck::cast_slice(&e), (q as usize * L::SPAWN_WORDS * 4) as u64);
        if q as usize + 1 > self.queue_high { self.queue_high = q as usize + 1; }
    }
    fn traits_of(d: Option<&[f32]>) -> [f64; 6] {
        match d { Some(d) => [d[0] as f64, d[1] as f64, d[2] as f64, d[3] as f64, d[4] as f64, d[5] as f64],
            None => [dt::AGGRESSION, dt::PATIENCE, dt::REACTION_TIME, dt::HEADWAY, dt::MAX_ACCEL, dt::MAX_DECEL] }
    }

    /// `_enqueueSpawns(tEnd)`: trips departing by tEnd + 2 s get a slot and a spawn-queue entry (pool / queue full: wait).
    pub fn enqueue_spawns(&mut self, t_end: f64) {
        if self.x.table.is_some() { return self.enqueue_stream(t_end); }
        let Some(pop) = self.x.pop.clone() else { return };
        let g = self.world.clone(); let g = &g.graph;
        while self.x.trip_ptr < self.x.trips.len() && self.x.trip_depart[self.x.trip_ptr] as f64 <= t_end + 2.0 {
            if self.free.is_empty() || self.queue_free.is_empty() { break; }
            let k = self.x.trips[self.x.trip_ptr]; let dep = self.x.trip_depart[self.x.trip_ptr] as f64; self.x.trip_ptr += 1;
            let id = (k >> 1) as usize;
            let slot = self.free.pop().unwrap(); let q = self.queue_free.pop().unwrap();
            let (roff, rlen) = (pop.route_offset[k as usize] as usize, (pop.route_offset[k as usize + 1] - pop.route_offset[k as usize]) as usize);
            let tr = Self::traits_of(pop.deltas.as_ref().map(|d| &d[id * 64..id * 64 + 6]));
            let ty = pop.vehicle_type[id] as usize;
            let mut len = 0.0; for &e in &pop.route_flat[roff..roff + rlen] { len += g.length[e as usize]; }
            let tank = Self::tank(ty);
            let fuel = if self.x.fuel_left[id] >= 0.0 { self.x.fuel_left[id] as f64 } else { pop.fuel_fill.as_ref().map_or(1.0, |f| f[id] as f64) * tank };
            let ro = (self.x.trip_route_base + roff) as u32;
            self.write_entry(q, slot, ty, tr, ro, rlen as u32, pop.work_start[id] as f64, len, k as u32, dep, fuel, tank);
            self.slot_trip[slot as usize] = k; self.slot_route[2 * slot as usize] = ro; self.slot_route[2 * slot as usize + 1] = rlen as u32;
        }
    }

    fn enqueue_stream(&mut self, t_end: f64) {
        let t = self.x.table.clone().unwrap();
        if !self.x.pending_stream.is_empty() {
            let pend = std::mem::take(&mut self.x.pending_stream); let mut keep = Vec::new();
            for id in pend { if !self.place_stream(&t, id) { keep.push(id); } }
            self.x.pending_stream = keep;
        }
        while self.x.table_ptr < t.n && t.sorted_depart[self.x.table_ptr] as f64 <= t_end + 2.0 {
            let id = t.order[self.x.table_ptr]; self.x.table_ptr += 1;
            if !self.place_stream(&t, id) { self.x.pending_stream.push(id); }
        }
    }
    fn place_stream(&mut self, t: &CommuterTable, id: u32) -> bool {
        let i = id as usize;
        if self.free.is_empty() || self.queue_free.is_empty() { self.x.stream_stats.pool_full += 1; return false; }
        let r = self.x.provider.as_mut().unwrap().get(t.home[i], t.work[i]);
        let Some(r) = r.filter(|r| !r.is_empty()) else { self.counters.no_route += 1; self.x.outcome[i] = 5; self.event("noRoute", -1, id as f64, 0.0); return true; };
        self.x.stream_stats.routed += 1;
        let words = self.pack_one(&r);
        let slot = self.free.pop().unwrap(); let q = self.queue_free.pop().unwrap();
        let (off, c) = self.arena_alloc(&words);
        { let a = self.x.arena.as_mut().unwrap(); a.slot[slot as usize] = off as i32; a.cls[slot as usize] = c; }
        let delta: Vec<f32> = match &self.x.stream_deltas { Some(d) => d[i * 64..i * 64 + 64].to_vec(), None => t.delta_for(i).to_vec() };
        let tr = Self::traits_of(Some(&delta));
        let ty = t.vehicle_type[i] as usize;
        let g = &self.world.graph; let mut len = 0.0; for &e in r.iter() { len += g.length[e as usize]; }
        let tank = Self::tank(ty);
        self.write_entry(q, slot, ty, tr, off as u32, r.len() as u32, t.work_start[i] as f64, len, 2 * id, t.depart[i] as f64, t.fuel_fill[i] as f64 * tank, tank);
        self.slot_trip[slot as usize] = 2 * id as i32; self.slot_route[2 * slot as usize] = off as u32; self.slot_route[2 * slot as usize + 1] = r.len() as u32;
        true
    }

    /// ARRIVE event: leave the station lot bookkeeping, release with the fuel left.
    pub(super) fn arrive(&mut self, slot: u32, fuel: f64) {
        if self.lot > 0 && self.slot_station[slot as usize] >= 0 && self.x.in_lot.remove(&slot) { self.lot -= 1; }
        self.release_fuel(slot, "arrived", fuel);
    }
    pub(super) fn mark_outcome(&mut self, slot: u32, code: u8) {
        let tr = self.slot_trip[slot as usize];
        if self.x.has_pop() && tr >= 0 && tr & 1 == 0 && self.x.outcome[(tr >> 1) as usize] == 0 { self.x.outcome[(tr >> 1) as usize] = code; }
    }

    // ------------------------------------------------------------------ fuel: plans, lots, pumps
    /// `_processRefuels`: low-fuel requests -> StationNetwork.plan within the per-frame budget.
    pub(super) fn process_refuels(&mut self) {
        if self.refuel_queue.is_empty() { return; }
        if self.x.stations.is_none() { for (slot, _) in std::mem::take(&mut self.refuel_queue) { self.op(op::FUEL_PLAN_FAILED, slot, 0, 0); } return; }
        let budget = self.x.refuel_budget_ms.unwrap_or(self.opts.refuel_budget_ms);
        let t0 = std::time::Instant::now();
        let mut q: VecDeque<(u32, u32)> = std::mem::take(&mut self.refuel_queue).into();
        while !q.is_empty() && t0.elapsed().as_secs_f64() * 1e3 < budget {
            let (slot, edge) = q.pop_front().unwrap();
            if self.slot_trip[slot as usize] < 0 { continue; }
            let route = self.route_of(slot).to_vec();
            if route.is_empty() { self.op(op::FUEL_PLAN_FAILED, slot, 0, 0); continue; }
            let g = &self.world.graph;
            let goal = g.to[(route[route.len() - 1] & 0xFFFFF) as usize] as i32; let from = g.to[edge as usize] as i32;
            if from == goal { self.op(op::FUEL_PLAN_FAILED, slot, 0, 0); continue; }
            self.ensure_router();
            let world = self.world.clone();
            let plan = { let st = self.x.stations.as_mut().unwrap(); let mut r = NavRef(self.router.as_mut().unwrap()); st.net.plan(&world.graph, &mut r, from, goal) };
            let trip = self.slot_trip[slot as usize];
            self.apply_refuel(slot, trip, edge, plan);
        }
        self.refuel_queue = q.into();
    }
    /// `_applyRefuel`: SET_ROUTE_REFUEL with [stop index + 1, total length, route words] (or FUEL_PLAN_FAILED).
    fn apply_refuel(&mut self, slot: u32, trip: i32, edge: u32, plan: Option<bayenv_sim::fuel::Plan>) {
        if self.slot_trip[slot as usize] != trip { return; }
        let Some(plan) = plan else { self.counters.refuel_plan_failed += 1; self.op(op::FUEL_PLAN_FAILED, slot, 0, 0); return; };
        let mut edges = Vec::with_capacity(plan.edges.len() + 1); edges.push(edge as i32); edges.extend_from_slice(&plan.edges);
        let g = &self.world.graph; let mut total = 0.0; for &e in &edges { total += g.length[e as usize]; }
        let words = self.pack_one(&edges);
        let mut block = Vec::with_capacity(words.len() + 2); block.push(plan.stop_index as u32 + 1); block.push(f32bits(total)); block.extend_from_slice(&words);
        let off = self.append_route(&block) + 2;
        self.op(op::SET_ROUTE_REFUEL, slot, off as u32, edges.len() as u32);
        self.slot_route[2 * slot as usize] = off as u32; self.slot_route[2 * slot as usize + 1] = edges.len() as u32;
        self.slot_station[slot as usize] = plan.station as i32;
        self.counters.refuel_planned += 1;
        self.event("refuelPlanned", slot as i64, plan.station as f64, plan.detour_time);
    }
    /// `_enterStation`: the car reached its station stop: FIFO pump queue (no stations / no plan: pumped at once).
    pub(super) fn enter_station(&mut self, slot: u32) {
        self.lot += 1; self.x.in_lot.insert(slot);
        let k = self.slot_station[slot as usize];
        if self.x.stations.is_none() || k < 0 { self.op(op::PUMPED, slot, f32bits(self.time), 0); return; }
        let st = self.x.stations.as_mut().unwrap(); let k = k as usize;
        st.queue[k].push_back((slot, self.time));
        if st.queue[k].len() > st.max_queue { st.max_queue = st.queue[k].len(); }
        let ql = st.queue[k].len();
        self.event("refuelArrive", slot as i64, k as f64, ql as f64);
    }
    /// `_pumps`: finished pumps -> PUMPED; free pumps take the next queued car for U(pumpMin, pumpMax) s.
    pub(super) fn pumps(&mut self) {
        if self.x.stations.is_none() || self.lot == 0 { return; }
        let pumping = std::mem::take(&mut self.x.pumping); let mut keep = Vec::new();
        for p in pumping {
            if self.time >= p.until {
                { let st = self.x.stations.as_mut().unwrap(); st.busy[p.station] -= 1; st.served[p.station] += 1; }
                self.counters.refuelled += 1; self.op(op::PUMPED, p.slot, f32bits(self.time), 0); self.event("refuelled", p.slot as i64, p.station as f64, 0.0);
            } else { keep.push(p); }
        }
        self.x.pumping = keep;
        let (pmin, pmax) = { let c = &self.x.stations.as_ref().unwrap().net.cfg; (c.pump_min, c.pump_max) };
        let ns = self.x.stations.as_ref().unwrap().busy.len();
        for k in 0..ns {
            loop {
                let st = self.x.stations.as_mut().unwrap();
                if st.queue[k].is_empty() || st.busy[k] >= st.net.pumps[k] { break; }
                let (slot, since) = st.queue[k].pop_front().unwrap(); st.busy[k] += 1;
                st.wait_sum += self.time - since; st.wait_n += 1;
                let until = self.time + pmin + (pmax - pmin) * self.rand();
                self.x.pumping.push(Pump { slot, station: k, until });
            }
        }
    }

    // ------------------------------------------------------------------ obstacles, wrecks
    fn lane_w(&self, e: usize, s: f64) -> f64 { self.world.lanes.lane_width(e, s) }

    /// writeCar of placeObstacles: a car on `route` at (e, s, lat); returns the slot (None: no free slot).
    #[allow(clippy::too_many_arguments)]
    fn write_car(&mut self, e: usize, route: &[i32], s: f64, lat: f64, conn: i64, st: u32, speed: f64, max_speed: Option<f64>, ty: u32) -> Option<u32> {
        let slot = self.free.pop()?;
        let words = self.pack_one(route);
        let route_off = self.append_route(&words);
        let g = self.world.clone(); let g = &g.graph;
        let mut total = 0.0; for &x in route { total += g.length[x as usize]; }
        let mut par = [0u32; L::PAR_WORDS];
        let tr = Self::traits_of(None);
        self.par_words(&mut par, ty as i64, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], route_off as u32, route.len() as u32, 0.0, total, 0xFFFF_FFFE);
        if let Some(m) = max_speed { par[6] = fw(m); par[19] = fw(m); }
        let lanes = g.lanes[e] as f64;
        let ln = jsmath::max(0.0, jsmath::min(lanes - 1.0, (lat / self.lane_w(e, jsmath::min(g.length[e], jsmath::max(0.0, s)))).floor())) as u32;
        let mut v = [0u32; L::VEH_WORDS];
        v[0] = e as u32; v[1] = fw(s); v[2] = fw(speed); v[4] = fw(lat); v[7] = st; v[10] = flag::NEW | if st == state::CRASHED { flag::DISABLED } else { 0 };
        v[11] = self.step as u32; v[12] = fw(-1.0); v[14] = L::pack_lanes(ln, ln, ln, -1); v[15] = fw(-1e9); v[16] = fw(10.0); v[17] = fw(50.0); v[18] = fw(self.time);
        v[20] = NONE; v[21] = fw(1e9); v[23] = NONE; v[30] = fw(if conn >= 0 { conn as f64 + 1.0 } else { 0.0 }); v[28] = fw(total); v[29] = fw(self.time); v[31] = fw(1e9);
        self.gpu.write("veh", bytemuck::cast_slice(&v), (slot as usize * L::VEH_WORDS * 4) as u64);
        self.gpu.write("par", bytemuck::cast_slice(&par), (slot as usize * L::PAR_WORDS * 4) as u64);
        self.slot_trip[slot as usize] = -3; self.new_pose = true;
        Some(slot)
    }
    fn static_car(&mut self, e: usize, s: f64, lat: f64, conn: i64, ty: u32) -> Option<u32> {
        let slot = self.write_car(e, &[e as i32], s, lat, conn, state::CRASHED, 0.0, None, ty)?;
        let g = &self.world.graph;
        let ln = jsmath::max(0.0, jsmath::min(g.lanes[e] as f64 - 1.0, (lat / self.lane_w(e, jsmath::min(g.length[e], jsmath::max(0.0, s)))).floor())) as u32;
        self.wrecks.insert(slot, Wreck { lo: ln, hi: ln, ty: 0, edge: e as u32, s });
        self.in_crash.insert(slot);
        self.open_scene(&[slot], "obstacle", "wreck", 1, 1e9, 1e9, false, &[slot]);
        Some(slot)
    }

    /// `placeObstacles(list)`: stopped, disabled cars at s = clamp(at, 1, length - 1), lat = lat ?? (lane + 0.5) x lane width.
    pub fn place_static_obstacles(&mut self, list: &[StaticObstacle]) -> Vec<Placed> {
        let mut placed = Vec::new();
        for o in list {
            if o.edge < 0 || o.edge as usize >= self.world.graph.edge_count { continue; }
            let e = o.edge as usize; let len = self.world.graph.length[e];
            let s = jsmath::min(len - 1.0, jsmath::max(1.0, o.at));
            let lat = o.lat.unwrap_or_else(|| (o.lane.unwrap_or(0.0) + 0.5) * self.lane_w(e, jsmath::min(len, jsmath::max(0.0, s))));
            if let Some(slot) = self.static_car(e, s, lat, -1, o.ty) { placed.push(Placed { slot, kind: ObstacleKind::Static, edge: e as u32, s }); }
        }
        self.x.obstacles = placed.clone();
        placed
    }

    /// `placeObstacles(spec)`: training obstacles (same RNG, pools and placement as gpuSim.js / sim.js).
    pub fn place_obstacles(&mut self, spec: &ObstacleSpec) -> Vec<Placed> {
        let world = self.world.clone(); let g = &world.graph;
        let mut rng = bayenv_sim::rng::Rng::new(spec.seed);
        let lw = |e: usize, s: f64| world.lanes.lane_width(e, jsmath::min(g.length[e], jsmath::max(0.0, s)));
        let mut pools: HashMap<ObstacleKind, Vec<u32>> = HashMap::new();
        let mut pool = |k: ObstacleKind| -> Vec<u32> {
            let key = match k { ObstacleKind::Partial | ObstacleKind::Slow | ObstacleKind::Braking => ObstacleKind::Midblock, x => x };
            pools.entry(key).or_insert_with(|| {
                let mut out = Vec::new();
                for e in 0..g.edge_count {
                    match key {
                        ObstacleKind::Midblock => { if g.rank[e] >= 1 && g.length[e] >= 150.0 { out.push(e as u32); } }
                        ObstacleKind::RampEnd => { if g.gore.get(e).map_or(false, |&x| x >= 0.0) && g.length[e] > 30.0 { out.push(e as u32); } }
                        ObstacleKind::AfterCurve | ObstacleKind::AfterMerge => {
                            if g.length[e] < 90.0 { continue; }
                            let ins: Vec<usize> = g.in_of(g.from[e] as usize).iter().filter(|&&q| q as i32 != g.reverse[e]).map(|&q| q as usize).collect();
                            let strict = |a: usize| bayenv_world::hops::hop_map(&world.lane_graph, a, e).strict;
                            if key == ObstacleKind::AfterMerge && ins.len() >= 2 && ins.iter().all(|&a| strict(a)) { out.push(e as u32); }
                            if key == ObstacleKind::AfterCurve && ins.len() == 1 {
                                let d = g.heading_start[e] - g.heading_end[ins[0]]; let d = jsmath::atan2(jsmath::sin(d), jsmath::cos(d)).abs();
                                if d > 0.3 && strict(ins[0]) { out.push(e as u32); }
                            }
                        }
                        _ => {}
                    }
                }
                out
            }).clone()
        };
        let mut placed = Vec::new(); let mut brakes: Vec<(u32, f64)> = Vec::new();
        for n in 0..spec.count {
            if spec.kinds.is_empty() { break; }
            let kind = spec.kinds[n % spec.kinds.len()];
            let (slot, e, s): (Option<u32>, usize, f64);
            if kind == ObstacleKind::Junction {
                let Some(j) = world.junctions.as_ref() else { continue };
                let mc = j.move_count(); if mc == 0 { continue; }
                let mut m = rng.int(mc); let mut d = j.describe(m); let mut t = 0.5 * d.length as f64;
                for _ in 0..20 {
                    m = rng.int(mc); d = j.describe(m); let ee = d.in_edge as usize;
                    let (mut bt, mut bd) = (-1.0, 1e9); let to = g.to[ee] as usize;
                    let mut u = 0.0; while u <= d.length as f64 { let q = j.arc_point(m, u); let dd = jsmath::hypot(q.x - g.nx[to], q.z - g.nz[to]); if dd < bd { bd = dd; bt = u; } u += 0.5; }
                    t = bt; if bd < 4.0 { break; }
                }
                e = d.in_edge as usize; s = d.s_in as f64 + t;
                slot = self.static_car(e, s, (d.in_lane as f64 + 0.5) * lw(e, d.s_in as f64), m as i64, 0);
            } else {
                let p = pool(kind); if p.is_empty() { continue; }
                e = p[rng.int(p.len())] as usize;
                let len = g.length[e]; let lanes = g.lanes[e] as usize;
                let lane = rng.int(lanes) as f64; let w = lw(e, len / 2.0);
                match kind {
                    ObstacleKind::Midblock => { s = len * rng.range(0.3, 0.7); slot = self.static_car(e, s, (lane + 0.5) * w, -1, 0); }
                    ObstacleKind::AfterCurve => { s = jsmath::min(len - 10.0, rng.range(5.0, 30.0)); slot = self.static_car(e, s, (lane + 0.5) * lw(e, s), -1, 0); }
                    ObstacleKind::AfterMerge => { s = jsmath::min(len - 10.0, rng.range(30.0, 80.0)); slot = self.static_car(e, s, (lane + 0.5) * lw(e, s), -1, 0); }
                    ObstacleKind::RampEnd => { s = jsmath::max(5.0, len - rng.range(3.0, 12.0)); slot = self.static_car(e, s, 0.5 * lw(e, s), -1, 0); }
                    ObstacleKind::Partial => { s = len * rng.range(0.3, 0.7); let k = if lanes > 1 { 1 + rng.int(lanes - 1) } else { rng.int(2) } as f64; let lat = k * lw(e, s) + rng.range(-0.4, 0.4); slot = self.static_car(e, s, lat, -1, 0); }
                    _ => {
                        s = len * rng.range(0.1, 0.5);
                        let mut route = vec![e as i32]; let mut cur = e;
                        for _ in 0..30 { let outs: Vec<u32> = g.out_of(g.to[cur] as usize).iter().copied().filter(|&o| o as i32 != g.reverse[cur]).collect(); if outs.is_empty() { break; } cur = outs[rng.int(outs.len())] as usize; route.push(cur as i32); }
                        if kind == ObstacleKind::Slow { let vmax = rng.range(spec.slow_kmh[0], spec.slow_kmh[1]) / 3.6; slot = self.write_car(e, &route, s, (lane + 0.5) * lw(e, s), -1, state::DRIVING, vmax, Some(vmax), 0); }
                        else {
                            slot = self.write_car(e, &route, s, (lane + 0.5) * lw(e, s), -1, state::DRIVING, jsmath::min(g.speed_limit[e], 15.0), None, 0);
                            if let Some(sl) = slot { brakes.push((sl, self.time + rng.range(spec.brake_every[0], spec.brake_every[1]))); }
                        }
                    }
                }
            }
            if let Some(slot) = slot { placed.push(Placed { slot, kind, edge: e as u32, s }); }
        }
        for (slot, next) in brakes {
            let mut r = [0u32; L::CTL_WORDS];
            r[0] = NONE; r[1] = NONE; r[5] = fw(1e9); r[7] = fw(1e9); r[9] = NONE; r[16] = NONE; r[20] = NONE;
            r[8] = fw(self.time); r[24] = fw(next); r[25] = fw(-1.0); r[26] = fw(spec.brake_every[0]); r[27] = fw(spec.brake_every[1]);
            self.gpu.write("ctl", bytemuck::cast_slice(&r), (slot as usize * L::CTL_WORDS * 4) as u64);
        }
        self.x.obstacles = placed.clone();
        placed
    }

    /// `placeWrecks(n)`: n static wrecks on random main-tier edges (rank >= 3, longer than 150 m) for the whole day.
    pub fn place_wrecks(&mut self, n: usize) -> usize {
        let world = self.world.clone(); let g = &world.graph;
        let cand: Vec<usize> = (0..g.edge_count).filter(|&e| g.rank[e] >= 3 && g.length[e] > 150.0).collect();
        let mut placed = 0;
        let mut k = 0;
        while k < n && !cand.is_empty() && !self.free.is_empty() {
            k += 1;
            let ei = cand[(self.rand() * cand.len() as f64).floor() as usize];
            let slot = self.free.pop().unwrap();
            let words = self.pack_one(&[ei as i32]); let route_off = self.append_route(&words);
            let mut par = [0u32; L::PAR_WORDS]; let tr = Self::traits_of(None);
            self.par_words(&mut par, 0, tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], route_off as u32, 1, 0.0, g.length[ei], 0xFFFF_FFFE);
            let lane = (self.rand() * g.lanes[ei] as f64).floor() as u32; let s_pos = g.length[ei] * (0.3 + 0.4 * self.rand());
            let mut v = [0u32; L::VEH_WORDS];
            v[0] = ei as u32; v[1] = fw(s_pos); v[4] = fw((lane as f64 + 0.5) * world.lanes.lane_width(ei, s_pos)); v[7] = state::CRASHED; v[10] = flag::NEW | flag::DISABLED;
            v[11] = self.step as u32; v[12] = fw(-1.0); v[14] = L::pack_lanes(lane, lane, lane, -1); v[15] = fw(-1e9); v[16] = fw(10.0); v[17] = fw(50.0); v[18] = fw(self.time);
            v[20] = NONE; v[21] = fw(1e9); v[23] = NONE; v[28] = fw(g.length[ei]); v[29] = fw(self.time); v[31] = fw(1e9);
            self.gpu.write("veh", bytemuck::cast_slice(&v), (slot as usize * L::VEH_WORDS * 4) as u64);
            self.gpu.write("par", bytemuck::cast_slice(&par), (slot as usize * L::PAR_WORDS * 4) as u64);
            self.slot_trip[slot as usize] = -3;
            self.wrecks.insert(slot, Wreck { lo: lane, hi: lane, ty: 0, edge: ei as u32, s: s_pos });
            self.in_crash.insert(slot);
            self.open_scene(&[slot], "wreck", "wreck", 1, 1e9, 1e9, false, &[slot]);
            placed += 1;
        }
        self.new_pose = true;
        placed
    }

    // ------------------------------------------------------------------ snapshot import, reset, agents
    /// `importSnapshot(snap, commuters)`: a CPU snapshot slot for slot (incl. the reaction-time rings) and its clock.
    pub fn import_snapshot(&mut self, snap: &Snapshot) -> Result<(), String> {
        let (cap, high) = (self.cap, snap.high);
        if high > cap { return Err(format!("capacity {cap} < cpu high water {high}")); }
        let (vw, pw) = (L::VEH_WORDS, L::PAR_WORDS);
        let mut vehw = vec![0u32; cap * vw]; let mut parw = vec![0u32; cap * pw];
        let a = |k: &str| snap.a(k);
        let (st, ro_a, rl_a, edge, com) = (a("state"), a("routeOff"), a("routeLen"), a("edge"), a("commuter"));
        let packer = self.packer_arc(); let g = self.world.clone(); let g = &g.graph;
        let mut all: Vec<u32> = Vec::new();
        let get = |k: &str, i: usize| -> f64 { snap.arrays.get(k).and_then(|v| v.get(i)).copied().unwrap_or(f64::NAN) };
        let has_vlat = snap.arrays.contains_key("vLat"); let has_sl = snap.arrays.contains_key("steerSl");
        for i in 0..high {
            if st[i] == 0.0 { continue; }
            let (ro, rl) = (ro_a[i], rl_a[i]);
            let route: Vec<i32> = if ro >= 0.0 { snap.route_flat[ro as usize..(ro + rl) as usize].to_vec() } else { vec![edge[i] as i32] };
            let words = packer.pack_routes(g, &route, &[0, route.len() as u32]);
            let r_off = all.len() as u32; all.extend_from_slice(&words);
            let ws = match (&snap.work_start, com[i] >= 0.0) { (Some(w), true) => w.get(com[i] as usize).copied(), _ => None };
            let v = &mut vehw[i * vw..(i + 1) * vw]; let p = &mut parw[i * pw..(i + 1) * pw];
            let f = |k: &str| get(k, i);
            v[0] = iw(f("edge")); v[1] = fw(f("s")); v[2] = fw(f("speed")); v[3] = fw(f("accel"));
            v[4] = fw(f("lat")); v[5] = fw(f("headingRel")); v[6] = fw(f("steerAngle")); v[7] = iw(f("state"));
            v[8] = iw(f("routePos")); v[9] = iw(f("seg")); let fl = f("flags") as u32; v[10] = fl & !32; v[11] = iw(f("cmdStep"));
            v[12] = fw(f("stoppedSince")); v[13] = fw(f("odometer"));
            let (ln, lo, hi, tl) = (f("lane") as i32, f("laneLo") as i32, f("laneHi") as i32, f("targetLane") as i32);
            v[14] = if hi < lo { L::pack_lanes(ln as u32, (ln + 1) as u32, ln as u32, tl) } else { L::pack_lanes(ln as u32, lo as u32, hi as u32, tl) };
            // (gpuSim.js leaves word 15, lastDecision, at 0: its assignment sits inside a comment)
            v[16] = fw(f("fuel")); v[17] = fw(f("tank")); v[18] = fw(f("spawnTime")); v[19] = fw(f("routeDist"));
            v[20] = iw(f("leader")); v[21] = fw(f("gap")); v[22] = fw(f("pumpUntil")); v[23] = iw(f("stopPos"));
            v[24] = iw(f("refuelState")); v[25] = fw(if has_sl { f("steerSl") } else { f("steerAngle") }); v[26] = fw(f("gas")); v[27] = fw(f("brake"));
            v[28] = fw(f("routeTotal")); v[29] = fw(f("lastFuelCheck")); v[30] = fw(f("taper")); v[31] = fw(f("laneEndDist")); v[32] = iw(f("laneEndSide"));
            if has_vlat { v[33] = fw(f("vLat")); v[34] = fw(f("yawRate")); v[35] = fw(f("axPrev")); v[36] = fw(f("px")); v[37] = fw(f("pz")); v[38] = fw(f("pyaw")); v[39] = fw(f("power")); }
            if fl & 32 != 0 { v[10] |= flag::DISABLED; }
            let ty = f("typeIdx");
            p[0] = r_off; p[1] = route.len() as u32; p[2] = iw(ty); p[3] = iw(f("delaySteps"));
            p[4] = fw(f("length")); p[5] = fw(f("width")); p[6] = fw(f("maxSpeed")); p[7] = fw(f("maxAccel"));
            p[8] = fw(f("maxDecel")); p[9] = fw(f("headwayPref")); p[10] = fw(f("aggression")); p[11] = fw(f("patience"));
            p[12] = fw(f("reactionTime")); p[13] = fw(ws.unwrap_or(0.0)); p[14] = fw(f("routeTotal")); p[15] = iw(jsmath::max(0.0, com[i]) * 2.0 + f("leg"));
            p[16] = fw(f("wheelbase")); p[17] = fw(f("maxSteer")); p[18] = fw(f("maxSteerRate")); p[19] = fw(VEHICLE_PARAMS[if (0.0..4.0).contains(&ty) { ty as usize } else { 0 }].vmax);
            self.slot_trip[i] = (com[i] * 2.0 + f("leg")) as i32;
        }
        if all.is_empty() { all.push(0); }
        self.route_words = 0; self.route_mirror.clear(); self.ensure_routes(all.len() + 1024); self.append_route(&all);
        let mut ring = vec![0f32; cap * L::CTRL_RING * L::CTRL_STRIDE];
        let m = snap.ctrl_ring.len().min(ring.len()); ring[..m].copy_from_slice(&snap.ctrl_ring[..m]);
        self.time = snap.time; self.step = snap.step; self.dt = snap.dt;
        self.gpu.write("veh", bytemuck::cast_slice(&vehw), 0);
        self.gpu.write("par", bytemuck::cast_slice(&parw), 0);
        self.gpu.write("ring", bytemuck::cast_slice(&ring), 0);
        let mut pose = vec![0f32; cap * 4];
        let (x, y, z, h) = (a("x"), a("y"), a("z"), a("heading"));
        for i in 0..cap { if i >= high || st[i] == 0.0 { pose[i * 4 + 1] = -1000.0; continue; } pose[i * 4] = x[i] as f32; pose[i * 4 + 1] = y[i] as f32; pose[i * 4 + 2] = z[i] as f32; pose[i * 4 + 3] = h[i] as f32; }
        self.gpu.write("pose", bytemuck::cast_slice(&pose), 0);
        let mut iow = vec![0u32; L::io::EVDATA]; iow[0] = self.step as u32; self.gpu.write("io", bytemuck::cast_slice(&iow), 0);
        self.free = (0..cap as u32).rev().filter(|&i| i as usize >= high || st[i as usize] == 0.0).collect();
        Ok(())
    }

    /// `resetDay(startTime)`: a new simulated day with the same world / population / capacity.
    pub fn reset_day(&mut self, start_time: Option<f64>) {
        self.sync();
        let start = start_time.unwrap_or(self.opts.start_time);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        for k in ["veh", "ring", "agents", "queue", "laneClaim", "claim", "telemetry", "ctl"] { enc.clear_buffer(&self.gpu.get(k), 0, None); }
        if self.opts.obs == crate::defines::ObsMode::V3 { enc.clear_buffer(&self.gpu.get("inputs"), (self.cap * self.obs_size * 4) as u64, Some((self.cap * L::LH_W * 4) as u64)); }
        enc.clear_buffer(&self.gpu.get("world"), (self.blob.off.eb as usize * 4) as u64, Some((self.blob.e * 4) as u64));
        self.gpu.queue.submit([enc.finish()]);
        let mut hidden = vec![0f32; self.cap * 4]; for j in (1..hidden.len()).step_by(4) { hidden[j] = -1000.0; }
        self.gpu.write("pose", bytemuck::cast_slice(&hidden), 0);
        self.time = start; self.step = 0;
        self.gpu.write("io", bytemuck::cast_slice(&vec![0u32; L::io::EVDATA]), 0);
        self.gpu_stats = [0; 64];
        self.free = (0..self.cap as u32).rev().collect();
        self.slot_trip.iter_mut().for_each(|x| *x = -1); self.slot_station.iter_mut().for_each(|x| *x = -1);
        self.queue_free = (0..self.opts.spawn_queue as u32).rev().collect();
        self.queue_high = 0; self.ops_pending.clear(); self.wrecks.clear(); self.in_crash.clear(); self.refuel_queue.clear(); self.x.pumping.clear(); self.lot = 0; self.x.in_lot.clear();
        self.crashes.clear(); self.active.clear(); self.edge_head.iter_mut().for_each(|x| *x = 0); self.block_next.iter_mut().for_each(|x| *x = 0);
        self.block_free = (0..self.opts.max_blocks as u32).rev().collect();
        self.counters = Default::default(); self.crash_x.clear();
        self.reroute_queue.clear(); self.last_reroute.clear(); self.closed_key = None; self.closed_n = 0; self.closed_edges.clear();
        self.edge_speed_ema.clear(); self.edge_speed_order.clear(); self.traffic_penalty.clear(); self.penalty_order.clear(); self.pending_penalty = None; self.next_traffic = -1e9;
        if let Some(t) = self.x.table.clone() {
            self.x.table_ptr = t.index_at(start - 3600.0); self.x.pending_stream.clear();
            self.x.fuel_left.iter_mut().for_each(|x| *x = -1.0); self.x.outcome.iter_mut().for_each(|x| *x = 0);
            self.x.arena = Some(Arena { free: HashMap::new(), slot: vec![-1; self.cap], cls: vec![0; self.cap] }); self.route_words = 1;
        } else if self.x.pop.is_some() {
            self.x.trip_ptr = 0; self.x.fuel_left.iter_mut().for_each(|x| *x = -1.0); self.x.outcome.iter_mut().for_each(|x| *x = 0);
            if let Some(e) = self.x.trip_route_end { self.route_words = e; }
            if let Some(st) = self.x.stations.as_mut() { st.busy.iter_mut().for_each(|x| *x = 0); st.queue.iter_mut().for_each(|q| q.clear()); }
        }
        self.events.clear();
    }

    /// `readAgents()`: per-agent reward sums, outcomes (outbound / return leg), progress, trip time, crashes.
    pub fn read_agents(&mut self) -> Agents {
        self.sync();
        let n = self.n_agents; let mut u = vec![0u32; n * 4];
        self.gpu.read_into("agents", 0, bytemuck::cast_slice_mut(&mut u));
        let mut o = Agents::default();
        for a in 0..n {
            o.reward_sum.push(f32::from_bits(u[4 * a])); o.outcome.push((u[4 * a + 1] & 255) as u8); o.outcome_back.push(((u[4 * a + 1] >> 8) & 255) as u8);
            o.progress.push(((u[4 * a + 1] >> 16) as f64 / 65535.0) as f32); o.trip_time.push(f32::from_bits(u[4 * a + 2])); o.crashes.push(u[4 * a + 3]);
        }
        o
    }
    pub fn n_agents(&self) -> usize { self.n_agents }
    // ---- host-step access (oracle checks / custom loops; submit_steps calls these itself)
    /// Process events as if read back from the GPU (`_ingest`): [type, slot, a, b] per event.
    pub fn ingest_events(&mut self, evs: &[[u32; 4]]) {
        let need = L::io::EVDATA + 4 * evs.len(); if self.io_host.len() < need { self.io_host.resize(need, 0); }
        self.io_host[L::io::EVCOUNT] = evs.len() as u32;
        for (k, e) in evs.iter().enumerate() { self.io_host[L::io::EVDATA + 4 * k..L::io::EVDATA + 4 * k + 4].copy_from_slice(e); }
        self.ingest();
    }
    /// Pending ops (code, slot, a, b), taken.
    pub fn take_ops(&mut self) -> Vec<u32> { std::mem::take(&mut self.ops_pending) }
    pub fn queue_refuel(&mut self, slot: u32, edge: u32) { self.refuel_queue.push((slot, edge)); }
    pub fn run_refuels(&mut self) { self.process_refuels(); }
    pub fn run_pumps(&mut self) { self.pumps(); }
    pub fn write_params_now(&mut self, op_count: usize) { self.write_params(op_count); }
    pub fn lot(&self) -> u32 { self.lot }
    pub fn slot_station(&self) -> &[i32] { &self.slot_station }
    pub fn queue_high(&self) -> usize { self.queue_high }
    pub fn free_len(&self) -> usize { self.free.len() }
    pub fn route_of_slot(&self, slot: u32) -> Vec<u32> { self.route_of(slot).to_vec() }
    pub fn edge_head(&self) -> &[u32] { &self.edge_head }
    pub fn scene_rows(&self) -> Vec<(u32, u32, f64, i32, f64, &'static str)> { self.active.iter().map(|&i| { let r = &self.crashes[i]; (r.edge, r.lane_mask, r.s, r.block, r.squeeze_at, r.cause) }).collect() }
    /// Trips not yet enqueued (population mode).
    pub fn trips_remaining(&self) -> usize { self.x.trips.len() - self.x.trip_ptr }
    pub fn stations(&self) -> Option<&Stations> { self.x.stations.as_ref() }
}
