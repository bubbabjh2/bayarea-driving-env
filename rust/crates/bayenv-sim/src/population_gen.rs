//! Population generation, bit-exact with the JS oracle:
//!  * src/sim/population.js `generatePopulation(graph, opts)` (SPEC commuter generator; mergeCfg defaults, _context,
//!    largestSCC (src/sim/route.js), graph.nearestNode (src/sim/roadGraph.js makeNearestNode), generateCommuter,
//!    deltaFor -> src/sim/traits.js makeGenome / mutate / clampDelta, assemble), and
//!  * src/scenarios/load.js `scenarioPopulation(graph, scenario, opts)` (scenario trip lists).
//!
//! Every commuter has its own RNG stream (mulberry32 seeded by hashSeed(`${seed}:${id}`), delta stream
//! `${seed}:delta:${id}`), so the result does not depend on chunking: popParallel.js (worker_threads, what env.js uses
//! for n >= 4000 on the real world) gives the same population as the serial generator, and so does the rayon split
//! here (per-thread routers, per-chunk results concatenated in id order).
//!
//! Routing goes through the [`Router`] trait (src/sim/route.js makeRouter). population.js routes with
//! `{ weight: cfg.astarWeight (1.25), cacheMax: 0 }`, load.js with `{ weight: 1, cacheMax: 0 }`; the cache never
//! changes a route (A* is deterministic per (start, goal)), only the weight matters. The default router is
//! bayenv_world::nav::router::Router through [`crate::route_stub::NavRouter`].
//!
//! The generator is generic over [`PopGraph`] (a [`PlanGraph`] supertrait), implemented for the sim's [`World`] and
//! for `bayenv_world::graph::SimGraph`, so the GPU host generates directly on the shared graph (no World copy).
//! Both impls expose the same values (SimGraph u32 / u8 integers widened), so output is bit-identical.
//!
//! Region filters (population.js regionFilter / REGION_PRESETS / mergeCfg preset `pop` overrides) restrict homes AND
//! workplaces (the `all` / `homes` pools and the nearestNode filter); see [`Region`]. JS function regions
//! (`region: (x, z) => bool`) are not representable; use [`Region::Bbox`] etc.
//!
//! Not ported (throw in JS-only paths / unused by env.js): function regions, `cfg.router`, `snapTolerance` overrides.
use crate::jm;
use crate::population::Population;
use crate::rng::{hash_seed, Rng};
use crate::sim::Router;
use crate::fuel::PlanGraph;
use crate::world::World;
use bayenv_world::graph::SimGraph;
use rayon::prelude::*;
use serde_json::Value;
use std::sync::Arc;

const H: f64 = 3600.0;
const MIN: f64 = 60.0;
pub const DELTA_SIZE: usize = 64;
const D_LEARN_START: usize = 6;
const D_LEARN_END: usize = 47;
/// traits.js TRAIT_RANGES in TRAIT_NAMES order (aggression, patience, reactionTime, preferredHeadway, maxAccel, maxDecel)
const TRAIT_RANGES: [(f64, f64); 6] = [(0.0, 1.0), (0.0, 1.0), (0.6, 1.8), (0.6, 3.0), (1.0, 4.5), (2.0, 9.0)];
const DEFAULT_TRAITS: [f64; 6] = [0.5, 0.5, 0.9, 1.4, 2.5, 5.0];
/// src/sim/vehicleTypes.js VEHICLE_TYPE_WEIGHTS / population.js REALISM_MIX.vehicleWeights
const TYPE_WEIGHTS: [f64; 4] = [0.62, 0.30, 0.06, 0.02];
const REALISM_TYPE_WEIGHTS: [f64; 4] = [0.452, 0.51, 0.035, 0.003];
const REALISM_BINS: [(f64, f64, f64); 6] = [(0.5, 5.0, 0.183), (5.0, 15.0, 0.282), (15.0, 30.0, 0.241), (30.0, 50.0, 0.125), (50.0, 75.0, 0.065), (75.0, 75.0, 0.104)];
const REALISM_P_CROSS: f64 = 0.05;
/// roadGraph.js HIGHWAY_RANK of the default homeClasses ['residential', 'tertiary', 'living_street']
pub const DEFAULT_HOME_RANKS: [i32; 3] = [1, 3, 0];
const R_EARTH: f64 = 6371008.8;
/// POPULATION_DEFAULTS.projection (lat0, lon0)
pub const DEFAULT_PROJECTION: (f64, f64) = (37.55, -122.15);

/// The graph queries population generation needs, on top of [`PlanGraph`] (edge from / to / length / speed, node x / z).
pub trait PopGraph: PlanGraph {
    fn node_count(&self) -> usize;
    fn edge_count(&self) -> usize;
    /// node x / z coordinates (length node_count)
    fn nx(&self) -> &[f64];
    fn nz(&self) -> &[f64];
    /// roadGraph.js HIGHWAY_RANK of the edge
    fn edge_rank(&self, e: usize) -> i32;
    /// CSR out adjacency: `out_edge(q)` for q in `out_range(n)`
    fn out_range(&self, n: usize) -> std::ops::Range<usize>;
    fn out_edge(&self, q: usize) -> usize;
    /// CSR in adjacency: `in_edge(q)` for q in `in_range(n)`
    fn in_range(&self, n: usize) -> std::ops::Range<usize>;
    fn in_edge(&self, q: usize) -> usize;
}
impl PopGraph for World {
    #[inline] fn node_count(&self) -> usize { self.node_count }
    #[inline] fn edge_count(&self) -> usize { self.edge_count }
    #[inline] fn nx(&self) -> &[f64] { &self.nx }
    #[inline] fn nz(&self) -> &[f64] { &self.nz }
    #[inline] fn edge_rank(&self, e: usize) -> i32 { self.e_rank[e] }
    #[inline] fn out_range(&self, n: usize) -> std::ops::Range<usize> { self.out_start[n] as usize..self.out_start[n + 1] as usize }
    #[inline] fn out_edge(&self, q: usize) -> usize { self.out_edges[q] as usize }
    #[inline] fn in_range(&self, n: usize) -> std::ops::Range<usize> { self.in_start[n] as usize..self.in_start[n + 1] as usize }
    #[inline] fn in_edge(&self, q: usize) -> usize { self.in_edges[q] as usize }
}
impl<G: PopGraph + ?Sized> PopGraph for Arc<G> {
    #[inline] fn node_count(&self) -> usize { (**self).node_count() }
    #[inline] fn edge_count(&self) -> usize { (**self).edge_count() }
    #[inline] fn nx(&self) -> &[f64] { (**self).nx() }
    #[inline] fn nz(&self) -> &[f64] { (**self).nz() }
    #[inline] fn edge_rank(&self, e: usize) -> i32 { (**self).edge_rank(e) }
    #[inline] fn out_range(&self, n: usize) -> std::ops::Range<usize> { (**self).out_range(n) }
    #[inline] fn out_edge(&self, q: usize) -> usize { (**self).out_edge(q) }
    #[inline] fn in_range(&self, n: usize) -> std::ops::Range<usize> { (**self).in_range(n) }
    #[inline] fn in_edge(&self, q: usize) -> usize { (**self).in_edge(q) }
}
impl PopGraph for SimGraph {
    #[inline] fn node_count(&self) -> usize { self.node_count }
    #[inline] fn edge_count(&self) -> usize { self.edge_count }
    #[inline] fn nx(&self) -> &[f64] { &self.nx }
    #[inline] fn nz(&self) -> &[f64] { &self.nz }
    #[inline] fn edge_rank(&self, e: usize) -> i32 { self.rank[e] as i32 }
    #[inline] fn out_range(&self, n: usize) -> std::ops::Range<usize> { self.out_start[n] as usize..self.out_start[n + 1] as usize }
    #[inline] fn out_edge(&self, q: usize) -> usize { self.out_edges[q] as usize }
    #[inline] fn in_range(&self, n: usize) -> std::ops::Range<usize> { self.in_start[n] as usize..self.in_start[n + 1] as usize }
    #[inline] fn in_edge(&self, q: usize) -> usize { self.in_edges[q] as usize }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Departures { Curve, Spec }
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mix { Spec, Realism }

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct P2 { pub x: f64, pub z: f64 }

/// population.js finishAxis result (SJ -> SF axis u, lateral v, bay centreline through `mid`).
#[derive(Clone, Copy, Debug)]
pub struct Axis { pub sj: P2, pub sf: P2, pub mid: P2, pub d: f64, pub u: P2, pub v: P2 }

impl Axis {
    pub fn finish(sj: P2, sf: P2, mid: P2) -> Axis {
        let (dx, dz) = (sf.x - sj.x, sf.z - sj.z);
        let d = jm::or(jm::hypot(dx, dz), 1.0);
        let u = P2 { x: dx / d, z: dz / d };
        let v = P2 { x: -u.z, z: u.x };
        Axis { sj, sf, mid, d, u, v }
    }
    /// makeAxis(cfg, graph.meta): SJ / SF / San Mateo Bridge midpoint projected with (lat0, lon0).
    pub fn from_cfg(cfg: &PopConfig, proj: (f64, f64)) -> Axis {
        Axis::finish(project_lat_lon(cfg.san_jose, proj), project_lat_lon(cfg.san_francisco, proj), project_lat_lon(cfg.san_mateo_bridge_mid, proj))
    }
    /// synthAxis(graph): long axis = -z, bay centreline x = mid of the bounds.
    pub fn synth<G: PopGraph + ?Sized>(w: &G) -> Axis {
        let (min_x, max_x, min_z, max_z) = node_bounds(w);
        Axis::finish(P2 { x: 0.0, z: max_z }, P2 { x: 0.0, z: min_z }, P2 { x: 0.5 * (min_x + max_x), z: 0.5 * (min_z + max_z) })
    }
    pub fn side_of(&self, x: f64, z: f64) -> u8 { if (x - self.mid.x) * self.v.x + (z - self.mid.z) * self.v.z > 0.0 { 1 } else { 0 } }
}

/// projectLatLon({lat, lon}, {lat0, lon0})
pub fn project_lat_lon((lat, lon): (f64, f64), (lat0, lon0): (f64, f64)) -> P2 {
    let kx = jm::cos(lat0 * jm::PI / 180.0) * R_EARTH * jm::PI / 180.0;
    let kz = R_EARTH * jm::PI / 180.0;
    P2 { x: (lon - lon0) * kx, z: -(lat - lat0) * kz }
}

/// Which axis generateCommuter uses (cfg.axis || makeAxis(cfg, graph.meta)).
#[derive(Clone, Copy, Debug)]
pub enum AxisSpec {
    /// makeAxis with the graph's meta.projection (lat0, lon0), or cfg.projection when None.
    Projection(Option<(f64, f64)>),
    /// synthAxis(graph) (tools/rust/trace.mjs synthetic world)
    Synth,
    Explicit(Axis),
}

/// mergeCfg(options) with POPULATION_DEFAULTS (population.js). Field names follow the JS keys.
#[derive(Clone, Debug)]
pub struct PopConfig {
    pub n: usize,
    pub seed: f64,
    pub local_share: f64,
    pub local_median_m: f64, pub local_sigma: f64, pub local_min_m: f64, pub local_max_m: f64,
    pub long_min_m: f64, pub long_max_m: f64, pub long_lateral_sigma_m: f64, pub long_p_cross: f64,
    pub max_attempts: usize,
    pub min_route_ratio: f64,
    pub work_start_mean: f64, pub work_start_sd: f64, pub work_start_min: f64, pub work_start_max: f64,
    pub travel_speed_factor: f64,
    pub slack_min: f64, pub slack_max: f64,
    pub rt_median: f64, pub rt_p95: f64, pub rt_min: f64, pub rt_max: f64,
    pub work_duration_mean: f64, pub work_duration_sd: f64,
    pub fuel_fill_min: f64, pub fuel_fill_max: f64,
    pub genome_sigma: f64, pub genome_trait_sigma: f64,
    pub return_trips: bool,
    /// cfg.projection (lat0, lon0): used by makeAxis when the graph has no meta.projection
    pub projection: (f64, f64),
    pub san_jose: (f64, f64), pub san_francisco: (f64, f64), pub san_mateo_bridge_mid: (f64, f64),
    /// HIGHWAY_RANK of cfg.homeClasses
    pub home_ranks: Vec<i32>,
    pub astar_weight: f64,
    pub departures: Departures,
    pub curve_start: f64, pub curve_bin: f64, pub curve_share: Vec<f64>,
    pub mix: Mix,
    pub axis: AxisSpec,
    /// cfg.baseGenome (delta, clamped by makeGenome); None = makeGenome() defaults
    pub base_delta: Option<Vec<f32>>,
    /// traits.js DELTA_NORM_CAP (setDeltaNormCap)
    pub delta_norm_cap: f64,
    /// options.range: commuter ids [a, b) (default [0, n))
    pub range: Option<(usize, usize)>,
    /// cfg.region: restrict homes and workplaces (None = whole graph). Set it through [`PopConfig::set_region`]
    /// (or apply_json) so that preset `pop` overrides (e.g. `bridges`) are applied like mergeCfg does.
    pub region: Option<Region>,
}

/// population.js cfg.region (regionFilter): homes AND workplaces must satisfy it (routes may leave it).
#[derive(Clone, Debug, PartialEq)]
pub enum Region {
    /// `{ bbox: [minX, minZ, maxX, maxZ] }` in graph x / z
    Bbox([f64; 4]),
    /// `{ side: 0 | 1 }`: bay side (axis.sideOf), 0 = Peninsula / west
    Side(u8),
    /// `{ preset: key }`: REGION_PRESETS key (sf, pen, sj, eb, ebs, marin, bridges, hills, wsj, bay) or the legacy
    /// 'peninsula-101' / 'eastbay-880'. 'bay' = no restriction.
    Preset(String),
}

/// population.js REGION_PRESETS boxes: [latMin, latMax, lonMin, lonMax] (None for 'bay'; 'sf' / 'wsj' / legacy presets
/// use their x / z definitions in regionFilter and these boxes are only descriptive for sf / wsj).
pub fn region_preset_boxes(key: &str) -> Option<&'static [[f64; 4]]> {
    Some(match key {
        "sf" => &[[37.712, 37.838, -122.499, -122.340]],
        "pen" => &[[37.43, 37.66, -122.46, -122.13]],
        "sj" => &[[37.25, 37.43, -122.10, -121.80]],
        "eb" => &[[37.72, 37.90, -122.32, -122.14]],
        "ebs" => &[[37.48, 37.70, -122.14, -121.90]],
        "marin" => &[[37.83, 37.97, -122.56, -122.44], [37.77, 37.81, -122.52, -122.40]],
        "bridges" => &[[37.77, 37.84, -122.42, -122.26], [37.53, 37.65, -122.33, -122.06], [37.44, 37.54, -122.20, -121.99]],
        "hills" => &[[37.38, 37.53, -122.40, -122.22], [37.25, 37.30, -122.08, -121.93], [37.80, 37.90, -122.25, -122.16], [37.30, 37.42, -121.84, -121.80]],
        "wsj" => &[[37.223, 37.353, -122.044, -121.884]],
        _ => return None,
    })
}
/// REGION_PRESETS keys (population.js order)
pub const REGION_PRESET_KEYS: [&str; 10] = ["sf", "pen", "sj", "eb", "ebs", "marin", "bridges", "hills", "wsj", "bay"];
/// population.js BREAKDOWN_REGIONS
pub const BREAKDOWN_REGIONS: [&str; 8] = ["sf", "marin", "eb", "bridges", "pen", "ebs", "hills", "sj"];
/// REGION_PRESETS[key].pop (mergeCfg preset overrides, explicit options win), as generatePopulation options JSON.
pub fn region_preset_pop(key: &str) -> Option<Value> {
    match key { "bridges" => Some(serde_json::json!({ "localShare": 0.9, "local": { "medianM": 9000, "minM": 3000, "maxM": 20000 } })), _ => None }
}

impl Region {
    /// Parse a JS region object ({ preset } | { bbox } | { side }; checked in regionFilter's order: bay, bbox, side, preset).
    pub fn from_json(v: &Value) -> Result<Option<Region>, String> {
        if v.is_null() { return Ok(None); }
        let preset = v.get("preset").and_then(|p| p.as_str());
        if preset == Some("bay") { return Ok(None); }
        if let Some(b) = v.get("bbox").and_then(|b| b.as_array()) {
            if b.len() != 4 { return Err(format!("region bbox needs 4 numbers: {v}")); }
            let g = |i: usize| b[i].as_f64().ok_or_else(|| format!("region bbox: {v}"));
            return Ok(Some(Region::Bbox([g(0)?, g(1)?, g(2)?, g(3)?])));
        }
        if let Some(s) = v.get("side").filter(|s| !s.is_null()) { return Ok(Some(Region::Side(s.as_f64().ok_or("region side")? as u8))); }
        match preset {
            Some(p) if matches!(p, "peninsula-101" | "eastbay-880") || region_preset_boxes(p).is_some() => Ok(Some(Region::Preset(p.to_string()))),
            _ => Err(format!("unknown region {v}")),
        }
    }
    pub fn preset(key: &str) -> Region { Region::Preset(key.to_string()) }
}

/// regionFilter(region, axis) resolved: (x, z) -> inside.
#[derive(Clone, Debug)]
pub enum RegionFilter {
    Bbox([f64; 4]),
    Side(u8, Axis),
    /// peninsula-101 (side 0) / eastbay-880 (side 1): |signed distance from the bay centreline| < 8 km
    Strip(u8, Axis),
    /// 'sf': 14 km box around axis.sf
    SfBox(P2),
    /// lat/lon boxes converted to [minX, minZ, maxX, maxZ] (latLonBoxesFilter; 'wsj' is its literal x / z box)
    Boxes(Vec<[f64; 4]>),
}
impl RegionFilter {
    /// `proj`: axis.proj || POPULATION_DEFAULTS.projection
    pub fn new(region: &Region, axis: &Axis, proj: (f64, f64)) -> Result<Option<RegionFilter>, String> {
        Ok(Some(match region {
            Region::Bbox(b) => RegionFilter::Bbox(*b),
            Region::Side(s) => RegionFilter::Side(*s, *axis),
            Region::Preset(p) => match p.as_str() {
                "bay" => return Ok(None),
                "peninsula-101" => RegionFilter::Strip(0, *axis),
                "eastbay-880" => RegionFilter::Strip(1, *axis),
                "sf" => RegionFilter::SfBox(axis.sf),
                "wsj" => RegionFilter::Boxes(vec![[9382.0, 21865.0, 23487.0, 36321.0]]),
                k => match region_preset_boxes(k) {
                    Some(bx) => RegionFilter::Boxes(lat_lon_boxes(bx, proj)),
                    None => return Err(format!("unknown region preset {k}")),
                },
            },
        }))
    }
    #[inline]
    pub fn contains(&self, x: f64, z: f64) -> bool {
        match self {
            RegionFilter::Bbox([a, b, c, d]) => x >= *a && x <= *c && z >= *b && z <= *d,
            RegionFilter::Side(s, ax) => ax.side_of(x, z) == *s,
            RegionFilter::Strip(s, ax) => ax.side_of(x, z) == *s && ((x - ax.mid.x) * ax.v.x + (z - ax.mid.z) * ax.v.z).abs() < 8000.0,
            RegionFilter::SfBox(c) => (x - c.x).abs() < 7000.0 && (z - c.z).abs() < 7000.0,
            RegionFilter::Boxes(bb) => bb.iter().any(|b| x >= b[0] && x <= b[2] && z >= b[1] && z <= b[3]),
        }
    }
}

/// latLonBoxesFilter boxes in the graph projection: [a.x, a.z, b.x, b.z], a = (latMax, lonMin), b = (latMin, lonMax).
pub fn lat_lon_boxes(boxes: &[[f64; 4]], proj: (f64, f64)) -> Vec<[f64; 4]> {
    boxes.iter().map(|&[la0, la1, lo0, lo1]| { let a = project_lat_lon((la1, lo0), proj); let b = project_lat_lon((la0, lo1), proj); [a.x, a.z, b.x, b.z] }).collect()
}

/// regionClassifier(proj, BREAKDOWN_REGIONS): (x, z) -> first preset key whose boxes contain it, or "other".
pub fn region_classifier(proj: (f64, f64)) -> impl Fn(f64, f64) -> &'static str {
    let fs: Vec<(&'static str, Vec<[f64; 4]>)> = BREAKDOWN_REGIONS.iter().map(|&k| (k, lat_lon_boxes(region_preset_boxes(k).unwrap(), proj))).collect();
    move |x, z| { for (k, bb) in &fs { if bb.iter().any(|b| x >= b[0] && x <= b[2] && z >= b[1] && z <= b[3]) { return *k; } } "other" }
}

impl Default for PopConfig {
    fn default() -> Self {
        PopConfig {
            n: 2000, seed: 42.0, local_share: 0.8,
            local_median_m: 5000.0, local_sigma: jm::log(12.0 / 5.0) / 1.2816, local_min_m: 500.0, local_max_m: 15000.0,
            long_min_m: 10000.0, long_max_m: 75000.0, long_lateral_sigma_m: 4000.0, long_p_cross: 0.4,
            max_attempts: 12, min_route_ratio: 0.7,
            work_start_mean: 8.5 * H, work_start_sd: 40.0 * MIN, work_start_min: 6.5 * H, work_start_max: 10.5 * H,
            travel_speed_factor: 0.8, slack_min: 0.83, slack_max: 1.33,
            rt_median: 0.9, rt_p95: 1.4, rt_min: 0.6, rt_max: 1.8,
            work_duration_mean: 8.5 * H, work_duration_sd: 20.0 * MIN,
            fuel_fill_min: 0.1, fuel_fill_max: 1.0, genome_sigma: 0.03, genome_trait_sigma: 0.12,
            return_trips: true,
            projection: (37.55, -122.15), san_jose: (37.3382, -121.8863), san_francisco: (37.7749, -122.4194), san_mateo_bridge_mid: (37.595, -122.192),
            home_ranks: DEFAULT_HOME_RANKS.to_vec(), astar_weight: 1.25,
            departures: Departures::Curve, curve_start: 6.0 * H, curve_bin: 15.0 * MIN,
            curve_share: vec![0.037, 0.042, 0.037, 0.041, 0.079, 0.08, 0.076, 0.079, 0.106, 0.095, 0.073, 0.067, 0.057, 0.049, 0.044, 0.038],
            mix: Mix::Spec, axis: AxisSpec::Projection(None), base_delta: None, delta_norm_cap: 0.5, range: None, region: None,
        }
    }
}

impl PopConfig {
    /// createTrainingEnv's population options: { n, seed, returnTrips: false } (src/sim/env.js).
    pub fn env(n: usize, seed: f64) -> PopConfig { PopConfig { n, seed, return_trips: false, ..Default::default() } }
    /// tools/rust/trace.mjs buildEnv('synthetic'): env options + { axis: synthAxis(graph), long: {minM: 5000, maxM: 16000}, departures: 'spec' }.
    pub fn env_synthetic(n: usize, seed: f64) -> PopConfig {
        PopConfig { axis: AxisSpec::Synth, long_min_m: 5000.0, long_max_m: 16000.0, departures: Departures::Spec, ..PopConfig::env(n, seed) }
    }
    /// Set cfg.region like generatePopulation({ region }) does: a preset's `pop` overrides (REGION_PRESETS[key].pop,
    /// e.g. bridges) are applied to this config first. Explicit options set after this call win (mergeCfg); use
    /// apply_json with both `region` and the options to get the same precedence in one step.
    pub fn set_region(&mut self, r: Option<Region>) {
        if let Some(Region::Preset(p)) = &r { if let Some(pop) = region_preset_pop(p) { self.apply_json(&pop); } }
        self.region = r;
    }
    /// Apply a JSON object of generatePopulation options (the subset ported here; nested objects merge like mergeCfg).
    /// `region` ({ preset } | { bbox } | { side }) is applied first, with its preset `pop` overrides, then the other
    /// options on top (explicit options win, as in mergeCfg). Panics on an unknown region.
    pub fn apply_json(&mut self, o: &Value) {
        if let Some(r) = o.get("region") {
            let reg = Region::from_json(r).unwrap_or_else(|e| panic!("population_gen: {e}"));
            // mergeCfg keys the pop overrides off options.region.preset (also when bbox / side take precedence in regionFilter)
            if let Some(pop) = r.get("preset").and_then(|p| p.as_str()).and_then(region_preset_pop) { self.apply_json(&pop); }
            self.region = reg;
        }
        let f = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_f64());
        if let Some(x) = f(o, "n") { self.n = x as usize; }
        if let Some(x) = f(o, "seed") { self.seed = x; }
        if let Some(x) = f(o, "localShare") { self.local_share = x; }
        if let Some(l) = o.get("local") { if let Some(x) = f(l, "medianM") { self.local_median_m = x; } if let Some(x) = f(l, "sigma") { self.local_sigma = x; } if let Some(x) = f(l, "minM") { self.local_min_m = x; } if let Some(x) = f(l, "maxM") { self.local_max_m = x; } }
        if let Some(l) = o.get("long") { if let Some(x) = f(l, "minM") { self.long_min_m = x; } if let Some(x) = f(l, "maxM") { self.long_max_m = x; } if let Some(x) = f(l, "lateralSigmaM") { self.long_lateral_sigma_m = x; } if let Some(x) = f(l, "pCross") { self.long_p_cross = x; } }
        if let Some(x) = f(o, "maxAttempts") { self.max_attempts = x as usize; }
        if let Some(x) = f(o, "minRouteRatio") { self.min_route_ratio = x; }
        if let Some(l) = o.get("workStart") { if let Some(x) = f(l, "mean") { self.work_start_mean = x; } if let Some(x) = f(l, "sd") { self.work_start_sd = x; } if let Some(x) = f(l, "min") { self.work_start_min = x; } if let Some(x) = f(l, "max") { self.work_start_max = x; } }
        if let Some(x) = f(o, "travelSpeedFactor") { self.travel_speed_factor = x; }
        if let Some(l) = o.get("slack") { if let Some(x) = f(l, "min") { self.slack_min = x; } if let Some(x) = f(l, "max") { self.slack_max = x; } }
        if let Some(l) = o.get("reactionTime") { if let Some(x) = f(l, "median") { self.rt_median = x; } if let Some(x) = f(l, "p95") { self.rt_p95 = x; } if let Some(x) = f(l, "min") { self.rt_min = x; } if let Some(x) = f(l, "max") { self.rt_max = x; } }
        if let Some(l) = o.get("workDuration") { if let Some(x) = f(l, "mean") { self.work_duration_mean = x; } if let Some(x) = f(l, "sd") { self.work_duration_sd = x; } }
        if let Some(l) = o.get("fuelFill") { if let Some(x) = f(l, "min") { self.fuel_fill_min = x; } if let Some(x) = f(l, "max") { self.fuel_fill_max = x; } }
        if let Some(l) = o.get("genome") { if let Some(x) = f(l, "sigma") { self.genome_sigma = x; } if let Some(x) = f(l, "traitSigma") { self.genome_trait_sigma = x; } }
        if let Some(x) = o.get("returnTrips").and_then(|x| x.as_bool()) { self.return_trips = x; }
        if let Some(x) = f(o, "astarWeight") { self.astar_weight = x; }
        if let Some(x) = o.get("departures").and_then(|x| x.as_str()) { self.departures = if x == "curve" { Departures::Curve } else { Departures::Spec }; }
        if let Some(x) = o.get("mix").and_then(|x| x.as_str()) { self.mix = if x == "realism" { Mix::Realism } else { Mix::Spec }; }
        if let Some(l) = o.get("departureCurve") {
            if let Some(x) = f(l, "start") { self.curve_start = x; } if let Some(x) = f(l, "bin") { self.curve_bin = x; }
            if let Some(a) = l.get("share").and_then(|a| a.as_array()) { self.curve_share = a.iter().map(|v| v.as_f64().unwrap()).collect(); }
        }
    }
}

/// Per-commuter fields of the JS population that `Population` does not carry (pop.arrays / commuter records), f64.
#[derive(Clone, Debug, Default)]
pub struct PopExtras {
    pub kind: Vec<u8>, pub intended: Vec<f64>, pub slack: Vec<f64>, pub reaction_time: Vec<f64>,
    pub length_ok: Vec<u8>, pub attempts: Vec<u32>, pub crosses_bay: Vec<u8>,
    /// scenario populations only: pop.tripTile / pop.tripSpec
    pub trip_tile: Vec<i32>, pub trip_spec: Vec<i32>,
}

pub struct Generated { pub pop: Population, pub extras: PopExtras }

/// One commuter record (generateCommuter result).
#[derive(Clone, Debug)]
pub struct Rec {
    pub home: i32, pub work: i32, pub local: bool, pub d: f64, pub l: f64, pub ok: bool, pub attempts: u32,
    pub route_out: Vec<i32>, pub route_back: Vec<i32>,
    pub expected_travel: f64, pub work_start: f64, pub slack: f64, pub depart: f64, pub return_depart: f64,
    pub vehicle_type: u8, pub fuel_fill: f64, pub reaction_time: f64, pub delta: [f32; DELTA_SIZE], pub crosses_bay: bool,
}

/// JS `${x}` for a number (seed strings of the per-commuter RNG streams).
pub fn js_num_str(x: f64) -> String {
    if x.is_nan() { return "NaN".into(); }
    if x.is_infinite() { return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() }; }
    if x == 0.0 { return "0".into(); }
    let a = x.abs();
    if (1e-6..1e21).contains(&a) { return format!("{}", x); }
    let s = format!("{:e}", x);
    let (m, e) = s.split_once('e').unwrap();
    if e.starts_with('-') { format!("{m}e{e}") } else { format!("{m}e+{e}") }
}

fn node_bounds<G: PopGraph + ?Sized>(w: &G) -> (f64, f64, f64, f64) {
    let (nx, nz) = (w.nx(), w.nz());
    if w.node_count() == 0 { return (0.0, 0.0, 0.0, 0.0); }
    let (mut min_x, mut max_x, mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..w.node_count() {
        let (x, z) = (nx[i], nz[i]);
        if x < min_x { min_x = x; } if x > max_x { max_x = x; } if z < min_z { min_z = z; } if z > max_z { max_z = z; }
    }
    (min_x, max_x, min_z, max_z)
}

fn prefix_sums(w: &[f64]) -> Vec<f64> { let mut acc = 0.0; w.iter().map(|&x| { acc += x; acc }).collect() }
#[inline] fn clip(v: f64, lo: f64, hi: f64) -> f64 { if v < lo { lo } else if v > hi { hi } else { v } }

/// route.js largestSCC (iterative Kosaraju): 1 for nodes of the largest strongly connected component.
pub fn largest_scc<G: PopGraph + ?Sized>(g: &G) -> Vec<u8> {
    let n = g.node_count();
    let mut order = vec![0usize; n]; let mut on = 0;
    let mut vis = vec![0u8; n];
    let mut stack_n = vec![0usize; n]; let mut stack_i = vec![0usize; n];
    for s in 0..n {
        if vis[s] != 0 { continue; }
        let mut sp: isize = 0; stack_n[0] = s; stack_i[0] = g.out_range(s).start; vis[s] = 1;
        while sp >= 0 {
            let k = sp as usize; let nd = stack_n[k];
            if stack_i[k] < g.out_range(nd).end {
                let m = g.edge_to(g.out_edge(stack_i[k])) as usize; stack_i[k] += 1;
                if vis[m] == 0 { vis[m] = 1; sp += 1; stack_n[sp as usize] = m; stack_i[sp as usize] = g.out_range(m).start; }
            } else { order[on] = nd; on += 1; sp -= 1; }
        }
    }
    let mut comp = vec![-1i32; n];
    let (mut nc, mut best, mut best_size) = (0i32, -1i32, 0usize);
    let mut st = vec![0usize; n];
    for k in (0..on).rev() {
        let s = order[k]; if comp[s] >= 0 { continue; }
        let mut sp: isize = 0; let mut size = 0; st[0] = s; comp[s] = nc;
        while sp >= 0 {
            let nd = st[sp as usize]; sp -= 1; size += 1;
            for i in g.in_range(nd) { let m = g.edge_from(g.in_edge(i)) as usize; if comp[m] < 0 { comp[m] = nc; sp += 1; st[sp as usize] = m; } }
        }
        if size > best_size { best_size = size; best = nc; }
        nc += 1;
    }
    comp.iter().map(|&c| (c == best) as u8).collect()
}

/// roadGraph.js makeNearestNode: 250 m bucket grid over the graph bounds, ring search.
pub struct NearestNode { min_x: f64, min_z: f64, cols: i64, rows: i64, start: Vec<u32>, items: Vec<u32> }
const CELL: f64 = 250.0;
impl NearestNode {
    pub fn new<G: PopGraph + ?Sized>(w: &G) -> NearestNode {
        let (nx, nz, nn) = (w.nx(), w.nz(), w.node_count());
        let (min_x, max_x, min_z, max_z) = node_bounds(w);
        let cols = jm::max(1.0, jm::ceil((max_x - min_x) / CELL) + 1.0) as i64;
        let rows = jm::max(1.0, jm::ceil((max_z - min_z) / CELL) + 1.0) as i64;
        let nc = (cols * rows) as usize;
        let cell_of = |x: f64, z: f64| -> usize {
            let cx = jm::min((cols - 1) as f64, jm::max(0.0, jm::floor((x - min_x) / CELL)));
            let cz = jm::min((rows - 1) as f64, jm::max(0.0, jm::floor((z - min_z) / CELL)));
            (cz as i64 * cols + cx as i64) as usize
        };
        let mut start = vec![0u32; nc + 1];
        for i in 0..nn { start[cell_of(nx[i], nz[i]) + 1] += 1; }
        for i in 0..nc { start[i + 1] += start[i]; }
        let mut fill = start[..nc].to_vec();
        let mut items = vec![0u32; nn];
        for i in 0..nn { let c = cell_of(nx[i], nz[i]); items[fill[c] as usize] = i as u32; fill[c] += 1; }
        NearestNode { min_x, min_z, cols, rows, start, items }
    }
    /// nearestNode(x, z, filter): -1 if none.
    pub fn nearest<G: PopGraph + ?Sized>(&self, w: &G, x: f64, z: f64, filter: impl Fn(usize) -> bool) -> i32 {
        if w.node_count() == 0 { return -1; }
        let (nx, nz) = (w.nx(), w.nz());
        let cx0 = jm::floor((x - self.min_x) / CELL) as i64; let cz0 = jm::floor((z - self.min_z) / CELL) as i64;
        let (mut best, mut best_d) = (-1i32, f64::INFINITY);
        let max_r = self.cols.max(self.rows) + 1;
        let visit = |c: usize, best: &mut i32, best_d: &mut f64| {
            for i in self.start[c]..self.start[c + 1] {
                let k = self.items[i as usize] as usize;
                if !filter(k) { continue; }
                let (dx, dz) = (nx[k] - x, nz[k] - z);
                let d = dx * dx + dz * dz;   // JS `**2` (fdlibm pow: y == 2 -> x * x)
                if d < *best_d { *best_d = d; *best = k as i32; }
            }
        };
        for r in 0..=max_r {
            if best >= 0 && jm::sqrt(best_d) <= ((r - 1) as f64) * CELL { break; }
            for cz in (cz0 - r)..=(cz0 + r) {
                if cz < 0 || cz >= self.rows { continue; }
                let edge_row = (cz - cz0).abs() == r;
                // same visiting order as the JS double loop (cx ascending; interior rows only touch cx0 +- r)
                if edge_row {
                    for cx in (cx0 - r).max(0)..=(cx0 + r).min(self.cols - 1) { visit((cz * self.cols + cx) as usize, &mut best, &mut best_d); }
                } else {
                    for cx in [cx0 - r, cx0 + r] { if cx >= 0 && cx < self.cols { visit((cz * self.cols + cx) as usize, &mut best, &mut best_d); } }
                }
            }
        }
        best
    }
}

/// traits.js makeGenome(): default traits (or clampDelta of a given delta).
pub fn make_genome(base: Option<&[f32]>, cap: f64) -> [f32; DELTA_SIZE] {
    let mut d = [0f32; DELTA_SIZE];
    match base {
        Some(b) => { d.copy_from_slice(&b[..DELTA_SIZE]); clamp_delta(&mut d, cap); }
        None => { for i in 0..6 { let (lo, hi) = TRAIT_RANGES[i]; d[i] = clip(DEFAULT_TRAITS[i], lo, hi) as f32; } }
    }
    d
}

/// traits.js clampDelta
pub fn clamp_delta(d: &mut [f32; DELTA_SIZE], cap: f64) {
    for i in 0..6 { let (lo, hi) = TRAIT_RANGES[i]; let v = d[i] as f64; d[i] = (if v < lo { lo } else if v > hi { hi } else { v }) as f32; }
    for v in d.iter_mut().take(D_LEARN_END).skip(D_LEARN_START) { let x = *v as f64; *v = (if x < -2.0 { -2.0 } else if x > 2.0 { 2.0 } else { x }) as f32; }
    for v in d.iter_mut().skip(D_LEARN_END) { *v = 0.0; }
    let mut q = 0.0; for &v in &d[D_LEARN_START..D_LEARN_END] { q += v as f64 * v as f64; }
    if q > cap * cap { let k = cap / jm::sqrt(q); for v in &mut d[D_LEARN_START..D_LEARN_END] { *v = (*v as f64 * k) as f32; } }
}

/// population.js deltaFor: mutate(base, sigma, rng `${seed}:delta:${id}` (src/sim/rng.js gaussian), {weightSigma, traitSigma}); delta[2] = reactionTime.
pub fn delta_for(seed_str: &str, cfg: &PopConfig, base: &[f32; DELTA_SIZE], id: usize, reaction_time: f64) -> [f32; DELTA_SIZE] {
    let mut rng = Rng::from_u32(hash_seed(&format!("{seed_str}:delta:{id}")));
    let rt_sigma = jm::log(1.4 / 0.9) / 1.645;   // traits.js REACTION_LOGNORMAL.sigma
    let (ws, ts) = (cfg.genome_sigma, cfg.genome_trait_sigma);
    let mut d = *base;
    for i in 0..6 {
        let (lo, hi) = TRAIT_RANGES[i];
        if i == 2 { d[i] = (d[i] as f64 * jm::exp(rng.gaussian() * ts * 2.0 * rt_sigma)) as f32; }
        else { d[i] = (d[i] as f64 + rng.gaussian() * ts * (hi - lo)) as f32; }
    }
    for v in &mut d[D_LEARN_START..D_LEARN_END] { *v = (*v as f64 + rng.gaussian() * ws) as f32; }
    clamp_delta(&mut d, cfg.delta_norm_cap);
    d[2] = reaction_time as f32;
    d
}

/// population.js _context (pools, axis, type CDFs, base genome); the router is supplied per thread.
pub struct Context {
    /// 1 for nodes allowed as a workplace snap target (ctx.inScc: SCC and inside the region)
    pub in_pool: Vec<u8>,
    pub region: Option<RegionFilter>,
    pub scc: Vec<u8>, pub all: Vec<i32>, pub homes: Vec<i32>, pub axis: Axis,
    pub type_cum: Vec<f64>, pub type_cum_realism: Vec<f64>, pub base: [f32; DELTA_SIZE], pub nearest: NearestNode, pub seed_str: String,
}

impl Context {
    pub fn new<G: PopGraph + ?Sized>(g: &G, cfg: &PopConfig) -> Result<Context, String> {
        let scc = largest_scc(g);
        let (axis, proj) = match cfg.axis {
            AxisSpec::Explicit(a) => (a, DEFAULT_PROJECTION), AxisSpec::Synth => (Axis::synth(g), DEFAULT_PROJECTION),
            AxisSpec::Projection(p) => { let pr = p.unwrap_or(cfg.projection); (Axis::from_cfg(cfg, pr), pr) }
        };
        let region = match &cfg.region { Some(r) => RegionFilter::new(r, &axis, proj)?, None => None };
        let (nx, nz) = (g.nx(), g.nz());
        let (mut homes, mut all) = (Vec::new(), Vec::new());
        // nodeLowRank (roadGraph.js): min rank over the edges touching the node (127 if none)
        let mut low = vec![127i32; g.node_count()];
        for e in 0..g.edge_count() { let r = g.edge_rank(e); for k in [g.edge_from(e) as usize, g.edge_to(e) as usize] { if r < low[k] { low[k] = r; } } }
        for i in 0..g.node_count() {
            if scc[i] == 0 { continue; }
            if let Some(f) = &region { if !f.contains(nx[i], nz[i]) { continue; } }
            all.push(i as i32);
            if cfg.home_ranks.contains(&low[i]) { homes.push(i as i32); }
        }
        if all.len() < 2 { return Err("graph has too few routable nodes".into()); }
        // inScc: SCC node inside the region (the workplace snap filter)
        let in_pool = match &region { None => scc.clone(), Some(f) => (0..g.node_count()).map(|k| (scc[k] == 1 && f.contains(nx[k], nz[k])) as u8).collect() };
        Ok(Context {
            in_pool, region,
            homes: if homes.is_empty() { all.clone() } else { homes }, all, scc, axis,
            type_cum: prefix_sums(&TYPE_WEIGHTS), type_cum_realism: prefix_sums(&REALISM_TYPE_WEIGHTS),
            base: make_genome(cfg.base_delta.as_deref(), cfg.delta_norm_cap), nearest: NearestNode::new(g), seed_str: js_num_str(cfg.seed),
        })
    }
}

fn route_length<G: PopGraph + ?Sized>(g: &G, r: &[i32]) -> f64 { let mut l = 0.0; for &e in r { l += g.edge_length(e as usize); } l }

/// population.js generateCommuter (pure function of (cfg.seed, id) given the graph and router).
pub fn generate_commuter<G: PopGraph + ?Sized>(g: &G, cfg: &PopConfig, ctx: &Context, router: &mut dyn Router, id: usize) -> Result<Rec, String> {
    let mut rng = Rng::from_u32(hash_seed(&format!("{}:{}", ctx.seed_str, id)));
    let (nx, nz) = (g.nx(), g.nz());
    let ax = &ctx.axis;
    let (sj, dd, u, v, mid) = (ax.sj, ax.d, ax.u, ax.v, ax.mid);
    let realism = cfg.mix == Mix::Realism;
    let mut is_local = rng.random() < cfg.local_share;
    let mut r_bin = REALISM_BINS[0];
    if realism {
        let mut q = rng.random(); let mut found = None;
        for b in REALISM_BINS { if q < b.2 { found = Some(b); break; } q -= b.2; }
        r_bin = found.unwrap_or(REALISM_BINS[0]); is_local = r_bin.1 <= 15.0;
    }
    let in_scc = |k: usize| ctx.in_pool[k] == 1;
    let mut home = ctx.homes[rng.int(ctx.homes.len())];
    struct Best { work: i32, route: Vec<i32>, l: f64, d: f64, ok: bool }
    let mut best: Option<Best> = None;
    let (mut attempts, mut home_tries) = (0u32, 0u32);
    loop {
        for _a in 0..cfg.max_attempts {
            attempts += 1;
            let d; let (mut cx, mut cz);
            let (hx, hz) = (nx[home as usize], nz[home as usize]);
            if is_local {
                d = if realism { rng.range(r_bin.0, r_bin.1) * 1000.0 } else { clip(jm::exp(jm::log(cfg.local_median_m) + cfg.local_sigma * rng.gaussian()), cfg.local_min_m, cfg.local_max_m) };
                let th = rng.random() * 2.0 * jm::PI;
                cx = hx + d * jm::cos(th); cz = hz + d * jm::sin(th);
            } else {
                d = if realism { rng.range(r_bin.0, r_bin.1) * 1000.0 } else { rng.range(cfg.long_min_m, cfg.long_max_m) };
                let t = (hx - sj.x) * u.x + (hz - sj.z) * u.z;
                let (room_sf, room_sj) = (dd - t, t);
                let s: f64 = if d <= room_sf && d <= room_sj { if rng.random() * (room_sf + room_sj) < room_sf { 1.0 } else { -1.0 } }
                    else if d <= room_sf { 1.0 } else if d <= room_sj { -1.0 } else if room_sf >= room_sj { 1.0 } else { -1.0 };
                let jit = rng.gaussian() * cfg.long_lateral_sigma_m;
                cx = hx + s * d * u.x + jit * v.x; cz = hz + s * d * u.z + jit * v.z;
                if rng.random() < (if realism { REALISM_P_CROSS } else { cfg.long_p_cross }) { let k = 2.0 * ((cx - mid.x) * v.x + (cz - mid.z) * v.z); cx -= k * v.x; cz -= k * v.z; }
            }
            let w = ctx.nearest.nearest(g, cx, cz, in_scc);
            if w < 0 || w == home { continue; }
            if jm::hypot(nx[w as usize] - cx, nz[w as usize] - cz) > jm::max(1000.0, d / 4.0) { continue; }
            let r = match router.route(home, w, None, None) { Some(r) if !r.is_empty() => r, _ => continue };
            let l = route_length(g, &r);
            let ok = l >= cfg.min_route_ratio * d;
            let better = match &best { None => true, Some(b) => (ok && !b.ok) || (ok == b.ok && l / d > b.l / b.d) };
            if better { best = Some(Best { work: w, route: r, l, d, ok }); }
            if ok { break; }
        }
        if best.is_some() { break; }
        home_tries += 1;
        if home_tries > 3 {
            let mut k = 0;
            while k < 50 && best.is_none() {
                k += 1;
                let w = ctx.all[rng.int(ctx.all.len())]; if w == home { continue; }
                if let Some(r) = router.route(home, w, None, None) { if !r.is_empty() { let l = route_length(g, &r); best = Some(Best { work: w, route: r, l, d: l, ok: false }); } }
            }
            if best.is_some() { break; }
            if home_tries > 20 { return Err(format!("commuter {id}: no workplace found")); }
        }
        home = ctx.homes[rng.int(ctx.homes.len())];
    }
    let Best { work, route: route_out, l, d, ok } = best.unwrap();
    let route_back = if cfg.return_trips { router.route(work, home, None, None).unwrap_or_default() } else { Vec::new() };
    let mut tt = 0.0; for &e in &route_out { let e = e as usize; tt += g.edge_length(e) / (g.edge_speed(e) * cfg.travel_speed_factor); }
    let slack = rng.range(cfg.slack_min, cfg.slack_max);
    let (work_start, depart);
    if cfg.departures == Departures::Curve {
        let sh = &cfg.curve_share; let mut tot = 0.0; for &x in sh { tot += x; }
        let mut q = rng.random() * tot; let mut k = 0usize;
        while k < sh.len() - 1 && q >= sh[k] { q -= sh[k]; k += 1; }
        depart = cfg.curve_start + (k as f64 + rng.random()) * cfg.curve_bin;
        work_start = depart + slack * tt;
    } else {
        work_start = clip(cfg.work_start_mean + rng.gaussian() * cfg.work_start_sd, cfg.work_start_min, cfg.work_start_max);
        depart = work_start - slack * tt;
    }
    let return_depart = work_start + jm::max(4.0 * H, cfg.work_duration_mean + rng.gaussian() * cfg.work_duration_sd);
    let vehicle_type = rng.weighted_index(if realism { &ctx.type_cum_realism } else { &ctx.type_cum }) as u8;
    let fuel_fill = rng.range(cfg.fuel_fill_min, cfg.fuel_fill_max);
    let sig = jm::log(cfg.rt_p95 / cfg.rt_median) / 1.6449;
    let reaction_time = clip(cfg.rt_median * jm::exp(sig * rng.gaussian()), cfg.rt_min, cfg.rt_max);
    let delta = delta_for(&ctx.seed_str, cfg, &ctx.base, id, reaction_time);
    let (sh, sw) = (ax.side_of(nx[home as usize], nz[home as usize]), ax.side_of(nx[work as usize], nz[work as usize]));
    Ok(Rec { home, work, local: is_local, d, l, ok, attempts, route_out, route_back, expected_travel: tt, work_start, slack, depart, return_depart,
        vehicle_type, fuel_fill, reaction_time, delta, crosses_bay: sh != sw })
}

/// population.js assemble: struct-of-arrays population (returnDepart = Infinity without return trips).
pub fn assemble(recs: &[Rec], return_trips: bool) -> Generated {
    let n = recs.len();
    let total: usize = recs.iter().map(|r| r.route_out.len() + r.route_back.len()).sum();
    let mut p = Population { n, ..Default::default() };
    let mut x = PopExtras::default();
    p.route_flat.reserve(total); p.route_offset.reserve(2 * n + 1); p.deltas.reserve(n * DELTA_SIZE);
    for r in recs {
        p.route_offset.push(p.route_flat.len() as u32); p.route_flat.extend_from_slice(&r.route_out);
        p.route_offset.push(p.route_flat.len() as u32); p.route_flat.extend_from_slice(&r.route_back);
        p.deltas.extend_from_slice(&r.delta);
        p.vehicle_type.push(r.vehicle_type); p.home.push(r.home); p.work.push(r.work);
        p.depart.push(r.depart); p.return_depart.push(if return_trips { r.return_depart } else { f64::INFINITY }); p.work_start.push(r.work_start);
        p.fuel_fill.push(r.fuel_fill); p.route_length.push(r.l); p.expected_travel.push(r.expected_travel);
        x.kind.push(if r.local { 0 } else { 1 }); x.intended.push(r.d); x.slack.push(r.slack); x.reaction_time.push(r.reaction_time);
        x.length_ok.push(r.ok as u8); x.attempts.push(r.attempts); x.crosses_bay.push(r.crosses_bay as u8);
    }
    p.route_offset.push(p.route_flat.len() as u32);
    Generated { pop: p, extras: x }
}

/// generatePopulation(graph, cfg) with `make_router()` per worker (each must behave like makeRouter(graph, { weight: cfg.astarWeight })).
/// `threads`: None = the current rayon pool, Some(1) = serial on this thread, Some(k) = a dedicated k-thread pool.
/// Output does not depend on the thread count.
pub fn generate_population_with<G: PopGraph + Sync + ?Sized, R: Router + Send, F: Fn() -> R + Sync>(g: &G, cfg: &PopConfig, make_router: F, threads: Option<usize>) -> Result<Generated, String> {
    let ctx = Context::new(g, cfg)?;
    let (a, b) = cfg.range.unwrap_or((0, cfg.n));
    let recs: Vec<Rec> = if threads == Some(1) || b - a < 64 {
        let mut r = make_router();
        (a..b).map(|id| generate_commuter(g, cfg, &ctx, &mut r, id)).collect::<Result<_, _>>()?
    } else {
        let run = || -> Result<Vec<Rec>, String> {
            let nt = rayon::current_num_threads();
            let chunk = ((b - a) / (nt * 8)).clamp(16, 2048);
            let chunks: Vec<(usize, usize)> = (a..b).step_by(chunk).map(|s| (s, (s + chunk).min(b))).collect();
            // router pool: at most one router per worker thread (a router holds per-node scratch for the whole graph)
            let pool: std::sync::Mutex<Vec<R>> = std::sync::Mutex::new(Vec::new());
            let parts: Vec<Result<Vec<Rec>, String>> = chunks.par_iter().map(|&(s, e)| {
                let popped = pool.lock().unwrap().pop();
                let mut r = popped.unwrap_or_else(&make_router);
                let out = (s..e).map(|id| generate_commuter(g, cfg, &ctx, &mut r, id)).collect();
                pool.lock().unwrap().push(r);
                out
            }).collect();
            let mut out = Vec::with_capacity(b - a);
            for p in parts { out.extend(p?); }
            Ok(out)
        };
        match threads {
            Some(k) => rayon::ThreadPoolBuilder::new().num_threads(k).build().map_err(|e| e.to_string())?.install(run)?,
            None => run()?,
        }
    };
    Ok(assemble(&recs, cfg.return_trips))
}

/// generatePopulation with the default router (bayenv_world::nav via route_stub::NavRouter, weight = cfg.astarWeight).
pub fn generate_population(g: &Arc<World>, cfg: &PopConfig, threads: Option<usize>) -> Result<Generated, String> {
    let w = cfg.astar_weight;
    generate_population_with(&**g, cfg, || crate::route_stub::NavRouter::new(g, w), threads)
}

/// generatePopulation directly on the shared graph (bayenv_world::World::graph): same output as
/// [`generate_population`] on the sim World built from it, without the World copy.
pub fn generate_population_sim_graph(g: &SimGraph, cfg: &PopConfig, threads: Option<usize>) -> Result<Generated, String> {
    let w = cfg.astar_weight;
    generate_population_with(g, cfg, || crate::route_stub::NavRouter::from_sim_graph(g, w), threads)
}

/// scenarioPopulation options (src/scenarios/load.js).
#[derive(Clone, Debug)]
pub struct ScenarioOpts { pub seed: Option<f64>, pub scale: f64, pub fuel_fill: f64, pub max_agents: usize, pub delta_base: Option<Vec<f32>>, pub time_shift: f64 }
impl Default for ScenarioOpts { fn default() -> Self { ScenarioOpts { seed: None, scale: 1.0, fuel_fill: 1.0, max_agents: usize::MAX, delta_base: None, time_shift: 0.0 } } }

/// scenarioPopulation(graph, scenario, opts): `scenario` is the parsed scenario.json; `router` must behave like
/// makeRouter(graph, { weight: 1 }).
pub fn scenario_population_with<G: PopGraph + ?Sized>(g: &G, scenario: &Value, o: &ScenarioOpts, router: &mut dyn Router) -> Result<Generated, String> {
    let seed = o.seed.or_else(|| scenario.get("seed").and_then(|s| s.as_f64())).unwrap_or(1.0);
    let cfg = PopConfig { seed, n: 0, return_trips: false, ..Default::default() };
    let seed_str = js_num_str(seed);
    let base = make_genome(o.delta_base.as_deref(), cfg.delta_norm_cap);
    let type_cum = prefix_sums(&TYPE_WEIGHTS);
    let sig = jm::log(cfg.rt_p95 / cfg.rt_median) / 1.6449;
    let name = scenario.get("name").and_then(|s| s.as_str()).unwrap_or("?");
    let (mut recs, mut tile, mut spec) = (Vec::new(), Vec::new(), Vec::new());
    let num = |v: &Value| v.as_f64().unwrap_or(0.0);
    for (ti, t) in scenario["trips"].as_array().ok_or("scenario has no trips")?.iter().enumerate() {
        let count = t.get("count").map(num).unwrap_or(0.0);
        let n = if count > 0.0 { jm::max(1.0, jm::round(count * o.scale)) as usize } else { 0 };
        let (from, to) = (num(&t["from"]) as i32, num(&t["to"]) as i32);
        let mut legs = vec![from];
        if let Some(v) = t.get("via").and_then(|v| v.as_array()) { legs.extend(v.iter().map(|x| num(x) as i32)); }
        legs.push(to);
        let mut r: Vec<i32> = Vec::new(); let mut okr = true;
        for k in 0..legs.len() - 1 {
            match router.route(legs[k], legs[k + 1], None, None) { Some(p) if !p.is_empty() => r.extend_from_slice(&p), _ => { okr = false; break; } }
        }
        if !okr { return Err(format!("scenario {name}: trip {ti} ({from} -> {to}) has no route")); }
        if let Some(rt) = t.get("route").and_then(|v| v.as_array()) { r = rt.iter().map(|x| num(x) as i32).collect(); }
        let (mut l, mut tt) = (0.0, 0.0);
        for &e in &r { let e = e as usize; l += g.edge_length(e); tt += g.edge_length(e) / (g.edge_speed(e) * cfg.travel_speed_factor); }
        let dep = t["depart"].as_array().ok_or("trip depart")?;
        let (d0, d1) = (num(&dep[0]), num(&dep[1]));
        let types: Option<Vec<u8>> = t.get("types").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| num(x) as u8).collect());
        let mut j = 0;
        while j < n && recs.len() < o.max_agents {
            let mut rng = Rng::from_u32(hash_seed(&format!("{seed_str}:scn:{ti}:{j}")));
            let depart = o.time_shift + d0 + if n > 1 { (d1 - d0) * (j as f64 + rng.random() * 0.8) / n as f64 } else { rng.random() * (d1 - d0) };
            let slack = rng.range(cfg.slack_min, cfg.slack_max);
            let reaction_time = jm::min(cfg.rt_max, jm::max(cfg.rt_min, cfg.rt_median * jm::exp(sig * rng.gaussian())));
            let vehicle_type = match &types { Some(ts) => ts[rng.int(ts.len())], None => rng.weighted_index(&type_cum) as u8 };
            let id = recs.len();
            recs.push(Rec { home: from, work: to, local: true, d: l, l, ok: true, attempts: 1, route_out: r.clone(), route_back: Vec::new(),
                expected_travel: tt, work_start: depart + slack * tt, slack, depart, return_depart: f64::INFINITY, vehicle_type, fuel_fill: o.fuel_fill,
                reaction_time, delta: delta_for(&seed_str, &cfg, &base, id, reaction_time), crosses_bay: false });
            tile.push(t.get("tile").and_then(|x| x.as_f64()).unwrap_or(0.0) as i32); spec.push(ti as i32);
            j += 1;
        }
    }
    let mut out = assemble(&recs, false);
    out.extras.trip_tile = tile; out.extras.trip_spec = spec;
    Ok(out)
}

/// scenarioPopulation with the default router (bayenv_world::nav via route_stub::NavRouter, weight 1).
pub fn scenario_population(g: &Arc<World>, scenario: &Value, o: &ScenarioOpts) -> Result<Generated, String> {
    let mut r = crate::route_stub::NavRouter::new(g, 1.0);
    scenario_population_with(&**g, scenario, o, &mut r)
}

/// scenarioPopulation directly on the shared graph (NavRouter::from_sim_graph, weight 1).
pub fn scenario_population_sim_graph(g: &SimGraph, scenario: &Value, o: &ScenarioOpts) -> Result<Generated, String> {
    let mut r = crate::route_stub::NavRouter::from_sim_graph(g, 1.0);
    scenario_population_with(g, scenario, o, &mut r)
}

/// Repository data dir (…/data) relative to this crate.
pub fn default_data_dir() -> std::path::PathBuf { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data") }

impl Population {
    /// The population createTrainingEnv builds for tools/rust/trace.mjs buildEnv(world, n, seed):
    ///   "synthetic" -> generatePopulation({ n, seed, returnTrips: false, axis: synthAxis, long: 5..16 km, departures: 'spec' })
    ///   "real"      -> generatePopulation({ n, seed, returnTrips: false })   (== popParallel for n >= 4000)
    ///   "scenario:<name>" -> scenarioPopulation(graph, <data_dir>/scenarios/<name>/scenario.json, { seed })  (n ignored)
    /// `data_dir` defaults to the repository's data/ directory.
    pub fn generate_like_env(world: &Arc<World>, world_name: &str, n: usize, seed: f64, data_dir: Option<&std::path::Path>) -> Result<Population, String> {
        Ok(match world_name {
            "synthetic" => generate_population(world, &PopConfig::env_synthetic(n, seed), None)?.pop,
            "real" => generate_population(world, &PopConfig::env(n, seed), None)?.pop,
            s if s.starts_with("scenario:") => {
                let dir = data_dir.map(|p| p.to_path_buf()).unwrap_or_else(default_data_dir).join("scenarios").join(&s[9..]).join("scenario.json");
                let txt = std::fs::read_to_string(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let sc: Value = serde_json::from_str(&txt).map_err(|e| e.to_string())?;
                scenario_population(world, &sc, &ScenarioOpts { seed: Some(seed), ..Default::default() })?.pop
            }
            _ => return Err(format!("unknown world {world_name}")),
        })
    }
}
