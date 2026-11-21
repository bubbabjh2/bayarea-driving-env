//! src/sim/fuel.js (FUEL_DEFAULTS, litresPerKm, StationNetwork + nav/stations.js routeViaStation) and
//! src/sim/routeStore.js offsets (route contents live per slot; only the offsets are observable).
use crate::jm;
use crate::physics::{fuel_flow, resistance, VEHICLE_PARAMS};
use crate::population::Population;
use crate::sim::Router;
use crate::vehicle::{VehicleStore, STATE_FREE};
use crate::world::World;
use crate::fx::HashMap;

#[derive(Clone, Debug)]
pub struct FuelCfg { pub low_fraction: f64, pub reserve_km: f64, pub plan_speed: f64, pub plan_factor: f64, pub pump_min: f64, pub pump_max: f64,
    pub tow_median: f64, pub tow_sigma: f64, pub check_interval: f64, pub max_candidates: usize, pub min_fill_to_skip: f64 }
impl Default for FuelCfg {
    fn default() -> Self { FuelCfg { low_fraction: 0.12, reserve_km: 15.0, plan_speed: 22.2, plan_factor: 1.3, pump_min: 180.0, pump_max: 300.0,
        tow_median: 30.0 * 60.0, tow_sigma: 0.3, check_interval: 60.0, max_candidates: 6, min_fill_to_skip: 0.5 } }
}

pub fn litres_per_km(type_idx: usize, v: f64) -> f64 {
    let p = &VEHICLE_PARAMS[if type_idx < 4 { type_idx } else { 0 }];
    let pw = resistance(p, v, 0.0, 1.0) * v / 0.88 + p.accessory_w;
    fuel_flow(p, pw) * 1000.0 / jm::max(v, 0.1)
}

/// detour_time = via-station route time - direct route time (s, free flow), as routeViaStation's detourTime.
#[derive(Clone, Debug)]
pub struct Plan { pub edges: Vec<i32>, pub station: usize, pub stop_index: usize, pub detour_time: f64 }

pub struct StationNetwork {
    pub cfg: FuelCfg,
    pub node: Vec<i32>, pub pumps: Vec<i32>, pub x: Vec<f64>, pub z: Vec<f64>,
    pub busy: Vec<i32>, pub served: Vec<i32>, pub queue: Vec<Vec<usize>>, pub max_queue: usize, pub planned: u64, pub plan_failed: u64,
}

/// The graph queries refuel planning needs (implemented for the sim's `World` and the shared `SimGraph`, so the
/// GPU host plans on `bayenv_world` without a copy).
pub trait PlanGraph {
    fn edge_from(&self, e: usize) -> i32;
    fn edge_to(&self, e: usize) -> i32;
    fn edge_length(&self, e: usize) -> f64;
    fn edge_speed(&self, e: usize) -> f64;
    fn node_x(&self, n: usize) -> f64;
    fn node_z(&self, n: usize) -> f64;
}
impl PlanGraph for World {
    fn edge_from(&self, e: usize) -> i32 { self.e_from[e] }
    fn edge_to(&self, e: usize) -> i32 { self.e_to[e] }
    fn edge_length(&self, e: usize) -> f64 { self.e_length[e] }
    fn edge_speed(&self, e: usize) -> f64 { self.e_speed[e] }
    fn node_x(&self, n: usize) -> f64 { self.nx[n] }
    fn node_z(&self, n: usize) -> f64 { self.nz[n] }
}
impl PlanGraph for bayenv_world::graph::SimGraph {
    fn edge_from(&self, e: usize) -> i32 { self.from[e] as i32 }
    fn edge_to(&self, e: usize) -> i32 { self.to[e] as i32 }
    fn edge_length(&self, e: usize) -> f64 { self.length[e] }
    fn edge_speed(&self, e: usize) -> f64 { self.speed_limit[e] }
    fn node_x(&self, n: usize) -> f64 { self.nx[n] }
    fn node_z(&self, n: usize) -> f64 { self.nz[n] }
}

/// `&Arc<World>` callers keep working with the generic planners / population generator.
impl<G: PlanGraph + ?Sized> PlanGraph for std::sync::Arc<G> {
    #[inline] fn edge_from(&self, e: usize) -> i32 { (**self).edge_from(e) }
    #[inline] fn edge_to(&self, e: usize) -> i32 { (**self).edge_to(e) }
    #[inline] fn edge_length(&self, e: usize) -> f64 { (**self).edge_length(e) }
    #[inline] fn edge_speed(&self, e: usize) -> f64 { (**self).edge_speed(e) }
    #[inline] fn node_x(&self, n: usize) -> f64 { (**self).node_x(n) }
    #[inline] fn node_z(&self, n: usize) -> f64 { (**self).node_z(n) }
}

struct RouteRes { edges: Vec<i32>, nodes: Vec<i32>, length: f64, time: f64 }
fn to_result<G: PlanGraph + ?Sized>(g: &G, edges: Option<Vec<i32>>) -> Option<RouteRes> {
    let edges = edges?;
    if edges.is_empty() { return None; }
    let mut nodes = Vec::with_capacity(edges.len() + 1);
    nodes.push(g.edge_from(edges[0] as usize));
    let (mut length, mut time) = (0.0, 0.0);
    for &e in &edges { let e = e as usize; nodes.push(g.edge_to(e)); length += g.edge_length(e); time += g.edge_length(e) / g.edge_speed(e); }
    Some(RouteRes { edges, nodes, length, time })
}

impl StationNetwork {
    pub fn new(g: &World, cfg: FuelCfg) -> Self {
        let n = g.stations.len();
        StationNetwork { cfg, node: g.stations.iter().map(|s| s.node).collect(), pumps: g.stations.iter().map(|s| s.pumps).collect(),
            x: g.stations.iter().map(|s| s.x).collect(), z: g.stations.iter().map(|s| s.z).collect(),
            busy: vec![0; n], served: vec![0; n], queue: vec![Vec::new(); n], max_queue: 0, planned: 0, plan_failed: 0 }
    }
    pub fn needs_fuel(&self, type_idx: usize, fuel: f64, tank: f64, remaining_m: f64) -> bool {
        let f = fuel / tank;
        if f >= self.cfg.min_fill_to_skip { return false; }
        if f < self.cfg.low_fraction { return true; }
        let lpk = litres_per_km(type_idx, self.cfg.plan_speed) * self.cfg.plan_factor;
        fuel / lpk < remaining_m / 1000.0 + self.cfg.reserve_km
    }
    /// StationNetwork.plan (routeViaStation)
    pub fn plan<G: PlanGraph + ?Sized>(&mut self, g: &G, router: &mut dyn Router, from: i32, goal: i32) -> Option<Plan> {
        if self.node.is_empty() { return None; }
        let r = self.plan_many(g, router, &[(from, goal)]).pop().unwrap();
        self.count(&r);
        r
    }
    /// The planned / planFailed bookkeeping of one plan() result.
    pub fn count(&mut self, r: &Option<Plan>) { if r.is_some() { self.planned += 1; } else { self.plan_failed += 1; } }

    /// plan() for many (from, goal) pairs without bookkeeping: the direct routes as one router batch, then both legs
    /// of every candidate station as one batch. Each result equals plan(from, goal) (a pure function of the pair).
    pub fn plan_many<G: PlanGraph + ?Sized>(&self, g: &G, router: &mut dyn Router, ods: &[(i32, i32)]) -> Vec<Option<Plan>> {
        if self.node.is_empty() { return ods.iter().map(|_| None).collect(); }
        let directs = router.route_batch(ods, None, None);
        let mut cand: Vec<(usize, Vec<usize>, f64)> = Vec::new();   // (od index, candidate stations, direct time)
        let mut qs: Vec<(i32, i32)> = Vec::new();
        for (k, d) in directs.into_iter().enumerate() {
            let Some(direct) = to_result(g, d) else { continue };
            let (from, goal) = ods[k];
            let stride = 1usize.max((direct.nodes.len() as f64 / 400.0).floor() as usize);
            let mut scored: Vec<(usize, f64)> = Vec::new();
            for s in 0..self.node.len() {
                if self.node[s] == from || self.node[s] == goal { continue; }
                let mut best = f64::INFINITY;
                let mut i = 0;
                while i < direct.nodes.len() { let n = direct.nodes[i] as usize; let dx = g.node_x(n) - self.x[s]; let dz = g.node_z(n) - self.z[s]; let d2 = dx * dx + dz * dz; if d2 < best { best = d2; } i += stride; }
                scored.push((s, best));
            }
            scored.sort_by(|a, b| jm::cmp_of(a.1 - b.1));
            let cs: Vec<usize> = scored.iter().take(self.cfg.max_candidates).map(|c| c.0).collect();
            for &s in &cs { qs.push((from, self.node[s])); qs.push((self.node[s], goal)); }
            cand.push((k, cs, direct.time));
        }
        let legs = router.route_batch(&qs, None, None);
        let mut out: Vec<Option<Plan>> = ods.iter().map(|_| None).collect();
        let mut q = 0usize;
        for (k, cs, direct_time) in cand {
            let mut best: Option<(RouteRes, usize)> = None;
            for &s in &cs {
                let (l1, l2) = (&legs[q], &legs[q + 1]); q += 2;
                let Some(a) = l1 else { continue };
                let Some(b) = l2 else { continue };
                let mut all = a.clone(); all.extend_from_slice(b);
                let Some(r) = to_result(g, Some(all)) else { continue };
                if best.as_ref().is_none_or(|(br, _)| r.time < br.time) { best = Some((r, s)); }
            }
            if let Some((route, st)) = best {
                if let Some(si) = route.edges.iter().position(|&e| g.edge_to(e as usize) == self.node[st]) {
                    out[k] = Some(Plan { detour_time: route.time - direct_time, edges: route.edges, station: st, stop_index: si });
                }
            }
        }
        out
    }
}

/// RouteStore offsets: [0, baseLen) population pack, dynamic routes bump-allocated after it, compacted on overflow.
pub struct RouteStore { pub base_len: usize, pub flat_len: usize, pub used: usize, pub pop_offset: Vec<u32>, pub compactions: u64 }

impl RouteStore {
    pub fn new(pop: &Population) -> Self { let b = pop.route_flat.len(); RouteStore { base_len: b, flat_len: b + (1 << 20), used: b, pop_offset: pop.route_offset.clone(), compactions: 0 } }
    pub fn pop_offset(&self, k: usize) -> i32 { self.pop_offset[k] as i32 }
    /// add(edges): returns the offset; relocates live dynamic routes (vs.route_off) on compaction.
    pub fn add(&mut self, edges: &[i32], vs: &mut VehicleStore) -> i32 {
        let n = edges.len();
        if self.used + n > self.flat_len {
            let mut keep: Vec<(usize, i32, i32)> = Vec::new();
            for k in 0..vs.high { if vs.state[k] != STATE_FREE && vs.route_off[k] >= 0 && vs.route_off[k] as usize >= self.base_len { keep.push((k, vs.route_off[k], vs.route_len[k])); } }
            let mut total = self.base_len; for &(_, _, l) in &keep { total += l as usize; }
            let cap = self.flat_len.max(total + n + (1 << 18));
            let mut moved: HashMap<i32, i32> = HashMap::default();
            let mut p = self.base_len as i32;
            for (slot, off, len) in keep {
                let no = *moved.entry(off).or_insert_with(|| { let q = p; p += len; q });
                vs.route_off[slot] = no;
            }
            self.flat_len = cap; self.used = p as usize; self.compactions += 1;
        }
        let off = self.used;
        self.used += n;
        off as i32
    }
}
