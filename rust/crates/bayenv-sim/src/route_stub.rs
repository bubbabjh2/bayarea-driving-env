//! TEMPORARY stand-in for src/sim/route.js makeRouter (weighted A*, banned hops, avoid set, traffic penalty),
//! used by the differential tests until bayenv_world::nav::Router lands (routing is owned by the world crate).
use crate::jm;
use crate::sim::Router;
use crate::world::World;
use crate::fx::HashMap;
use std::sync::Arc;

pub struct AStar { g: Arc<World>, e_time: Vec<f64>, g_score: Vec<f64>, came: Vec<i32>, stamp: Vec<i32>, closed: Vec<u8>, gen: i32, max_speed: f64, weight: f64, junction_penalty: f64,
    cache: HashMap<i64, Option<Vec<i32>>>, cache_max: usize }

impl AStar {
    pub fn new(g: Arc<World>, weight: f64) -> AStar {
        let n = g.node_count;
        let e_time = (0..g.edge_count).map(|e| g.e_length[e] / g.e_speed[e]).collect();
        let mut ms: f64 = g.max_speed;
        if ms == 0.0 { for e in 0..g.edge_count { if g.e_speed[e] > ms { ms = g.e_speed[e]; } } }
        AStar { e_time, g_score: vec![0.0; n], came: vec![0; n], stamp: vec![0; n], closed: vec![0; n], gen: 0, max_speed: jm::max(1.0, ms), weight, junction_penalty: 2.0, cache: HashMap::default(), cache_max: 50000, g }
    }
}

struct Heap { k: Vec<f64>, v: Vec<i32> }
impl Heap {
    fn push(&mut self, key: f64, val: i32) {
        let mut i = self.k.len(); self.k.push(key); self.v.push(val);
        while i > 0 { let p = (i - 1) >> 1; if self.k[p] <= key { break; } self.k[i] = self.k[p]; self.v[i] = self.v[p]; i = p; }
        self.k[i] = key; self.v[i] = val;
    }
    fn pop(&mut self) -> i32 {
        let top = self.v[0];
        let lk = self.k.pop().unwrap(); let lv = self.v.pop().unwrap();
        let n = self.k.len();
        if n > 0 {
            let mut i = 0;
            loop {
                let l = 2 * i + 1; let r = l + 1;
                let (mut m, mut mk) = (i, lk);
                if l < n && self.k[l] < mk { m = l; mk = self.k[l]; }
                if r < n && self.k[r] < mk { m = r; }
                if m == i { break; }
                self.k[i] = self.k[m]; self.v[i] = self.v[m]; i = m;
            }
            self.k[i] = lk; self.v[i] = lv;
        }
        top
    }
}

impl Router for AStar {
    fn route(&mut self, start: i32, goal: i32, avoid: Option<&std::collections::HashSet<i32>>, penalty: Option<&std::collections::HashMap<i32, f64>>) -> Option<Vec<i32>> {
        if start == goal { return Some(Vec::new()); }
        let g = self.g.clone();
        let key = start as i64 * g.node_count as i64 + goal as i64;
        let bypass = avoid.is_some() || penalty.is_some();
        if !bypass { if let Some(h) = self.cache.get(&key) { return h.clone(); } }
        self.gen += 1;
        let gen = self.gen;
        let (nx, nz) = (&g.nx, &g.nz);
        let (gx, gz) = (nx[goal as usize], nz[goal as usize]);
        let h = |n: usize| self.weight * jm::hypot(nx[n] - gx, nz[n] - gz) / self.max_speed;
        let mut heap = Heap { k: Vec::new(), v: Vec::new() };
        let s = start as usize;
        self.g_score[s] = 0.0; self.stamp[s] = gen; self.came[s] = -1; self.closed[s] = 0;
        heap.push(h(s), start);
        let mut found = false;
        while !heap.k.is_empty() {
            let n = heap.pop() as usize;
            if self.closed[n] == 1 && self.stamp[n] == gen { continue; }
            self.closed[n] = 1;
            if n == goal as usize { found = true; break; }
            let gn = self.g_score[n];
            let mut banned: Option<&Vec<i32>> = None;
            if !g.banned_hops.is_empty() && n != s {
                let via = self.came[n];
                let pre = g.e_from[via as usize] as usize;
                let inn = if pre != s && self.stamp[pre] == gen { self.came[pre] } else { -1 };
                if inn >= 0 { banned = g.banned_hops.get(&(inn as i64 * 4194304 + via as i64)); }
            }
            for q in g.out_start[n]..g.out_start[n + 1] {
                let ei = g.out_edges[q as usize];
                if let Some(b) = banned { if b.contains(&ei) { continue; } }
                if let Some(a) = avoid { if a.contains(&ei) { continue; } }
                let m = g.e_to[ei as usize] as usize;
                let mut tent = gn + self.e_time[ei as usize] + self.junction_penalty / (1 + g.e_rank[ei as usize]) as f64;
                if let Some(p) = penalty { if let Some(&x) = p.get(&ei) { if x != 0.0 && !x.is_nan() { tent += x; } } }
                if self.stamp[m] != gen || tent < self.g_score[m] {
                    self.stamp[m] = gen; self.closed[m] = 0; self.g_score[m] = tent; self.came[m] = ei;
                    heap.push(tent + h(m), m as i32);
                }
            }
        }
        let result = if found {
            let mut path = Vec::new();
            let mut n = goal as usize;
            while n != s { let ei = self.came[n]; path.push(ei); n = g.e_from[ei as usize] as usize; }
            path.reverse();
            Some(path)
        } else { None };
        if bypass { return result; }
        if self.cache.len() >= self.cache_max { self.cache.clear(); }
        self.cache.insert(key, result.clone());
        result
    }
}

/// The shared router (bayenv_world::nav::Router, the route.js makeRouter port) behind the sim's Router trait.
pub struct NavRouter { r: bayenv_world::nav::router::Router, avoid: Vec<u32>, pen: Vec<(u32, f64)>,
    /// parallel batch routing (with_pool; the pool of search states lives in the nav Router)
    graph: Option<std::sync::Arc<bayenv_world::graph::SimGraph>>, weight: f64, pool_size: usize }

impl NavRouter {
    /// makeRouter(graph, { weight }) over the sim's world.
    pub fn new(w: &World, weight: f64) -> NavRouter { NavRouter::from_sim_graph(&sim_graph_of(w), weight) }
}

/// The routing view of a sim World as a bayenv_world SimGraph.
pub fn sim_graph_of(w: &World) -> bayenv_world::graph::SimGraph {
    {
        let mut g = bayenv_world::graph::SimGraph { node_count: w.node_count, edge_count: w.edge_count, nx: w.nx.clone(), nz: w.nz.clone(), max_speed: w.max_speed, ..Default::default() };
        g.from = w.e_from.iter().map(|&x| x as u32).collect(); g.to = w.e_to.iter().map(|&x| x as u32).collect();
        g.length = w.e_length.clone(); g.speed_limit = w.e_speed.clone(); g.rank = w.e_rank.iter().map(|&x| x as u8).collect();
        g.out_start = w.out_start.iter().map(|&x| x as u32).collect(); g.out_edges = w.out_edges.iter().map(|&x| x as u32).collect();
        g.in_start = w.in_start.iter().map(|&x| x as u32).collect(); g.in_edges = w.in_edges.iter().map(|&x| x as u32).collect();
        g.banned_hops = w.banned_hops.iter().map(|(&k, v)| (k as u64, v.iter().map(|&x| x as u32).collect())).collect();
        g.bounds = bayenv_world::graph::Bounds { min_x: w.bounds.0, max_x: w.bounds.1, min_z: w.bounds.2, max_z: w.bounds.3 };
        g
    }
}

impl NavRouter {
    /// makeRouter(graph, { weight }) directly over the shared graph (no World copy).
    pub fn from_sim_graph(g: &bayenv_world::graph::SimGraph, weight: f64) -> NavRouter {
        let o = bayenv_world::nav::router::RouterOpts { weight, ..Default::default() };
        NavRouter { r: bayenv_world::nav::router::Router::new(g, &o), avoid: Vec::new(), pen: Vec::new(), graph: None, weight, pool_size: 0 }
    }
    /// route_batch runs avoid / penalty queries on up to `threads` routers in parallel (built on first use; each
    /// holds its own search state, ~ one graph's worth of node arrays). Results equal sequential `route`.
    /// Another router on the same search graph and OD cache (own search state): cheap, for one per worker thread.
    pub fn fork(&self) -> NavRouter { NavRouter { r: self.r.fork(), avoid: Vec::new(), pen: Vec::new(), graph: self.graph.clone(), weight: self.weight, pool_size: self.pool_size } }
    pub fn with_pool(mut self, g: std::sync::Arc<bayenv_world::graph::SimGraph>, threads: usize) -> NavRouter { self.graph = Some(g); self.pool_size = threads; self }
}

impl Router for NavRouter {
    fn route(&mut self, start: i32, goal: i32, avoid: Option<&std::collections::HashSet<i32>>, penalty: Option<&std::collections::HashMap<i32, f64>>) -> Option<Vec<i32>> {
        let to_i = |p: Vec<u32>| p.into_iter().map(|x| x as i32).collect::<Vec<i32>>();
        if avoid.is_none() && penalty.is_none() { return self.r.route(start as usize, goal as usize).map(to_i); }
        self.avoid.clear(); self.pen.clear();
        if let Some(a) = avoid { self.avoid.extend(a.iter().map(|&x| x as u32)); }
        if let Some(p) = penalty { self.pen.extend(p.iter().map(|(&e, &v)| (e as u32, v))); }
        let q = bayenv_world::nav::router::RouteQuery { avoid: &self.avoid, penalty: &self.pen };
        self.r.route_with(start as usize, goal as usize, &q).map(to_i)
    }
    fn route_batch(&mut self, queries: &[(i32, i32)], avoid: Option<&std::collections::HashSet<i32>>, penalty: Option<&std::collections::HashMap<i32, f64>>) -> Vec<Option<Vec<i32>>> {
        if self.pool_size < 2 || queries.len() < 4 || self.graph.is_none() { return queries.iter().map(|&(s, g)| self.route(s, g, avoid, penalty)).collect(); }
        // bayenv_world's route_batch: rayon over pooled search states on the router's shared graph (+ OD cache for
        // plain queries); each result equals route_with of the same query
        let a: Vec<u32> = avoid.map(|s| s.iter().map(|&x| x as u32).collect()).unwrap_or_default();
        let p: Vec<(u32, f64)> = penalty.map(|m| m.iter().map(|(&e, &v)| (e as u32, v)).collect()).unwrap_or_default();
        let q = bayenv_world::nav::router::RouteQuery { avoid: &a, penalty: &p };
        let qs: Vec<(usize, usize, bayenv_world::nav::router::RouteQuery)> = queries.iter().map(|&(s, g)| (s as usize, g as usize, q)).collect();
        self.r.route_batch(&qs).into_iter().map(|r| r.map(|v| v.into_iter().map(|x| x as i32).collect())).collect()
    }
}
