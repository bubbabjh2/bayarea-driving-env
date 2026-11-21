//! A* router: port of `src/sim/route.js makeRouter(graph, opts)` (same as `makeRouter(routingPack(graph))`).
//!
//! Cost of an edge = `length / speedLimit + junctionPenalty / (1 + rank)` (+ live penalty), heuristic
//! `weight * hypot(dx, dz) / maxSpeed`, banned lane paths (`bannedHops`, in -> via -> out), an optional avoid set.
//! Results equal the JS bit for bit, including tie-breaking: the binary heap is route.js `MinHeap` (keys f64,
//! same sift order), relaxation order = out-edge CSR order, the same floating-point expression order.
//!
//! All search state is preallocated (node state and out-edges interleaved for locality); a query allocates only
//! its result ([`Router::route_into`] not even that). Avoid / penalty sets are marked into per-edge
//! generation-stamped arrays (no hashing on the hot path).
use crate::graph::SimGraph;
use crate::jsmath;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct RouterOpts {
    /// > 1 = weighted A* (gpuSim / sim.js / workers use 1.25)
    pub weight: f64,
    /// seconds per junction, divided by (1 + rank)
    pub junction_penalty: f64,
    /// 0 = graph.maxSpeed
    pub max_speed: f64,
    /// result cache (start, goal) for queries without avoid / penalty; cleared when full; 0 = off. JS default 50000
    pub cache_max: usize,
}
impl Default for RouterOpts { fn default() -> Self { RouterOpts { weight: 1.0, junction_penalty: 2.0, max_speed: 0.0, cache_max: 50000 } } }

/// Live constraints of one query (route.js `avoid` Set and `penalty` Map).
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteQuery<'a> {
    /// edges never used
    pub avoid: &'a [u32],
    /// (edge, extra seconds); a later entry for the same edge replaces an earlier one (Map.set); 0 / NaN = none
    pub penalty: &'a [(u32, f64)],
}

/// One out-edge in CSR order.
#[derive(Clone, Copy, Default)]
struct Adj { to: u32, edge: u32, time: f64, jp: f64 }
#[derive(Clone, Copy, Default)]
struct NodeSt { g: f64, came: i32, stamp: u32, closed: bool }

/// V8 `Math.hypot(a, b)` for finite inputs: the max term divides to exactly 1, its Kahan step has zero
/// compensation, so it reduces to `sqrt(1 + r * r) * max` with r = min / max (bit-identical to [`jsmath::hypot`]).
#[inline]
fn hypot2(a: f64, b: f64) -> f64 {
    let (x, y) = (a.abs(), b.abs());
    let (mx, mn) = if x > y { (x, y) } else { (y, x) };
    if mx == 0.0 { return 0.0; }
    let r = mn / mx;
    (1.0 + r * r).sqrt() * mx
}

pub struct Router {
    n: usize,
    xz: Vec<[f64; 2]>,
    out_start: Vec<u32>, adj: Vec<Adj>,
    /// from-node (internal numbering) per edge
    e_from: Vec<u32>,
    /// graph node -> internal node
    new_of: Vec<u32>,
    weight: f64, max_speed: f64,
    banned: HashMap<u64, Vec<u32>>,
    /// any banned (in, via) key with this via edge
    banned_via: Vec<bool>,
    // search state
    st: Vec<NodeSt>, gen: u32,
    heap: Vec<(f64, u32)>,
    avoid_stamp: Vec<u32>, pen_stamp: Vec<u32>, pen: Vec<f64>, qgen: u32,
    cache: HashMap<u64, Option<Vec<u32>>>, cache_max: usize,
    /// heap pops of the last query (diagnostics)
    pub last_pops: usize,
}

impl Router {
    pub fn new(g: &SimGraph, o: &RouterOpts) -> Router {
        let e_n = g.edge_count; let n = g.node_count;
        let mut max_speed = if o.max_speed > 0.0 { o.max_speed } else { g.max_speed };
        if !(max_speed > 0.0) { max_speed = g.speed_limit.iter().fold(0.0, |m, &v| if v > m { v } else { m }); }
        let max_speed = jsmath::max(1.0, max_speed);
        let mut banned_via = vec![false; e_n];
        for k in g.banned_hops.keys() { let via = (k % 4194304) as usize; if via < e_n { banned_via[via] = true; } }
        // internal node numbering in Morton (Z-curve) order of (x, z) for cache locality. Results do not depend on
        // it: heap order depends only on keys and push order, relaxation order on each node's own CSR order.
        let b = g.bounds;
        let q = |v: f64, lo: f64, hi: f64| -> u64 { let t = if hi > lo { (v - lo) / (hi - lo) } else { 0.0 }; (t.clamp(0.0, 1.0) * 65535.0) as u64 };
        let spread = |mut v: u64| { v &= 0xFFFF; v = (v | (v << 8)) & 0x00FF00FF; v = (v | (v << 4)) & 0x0F0F0F0F; v = (v | (v << 2)) & 0x33333333; (v | (v << 1)) & 0x55555555 };
        let mut order: Vec<(u64, u32)> = (0..n).map(|i| (spread(q(g.nx[i], b.min_x, b.max_x)) | spread(q(g.nz[i], b.min_z, b.max_z)) << 1, i as u32)).collect();
        order.sort_unstable();
        let old_of: Vec<u32> = order.iter().map(|p| p.1).collect();
        let mut new_of = vec![0u32; n]; for (k, &o) in old_of.iter().enumerate() { new_of[o as usize] = k as u32; }
        let mut out_start = Vec::with_capacity(n + 1); let mut adj = Vec::with_capacity(e_n); out_start.push(0u32);
        for &on in &old_of {
            for &e in g.out_of(on as usize) { let e = e as usize;
                adj.push(Adj { to: new_of[g.to[e] as usize], edge: e as u32, time: g.length[e] / g.speed_limit[e], jp: o.junction_penalty / (1.0 + g.rank[e] as f64) }); }
            out_start.push(adj.len() as u32);
        }
        Router {
            n, xz: old_of.iter().map(|&i| [g.nx[i as usize], g.nz[i as usize]]).collect(), out_start, adj,
            e_from: g.from.iter().map(|&f| new_of[f as usize]).collect(), new_of,
            weight: o.weight, max_speed, banned: g.banned_hops.clone(), banned_via,
            st: vec![NodeSt::default(); n], gen: 0, heap: Vec::with_capacity(1 << 16),
            avoid_stamp: vec![0; e_n], pen_stamp: vec![0; e_n], pen: vec![0.0; e_n], qgen: 0,
            cache: HashMap::new(), cache_max: o.cache_max, last_pops: 0,
        }
    }

    pub fn node_count(&self) -> usize { self.n }
    pub fn clear_cache(&mut self) { self.cache.clear(); }

    /// `route(start, goal)`: edge path or None (no path). Cached like the JS when `cache_max > 0`.
    pub fn route(&mut self, start: usize, goal: usize) -> Option<Vec<u32>> {
        if start == goal { return Some(Vec::new()); }
        let key = start as u64 * self.n as u64 + goal as u64;
        if self.cache_max > 0 { if let Some(h) = self.cache.get(&key) { return h.clone(); } }
        let mut out = Vec::new();
        let r = if self.search(start, goal, false, false, &mut out) { Some(out) } else { None };
        if self.cache_max > 0 { if self.cache.len() >= self.cache_max { self.cache.clear(); } self.cache.insert(key, r.clone()); }
        r
    }

    /// `route(start, goal, avoid, penalty)` (never cached, as in the JS).
    pub fn route_with(&mut self, start: usize, goal: usize, q: &RouteQuery) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        if self.route_into(start, goal, q, &mut out) { Some(out) } else { None }
    }

    /// Allocation-free variant (no cache): writes the path into `out` (cleared); false = no path.
    pub fn route_into(&mut self, start: usize, goal: usize, q: &RouteQuery, out: &mut Vec<u32>) -> bool {
        out.clear();
        if start == goal { return true; }
        let (ua, up) = (!q.avoid.is_empty(), !q.penalty.is_empty());
        if ua || up {
            self.qgen = self.qgen.wrapping_add(1);
            if self.qgen == 0 { self.avoid_stamp.fill(0); self.pen_stamp.fill(0); self.qgen = 1; }
            let qg = self.qgen;
            for &e in q.avoid { if (e as usize) < self.avoid_stamp.len() { self.avoid_stamp[e as usize] = qg; } }
            for &(e, p) in q.penalty { if (e as usize) < self.pen.len() { self.pen_stamp[e as usize] = qg; self.pen[e as usize] = p; } }
        }
        self.search(start, goal, ua, up, out)
    }

    #[inline]
    fn push(heap: &mut Vec<(f64, u32)>, k: f64, v: u32) {
        let mut i = heap.len(); heap.push((k, v));
        while i > 0 {
            let p = (i - 1) >> 1;
            if heap[p].0 <= k { break; }
            heap[i] = heap[p]; i = p;
        }
        heap[i] = (k, v);
    }
    #[inline]
    fn pop(heap: &mut Vec<(f64, u32)>) -> u32 {
        let top = heap[0].1;
        let last = heap.pop().unwrap();
        let n = heap.len();
        if n > 0 {
            let lk = last.0; let mut i = 0;
            loop {
                let l = 2 * i + 1; let r = l + 1;
                let (mut m, mut mk) = (i, lk);
                if l < n && heap[l].0 < mk { m = l; mk = heap[l].0; }
                if r < n && heap[r].0 < mk { m = r; }
                if m == i { break; }
                heap[i] = heap[m]; i = m;
            }
            heap[i] = last;
        }
        top
    }

    fn search(&mut self, start: usize, goal: usize, use_avoid: bool, use_pen: bool, out: &mut Vec<u32>) -> bool {
        let (start, goal) = (self.new_of[start] as usize, self.new_of[goal] as usize);
        self.gen += 1;
        if self.gen > 2_000_000_000 { for s in self.st.iter_mut() { s.stamp = 0; } self.gen = 1; }
        let gen = self.gen; let qg = self.qgen;
        let (w, ms) = (self.weight, self.max_speed);
        let [gx, gz] = self.xz[goal];
        let xz = &self.xz;
        let h = |m: usize| { let p = xz[m]; w * hypot2(p[0] - gx, p[1] - gz) / ms };
        let (st, heap, adj, out_start) = (&mut self.st, &mut self.heap, &self.adj, &self.out_start);
        heap.clear();
        st[start] = NodeSt { g: 0.0, came: -1, stamp: gen, closed: false };
        Self::push(heap, h(start), start as u32);
        let mut found = false; let mut pops = 0;
        let has_banned = !self.banned.is_empty();
        let mut ban_buf: Vec<u32> = Vec::new();
        while !heap.is_empty() {
            let n = Self::pop(heap) as usize; pops += 1;
            let s = &mut st[n];
            if s.closed && s.stamp == gen { continue; }
            s.closed = true;
            if n == goal { found = true; break; }
            let gn = s.g;
            let mut use_ban = false;
            if has_banned && n != start {
                let via = s.came as usize;
                if self.banned_via[via] {
                    let pre = self.e_from[via] as usize;
                    let inn = if pre != start && st[pre].stamp == gen { st[pre].came } else { -1 };
                    if inn >= 0 { if let Some(b) = self.banned.get(&(inn as u64 * 4194304 + via as u64)) { ban_buf.clear(); ban_buf.extend_from_slice(b); use_ban = true; } }
                }
            }
            for a in &adj[out_start[n] as usize..out_start[n + 1] as usize] {
                let ei = a.edge as usize;
                if use_ban && ban_buf.contains(&a.edge) { continue; }
                if use_avoid && self.avoid_stamp[ei] == qg { continue; }
                let m = a.to as usize;
                let mut tentative = gn + a.time + a.jp;
                if use_pen && self.pen_stamp[ei] == qg { let p = self.pen[ei]; if p != 0.0 && !p.is_nan() { tentative += p; } }
                let sm = &mut st[m];
                if sm.stamp != gen || tentative < sm.g {
                    *sm = NodeSt { g: tentative, came: ei as i32, stamp: gen, closed: false };
                    Self::push(heap, tentative + h(m), m as u32);
                }
            }
        }
        self.last_pops = pops;
        if found {
            let mut n = goal;
            while n != start { let ei = st[n].came as usize; out.push(ei as u32); n = self.e_from[ei] as usize; }
            out.reverse();
        }
        found
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hypot2_equals_v8_hypot() {
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut rnd = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
        for i in 0..2_000_000 {
            let r = rnd();
            let (a, b) = if i % 3 == 0 { (f64::from_bits(r & 0x7FEF_FFFF_FFFF_FFFF), f64::from_bits(rnd() & 0x7FEF_FFFF_FFFF_FFFF)) }
                else { ((r % 200_000) as f64 * 0.37 - 37000.0, (rnd() % 200_000) as f64 * 0.41 - 41000.0) };
            assert_eq!(super::hypot2(a, b).to_bits(), crate::jsmath::hypot(a, b).to_bits(), "{a} {b}");
        }
    }
}
