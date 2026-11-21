//! A* router: port of `src/sim/route.js makeRouter(graph, opts)` (same as `makeRouter(routingPack(graph))`).
//!
//! Cost of an edge = `length / speedLimit + junctionPenalty / (1 + rank)` (+ live penalty), heuristic
//! `weight * hypot(dx, dz) / maxSpeed`, banned lane paths (`bannedHops`, in -> via -> out), an optional avoid set.
//! Results equal the JS bit for bit, including tie-breaking and "no path".
//!
//! Layout: the immutable search graph ([`RouterGraph`], shared through an `Arc`) is split from the per-query
//! search state ([`SearchState`]), so one graph serves any number of threads: [`Router::fork`] makes another
//! router on the same graph and OD cache, [`Router::route_batch`] routes a slice of queries in parallel (rayon)
//! with a pool of search states. Plain queries (no avoid / penalty) go through a shared, bounded, sharded OD
//! cache ([`OdCache`]); a cached result equals a fresh one, as in the JS.
//!
//! Exactness of the fast search (see `rust/ARCHITECTURE.md`, "Router"): the JS pops its binary `MinHeap`, whose
//! order among EQUAL keys depends on the heap's array layout. The fast path uses a 4-ary heap and skips pushing
//! entries whose key exceeds the best goal key in the heap (they can never be popped before the goal), and checks
//! at every pop that the popped key is the unique minimum. If every pop is unique, the pop sequence is a function
//! of the key multiset alone, so it equals the JS binary heap's sequence (the skipped entries are all strictly
//! above every key popped before the goal). If any pop ties, the query is rerun with the exact JS heap (sift order
//! of route.js `MinHeap`, no skipping): ties are rare, so this costs nearly nothing on average.
//! Queries whose goal is in a strongly connected component that cannot be reached from the start's (Tarjan order
//! of the plain graph; bans / avoid only remove edges) return "no path" without a search.
use crate::graph::SimGraph;
use crate::jsmath;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct RouterOpts {
    /// > 1 = weighted A* (gpuSim / sim.js / workers use 1.25)
    pub weight: f64,
    /// seconds per junction, divided by (1 + rank)
    pub junction_penalty: f64,
    /// 0 = graph.maxSpeed
    pub max_speed: f64,
    /// result cache (start, goal) for queries without avoid / penalty; a full shard is cleared; 0 = off. JS default 50000
    pub cache_max: usize,
}
impl Default for RouterOpts { fn default() -> Self { RouterOpts { weight: 1.0, junction_penalty: 2.0, max_speed: 0.0, cache_max: 50000 } } }

/// Live constraints of one query (route.js `avoid` Set and `penalty` Map).
#[derive(Clone, Copy, Debug, Default)]
pub struct RouteQuery<'a> {
    /// edges never used
    pub avoid: &'a [u32],
    /// (edge, extra seconds); a later entry for the same edge replaces an earlier one (Map.set); 0 / NaN = none
    /// (a negative value that makes a cycle's cost negative never terminates, as in the JS)
    pub penalty: &'a [(u32, f64)],
}
impl RouteQuery<'_> {
    #[inline] pub fn is_plain(&self) -> bool { self.avoid.is_empty() && self.penalty.is_empty() }
}
/// No avoid set, no penalty map.
pub const PLAIN: RouteQuery<'static> = RouteQuery { avoid: &[], penalty: &[] };

/// One out-edge in CSR order, with the head node's coordinates (the heuristic needs no second random access).
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Adj { time: f64, x: f64, z: f64, to: u32, edge: u32, span: u32, rank: u8 }
/// Heap entry: key, node, the node's out-edge span (`lo << 8 | degree`, or NO_SPAN: read out_start).
#[derive(Clone, Copy)]
#[repr(C)]
struct Ent { k: f64, v: u32, span: u32 }
const NO_SPAN: u32 = u32::MAX;

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "aarch64")]
    unsafe { std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)); }
    #[cfg(target_arch = "x86_64")]
    unsafe { std::arch::x86_64::_mm_prefetch(p as *const i8, std::arch::x86_64::_MM_HINT_T0); }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = p;
}
/// `stamp = gen << 1 | closed`
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct NodeSt { g: f64, came: i32, stamp: u32 }
#[derive(Clone, Copy, Default)]
struct EdgeMark { avoid: u32, pen: u32, p: f64 }

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

/// Immutable search graph (internal Morton node numbering), shared by every router / thread built from it.
pub struct RouterGraph {
    n: usize, e_n: usize,
    xz: Vec<[f64; 2]>,
    out_start: Vec<u32>, adj: Vec<Adj>, spans: Vec<u32>,
    /// junction penalty by rank
    jp: Box<[f64; 256]>,
    /// from-node (internal numbering) per edge
    e_from: Vec<u32>,
    /// graph node -> internal node
    new_of: Vec<u32>,
    weight: f64, max_speed: f64,
    banned: HashMap<u64, Vec<u32>>,
    /// any banned (in, via) key with this via edge
    banned_via: Vec<bool>,
    /// Tarjan SCC index per internal node: an edge a -> b implies comp[b] <= comp[a]
    comp: Vec<u32>,
    /// all costs and coordinates finite (the fast path's tie argument needs NaN-free keys)
    finite: bool,
}

impl RouterGraph {
    pub fn new(g: &SimGraph, o: &RouterOpts) -> RouterGraph {
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
            for &e in g.out_of(on as usize) { let e = e as usize; let t = g.to[e] as usize;
                adj.push(Adj { time: g.length[e] / g.speed_limit[e], rank: g.rank[e], x: g.nx[t], z: g.nz[t], to: new_of[t], edge: e as u32, span: 0 }); }
            out_start.push(adj.len() as u32);
        }
        let span_of = |m: usize| { let (lo, hi) = (out_start[m], out_start[m + 1]); if lo < (1 << 24) && hi - lo < 256 { lo << 8 | (hi - lo) } else { NO_SPAN } };
        let spans: Vec<u32> = (0..n).map(span_of).collect();
        for a in adj.iter_mut() { a.span = spans[a.to as usize]; }
        let xz: Vec<[f64; 2]> = old_of.iter().map(|&i| [g.nx[i as usize], g.nz[i as usize]]).collect();
        // junctionPenalty / (1 + rank) per rank value (same expression as the JS, so the same bits)
        let jp: Box<[f64; 256]> = Box::new(std::array::from_fn(|r| o.junction_penalty / (1.0 + r as f64)));
        let finite = o.weight.is_finite() && adj.iter().all(|a| a.time.is_finite() && a.time.abs() < 1e12 && jp[a.rank as usize].is_finite() && jp[a.rank as usize].abs() < 1e12)
            && xz.iter().all(|p| p[0].is_finite() && p[1].is_finite());
        let comp = tarjan(n, &out_start, &adj);
        RouterGraph { n, e_n, xz, out_start, adj, spans, jp, e_from: g.from.iter().map(|&f| new_of[f as usize]).collect(), new_of,
            weight: o.weight, max_speed, banned: g.banned_hops.clone(), banned_via, comp, finite }
    }
    pub fn node_count(&self) -> usize { self.n }
    /// True when `goal` is provably unreachable from `start` (internal ids) in the plain graph.
    #[inline] fn unreachable(&self, s: usize, g: usize) -> bool { self.comp[g] > self.comp[s] }
}

/// Iterative Tarjan: SCC index per node, components numbered in completion (reverse topological) order.
fn tarjan(n: usize, os: &[u32], adj: &[Adj]) -> Vec<u32> {
    const NONE: u32 = u32::MAX;
    let mut index = vec![NONE; n]; let mut low = vec![0u32; n]; let mut comp = vec![NONE; n];
    let mut on = vec![false; n]; let mut stack: Vec<u32> = Vec::new(); let mut call: Vec<(u32, u32)> = Vec::new();
    let (mut idx, mut count) = (0u32, 0u32);
    for root in 0..n {
        if index[root] != NONE { continue; }
        index[root] = idx; low[root] = idx; idx += 1; stack.push(root as u32); on[root] = true; call.push((root as u32, os[root]));
        while let Some(top) = call.last_mut() {
            let v = top.0 as usize;
            if top.1 < os[v + 1] {
                let w = adj[top.1 as usize].to as usize; top.1 += 1;
                if index[w] == NONE { index[w] = idx; low[w] = idx; idx += 1; stack.push(w as u32); on[w] = true; call.push((w as u32, os[w])); }
                else if on[w] && index[w] < low[v] { low[v] = index[w]; }
            } else {
                if low[v] == index[v] { loop { let w = stack.pop().unwrap() as usize; on[w] = false; comp[w] = count; if w == v { break; } } count += 1; }
                call.pop();
                if let Some(&(u, _)) = call.last() { let u = u as usize; if low[v] < low[u] { low[u] = low[v]; } }
            }
        }
    }
    comp
}

/// Mutable per-query state (one per thread). 16 bytes per node, plus 16 bytes per edge once a query uses avoid /
/// penalty.
pub struct SearchState {
    st: Vec<NodeSt>, gen: u32,
    heap: Vec<Ent>,
    marks: Vec<EdgeMark>, qgen: u32,
    ban_buf: Vec<u32>,
    /// heap pops of the last query (diagnostics)
    pub last_pops: usize,
    /// queries rerun with the exact JS heap because the fast search saw a key tie
    pub fallbacks: u64,
    /// always use the exact JS heap (tests of the fallback path)
    #[doc(hidden)]
    pub exact_only: bool,
}

impl SearchState {
    pub fn new(g: &RouterGraph) -> SearchState {
        SearchState { st: vec![NodeSt::default(); g.n], gen: 0, heap: Vec::with_capacity(1 << 14), marks: Vec::new(), qgen: 0, ban_buf: Vec::new(), last_pops: 0, fallbacks: 0, exact_only: false }
    }

    /// `route(start, goal, avoid, penalty)` without the cache: path into `out` (cleared); false = no path.
    pub fn route_into(&mut self, g: &RouterGraph, start: usize, goal: usize, q: &RouteQuery, out: &mut Vec<u32>) -> bool {
        out.clear();
        if start == goal { return true; }
        let (ua, up) = (!q.avoid.is_empty(), !q.penalty.is_empty());
        if ua || up {
            if self.marks.len() != g.e_n { self.marks = vec![EdgeMark::default(); g.e_n]; }
            self.qgen = self.qgen.wrapping_add(1);
            if self.qgen == 0 { self.marks.fill(EdgeMark::default()); self.qgen = 1; }
            let qg = self.qgen;
            for &e in q.avoid { if let Some(m) = self.marks.get_mut(e as usize) { m.avoid = qg; } }
            for &(e, p) in q.penalty { if let Some(m) = self.marks.get_mut(e as usize) { m.pen = qg; m.p = p; } }
        }
        let (s, t) = (g.new_of[start] as usize, g.new_of[goal] as usize);
        if g.unreachable(s, t) { self.last_pops = 0; return false; }
        let fast = !self.exact_only && g.finite && q.penalty.iter().all(|&(_, p)| p.is_nan() || (p.is_finite() && p.abs() < 1e12));
        if fast {
            let r = match (ua, up) {
                (false, false) => self.search::<true, false, false>(g, s, t, out),
                (true, false) => self.search::<true, true, false>(g, s, t, out),
                (false, true) => self.search::<true, false, true>(g, s, t, out),
                (true, true) => self.search::<true, true, true>(g, s, t, out),
            };
            if let Some(found) = r { return found; }
            self.fallbacks += 1; out.clear();
        }
        let r = match (ua, up) {
            (false, false) => self.search::<false, false, false>(g, s, t, out),
            (true, false) => self.search::<false, true, false>(g, s, t, out),
            (false, true) => self.search::<false, false, true>(g, s, t, out),
            (true, true) => self.search::<false, true, true>(g, s, t, out),
        };
        r.unwrap()
    }

    /// route.js `MinHeap.push` (binary, exact sift order).
    #[inline]
    fn push2(heap: &mut Vec<Ent>, e: Ent) {
        let mut i = heap.len(); heap.push(e);
        while i > 0 {
            let p = (i - 1) >> 1;
            if heap[p].k <= e.k { break; }
            heap[i] = heap[p]; i = p;
        }
        heap[i] = e;
    }
    /// route.js `MinHeap.pop`.
    #[inline]
    fn pop2(heap: &mut Vec<Ent>) -> Ent {
        let top = heap[0];
        let last = heap.pop().unwrap();
        let n = heap.len();
        if n > 0 {
            let lk = last.k; let mut i = 0;
            loop {
                let l = 2 * i + 1; let r = l + 1;
                let (mut m, mut mk) = (i, lk);
                if l < n && heap[l].k < mk { m = l; mk = heap[l].k; }
                if r < n && heap[r].k < mk { m = r; }
                if m == i { break; }
                heap[i] = heap[m]; i = m;
            }
            heap[i] = last;
        }
        top
    }
    /// 4-ary min-heap (any order among equal keys: only used while every pop is a unique minimum).
    #[inline]
    fn push4(heap: &mut Vec<Ent>, e: Ent) {
        let mut i = heap.len(); heap.push(e);
        while i > 0 {
            let p = (i - 1) >> 2;
            if heap[p].k <= e.k { break; }
            heap[i] = heap[p]; i = p;
        }
        heap[i] = e;
    }
    #[inline]
    fn pop4(heap: &mut Vec<Ent>) -> Ent {
        let top = heap[0];
        let last = heap.pop().unwrap();
        let n = heap.len();
        if n > 0 {
            let lk = last.k; let mut i = 0;
            loop {
                let c = 4 * i + 1;
                if c >= n { break; }
                let (m, mk);
                if c + 3 < n {
                    let (k0, k1, k2, k3) = (heap[c].k, heap[c + 1].k, heap[c + 2].k, heap[c + 3].k);
                    let (a, ak) = if k1 < k0 { (c + 1, k1) } else { (c, k0) };
                    let (b, bk) = if k3 < k2 { (c + 3, k3) } else { (c + 2, k2) };
                    if bk < ak { m = b; mk = bk; } else { m = a; mk = ak; }
                } else {
                    let (mut mm, mut kk) = (c, heap[c].k);
                    for j in c + 1..n { if heap[j].k < kk { mm = j; kk = heap[j].k; } }
                    m = mm; mk = kk;
                }
                if mk >= lk { break; }
                heap[i] = heap[m]; i = m;
            }
            heap[i] = last;
        }
        top
    }

    /// One A* search on internal ids. FAST: 4-ary heap, goal-bound push skipping, tie check (None = a pop tied,
    /// rerun exact). !FAST: the JS heap, never None.
    #[inline(never)]
    fn search<const FAST: bool, const AV: bool, const PEN: bool>(&mut self, g: &RouterGraph, start: usize, goal: usize, out: &mut Vec<u32>) -> Option<bool> {
        self.gen += 1;
        if self.gen >= 0x7FFF_FFFF { for s in self.st.iter_mut() { s.stamp = 0; } self.gen = 1; }
        let gen = self.gen; let open = gen << 1; let closed = open | 1; let qg = self.qgen;
        let (w, ms) = (g.weight, g.max_speed);
        let [gx, gz] = g.xz[goal];
        let h = |x: f64, z: f64| w * hypot2(x - gx, z - gz) / ms;
        let (st, heap, marks, ban_buf) = (&mut self.st, &mut self.heap, &self.marks, &mut self.ban_buf);
        let (adj, out_start, jp) = (&g.adj, &g.out_start, &*g.jp);
        heap.clear();
        st[start] = NodeSt { g: 0.0, came: -1, stamp: open };
        let [sx, sz] = g.xz[start];
        heap.push(Ent { k: h(sx, sz), v: start as u32, span: g.spans[start] });
        let mut found = false; let mut pops = 0usize;
        let has_banned = !g.banned.is_empty();
        // smallest key of a goal entry in the heap; FAST skips pushes above it
        let mut bound = f64::INFINITY;
        let mut tied = false;
        while !heap.is_empty() {
            let e = if FAST { Self::pop4(heap) } else { Self::pop2(heap) };
            let n = e.v as usize; pops += 1;
            if FAST { if let Some(t) = heap.first() { if t.k == e.k { tied = true; break; } } }
            let s = &mut st[n];
            if s.stamp == closed { continue; }
            s.stamp = closed;
            if n == goal { found = true; break; }
            let gn = s.g;
            let mut use_ban = false;
            if has_banned && n != start {
                let via = s.came as usize;
                if g.banned_via[via] {
                    let pre = g.e_from[via] as usize;
                    let inn = if pre != start && (st[pre].stamp >> 1) == gen { st[pre].came } else { -1 };
                    if inn >= 0 { if let Some(b) = g.banned.get(&(inn as u64 * 4194304 + via as u64)) { ban_buf.clear(); ban_buf.extend_from_slice(b); use_ban = true; } }
                }
            }
            let (lo, hi) = if e.span != NO_SPAN { ((e.span >> 8) as usize, ((e.span >> 8) + (e.span & 255)) as usize) } else { (out_start[n] as usize, out_start[n + 1] as usize) };
            for a in &adj[lo..hi] {
                let ei = a.edge as usize;
                if use_ban && ban_buf.contains(&a.edge) { continue; }
                if AV && marks[ei].avoid == qg { continue; }
                let m = a.to as usize;
                let mut tentative = gn + a.time + jp[a.rank as usize];
                if PEN { let mk = &marks[ei]; if mk.pen == qg { let p = mk.p; if p != 0.0 && !p.is_nan() { tentative += p; } } }
                let sm = &mut st[m];
                if (sm.stamp >> 1) != gen || tentative < sm.g {
                    *sm = NodeSt { g: tentative, came: ei as i32, stamp: open };
                    let key = tentative + h(a.x, a.z);
                    if FAST {
                        if m == goal { if key < bound { bound = key; } }
                        else if key > bound { continue; }
                        if a.span != NO_SPAN { prefetch(adj.as_ptr().wrapping_add((a.span >> 8) as usize)); }
                        Self::push4(heap, Ent { k: key, v: m as u32, span: a.span });
                    } else { Self::push2(heap, Ent { k: key, v: m as u32, span: a.span }); }
                }
            }
        }
        self.last_pops = pops;
        if tied { return None; }
        if found {
            let mut n = goal;
            while n != start { let ei = st[n].came as usize; out.push(ei as u32); n = g.e_from[ei] as usize; }
            out.reverse();
        }
        Some(found)
    }
}

/// Shared OD cache (start, goal) -> route for plain queries: thread-safe (64 mutex shards), bounded (a shard
/// holding `max / 64` entries is cleared before the next insert, like the JS `cache.clear()` when full).
pub struct OdCache {
    shards: Box<[Mutex<HashMap<u64, Option<Arc<[u32]>>>>]>,
    per_shard: usize,
    pub hits: AtomicU64, pub misses: AtomicU64,
}
const SHARDS: usize = 64;
impl OdCache {
    pub fn new(max: usize) -> OdCache {
        OdCache { shards: (0..SHARDS).map(|_| Mutex::new(HashMap::new())).collect(), per_shard: max.div_ceil(SHARDS).max(1), hits: AtomicU64::new(0), misses: AtomicU64::new(0) }
    }
    #[inline] fn shard(&self, key: u64) -> &Mutex<HashMap<u64, Option<Arc<[u32]>>>> { &self.shards[(key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58) as usize] }
    pub fn get(&self, key: u64) -> Option<Option<Arc<[u32]>>> {
        let r = self.shard(key).lock().unwrap().get(&key).cloned();
        if r.is_some() { self.hits.fetch_add(1, Ordering::Relaxed); } else { self.misses.fetch_add(1, Ordering::Relaxed); }
        r
    }
    pub fn insert(&self, key: u64, v: Option<Arc<[u32]>>) {
        let mut m = self.shard(key).lock().unwrap();
        if m.len() >= self.per_shard && !m.contains_key(&key) { m.clear(); }
        m.insert(key, v);
    }
    pub fn clear(&self) { for s in self.shards.iter() { s.lock().unwrap().clear(); } }
    pub fn len(&self) -> usize { self.shards.iter().map(|s| s.lock().unwrap().len()).sum() }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

/// A router: shared graph + OD cache, its own search state, and a pool of states for [`Router::route_batch`].
pub struct Router {
    graph: Arc<RouterGraph>,
    cache: Option<Arc<OdCache>>,
    st: SearchState,
    pool: Mutex<Vec<SearchState>>,
    /// heap pops of the last single query (diagnostics)
    pub last_pops: usize,
}

impl Router {
    pub fn new(g: &SimGraph, o: &RouterOpts) -> Router {
        let graph = Arc::new(RouterGraph::new(g, o));
        let cache = if o.cache_max > 0 { Some(Arc::new(OdCache::new(o.cache_max))) } else { None };
        Router::from_shared(graph, cache)
    }
    /// A router on an existing graph / cache (no copy).
    pub fn from_shared(graph: Arc<RouterGraph>, cache: Option<Arc<OdCache>>) -> Router {
        let st = SearchState::new(&graph);
        Router { graph, cache, st, pool: Mutex::new(Vec::new()), last_pops: 0 }
    }
    /// Another router on the same graph and OD cache (own search state): one per thread.
    pub fn fork(&self) -> Router { Router::from_shared(self.graph.clone(), self.cache.clone()) }
    pub fn graph(&self) -> &Arc<RouterGraph> { &self.graph }
    pub fn cache(&self) -> Option<&Arc<OdCache>> { self.cache.as_ref() }
    pub fn node_count(&self) -> usize { self.graph.n }
    pub fn clear_cache(&mut self) { if let Some(c) = &self.cache { c.clear(); } }
    /// Fast-path ties rerun with the exact heap (this router's own state; diagnostics).
    pub fn fallbacks(&self) -> u64 { self.st.fallbacks }
    /// Tests: single queries always take the exact JS-heap search.
    #[doc(hidden)]
    pub fn set_exact_only(&mut self, on: bool) { self.st.exact_only = on; }

    #[inline] fn key(&self, start: usize, goal: usize) -> u64 { start as u64 * self.graph.n as u64 + goal as u64 }

    /// `route(start, goal)`: edge path or None (no path). Cached when `cache_max > 0`.
    pub fn route(&mut self, start: usize, goal: usize) -> Option<Vec<u32>> {
        if start == goal { return Some(Vec::new()); }
        let key = self.key(start, goal);
        if let Some(c) = &self.cache { if let Some(h) = c.get(key) { return h.map(|p| p.to_vec()); } }
        let mut out = Vec::new();
        let ok = self.st.route_into(&self.graph, start, goal, &PLAIN, &mut out);
        self.last_pops = self.st.last_pops;
        if let Some(c) = &self.cache { c.insert(key, if ok { Some(out.as_slice().into()) } else { None }); }
        if ok { Some(out) } else { None }
    }

    /// `route(start, goal, avoid, penalty)` (never cached, as in the JS).
    pub fn route_with(&mut self, start: usize, goal: usize, q: &RouteQuery) -> Option<Vec<u32>> {
        let mut out = Vec::new();
        if self.route_into(start, goal, q, &mut out) { Some(out) } else { None }
    }

    /// Allocation-free variant (no cache): writes the path into `out` (cleared); false = no path.
    pub fn route_into(&mut self, start: usize, goal: usize, q: &RouteQuery, out: &mut Vec<u32>) -> bool {
        let r = self.st.route_into(&self.graph, start, goal, q, out);
        self.last_pops = self.st.last_pops;
        r
    }

    /// Routes every `(start, goal, query)` in parallel (rayon, pooled search states, one per worker); result `i`
    /// equals `route_with(queries[i])`, order preserved. Plain queries (empty avoid and penalty) are answered
    /// from the OD cache when present, searched once per distinct (start, goal) in the batch, and cached.
    pub fn route_batch(&self, queries: &[(usize, usize, RouteQuery)]) -> Vec<Option<Vec<u32>>> {
        use rayon::prelude::*;
        struct Lease<'a> { pool: &'a Mutex<Vec<SearchState>>, st: Option<SearchState> }
        impl Drop for Lease<'_> { fn drop(&mut self) { if let Some(s) = self.st.take() { self.pool.lock().unwrap().push(s); } } }
        let g = &*self.graph;
        // cheap pass: trivial queries, cache hits, duplicate plain ODs (-> index of the first)
        let mut res: Vec<Option<Option<Vec<u32>>>> = vec![None; queries.len()];
        let mut todo: Vec<usize> = Vec::new(); let mut dup: Vec<(usize, usize)> = Vec::new();
        let mut first: HashMap<u64, usize> = HashMap::new();
        for (i, &(s, t, ref q)) in queries.iter().enumerate() {
            if s == t { res[i] = Some(Some(Vec::new())); continue; }
            if q.is_plain() {
                let key = self.key(s, t);
                if let Some(c) = &self.cache { if let Some(h) = c.get(key) { res[i] = Some(h.map(|p| p.to_vec())); continue; } }
                if let Some(&j) = first.get(&key) { dup.push((i, j)); continue; }
                first.insert(key, i);
            }
            todo.push(i);
        }
        // one worker per rayon thread (at most), each with one pooled search state, pulling queries from a shared
        // counter: the pool never holds more states than threads (rayon map_init would make one per split)
        let workers = rayon::current_num_threads().min(todo.len()).max(1);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let parts: Vec<Vec<(usize, Option<Vec<u32>>)>> = (0..workers).into_par_iter().map(|_| {
            let mut lease = Lease { pool: &self.pool, st: Some(self.pool.lock().unwrap().pop().unwrap_or_else(|| SearchState::new(g))) };
            let st = lease.st.as_mut().unwrap();
            let mut v = Vec::new();
            loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= todo.len() { break; }
                let (s, t, ref q) = queries[todo[k]];
                let mut out = Vec::new();
                let ok = st.route_into(g, s, t, q, &mut out);
                v.push((k, if ok { Some(out) } else { None }));
            }
            v
        }).collect();
        let mut found: Vec<Option<Vec<u32>>> = vec![None; todo.len()];
        for p in parts { for (k, r) in p { found[k] = r; } }
        for (&i, r) in todo.iter().zip(found) {
            let (s, t, ref q) = queries[i];
            if q.is_plain() { if let Some(c) = &self.cache { c.insert(self.key(s, t), r.as_deref().map(Arc::from)); } }
            res[i] = Some(r);
        }
        for (i, j) in dup { res[i] = res[j].clone(); }
        res.into_iter().map(|r| r.unwrap()).collect()
    }
    /// Pooled search-state fast-path fallbacks so far (diagnostics).
    pub fn batch_fallbacks(&self) -> u64 { self.pool.lock().unwrap().iter().map(|s| s.fallbacks).sum() }
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
