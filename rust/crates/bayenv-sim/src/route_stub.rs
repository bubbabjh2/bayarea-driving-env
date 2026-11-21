//! TEMPORARY stand-in for src/sim/route.js makeRouter (weighted A*, banned hops, avoid set, traffic penalty),
//! used by the differential tests until bayenv_world::nav::Router lands (routing is owned by the world crate).
use crate::jm;
use crate::sim::Router;
use crate::world::World;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub struct AStar { g: Arc<World>, e_time: Vec<f64>, g_score: Vec<f64>, came: Vec<i32>, stamp: Vec<i32>, closed: Vec<u8>, gen: i32, max_speed: f64, weight: f64, junction_penalty: f64,
    cache: HashMap<i64, Option<Vec<i32>>>, cache_max: usize }

impl AStar {
    pub fn new(g: Arc<World>, weight: f64) -> AStar {
        let n = g.node_count;
        let e_time = (0..g.edge_count).map(|e| g.e_length[e] / g.e_speed[e]).collect();
        let mut ms: f64 = g.max_speed;
        if ms == 0.0 { for e in 0..g.edge_count { if g.e_speed[e] > ms { ms = g.e_speed[e]; } } }
        AStar { e_time, g_score: vec![0.0; n], came: vec![0; n], stamp: vec![0; n], closed: vec![0; n], gen: 0, max_speed: jm::max(1.0, ms), weight, junction_penalty: 2.0, cache: HashMap::new(), cache_max: 50000, g }
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
    fn route(&mut self, start: i32, goal: i32, avoid: Option<&HashSet<i32>>, penalty: Option<&HashMap<i32, f64>>) -> Option<Vec<i32>> {
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
