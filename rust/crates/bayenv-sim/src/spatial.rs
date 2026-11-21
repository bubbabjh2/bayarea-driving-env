//! src/sim/spatial.js: SpatialHash (wrapped 2-D grid, counting sort) and LaneIndex (per (edge, lane) lists by s).
use crate::vehicle::{VehicleStore, STATE_CRASHED, STATE_DRIVING, MAX_LANES, MAX_OCC};

pub struct SpatialHash {
    pub cell_size: f64, w_bits: u32, w_mask: i32, h_mask: i32,
    /// per cell: first item and item count (only cells touched by the last rebuild are non-zero; JS keeps a dense
    /// CSR over the whole 2^tableBits table, which gives the same per-cell item lists: slots in ascending order)
    beg: Vec<i32>, cnt: Vec<i32>, touched: Vec<u32>, pub items: Vec<i32>, cell_of_item: Vec<i32>,
}

impl SpatialHash {
    pub fn new(cell_size: f64, table_bits: u32, capacity: usize) -> Self {
        let ts = 1usize << table_bits;
        let w_bits = table_bits.div_ceil(2);
        SpatialHash { cell_size, w_bits, w_mask: (1 << w_bits) - 1, h_mask: (1 << (table_bits - w_bits)) - 1,
            beg: vec![0; ts], cnt: vec![0; ts], touched: Vec::new(), items: vec![0; capacity], cell_of_item: vec![0; capacity] }
    }
    #[inline(always)]
    pub fn hash(&self, cx: f64, cz: f64) -> usize {
        let (cx, cz) = (crate::jm::to_i32(cx), crate::jm::to_i32(cz));
        ((cx & self.w_mask) | ((cz & self.h_mask) << self.w_bits)) as usize
    }
    pub fn rebuild(&mut self, vs: &VehicleStore) {
        let high = vs.high;
        if self.items.len() < high { self.items = vec![0; high * 2]; self.cell_of_item = vec![0; high * 2]; }
        for &c in &self.touched { self.cnt[c as usize] = 0; }
        self.touched.clear();
        let inv = 1.0 / self.cell_size;
        for i in 0..high {
            if vs.state[i] != STATE_DRIVING && vs.state[i] != STATE_CRASHED { self.cell_of_item[i] = -1; continue; }
            let c = self.hash((vs.x[i] as f64 * inv).floor(), (vs.z[i] as f64 * inv).floor());
            self.cell_of_item[i] = c as i32;
            if self.cnt[c] == 0 { self.touched.push(c as u32); }
            self.cnt[c] += 1;
        }
        let mut acc = 0i32;
        for &c in &self.touched { let c = c as usize; self.beg[c] = acc; acc += self.cnt[c]; }
        // fill cursor: reuse beg + running count (beg is restored below)
        for i in 0..high { let c = self.cell_of_item[i]; if c >= 0 { let c = c as usize; self.items[self.beg[c] as usize] = i as i32; self.beg[c] += 1; } }
        for &c in &self.touched { let c = c as usize; self.beg[c] -= self.cnt[c]; }
    }
    /// forEachNear: calls f(slot) for every car within r of (x, z) (positions read live from vs), in cell order.
    /// f returns false to stop.
    #[inline]
    pub fn for_each_near(&self, vs_x: &[f32], vs_z: &[f32], x: f64, z: f64, r: f64, mut f: impl FnMut(usize) -> bool) {
        let inv = 1.0 / self.cell_size;
        let cx0 = ((x - r) * inv).floor() as i64; let cx1 = ((x + r) * inv).floor() as i64;
        let cz0 = ((z - r) * inv).floor() as i64; let cz1 = ((z + r) * inv).floor() as i64;
        let r2 = r * r;
        let mut cz = cz0;
        while cz <= cz1 {
            let mut cx = cx0;
            while cx <= cx1 {
                let c = self.hash(cx as f64, cz as f64);
                let b = self.beg[c];
                for k in b..b + self.cnt[c] {
                    let i = self.items[k as usize] as usize;
                    let dx = vs_x[i] as f64 - x; let dz = vs_z[i] as f64 - z;
                    if dx * dx + dz * dz <= r2 && !f(i) { return; }
                }
                cx += 1;
            }
            cz += 1;
        }
    }
    /// Collect forEachNear results (the JS callbacks never touch the hash, so collecting first is equivalent).
    pub fn near_into(&self, vs_x: &[f32], vs_z: &[f32], x: f64, z: f64, r: f64, out: &mut Vec<usize>) {
        out.clear();
        self.for_each_near(vs_x, vs_z, x, z, r, |i| { out.push(i); true });
    }
}

pub struct LaneIndex {
    pub lstart: Vec<i32>, pub lend: Vec<i32>, count: Vec<i32>, touched: Vec<i32>, pub order: Vec<i32>, pub ent_pos: Vec<i32>, n_touched: usize,
}

impl LaneIndex {
    pub fn new(edge_count: usize, capacity: usize) -> Self {
        let keys = edge_count * MAX_LANES as usize;
        LaneIndex { lstart: vec![0; keys], lend: vec![0; keys], count: vec![0; keys], touched: vec![0; capacity * 2], order: vec![0; capacity * 2], ent_pos: vec![0; capacity * MAX_OCC as usize], n_touched: 0 }
    }
    pub fn rebuild(&mut self, vs: &VehicleStore) {
        let high = vs.high;
        let mut total = 0usize;
        for i in 0..high { let st = vs.state[i]; if st == STATE_DRIVING || st == STATE_CRASHED { total = (total as i64 + vs.lane_hi[i] as i64 - vs.lane_lo[i] as i64 + 1) as usize; } }
        if self.order.len() < total { self.order = vec![0; total * 2]; }
        if self.touched.len() < total { self.touched.resize(total * 2, 0); }
        if self.ent_pos.len() < high * MAX_OCC as usize { self.ent_pos = vec![0; high * MAX_OCC as usize * 2]; }
        for q in 0..self.n_touched { let k = self.touched[q] as usize; self.lstart[k] = 0; self.lend[k] = 0; }
        let mut nt = 0usize;
        for i in 0..high {
            let st = vs.state[i]; if st != STATE_DRIVING && st != STATE_CRASHED { continue; }
            let base = vs.edge[i] as i64 * MAX_LANES as i64;
            let mut l = vs.lane_lo[i] as i64;
            while l <= vs.lane_hi[i] as i64 { let k = (base + l) as usize; if self.count[k] == 0 { self.touched[nt] = k as i32; nt += 1; } self.count[k] += 1; l += 1; }
        }
        // load-bearing: the lane ranges laid out in key order (dropping this sort breaks the obs / v2 traces)
        self.touched[..nt].sort_unstable();
        let mut acc = 0i32;
        for q in 0..nt { let k = self.touched[q] as usize; self.lstart[k] = acc; acc += self.count[k]; self.lend[k] = self.lstart[k]; }
        for i in 0..high {
            let st = vs.state[i]; if st != STATE_DRIVING && st != STATE_CRASHED { continue; }
            let base = vs.edge[i] as i64 * MAX_LANES as i64;
            let mut l = vs.lane_lo[i] as i64;
            while l <= vs.lane_hi[i] as i64 { let k = (base + l) as usize; self.order[self.lend[k] as usize] = i as i32; self.lend[k] += 1; l += 1; }
        }
        let s = &vs.s;
        for q in 0..nt {
            let k = self.touched[q] as usize;
            let (a, b) = (self.lstart[k] as usize, self.lend[k] as usize);
            let lane = (k % MAX_LANES as usize) as i64;
            for i in a + 1..b {
                let v = self.order[i]; let sv = s[v as usize];
                let mut j = i as i64 - 1;
                while j >= a as i64 { let oj = self.order[j as usize]; let so = s[oj as usize]; if so > sv || (so == sv && oj > v) { self.order[j as usize + 1] = oj; j -= 1; } else { break; } }
                self.order[(j + 1) as usize] = v;
            }
            for i in a..b { let v = self.order[i] as usize; self.ent_pos[(v as i64 * MAX_OCC as i64 + lane - vs.lane_lo[v] as i64) as usize] = i as i32; }
            self.count[k] = 0;
        }
        self.n_touched = nt;
    }
    #[inline]
    pub fn ahead(&self, vs: &VehicleStore, i: usize, lane: i32) -> i32 {
        if lane < vs.lane_lo[i] as i32 || lane > vs.lane_hi[i] as i32 { return -1; }
        let k = (vs.edge[i] * MAX_LANES + lane) as usize;
        let p = self.ent_pos[i * MAX_OCC as usize + (lane - vs.lane_lo[i] as i32) as usize] + 1;
        if p < self.lend[k] { self.order[p as usize] } else { -1 }
    }
    #[inline]
    pub fn behind(&self, vs: &VehicleStore, i: usize, lane: i32) -> i32 {
        if lane < vs.lane_lo[i] as i32 || lane > vs.lane_hi[i] as i32 { return -1; }
        let k = (vs.edge[i] * MAX_LANES + lane) as usize;
        let p = self.ent_pos[i * MAX_OCC as usize + (lane - vs.lane_lo[i] as i32) as usize] - 1;
        if p >= self.lstart[k] { self.order[p as usize] } else { -1 }
    }
    #[inline]
    fn key(edge: i32, lane: i32) -> usize { (edge * MAX_LANES + lane) as usize }
    #[inline] pub fn first(&self, edge: i32, lane: i32) -> i32 { let k = Self::key(edge, lane); if self.lend[k] > self.lstart[k] { self.order[self.lstart[k] as usize] } else { -1 } }
    #[inline] pub fn last(&self, edge: i32, lane: i32) -> i32 { let k = Self::key(edge, lane); if self.lend[k] > self.lstart[k] { self.order[self.lend[k] as usize - 1] } else { -1 } }
    #[inline] pub fn range(&self, edge: i32, lane: i32) -> (usize, usize) { let k = Self::key(edge, lane); (self.lstart[k] as usize, self.lend[k] as usize) }
    /// neighborsAt: [first with s >= s (ahead), last with s < s (behind)], skipping `me`.
    pub fn neighbors_at(&self, vs: &VehicleStore, edge: i32, lane: i32, s: f64, me: i32) -> (i32, i32) {
        let k = Self::key(edge, lane);
        let (a, b) = (self.lstart[k] as usize, self.lend[k] as usize);
        let (mut lo, mut hi) = (a, b);
        while lo < hi { let m = (lo + hi) >> 1; if (vs.s[self.order[m] as usize] as f64) < s { lo = m + 1; } else { hi = m; } }
        let mut p = lo; while p < b && self.order[p] == me { p += 1; }
        let mut q = lo as i64 - 1; while q >= a as i64 && self.order[q as usize] == me { q -= 1; }
        (if p < b { self.order[p] } else { -1 }, if q >= a as i64 { self.order[q as usize] } else { -1 })
    }
}
