//! The sim's directed road graph (`src/sim/roadGraph.js` `fromRoadSet` + `finishGraph`), struct-of-arrays.
//!
//! Numbering = `fromRoadSet`: main-tier edges keep their ids, local edges follow at `main_edge_count + e`;
//! nodes as [`crate::lanegraph::merged_node_maps`]. Floating-point fields keep the JS precision (f64 where the
//! JS holds a double, f32 where it holds a Float32Array value) so ports can be bit-exact.
use crate::jsmath;
use crate::lanegraph::merged_node_maps;
use crate::roadpack::{RoadSet, CLASSES_ALL};
use std::collections::HashMap;

pub const LANE_WIDTH: f64 = 3.7;
pub const RAMP_M: f64 = 140.0;
/// `HIGHWAY_RANK` per class id.
pub const HIGHWAY_RANK: [u8; 13] = [7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 0];
/// `CLASS_RANK` (nav rank) per class id.
pub const CLASS_RANK: [u8; 13] = [9, 7, 8, 6, 5, 4, 3, 3, 2, 2, 1, 1, 0];

/// Edge flag bits of [`SimGraph::eflags`].
pub mod ef {
    pub const ONEWAY: u8 = 1; pub const BRIDGE: u8 = 2; pub const TUNNEL: u8 = 4; pub const LINK: u8 = 8;
    /// freeway system: motorway, motorway_link, trunk, trunk_link
    pub const FREEWAY_SYS: u8 = 16;
    /// motorway / trunk mainline
    pub const FREEWAY: u8 = 32;
    /// edge has per-point heights (`ys`), else flat at `y`
    pub const HAS_YS: u8 = 64;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Bounds { pub min_x: f64, pub max_x: f64, pub min_z: f64, pub max_z: f64 }

/// SimGraph (SoA). Per-edge arrays have length `edge_count`; CSR `*_start` arrays have length `n + 1`.
#[derive(Clone, Default)]
pub struct SimGraph {
    pub node_count: usize,
    pub edge_count: usize,
    pub main_edge_count: usize,
    // ---- nodes
    pub nx: Vec<f64>, pub nz: Vec<f64>, pub node_y: Vec<f32>,
    pub node_rank: Vec<i8>, pub node_low_rank: Vec<i8>,
    /// per tier (0 main, 1 local): tier node -> merged node
    pub tier_node_map: Vec<Vec<i32>>,
    // ---- edges
    pub from: Vec<u32>, pub to: Vec<u32>,
    /// opposite direction edge (`e.reverse`), -1 none
    pub reverse: Vec<i32>,
    /// clamped to 1..=8 (`e.lanes`)
    pub lanes: Vec<u8>,
    pub opp_lanes: Vec<u8>,
    /// class id (index into [`CLASSES_ALL`])
    pub class: Vec<u8>,
    pub rank: Vec<u8>, pub nav_rank: Vec<u8>,
    pub eflags: Vec<u8>,
    /// m/s (speedCmps / 100, a JS double)
    pub speed_limit: Vec<f64>,
    /// `max(sum of hypot segment lengths, 0.01)`
    pub length: Vec<f64>,
    pub heading_start: Vec<f64>, pub heading_end: Vec<f64>,
    /// flat edge height (first node y)
    pub y: Vec<f32>,
    /// source edge id within its tier and the tier (0 main / 1 local)
    pub src_edge: Vec<u32>, pub tier: Vec<u8>,
    // ---- polyline points (CSR by edge)
    pub pt_start: Vec<u32>,
    pub px: Vec<f64>, pub pz: Vec<f64>, pub cum: Vec<f64>,
    /// per-point heights (valid for every point; equals `y` for flat edges)
    pub ys: Vec<f32>,
    pub node_idx: Vec<u32>,
    // ---- lanes (CSR by edge): lane centre offsets, right-positive, lane 0 = rightmost
    pub lane_start: Vec<u32>,
    pub lane_off: Vec<f32>,
    /// `laneOff[0] + 3.7 / 2`
    pub right_edge: Vec<f64>,
    // ---- adjacency
    pub out_start: Vec<u32>, pub out_edges: Vec<u32>, pub in_start: Vec<u32>, pub in_edges: Vec<u32>,
    /// key `in * 4194304 + via` -> banned out edges
    pub banned_hops: HashMap<u64, Vec<u32>>,
    // ---- attached later by `World` (surfaces / junctions); defaults: asphalt, no junction
    pub surf: Vec<u8>, pub gnd_r: Vec<u8>, pub gnd_l: Vec<u8>,
    /// junction mouths (`e.sExit`, `e.sStop`) and ramp gore (`e.gore`, -1 none)
    pub s_exit: Vec<f32>, pub s_stop: Vec<f32>, pub gore: Vec<f32>,
    pub bounds: Bounds,
    pub max_speed: f64,
}

impl SimGraph {
    #[inline] pub fn pts(&self, e: usize) -> std::ops::Range<usize> { self.pt_start[e] as usize..self.pt_start[e + 1] as usize }
    #[inline] pub fn lane_offs(&self, e: usize) -> &[f32] { &self.lane_off[self.lane_start[e] as usize..self.lane_start[e + 1] as usize] }
    #[inline] pub fn out_of(&self, n: usize) -> &[u32] { &self.out_edges[self.out_start[n] as usize..self.out_start[n + 1] as usize] }
    #[inline] pub fn in_of(&self, n: usize) -> &[u32] { &self.in_edges[self.in_start[n] as usize..self.in_start[n + 1] as usize] }
    #[inline] pub fn out_degree(&self, n: usize) -> usize { (self.out_start[n + 1] - self.out_start[n]) as usize }
    #[inline] pub fn in_degree(&self, n: usize) -> usize { (self.in_start[n + 1] - self.in_start[n]) as usize }
    #[inline] pub fn has(&self, e: usize, f: u8) -> bool { self.eflags[e] & f != 0 }
    #[inline] pub fn class_name(&self, e: usize) -> &'static str { CLASSES_ALL[self.class[e] as usize] }
    #[inline] pub fn is_link(&self, e: usize) -> bool { self.has(e, ef::LINK) }
    /// `edgeBetween(a, b)`
    pub fn edge_between(&self, a: usize, b: usize) -> i64 {
        for &e in self.out_of(a) { if self.to[e as usize] as usize == b { return e as i64; } } -1
    }

    /// `fromRoadSet(main, local)` (lane geometry / junctions / surfaces are attached by [`crate::World`]).
    pub fn from_road_sets(main: &RoadSet, local: Option<&RoadSet>) -> SimGraph {
        let sets: Vec<&RoadSet> = std::iter::once(main).chain(local).collect();
        let (n, maps) = merged_node_maps(&sets);
        let mut g = SimGraph { node_count: n, nx: vec![0.0; n], nz: vec![0.0; n], node_y: vec![0.0; n], ..Default::default() };
        let mut e_tot = 0; let mut p_tot = 0;
        for (si, rs) in sets.iter().enumerate() {
            for i in 0..rs.node_count { let k = maps[si][i] as usize; g.nx[k] = rs.x(i); g.nz[k] = rs.z(i); g.node_y[k] = rs.xyz[3 * i + 1]; }
            e_tot += rs.edge_count; p_tot += rs.poly.len();
        }
        g.edge_count = e_tot; g.main_edge_count = main.edge_count;
        macro_rules! cap { ($($f:ident),*) => { $( g.$f.reserve(e_tot); )* } }
        cap!(from, to, reverse, lanes, opp_lanes, class, rank, nav_rank, eflags, speed_limit, length, heading_start, heading_end, y, src_edge, tier, right_edge);
        g.pt_start.reserve(e_tot + 1); g.lane_start.reserve(e_tot + 1);
        g.px.reserve(p_tot); g.pz.reserve(p_tot); g.cum.reserve(p_tot); g.ys.reserve(p_tot); g.node_idx.reserve(p_tot);
        g.pt_start.push(0); g.lane_start.push(0);
        let mut base = 0i64;
        for (si, rs) in sets.iter().enumerate() {
            for e in 0..rs.edge_count {
                let nodes = rs.edge_nodes(e); let np = nodes.len();
                let p0 = g.px.len();
                let mut l = 0.0f64;
                for (i, &nd) in nodes.iter().enumerate() {
                    let k = maps[si][nd as usize] as usize;
                    g.node_idx.push(k as u32); g.px.push(g.nx[k]); g.pz.push(g.nz[k]);
                    if i > 0 { l += jsmath::hypot(g.px[p0 + i] - g.px[p0 + i - 1], g.pz[p0 + i] - g.pz[p0 + i - 1]); }
                    g.cum.push(l);
                }
                let fl = rs.flags(e); let c = rs.class_id(e) as usize; let c = if c < 13 { c } else { 10 };
                g.from.push(g.node_idx[p0]); g.to.push(g.node_idx[p0 + np - 1]);
                g.lanes.push(rs.lanes(e).clamp(1, 8) as u8); g.opp_lanes.push(rs.opp_lanes(e) as u8);
                g.class.push(c as u8); g.rank.push(HIGHWAY_RANK[c]); g.nav_rank.push(CLASS_RANK[c]);
                let mut f = 0u8;
                if fl & 1 != 0 { f |= ef::ONEWAY; } if fl & 2 != 0 { f |= ef::BRIDGE; } if fl & 4 != 0 { f |= ef::TUNNEL; }
                if CLASSES_ALL[c].ends_with("_link") { f |= ef::LINK; }
                if c <= 3 { f |= ef::FREEWAY_SYS; } if c == 0 || c == 2 { f |= ef::FREEWAY; }
                g.speed_limit.push(rs.speed_mps(e)); g.length.push(jsmath::max(l, 0.01));
                g.heading_start.push(jsmath::atan2(g.pz[p0 + 1] - g.pz[p0], g.px[p0 + 1] - g.px[p0]));
                g.heading_end.push(jsmath::atan2(g.pz[p0 + np - 1] - g.pz[p0 + np - 2], g.px[p0 + np - 1] - g.px[p0 + np - 2]));
                // baked heights: y = first node y; ys only when not flat
                let y0 = g.node_y[g.node_idx[p0] as usize];
                let flat = (1..np).all(|i| g.node_y[g.node_idx[p0 + i] as usize] == y0);
                g.y.push(y0);
                for i in 0..np { g.ys.push(g.node_y[g.node_idx[p0 + i] as usize]); }
                if !flat { *g.eflags_push(f) |= ef::HAS_YS; } else { g.eflags.push(f); }
                g.pt_start.push(g.px.len() as u32);
                // lane offsets: reversed laneOffsets (lane 0 = rightmost)
                let offs = rs.lane_offsets(e); let nl = *g.lanes.last().unwrap() as usize;
                for li in 0..nl { g.lane_off.push(offs.get(nl - 1 - li).copied().unwrap_or(f32::NAN)); }
                g.lane_start.push(g.lane_off.len() as u32);
                g.right_edge.push(g.lane_off[g.lane_off.len() - nl] as f64 + LANE_WIDTH / 2.0);
                let t = rs.twin(e); g.reverse.push(if t >= 0 { (base + t) as i32 } else { -1 });
                g.src_edge.push(e as u32); g.tier.push(si as u8);
            }
            base += rs.edge_count as i64;
        }
        if let Some(bh) = &main.banned_hops {
            for k in (0..bh.len().saturating_sub(2)).step_by(3) {
                g.banned_hops.entry(bh[k] as u64 * 4194304 + bh[k + 1] as u64).or_default().push(bh[k + 2]);
            }
        }
        g.tier_node_map = maps;
        g.finish();
        g
    }

    fn eflags_push(&mut self, f: u8) -> &mut u8 { self.eflags.push(f); self.eflags.last_mut().unwrap() }

    fn finish(&mut self) {
        let (n, e) = (self.node_count, self.edge_count);
        let mut os = vec![0u32; n + 1]; let mut is = vec![0u32; n + 1];
        for k in 0..e { os[self.from[k] as usize + 1] += 1; is[self.to[k] as usize + 1] += 1; }
        for i in 0..n { os[i + 1] += os[i]; is[i + 1] += is[i]; }
        let (mut oc, mut ic) = (os[..n].to_vec(), is[..n].to_vec());
        let (mut oe, mut ie) = (vec![0u32; e], vec![0u32; e]);
        for k in 0..e {
            let (f, t) = (self.from[k] as usize, self.to[k] as usize);
            oe[oc[f] as usize] = k as u32; oc[f] += 1; ie[ic[t] as usize] = k as u32; ic[t] += 1;
        }
        self.out_start = os; self.in_start = is; self.out_edges = oe; self.in_edges = ie;
        self.node_rank = vec![-1; n]; self.node_low_rank = vec![127; n];
        for k in 0..e { let r = self.rank[k] as i8; for nd in [self.from[k] as usize, self.to[k] as usize] {
            if r > self.node_rank[nd] { self.node_rank[nd] = r; } if r < self.node_low_rank[nd] { self.node_low_rank[nd] = r; } } }
        let mut b = Bounds { min_x: f64::INFINITY, max_x: f64::NEG_INFINITY, min_z: f64::INFINITY, max_z: f64::NEG_INFINITY };
        for i in 0..n { let (x, z) = (self.nx[i], self.nz[i]);
            if x < b.min_x { b.min_x = x; } if x > b.max_x { b.max_x = x; } if z < b.min_z { b.min_z = z; } if z > b.max_z { b.max_z = z; } }
        self.bounds = if n > 0 { b } else { Bounds::default() };
        self.max_speed = self.speed_limit.iter().fold(0.0, |m, &v| if v > m { v } else { m });
        self.surf = vec![0; e]; self.gnd_r = vec![0; e]; self.gnd_l = vec![0; e];
        self.s_exit = vec![0.0; e]; self.s_stop = self.length.iter().map(|&l| l as f32).collect(); self.gore = vec![-1.0; e];
    }

    /// `edgePoint(e, s, offset)`: point s m along edge e, `offset` m to the right. Returns (x, z, y, hx, hz, seg).
    pub fn edge_point(&self, e: usize, s: f64, offset: f64, seg_hint: usize) -> EdgePoint {
        let r = self.pts(e); let (b, n) = (r.start, r.len());
        let (px, pz, cum) = (&self.px[r.clone()], &self.pz[r.clone()], &self.cum[r]);
        let mut i = seg_hint.min(n - 2);
        while i > 0 && cum[i] > s { i -= 1; }
        while i < n - 2 && cum[i + 1] < s { i += 1; }
        let seg = cum[i + 1] - cum[i];
        let t = if seg > 0.0 { jsmath::min(1.0, jsmath::max(0.0, (s - cum[i]) / seg)) } else { 0.0 };
        let (mut dx, mut dz) = (px[i + 1] - px[i], pz[i + 1] - pz[i]);
        let mut l = jsmath::hypot(dx, dz); if l == 0.0 { l = 1.0; }
        dx /= l; dz /= l;
        let y = if self.has(e, ef::HAS_YS) { let ys = &self.ys[b..]; ys[i] as f64 + (ys[i + 1] as f64 - ys[i] as f64) * t } else { self.y[e] as f64 };
        EdgePoint { x: px[i] + dx * t * seg - dz * offset, z: pz[i] + dz * t * seg + dx * offset, y, hx: dx, hz: dz, seg: i }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EdgePoint { pub x: f64, pub z: f64, pub y: f64, pub hx: f64, pub hz: f64, pub seg: usize }
