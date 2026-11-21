//! World data as the JS sim sees it after loading (src/sim/roadGraph.js SimGraph, src/geom LaneGeometry /
//! LaneGraph / JunctionSet, src/sim/controls.js TrafficControls, stations), plus the read-only geometry
//! queries the step uses (laneW, edgePoint, latPoint, bounds, hopMap / mapLane / mapLat, junction movements,
//! connector points, edge grade). Loaded from a world pack (tools/rust/export.mjs).
//!
//! Every float is held as f64; values that JS keeps in Float32Arrays are widened exactly, so all
//! arithmetic sees the same numbers as V8.
use crate::jm;
use crate::pack::Pack;
use crate::fx::{HashMap, HashSet};

pub const LANE_WIDTH: f64 = 3.7;
const TAPER_M: f64 = 90.0;
pub const MAX_GRADE: f64 = 0.35;
pub const GRADE_HALF: f64 = 6.0;

/// World point + unit heading (out objects of edgePoint / offsetPoint / arcPoint).
#[derive(Clone, Copy, Debug, Default)]
pub struct Pt { pub x: f64, pub y: f64, pub z: f64, pub hx: f64, pub hz: f64, pub seg: i32 }

#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds { pub right: f64, pub left: f64, pub r: f64 }

#[derive(Clone, Copy, Debug)]
pub struct Hop { pub strict: bool, pub shift: i32, pub scale: f64 }

/// Highway class bits (per edge).
pub const HW_SHOULDER: u8 = 1; // motorway, trunk, motorway_link, trunk_link (freeway system)
pub const HW_PULLOVER: u8 = 2; // motorway, trunk
pub const HW_MAINLINE: u8 = 4; // motorway or trunk (density sampling)

pub struct LaneGeom {
    pub v_first: Vec<u32>, pub vs: Vec<f64>,
    /// per vertex [x, y, z, mx, mz] interleaved (offset_point reads vertices i and i + 1: one or two cache lines)
    pub vtx: Vec<[f64; 5]>,
    pub rd: Vec<f64>, pub rend: Vec<f64>, pub tau: Vec<f64>, pub x_r: Vec<f64>, pub x_l: Vec<f64>, pub lane_w: Vec<f64>, pub lane_w_end: Vec<f64>,
    pub tap_r: Vec<u8>, pub tap_l: Vec<u8>,
    /// LaneGraph (src/geom/graph.js), same numbering as the SimGraph
    pub g_length: Vec<f64>, pub g_lanes: Vec<i32>, pub g_opp: Vec<i32>, pub g_twin: Vec<i32>, pub g_cls: Vec<u8>,
    pub g_first: Vec<u32>, pub g_count: Vec<u32>, pub g_poly: Vec<u32>, pub g_xyz: Vec<f64>,
    pub g_out_start: Vec<u32>, pub g_out_list: Vec<u32>, pub g_in_start: Vec<u32>, pub g_in_list: Vec<u32>,
}

fn interleave_vtx(x: &[f64], y: &[f64], z: &[f64], mx: &[f64], mz: &[f64]) -> Vec<[f64; 5]> {
    (0..x.len()).map(|k| [x[k], y[k], z[k], mx[k], mz[k]]).collect()
}

#[inline(always)]
fn smooth(u: f64) -> f64 { let u = if u < 0.0 { 0.0 } else if u > 1.0 { 1.0 } else { u }; u * u * (3.0 - 2.0 * u) }

impl LaneGeom {
    #[inline]
    pub fn right_edge(&self, e: usize, s: f64) -> f64 {
        let l = self.g_length[e];
        let rd = self.rd[e];
        let mut r = rd;
        let ts = self.tau[2 * e];
        let te = self.tau[2 * e + 1];
        if ts > 0.0 { r += (self.rend[2 * e] - rd) * smooth(1.0 - s / ts); }
        if te > 0.0 { r += (self.rend[2 * e + 1] - rd) * smooth(1.0 - (l - s) / te); }
        r
    }
    #[inline]
    fn right_edge_slope(&self, e: usize, s: f64) -> f64 {
        let d = 0.5;
        (self.right_edge(e, jm::min(self.g_length[e], s + d)) - self.right_edge(e, jm::max(0.0, s - d))) / (2.0 * d)
    }
    #[inline]
    pub fn lane_width(&self, e: usize, s: f64) -> f64 {
        let l = self.g_length[e];
        let w = self.lane_w[e];
        let ts = self.tau[2 * e];
        let te = self.tau[2 * e + 1];
        let mut r = w;
        if ts > 0.0 { r += (self.lane_w_end[2 * e] - w) * smooth(1.0 - s / ts); }
        if te > 0.0 { r += (self.lane_w_end[2 * e + 1] - w) * smooth(1.0 - (l - s) / te); }
        r
    }
    pub fn bounds(&self, e: usize, s: f64) -> Bounds {
        let n = self.g_lanes[e] as f64;
        let w = self.lane_width(e, s);
        let l = self.g_length[e];
        let r = self.right_edge(e, s);
        let t = jm::min(TAPER_M, l);
        let fs = if t > 0.0 { jm::max(0.0, 1.0 - s / t) } else { 0.0 };
        let fe = if t > 0.0 { jm::max(0.0, 1.0 - (l - s) / t) } else { 0.0 };
        let ex_r = (self.x_r[2 * e] * fs + self.x_r[2 * e + 1] * fe) * w;
        let ex_l = (self.x_l[2 * e] * fs + self.x_l[2 * e + 1] * fe) * w;
        Bounds { right: r + ex_r, left: r - n * w - ex_l, r }
    }
    #[inline]
    fn locate(&self, e: usize, s: f64) -> (usize, f64) {
        let v0 = self.v_first[e] as usize;
        let v1 = self.v_first[e + 1] as usize - 1;
        let vs = &self.vs;
        if s <= 0.0 { return (v0, 0.0); }
        if s >= vs[v1] { return (v1 - 1, 1.0); }
        let (mut lo, mut hi) = (v0, v1);
        while hi - lo > 1 { let m = (lo + hi) >> 1; if vs[m] <= s { lo = m; } else { hi = m; } }
        let d = vs[lo + 1] - vs[lo];
        (lo, if d > 0.0 { (s - vs[lo]) / d } else { 0.0 })
    }
    pub fn offset_point(&self, e: usize, s: f64, off: f64, off_slope: f64) -> Pt {
        let (i, t) = self.locate(e, s);
        let j = i + 1;
        let (a, b) = (&self.vtx[i], &self.vtx[j]);
        let ([vxi, vyi, vzi, mxi, mzi], [vxj, vyj, vzj, mxj, mzj]) = (*a, *b);
        let nx = mxi + (mxj - mxi) * t;
        let nz = mzi + (mzj - mzi) * t;
        let x = vxi + (vxj - vxi) * t + off * nx;
        let y = vyi + (vyj - vyi) * t;
        let z = vzi + (vzj - vzi) * t + off * nz;
        let ds = jm::or(self.vs[j] - self.vs[i], 1.0);
        let hx = (vxj - vxi + off * (mxj - mxi)) / ds + off_slope * nx;
        let hz = (vzj - vzi + off * (mzj - mzi)) / ds + off_slope * nz;
        let l = jm::or(jm::hypot(hx, hz), 1.0);
        Pt { x, y, z, hx: hx / l, hz: hz / l, seg: 0 }
    }
    #[inline]
    pub fn lat_point(&self, e: usize, s: f64, lat: f64) -> Pt {
        self.offset_point(e, s, self.right_edge(e, s) - lat, self.right_edge_slope(e, s))
    }
    // ---- LaneGraph helpers (src/geom/graph.js, hops.js)
    #[inline] fn gx(&self, n: usize) -> f64 { self.g_xyz[3 * n] }
    #[inline] fn gz(&self, n: usize) -> f64 { self.g_xyz[3 * n + 2] }
    #[inline] fn gfrom(&self, e: usize) -> usize { self.g_poly[self.g_first[e] as usize] as usize }
    #[inline] fn gto(&self, e: usize) -> usize { self.g_poly[(self.g_first[e] + self.g_count[e] - 1) as usize] as usize }
    fn heading(&self, e: usize, end: bool) -> f64 {
        let f = self.g_first[e] as usize;
        let c = self.g_count[e] as usize;
        let (a, b) = if end { (self.g_poly[f + c - 2], self.g_poly[f + c - 1]) } else { (self.g_poly[f], self.g_poly[f + 1]) };
        let (a, b) = (a as usize, b as usize);
        jm::atan2(self.gz(b) - self.gz(a), self.gx(b) - self.gx(a))
    }
    fn turn_angle(&self, e: usize, en: usize) -> f64 {
        let mut d = self.heading(en, false) - self.heading(e, true);
        while d > jm::PI { d -= 2.0 * jm::PI; }
        while d < -jm::PI { d += 2.0 * jm::PI; }
        d
    }
    fn point_into(&self, e: usize, dist: f64, from_end: bool) -> (f64, f64) {
        let f = self.g_first[e] as usize;
        let c = self.g_count[e] as usize;
        let mut acc = 0.0;
        for k in 0..c - 1 {
            let a = self.g_poly[if from_end { f + c - 1 - k } else { f + k }] as usize;
            let b = self.g_poly[if from_end { f + c - 2 - k } else { f + k + 1 }] as usize;
            let l = jm::hypot(self.gx(b) - self.gx(a), self.gz(b) - self.gz(a));
            if acc + l >= dist {
                let t = if l > 0.0 { (dist - acc) / l } else { 0.0 };
                return (self.gx(a) + (self.gx(b) - self.gx(a)) * t, self.gz(a) + (self.gz(b) - self.gz(a)) * t);
            }
            acc += l;
        }
        let n = self.g_poly[if from_end { f } else { f + c - 1 }] as usize;
        (self.gx(n), self.gz(n))
    }
    fn side(&self, h: f64, node: usize, p: (f64, f64)) -> f64 {
        let dx = jm::cos(h);
        let dz = jm::sin(h);
        (p.0 - self.gx(node)) * -dz + (p.1 - self.gz(node)) * dx
    }
    fn is_freeway_cls(&self, e: usize) -> bool { self.g_cls[e] <= 3 }
    /// src/geom/hops.js hopMap
    pub fn hop_map(&self, e: usize, en: usize) -> Hop {
        const TURN_STRICT: f64 = 1.2;
        let node = self.gto(e);
        let mut outs: Vec<usize> = Vec::new();
        let mut ins: Vec<usize> = Vec::new();
        for k in self.g_out_start[node]..self.g_out_start[node + 1] { let o = self.g_out_list[k as usize] as usize; if o as i32 != self.g_twin[e] { outs.push(o); } }
        let fen = self.gfrom(en);
        for k in self.g_in_start[fen]..self.g_in_start[fen + 1] { let i = self.g_in_list[k as usize] as usize; if i as i32 != self.g_twin[en] { ins.push(i); } }
        let simple = outs.len() <= 1 && ins.len() <= 1;
        let strict = simple || (self.is_freeway_cls(e) && self.is_freeway_cls(en)
            && (outs.len() <= 1 || ins.len() <= 1 || jm::abs(self.turn_angle(e, en)) < TURN_STRICT));
        let mut shift = 0i32;
        if strict && simple { shift = self.g_lanes[e] - self.g_lanes[en]; }
        else if strict && self.g_opp[e] > 0 && self.g_opp[en] > 0 { shift = self.g_lanes[e] - self.g_lanes[en]; }
        else if strict {
            let (mut div, mut mer) = (0i32, 0i32);
            if outs.len() > 1 {
                let he = self.heading(e, true);
                let sd: HashMap<usize, f64> = outs.iter().map(|&o| (o, self.side(he, node, self.point_into(o, 50.0, false)))).collect();
                jm::v8_sort_small(&mut outs, |a, b| if jm::abs(sd[&a] - sd[&b]) >= 1.0 { sd[&b] - sd[&a] } else { self.turn_angle(e, b) - self.turn_angle(e, a) });
                for &o in &outs { if o == en { break; } div += self.g_lanes[o]; }
                div = 0.max(div.min(self.g_lanes[e] - self.g_lanes[en]));
            }
            if ins.len() > 1 {
                let hn = self.heading(en, false);
                let sd: HashMap<usize, f64> = ins.iter().map(|&i| (i, self.side(hn, node, self.point_into(i, 50.0, true)))).collect();
                jm::v8_sort_small(&mut ins, |a, b| if jm::abs(sd[&a] - sd[&b]) >= 1.0 { sd[&b] - sd[&a] } else { self.turn_angle(b, en) - self.turn_angle(a, en) });
                for &i in &ins { if i == e { break; } mer += self.g_lanes[i]; }
                mer = 0.max(mer.min(self.g_lanes[en] - self.g_lanes[e]));
            }
            shift = div - mer;
        }
        Hop { strict, shift, scale: self.g_lanes[en] as f64 / self.g_lanes[e] as f64 }
    }
}

pub struct Junctions {
    pub mouth_s: Vec<f64>, pub gore_s: Vec<f64>,
    pub j_node: Vec<u32>, pub j_first_m: Vec<u32>, pub j_count_m: Vec<u32>,
    /// per movement: [inEdge, inLane, outEdge, outLane, firstPt, nPts, preferred]
    pub m_info: Vec<u32>,
    /// per movement: [sIn, sOut, length, turn]
    pub m_f: Vec<f64>,
    pub m_len: Vec<f64>,
    /// per point: [x, y, z, s]
    pub pts: Vec<f64>,
    pub move_count: usize,
    /// node -> junction (-1 none)
    pub j_of_node: Vec<i32>,
    /// movements by in-edge (CSR, ascending m)
    by_in_start: Vec<u32>, by_in: Vec<u32>,
    /// conflicts per movement (CSR in JunctionSet.conflictsOf order)
    pub conf_start: Vec<u32>, pub conf_list: Vec<Conflict>,
}

#[derive(Clone, Copy, Debug)]
pub struct Conflict { pub other: u32, pub ty: u32, pub s: f64, pub s_other: f64 }

impl Junctions {
    #[allow(clippy::too_many_arguments)]
    pub fn from_arrays(n: usize, ne: usize, m_info: Vec<u32>, m_f: Vec<f64>, pts: Vec<f64>, conf: Vec<u32>, conf_s: Vec<f64>, mouth_s: Vec<f64>, gore_s: Vec<f64>,
        j_node: Vec<u32>, j_first_m: Vec<u32>, j_count_m: Vec<u32>) -> Junctions {
        let mc = m_info.len() / 7;
        let mut by_in_start = vec![0u32; ne + 1];
        for m in 0..mc { by_in_start[m_info[7 * m] as usize + 1] += 1; }
        for e in 0..ne { by_in_start[e + 1] += by_in_start[e]; }
        let mut fill = by_in_start.clone();
        let mut by_in = vec![0u32; mc];
        for m in 0..mc { let a = m_info[7 * m] as usize; by_in[fill[a] as usize] = m as u32; fill[a] += 1; }
        let nc = conf.len() / 3;
        let mut cnt = vec![0u32; mc + 1];
        for c in 0..nc { cnt[conf[3 * c] as usize + 1] += 1; cnt[conf[3 * c + 1] as usize + 1] += 1; }
        for m in 0..mc { cnt[m + 1] += cnt[m]; }
        let mut fill = cnt.clone();
        let mut list = vec![Conflict { other: 0, ty: 0, s: 0.0, s_other: 0.0 }; 2 * nc];
        for c in 0..nc {
            let (a, b, t) = (conf[3 * c] as usize, conf[3 * c + 1] as usize, conf[3 * c + 2]);
            let (sa, sb) = (conf_s[2 * c], conf_s[2 * c + 1]);
            list[fill[a] as usize] = Conflict { other: b as u32, ty: t, s: sa, s_other: sb }; fill[a] += 1;
            list[fill[b] as usize] = Conflict { other: a as u32, ty: t, s: sb, s_other: sa }; fill[b] += 1;
        }
        let mut j_of_node = vec![-1i32; n];
        for (jj, &nd) in j_node.iter().enumerate() { if (nd as usize) < n { j_of_node[nd as usize] = jj as i32; } }
        let m_len = (0..mc).map(|m| (m_f[4 * m + 2] as f32) as f64).collect();
        Junctions { mouth_s, gore_s, j_node, j_first_m, j_count_m, m_f, m_len, pts, move_count: mc, j_of_node, by_in_start, by_in, conf_start: cnt, conf_list: list, m_info }
    }
    #[inline] pub fn info(&self, m: usize, k: usize) -> u32 { self.m_info[7 * m + k] }
    #[inline] pub fn s_in(&self, m: usize) -> f64 { self.m_f[4 * m] }
    #[inline] pub fn s_out(&self, m: usize) -> f64 { self.m_f[4 * m + 1] }
    #[inline] pub fn turn(&self, m: usize) -> f64 { self.m_f[4 * m + 3] }
    /// JunctionSet.movement(a, la, b): movement from (in-edge a, lane la) onto out-edge b, -1 none.
    #[inline]
    pub fn movement(&self, a: usize, la: i32, b: i32) -> i32 {
        if la < 0 || a + 1 >= self.by_in_start.len() { return -1; }
        for k in self.by_in_start[a]..self.by_in_start[a + 1] {
            let m = self.by_in[k as usize] as usize;
            if self.m_info[7 * m + 1] as i32 == la && self.m_info[7 * m + 2] as i32 == b { return m as i32; }
        }
        -1
    }
    pub fn movements_at(&self, node: usize) -> std::ops::Range<usize> {
        let j = if node < self.j_of_node.len() { self.j_of_node[node] } else { -1 };
        if j < 0 { return 0..0; }
        let j = j as usize;
        let f = self.j_first_m[j] as usize;
        f..f + self.j_count_m[j] as usize
    }
    #[inline]
    pub fn conflicts_of(&self, m: usize) -> &[Conflict] { &self.conf_list[self.conf_start[m] as usize..self.conf_start[m + 1] as usize] }
    /// JunctionSet.arcPoint
    pub fn arc_point(&self, m: usize, t: f64) -> Pt {
        let first = self.m_info[7 * m + 4] as usize;
        let n = self.m_info[7 * m + 5] as usize;
        let p = &self.pts;
        let mut k = 0usize;
        while k + 2 < n && p[4 * (first + k + 1) + 3] < t { k += 1; }
        let a = 4 * (first + k);
        let b = 4 * (first + k + 1);
        let d = jm::or(p[b + 3] - p[a + 3], 1.0);
        let u = jm::max(0.0, jm::min(1.0, (t - p[a + 3]) / d));
        let hx = p[b] - p[a];
        let hz = p[b + 2] - p[a + 2];
        let l = jm::or(jm::hypot(hx, hz), 1.0);
        Pt { x: p[a] + (p[b] - p[a]) * u, y: p[a + 1] + (p[b + 1] - p[a + 1]) * u, z: p[a + 2] + (p[b + 2] - p[a + 2]) * u, hx: hx / l, hz: hz / l, seg: 0 }
    }
    /// vehicle.js connectorPoint: arcPoint position with a smooth heading.
    pub fn connector_point(&self, m: usize, t: f64) -> Pt {
        let mut out = self.arc_point(m, t);
        let first = self.m_info[7 * m + 4] as usize;
        let n = self.m_info[7 * m + 5] as usize;
        let p = &self.pts;
        if n < 3 { return out; }
        let mut k = 0usize;
        while k + 2 < n && p[4 * (first + k + 1) + 3] < t { k += 1; }
        let a = 4 * (first + k);
        let b = a + 4;
        let d = jm::or(p[b + 3] - p[a + 3], 1.0);
        let u = jm::max(0.0, jm::min(1.0, (t - p[a + 3]) / d));
        let cx = p[b] - p[a];
        let cz = p[b + 2] - p[a + 2];
        let cl = jm::or(jm::hypot(cx, cz), 1.0);
        let (mut ax, mut az) = (cx / cl, cz / cl);
        let (mut bx, mut bz) = (ax, az);
        if k > 0 { let px = p[a] - p[a - 4]; let pz = p[a + 2] - p[a - 2]; let pl = jm::or(jm::hypot(px, pz), 1.0); ax += px / pl; az += pz / pl; }
        if k + 2 < n { let nx = p[b + 4] - p[b]; let nz = p[b + 6] - p[b + 2]; let nl = jm::or(jm::hypot(nx, nz), 1.0); bx += nx / nl; bz += nz / nl; }
        let al = jm::or(jm::hypot(ax, az), 1.0);
        let bl = jm::or(jm::hypot(bx, bz), 1.0);
        let hx = (1.0 - u) * ax / al + u * bx / bl;
        let hz = (1.0 - u) * az / al + u * bz / bl;
        let hl = jm::or(jm::hypot(hx, hz), 1.0);
        out.hx = hx / hl; out.hz = hz / hl;
        out
    }
}

/// TrafficControls (src/sim/controls.js), derived arrays.
pub struct Controls {
    pub count: usize,
    pub ty: Vec<i8>, pub flags: Vec<u8>, pub cycle: Vec<f64>, pub offset: Vec<f64>,
    pub ap_control: Vec<i32>, pub ap_stop_at: Vec<f64>, pub ap_edge: Vec<i32>, pub ap_group: Vec<i8>, pub ap_prot_left: Vec<u8>,
    pub cluster_of: Vec<i32>, pub edge_ap_start: Vec<i32>, pub edge_ap_list: Vec<i32>,
    /// phases per control (CSR): [group, left, g, y, r]
    pub phase_start: Vec<u32>, pub phases: Vec<[f64; 5]>,
    pub ramp_start: f64, pub ramp_end: f64,
}

pub struct Station { pub node: i32, pub pumps: i32, pub x: f64, pub z: f64 }

/// The SimGraph and everything attached to it.
pub struct World {
    pub node_count: usize,
    pub nx: Vec<f64>, pub nz: Vec<f64>, pub node_y: Vec<f64>,
    pub edge_count: usize,
    pub e_from: Vec<i32>, pub e_to: Vec<i32>, pub e_rev: Vec<i32>, pub e_lanes: Vec<i32>, pub e_opp: Vec<i32>,
    pub e_speed: Vec<f64>, pub e_length: Vec<f64>, pub e_rank: Vec<i32>, pub e_class: Vec<u8>, pub e_hw: Vec<u8>,
    pub e_is_link: Vec<bool>, pub e_has_ys: Vec<bool>,
    pub e_head_start: Vec<f64>, pub e_head_end: Vec<f64>, pub e_y: Vec<f64>, pub e_right: Vec<f64>,
    pub e_s_exit: Vec<f64>, pub e_s_stop: Vec<f64>, pub e_gore: Vec<f64>,
    pub e_surf: Vec<u8>, pub e_gnd_r: Vec<u8>, pub e_gnd_l: Vec<u8>,
    pub v_start: Vec<u32>, pub px: Vec<f64>, pub pz: Vec<f64>, pub cum: Vec<f64>, pub ys: Vec<f64>,
    pub l_start: Vec<u32>, pub lane_off: Vec<f64>,
    pub out_start: Vec<i32>, pub out_edges: Vec<i32>, pub in_start: Vec<i32>, pub in_edges: Vec<i32>,
    pub highway_names: Vec<String>,
    pub max_speed: f64,
    pub banned_hops: HashMap<i64, Vec<i32>>,
    pub contact_pairs: Option<HashSet<i64>>,
    pub lg: Option<LaneGeom>,
    pub j: Option<Junctions>,
    pub controls: Option<Controls>,
    pub stations: Vec<Station>,
    pub surfaces: bool,
    /// per junction movement: sim._conflicts(m) (world conflicts + paths within 4.2 m), filled on first use
    pub conf_ext: Vec<std::sync::OnceLock<Box<[Conflict]>>>,
    /// the sim's router (weight 1.25) built once per world; envs fork it (shared search graph, own search state)
    pub nav_base: std::sync::OnceLock<crate::route_stub::NavRouter>,
    /// junction movements by out-edge (junction.rs moves_into), built once per world
    pub mv_into: std::sync::OnceLock<Option<crate::fx::HashMap<u32, Vec<usize>>>>,
    /// hop table: for edge e, entry k = hop to out_edges[out_start[e_to[e]] + k]
    hop_base: Vec<u32>,
    /// (out edge, hop) per (edge, out edge of its head node), CSR order; edge e at [hop_base[e], hop_base[e + 1])
    hop_tab: Vec<(u32, Hop)>,
    pub bounds: (f64, f64, f64, f64),
    /// the lane geometry and connectors are the smoothed frame geometry (bayenv_world::smooth, DECISIONS.md 33);
    /// a Sim on this world needs SimOpts::smooth_geometry (and curvature_dynamics)
    pub geometry_smoothed: bool,
    /// the junction connectors are re-fitted to vehicle-feasible radii (bayenv_world::junction_radii, DECISIONS.md
    /// 34); a Sim on this world needs SimOpts::junction_radii
    pub junction_radii: bool,
}

impl World {
    pub fn load(path: &str) -> World { World::from_pack(&Pack::read(path).expect("world pack")) }

    pub fn from_pack(p: &Pack) -> World {
        let meta = &p.meta;
        let n = meta["nodeCount"].as_u64().unwrap() as usize;
        let ne = meta["edgeCount"].as_u64().unwrap() as usize;
        let names: Vec<String> = meta["highway"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        let e_class = p.u8s("eClass");
        let hwbits: Vec<u8> = names.iter().map(|s| {
            let mut b = 0u8;
            if matches!(s.as_str(), "motorway" | "trunk" | "motorway_link" | "trunk_link") { b |= HW_SHOULDER; }
            if matches!(s.as_str(), "motorway" | "trunk") { b |= HW_PULLOVER | HW_MAINLINE; }
            b
        }).collect();
        let flags = p.u8s("eFlags");
        let lg = if meta["laneGeom"].as_bool().unwrap_or(false) {
            Some(LaneGeom {
                v_first: p.u32s("LG.vFirst"), vs: p.f64s("LG.vs"),
                vtx: interleave_vtx(&p.f64s("LG.vx"), &p.f64s("LG.vy"), &p.f64s("LG.vz"), &p.f64s("LG.mx"), &p.f64s("LG.mz")), rd: p.f64s("LG.Rd"), rend: p.f64s("LG.Rend"), tau: p.f64s("LG.tau"),
                x_r: p.f64s("LG.xR"), x_l: p.f64s("LG.xL"), lane_w: p.f64s("LG.laneW"), lane_w_end: p.f64s("LG.laneWEnd"),
                tap_r: p.u8s("LG.tapR"), tap_l: p.u8s("LG.tapL"),
                g_length: p.f64s("G.length"), g_lanes: p.i32s("G.lanes"), g_opp: p.i32s("G.opp"), g_twin: p.i32s("G.twin"), g_cls: p.u8s("G.cls"),
                g_first: p.u32s("G.first"), g_count: p.u32s("G.count"), g_poly: p.u32s("G.poly"), g_xyz: p.f64s("G.xyz"),
                g_out_start: p.u32s("G.outStart"), g_out_list: p.u32s("G.outList"), g_in_start: p.u32s("G.inStart"), g_in_list: p.u32s("G.inList"),
            })
        } else { None };
        let j = if meta["junctions"].as_bool().unwrap_or(false) {
            Some(Junctions::from_arrays(n, ne, p.u32s("J.mInfo"), p.f64s("J.mF"), p.f64s("J.pts"), p.u32s("J.conf"), p.f64s("J.confS"),
                p.f64s("J.mouthS"), p.f64s("J.goreS"), p.u32s("J.jNode"), p.u32s("J.jFirstM"), p.u32s("J.jCountM")))
        } else { None };
        let controls = if p.has("C.type") {
            let ps = p.u32s("C.phaseStart");
            let pf = p.f64s("C.phases");
            let cm = &meta["controls"];
            Some(Controls {
                count: cm["count"].as_u64().unwrap() as usize,
                ty: p.i8s("C.type"), flags: p.u8s("C.flags"), cycle: p.f64s("C.cycle"), offset: p.f64s("C.offset"),
                ap_control: p.i32s("C.apControl"), ap_stop_at: p.f64s("C.apStopAt"), ap_edge: p.i32s("C.apEdge"), ap_group: p.i8s("C.apGroup"),
                ap_prot_left: p.u8s("C.apProtLeft"), cluster_of: p.i32s("C.clusterOf"), edge_ap_start: p.i32s("C.edgeApStart"), edge_ap_list: p.i32s("C.edgeApList"),
                phase_start: ps, phases: pf.chunks_exact(5).map(|c| [c[0], c[1], c[2], c[3], c[4]]).collect(),
                ramp_start: cm["rampStart"].as_f64().unwrap(), ramp_end: cm["rampEnd"].as_f64().unwrap(),
            })
        } else { None };
        let st_node = p.i32s("stNode");
        let st_pumps = p.i32s("stPumps");
        let st_x = p.f64s("stX");
        let st_z = p.f64s("stZ");
        let stations = (0..st_node.len()).map(|k| Station { node: st_node[k], pumps: st_pumps[k], x: st_x[k], z: st_z[k] }).collect();
        let mut banned_hops: HashMap<i64, Vec<i32>> = HashMap::default();
        if p.has("bannedHops") {
            let t = p.i32s("bannedHops");
            for c in t.chunks_exact(3) { banned_hops.entry(c[0] as i64 * 4194304 + c[1] as i64).or_default().push(c[2]); }
        }
        let contact_pairs = if p.has("contactPairs") { Some(p.f64s("contactPairs").into_iter().map(|k| k as i64).collect()) } else { None };
        let b = &meta["bounds"];
        let mut w = World {
            node_count: n, nx: p.f64s("nx"), nz: p.f64s("nz"), node_y: p.f64s("nodeY"), edge_count: ne,
            e_from: p.i32s("eFrom"), e_to: p.i32s("eTo"), e_rev: p.i32s("eRev"), e_lanes: p.i32s("eLanes"), e_opp: p.i32s("eOpp"),
            e_speed: p.f64s("eSpeed"), e_length: p.f64s("eLength"), e_rank: p.i32s("eRank"),
            e_hw: e_class.iter().map(|&c| hwbits[c as usize]).collect(), e_class,
            e_is_link: flags.iter().map(|f| f & 1 != 0).collect(), e_has_ys: flags.iter().map(|f| f & 16 != 0).collect(),
            e_head_start: p.f64s("eHeadStart"), e_head_end: p.f64s("eHeadEnd"), e_y: p.f64s("eY"), e_right: p.f64s("eRight"),
            e_s_exit: p.f64s("eSExit"), e_s_stop: p.f64s("eSStop"), e_gore: p.f64s("eGore"),
            e_surf: p.u8s("eSurf"), e_gnd_r: p.u8s("eGndR"), e_gnd_l: p.u8s("eGndL"),
            v_start: p.u32s("vStart"), px: p.f64s("px"), pz: p.f64s("pz"), cum: p.f64s("cum"), ys: p.f64s("ys"),
            l_start: p.u32s("lStart"), lane_off: p.f64s("laneOff"),
            out_start: p.i32s("outStart"), out_edges: p.i32s("outEdges"), in_start: p.i32s("inStart"), in_edges: p.i32s("inEdges"),
            highway_names: names, max_speed: meta["maxSpeed"].as_f64().unwrap_or(0.0), banned_hops, contact_pairs, lg, j, controls, stations,
            surfaces: meta["surfaces"].as_bool().unwrap_or(false), hop_base: Vec::new(), hop_tab: Vec::new(), conf_ext: Vec::new(), nav_base: std::sync::OnceLock::new(), mv_into: std::sync::OnceLock::new(),
            bounds: (b["minX"].as_f64().unwrap_or(0.0), b["maxX"].as_f64().unwrap_or(0.0), b["minZ"].as_f64().unwrap_or(0.0), b["maxZ"].as_f64().unwrap_or(0.0)), geometry_smoothed: false, junction_radii: false,
        };
        w.build_hops();
        w
    }

    /// From the shared world crate (bayenv_world::World::load): same numbering and values as the JS-loaded world.
    pub fn from_bayenv(w: &bayenv_world::World) -> World {
        use bayenv_world::graph::ef;
        let g = &w.graph;
        let (n, ne) = (g.node_count, g.edge_count);
        let names: Vec<String> = bayenv_world::roadpack::CLASSES_ALL.iter().map(|s| s.to_string()).collect();
        let hwbits: Vec<u8> = names.iter().map(|s| {
            let mut b = 0u8;
            if matches!(s.as_str(), "motorway" | "trunk" | "motorway_link" | "trunk_link") { b |= HW_SHOULDER; }
            if matches!(s.as_str(), "motorway" | "trunk") { b |= HW_PULLOVER | HW_MAINLINE; }
            b
        }).collect();
        let f64v = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let u8f = |v: &[u8]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let lgs = &w.lanes;
        let lgr = &w.lane_graph;
        let lg = LaneGeom {
            v_first: lgs.v_first.to_vec(), vs: lgs.vs.to_vec(), vtx: interleave_vtx(&lgs.vx, &f64v(&lgs.vy), &lgs.vz, &lgs.mx, &lgs.mz),
            rd: f64v(&lgs.rd), rend: f64v(&lgs.rend), tau: f64v(&lgs.tau), x_r: u8f(&lgs.x_r), x_l: u8f(&lgs.x_l), lane_w: f64v(&lgs.lane_w), lane_w_end: f64v(&lgs.lane_w_end),
            tap_r: lgs.tap_r.to_vec(), tap_l: lgs.tap_l.to_vec(),
            g_length: f64v(&lgr.length), g_lanes: lgr.lanes.iter().map(|&x| x as i32).collect(), g_opp: lgr.opp.iter().map(|&x| x as i32).collect(), g_twin: lgr.twin.clone(), g_cls: lgr.cls.clone(),
            g_first: lgr.first.clone(), g_count: lgr.count.clone(), g_poly: lgr.poly.clone(), g_xyz: lgr.xyz.clone(),
            g_out_start: lgr.out_start.clone(), g_out_list: lgr.out_list.clone(), g_in_start: lgr.in_start.clone(), g_in_list: lgr.in_list.clone(),
        };
        let j = w.junctions.as_ref().map(|js| Junctions::from_arrays(n, ne, js.m_info.to_vec(), f64v(&js.m_f), f64v(&js.pts), js.conf.to_vec(), f64v(&js.conf_s),
            f64v(&js.mouth_s), f64v(&js.gore_s), js.j_node.to_vec(), js.j_first_m.to_vec(), js.j_count_m.to_vec()));
        let has_j = j.is_some();
        let nan = |v: &[f32]| if has_j { f64v(v) } else { vec![f64::NAN; ne] };
        let controls = w.controls.as_ref().map(|c| {
            let mut phase_start = vec![0u32; c.count + 1];
            let mut phases = Vec::new();
            for k in 0..c.count { phase_start[k] = phases.len() as u32; for p in &c.phases[k] { phases.push([p.group as f64, if p.left { 1.0 } else { 0.0 }, p.g, p.y, p.r]); } }
            phase_start[c.count] = phases.len() as u32;
            Controls { count: c.count, ty: c.ty.clone(), flags: c.flags.clone(), cycle: f64v(&c.cycle), offset: f64v(&c.offset), ap_control: c.ap_control.clone(), ap_stop_at: f64v(&c.ap_stop_at),
                ap_edge: c.ap_edge.clone(), ap_group: c.ap_group.clone(), ap_prot_left: c.ap_prot_left.clone(), cluster_of: c.cluster_of.clone(), edge_ap_start: c.edge_ap_start.clone(),
                edge_ap_list: c.edge_ap_list.clone(), phase_start, phases, ramp_start: c.ramp_start, ramp_end: c.ramp_end }
        });
        let i32v = |v: &[u32]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
        let mut world = World {
            node_count: n, nx: g.nx.clone(), nz: g.nz.clone(), node_y: f64v(&g.node_y), edge_count: ne,
            e_from: i32v(&g.from), e_to: i32v(&g.to), e_rev: g.reverse.clone(), e_lanes: g.lanes.iter().map(|&x| x as i32).collect(), e_opp: g.opp_lanes.iter().map(|&x| x as i32).collect(),
            e_speed: g.speed_limit.clone(), e_length: g.length.clone(), e_rank: g.rank.iter().map(|&x| x as i32).collect(),
            e_hw: g.class.iter().map(|&c| hwbits[c as usize]).collect(), e_class: g.class.clone(),
            e_is_link: g.eflags.iter().map(|&f| f & ef::LINK != 0).collect(), e_has_ys: g.eflags.iter().map(|&f| f & ef::HAS_YS != 0).collect(),
            e_head_start: g.heading_start.clone(), e_head_end: g.heading_end.clone(), e_y: f64v(&g.y), e_right: g.right_edge.clone(),
            e_s_exit: nan(&g.s_exit), e_s_stop: nan(&g.s_stop), e_gore: nan(&g.gore),
            e_surf: g.surf.clone(), e_gnd_r: g.gnd_r.clone(), e_gnd_l: g.gnd_l.clone(),
            v_start: g.pt_start.clone(), px: g.px.clone(), pz: g.pz.clone(), cum: g.cum.clone(), ys: f64v(&g.ys),
            l_start: g.lane_start.clone(), lane_off: f64v(&g.lane_off),
            out_start: i32v(&g.out_start), out_edges: i32v(&g.out_edges), in_start: i32v(&g.in_start), in_edges: i32v(&g.in_edges),
            highway_names: names, max_speed: g.max_speed,
            banned_hops: g.banned_hops.iter().map(|(&k, v)| (k as i64, v.iter().map(|&x| x as i32).collect())).collect(),
            contact_pairs: w.contact_pairs.as_ref().map(|c| c.set.iter().map(|&k| k as i64).collect()),
            lg: Some(lg), j, controls,
            stations: w.stations.iter().map(|s| Station { node: s.node as i32, pumps: s.pumps as i32, x: s.x, z: s.z }).collect(),
            surfaces: w.surfaces, hop_base: Vec::new(), hop_tab: Vec::new(), conf_ext: Vec::new(), nav_base: std::sync::OnceLock::new(), mv_into: std::sync::OnceLock::new(),
            bounds: (g.bounds.min_x, g.bounds.max_x, g.bounds.min_z, g.bounds.max_z), geometry_smoothed: w.smoothing.is_some(), junction_radii: w.junction_radii.is_some(),
        };
        world.build_hops();
        world
    }

    /// Graph-only copy of the shared world for routing / population generation / refuel planning (bayenv-gpu):
    /// nodes, edge from / to / reverse / lanes / speed / length / rank / headings, CSR adjacency, banned hops,
    /// stations, bounds. Polylines, lane geometry, junctions, controls and the hop table are left empty (~40 MB
    /// instead of the full copy of `from_bayenv`).
    pub fn routing_from_bayenv(w: &bayenv_world::World) -> World {
        let g = &w.graph;
        let i32v = |v: &[u32]| v.iter().map(|&x| x as i32).collect::<Vec<i32>>();
        World {
            node_count: g.node_count, nx: g.nx.clone(), nz: g.nz.clone(), node_y: Vec::new(), edge_count: g.edge_count,
            e_from: i32v(&g.from), e_to: i32v(&g.to), e_rev: g.reverse.clone(), e_lanes: g.lanes.iter().map(|&x| x as i32).collect(), e_opp: Vec::new(),
            e_speed: g.speed_limit.clone(), e_length: g.length.clone(), e_rank: g.rank.iter().map(|&x| x as i32).collect(), e_class: g.class.clone(), e_hw: Vec::new(),
            e_is_link: Vec::new(), e_has_ys: Vec::new(), e_head_start: g.heading_start.clone(), e_head_end: g.heading_end.clone(), e_y: Vec::new(), e_right: Vec::new(),
            e_s_exit: Vec::new(), e_s_stop: Vec::new(), e_gore: Vec::new(), e_surf: Vec::new(), e_gnd_r: Vec::new(), e_gnd_l: Vec::new(),
            v_start: Vec::new(), px: Vec::new(), pz: Vec::new(), cum: Vec::new(), ys: Vec::new(), l_start: Vec::new(), lane_off: Vec::new(),
            out_start: i32v(&g.out_start), out_edges: i32v(&g.out_edges), in_start: i32v(&g.in_start), in_edges: i32v(&g.in_edges),
            highway_names: Vec::new(), max_speed: g.max_speed,
            banned_hops: g.banned_hops.iter().map(|(&k, v)| (k as i64, v.iter().map(|&x| x as i32).collect())).collect(),
            contact_pairs: None, lg: None, j: None, controls: None,
            stations: w.stations.iter().map(|s| Station { node: s.node as i32, pumps: s.pumps as i32, x: s.x, z: s.z }).collect(),
            surfaces: false, conf_ext: Vec::new(), hop_base: Vec::new(), hop_tab: Vec::new(), nav_base: std::sync::OnceLock::new(), mv_into: std::sync::OnceLock::new(),
            bounds: (g.bounds.min_x, g.bounds.max_x, g.bounds.min_z, g.bounds.max_z), geometry_smoothed: w.smoothing.is_some(), junction_radii: w.junction_radii.is_some(),
        }
    }

    /// Smooth this world's lane geometry and connectors in place (bayenv_world::smooth with the default config; the
    /// same arrays as `from_bayenv` of a world loaded with `LoadOptions::smooth_geometry`). For worlds from a pack
    /// (synthetic test worlds); returns the statistics, None when there is no lane geometry or already smoothed.
    pub fn smooth_geometry(&mut self) -> Option<bayenv_world::smooth::SmoothStats> {
        if self.geometry_smoothed { return None; }
        let lg = self.lg.as_ref()?;
        let f32v = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let u8v = |v: &[f64]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
        let ne = self.edge_count;
        let col = |k: usize| lg.vtx.iter().map(|v| v[k]).collect::<Vec<f64>>();
        let blg = bayenv_world::LaneGeometry {
            v_first: lg.v_first.clone().into(), vx: col(0).into(), vy: f32v(&col(1)).into(), vz: col(2).into(), vs: lg.vs.clone().into(), mx: col(3).into(), mz: col(4).into(),
            rd: f32v(&lg.rd).into(), rend: f32v(&lg.rend).into(), tau: f32v(&lg.tau).into(), tap_r: lg.tap_r.clone().into(), tap_l: lg.tap_l.clone().into(),
            x_r: u8v(&lg.x_r).into(), x_l: u8v(&lg.x_l).into(), lw: LANE_WIDTH, lane_w: f32v(&lg.lane_w).into(), lane_w_end: f32v(&lg.lane_w_end).into(),
            length: f32v(&lg.g_length).into(), lanes: lg.g_lanes.iter().map(|&x| x as u8).collect::<Vec<u8>>().into(),
        };
        let mut g = bayenv_world::LaneGraph {
            xyz: lg.g_xyz.clone(), first: lg.g_first.clone(), count: lg.g_count.clone(), poly: lg.g_poly.clone(),
            lanes: lg.g_lanes.iter().map(|&x| x as u8).collect(), opp: lg.g_opp.iter().map(|&x| x as u8).collect(), twin: lg.g_twin.clone(), cls: lg.g_cls.clone(),
            length: f32v(&lg.g_length), out_start: lg.g_out_start.clone(), in_start: lg.g_in_start.clone(), out_list: lg.g_out_list.clone(), in_list: lg.g_in_list.clone(),
            ..Default::default()
        };
        g.node_count = lg.g_xyz.len() / 3; g.edge_count = ne;
        let js = self.j.as_ref().map(|j| bayenv_world::JunctionSet {
            m_info: j.m_info.clone().into(), m_f: f32v(&j.m_f).into(), pts: f32v(&j.pts).into(), j_node: j.j_node.clone().into(), ..Default::default()
        });
        let (slg, sj, st) = bayenv_world::smooth::smooth_geometry(&blg, &g, &self.e_speed, js.as_ref(), &bayenv_world::smooth::SmoothCfg::default());
        let f64v = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
        let lgm = self.lg.as_mut().unwrap();
        lgm.v_first = slg.v_first.to_vec(); lgm.vs = slg.vs.to_vec();
        lgm.vtx = interleave_vtx(&slg.vx, &f64v(&slg.vy), &slg.vz, &slg.mx, &slg.mz);
        if let (Some(j), Some(sj)) = (self.j.as_mut(), sj) { j.pts = f64v(&sj.pts); j.m_info = sj.m_info.to_vec(); }
        self.geometry_smoothed = true;
        self.build_hops();
        Some(st)
    }

    fn build_hops(&mut self) {
        self.conf_ext = (0..self.j.as_ref().map_or(0, |j| j.move_count)).map(|_| std::sync::OnceLock::new()).collect();
        let ne = self.edge_count;
        let mut base = vec![0u32; ne + 1];
        for e in 0..ne { let t = self.e_to[e] as usize; base[e + 1] = base[e] + (self.out_start[t + 1] - self.out_start[t]) as u32; }
        let mut tab = Vec::with_capacity(base[ne] as usize);
        for e in 0..ne {
            let t = self.e_to[e] as usize;
            for q in self.out_start[t]..self.out_start[t + 1] { let en = self.out_edges[q as usize] as usize; tab.push((en as u32, self.hop_compute(e, en))); }
        }
        self.hop_base = base;
        self.hop_tab = tab;
    }

    /// Check the hop table against the JS reference (world pack `hopRef`). Returns mismatches.
    pub fn check_hops(&self, p: &Pack) -> usize {
        let r = p.i32s("hopRef");
        let mut bad = 0;
        for c in r.chunks_exact(4) {
            let h = self.hop(c[0] as usize, c[1] as usize);
            if h.strict != (c[2] != 0) || h.shift != c[3] { bad += 1; }
        }
        bad
    }

    /// roadGraph.js turnAngle (> 0 = right turn), on SimGraph headings.
    fn turn_angle_sim(&self, a: usize, b: usize) -> f64 {
        let mut d = self.e_head_start[b] - self.e_head_end[a];
        while d > jm::PI { d -= 2.0 * jm::PI; }
        while d < -jm::PI { d += 2.0 * jm::PI; }
        d
    }

    fn hop_compute(&self, e: usize, en: usize) -> Hop {
        if let Some(lg) = &self.lg {
            let h = lg.hop_map(e, en);
            return Hop { strict: h.strict, shift: h.shift, scale: self.e_lanes[en] as f64 / self.e_lanes[e] as f64 };
        }
        // roadGraph.js hopMap without lane geometry (synthetic graphs)
        let mut outs: Vec<usize> = Vec::new();
        let mut ins: Vec<usize> = Vec::new();
        let to = self.e_to[e] as usize;
        for q in self.out_start[to]..self.out_start[to + 1] { let o = self.out_edges[q as usize] as usize; if o as i32 != self.e_rev[e] { outs.push(o); } }
        let fr = self.e_from[en] as usize;
        for q in self.in_start[fr]..self.in_start[fr + 1] { let o = self.in_edges[q as usize] as usize; if o as i32 != self.e_rev[en] { ins.push(o); } }
        let simple = outs.len() <= 1 && ins.len() <= 1;
        let strict = simple || (self.e_hw[e] & HW_SHOULDER != 0 && self.e_hw[en] & HW_SHOULDER != 0 && jm::abs(self.turn_angle_sim(e, en)) < 0.5);
        let mut shift = 0;
        if strict {
            if simple { shift = self.e_lanes[e] - self.e_lanes[en]; } else {
                let (mut div, mut mer) = (0i32, 0i32);
                if outs.len() > 1 {
                    outs.sort_by(|&a, &b| jm::cmp_of(self.turn_angle_sim(e, b) - self.turn_angle_sim(e, a)));
                    for &o in &outs { if o == en { break; } div += self.e_lanes[o]; }
                    div = 0.max(div.min(self.e_lanes[e] - self.e_lanes[en]));
                }
                if ins.len() > 1 {
                    ins.sort_by(|&a, &b| jm::cmp_of(self.turn_angle_sim(b, en) - self.turn_angle_sim(a, en)));
                    for &o in &ins { if o == e { break; } mer += self.e_lanes[o]; }
                    mer = 0.max(mer.min(self.e_lanes[en] - self.e_lanes[e]));
                }
                shift = div - mer;
            }
        }
        Hop { strict, shift, scale: self.e_lanes[en] as f64 / self.e_lanes[e] as f64 }
    }

    /// hopMap(g, e, en) (cached table for adjacent pairs).
    #[inline]
    pub fn hop(&self, e: usize, en: usize) -> Hop {
        // entries (out edge, hop) of e's head node, contiguous: [hop_base[e], hop_base[e + 1])
        let (b, b1) = (self.hop_base[e] as usize, self.hop_base[e + 1] as usize);
        for (to, h) in &self.hop_tab[b..b1] { if *to as usize == en { return *h; } }
        self.hop_compute(e, en)
    }
    /// mapLane(g, e, en, l)
    #[inline]
    pub fn map_lane(&self, e: usize, en: usize, l: i32) -> i32 {
        let h = self.hop(e, en);
        if h.strict { l - h.shift } else { jm::min((self.e_lanes[en] - 1) as f64, jm::floor((l as f64 + 0.5) * h.scale)) as i32 }
    }
    /// laneW(e, s)
    #[inline]
    pub fn lane_w(&self, e: usize, s: f64) -> f64 {
        match &self.lg {
            Some(lg) => { let len = self.e_length[e]; lg.lane_width(e, if s < 0.0 { 0.0 } else if s > len { len } else { s }) }
            None => LANE_WIDTH,
        }
    }
    /// edgePoint(e, s, offset, out, segHint)
    pub fn edge_point(&self, e: usize, s: f64, offset: f64, seg_hint: i32) -> Pt {
        let v0 = self.v_start[e] as usize;
        let n = self.v_start[e + 1] as usize - v0;
        let (px, pz, cum) = (&self.px[v0..v0 + n], &self.pz[v0..v0 + n], &self.cum[v0..v0 + n]);
        let mut i = seg_hint;
        if i < 0 { i = 0; }
        if i > n as i32 - 2 { i = n as i32 - 2; }
        let mut i = i as usize;
        while i > 0 && cum[i] > s { i -= 1; }
        while i + 2 < n && cum[i + 1] < s { i += 1; }
        let seg_len = cum[i + 1] - cum[i];
        let t = if seg_len > 0.0 { jm::min(1.0, jm::max(0.0, (s - cum[i]) / seg_len)) } else { 0.0 };
        let mut dx = px[i + 1] - px[i];
        let mut dz = pz[i + 1] - pz[i];
        let l = jm::or(jm::hypot(dx, dz), 1.0);
        dx /= l; dz /= l;
        let y = if self.e_has_ys[e] { let ys = &self.ys[v0..v0 + n]; ys[i] + (ys[i + 1] - ys[i]) * t } else { self.e_y[e] };
        Pt { x: px[i] + dx * t * seg_len - dz * offset, z: pz[i] + dz * t * seg_len + dx * offset, y, hx: dx, hz: dz, seg: i as i32 }
    }
    /// sim.js edgePointDir: road direction at s.
    #[inline]
    pub fn edge_point_dir(&self, e: usize, s: f64, seg: i32) -> Pt {
        match &self.lg { Some(lg) => lg.offset_point(e, s, 0.0, 0.0), None => self.edge_point(e, s, 0.0, seg) }
    }
    /// mapLat(g, e, en, lat, he)
    pub fn map_lat(&self, e: usize, en: usize, lat: f64, he: f64) -> f64 {
        let h = self.hop(e, en);
        if h.strict { return lat - h.shift as f64 * self.lane_w(e, self.e_length[e]); }
        let w = self.e_lanes[en] as f64 * self.lane_w(en, 0.0);
        if let Some(lg) = &self.lg {
            let p = lg.lat_point(e, self.e_length[e], lat);
            let q = lg.offset_point(en, 0.0, 0.0, 0.0);
            let mut l2 = lg.right_edge(en, 0.0) - ((p.x - q.x) * -q.hz + (p.z - q.z) * q.hx);
            if l2 < he { l2 = jm::min(he, w / 2.0); } else if l2 > w - he { l2 = jm::max(w - he, w / 2.0); }
            return l2;
        }
        let v0 = self.v_start[e] as usize;
        let n = self.v_start[e + 1] as usize - v0;
        let (px, pz) = (&self.px[v0..v0 + n], &self.pz[v0..v0 + n]);
        let mut dx = px[n - 1] - px[n - 2];
        let mut dz = pz[n - 1] - pz[n - 2];
        let mut l = jm::or(jm::hypot(dx, dz), 1.0);
        dx /= l; dz /= l;
        let o = self.e_right[e] - lat;
        let wx = px[n - 1] - dz * o;
        let wz = pz[n - 1] + dx * o;
        let u0 = self.v_start[en] as usize;
        let mut ex = self.px[u0 + 1] - self.px[u0];
        let mut ez = self.pz[u0 + 1] - self.pz[u0];
        l = jm::or(jm::hypot(ex, ez), 1.0);
        ex /= l; ez /= l;
        let o2 = (wx - self.px[u0]) * -ez + (wz - self.pz[u0]) * ex;
        let mut l2 = self.e_right[en] - o2;
        if l2 < he { l2 = jm::min(he, w / 2.0); } else if l2 > w - he { l2 = jm::max(w - he, w / 2.0); }
        l2
    }
    /// vehicle.js edgeGrade(e, s, seg)
    pub fn edge_grade(&self, e: usize, s: f64, seg: i32) -> f64 {
        if !self.e_has_ys[e] { return 0.0; }
        let v0 = self.v_start[e] as usize;
        let n = self.v_start[e + 1] as usize - v0;
        let cum = &self.cum[v0..v0 + n];
        let ys = &self.ys[v0..v0 + n];
        let l = cum[n - 1];
        let s0 = if s - GRADE_HALF > 0.0 { s - GRADE_HALF } else { 0.0 };
        let s1 = if s + GRADE_HALF < l { s + GRADE_HALF } else { l };
        if !(s1 - s0 > 0.5) { return 0.0; }
        let gr = (y_along(ys, cum, s1, seg) - y_along(ys, cum, s0, seg)) / (s1 - s0);
        if gr > MAX_GRADE { MAX_GRADE } else if gr < -MAX_GRADE { -MAX_GRADE } else { gr }
    }
    #[inline] pub fn is_shoulder_road(&self, e: usize) -> bool { self.e_hw[e] & HW_SHOULDER != 0 }
    #[inline] pub fn highway(&self, e: usize) -> &str { &self.highway_names[self.e_class[e] as usize] }
    #[inline] pub fn contact_pair_key(a: i64, b: i64) -> i64 { if a < b { a * 4194304 + b } else { b * 4194304 + a } }
}

fn y_along(ys: &[f64], cum: &[f64], s: f64, seg: i32) -> f64 {
    let n = cum.len() as i32;
    let mut k = if seg < 0 { 0 } else if seg > n - 2 { n - 2 } else { seg } as usize;
    while k > 0 && cum[k] > s { k -= 1; }
    while (k as i32) < n - 2 && cum[k + 1] < s { k += 1; }
    let d = cum[k + 1] - cum[k];
    let t = if d > 0.0 { (s - cum[k]) / d } else { 0.0 };
    ys[k] + (ys[k + 1] - ys[k]) * (if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t })
}
