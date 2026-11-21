//! Lane geometry (`src/geom/lanes.js`): THE definition of where every lane is.
//! `s` is the data edge-length coordinate (0..=LaneGraph.length[e]); offsets are metres right of the
//! reference polyline; sim `lat` is metres from the right carriageway edge (positive = left).
use crate::blob::Slice;
use crate::jsmath;
use crate::lanegraph::LaneGraph;

pub const TAPER_M: f64 = 90.0;

#[inline]
pub fn smooth(u: f64) -> f64 { let u = if u < 0.0 { 0.0 } else if u > 1.0 { 1.0 } else { u }; u * u * (3.0 - 2.0 * u) }

/// Arrays of `LaneGeometry` with the JS element types (vx/vz/vs/mx/mz Float64, vy Float32, ...).
#[derive(Clone, Default)]
pub struct LaneGeometry {
    /// E + 1 vertex starts
    pub v_first: Slice<u32>,
    pub vx: Slice<f64>, pub vy: Slice<f32>, pub vz: Slice<f64>, pub vs: Slice<f64>,
    /// unit miter normal (right) per vertex
    pub mx: Slice<f64>, pub mz: Slice<f64>,
    /// right edge offset per edge, per end (2E), transition length per end (2E)
    pub rd: Slice<f32>, pub rend: Slice<f32>, pub tau: Slice<f32>,
    /// lanes ending / starting per end, right / left (2E)
    pub tap_r: Slice<u8>, pub tap_l: Slice<u8>,
    /// paved run-outs per end (2E)
    pub x_r: Slice<u8>, pub x_l: Slice<u8>,
    pub lw: f64,
    /// lane width per edge, and at each end (2E)
    pub lane_w: Slice<f32>, pub lane_w_end: Slice<f32>,
    /// copy of LaneGraph.length (f32) and lanes, so the accessors need no graph reference
    pub length: Slice<f32>, pub lanes: Slice<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Pt { pub x: f64, pub y: f64, pub z: f64, pub hx: f64, pub hz: f64 }

#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds { pub right: f64, pub left: f64, pub r: f64, pub lane_right: f64, pub lane_left: f64, pub lane_width: f64 }

impl LaneGeometry {
    /// `buildLaneGeometry(G)` (port in progress: see rust/ARCHITECTURE.md).
    pub fn build(g: &LaneGraph) -> LaneGeometry { crate::lanes_build::build_lane_geometry(g) }

    pub fn right_edge(&self, e: usize, s: f64) -> f64 {
        let l = self.length[e] as f64; let rd = self.rd[e] as f64; let mut r = rd;
        let (ts, te) = (self.tau[2 * e] as f64, self.tau[2 * e + 1] as f64);
        if ts > 0.0 { r += (self.rend[2 * e] as f64 - rd) * smooth(1.0 - s / ts); }
        if te > 0.0 { r += (self.rend[2 * e + 1] as f64 - rd) * smooth(1.0 - (l - s) / te); }
        r
    }
    pub fn right_edge_slope(&self, e: usize, s: f64) -> f64 {
        let d = 0.5; let l = self.length[e] as f64;
        (self.right_edge(e, jsmath::min(l, s + d)) - self.right_edge(e, jsmath::max(0.0, s - d))) / (2.0 * d)
    }
    pub fn lane_width(&self, e: usize, s: f64) -> f64 {
        let l = self.length[e] as f64; let w = self.lane_w[e] as f64;
        let (ts, te) = (self.tau[2 * e] as f64, self.tau[2 * e + 1] as f64);
        let mut r = w;
        if ts > 0.0 { r += (self.lane_w_end[2 * e] as f64 - w) * smooth(1.0 - s / ts); }
        if te > 0.0 { r += (self.lane_w_end[2 * e + 1] as f64 - w) * smooth(1.0 - (l - s) / te); }
        r
    }
    pub fn bounds(&self, e: usize, s: f64) -> Bounds {
        let n = self.lanes[e] as f64; let w = self.lane_width(e, s); let l = self.length[e] as f64;
        let r = self.right_edge(e, s); let t = jsmath::min(TAPER_M, l);
        let fs = if t > 0.0 { jsmath::max(0.0, 1.0 - s / t) } else { 0.0 };
        let fe = if t > 0.0 { jsmath::max(0.0, 1.0 - (l - s) / t) } else { 0.0 };
        let ex_r = (self.x_r[2 * e] as f64 * fs + self.x_r[2 * e + 1] as f64 * fe) * w;
        let ex_l = (self.x_l[2 * e] as f64 * fs + self.x_l[2 * e + 1] as f64 * fe) * w;
        Bounds { right: r + ex_r, left: r - n * w - ex_l, r, lane_right: r, lane_left: r - n * w, lane_width: w }
    }
    fn locate(&self, e: usize, s: f64) -> (usize, f64) {
        let v0 = self.v_first[e] as usize; let v1 = self.v_first[e + 1] as usize - 1; let vs = &self.vs;
        if s <= 0.0 { return (v0, 0.0); }
        if s >= vs[v1] { return (v1 - 1, 1.0); }
        let (mut lo, mut hi) = (v0, v1);
        while hi - lo > 1 { let m = (lo + hi) >> 1; if vs[m] <= s { lo = m; } else { hi = m; } }
        let d = vs[lo + 1] - vs[lo];
        (lo, if d > 0.0 { (s - vs[lo]) / d } else { 0.0 })
    }
    pub fn offset_point(&self, e: usize, s: f64, off: f64, off_slope: f64) -> Pt {
        let (i, t) = self.locate(e, s); let j = i + 1;
        let (vx, vy, vz, mx, mz) = (&self.vx, &self.vy, &self.vz, &self.mx, &self.mz);
        let nx = mx[i] + (mx[j] - mx[i]) * t; let nz = mz[i] + (mz[j] - mz[i]) * t;
        let x = vx[i] + (vx[j] - vx[i]) * t + off * nx;
        let y = vy[i] as f64 + (vy[j] as f64 - vy[i] as f64) * t;
        let z = vz[i] + (vz[j] - vz[i]) * t + off * nz;
        let mut ds = self.vs[j] - self.vs[i]; if ds == 0.0 || ds.is_nan() { ds = 1.0; }
        let hx = (vx[j] - vx[i] + off * (mx[j] - mx[i])) / ds + off_slope * nx;
        let hz = (vz[j] - vz[i] + off * (mz[j] - mz[i])) / ds + off_slope * nz;
        let mut l = jsmath::hypot(hx, hz); if l == 0.0 || l.is_nan() { l = 1.0; }
        Pt { x, y, z, hx: hx / l, hz: hz / l }
    }
    pub fn lat_point(&self, e: usize, s: f64, lat: f64) -> Pt { self.offset_point(e, s, self.right_edge(e, s) - lat, self.right_edge_slope(e, s)) }
    pub fn lane_point(&self, e: usize, s: f64, lane: usize) -> Pt { self.lat_point(e, s, (lane as f64 + 0.5) * self.lane_width(e, s)) }
    pub fn lat_to_lane(&self, e: usize, s: f64, lat: f64) -> i64 { (lat / self.lane_width(e, s)).floor() as i64 }
}
