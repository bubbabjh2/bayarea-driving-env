//! `buildJunctions` / `crosswalkLegs` port (src/geom/junctions.js). OWNER: geom port agent. Output equals
//! `npm run build:junctions` (data/junctions.bin) byte for byte: same evaluation order, f64 math where the JS
//! keeps doubles (connector points, conflicts inside a junction) and f32 where it reads its Float32Arrays
//! (crossingFix, cross-junction conflicts), JS Map insertion order for the crossing list.
//! Parallel (rayon) only over independent per-connector / per-junction / per-edge work, merged in JS order.
use crate::hops::{turn_angle, v8_sort, HopTable};
use crate::jsmath::{self, hypot};
use crate::junctions::{CrosswalkLegs, JunctionArrays};
use crate::lanegraph::LaneGraph;
use crate::lanes::{LaneGeometry, Pt};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::f64::consts::PI;

const CURB: f64 = 1.0;
const CROSSWALK: f64 = 3.5;
const R_MIN: f64 = 6.0;
const CONFLICT_SEP: f64 = 3.0;
const GORE_CLEAR: f64 = 1.3;

#[inline] fn max(a: f64, b: f64) -> f64 { jsmath::max(a, b) }
#[inline] fn min(a: f64, b: f64) -> f64 { jsmath::min(a, b) }
#[inline] fn or1(x: f64) -> f64 { if x == 0.0 || x.is_nan() { 1.0 } else { x } }

type P3 = [f64; 3];

/// One control record as `crosswalkLegs` reads it (`data/controls.json` controls[]).
#[derive(Clone, Debug)]
pub struct ControlLegs { pub node: f64, pub ty: String, pub all_way: bool, pub approach_edges: Vec<u32> }

/// `crosswalkLegs(LG, controls, crosswalkPoints)`.
pub fn crosswalk_legs(g: &LaneGraph, controls: Option<&[ControlLegs]>, points: Option<&[[f64; 2]]>) -> CrosswalkLegs {
    let mut legs = CrosswalkLegs::default();
    let mark = |legs: &mut CrosswalkLegs, e: u32, at_end: bool| {
        let t = g.twin[e as usize];
        if at_end { legs.crosswalk_stop.insert(e); if t >= 0 { legs.crosswalk_exit.insert(t as u32); } }
        else { legs.crosswalk_exit.insert(e); if t >= 0 { legs.crosswalk_stop.insert(t as u32); } }
    };
    if let Some(cs) = controls {
        for c in cs {
            if c.node < 0.0 { continue; }
            if c.ty == "signal" || (c.ty == "stop" && c.all_way) { for &a in &c.approach_edges { mark(&mut legs, a, true); } }
        }
    }
    if let Some(points) = points.filter(|p| !p.is_empty()) {
        let cell = 50.0;
        let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
        for e in 0..g.edge_count {
            if g.twin[e] >= 0 && (g.twin[e] as usize) < e { continue; }
            let nd = g.nodes_of(e);
            for k in 0..nd.len().saturating_sub(1) {
                let (a, b) = (nd[k] as usize, nd[k + 1] as usize);
                let (x0, x1) = (min(g.x(a), g.x(b)), max(g.x(a), g.x(b)));
                let (z0, z1) = (min(g.z(a), g.z(b)), max(g.z(a), g.z(b)));
                let (gx0, gx1, gz0, gz1) = ((x0 / cell).floor() as i64, (x1 / cell).floor() as i64, (z0 / cell).floor() as i64, (z1 / cell).floor() as i64);
                for gx in gx0..=gx1 { for gz in gz0..=gz1 { let l = grid.entry((gx, gz)).or_default(); l.push(e as u32); l.push(k as u32); } }
            }
        }
        for &[px, pz] in points {
            let Some(l) = grid.get(&((px / cell).floor() as i64, (pz / cell).floor() as i64)) else { continue };
            let (mut best, mut be, mut bs) = (4.0f64, -1i64, 0.0f64);
            for i in (0..l.len()).step_by(2) {
                let (e, k) = (l[i] as usize, l[i + 1] as usize); let nd = g.nodes_of(e);
                let (a, b) = (nd[k] as usize, nd[k + 1] as usize);
                let (ax, az, bx, bz) = (g.x(a), g.z(a), g.x(b), g.z(b));
                let (dx, dz) = (bx - ax, bz - az); let ll = or1(dx * dx + dz * dz);
                let t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / ll));
                let d = hypot(ax + dx * t - px, az + dz * t - pz);
                if d < best {
                    let mut acc = 0.0;
                    for q in 0..k { let (n0, n1) = (nd[q] as usize, nd[q + 1] as usize); acc += hypot(g.x(n1) - g.x(n0), g.z(n1) - g.z(n0)); }
                    best = d; be = e as i64; bs = acc + ll.sqrt() * t;
                }
            }
            if be < 0 { continue; }
            let l = g.length[be as usize] as f64;
            if l - bs <= 25.0 { mark(&mut legs, be as u32, true); }
            if bs <= 25.0 { mark(&mut legs, be as u32, false); }
        }
    }
    legs
}

/// Setback (m) from node n back along edge e (or forward along it, at its start) to the mouth.
fn setback_at(g: &LaneGraph, lg: &LaneGeometry, n: usize, e: usize) -> f64 {
    let gt = g.twin[e];
    let mut roads: Vec<u32> = Vec::new();
    let mut hw = 0.0f64;
    let mut visit = |h: u32| {
        if h as usize == e || h as i32 == gt { return; }
        let t = g.twin[h as usize];
        let r = if t >= 0 { h.min(t as u32) } else { h };
        if !roads.contains(&r) { roads.push(r); }
        hw = max(hw, (g.lanes[h as usize] as f64 + g.opp[h as usize] as f64) * lg.lane_w[h as usize] as f64 / 2.0);
    };
    for &h in g.out_edges(n) { visit(h); }
    for &h in g.in_edges(n) { visit(h); }
    if roads.len() >= 2 { hw + CURB } else { 0.0 }
}

#[inline]
fn lane_pt(lg: &LaneGeometry, e: usize, s: f64, lane: f64) -> Pt { lg.lat_point(e, s, (lane + 0.5) * lg.lane_width(e, s)) }

#[derive(Clone, Debug)]
struct Conn { curve: Vec<P3>, s_in: f64, s_out: f64, radius: f64 }

/// `connector(LG, a, la, b, lb, sIn0, sOut0, th, rMin, lim)`.
#[allow(clippy::too_many_arguments)]
fn connector(g: &LaneGraph, lg: &LaneGeometry, a: usize, la: usize, b: usize, lb: usize, s_in0: f64, s_out0: f64, th: f64, r_min: f64,
             floor_in: f64, cap_out: f64) -> Conn {
    let (la_len, lb_len) = (g.length[a] as f64, g.length[b] as f64);
    let min_in = max(max(min(s_in0, 0.05 * la_len), s_in0 - 40.0), floor_in);
    let max_out = min(min(max(s_out0, 0.95 * lb_len), s_out0 + 40.0), cap_out);
    let (mut s_in, mut s_out) = (s_in0, s_out0);
    let mut curve: Vec<P3> = Vec::new();
    let (la_f, lb_f) = (la as f64, lb as f64);
    for iter in 0..10 {
        let pa = lane_pt(lg, a, s_in, la_f); let pb = lane_pt(lg, b, s_out, lb_f);
        let cr = pa.hx * pb.hz - pa.hz * pb.hx; let dt = pa.hx * pb.hx + pa.hz * pb.hz;
        let turn = jsmath::atan2(cr, dt);
        let (mut t0, mut t1) = (-1.0f64, -1.0f64);
        if cr.abs() > 1e-3 {
            t0 = ((pb.x - pa.x) * pb.hz - (pb.z - pa.z) * pb.hx) / cr;
            t1 = -((pb.x - pa.x) * pa.hz - (pb.z - pa.z) * pa.hx) / cr;
        }
        let at = turn.abs(); let straight_th = th.abs() < 0.35;
        let corner = at > 0.15 && at < 2.6 && t0 > 0.05 && t1 > 0.05 && !straight_th;
        if corner {
            let tn = jsmath::tan(at / 2.0); let need = r_min * tn;
            if (t0 < need - 0.01 || t1 < need - 0.01) && iter < 9 {
                let n_in = max(min_in, s_in - max(0.0, need - t0) * 1.2 - 0.5);
                let n_out = min(max_out, s_out + max(0.0, need - t1) * 1.2 - 0.0 + 0.5);
                if n_in < s_in - 0.05 || n_out > s_out + 0.05 { s_in = n_in; s_out = n_out; continue; }
            }
            let r = min(40.0, min(t0, t1) / tn); let tt = r * tn;
            curve = fillet(&pa, &pb, t0, t1, tt, r, turn);
            break;
        }
        if !corner && at > 0.15 && at < 2.6 && !straight_th && iter < 9 {
            let n_in = if t0 <= 0.05 { max(min_in, s_in - 4.0) } else { s_in };
            let n_out = if t1 <= 0.05 { min(max_out, s_out + 4.0) } else { s_out };
            if n_in < s_in - 0.05 || n_out > s_out + 0.05 { s_in = n_in; s_out = n_out; continue; }
        }
        if !straight_th && (at >= 2.6 || (at > 0.15 && !corner)) { curve = dubins(&pa, &pb, if turn >= 0.0 { 1.0 } else { -1.0 }, r_min); break; }
        let chord = hypot(pb.x - pa.x, pb.z - pa.z); let k = max(0.5, chord * 0.4);
        let p0 = [pa.x, pa.y, pa.z]; let p1 = [pa.x + pa.hx * k, pa.y, pa.z + pa.hz * k];
        let p2 = [pb.x - pb.hx * k, pb.y, pb.z - pb.hz * k]; let p3 = [pb.x, pb.y, pb.z];
        let n = min(40.0, max(4.0, (chord / 1.5).ceil())) as usize;
        curve = (0..=n).map(|i| bezier(&p0, &p1, &p2, &p3, n, i)).collect();
        if min_radius(&curve) >= r_min - 0.05 || iter >= 9 { break; }
        let n_in = max(min_in, s_in - 4.0); let n_out = min(max_out, s_out + 4.0);
        if n_in >= s_in - 0.05 && n_out <= s_out + 0.05 { break; }
        s_in = n_in; s_out = n_out;
    }
    let pa = lane_pt(lg, a, s_in, la_f); let pb = lane_pt(lg, b, s_out, lb_f);
    let curve = tangent_ends(curve, &pa, &pb);
    let radius = min_radius(&curve);
    Conn { curve, s_in, s_out, radius }
}

fn tangent_ends(c: Vec<P3>, a: &Pt, b: &Pt) -> Vec<P3> {
    if c.len() < 2 { return c; }
    let ang = |dx: f64, dz: f64, hx: f64, hz: f64| jsmath::acos(max(-1.0, min(1.0, (dx * hx + dz * hz) / or1(hypot(dx, dz)))));
    let mut out = c;
    out[0] = [a.x, a.y, a.z]; let last = out.len() - 1; out[last] = [b.x, b.y, b.z];
    let d0 = hypot(out[1][0] - out[0][0], out[1][2] - out[0][2]);
    if ang(out[1][0] - out[0][0], out[1][2] - out[0][2], a.hx, a.hz) > 0.005 && d0 > 0.03 {
        let k = min(0.5, d0 / 3.0);
        out.insert(1, [a.x + a.hx * k, a.y, a.z + a.hz * k]);
    }
    let n = out.len();
    let d1 = hypot(out[n - 1][0] - out[n - 2][0], out[n - 1][2] - out[n - 2][2]);
    if ang(out[n - 1][0] - out[n - 2][0], out[n - 1][2] - out[n - 2][2], b.hx, b.hz) > 0.005 && d1 > 0.03 {
        let k = min(0.5, d1 / 3.0);
        out.insert(n - 1, [b.x - b.hx * k, b.y, b.z - b.hz * k]);
    }
    out
}

fn fillet(a: &Pt, b: &Pt, t0: f64, t1: f64, tt: f64, r: f64, turn: f64) -> Vec<P3> {
    let mut out: Vec<P3> = Vec::new();
    let lerp_line = |out: &mut Vec<P3>, x0: f64, y0: f64, z0: f64, x1: f64, y1: f64, z1: f64, step: f64| {
        let l = hypot(x1 - x0, z1 - z0); let n = max(1.0, (l / step).ceil()) as usize; let nf = n as f64;
        for i in 0..n { let f = i as f64 / nf; out.push([x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, z0 + (z1 - z0) * f]); }
    };
    let (cx, cz) = (a.x + a.hx * t0, a.z + a.hz * t0);
    let (tax, taz, tbx, tbz) = (cx - a.hx * tt, cz - a.hz * tt, cx + b.hx * tt, cz + b.hz * tt);
    let total = or1((t0 - tt) + turn.abs() * r + (t1 - tt));
    let y_at = |d: f64| a.y + (b.y - a.y) * min(1.0, d / total);
    lerp_line(&mut out, a.x, a.y, a.z, tax, y_at(t0 - tt), taz, 6.0);
    let side = if turn >= 0.0 { 1.0 } else { -1.0 };
    let (nx, nz) = if side > 0.0 { (-a.hz, a.hx) } else { (a.hz, -a.hx) };
    let (ox, oz) = (tax + nx * r, taz + nz * r);
    let a0 = jsmath::atan2(taz - oz, tax - ox); let sweep = turn.abs();
    let n = max(2.0, (sweep / 0.12).ceil()) as usize; let nf = n as f64;
    for i in 0..n {
        let f = i as f64 / nf;
        let ang = a0 + side * sweep * f;
        out.push([ox + r * jsmath::cos(ang), y_at(t0 - tt + r * sweep * f), oz + r * jsmath::sin(ang)]);
    }
    lerp_line(&mut out, tbx, y_at(t0 - tt + r * sweep), tbz, b.x, b.y, b.z, 6.0);
    out.push([b.x, b.y, b.z]);
    let mut res: Vec<P3> = vec![out[0]];
    for i in 1..out.len() {
        let q = *res.last().unwrap();
        if hypot(out[i][0] - q[0], out[i][2] - q[2]) > 0.05 { res.push(out[i]); }
        else if i == out.len() - 1 { *res.last_mut().unwrap() = out[i]; }
    }
    res
}

fn min_radius(c: &[P3]) -> f64 {
    let mut mr = 1e9;
    for k in 1..c.len().saturating_sub(1) {
        let ([ax, _, az], [bx, _, bz], [cx, _, cz]) = (c[k - 1], c[k], c[k + 1]);
        let (ab, bc, ca) = (hypot(bx - ax, bz - az), hypot(cx - bx, cz - bz), hypot(ax - cx, az - cz));
        if ab < 0.5 || bc < 0.5 { continue; }
        let area2 = ((bx - ax) * (cz - az) - (bz - az) * (cx - ax)).abs();
        if area2 > 1e-9 { mr = min(mr, (ab * bc * ca) / (2.0 * area2)); }
    }
    mr
}

/// Cubic Bezier at t = i / n. JS `x ** 3` is V8's fdlibm pow (not x * x * x): jsmath::pow.
fn bezier(p0: &P3, p1: &P3, p2: &P3, p3: &P3, n: usize, i: usize) -> P3 {
    let t = i as f64 / n as f64;
    let a = jsmath::pow(1.0 - t, 3.0);
    let b = 3.0 * jsmath::pow(1.0 - t, 2.0) * t;
    let c = 3.0 * (1.0 - t) * t * t;
    let d = jsmath::pow(t, 3.0);
    [a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1], a * p0[2] + b * p1[2] + c * p2[2] + d * p3[2]]
}

/// Closest distance between segments AB and CD: (d, u on AB, v on CD).
#[allow(clippy::too_many_arguments)]
fn seg_seg_dist(ax: f64, az: f64, bx: f64, bz: f64, cx: f64, cz: f64, dx: f64, dz: f64) -> (f64, f64, f64) {
    let pt = |px: f64, pz: f64, x0: f64, z0: f64, x1: f64, z1: f64| -> (f64, f64) {
        let (ex, ez) = (x1 - x0, z1 - z0); let mut l = ex * ex + ez * ez; if l == 0.0 || l.is_nan() { l = 1e-12; }
        let t = max(0.0, min(1.0, ((px - x0) * ex + (pz - z0) * ez) / l));
        (hypot(x0 + ex * t - px, z0 + ez * t - pz), t)
    };
    let mut best = (1e18, 0.0, 0.0);
    let r = pt(ax, az, cx, cz, dx, dz); if r.0 < best.0 { best = (r.0, 0.0, r.1); }
    let r = pt(bx, bz, cx, cz, dx, dz); if r.0 < best.0 { best = (r.0, 1.0, r.1); }
    let r = pt(cx, cz, ax, az, bx, bz); if r.0 < best.0 { best = (r.0, r.1, 0.0); }
    let r = pt(dx, dz, ax, az, bx, bz); if r.0 < best.0 { best = (r.0, r.1, 1.0); }
    best
}

#[allow(clippy::too_many_arguments)]
fn seg_cross(ax: f64, az: f64, bx: f64, bz: f64, cx: f64, cz: f64, dx: f64, dz: f64) -> f64 {
    let (d1x, d1z, d2x, d2z) = (bx - ax, bz - az, dx - cx, dz - cz);
    let den = d1x * d2z - d1z * d2x;
    if den.abs() < 1e-12 { return -1.0; }
    let u = ((cx - ax) * d2z - (cz - az) * d2x) / den; let v = ((cx - ax) * d1z - (cz - az) * d1x) / den;
    if (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v) { u } else { -1.0 }
}

fn dubins(a: &Pt, b: &Pt, side: f64, r: f64) -> Vec<P3> {
    let n_a = if side > 0.0 { [-a.hz, a.hx] } else { [a.hz, -a.hx] };
    let n_b = if side > 0.0 { [-b.hz, b.hx] } else { [b.hz, -b.hx] };
    let c1 = [a.x + r * n_a[0], a.z + r * n_a[1]]; let c2 = [b.x + r * n_b[0], b.z + r * n_b[1]];
    let (mut ux, mut uz) = (c2[0] - c1[0], c2[1] - c1[1]);
    let d = hypot(ux, uz);
    if d < 1e-6 { ux = a.hx; uz = a.hz; } else { ux /= d; uz /= d; }
    let n_u = if side > 0.0 { [-uz, ux] } else { [uz, -ux] };
    let t1 = [c1[0] - r * n_u[0], c1[1] - r * n_u[1]]; let t2 = [c2[0] - r * n_u[0], c2[1] - r * n_u[1]];
    let ang = |p: [f64; 2], c: [f64; 2]| jsmath::atan2(p[1] - c[1], p[0] - c[0]);
    let sweep = |from: f64, to: f64| { let a = if side > 0.0 { to - from } else { from - to }; ((a % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI) };
    let mut out: Vec<P3> = Vec::new();
    let arc = |out: &mut Vec<P3>, c: [f64; 2], a0: f64, phi: f64, y: f64| {
        let n = max(1.0, (phi / 0.25).ceil()) as usize; let nf = n as f64;
        for i in 0..=n { let an = a0 + side * phi * (i as f64 / nf); out.push([c[0] + r * jsmath::cos(an), y, c[1] + r * jsmath::sin(an)]); }
    };
    let a1 = ang([a.x, a.z], c1); let p1 = sweep(a1, ang(t1, c1));
    arc(&mut out, c1, a1, p1, a.y);
    let a2 = ang(t2, c2); let p2 = sweep(a2, ang([b.x, b.z], c2));
    let n_line = max(1.0, (d / 3.0).ceil()) as usize; let nl = n_line as f64;
    for i in 1..n_line { let f = i as f64 / nl; out.push([t1[0] + (t2[0] - t1[0]) * f, a.y + (b.y - a.y) * f, t1[1] + (t2[1] - t1[1]) * f]); }
    let start = out.len();
    arc(&mut out, c2, a2, p2, b.y);
    let last = out.len() - 1; out[last] = [b.x, b.y, b.z];
    out[0] = [a.x, a.y, a.z];
    let mut res = Vec::with_capacity(out.len());
    for i in 0..out.len() {
        if i == 0 || hypot(out[i][0] - out[i - 1][0], out[i][2] - out[i - 1][2]) > 1e-3 || i == start { res.push(out[i]); }
    }
    res
}

fn convex_hull(p: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut pts = p.to_vec();
    v8_sort(&mut pts, |a, b| { let d = a[0] - b[0]; if d != 0.0 && !d.is_nan() { d } else { a[1] - b[1] } });
    if pts.len() < 3 { return pts; }
    let cross = |o: &[f64; 2], a: &[f64; 2], b: &[f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let (mut lower, mut upper): (Vec<[f64; 2]>, Vec<[f64; 2]>) = (Vec::new(), Vec::new());
    for q in &pts { while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], q) <= 0.0 { lower.pop(); } lower.push(*q); }
    for q in pts.iter().rev() { while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], q) <= 0.0 { upper.pop(); } upper.push(*q); }
    lower.pop(); upper.pop();
    lower.extend(upper);
    lower
}

struct Spec { n: u32, a: u32, la: u32, b: u32, lb: u32, th: f64, cn: Conn }

/// Built junction set (f32 / u32 arrays as the JS JunctionSet holds them).
struct Built {
    mouth_s: Vec<f32>, gore_s: Vec<f32>, j_node: Vec<u32>, j_first_m: Vec<u32>, j_count_m: Vec<u32>, j_first_c: Vec<u32>, j_count_c: Vec<u32>,
    j_first_h: Vec<u32>, j_count_h: Vec<u32>, m_info: Vec<u32>, m_f: Vec<f32>, pts: Vec<f32>, conf: Vec<u32>, conf_s: Vec<f32>, hull: Vec<f32>,
}

struct Ctx<'a> { g: &'a LaneGraph, lg: &'a LaneGeometry, ht: HopTable, is_j: Vec<bool>, cw_stop: HashSet<u32>, cw_exit: HashSet<u32> }

/// Per-junction output with junction-local movement / point indices.
#[derive(Default)]
struct JLocal { m_info: Vec<u32>, m_f: Vec<f32>, pts: Vec<f64>, pts32: Vec<f32>, conf: Vec<u32>, conf_s: Vec<f32>, hull: Vec<f32>, too_tight: usize }

fn junction_local(ctx: &Ctx, mouth_s: &[f32], n: usize, specs: &[Spec]) -> JLocal {
    let (g, lg) = (ctx.g, ctx.lg);
    let (ins, outs) = (g.in_edges(n), g.out_edges(n));
    let mut o = JLocal::default();
    let mut corners: Vec<[f64; 2]> = Vec::with_capacity(2 * (ins.len() + outs.len()));
    for &e in ins.iter().chain(outs.iter()) {
        let e = e as usize;
        let s = if ins.contains(&(e as u32)) { mouth_s[2 * e + 1] } else { mouth_s[2 * e] } as f64;
        let bd = lg.bounds(e, s);
        let p = lg.offset_point(e, s, bd.right, 0.0); corners.push([p.x, p.z]);
        let p = lg.offset_point(e, s, bd.left, 0.0); corners.push([p.x, p.z]);
    }
    for sp in specs {
        let na = g.lanes[sp.a as usize] as u32;
        if sp.cn.radius < R_MIN - 0.05 { o.too_tight += 1; }
        let first = (o.pts.len() / 4) as u32;
        let mut acc = 0.0; let mut prev: Option<&P3> = None;
        for p in &sp.cn.curve {
            if let Some(q) = prev { acc += hypot(p[0] - q[0], p[2] - q[2]); }
            o.pts.extend_from_slice(&[p[0], p[1], p[2], acc]);
            prev = Some(p);
        }
        let pref = if sp.th > 0.5 { sp.la == 0 } else if sp.th < -0.5 { sp.la == na - 1 } else { true };
        o.m_info.extend_from_slice(&[sp.a, sp.la, sp.b, sp.lb, first, sp.cn.curve.len() as u32, pref as u32]);
        o.m_f.extend_from_slice(&[sp.cn.s_in as f32, sp.cn.s_out as f32, acc as f32, sp.th as f32]);
    }
    let (mi, pts) = (&o.m_info, &o.pts);
    let m1 = mi.len() / 7;
    let mut conf = Vec::new(); let mut conf_s = Vec::new();
    for i in 0..m1 {
        for j in i + 1..m1 {
            let (fi, ci, fj, cj) = (mi[7 * i + 4] as usize, mi[7 * i + 5] as usize, mi[7 * j + 4] as usize, mi[7 * j + 5] as usize);
            let same_in = mi[7 * i] == mi[7 * j] && mi[7 * i + 1] == mi[7 * j + 1];
            let same_out = mi[7 * i + 2] == mi[7 * j + 2] && mi[7 * i + 3] == mi[7 * j + 3];
            if same_in { continue; }
            let (exi, exj) = (4 * (fi + ci - 1), 4 * (fj + cj - 1));
            if same_out || hypot(pts[exi] - pts[exj], pts[exi + 2] - pts[exj + 2]) < CONFLICT_SEP {
                conf.extend_from_slice(&[i as u32, j as u32, 1]); conf_s.extend_from_slice(&[pts[exi + 3] as f32, pts[exj + 3] as f32]); continue;
            }
            let mut hit = false;
            'p: for p in 0..ci.saturating_sub(1) {
                let (ax, az, bx, bz) = (pts[4 * (fi + p)], pts[4 * (fi + p) + 2], pts[4 * (fi + p + 1)], pts[4 * (fi + p + 1) + 2]);
                for q in 0..cj.saturating_sub(1) {
                    let (cx, cz, dx, dz) = (pts[4 * (fj + q)], pts[4 * (fj + q) + 2], pts[4 * (fj + q + 1)], pts[4 * (fj + q + 1) + 2]);
                    if max(ax, bx) + 2.0 < min(cx, dx) || max(cx, dx) + 2.0 < min(ax, bx) || max(az, bz) + 2.0 < min(cz, dz) || max(cz, dz) + 2.0 < min(az, bz) { continue; }
                    let u = seg_cross(ax, az, bx, bz, cx, cz, dx, dz);
                    if u >= 0.0 {
                        let si = pts[4 * (fi + p) + 3] + u * (pts[4 * (fi + p + 1) + 3] - pts[4 * (fi + p) + 3]);
                        conf.extend_from_slice(&[i as u32, j as u32, 0]); conf_s.extend_from_slice(&[si as f32, pts[4 * (fj + q) + 3] as f32]);
                        hit = true; break 'p;
                    }
                }
            }
            if hit { continue; }
            'p2: for p in 0..ci.saturating_sub(1) {
                let (ax, az, bx, bz) = (pts[4 * (fi + p)], pts[4 * (fi + p) + 2], pts[4 * (fi + p + 1)], pts[4 * (fi + p + 1) + 2]);
                for q in 0..cj.saturating_sub(1) {
                    let (cx, cz, dx, dz) = (pts[4 * (fj + q)], pts[4 * (fj + q) + 2], pts[4 * (fj + q + 1)], pts[4 * (fj + q + 1) + 2]);
                    if max(ax, bx) + CONFLICT_SEP < min(cx, dx) || max(cx, dx) + CONFLICT_SEP < min(ax, bx) || max(az, bz) + CONFLICT_SEP < min(cz, dz) || max(cz, dz) + CONFLICT_SEP < min(az, bz) { continue; }
                    let (d, u, v) = seg_seg_dist(ax, az, bx, bz, cx, cz, dx, dz);
                    if d < CONFLICT_SEP {
                        let si = pts[4 * (fi + p) + 3] + u * (pts[4 * (fi + p + 1) + 3] - pts[4 * (fi + p) + 3]);
                        let sj = pts[4 * (fj + q) + 3] + v * (pts[4 * (fj + q + 1) + 3] - pts[4 * (fj + q) + 3]);
                        conf.extend_from_slice(&[i as u32, j as u32, 0]); conf_s.extend_from_slice(&[si as f32, sj as f32]);
                        break 'p2;
                    }
                }
            }
        }
    }
    o.conf = conf; o.conf_s = conf_s;
    // the conflicts above use the doubles; the set keeps Float32Array.from(pts)
    o.pts32 = o.pts.iter().map(|&v| v as f32).collect(); o.pts = Vec::new();
    for [x, z] in convex_hull(&corners) { o.hull.push(x as f32); o.hull.push(z as f32); }
    o
}

/// Ramp gore wait point of in-edge a at merge node n (out-edge b), or None.
fn gore_of(ctx: &Ctx, n: usize, a: usize, b: usize) -> Option<(f32, bool)> {
    let (g, lg) = (ctx.g, ctx.lg);
    let ins = g.in_edges(n);
    if !(g.cls[a] == 1 || g.cls[a] == 3) { return None; }
    let h = ctx.ht.get(g, a, b);
    if !h.strict { return None; }
    let mut m: i64 = -1;
    for &o in ins { let o = o as usize; if o != a && (m < 0 || g.lanes[o] > g.lanes[m as usize]) { m = o as i64; } }
    if m < 0 { return None; }
    let m = m as usize;
    let hm = ctx.ht.get(g, m, b);
    let m_lo = -hm.shift; let r_hi = g.lanes[a] as i32 - 1 - h.shift;
    if r_hi < m_lo { return None; }
    let (l, lm) = (g.length[a] as f64, g.length[m] as f64);
    let side_of = |p: &Pt, right: f64| -> f64 {
        let (mut best, mut sd) = (1e9f64, 0.0f64);
        let mut sm = lm;
        while sm >= max(0.0, lm - 450.0) {
            let q = lg.offset_point(m, sm, right, 0.0);
            let d = hypot(p.x - q.x, p.z - q.z);
            if d < best { best = d; sd = (p.x - q.x) * -q.hz + (p.z - q.z) * q.hx; }
            sm -= 3.0;
        }
        sd
    };
    let pa = lane_pt(lg, a, max(0.0, l - 400.0), 0.0);
    let from_right = side_of(&pa, 0.0) >= 0.0;
    let mut edge_pts: Vec<f64> = Vec::new();
    let mut sm = lm;
    while sm >= max(0.0, lm - 450.0) {
        let bd = lg.bounds(m, sm);
        let q = lg.offset_point(m, sm, if from_right { bd.lane_right } else { bd.lane_left }, 0.0);
        edge_pts.extend_from_slice(&[q.x, q.z, -q.hz, q.hx]);
        sm -= 3.0;
    }
    let (mut sg, mut best_s, mut best_side) = (-1.0f64, max(0.0, l - 60.0), -1e9f64);
    let mut s = l;
    while s >= max(0.0, l - 400.0) {
        let p = lane_pt(lg, a, s, if from_right { 0.0 } else { g.lanes[a] as f64 - 1.0 });
        let (mut best, mut side) = (1e9f64, 0.0f64);
        for q in (0..edge_pts.len()).step_by(4) {
            let d = hypot(p.x - edge_pts[q], p.z - edge_pts[q + 1]);
            if d < best { best = d; side = (p.x - edge_pts[q]) * edge_pts[q + 2] + (p.z - edge_pts[q + 1]) * edge_pts[q + 3]; }
        }
        if !from_right { side = -side; }
        if side > best_side + 0.1 { best_side = side; best_s = s; }
        if side > 0.0 && side - lg.lane_width(a, s) / 2.0 >= GORE_CLEAR { sg = s; break; }
        s -= 2.0;
    }
    Some((if sg >= 0.0 { sg } else { best_s } as f32, sg < 0.0))
}

fn build_once(ctx: &Ctx, mouth_adjust: &[(u32, f64, f64)]) -> (Built, usize, usize) {
    let (g, lg) = (ctx.g, ctx.lg);
    let (e_n, n_n) = (g.edge_count, g.node_count);
    // ---- mouths
    let mut mouth_s = vec![0f32; 2 * e_n];
    for e in 0..e_n {
        let l = g.length[e] as f64;
        let s0 = if ctx.is_j[g.from(e)] { min(setback_at(g, lg, g.from(e), e) + if ctx.cw_exit.contains(&(e as u32)) { CROSSWALK } else { 0.0 }, 0.45 * l) } else { 0.0 };
        let s1 = if ctx.is_j[g.to(e)] { min(setback_at(g, lg, g.to(e), e) + if ctx.cw_stop.contains(&(e as u32)) { CROSSWALK } else { 0.0 }, 0.45 * l) } else { 0.0 };
        mouth_s[2 * e] = s0 as f32; mouth_s[2 * e + 1] = (l - s1) as f32;
    }
    for &(e, sx, st) in mouth_adjust {
        let e = e as usize; let l = g.length[e] as f64;
        mouth_s[2 * e] = min(l, max(mouth_s[2 * e] as f64, sx)) as f32;
        mouth_s[2 * e + 1] = min(l, max(mouth_s[2 * e] as f64, min(mouth_s[2 * e + 1] as f64, st))) as f32;
    }
    // ---- connectors
    let mut keys: Vec<(u32, u32, u32, u32, u32, f64)> = Vec::new();
    for n in 0..n_n {
        if !ctx.is_j[n] { continue; }
        let (ins, outs) = (g.in_edges(n), g.out_edges(n));
        for &a in ins { for &b in outs {
            let (a, b) = (a as usize, b as usize);
            if b as i32 == g.twin[a] && ins.len() + outs.len() > 2 { continue; }
            if ctx.ht.get(g, a, b).strict { continue; }
            let (na, nb) = (g.lanes[a] as usize, g.lanes[b] as usize); let th = turn_angle(g, a, b);
            for la in 0..na {
                let lb = min(nb as f64 - 1.0, (((la as f64 + 0.5) * nb as f64) / na as f64).floor()) as u32;
                keys.push((n as u32, a as u32, la as u32, b as u32, lb, th));
            }
        } }
    }
    let ms = &mouth_s;
    let mut specs: Vec<Spec> = keys.par_iter().map(|&(n, a, la, b, lb, th)| {
        let cn = connector(g, lg, a as usize, la as usize, b as usize, lb as usize, ms[2 * a as usize + 1] as f64, ms[2 * b as usize] as f64, th, R_MIN, f64::NEG_INFINITY, f64::INFINITY);
        Spec { n, a, la, b, lb, th, cn }
    }).collect();
    drop(keys);
    let mut cap_out = vec![f64::INFINITY; e_n]; let mut floor_in = vec![f64::NEG_INFINITY; e_n];
    let mut split = vec![false; e_n];
    for _pass in 0..6 {
        let mut max_out_e = vec![-1f64; e_n]; let mut min_in_e = vec![f64::INFINITY; e_n];
        for sp in &specs { let (a, b) = (sp.a as usize, sp.b as usize); max_out_e[b] = max(max_out_e[b], sp.cn.s_out); min_in_e[a] = min(min_in_e[a], sp.cn.s_in); }
        let mut changed = 0;
        for e in 0..e_n {
            if max_out_e[e] >= 0.0 && min_in_e[e] < f64::INFINITY && max_out_e[e] > min_in_e[e] + 1e-6 {
                let mid = max(mouth_s[2 * e] as f64, min(mouth_s[2 * e + 1] as f64, (min(max_out_e[e], cap_out[e]) + max(min_in_e[e], floor_in[e])) / 2.0));
                cap_out[e] = min(cap_out[e], mid); floor_in[e] = max(floor_in[e], min(mid, cap_out[e]));
                split[e] = true; changed += 1;
            }
        }
        if changed == 0 { break; }
        let (co, fi) = (&cap_out, &floor_in);
        specs.par_iter_mut().for_each(|sp| {
            let (a, b) = (sp.a as usize, sp.b as usize);
            if sp.cn.s_out > co[b] + 1e-6 || sp.cn.s_in < fi[a] - 1e-6 {
                sp.cn = connector(g, lg, a, sp.la as usize, b, sp.lb as usize, ms[2 * a + 1] as f64, ms[2 * b] as f64, sp.th, R_MIN, fi[a], co[b]);
            }
        });
    }
    let chains_split = split.iter().filter(|&&s| s).count();
    // ---- movements + conflicts per junction (parallel, merged in node order)
    let jnodes: Vec<u32> = (0..n_n as u32).filter(|&n| ctx.is_j[n as usize]).collect();
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(jnodes.len());
    let mut si = 0;
    for &n in &jnodes { let s0 = si; while si < specs.len() && specs[si].n == n { si += 1; } ranges.push((s0, si)); }
    assert_eq!(si, specs.len());
    let locals: Vec<JLocal> = jnodes.par_iter().zip(ranges.par_iter()).map(|(&n, &(s0, s1))| junction_local(ctx, &mouth_s, n as usize, &specs[s0..s1])).collect();
    drop(specs);
    let nj = jnodes.len();
    let (mut j_first_m, mut j_count_m, mut j_first_c, mut j_count_c, mut j_first_h, mut j_count_h) =
        (Vec::with_capacity(nj), Vec::with_capacity(nj), Vec::with_capacity(nj), Vec::with_capacity(nj), Vec::with_capacity(nj), Vec::with_capacity(nj));
    let tot = |f: &dyn Fn(&JLocal) -> usize| locals.iter().map(f).sum::<usize>();
    let (mut m_info, mut m_f, mut pts, mut conf, mut conf_s, mut hull) = (Vec::with_capacity(tot(&|l| l.m_info.len())), Vec::with_capacity(tot(&|l| l.m_f.len())),
        Vec::with_capacity(tot(&|l| l.pts32.len())), Vec::with_capacity(tot(&|l| l.conf.len())), Vec::with_capacity(tot(&|l| l.conf_s.len())), Vec::with_capacity(tot(&|l| l.hull.len())));
    let mut too_tight = 0;
    for l in &locals {
        let (m0, p0, c0, h0) = ((m_info.len() / 7) as u32, (pts.len() / 4) as u32, (conf.len() / 3) as u32, (hull.len() / 2) as u32);
        for (k, &v) in l.m_info.iter().enumerate() { m_info.push(if k % 7 == 4 { v + p0 } else { v }); }
        m_f.extend_from_slice(&l.m_f);
        pts.extend_from_slice(&l.pts32);
        for (k, &v) in l.conf.iter().enumerate() { conf.push(if k % 3 == 2 { v } else { v + m0 }); }
        conf_s.extend_from_slice(&l.conf_s);
        hull.extend_from_slice(&l.hull);
        j_first_m.push(m0); j_count_m.push((l.m_info.len() / 7) as u32); j_first_c.push(c0); j_count_c.push((l.conf.len() / 3) as u32);
        j_first_h.push(h0); j_count_h.push((l.hull.len() / 2) as u32);
        too_tight += l.too_tight;
    }
    drop(locals);
    // ---- ramp gores
    let gores: Vec<Vec<(u32, f32, bool)>> = (0..n_n).into_par_iter().map(|n| {
        let (ins, outs) = (g.in_edges(n), g.out_edges(n));
        if ins.len() < 2 || outs.len() != 1 { return Vec::new(); }
        let b = outs[0] as usize;
        ins.iter().filter_map(|&a| gore_of(ctx, n, a as usize, b).map(|(v, fb)| (a, v, fb))).collect()
    }).collect();
    let mut gore_s = vec![-1f32; e_n];
    for l in gores { for (a, v, _) in l { gore_s[a as usize] = v; } }
    (Built { mouth_s, gore_s, j_node: jnodes, j_first_m, j_count_m, j_first_c, j_count_c, j_first_h, j_count_h, m_info, m_f, pts, conf, conf_s, hull },
     too_tight, chains_split)
}

#[derive(Clone, Copy, Debug)]
struct Hit { e: u32, s1: f64, s2: f64, n: u32 }

/// `connectorEdgeCrossings(J)`: connector paths over another edge's lanes outside its mouths (JS Map order).
fn connector_edge_crossings(g: &LaneGraph, lg: &LaneGeometry, j: &Built) -> Vec<Hit> {
    let p = &j.pts; let mi = &j.m_info;
    let m_n = mi.len() / 7; let np = p.len() / 4; let e_n = g.edge_count;
    let mut owner = vec![0u32; np];
    for m in 0..m_n { let (f, n) = (mi[7 * m + 4] as usize, mi[7 * m + 5] as usize); for k in 0..n { owner[f + k] = m as u32; } }
    let (mut x0, mut z0, mut x1, mut z1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for q in 0..np { let (x, z) = (p[4 * q] as f64, p[4 * q + 2] as f64); if x < x0 { x0 = x; } if x > x1 { x1 = x; } if z < z0 { z0 = z; } if z > z1 { z1 = z; } }
    if np == 0 { return Vec::new(); }
    const C: f64 = 20.0;
    let nx = ((x1 - x0) / C).ceil() as usize + 1; let nz = ((z1 - z0) / C).ceil() as usize + 1;
    let cell_of = |x: f64, z: f64| ((x - x0) / C).floor() as usize * nz + ((z - z0) / C).floor() as usize;
    let mut start = vec![0u32; nx * nz + 1];
    for q in 0..np { start[cell_of(p[4 * q] as f64, p[4 * q + 2] as f64) + 1] += 1; }
    for c in 0..nx * nz { start[c + 1] += start[c]; }
    let mut fill = start[..nx * nz].to_vec(); let mut idx = vec![0u32; np];
    for q in 0..np { let c = cell_of(p[4 * q] as f64, p[4 * q + 2] as f64); idx[fill[c] as usize] = q as u32; fill[c] += 1; }
    drop(fill);
    let per: Vec<Vec<Hit>> = (0..e_n).into_par_iter().map(|e| {
        let mut hits: Vec<Hit> = Vec::new(); let mut at: HashMap<u32, usize> = HashMap::new();
        let (s_stop, s_exit) = (j.mouth_s[2 * e + 1] as f64, j.mouth_s[2 * e] as f64);
        let mut s = s_exit;
        while s <= s_stop {
            let b = lg.bounds(e, s); let mid = (b.lane_right + b.lane_left) / 2.0; let hw = (b.lane_right - b.lane_left) / 2.0;
            let q = lg.offset_point(e, s, mid, 0.0);
            if !(q.x < x0 - C || q.x > x1 + C || q.z < z0 - C || q.z > z1 + C) {
                let (cx, cz) = (((q.x - x0) / C).floor() as i64, ((q.z - z0) / C).floor() as i64);
                for i in -1..=1i64 { for jj in -1..=1i64 {
                    let (xx, zz) = (cx + i, cz + jj);
                    if xx < 0 || zz < 0 || xx >= nx as i64 || zz >= nz as i64 { continue; }
                    let c = xx as usize * nz + zz as usize;
                    for k in start[c]..start[c + 1] {
                        let pp = idx[k as usize] as usize; let m = owner[pp] as usize;
                        if mi[7 * m] as usize == e || mi[7 * m + 2] as usize == e || mi[7 * m] == mi[7 * m + 2] { continue; }
                        let (dx, dz) = (p[4 * pp] as f64 - q.x, p[4 * pp + 2] as f64 - q.z);
                        if (dx * q.hx + dz * q.hz).abs() > 0.9 || (p[4 * pp + 1] as f64 - q.y).abs() > 3.0 { continue; }
                        if (dx * -q.hz + dz * q.hx).abs() > hw - 0.5 { continue; }
                        match at.get(&(m as u32)) {
                            Some(&h) => { let h = &mut hits[h]; h.n += 1; h.s1 = min(h.s1, s); h.s2 = max(h.s2, s); }
                            None => { at.insert(m as u32, hits.len()); hits.push(Hit { e: e as u32, s1: s, s2: s, n: 1 }); }
                        }
                    }
                } }
            }
            s += 1.5;
        }
        hits.retain(|h| h.n >= 2);
        hits
    }).collect();
    per.into_iter().flatten().collect()
}

/// `pairConflict(J, i, j)` on the f32 arrays.
pub(crate) fn pair_conflict(pts: &[f32], mi: &[u32], i: usize, j: usize) -> Option<(u32, f64, f64)> {
    let p = |k: usize| pts[k] as f64;
    let (fi, ci, fj, cj) = (mi[7 * i + 4] as usize, mi[7 * i + 5] as usize, mi[7 * j + 4] as usize, mi[7 * j + 5] as usize);
    if mi[7 * i] == mi[7 * j] && mi[7 * i + 1] == mi[7 * j + 1] { return None; }
    let (exi, exj) = (4 * (fi + ci - 1), 4 * (fj + cj - 1));
    let same_out = mi[7 * i + 2] == mi[7 * j + 2] && mi[7 * i + 3] == mi[7 * j + 3];
    if same_out || (hypot(p(exi) - p(exj), p(exi + 2) - p(exj + 2)) < CONFLICT_SEP && (p(exi + 1) - p(exj + 1)).abs() < 3.0) { return Some((1, p(exi + 3), p(exj + 3))); }
    let mut best: Option<(f64, f64, f64)> = None;
    for a in 0..ci.saturating_sub(1) {
        let (ax, az, bx, bz, ay) = (p(4 * (fi + a)), p(4 * (fi + a) + 2), p(4 * (fi + a + 1)), p(4 * (fi + a + 1) + 2), p(4 * (fi + a) + 1));
        for q in 0..cj.saturating_sub(1) {
            let (cx, cz, dx, dz) = (p(4 * (fj + q)), p(4 * (fj + q) + 2), p(4 * (fj + q + 1)), p(4 * (fj + q + 1) + 2));
            if (ay - p(4 * (fj + q) + 1)).abs() > 3.0 { continue; }
            if max(ax, bx) + CONFLICT_SEP < min(cx, dx) || max(cx, dx) + CONFLICT_SEP < min(ax, bx) || max(az, bz) + CONFLICT_SEP < min(cz, dz) || max(cz, dz) + CONFLICT_SEP < min(az, bz) { continue; }
            let u = seg_cross(ax, az, bx, bz, cx, cz, dx, dz);
            if u >= 0.0 { return Some((0, p(4 * (fi + a) + 3) + u * (p(4 * (fi + a + 1) + 3) - p(4 * (fi + a) + 3)), p(4 * (fj + q) + 3))); }
            let (d, uu, vv) = seg_seg_dist(ax, az, bx, bz, cx, cz, dx, dz);
            if d < CONFLICT_SEP && best.map_or(true, |b| d < b.0) {
                best = Some((d, p(4 * (fi + a) + 3) + uu * (p(4 * (fi + a + 1) + 3) - p(4 * (fi + a) + 3)), p(4 * (fj + q) + 3) + vv * (p(4 * (fj + q + 1) + 3) - p(4 * (fj + q) + 3))));
            }
        }
    }
    best.map(|b| (0, b.1, b.2))
}

/// `crossConflicts(J)`: conflicts between connectors of neighbouring junctions, appended to conf / confS.
fn cross_conflicts(g: &LaneGraph, j: &mut Built) -> usize {
    let (pts, mi) = (&j.pts, &j.m_info);
    let m_n = mi.len() / 7; let nj = j.j_node.len();
    let mut bb = vec![0f32; 4 * m_n];
    for m in 0..m_n {
        let (f, n) = (mi[7 * m + 4] as usize, mi[7 * m + 5] as usize);
        let (mut a, mut b, mut c, mut d) = (f32::INFINITY, f32::NEG_INFINITY, f32::INFINITY, f32::NEG_INFINITY);
        for k in 0..n { let (x, z) = (pts[4 * (f + k)], pts[4 * (f + k) + 2]); if x < a { a = x; } if x > b { b = x; } if z < c { c = z; } if z > d { d = z; } }
        bb[4 * m] = a; bb[4 * m + 1] = b; bb[4 * m + 2] = c; bb[4 * m + 3] = d;
    }
    const CELL: f64 = 120.0;
    let key = |x: f64, z: f64| ((x / CELL).floor() as i64, (z / CELL).floor() as i64);
    let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for jj in 0..nj { let n = j.j_node[jj] as usize; grid.entry(key(g.x(n), g.z(n))).or_default().push(jj as u32); }
    let mm = m_n as u64;
    let known: HashSet<u64> = (0..j.conf.len() / 3).map(|c| { let (a, b) = (j.conf[3 * c] as u64, j.conf[3 * c + 1] as u64); if a < b { a * mm + b } else { b * mm + a } }).collect();
    let bbf = |k: usize| bb[k] as f64;
    let adds: Vec<(Vec<u32>, Vec<f32>)> = (0..nj).into_par_iter().map(|jj| {
        let (mut add, mut add_s) = (Vec::new(), Vec::new());
        let n = j.j_node[jj] as usize; let (cx, cz) = key(g.x(n), g.z(n));
        for u in -1..=1 { for v in -1..=1 { let Some(l) = grid.get(&(cx + u, cz + v)) else { continue };
            for &k in l {
                let k = k as usize;
                if k <= jj { continue; }
                let n2 = j.j_node[k] as usize;
                if hypot(g.x(n2) - g.x(n), g.z(n2) - g.z(n)) > 120.0 { continue; }
                let (m0, m1) = (j.j_first_m[jj] as usize, (j.j_first_m[jj] + j.j_count_m[jj]) as usize);
                let (o0, o1) = (j.j_first_m[k] as usize, (j.j_first_m[k] + j.j_count_m[k]) as usize);
                for m in m0..m1 { for o in o0..o1 {
                    if bbf(4 * o) > bbf(4 * m + 1) + CONFLICT_SEP || bbf(4 * m) > bbf(4 * o + 1) + CONFLICT_SEP
                        || bbf(4 * o + 2) > bbf(4 * m + 3) + CONFLICT_SEP || bbf(4 * m + 2) > bbf(4 * o + 3) + CONFLICT_SEP { continue; }
                    if mi[7 * m + 2] == mi[7 * o] || mi[7 * o + 2] == mi[7 * m] { continue; }
                    let pk = if m < o { m as u64 * mm + o as u64 } else { o as u64 * mm + m as u64 };
                    if known.contains(&pk) { continue; }
                    if let Some((t, a, b)) = pair_conflict(pts, mi, m, o) { add.extend_from_slice(&[m as u32, o as u32, t]); add_s.extend_from_slice(&[a as f32, b as f32]); }
                } }
            }
        } }
        (add, add_s)
    }).collect();
    let mut added = 0;
    for (a, s) in adds { added += a.len() / 3; j.conf.extend_from_slice(&a); j.conf_s.extend_from_slice(&s); }
    added
}

/// Build statistics of [`build_junctions_full`].
#[derive(Clone, Debug, Default)]
pub struct JunctionStats {
    pub crossings_before: usize, pub crossings_after: usize, pub edges_adjusted: usize, pub rounds: usize, pub cross_conflicts: usize,
    pub too_tight: usize, pub chains_split: usize, pub builds: usize,
}

/// `buildJunctions(LG, legs)` with its crossingFix rounds (the JS default, 6 rounds).
pub fn build_junctions(lg: &LaneGeometry, g: &LaneGraph, legs: Option<&CrosswalkLegs>) -> JunctionArrays { build_junctions_full(lg, g, legs).0 }

pub fn build_junctions_full(lg: &LaneGeometry, g: &LaneGraph, legs: Option<&CrosswalkLegs>) -> (JunctionArrays, JunctionStats) {
    let ht = HopTable::new(g);
    let is_j: Vec<bool> = (0..g.node_count).into_par_iter().map(|n| {
        let (ins, outs) = (g.in_edges(n), g.out_edges(n));
        ins.iter().any(|&a| outs.iter().any(|&b| b as i32 != g.twin[a as usize] && !ht.get(g, a as usize, b as usize).strict))
    }).collect();
    let (cw_stop, cw_exit) = legs.map(|l| (l.crosswalk_stop.clone(), l.crosswalk_exit.clone())).unwrap_or_default();
    let ctx = Ctx { g, lg, ht, is_j, cw_stop, cw_exit };
    let mut st = JunctionStats::default();
    // adjust: insertion-ordered (edge, [sx, st])
    let mut adjust: Vec<(u32, f64, f64)> = Vec::new(); let mut adj_at: HashMap<u32, usize> = HashMap::new();
    let (j0, tt, cs) = build_once(&ctx, &adjust); st.builds = 1; st.too_tight = tt; st.chains_split = cs;
    let mut j = Some(j0);
    let mut first: Option<usize> = None;
    let mut rounds = 0;
    while rounds < 6 {
        let jr = j.as_ref().unwrap();
        let cr = connector_edge_crossings(g, lg, jr);
        if first.map_or(true, |f| f == 0) { first = Some(cr.len()); }
        let mut changed = 0;
        for c in &cr {
            let e = c.e as usize;
            let (s_stop, s_exit) = (jr.mouth_s[2 * e + 1] as f64, jr.mouth_s[2 * e] as f64);
            const PAD: f64 = 3.0; const MAXEXT: f64 = 30.0;
            let (can_start, can_end) = (ctx.is_j[g.from(e)], ctx.is_j[g.to(e)]);
            let ext_s = c.s2 + PAD - s_exit; let ext_e = s_stop - (c.s1 - PAD);
            let (mut sx, mut stv) = (f64::NEG_INFINITY, f64::INFINITY);
            if can_start && (!can_end || ext_s <= ext_e) && ext_s <= MAXEXT { sx = c.s2 + PAD; }
            else if can_end && ext_e <= MAXEXT { stv = c.s1 - PAD; }
            else { continue; }
            let cur = adj_at.get(&c.e).map(|&i| (adjust[i].1, adjust[i].2)).unwrap_or((f64::NEG_INFINITY, f64::INFINITY));
            let (nx, nt) = (max(cur.0, sx), min(cur.1, stv));
            if nx > cur.0 + 0.01 || nt < cur.1 - 0.01 {
                match adj_at.get(&c.e) { Some(&i) => { adjust[i].1 = nx; adjust[i].2 = nt; } None => { adj_at.insert(c.e, adjust.len()); adjust.push((c.e, nx, nt)); } }
                changed += 1;
            }
        }
        if changed == 0 { break; }
        drop(cr); drop(j.take());
        let (jn, tt, cs) = build_once(&ctx, &adjust); j = Some(jn); st.builds += 1; st.too_tight = tt; st.chains_split = cs;
        rounds += 1;
    }
    let mut j = j.unwrap();
    let added = cross_conflicts(g, &mut j);
    let left = connector_edge_crossings(g, lg, &j);
    st.crossings_before = first.unwrap_or(0); st.crossings_after = left.len(); st.edges_adjusted = adjust.len(); st.rounds = rounds; st.cross_conflicts = added;
    let crossing_fix = serde_json::json!({ "crossingsBefore": st.crossings_before, "crossingsAfter": st.crossings_after, "edgesAdjusted": st.edges_adjusted,
        "rounds": st.rounds, "crossConflicts": st.cross_conflicts });
    let Built { mouth_s, gore_s, j_node, j_first_m, j_count_m, j_first_c, j_count_c, j_first_h, j_count_h, m_info, m_f, pts, conf, conf_s, hull } = j;
    (JunctionArrays { mouth_s, gore_s, j_node, j_first_m, j_count_m, j_first_c, j_count_c, j_first_h, j_count_h, m_info, m_f, pts, conf, conf_s, hull,
        crossing_fix: Some(crossing_fix) }, st)
}

