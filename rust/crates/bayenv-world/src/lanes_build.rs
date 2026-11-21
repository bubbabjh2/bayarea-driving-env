//! `buildLaneGeometry` port (src/geom/lanes.js), with the defaults the committed data was built with
//! (opposing-carriageway pass on, taper slope limit on, medianMin 1.0). OWNER: geom port agent.
//! Reproduces the JS arrays bit for bit: same evaluation order, f32 rounding wherever the JS stores into a
//! Float32Array, JS Map / Set iteration order where it matters. Parallel only over independent per-edge /
//! per-sample work whose results are merged in the JS order.
use crate::hops::{turn_angle, HopTable};
use crate::jsmath::{self, hypot};
use crate::lanegraph::{LaneGraph, LANE_W};
use crate::lanes::{smooth, LaneGeometry};
use rayon::prelude::*;
use std::collections::HashMap;

const FILLET_MIN_TURN: f64 = 0.2;
const FILLET_STEP: f64 = 0.15;
const FILLET_MAX_CUT: f64 = 12.0;
const MIN_TAU: f64 = 60.0;
const TAU_PER_M: f64 = 30.0;
const TAPER_LIMIT: bool = true;

#[inline] fn max(a: f64, b: f64) -> f64 { jsmath::max(a, b) }
#[inline] fn min(a: f64, b: f64) -> f64 { jsmath::min(a, b) }
#[inline] fn or1(x: f64) -> f64 { if x == 0.0 || x.is_nan() { 1.0 } else { x } }
#[inline] fn sign(x: f64) -> f64 { if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { x } }

/// Build statistics (`LG.stats`), for checks against the JS dump.
#[derive(Clone, Debug, Default)]
pub struct LaneStats {
    pub strict_hops: usize, pub groups: usize, pub anchor_conflicts: usize, pub vertices: usize,
    pub median_shifted: usize, pub ramps_shifted: usize,
    pub opp_shifted: usize, pub opp_unresolved: usize, pub opp_same_dir: usize, pub opp_samples: usize, pub opp_left_hand: usize,
    pub opp_groups_raised: usize, pub opp_residual_samples: usize, pub opp_max_shift: f64,
}

pub fn build_lane_geometry(g: &LaneGraph) -> LaneGeometry { build_lane_geometry_stats(g).0 }

/// One edge's resampled polyline: (x, y, z, s) per vertex.
fn resample(g: &LaneGraph, e: usize) -> (Vec<f64>, Vec<f32>, Vec<f64>, Vec<f64>) {
    let (f, c) = (g.first[e] as usize, g.count[e] as usize);
    let nd = |k: usize| g.poly[f + k] as usize;
    let xx = |k: usize| g.x(nd(k)); let yy = |k: usize| g.y(nd(k)); let zz = |k: usize| g.z(nd(k));
    let mut seg_l = Vec::with_capacity(c);
    for k in 0..c.saturating_sub(1) { seg_l.push(hypot(xx(k + 1) - xx(k), zz(k + 1) - zz(k))); }
    let (mut px, mut py, mut pz) = (Vec::new(), Vec::new(), Vec::new());
    let mut push = |x: f64, y: f64, z: f64| { px.push(x); py.push(y); pz.push(z); };
    push(xx(0), yy(0), zz(0));
    for k in 1..c.saturating_sub(1) {
        let (ax, az, bx, bz) = (xx(k) - xx(k - 1), zz(k) - zz(k - 1), xx(k + 1) - xx(k), zz(k + 1) - zz(k));
        let (la, lb) = (seg_l[k - 1], seg_l[k]);
        if la < 1e-6 || lb < 1e-6 { continue; }
        let th = jsmath::acos(max(-1.0, min(1.0, (ax * bx + az * bz) / (la * lb))));
        if th < FILLET_MIN_TURN { push(xx(k), yy(k), zz(k)); continue; }
        let cut = min(min(FILLET_MAX_CUT, 0.45 * la), 0.45 * lb);
        let a = [xx(k) - (ax / la) * cut, yy(k) + (yy(k - 1) - yy(k)) * (cut / la), zz(k) - (az / la) * cut];
        let b = [xx(k) + (bx / lb) * cut, yy(k) + (yy(k + 1) - yy(k)) * (cut / lb), zz(k) + (bz / lb) * cut];
        let v = [xx(k), yy(k), zz(k)];
        let n = max(2.0, (th / FILLET_STEP).ceil()) as usize;
        for j in 0..=n {
            let t = j as f64 / n as f64;
            let (wa, wb, wd) = ((1.0 - t) * (1.0 - t), 2.0 * (1.0 - t) * t, t * t);
            push(wa * a[0] + wb * v[0] + wd * b[0], wa * a[1] + wb * v[1] + wd * b[1], wa * a[2] + wb * v[2] + wd * b[2]);
        }
    }
    push(xx(c - 1), yy(c - 1), zz(c - 1));
    let n = px.len();
    let mut ps = Vec::with_capacity(n);
    let mut acc = 0.0;
    ps.push(0.0);
    for i in 1..n { acc += hypot(px[i] - px[i - 1], pz[i] - pz[i - 1]); ps.push(acc); }
    let k = if acc > 0.0 { g.length[e] as f64 / acc } else { 1.0 };
    for s in ps.iter_mut() { *s *= k; }
    (px, py.into_iter().map(|y| y as f32).collect(), pz, ps)
}

/// Mutable right-edge state shared by the anchoring and the opposing pass.
struct St<'a> {
    g: &'a LaneGraph,
    we: Vec<f32>, rd: Vec<f32>, rend: Vec<f32>, wend: Vec<f32>, tau_min: Vec<f32>,
    /// adjacency per edge-end key: (other key, offset)
    adj: Vec<Vec<(u32, f64)>>,
    /// union-find root per key (final)
    root: Vec<u32>,
    /// groups in JS Map order: (root, keys)
    groups: Vec<(u32, Vec<u32>)>,
    anchor_of: HashMap<u32, u32>,
    group_raise: HashMap<u32, f64>,
    /// (capOf, floorOf) per edge once the opposing pass has set them (PC.groupCap / groupFloor)
    caps: Option<(Vec<f64>, Vec<f64>)>,
}

impl<'a> St<'a> {
    #[inline] fn two_way(&self, e: usize) -> bool { self.g.opp[e] > 0 }
    #[inline] fn has_adj(&self, k: usize) -> bool { !self.adj[k].is_empty() }
    fn tau_of(&self, e: usize, end: usize) -> f64 {
        let g = self.g; let l = g.length[e] as f64; let k = 2 * e + end;
        let d = max((self.rend[k] as f64 - self.rd[e] as f64).abs(), g.lanes[e] as f64 * (self.wend[k] as f64 - self.we[e] as f64).abs());
        if d < 1e-4 { return 0.0; }
        let link = g.cls[e] == 1 || g.cls[e] == 3;
        let t = if link { max(20.0, min(30.0, 6.0 * d)) } else { max(MIN_TAU, TAU_PER_M * d) };
        min(l / 2.0, max(t, self.tau_min[k] as f64))
    }
    /// `anchorGroups(count)`; returns the conflicts counted.
    fn anchor_groups(&mut self, count: bool) -> usize {
        let mut conflicts = 0;
        let groups = std::mem::take(&mut self.groups);
        let mut q: Vec<u32> = Vec::new();
        for (r, keys) in &groups {
            for &k in keys { self.rend[k as usize] = f32::NAN; }
            let anchor = self.anchor_of[r] as usize;
            self.rend[anchor] = self.rd[anchor >> 1];
            q.clear(); q.push(anchor as u32);
            while let Some(k) = q.pop() {
                let k = k as usize;
                for idx in 0..self.adj[k].len() {
                    let (k2, d) = self.adj[k][idx]; let k2 = k2 as usize;
                    let mut v = self.rend[k] as f64 + d;
                    if self.two_way(k2 >> 1) { v = self.g.lanes[k2 >> 1] as f64 * self.wend[k2] as f64; }
                    if self.rend[k2].is_nan() { self.rend[k2] = v as f32; q.push(k2 as u32); }
                    else if count && (self.rend[k2] as f64 - (self.rend[k] as f64 + d)).abs() > 1e-3 { conflicts += 1; }
                }
            }
            let mut gr = self.group_raise.get(r).copied().unwrap_or(0.0);
            if gr.is_nan() { gr = 0.0; }
            if gr != 0.0 {
                if let Some((cap, flo)) = &self.caps {
                    let (mut hi, mut lo) = (f64::INFINITY, f64::NEG_INFINITY);
                    for &k in keys {
                        let k = k as usize;
                        hi = min(hi, cap[k >> 1] - self.rend[k] as f64);
                        lo = max(lo, flo[k >> 1] - self.rend[k] as f64);
                    }
                    gr = min(max(gr, min(0.0, lo)), max(0.0, hi));
                    self.group_raise.insert(*r, gr);
                    for &k in keys { let k = k as usize; self.rend[k] = (self.rend[k] as f64 + gr) as f32; }
                }
            }
        }
        self.groups = groups;
        conflicts
    }
}

pub fn build_lane_geometry_stats(g: &LaneGraph) -> (LaneGeometry, LaneStats) {
    let e_n = g.edge_count;
    let lw = LANE_W;
    let mut stats = LaneStats::default();
    // ---------------------------------------------------------------- 1. resampled polylines
    let per: Vec<_> = (0..e_n).into_par_iter().map(|e| resample(g, e)).collect();
    let mut v_first = vec![0u32; e_n + 1];
    let total: usize = per.iter().map(|p| p.0.len()).sum();
    let (mut vx, mut vy, mut vz, mut vs) = (Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total), Vec::with_capacity(total));
    for (e, (x, y, z, s)) in per.into_iter().enumerate() {
        v_first[e] = vx.len() as u32;
        vx.extend_from_slice(&x); vy.extend_from_slice(&y); vz.extend_from_slice(&z); vs.extend_from_slice(&s);
    }
    v_first[e_n] = vx.len() as u32;
    let nv = vx.len();
    stats.vertices = nv;

    // ---------------------------------------------------------------- 2. strict groups at nodes
    let ht = HopTable::new(g);
    let mut parent: Vec<i32> = (0..2 * e_n as i32).collect();
    fn find(parent: &mut [i32], mut a: usize) -> usize {
        while parent[a] as usize != a { parent[a] = parent[parent[a] as usize]; a = parent[a] as usize; }
        a
    }
    let mut hops: Vec<(u32, u32, i32)> = Vec::new();
    for n in 0..g.node_count {
        let (ins, outs) = (g.in_edges(n), g.out_edges(n));
        if ins.is_empty() || outs.is_empty() { continue; }
        for &a in ins { for &b in outs {
            if b as i32 == g.twin[a as usize] { continue; }
            let h = ht.get(g, a as usize, b as usize);
            if !h.strict { continue; }
            hops.push((a, b, h.shift));
            let ra = find(&mut parent, 2 * a as usize + 1); let rb = find(&mut parent, 2 * b as usize);
            if ra != rb { parent[rb] = ra as i32; }
        } }
    }
    let nh = hops.len();
    stats.strict_hops = nh;
    let root: Vec<u32> = (0..2 * e_n).map(|k| find(&mut parent, k) as u32).collect();
    let seg_dir = |e: usize, end: bool| -> [f64; 2] {
        let (v0, v1) = (v_first[e] as usize, v_first[e + 1] as usize - 1);
        let (a, b) = if end { (v1 - 1, v1) } else { (v0, v0 + 1) };
        let (dx, dz) = (vx[b] - vx[a], vz[b] - vz[a]); let l = or1(hypot(dx, dz));
        [dx / l, dz / l]
    };
    let mut best_turn: HashMap<u32, f64> = HashMap::new();
    let mut frame: HashMap<u32, [f64; 2]> = HashMap::new();
    for &(a, b, _) in &hops {
        let (a, b) = (a as usize, b as usize); let r = root[2 * a + 1];
        let t = turn_angle(g, a, b).abs();
        let better = match best_turn.get(&r) { None => true, Some(&bt) => t < bt };
        if better {
            best_turn.insert(r, t);
            let (ua, ub) = (seg_dir(a, true), seg_dir(b, false));
            let (mut fx, mut fz) = (ua[0] + ub[0], ua[1] + ub[1]);
            let l = hypot(fx, fz);
            if l < 1e-6 { fx = ua[0]; fz = ua[1]; } else { fx /= l; fz /= l; }
            frame.insert(r, [fx, fz]);
        }
    }

    // ---------------------------------------------------------------- 3. right-edge anchoring
    let we: Vec<f32> = (0..e_n).map(|e| { let w = g.lane_w[e]; if w == 0.0 || w.is_nan() { LANE_W as f32 } else { w } }).collect();
    let mut rd: Vec<f32> = (0..e_n).map(|e| {
        let w = g.lanes[e] as f64 * we[e] as f64;
        (if g.opp[e] > 0 { w } else { w / 2.0 }) as f32
    }).collect();
    stats.median_shifted = median_clearance(g, &v_first, &vx, &vy, &vz, &mut rd, &we);
    stats.ramps_shifted = ramp_clearance(g, &v_first, &vx, &vy, &vz, &mut rd, &we);
    let mut groups: Vec<(u32, Vec<u32>)> = Vec::new();
    let mut gidx: HashMap<u32, usize> = HashMap::new();
    for &(a, b, _) in &hops {
        for key in [2 * a + 1, 2 * b] {
            let r = root[key as usize];
            let i = *gidx.entry(r).or_insert_with(|| { groups.push((r, Vec::new())); groups.len() - 1 });
            if !groups[i].1.contains(&key) { groups[i].1.push(key); }
        }
    }
    stats.groups = groups.len();
    let mut anchor_of = HashMap::new();
    let mut wend = vec![0f32; 2 * e_n];
    for e in 0..e_n { wend[2 * e] = we[e]; wend[2 * e + 1] = we[e]; }
    for (r, keys) in &groups {
        let (mut anchor, mut best) = (keys[0], -1i32);
        for &k in keys {
            let e = (k >> 1) as usize; let score = if g.opp[e] > 0 { 100 } else { 0 } + g.lanes[e] as i32;
            if score > best { best = score; anchor = k; }
        }
        anchor_of.insert(*r, anchor);
        for &k in keys { wend[k as usize] = we[(anchor >> 1) as usize]; }
    }
    let mut adj: Vec<Vec<(u32, f64)>> = vec![Vec::new(); 2 * e_n];
    for &(a, b, sh) in &hops {
        let wn = wend[2 * a as usize + 1] as f64;
        adj[2 * a as usize + 1].push((2 * b, -(sh as f64) * wn));
        adj[2 * b as usize].push((2 * a + 1, sh as f64 * wn));
    }
    let mut st = St { g, we, rd, rend: vec![f32::NAN; 2 * e_n], wend, tau_min: vec![0f32; 2 * e_n], adj, root, groups, anchor_of,
        group_raise: HashMap::new(), caps: None };
    stats.anchor_conflicts = st.anchor_groups(true);
    for k in 0..2 * e_n { if st.rend[k].is_nan() { st.rend[k] = st.rd[k >> 1]; } }
    opposing_clearance(&mut st, &v_first, &vx, &vy, &vz, &vs, &mut stats);
    let mut tau = vec![0f32; 2 * e_n];
    for e in 0..e_n { for end in 0..2 { tau[2 * e + end] = st.tau_of(e, end) as f32; } }

    // ---------------------------------------------------------------- 4. normals (mitres)
    let mut mx = vec![0f64; nv]; let mut mz = vec![0f64; nv];
    {
        let seg = |a: usize, b: usize| -> [f64; 2] { let (dx, dz) = (vx[b] - vx[a], vz[b] - vz[a]); let l = or1(hypot(dx, dz)); [dx / l, dz / l] };
        let chunks: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..e_n).into_par_iter().map(|e| {
            let (v0, v1) = (v_first[e] as usize, v_first[e + 1] as usize - 1);
            let (mut ox, mut oz) = (Vec::with_capacity(v1 + 1 - v0), Vec::with_capacity(v1 + 1 - v0));
            for i in v0..=v1 {
                let (mut ux, mut uz) = (0.0f64, 0.0f64);
                if i > v0 { let u = seg(i - 1, i); ux += u[0]; uz += u[1]; }
                if i < v1 { let u = seg(i, i + 1); ux += u[0]; uz += u[1]; }
                let l = or1(hypot(ux, uz)); ux /= l; uz /= l;
                let mut sc = 1.0;
                if i > v0 && i < v1 { let u = seg(i - 1, i); sc = 1.0 / max(0.5, u[0] * ux + u[1] * uz); }
                ox.push(-uz * sc); oz.push(ux * sc);
            }
            for end in 0..2 {
                let key = 2 * e + end; let r = st.root[key] as usize;
                if r == key && !st.has_adj(key) { continue; }
                let Some(f) = frame.get(&(r as u32)) else { continue };
                let i = if end == 1 { v1 } else { v0 } - v0;
                ox[i] = -f[1]; oz[i] = f[0];
            }
            (v0, ox, oz)
        }).collect();
        for (v0, ox, oz) in chunks { mx[v0..v0 + ox.len()].copy_from_slice(&ox); mz[v0..v0 + oz.len()].copy_from_slice(&oz); }
    }

    // ---------------------------------------------------------------- 5. lane ends / adds (tapers)
    let mut tap_r = vec![0u8; 2 * e_n]; let mut tap_l = vec![0u8; 2 * e_n];
    let mut outs_of: Vec<Vec<(u32, i32)>> = vec![Vec::new(); e_n];
    let mut ins_of: Vec<Vec<(u32, i32)>> = vec![Vec::new(); e_n];
    for &(a, b, sh) in &hops { outs_of[a as usize].push((b, sh)); ins_of[b as usize].push((a, sh)); }
    let runs = |n: usize, cont: &[u8]| -> Option<(u8, u8)> {
        let mut r = 0; while r < n && cont[r] == 0 { r += 1; }
        let mut t = 0; while t < n - r && cont[n - 1 - t] == 0 { t += 1; }
        if r < n { Some((r as u8, t as u8)) } else { None }
    };
    for a in 0..e_n {
        if outs_of[a].is_empty() { continue; }
        let n = g.lanes[a] as usize; let mut cont = vec![0u8; n];
        for &(b, sh) in &outs_of[a] { for l in 0..n as i32 { if l - sh >= 0 && l - sh < g.lanes[b as usize] as i32 { cont[l as usize] = 1; } } }
        if let Some((r, t)) = runs(n, &cont) { tap_r[2 * a + 1] = r; tap_l[2 * a + 1] = t; }
    }
    for b in 0..e_n {
        if ins_of[b].is_empty() { continue; }
        let n = g.lanes[b] as usize; let mut cont = vec![0u8; n];
        for &(a, sh) in &ins_of[b] { for l in 0..n as i32 { if l + sh >= 0 && l + sh < g.lanes[a as usize] as i32 { cont[l as usize] = 1; } } }
        if let Some((r, t)) = runs(n, &cont) { tap_r[2 * b] = r; tap_l[2 * b] = t; }
    }
    let mut x_r = vec![0u8; 2 * e_n]; let mut x_l = vec![0u8; 2 * e_n];
    let straightest = |lst: &[(u32, i32)], f: &dyn Fn(usize) -> f64| -> i64 {
        let (mut best, mut bt) = (-1i64, 9.0f64);
        for &(o, _) in lst { let t = f(o as usize).abs(); if t < bt { bt = t; best = o as i64; } }
        best
    };
    for a in 0..e_n {
        if outs_of[a].is_empty() || (tap_r[2 * a + 1] == 0 && tap_l[2 * a + 1] == 0) { continue; }
        let b = straightest(&outs_of[a], &|o| turn_angle(g, a, o));
        if b < 0 { continue; }
        let b = b as usize;
        x_r[2 * b] = x_r[2 * b].max(tap_r[2 * a + 1]); x_l[2 * b] = x_l[2 * b].max(tap_l[2 * a + 1]);
    }
    for b in 0..e_n {
        if ins_of[b].is_empty() || (tap_r[2 * b] == 0 && tap_l[2 * b] == 0) { continue; }
        let a = straightest(&ins_of[b], &|o| turn_angle(g, o, b));
        if a < 0 { continue; }
        let a = a as usize;
        x_r[2 * a + 1] = x_r[2 * a + 1].max(tap_r[2 * b]); x_l[2 * a + 1] = x_l[2 * a + 1].max(tap_l[2 * b]);
    }

    let lg = LaneGeometry {
        v_first: v_first.into(), vx: vx.into(), vy: vy.into(), vz: vz.into(), vs: vs.into(), mx: mx.into(), mz: mz.into(),
        rd: st.rd.into(), rend: st.rend.into(), tau: tau.into(), tap_r: tap_r.into(), tap_l: tap_l.into(), x_r: x_r.into(), x_l: x_l.into(),
        lw, lane_w: st.we.into(), lane_w_end: st.wend.into(), length: g.length.clone().into(), lanes: g.lanes.clone().into(),
    };
    (lg, stats)
}

#[inline] fn cell_key(x: f64, z: f64, cell: f64) -> (i64, i64) { ((x / cell).floor() as i64, (z / cell).floor() as i64) }

/// `medianClearance`: opposing one-way motorway / trunk carriageways shifted right of the midline.
fn median_clearance(g: &LaneGraph, v_first: &[u32], vx: &[f64], vy: &[f32], vz: &[f64], rd: &mut [f32], we: &[f32]) -> usize {
    let is_fw = |e: usize| (g.cls[e] == 0 || g.cls[e] == 2) && g.opp[e] == 0;
    #[derive(Clone, Copy)] struct Rec { e: u32, x: f64, z: f64, y: f64, hx: f64, hz: f64 }
    let cell = 60.0;
    let mut samples: Vec<Rec> = Vec::new();
    let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for e in 0..g.edge_count {
        if !is_fw(e) { continue; }
        let (v0, v1) = (v_first[e] as usize, v_first[e + 1] as usize - 1);
        for i in v0..v1 {
            let (dx, dz) = (vx[i + 1] - vx[i], vz[i + 1] - vz[i]); let l = hypot(dx, dz);
            if l < 1e-6 { continue; }
            let n = max(1.0, (l / 15.0).ceil()) as usize;
            for k in 0..n {
                let t = (k as f64 + 0.5) / n as f64;
                let rec = Rec { e: e as u32, x: vx[i] + dx * t, z: vz[i] + dz * t, y: vy[i] as f64 + (vy[i + 1] as f64 - vy[i] as f64) * t, hx: dx / l, hz: dz / l };
                grid.entry(cell_key(rec.x, rec.z, cell)).or_default().push(samples.len() as u32);
                samples.push(rec);
            }
        }
    }
    let res: Vec<Option<f64>> = samples.par_iter().map(|a| {
        let mut bd = 50.0; let mut best: Option<&Rec> = None;
        let (cx, cz) = cell_key(a.x, a.z, cell);
        for i in -1..=1 { for j in -1..=1 { if let Some(l) = grid.get(&(cx + i, cz + j)) { for &bi in l {
            let b = &samples[bi as usize];
            if b.e == a.e || a.hx * b.hx + a.hz * b.hz > -0.94 || (a.y - b.y).abs() > 3.0 { continue; }
            let d = hypot(b.x - a.x, b.z - a.z);
            if d < bd { bd = d; best = Some(b); }
        } } } }
        let best = best?;
        let gap = (best.x - a.x) * a.hz + (best.z - a.z) * -a.hx;
        if gap <= 0.0 { return None; }
        let w = g.lanes[a.e as usize] as f64 * we[a.e as usize] as f64;
        Some(w - max(0.0, gap / 2.0 - 0.3))
    }).collect();
    let mut need = vec![-1f32; g.edge_count];
    for (a, r) in samples.iter().zip(res) { if let Some(r) = r { if r > need[a.e as usize] as f64 { need[a.e as usize] = r as f32; } } }
    let mut shifted = 0;
    for e in 0..g.edge_count {
        if need[e] as f64 > rd[e] as f64 + 1e-3 { rd[e] = min(need[e] as f64, (g.lanes[e] as f64 + 2.0) * we[e] as f64) as f32; shifted += 1; }
    }
    shifted
}

/// Samples every <= `step` m along the resampled polyline (segment midpoints of n parts): (x, y, z, hx, hz, d).
fn edge_samples(v_first: &[u32], vx: &[f64], vy: &[f32], vz: &[f64], e: usize, step: f64, mut f: impl FnMut(f64, f64, f64, f64, f64, f64)) {
    let (v0, v1) = (v_first[e] as usize, v_first[e + 1] as usize - 1);
    let mut acc = 0.0;
    for i in v0..v1 {
        let (dx, dz) = (vx[i + 1] - vx[i], vz[i + 1] - vz[i]); let l = hypot(dx, dz);
        if l < 1e-6 { continue; }
        let n = max(1.0, (l / step).ceil()) as usize;
        for k in 0..n {
            let t = (k as f64 + 0.5) / n as f64;
            f(vx[i] + dx * t, vy[i] as f64 + (vy[i + 1] as f64 - vy[i] as f64) * t, vz[i] + dz * t, dx / l, dz / l, acc + l * t);
        }
        acc += l;
    }
}

/// `rampClearance`: ramps drawn alongside a same-direction freeway carriageway shifted clear of it.
fn ramp_clearance(g: &LaneGraph, v_first: &[u32], vx: &[f64], vy: &[f32], vz: &[f64], rd: &mut [f32], we: &[f32]) -> usize {
    let is_fw = |e: usize| (g.cls[e] == 0 || g.cls[e] == 2) && g.opp[e] == 0;
    let is_link = |e: usize| (g.cls[e] == 1 || g.cls[e] == 3) && g.opp[e] == 0;
    #[derive(Clone, Copy)] struct Rec { e: u32, x: f64, y: f64, z: f64, hx: f64, hz: f64, hw: f64 }
    let cell = 40.0;
    let mut recs: Vec<Rec> = Vec::new();
    let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for e in 0..g.edge_count {
        if !is_fw(e) { continue; }
        let w = g.lanes[e] as f64 * we[e] as f64; let c = rd[e] as f64 - w / 2.0;
        edge_samples(v_first, vx, vy, vz, e, 8.0, |x, y, z, hx, hz, _| {
            let rec = Rec { e: e as u32, x: x - hz * c, y, z: z + hx * c, hx, hz, hw: w / 2.0 };
            grid.entry(cell_key(rec.x, rec.z, cell)).or_default().push(recs.len() as u32);
            recs.push(rec);
        });
    }
    const GORE: f64 = 30.0;
    let rd_ro: &[f32] = rd;
    let shifts: Vec<f64> = (0..g.edge_count).into_par_iter().map(|e| {
        if !is_link(e) { return 0.0; }
        let w = g.lanes[e] as f64 * we[e] as f64; let c = rd_ro[e] as f64 - w / 2.0; let le = g.length[e] as f64;
        let (mut right, mut left, mut j_right, mut j_left) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        edge_samples(v_first, vx, vy, vz, e, 8.0, |x, y, z, hx, hz, d| {
            let (px, pz) = (x - hz * c, z + hx * c);
            let (cx, cz) = cell_key(px, pz, cell);
            for i in -1..=1 { for j in -1..=1 { if let Some(l) = grid.get(&(cx + i, cz + j)) { for &ri in l {
                let r = &recs[ri as usize];
                if r.hx * hx + r.hz * hz < 0.8 || (r.y - y).abs() > 4.5 { continue; }
                let re = r.e as usize;
                let at_end = g.to(e) == g.to(re) || g.to(e) == g.from(re);
                let at_start = g.from(e) == g.from(re) || g.from(e) == g.to(re);
                if (at_end && le - d < GORE) || (at_start && d < GORE) { continue; }
                let lon = (px - r.x) * r.hx + (pz - r.z) * r.hz;
                if lon.abs() > 6.0 { continue; }
                let lat = (px - r.x) * -r.hz + (pz - r.z) * r.hx;
                let over = w / 2.0 + r.hw + 0.3 - lat.abs();
                if over <= 0.0 { continue; }
                if lat >= 0.0 { right = max(right, over); } else { left = max(left, over); }
                if at_end || at_start {
                    if lat >= 0.0 { j_right = max(j_right, over); } else { j_left = max(j_left, over); }
                }
            } } } }
        });
        let mut sh = 0.0;
        if j_right > 0.0 && j_left == 0.0 { sh = max(j_right, right); }
        else if j_left > 0.0 && j_right == 0.0 { sh = -max(j_left, left); }
        else if right > 0.0 && left == 0.0 { sh = right; }
        else if left > 0.0 && right == 0.0 { sh = -left; }
        sign(sh) * min(sh.abs(), 4.0 * we[e] as f64)
    }).collect();
    let mut shifted = 0;
    for e in 0..g.edge_count { let sh = shifts[e]; if sh.abs() > 1e-3 { rd[e] = (rd[e] as f64 + sh) as f32; shifted += 1; } }
    shifted
}

/// `opposingClearance` (docs/carriageway-fix.md): inner pavement edges of opposite one-way carriageways kept
/// `medianMin` apart, through Rd / Rend / tauMin.
fn opposing_clearance(st: &mut St, v_first: &[u32], vx: &[f64], vy: &[f32], vz: &[f64], vs: &[f64], stats: &mut LaneStats) {
    let g = st.g; let e_n = g.edge_count;
    const CLR: f64 = 1.0; const NOSE: f64 = 30.0; const STEP: f64 = 3.0; const CELL: f64 = 16.0; const TMIN: f64 = 40.0;
    const SLOPE_MAX: f64 = 0.04; const SAME_W: f64 = 0.5;
    let one_way = |e: usize| g.opp[e] == 0 && g.twin[e] < 0;
    // ---- samples
    struct Smp { e: Vec<u32>, s: Vec<f64>, x: Vec<f64>, y: Vec<f64>, z: Vec<f64>, hx: Vec<f64>, hz: Vec<f64> }
    let per: Vec<Vec<[f64; 6]>> = (0..e_n).into_par_iter().map(|e| {
        let mut out = Vec::new();
        if !one_way(e) { return out; }
        let (v0, v1) = (v_first[e] as usize, v_first[e + 1] as usize - 1);
        for i in v0..v1 {
            let (dx, dz) = (vx[i + 1] - vx[i], vz[i + 1] - vz[i]); let l = hypot(dx, dz);
            if l < 1e-6 { continue; }
            let n = max(1.0, (l / STEP).ceil()) as usize; let last = i == v1 - 1;
            for k in 0..=(if last { n } else { n - 1 }) {
                let t = k as f64 / n as f64;
                out.push([vs[i] + (vs[i + 1] - vs[i]) * t, vx[i] + dx * t, vy[i] as f64 + (vy[i + 1] as f64 - vy[i] as f64) * t, vz[i] + dz * t, dx / l, dz / l]);
            }
        }
        out
    }).collect();
    let mut s_first = vec![0usize; e_n + 1];
    let ns_tot: usize = per.iter().map(|p| p.len()).sum();
    let mut sm_ = Smp { e: Vec::with_capacity(ns_tot), s: Vec::with_capacity(ns_tot), x: Vec::with_capacity(ns_tot), y: Vec::with_capacity(ns_tot),
        z: Vec::with_capacity(ns_tot), hx: Vec::with_capacity(ns_tot), hz: Vec::with_capacity(ns_tot) };
    for (e, p) in per.into_iter().enumerate() {
        s_first[e] = sm_.e.len();
        for r in p { sm_.e.push(e as u32); sm_.s.push(r[0]); sm_.x.push(r[1]); sm_.y.push(r[2]); sm_.z.push(r[3]); sm_.hx.push(r[4]); sm_.hz.push(r[5]); }
    }
    s_first[e_n] = sm_.e.len();
    let ns = sm_.e.len();
    stats.opp_samples = ns;
    let (se, ss, sx, sy, sz, shx, shz) = (&sm_.e, &sm_.s, &sm_.x, &sm_.y, &sm_.z, &sm_.hx, &sm_.hz);
    let gkey = |x: f64, z: f64| -> i64 { (x / CELL).floor() as i64 * 100000 + (z / CELL).floor() as i64 };
    let mut grid: HashMap<i64, Vec<u32>> = HashMap::new();
    for i in 0..ns { grid.entry(gkey(sx[i], sz[i])).or_default().push(i as u32); }
    let has_adj = |k: usize| !st.adj[k].is_empty();
    let d_strict = |i: usize| -> f64 {
        let e = se[i] as usize; let l = g.length[e] as f64;
        min(if has_adj(2 * e) { ss[i] } else { f64::INFINITY }, if has_adj(2 * e + 1) { l - ss[i] } else { f64::INFINITY })
    };
    // ---- neighbours with opposite headings
    let nb: Vec<Vec<(u32, f64)>> = (0..ns).into_par_iter().map(|i| {
        let mut out = Vec::new();
        let (a, hx, hz) = (se[i], shx[i], shz[i]);
        let (cx, cz) = ((sx[i] / CELL).floor() as i64, (sz[i] / CELL).floor() as i64);
        for u in -1..=1 { for w in -1..=1 { if let Some(l) = grid.get(&((cx + u) * 100000 + cz + w)) { for &j in l {
            let j = j as usize; let b = se[j];
            if b == a || hx * shx[j] + hz * shz[j] > -0.8 || (sy[i] - sy[j]).abs() > 3.0 { continue; }
            let (dx, dz) = (sx[j] - sx[i], sz[j] - sz[i]);
            if (dx * hx + dz * hz).abs() > STEP / 2.0 + 0.1 { continue; }
            let gg = dx * hz - dz * hx;
            if gg.abs() > 30.0 { continue; }
            out.push((j as u32, gg));
        } } } }
        out
    }).collect();
    let pk = |a: u32, b: u32| -> u64 { a.min(b) as u64 * 4194304 + a.max(b) as u64 };
    let mut pair_g: HashMap<u64, (f64, u32)> = HashMap::new();
    for i in 0..ns { for &(j, gg) in &nb[i] {
        let p = pair_g.entry(pk(se[i], se[j as usize])).or_insert((0.0, 0));
        if p.1 == 0 { *p = (gg, 1); } else { p.0 += gg; p.1 += 1; }
    } }
    let mut lb = vec![f32::NEG_INFINITY; ns]; let mut ubo = vec![f32::INFINITY; ns]; let mut ubs = vec![f32::INFINITY; ns];
    let mut lh_pairs: std::collections::HashSet<u64> = Default::default();
    for i in 0..ns { for &(j, gg) in &nb[i] {
        let j = j as usize; let (a, b) = (se[i], se[j]); let p = pair_g[&pk(a, b)];
        let c = CLR * min(min(1.0, d_strict(i) / NOSE), d_strict(j) / NOSE);
        if p.0 / (p.1 as f64) < -1.0 {
            // drawn left-hand: both move to their own left
            lh_pairs.insert(pk(a, b));
            let ub = (-gg - c) / 2.0;
            if ub < ubo[i] as f64 { ubo[i] = ub as f32; }
            continue;
        }
        let need = g.lanes[a as usize] as f64 * st.we[a as usize] as f64 + (c - gg) / 2.0;
        if need > lb[i] as f64 { lb[i] = need as f32; }
    } }
    drop(nb);
    stats.opp_left_hand = lh_pairs.len();
    // ---- same-direction freeway carriageways on the right
    let r0 = |e: usize, s: f64| -> f64 {
        let l = g.length[e] as f64; let rd = st.rd[e] as f64; let (ts, te) = (st.tau_of(e, 0), st.tau_of(e, 1));
        let mut r = rd;
        if ts > 0.0 { r += (st.rend[2 * e] as f64 - rd) * smooth(1.0 - s / ts); }
        if te > 0.0 { r += (st.rend[2 * e + 1] as f64 - rd) * smooth(1.0 - (l - s) / te); }
        r
    };
    let same: Vec<(f32, usize)> = (0..ns).into_par_iter().map(|i| {
        let mut ub_i = f32::INFINITY; let mut cnt = 0;
        if !(lb[i] > f32::NEG_INFINITY) && !(ubo[i] < f32::INFINITY) { return (ub_i, 0); }
        let a = se[i] as usize; let (hx, hz) = (shx[i], shz[i]); let la = g.length[a] as f64;
        let (cx, cz) = ((sx[i] / CELL).floor() as i64, (sz[i] / CELL).floor() as i64);
        for u in -1..=1 { for w in -1..=1 { if let Some(l) = grid.get(&((cx + u) * 100000 + cz + w)) { for &j in l {
            let j = j as usize; let b = se[j] as usize;
            if b == a || !(g.cls[b] <= 3) || hx * shx[j] + hz * shz[j] < 0.8 || (sy[i] - sy[j]).abs() > 4.5 { continue; }
            let (dx, dz) = (sx[j] - sx[i], sz[j] - sz[i]);
            if (dx * hx + dz * hz).abs() > STEP / 2.0 + 0.1 { continue; }
            let gr = -(dx * hz - dz * hx);
            if gr <= 0.0 || gr > 30.0 { continue; }
            let at_end = g.to(a) == g.to(b) || g.to(a) == g.from(b);
            let at_start = g.from(a) == g.from(b) || g.from(a) == g.to(b);
            if (at_end && la - ss[i] < 30.0) || (at_start && ss[i] < 30.0) { continue; }
            let ub = gr + r0(b, ss[j]) - g.lanes[b] as f64 * st.we[b] as f64 - 0.3;
            if ub < ub_i as f64 { ub_i = ub as f32; cnt += 1; }
        } } } }
        (ub_i, cnt)
    }).collect();
    let mut same_dir = 0;
    for (i, (u, c)) in same.into_iter().enumerate() { ubs[i] = u; same_dir += c; }
    stats.opp_same_dir = same_dir;
    drop(grid);

    // ---- fit the right-edge parameters of every edge with a violation
    let rd0: Vec<f32> = st.rd.clone(); let rend0: Vec<f32> = st.rend.clone();
    let cap: Vec<f64> = (0..e_n).map(|e| max(max(max(rd0[e] as f64, rend0[2 * e] as f64), rend0[2 * e + 1] as f64), g.lanes[e] as f64 * st.we[e] as f64 + CLR / 2.0 + 1.5)).collect();
    let flo: Vec<f64> = (0..e_n).map(|e| min(min(min(rd0[e] as f64, rend0[2 * e] as f64), rend0[2 * e + 1] as f64), -(CLR / 2.0 + 1.5))).collect();
    st.caps = Some((cap.clone(), flo.clone()));
    let mut set = vec![0u8; 2 * e_n];
    let mut changed = vec![false; e_n];
    let viol_of = |i: usize, r: f64| -> f64 { max(max(lb[i] as f64 - r, r - ubo[i] as f64), SAME_W * (r - ubs[i] as f64)) };
    let mut unresolved = 0;
    for _pass in 0..4 {
        let mut bad = 0;
        for e in 0..e_n {
            if s_first[e + 1] > s_first[e] && !fit_edge(st, &mut set, &mut changed, e, &s_first, ss, &lb, &ubo, &viol_of, &cap, &flo, TMIN, SLOPE_MAX) { bad += 1; }
        }
        st.anchor_groups(false);
        for k in 0..2 * e_n { if !has_adj_st(st, k) && set[k] == 0 { st.rend[k] = st.rd[k >> 1]; } }
        unresolved = bad;
    }
    stats.opp_unresolved = unresolved;
    // remaining violations (stats only)
    let tau_p = |st: &St, e: usize, end: usize| max(st.tau_of(e, end), min(g.length[e] as f64 / 2.0, st.tau_min[2 * e + end] as f64));
    let r_of = |st: &St, e: usize, s: f64| -> f64 {
        let l = g.length[e] as f64; let rd = st.rd[e] as f64; let (ts, te) = (tau_p(st, e, 0), tau_p(st, e, 1));
        let mut r = rd;
        if ts > 0.0 { r += (st.rend[2 * e] as f64 - rd) * smooth(1.0 - s / ts); }
        if te > 0.0 { r += (st.rend[2 * e + 1] as f64 - rd) * smooth(1.0 - (l - s) / te); }
        r
    };
    let mut viol = 0;
    for i in 0..ns { let r = r_of(st, se[i] as usize, ss[i]); if max(lb[i] as f64 - r, r - ubo[i] as f64) > 0.05 { viol += 1; } }
    stats.opp_residual_samples = viol;
    let mut moved = 0; let mut max_shift = 0.0f64;
    for e in 0..e_n { if !changed[e] { continue; }
        max_shift = max(max(max(max_shift, (st.rd[e] as f64 - rd0[e] as f64).abs()), (st.rend[2 * e] as f64 - rend0[2 * e] as f64).abs()), (st.rend[2 * e + 1] as f64 - rend0[2 * e + 1] as f64).abs());
        moved += 1;
    }
    stats.opp_shifted = moved; stats.opp_max_shift = max_shift; stats.opp_groups_raised = st.group_raise.len();
}

#[inline] fn has_adj_st(st: &St, k: usize) -> bool { !st.adj[k].is_empty() }

/// `fitEdge(e)` of opposingClearance.
#[allow(clippy::too_many_arguments)]
fn fit_edge(st: &mut St, set: &mut [u8], changed: &mut [bool], e: usize, s_first: &[usize], ss: &[f64], lb: &[f32], ubo: &[f32],
            viol_of: &dyn Fn(usize, f64) -> f64, cap_v: &[f64], flo_v: &[f64], tmin: f64, slope_max: f64) -> bool {
    let g = st.g;
    let sm = smooth;
    let (i0, i1, l) = (s_first[e], s_first[e + 1], g.length[e] as f64);
    let kind = |st: &St, key: usize| -> u8 {
        if !has_adj_st(st, key) { return 0; }
        let anchor = st.anchor_of[&st.root[key]] as usize;
        if g.opp[anchor >> 1] > 0 { 2 } else { 1 }
    };
    let (ks, ke) = (kind(st, 2 * e), kind(st, 2 * e + 1));
    let tau_p = |st: &St, end: usize| max(st.tau_of(e, end), min(l / 2.0, st.tau_min[2 * e + end] as f64));
    let r_at = |st: &St, s: f64| -> f64 {
        let rd = st.rd[e] as f64; let (ts, te) = (tau_p(st, 0), tau_p(st, 1));
        let mut r = rd;
        if ts > 0.0 { r += (st.rend[2 * e] as f64 - rd) * sm(1.0 - s / ts); }
        if te > 0.0 { r += (st.rend[2 * e + 1] as f64 - rd) * sm(1.0 - (l - s) / te); }
        r
    };
    let (mut ext_s, mut ext_e, mut cur) = (-1.0f64, -1.0f64, 0.0f64);
    for i in i0..i1 {
        let r = r_at(st, ss[i]); let v = viol_of(i, r);
        cur = max(cur, v);
        if max(lb[i] as f64 - r, r - ubo[i] as f64) <= 0.02 { continue; }
        if ss[i] <= l / 2.0 { ext_s = max(ext_s, ss[i]); } else { ext_e = max(ext_e, l - ss[i]); }
    }
    if ext_s < 0.0 && ext_e < 0.0 { return true; }
    changed[e] = true;
    let (cap, flo, rd0) = (cap_v[e], flo_v[e], st.rd[e] as f64);
    let plan = |st: &St, end: usize, ext: f64, k: u8| -> f64 {
        if k == 2 { -1.0 }
        else if ext >= 0.0 || set[2 * e + end] != 0 || k == 1 { min(l / 2.0, max(max(tmin, 2.0 * max(0.0, ext) + 10.0), st.tau_min[2 * e + end] as f64)) }
        else { 0.0 }
    };
    let (p_s, p_e) = (plan(st, 0, ext_s, ks), plan(st, 1, ext_e, ke));
    let mut cand = vec![rd0];
    for k in 1..=80 {
        let k = k as f64;
        if rd0 + 0.1 * k <= cap { cand.push(rd0 + 0.1 * k); }
        if rd0 - 0.1 * k >= flo { cand.push(rd0 - 0.1 * k); }
    }
    #[derive(Clone, Copy)] struct Best { ok: bool, cost: f64, viol: f64, rd: f64, rs: f64, re: f64, ts: f64, te: f64 }
    let mut best: Option<Best> = None;
    let pick = |base: f64, lo: f64, hi: f64, k: u8| -> f64 {
        if k == 2 { return base; }
        let (cp, fl) = (max(cap, base), min(flo, base));
        let (lo, hi) = (min(lo, cp), max(hi, fl));
        let v = if lo > hi { (lo + hi) / 2.0 } else { min(max(base, lo), hi) };
        min(cp, max(fl, v))
    };
    for &rd in &cand {
        st.rd[e] = rd as f32;
        let ts = if p_s < 0.0 { st.tau_of(e, 0) } else { p_s };
        let te = if p_e < 0.0 { st.tau_of(e, 1) } else { p_e };
        let base_s = if ks != 0 { st.rend[2 * e] as f64 } else if set[2 * e] != 0 { st.rend[2 * e] as f64 } else { rd };
        let base_e = if ke != 0 { st.rend[2 * e + 1] as f64 } else if set[2 * e + 1] != 0 { st.rend[2 * e + 1] as f64 } else { rd };
        let (mut s_lo, mut s_hi, mut e_lo, mut e_hi) = (f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY);
        for i in i0..i1 {
            let s = ss[i];
            let a_s = if ts > 0.0 { sm(1.0 - s / ts) } else { 0.0 };
            let a_e = if te > 0.0 { sm(1.0 - (l - s) / te) } else { 0.0 };
            let (lbi, ub) = (lb[i] as f64, ubo[i] as f64);
            if a_s > 0.02 && ks != 2 {
                if lbi > f64::NEG_INFINITY { s_lo = max(s_lo, (lbi - rd * (1.0 - a_s)) / a_s); }
                if ub < f64::INFINITY { s_hi = min(s_hi, (ub - rd * (1.0 - a_s)) / a_s); }
            } else if a_e > 0.02 && ke != 2 {
                if lbi > f64::NEG_INFINITY { e_lo = max(e_lo, (lbi - rd * (1.0 - a_e)) / a_e); }
                if ub < f64::INFINITY { e_hi = min(e_hi, (ub - rd * (1.0 - a_e)) / a_e); }
            }
        }
        let mut rs = pick(base_s, s_lo, s_hi, ks); let mut re = pick(base_e, e_lo, e_hi, ke);
        let taper = |k: u8, r: f64, base: f64, plan: f64| -> (f64, f64) {
            if !TAPER_LIMIT || k == 2 || (r - rd).abs() < 1e-3 || (r - base).abs() < 1e-3 { return (r, if plan < 0.0 { 0.0 } else { plan }); }
            let tau = min(l / 2.0, max(plan, TAU_PER_M * (r - rd).abs()));
            let mut r = r;
            if 1.5 * (r - rd).abs() / tau > slope_max { r = rd + sign(r - rd) * slope_max * tau / 1.5; }
            (r, tau)
        };
        let (mut ts_e, mut te_e) = (ts, te);
        if ks != 2 { let (a, b) = taper(ks, rs, base_s, ts); rs = a; ts_e = b; }
        if ke != 2 { let (a, b) = taper(ke, re, base_e, te); re = a; te_e = b; }
        let mut viol = 0.0f64;
        for i in i0..i1 {
            let s = ss[i];
            let a_s = if ts_e > 0.0 { sm(1.0 - s / ts_e) } else { 0.0 };
            let a_e = if te_e > 0.0 { sm(1.0 - (l - s) / te_e) } else { 0.0 };
            viol = max(viol, viol_of(i, rd * (1.0 - a_s - a_e) + rs * a_s + re * a_e));
        }
        let cost = (rd - rd0).abs() * l + (rs - base_s).abs() * ts_e / 2.0 + (re - base_e).abs() * te_e / 2.0;
        let ok = viol <= 0.02;
        let take = match &best { None => true, Some(b) => (ok && (!b.ok || cost < b.cost)) || (!ok && !b.ok && viol < b.viol - 1e-3) };
        if take { best = Some(Best { ok, cost, viol, rd, rs, re, ts: ts_e, te: te_e }); }
    }
    let best = best.unwrap();
    if !best.ok && best.viol >= cur - 1e-3 { st.rd[e] = rd0 as f32; return false; }
    st.rd[e] = best.rd as f32;
    for (end, k, r, tp) in [(0usize, ks, best.rs, best.ts), (1, ke, best.re, best.te)] {
        let key = 2 * e + end;
        if k == 2 { continue; }
        if k == 1 {
            let d = r - st.rend[key] as f64;
            if d.abs() > 1e-3 {
                let gr = st.root[key];
                let prev = st.group_raise.get(&gr).copied().unwrap_or(0.0);
                let prev = if prev.is_nan() { 0.0 } else { prev };
                st.group_raise.insert(gr, prev + d);
                st.rend[key] = r as f32; st.tau_min[key] = max(st.tau_min[key] as f64, tp) as f32;
            }
        } else if (r - best.rd).abs() > 1e-3 { st.rend[key] = r as f32; set[key] = 1; st.tau_min[key] = max(st.tau_min[key] as f64, tp) as f32; }
        else { st.rend[key] = best.rd as f32; set[key] = 0; }
    }
    best.ok
}
