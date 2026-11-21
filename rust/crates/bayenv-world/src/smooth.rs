//! Smoothed frame geometry for `curvature_dynamics` (rust/DECISIONS.md 33).
//!
//! The lane reference polylines (`LaneGeometry` vx / vz) and the junction connector polylines (`JunctionSet.pts`)
//! carry kinks with radii of a few metres. Once the road frame stops turning the car, every kink is a curve the car
//! must steer through, so this module derives a second, smoothed copy of that geometry at load time:
//!
//! * Every vertex (or run of vertices, see "merging") is replaced by a circular fillet arc tangent to the two
//!   segments around it, sampled every [`SmoothCfg::arc_step`] rad.
//! * The target radius comes from the road class and the edge's speed limit ([`target_radius`]): the curve radius
//!   for the speed limit, R = v^2 / (g (e + f)) (AASHTO superelevation e and side friction f), clamped to a floor and a
//!   cap per class. On straight-ish roads the radius is also capped so the arc stays within
//!   [`SmoothCfg::dev_max`] of the vertex (never below the class floor). The target applies to the innermost lane
//!   centre (the reference polyline gets the inner lane offset added).
//! * Where the segments are too short for the target, the largest radius that fits is used; two neighbouring
//!   fillets split a shared segment max-min fairly. Runs of short segments (a 0.7 rad turn digitised as three kinks
//!   within 2.5 m) are merged into one fillet at the intersection of the outer segments, when that gives a larger
//!   radius ("merging"; symmetric rounds, so a polyline and its reverse smooth alike).
//! * Consecutive edges at a node without a junction box and without any other edge (a pass-through node) are
//!   smoothed as one polyline, so the join kink gets a fillet too. Both edge ends move to the same arc point and
//!   share its normal, so lanes stay continuous across the hop.
//! * Connectors are re-fitted to the smoothed edges: their ends are the smoothed lane points at s_in / s_out, their
//!   end tangents the smoothed lane headings, and their interior is filleted like an edge.
//!
//! **s is preserved**: vertices outside a fillet keep their s exactly; a fillet's arc spans the s-range of the
//! polyline it replaces (from its first to its last tangent point), linear in angle. So edge lengths, s_exit /
//! s_stop, gore points, controls, connector s_in / s_out / lengths, conflict positions and the hop maps are
//! untouched. Inside a fillet the s-rate differs from the arc length by the factor (arc / replaced polyline), as
//! the JS geometry already does for lanes off the reference line.
use crate::junctions::JunctionSet;
use crate::lanegraph::LaneGraph;
use crate::lanes::LaneGeometry;
use rayon::prelude::*;

/// Smoothing parameters (the defaults are what the sim uses; DECISIONS.md 33).
#[derive(Clone, Debug)]
pub struct SmoothCfg {
    /// max angle per arc segment (rad) on edges (the lane frame heading is constant along a segment)
    pub arc_step: f64,
    /// ... and on connectors (`connector_point` blends the heading between segments, so coarser is fine)
    pub conn_arc_step: f64,
    /// a fillet's apex may move the line at most this far from its vertex (m), unless the class floor needs more
    pub dev_max: f64,
    /// ... but never more than this (m): a hairpin keeps a tight radius rather than cutting its corner off
    pub dev_hard: f64,
    /// straight run kept at an edge end that is not smoothed across (m, at most 30 % of the end segment)
    pub end_margin: f64,
    /// straight run kept at a connector end (m, at most 30 % of the end segment)
    pub conn_end_margin: f64,
    /// merged fillets may turn at most this much (rad): beyond it the outer segments' intersection runs away
    pub max_merge_turn: f64,
    /// connector target radius: clamp(min(target of in-edge, out-edge), conn_floor, conn_cap)
    pub conn_floor: f64,
    pub conn_cap: f64,
}
impl Default for SmoothCfg {
    fn default() -> Self { SmoothCfg { arc_step: 0.04, conn_arc_step: 0.1, dev_max: 1.0, dev_hard: 5.0, end_margin: 2.0, conn_end_margin: 0.5, max_merge_turn: 2.6, conn_floor: 12.0, conn_cap: 30.0 } }
}

/// Per-class radius rule: (superelevation e, floor m, cap m). Side friction f is a function of the speed
/// ([`side_friction`]). Index = `CLASSES_ALL`.
pub const CLASS_RADIUS: [(f64, f64, f64); 13] = [
    (0.06, 150.0, 900.0), // motorway
    (0.06, 30.0, 250.0),  // motorway_link
    (0.06, 150.0, 900.0), // trunk
    (0.06, 30.0, 250.0),  // trunk_link
    (0.04, 20.0, 150.0),  // primary
    (0.04, 15.0, 80.0),   // primary_link
    (0.04, 15.0, 120.0),  // secondary
    (0.04, 12.0, 60.0),   // secondary_link
    (0.02, 12.0, 60.0),   // tertiary
    (0.02, 12.0, 40.0),   // tertiary_link
    (0.0, 10.0, 15.0),    // unclassified
    (0.0, 10.0, 15.0),    // residential
    (0.0, 10.0, 12.0),    // living_street
];

/// AASHTO Green Book side-friction factor for curve design (Table 3-7, low- and high-speed values blended),
/// piecewise linear in km/h.
pub fn side_friction(kmh: f64) -> f64 {
    const T: [(f64, f64); 7] = [(20.0, 0.35), (40.0, 0.23), (60.0, 0.17), (80.0, 0.14), (100.0, 0.12), (120.0, 0.09), (140.0, 0.08)];
    if kmh <= T[0].0 { return T[0].1; }
    for w in T.windows(2) { if kmh <= w[1].0 { let u = (kmh - w[0].0) / (w[1].0 - w[0].0); return w[0].1 + (w[1].1 - w[0].1) * u; } }
    T[T.len() - 1].1
}

/// Target minimum radius (m) of the lane centres of an edge of class `cls` (`CLASSES_ALL` index) with speed limit
/// `speed` (m/s): clamp(v^2 / (g (e + f)), floor, cap).
pub fn target_radius(cls: u8, speed: f64) -> f64 {
    let (e, lo, hi) = CLASS_RADIUS[(cls as usize).min(12)];
    let v = if speed.is_finite() && speed > 0.0 { speed } else { 11.0 };
    let r = v * v / (9.81 * (e + side_friction(v * 3.6)));
    r.max(lo).min(hi)
}

/// What a vertex of a chain wants: target lane radius, class floor, and the lane-centre offset on the inside of a
/// right (> 0) / left (< 0) turn that is added to the reference-line radius.
#[derive(Clone, Copy, Debug)]
pub struct Want { pub r: f64, pub floor: f64, pub inner_right: f64, pub inner_left: f64 }

/// Statistics of one smoothing run.
#[derive(Clone, Debug, Default)]
pub struct SmoothStats {
    pub edges: usize, pub joins: usize, pub join_chains: usize, pub connectors: usize,
    /// fillets made (vertices with a turn > 1e-6 rad, after merging) on edges and on connectors
    pub edge_fillets: usize, pub conn_fillets: usize,
    /// of those: radius below the target because the segments are too short (largest radius that fits)
    pub edge_fit_limited: usize, pub conn_fit_limited: usize,
    /// of those: no fillet possible at all (merged run that could not be undone, degenerate segments)
    pub edge_unfilleted: usize, pub conn_unfilleted: usize,
    /// merges of neighbouring fillets
    pub merges: usize,
    /// vertices before / after
    pub lane_vertices: (usize, usize), pub conn_points: (usize, usize),
    /// largest move of an edge end (m) and of a connector end (m) to the smoothed lane point
    pub max_end_move: f64, pub max_conn_end_move: f64,
    /// largest apex move (m) among fillets turning < 0.1 rad (straight-ish roads), and among all fillets
    pub max_dev_straightish: f64, pub max_dev: f64,
}

/// One input polyline: points (x, z), anchor parameter s, height y.
#[derive(Clone, Copy, Debug)]
struct P { x: f64, z: f64, s: f64, y: f64 }

/// One output point: position, s, y, tangent (unit) when on an arc, original vertex index when kept.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Q { pub x: f64, pub z: f64, pub s: f64, pub y: f64, pub tan: Option<[f64; 2]>, pub orig: Option<usize>, pub keep: bool }

#[derive(Clone, Copy, Debug)]
struct Group { i: usize, j: usize, turn: f64, alpha: f64, beta: f64, vx: f64, vz: f64, want: f64, dev_ok: f64 }

#[inline] fn unit(dx: f64, dz: f64) -> (f64, f64, f64) { let l = (dx * dx + dz * dz).sqrt(); if l > 0.0 { (dx / l, dz / l, l) } else { (0.0, 0.0, 0.0) } }
#[inline] fn sec_m1(half: f64) -> f64 { 1.0 / half.cos() - 1.0 }

/// Result of smoothing one chain.
#[derive(Default)]
pub(crate) struct ChainOut { pub pts: Vec<Q>, pub fillets: usize, pub fit_limited: usize, pub unfilleted: usize, pub merges: usize, pub max_dev_straightish: f64, pub max_dev: f64 }

struct Chain<'a> { p: &'a [P], seg: Vec<(f64, f64, f64)>, wants: &'a [Want], margin: (f64, f64), cfg: &'a SmoothCfg }

impl<'a> Chain<'a> {
    /// a group over vertices i..=j: virtual vertex at the intersection of the segment into i and out of j
    fn group(&self, i: usize, j: usize) -> Option<Group> {
        let (ux, uz, _) = self.seg[i - 1]; let (wx, wz, _) = self.seg[j];
        let turn: f64 = (i..=j).map(|k| self.turn_at(k)).sum();
        let t2 = (ux * wz - uz * wx).atan2(ux * wx + uz * wz);
        if (t2 - turn).abs() > 1e-6 { return None; }
        let (pi, pj) = (&self.p[i], &self.p[j]);
        let (alpha, beta, vx, vz) = if i == j { (0.0, 0.0, pi.x, pi.z) } else {
            // pi + alpha u = pj - beta w, i.e. alpha u + beta w = pj - pi
            let det = ux * wz - uz * wx;
            if det.abs() < 1e-12 { return None; }
            let (rx, rz) = (pj.x - pi.x, pj.z - pi.z);
            let alpha = (rx * wz - rz * wx) / det;
            let beta = (ux * rz - uz * rx) / det;
            if !(alpha >= -1e-9 && beta >= -1e-9) { return None; }
            (alpha.max(0.0), beta.max(0.0), pi.x + alpha * ux, pi.z + alpha * uz)
        };
        let (mut r, mut floor, mut inner) = (f64::INFINITY, f64::INFINITY, 0.0f64);
        for k in i..=j { let w = &self.wants[k]; r = r.min(w.r); floor = floor.min(w.floor); inner = inner.max(if turn > 0.0 { w.inner_right } else { w.inner_left }); }
        let h = 0.5 * turn.abs();
        let inner = inner.max(0.0);
        let (soft, hard) = if h > 1e-9 { (self.cfg.dev_max / sec_m1(h), self.cfg.dev_hard / sec_m1(h)) } else { (f64::INFINITY, f64::INFINITY) };
        // the target, capped by the soft apex move; never below the floor, unless that moves the apex more than dev_hard
        let want = (floor + inner).min(hard).max((r + inner).min(soft));
        // the largest apex move a merge may cause: that of the floor-radius fillet (or dev_max), at most dev_hard
        let dev_ok = self.cfg.dev_max.max(((floor + inner) * sec_m1(h)).min(self.cfg.dev_hard));
        Some(Group { i, j, turn, alpha, beta, vx, vz, want, dev_ok })
    }
    #[inline] fn turn_at(&self, k: usize) -> f64 {
        let (ux, uz, _) = self.seg[k - 1]; let (wx, wz, _) = self.seg[k];
        (ux * wz - uz * wx).atan2(ux * wx + uz * wz)
    }
    /// tangent length the group wants on each side (beyond its alpha / beta)
    #[inline] fn need(&self, g: &Group) -> (f64, f64) { if g.turn.abs() > self.cfg.max_merge_turn { return (0.0, 0.0); } let a = g.want * (0.5 * g.turn.abs()).tan(); ((a - g.alpha).max(0.0), (a - g.beta).max(0.0)) }
    /// tangent length a of group gs[t] given its neighbours' needs (None = infeasible)
    fn fit(&self, gs: &[Group], t: usize, g: &Group) -> Option<f64> {
        let tn = (0.5 * g.turn.abs()).tan();
        if tn < 1e-12 { return Some(0.0); }
        let a_want = g.want * tn;
        let n = self.p.len();
        let share = |l: f64, other: f64| l.max(0.0).min((0.5 * l).max(l - other));
        // segment into g: i - 1 .. i
        let li = self.seg[g.i - 1].2;
        let avail_in = if g.i - 1 == 0 { li - self.margin.0.min(0.3 * li) } else {
            let other = if t > 0 && gs[t - 1].j == g.i - 1 { self.need(&gs[t - 1]).1 } else { 0.0 };
            share(li, other)
        };
        let lo = self.seg[g.j].2;
        let avail_out = if g.j + 1 == n - 1 { lo - self.margin.1.min(0.3 * lo) } else {
            let other = if t + 1 < gs.len() && gs[t + 1].i == g.j + 1 { self.need(&gs[t + 1]).0 } else { 0.0 };
            share(lo, other)
        };
        let a = a_want.min(g.alpha + avail_in.max(0.0)).min(g.beta + avail_out.max(0.0));
        if a < g.alpha.max(g.beta) - 1e-9 { return None; }
        Some(a)
    }
    fn ratio(&self, gs: &[Group], t: usize, g: &Group) -> f64 {
        if (0.5 * g.turn.abs()).tan() < 1e-12 || g.turn.abs() > self.cfg.max_merge_turn { return 1.0; }
        match self.fit(gs, t, g) { Some(a) => a / (g.want * (0.5 * g.turn.abs()).tan()), None => -1.0 }
    }
    /// largest distance of the member vertices from the fillet arc of radius r
    fn member_dev(&self, g: &Group, a: f64) -> f64 {
        let (ux, uz, _) = self.seg[g.i - 1];
        let tn = (0.5 * g.turn.abs()).tan(); if tn < 1e-12 { return 0.0; }
        let r = a / tn; let sg = g.turn.signum();
        let (t1x, t1z) = (g.vx - a * ux, g.vz - a * uz);
        let (cx, cz) = (t1x - sg * r * uz, t1z + sg * r * ux);
        (g.i..=g.j).map(|k| (((self.p[k].x - cx).powi(2) + (self.p[k].z - cz).powi(2)).sqrt() - r).abs()).fold(0.0, f64::max)
    }
}

/// Smooth one polyline with fixed end points and end tangents. `wants[k]` for interior vertices (ends ignored);
/// `bounds`: anchor-s values strictly inside the chain at which an output point must exist (edge boundaries of a
/// joined chain); `margin`: straight run kept at the start / end.
/// `min_gap`: output points closer than this (m, in position or s) to the previous one are dropped.
pub(crate) fn smooth_chain(p_in: &[[f64; 4]], wants: &[Want], bounds: &[f64], margin: (f64, f64), min_gap: f64, cfg: &SmoothCfg) -> ChainOut {
    let p: Vec<P> = p_in.iter().map(|q| P { x: q[0], z: q[1], s: q[2], y: q[3] }).collect();
    let n = p.len();
    let mut out = ChainOut::default();
    let keep_all = |out: &mut ChainOut| { out.pts = p.iter().enumerate().map(|(k, q)| Q { x: q.x, z: q.z, s: q.s, y: q.y, tan: None, orig: Some(k), keep: true }).collect(); };
    if n < 3 { keep_all(&mut out); return out; }
    let seg: Vec<(f64, f64, f64)> = (0..n - 1).map(|k| unit(p[k + 1].x - p[k].x, p[k + 1].z - p[k].z)).collect();
    if seg.iter().any(|s| s.2 < 1e-9) { keep_all(&mut out); out.unfilleted = n - 2; return out; }
    let ch = Chain { p: &p, seg, wants, margin, cfg };
    let mut gs: Vec<Group> = (1..n - 1).filter_map(|k| ch.group(k, k)).collect();
    // merging rounds
    for _round in 0..64 {
        let m = gs.len(); if m < 2 { break; }
        let q: Vec<f64> = (0..m).map(|t| ch.ratio(&gs, t, &gs[t])).collect();
        // merging groups t0..=t1 into one: (benefit, merged group)
        let eval = |t0: usize, t1: usize| -> Option<(f64, Group)> {
            let worst = q[t0..=t1].iter().copied().fold(f64::INFINITY, f64::min);
            if worst >= 1.0 - 1e-9 { return None; }
            let (mut pos, mut neg) = (false, false);
            for g in &gs[t0..=t1] { if g.turn > 1e-6 { pos = true; } if g.turn < -1e-6 { neg = true; } }
            if pos && neg { return None; }
            for t in t0..t1 { if gs[t + 1].i != gs[t].j + 1 { return None; } }
            let g = ch.group(gs[t0].i, gs[t1].j);
            let g = g?;
            if g.turn.abs() > cfg.max_merge_turn { return None; }
            let mut tmp: Vec<Group> = Vec::with_capacity(3);
            if t0 > 0 { tmp.push(gs[t0 - 1]); }
            let k = tmp.len(); tmp.push(g);
            if t1 + 1 < m { tmp.push(gs[t1 + 1]); }
            let af = ch.fit(&tmp, k, &g)?;
            if ch.member_dev(&g, af) > g.dev_ok + 1e-9 { return None; }
            let ben = ch.ratio(&tmp, k, &g) - worst;
            if ben > 1e-6 { Some((ben, g)) } else { None }
        };
        // 1. clusters: maximal runs of same-sign neighbours whose fillets overlap on the segment between them; each
        //    merges as a whole when that improves its worst radius (symmetric under reversal)
        {
            let conflict = |t: usize| -> bool {
                let (a, b) = (&gs[t], &gs[t + 1]);
                if b.i != a.j + 1 || a.turn * b.turn < 0.0 { return false; }
                ch.need(a).1 + ch.need(b).0 > ch.seg[a.j].2 + 1e-9
            };
            let mut next: Vec<Group> = Vec::with_capacity(m);
            let mut merged = false;
            let mut t = 0;
            while t < m {
                let mut u = t; while u + 1 < m && conflict(u) { u += 1; }
                if u > t { if let Some((_, g)) = eval(t, u) { next.push(g); merged = true; out.merges += 1; t = u + 1; continue; } }
                for k in t..=u { next.push(gs[k]); }
                t = u + 1;
            }
            if merged { gs = next; continue; }
        }
        // 2. pairs
        // candidate merges of t and t + 1 with their benefit
        let cand: Vec<Option<(f64, Group)>> = (0..m - 1).map(|t| eval(t, t + 1)).collect();
        let ben = |t: usize| cand[t].map_or(f64::NEG_INFINITY, |c| c.0);
        let same = |a: f64, b: f64| (a - b).abs() <= 1e-9 * a.abs().max(1.0);
        // merge the candidates whose benefit is a strict local maximum; a plateau of equal benefits (symmetric
        // geometry) merges as one run, so the result does not depend on the polyline's direction
        let mut merged = false;
        let mut next: Vec<Group> = Vec::with_capacity(m);
        let mut t = 0;
        while t < m {
            if t + 1 < m && cand[t].is_some() {
                let b = ben(t);
                let mut u = t; while u + 1 < m - 1 && cand[u + 1].is_some() && same(ben(u + 1), b) { u += 1; }
                let left = if t > 0 { ben(t - 1) } else { f64::NEG_INFINITY };
                let right = if u + 1 < m - 1 { ben(u + 1) } else { f64::NEG_INFINITY };
                if b > left && !same(b, left) && b > right && !same(b, right) {
                    let g = if u == t { cand[t].map(|c| c.1) } else { eval(t, u + 1).map(|c| c.1) };
                    if let Some(g) = g { next.push(g); t = u + 2; merged = true; out.merges += 1; continue; }
                }
                // not merged: keep t..=u as they are (u + 1 may still merge with its right neighbour)
                for k in t..=u { next.push(gs[k]); }
                t = u + 1; continue;
            }
            next.push(gs[t]); t += 1;
        }
        gs = next;
        if !merged { break; }
    }
    // emit
    let mut pts: Vec<Q> = Vec::with_capacity(n * 2);
    pts.push(Q { x: p[0].x, z: p[0].z, s: p[0].s, y: p[0].y, tan: None, orig: Some(0), keep: true });
    let y_at = |s: f64| -> f64 {
        if s <= p[0].s { return p[0].y; }
        let k = p.partition_point(|q| q.s <= s);
        if k >= n { return p[n - 1].y; }
        let (a, b) = (&p[k - 1], &p[k]); let d = b.s - a.s;
        if d > 0.0 { a.y + (b.y - a.y) * (s - a.s) / d } else { a.y }
    };
    // append in s order; a point closer than min_gap to the previous one (a tangent point next to a kept vertex or
    // another tangent point) is dropped, and a point that must stay (an input vertex, an edge boundary) replaces an
    // arc / tangent point there
    let push = |pts: &mut Vec<Q>, q: Q| {
        let l = *pts.last().unwrap();
        if (q.x - l.x).hypot(q.z - l.z) < min_gap || q.s <= l.s + min_gap {
            if q.keep && !l.keep && pts.len() > 1 { *pts.last_mut().unwrap() = q; }
            else if q.keep && q.s > l.s && (q.x - l.x).hypot(q.z - l.z) > 1e-9 { pts.push(q); }
            return;
        }
        pts.push(q);
    };
    let gs_len = gs.len();
    for t in 0..gs_len {
        let g = gs[t];
        let tn = (0.5 * g.turn.abs()).tan();
        // no fillet for a near-reversal (a dead-end stub that turns back): its tangent points would run away
        let a = if tn < 1e-12 || g.turn.abs() > cfg.max_merge_turn { None } else { ch.fit(&gs, t, &g) };
        let a = match a { Some(a) if a > 1e-6 => a, _ => {
            if tn >= 1e-12 { out.unfilleted += 1; }
            for k in g.i..=g.j { push(&mut pts, Q { x: p[k].x, z: p[k].z, s: p[k].s, y: p[k].y, tan: None, orig: Some(k), keep: true }); }
            continue;
        } };
        out.fillets += 1;
        if a < g.want * tn * (1.0 - 1e-6) { out.fit_limited += 1; }
        let r = a / tn;
        let dev = ch.member_dev(&g, a);
        out.max_dev = out.max_dev.max(dev);
        if g.turn.abs() < 0.1 { out.max_dev_straightish = out.max_dev_straightish.max(dev); }
        let (ux, uz, li) = ch.seg[g.i - 1]; let (wx, wz, lo) = ch.seg[g.j];
        let (t1x, t1z) = (g.vx - a * ux, g.vz - a * uz);
        let (t2x, t2z) = (g.vx + a * wx, g.vz + a * wz);
        let (pa, pb) = (&p[g.i - 1], &p[g.i]);
        let s1 = pb.s - (a - g.alpha) * (pb.s - pa.s) / li;
        let (pc, pd) = (&p[g.j], &p[g.j + 1]);
        let s2 = pc.s + (a - g.beta) * (pd.s - pc.s) / lo;
        let sg = g.turn.signum();
        let (cx, cz) = (t1x - sg * r * uz, t1z + sg * r * ux);
        let (rx, rz) = (t1x - cx, t1z - cz);
        let at = |phi: f64| -> (f64, f64, [f64; 2]) {
            let (c, s) = ((sg * phi).cos(), (sg * phi).sin());
            (cx + rx * c - rz * s, cz + rx * s + rz * c, [ux * c - uz * s, ux * s + uz * c])
        };
        push(&mut pts, Q { x: t1x, z: t1z, s: s1, y: y_at(s1), tan: Some([ux, uz]), orig: None, keep: false });
        let th = g.turn.abs();
        // arc samples every <= arc_step, with the required boundary points as sample points (pieces between them
        // sampled evenly)
        let mut cuts: Vec<f64> = vec![0.0];
        for &b in bounds { if b > s1 && b < s2 { cuts.push((b - s1) / (s2 - s1)); } }
        cuts.push(1.0);
        cuts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in cuts.windows(2) {
            let (f0, f1) = (w[0], w[1]);
            let ns = ((f1 - f0) * th / cfg.arc_step).ceil().max(1.0) as usize;
            for k in 1..=ns {
                if k == ns && f1 >= 1.0 { break; }
                let f = f0 + (f1 - f0) * k as f64 / ns as f64;
                let (x, z, tg) = at(f * th);
                let mut s = s1 + f * (s2 - s1);
                let mut keep = false;
                if k == ns { for &b in bounds { if (s - b).abs() < 1e-6 * (1.0 + b.abs()) { s = b; keep = true; } } }
                push(&mut pts, Q { x, z, s, y: y_at(s), tan: Some(tg), orig: None, keep });
            }
        }
        push(&mut pts, Q { x: t2x, z: t2z, s: s2, y: y_at(s2), tan: Some([wx, wz]), orig: None, keep: false });
    }
    push(&mut pts, Q { x: p[n - 1].x, z: p[n - 1].z, s: p[n - 1].s, y: p[n - 1].y, tan: None, orig: Some(n - 1), keep: true });
    out.pts = pts;
    out
}

/// Pass-through joins (a, b): node `to(a)` has no junction box, `a`'s only successor (bar its twin) is `b` and
/// `b`'s only predecessor (bar its twin) is `a`.
pub fn pass_through_joins(g: &LaneGraph, junction_nodes: &[u32]) -> Vec<(u32, u32)> {
    let mut jn = vec![false; g.node_count];
    for &n in junction_nodes { if (n as usize) < jn.len() { jn[n as usize] = true; } }
    let mut out = Vec::new();
    for a in 0..g.edge_count {
        let n = g.to(a);
        if jn[n] { continue; }
        let outs: Vec<u32> = g.out_edges(n).iter().copied().filter(|&o| o as i32 != g.twin[a]).collect();
        if outs.len() != 1 { continue; }
        let b = outs[0] as usize;
        if b == a { continue; }
        let ins: Vec<u32> = g.in_edges(n).iter().copied().filter(|&i| i as i32 != g.twin[b]).collect();
        if ins.len() != 1 || ins[0] as usize != a { continue; }
        out.push((a as u32, b as u32));
    }
    out
}

fn edge_want(lg: &LaneGeometry, g: &LaneGraph, speed: &[f64], e: usize) -> Want {
    let w = { let w = lg.lane_w[e] as f64; if w > 0.0 { w } else { 3.7 } };
    let rd = lg.rd[e] as f64; let n = g.lanes[e].max(1) as f64;
    // lane-centre offsets right of the reference line: rd - w / 2 (rightmost) .. rd - n w + w / 2 (leftmost)
    Want { r: target_radius(g.cls[e], speed[e]), floor: CLASS_RADIUS[(g.cls[e] as usize).min(12)].1, inner_right: rd - 0.5 * w, inner_left: -(rd - n * w + 0.5 * w) }
}

/// Smooth the lane geometry and the junction connectors. `speed`: per-edge speed limit (m/s, SimGraph numbering).
pub fn smooth_geometry(lg: &LaneGeometry, g: &LaneGraph, speed: &[f64], j: Option<&JunctionSet>, cfg: &SmoothCfg) -> (LaneGeometry, Option<JunctionSet>, SmoothStats) {
    let e_n = g.edge_count;
    let mut st = SmoothStats { edges: e_n, ..Default::default() };
    let jnodes: Vec<u32> = j.map(|j| j.j_node.to_vec()).unwrap_or_default();
    let joins = pass_through_joins(g, &jnodes);
    st.joins = joins.len();
    // chains of joined edges
    let mut next = vec![-1i64; e_n]; let mut prev = vec![-1i64; e_n];
    for &(a, b) in &joins {
        // the two edges must meet exactly (shared node) and have >= 2 vertices
        let (va, vb) = (lg.v_first[a as usize + 1] as usize - 1, lg.v_first[b as usize] as usize);
        if (lg.vx[va] - lg.vx[vb]).abs() > 1e-6 || (lg.vz[va] - lg.vz[vb]).abs() > 1e-6 { continue; }
        next[a as usize] = b as i64; prev[b as usize] = a as i64;
    }
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut seen = vec![false; e_n];
    for e in 0..e_n {
        if seen[e] || prev[e] >= 0 { continue; }
        let mut c = vec![e]; seen[e] = true; let mut x = e;
        while next[x] >= 0 && !seen[next[x] as usize] { x = next[x] as usize; seen[x] = true; c.push(x); }
        chains.push(c);
    }
    // cycles: break at the lowest edge (its start join is not smoothed)
    for e in 0..e_n {
        if seen[e] { continue; }
        let mut c = vec![e]; seen[e] = true; let mut x = e;
        while next[x] >= 0 && !seen[next[x] as usize] { x = next[x] as usize; seen[x] = true; c.push(x); }
        chains.push(c);
    }
    st.join_chains = chains.iter().filter(|c| c.len() > 1).count();
    // smooth each chain
    struct EdgeOut { e: usize, x: Vec<f64>, y: Vec<f32>, z: Vec<f64>, s: Vec<f64>, mx: Vec<f64>, mz: Vec<f64> }
    let res: Vec<(Vec<EdgeOut>, ChainOut)> = chains.par_iter().map(|c| {
        let mut p: Vec<[f64; 4]> = Vec::new(); let mut wants: Vec<Want> = Vec::new();
        let mut bounds: Vec<f64> = Vec::new(); let mut base = 0.0f64;
        let mut starts: Vec<f64> = Vec::new(); // chain-s offset of each edge
        let mut ls: Vec<f64> = Vec::new(); // local s of each chain point
        for (k, &e) in c.iter().enumerate() {
            let (v0, v1) = (lg.v_first[e] as usize, lg.v_first[e + 1] as usize);
            let w = edge_want(lg, g, speed, e);
            let skip = if k == 0 { 0 } else { 1 };
            if k > 0 {
                // the join vertex: the stricter of the two edges
                let wp = wants.last_mut().unwrap();
                *wp = Want { r: wp.r.min(w.r), floor: wp.floor.min(w.floor), inner_right: wp.inner_right.max(w.inner_right), inner_left: wp.inner_left.max(w.inner_left) };
                bounds.push(base);
            }
            starts.push(base);
            for v in v0 + skip..v1 { p.push([lg.vx[v], lg.vz[v], base + lg.vs[v], lg.vy[v] as f64]); ls.push(lg.vs[v]); wants.push(w); }
            base += lg.vs[v1 - 1];
        }
        let cfg_m = (cfg.end_margin, cfg.end_margin);
        let out = smooth_chain(&p, &wants, &bounds, cfg_m, 0.05, cfg);
        // split per edge: edge k covers chain s in [starts[k].1, starts[k].1 + vs_last]
        let mut eo = Vec::with_capacity(c.len());
        let mut qi = 0usize;
        for (k, &e) in c.iter().enumerate() {
            let (v0, v1) = (lg.v_first[e] as usize, lg.v_first[e + 1] as usize);
            let (lo, hi) = (starts[k], if k + 1 < c.len() { starts[k + 1] } else { f64::INFINITY });
            let mut q: Vec<Q> = Vec::new();
            while qi < out.pts.len() && out.pts[qi].s < lo { qi += 1; }
            let mut qq = qi;
            while qq < out.pts.len() && out.pts[qq].s <= hi { q.push(out.pts[qq]); qq += 1; }
            // the next edge starts at the boundary point (shared)
            qi = if qq > 0 { qq - 1 } else { 0 };
            let nq = q.len();
            let mut x = Vec::with_capacity(nq); let mut y = Vec::with_capacity(nq); let mut z = Vec::with_capacity(nq); let mut s = Vec::with_capacity(nq);
            for (i, a) in q.iter().enumerate() {
                x.push(a.x); z.push(a.z); y.push(a.y as f32);
                // the original local s exactly for kept vertices and the edge ends
                s.push(if i == 0 { 0.0 } else if i + 1 == nq { lg.vs[v1 - 1] } else if let Some(o) = a.orig { ls[o] } else { a.s - lo });
            }
            // normals: mitres inside (as buildLaneGeometry); the ends keep their normal when the end vertex is kept,
            // else the arc normal there
            let mut mx = vec![0.0; nq]; let mut mz = vec![0.0; nq];
            let sd = |a: usize, b: usize| -> (f64, f64) { let (ux, uz, _) = unit(x[b] - x[a], z[b] - z[a]); (ux, uz) };
            for i in 0..nq {
                let (mut ux, mut uz) = (0.0, 0.0);
                if i > 0 { let u = sd(i - 1, i); ux += u.0; uz += u.1; }
                if i + 1 < nq { let u = sd(i, i + 1); ux += u.0; uz += u.1; }
                let l = (ux * ux + uz * uz).sqrt(); let l = if l > 0.0 { l } else { 1.0 }; ux /= l; uz /= l;
                let mut sc = 1.0;
                if i > 0 && i + 1 < nq { let u = sd(i - 1, i); sc = 1.0 / (u.0 * ux + u.1 * uz).max(0.5); }
                mx[i] = -uz * sc; mz[i] = ux * sc;
            }
            for (i, vi) in [(0usize, v0), (nq - 1, v1 - 1)] {
                let a = &q[i];
                if a.orig.is_some() { mx[i] = lg.mx[vi]; mz[i] = lg.mz[vi]; }
                else if let Some(t) = a.tan { mx[i] = -t[1]; mz[i] = t[0]; }
            }
            eo.push(EdgeOut { e, x, y, z, s, mx, mz });
        }
        (eo, out)
    }).collect();
    // assemble
    let mut per: Vec<Option<EdgeOut>> = (0..e_n).map(|_| None).collect();
    for (eo, co) in res {
        st.edge_fillets += co.fillets; st.edge_fit_limited += co.fit_limited; st.edge_unfilleted += co.unfilleted; st.merges += co.merges;
        st.max_dev = st.max_dev.max(co.max_dev); st.max_dev_straightish = st.max_dev_straightish.max(co.max_dev_straightish);
        for o in eo { let e = o.e; per[e] = Some(o); }
    }
    let mut v_first = vec![0u32; e_n + 1];
    let (mut vx, mut vy, mut vz, mut vs, mut mx, mut mz) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for e in 0..e_n {
        v_first[e] = vx.len() as u32;
        let o = per[e].take().expect("every edge is in a chain");
        let (v0, v1) = (lg.v_first[e] as usize, lg.v_first[e + 1] as usize - 1);
        let mv = ((o.x[0] - lg.vx[v0]).hypot(o.z[0] - lg.vz[v0])).max((o.x[o.x.len() - 1] - lg.vx[v1]).hypot(o.z[o.z.len() - 1] - lg.vz[v1]));
        st.max_end_move = st.max_end_move.max(mv);
        vx.extend(o.x); vy.extend(o.y); vz.extend(o.z); vs.extend(o.s); mx.extend(o.mx); mz.extend(o.mz);
    }
    v_first[e_n] = vx.len() as u32;
    st.lane_vertices = (lg.vx.len(), vx.len());
    let slg = LaneGeometry { v_first: v_first.into(), vx: vx.into(), vy: vy.into(), vz: vz.into(), vs: vs.into(), mx: mx.into(), mz: mz.into(), ..lg.clone() };
    let sj = j.map(|j| smooth_connectors(&slg, g, speed, j, cfg, &mut st));
    (slg, sj, st)
}

/// The lane-centre point (and heading) of lane `lane` at s, as the junction builder's `lane_pt`.
fn lane_pt(lg: &LaneGeometry, e: usize, s: f64, lane: f64) -> crate::lanes::Pt { lg.lat_point(e, s, (lane + 0.5) * lg.lane_width(e, s)) }

fn smooth_connectors(slg: &LaneGeometry, g: &LaneGraph, speed: &[f64], j: &JunctionSet, cfg: &SmoothCfg, st: &mut SmoothStats) -> JunctionSet {
    let mc = j.move_count();
    st.connectors = mc;
    let res: Vec<(Vec<[f32; 4]>, ChainOut, f64)> = (0..mc).into_par_iter().map(|m| {
        let mi = &j.m_info[7 * m..7 * m + 7]; let mf = &j.m_f[4 * m..4 * m + 4];
        let (a, la, b, lb, first, n) = (mi[0] as usize, mi[1] as f64, mi[2] as usize, mi[3] as f64, mi[4] as usize, mi[5] as usize);
        let raw: Vec<[f64; 4]> = (0..n).map(|k| { let q = &j.pts[4 * (first + k)..4 * (first + k) + 4]; [q[0] as f64, q[2] as f64, q[3] as f64, q[1] as f64] }).collect();
        if n < 2 { return (raw.iter().map(|q| [q[0] as f32, q[3] as f32, q[1] as f32, q[2] as f32]).collect(), ChainOut::default(), 0.0); }
        let (pa, pb) = (lane_pt(slg, a, mf[0] as f64, la), lane_pt(slg, b, mf[1] as f64, lb));
        // move the ends onto the smoothed lanes, blending the displacement over the connector (smoothstep in s)
        let (d0, d1) = ([pa.x - raw[0][0], pa.z - raw[0][1]], [pb.x - raw[n - 1][0], pb.z - raw[n - 1][1]]);
        let mv = (d0[0].hypot(d0[1])).max(d1[0].hypot(d1[1]));
        let (s0, sl) = (raw[0][2], raw[n - 1][2]);
        let mut p: Vec<[f64; 4]> = raw.iter().map(|q| {
            let u = if sl > s0 { ((q[2] - s0) / (sl - s0)).clamp(0.0, 1.0) } else { 0.0 }; let w = u * u * (3.0 - 2.0 * u);
            [q[0] + d0[0] * (1.0 - w) + d1[0] * w, q[1] + d0[1] * (1.0 - w) + d1[1] * w, q[2], q[3]]
        }).collect();
        p[0][0] = pa.x; p[0][1] = pa.z; p[n - 1][0] = pb.x; p[n - 1][1] = pb.z;
        // end tangents = the smoothed lane headings: when the first / last segment is off, a 0.5 m lead segment along
        // the lane heading replaces the points within 0.6 m of the end (the fillet then blends it in)
        let ang = |dx: f64, dz: f64, hx: f64, hz: f64| { let (ux, uz, l) = unit(dx, dz); if l < 1e-9 { 0.0 } else { (ux * hz - uz * hx).atan2(ux * hx + uz * hz).abs() } };
        let k0 = 0.5f64.min(0.2 * (sl - s0));
        let y_of = |p: &[[f64; 4]], s: f64| -> f64 { let k = p.partition_point(|q| q[2] <= s).clamp(1, p.len() - 1); let (a, b) = (&p[k - 1], &p[k]); if b[2] > a[2] { a[3] + (b[3] - a[3]) * ((s - a[2]) / (b[2] - a[2])).clamp(0.0, 1.0) } else { a[3] } };
        if k0 > 0.02 && ang(p[1][0] - p[0][0], p[1][1] - p[0][1], pa.hx, pa.hz) > 1e-4 {
            let lead = [pa.x + pa.hx * k0, pa.z + pa.hz * k0, s0 + k0, y_of(&p, s0 + k0)];
            p.retain(|q| q[2] == s0 || q[2] == sl || q[2] > s0 + 1.2 * k0);
            p.insert(1, lead);
        }
        let n2 = p.len();
        if k0 > 0.02 && ang(p[n2 - 1][0] - p[n2 - 2][0], p[n2 - 1][1] - p[n2 - 2][1], pb.hx, pb.hz) > 1e-4 {
            let lead = [pb.x - pb.hx * k0, pb.z - pb.hz * k0, sl - k0, y_of(&p, sl - k0)];
            p.retain(|q| q[2] == s0 || q[2] == sl || q[2] < sl - 1.2 * k0 || (q[2] <= s0 + 1.0001 * k0 && q[2] > s0));
            let n3 = p.len();
            p.insert(n3 - 1, lead);
        }
        // drop points closer than 2 cm (f32 positions)
        let mut q: Vec<[f64; 4]> = Vec::with_capacity(p.len());
        for (k, x) in p.iter().enumerate() {
            if let Some(l) = q.last() { if (x[0] - l[0]).hypot(x[1] - l[1]) < 0.02 { if k + 1 < p.len() { continue; } q.pop(); } }
            q.push(*x);
        }
        let rt = target_radius(g.cls[a], speed[a]).min(target_radius(g.cls[b], speed[b])).max(cfg.conn_floor).min(cfg.conn_cap);
        let want = Want { r: rt, floor: cfg.conn_floor.min(rt), inner_right: 0.0, inner_left: 0.0 };
        let wants = vec![want; q.len()];
        let ccfg = SmoothCfg { dev_max: f64::INFINITY, dev_hard: f64::INFINITY, arc_step: cfg.conn_arc_step, ..cfg.clone() };
        let out = smooth_chain(&q, &wants, &[], (cfg.conn_end_margin, cfg.conn_end_margin), 0.05, &ccfg);
        let pts: Vec<[f32; 4]> = out.pts.iter().map(|o| [o.x as f32, o.y as f32, o.z as f32, o.s as f32]).collect();
        (pts, out, mv)
    }).collect();
    let mut m_info = j.m_info.to_vec();
    let mut pts: Vec<f32> = Vec::with_capacity(j.pts.len() * 2);
    for (m, (p, co, mv)) in res.into_iter().enumerate() {
        st.conn_fillets += co.fillets; st.conn_fit_limited += co.fit_limited; st.conn_unfilleted += co.unfilleted; st.merges += co.merges;
        st.max_conn_end_move = st.max_conn_end_move.max(mv);
        m_info[7 * m + 4] = (pts.len() / 4) as u32; m_info[7 * m + 5] = p.len() as u32;
        for q in p { pts.extend_from_slice(&q); }
    }
    st.conn_points = (j.pts.len() / 4, pts.len() / 4);
    JunctionSet { m_info: m_info.into(), pts: pts.into(), ..j.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn want(r: f64) -> Want { Want { r, floor: r, inner_right: 0.0, inner_left: 0.0 } }
    fn chain(pts: &[(f64, f64)]) -> Vec<[f64; 4]> {
        let mut s = 0.0; let mut out = Vec::new();
        for (k, &(x, z)) in pts.iter().enumerate() { if k > 0 { s += (x - pts[k - 1].0).hypot(z - pts[k - 1].1); } out.push([x, z, s, 0.0]); }
        out
    }
    /// fillet radius at each interior vertex: 0.5 min(adjacent) / tan(turn / 2)
    fn radii(q: &[Q]) -> Vec<(f64, f64)> {
        (1..q.len() - 1).map(|k| {
            let (a, b, c) = (&q[k - 1], &q[k], &q[k + 1]);
            let (l1, l2) = ((b.x - a.x).hypot(b.z - a.z), (c.x - b.x).hypot(c.z - b.z));
            let t = ((b.z - a.z).atan2(b.x - a.x) - (c.z - b.z).atan2(c.x - b.x)).sin().atan2(((b.z - a.z).atan2(b.x - a.x) - (c.z - b.z).atan2(c.x - b.x)).cos());
            (t.abs(), if t.abs() < 1e-9 { f64::INFINITY } else { 0.5 * l1.min(l2) / (t.abs() / 2.0).tan() })
        }).collect()
    }
    #[test]
    fn right_angle_gets_the_target_radius() {
        let p = chain(&[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)]);
        let o = smooth_chain(&p, &vec![want(12.0); 3], &[], (2.0, 2.0), 1e-6, &SmoothCfg::default());
        assert_eq!(o.fillets, 1); assert_eq!(o.fit_limited, 0);
        let rmin = radii(&o.pts).into_iter().filter(|r| r.0 > 1e-6).map(|r| r.1).fold(f64::INFINITY, f64::min);
        assert!(rmin > 12.0 * (0.02f64).cos() - 1e-9 && rmin < 12.0 * 1.001, "radius {rmin}");
        // ends and their tangents unchanged
        assert_eq!((o.pts[0].x, o.pts[0].z), (0.0, 0.0)); let l = o.pts.last().unwrap(); assert_eq!((l.x, l.z), (50.0, 50.0));
        // s anchored: the end keeps its s, straight parts keep theirs
        assert_eq!(l.s, 100.0);
        assert!(o.pts.windows(2).all(|w| w[1].s > w[0].s), "s increases");
        // the tangent points are 12 m from the corner
        let t1 = o.pts[1]; assert!((t1.x - 38.0).abs() < 1e-9 && t1.z.abs() < 1e-12 && (t1.s - 38.0).abs() < 1e-9);
    }
    #[test]
    fn short_segments_use_the_largest_radius_that_fits() {
        let p = chain(&[(0.0, 0.0), (6.0, 0.0), (6.0, 6.0)]);
        let o = smooth_chain(&p, &vec![want(12.0); 3], &[], (0.0, 0.0), 1e-6, &SmoothCfg::default());
        assert_eq!(o.fit_limited, 1);
        let rmin = radii(&o.pts).into_iter().filter(|r| r.0 > 1e-6).map(|r| r.1).fold(f64::INFINITY, f64::min);
        assert!(rmin > 5.9 && rmin < 6.1, "{rmin}");
    }
    #[test]
    fn a_kink_cluster_is_merged() {
        // 0.7 rad within 2.5 m, as three kinks, between long straights: separately ~3.5 m, merged >= the target
        let mut pts = vec![(0.0, 0.0), (40.0, 0.0)];
        let (mut x, mut z, mut h) = (40.0, 0.0, 0.0f64);
        for _ in 0..2 { h += 0.7 / 3.0; x += 1.25 * h.cos(); z += 1.25 * h.sin(); pts.push((x, z)); }
        h += 0.7 / 3.0; pts.push((x + 40.0 * h.cos(), z + 40.0 * h.sin()));
        let p = chain(&pts);
        let o = smooth_chain(&p, &vec![want(12.0); p.len()], &[], (2.0, 2.0), 1e-6, &SmoothCfg::default());
        assert!(o.merges >= 1, "merged");
        let rmin = radii(&o.pts).into_iter().filter(|r| r.0 > 1e-6).map(|r| r.1).fold(f64::INFINITY, f64::min);
        assert!(rmin > 11.5, "radius after merging {rmin}");
    }
    #[test]
    fn straight_ish_moves_little_and_reverse_is_symmetric() {
        let pts: Vec<(f64, f64)> = (0..12).map(|k| (k as f64 * 20.0, if k % 3 == 1 { 0.6 } else { 0.0 })).collect();
        let p = chain(&pts);
        let w = Want { r: 480.0, floor: 150.0, inner_right: 0.0, inner_left: 0.0 };
        let o = smooth_chain(&p, &vec![w; p.len()], &[], (2.0, 2.0), 1e-6, &SmoothCfg::default());
        assert!(o.max_dev <= 1.0 + 1e-9, "dev {}", o.max_dev);
        let rp: Vec<(f64, f64)> = pts.iter().rev().copied().collect();
        let r = smooth_chain(&chain(&rp), &vec![w; p.len()], &[], (2.0, 2.0), 1e-6, &SmoothCfg::default());
        assert_eq!(o.pts.len(), r.pts.len());
        for (a, b) in o.pts.iter().zip(r.pts.iter().rev()) { assert!((a.x - b.x).abs() < 1e-6 && (a.z - b.z).abs() < 1e-6, "{a:?} {b:?}"); }
    }
    #[test]
    fn boundaries_get_a_point() {
        let p = chain(&[(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)]);
        let o = smooth_chain(&p, &vec![want(12.0); 3], &[50.0], (2.0, 2.0), 1e-6, &SmoothCfg::default());
        assert!(o.pts.iter().any(|q| q.s == 50.0 && q.orig.is_none()));
    }
    #[test]
    fn target_table() {
        // residential 25 mph: clamped to its cap; motorway 65 mph: v^2 / (g (0.06 + f))
        assert_eq!(target_radius(11, 11.2), 15.0);
        let r = target_radius(0, 29.06); assert!(r > 400.0 && r < 520.0, "{r}");
        assert!(target_radius(4, 15.6) > 60.0);
    }
}
