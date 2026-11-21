//! Vehicle-feasible junction turns (`junction_radii`, rust/DECISIONS.md 34): turning connectors re-fitted as one
//! circular arc of a target radius by road class, with s_in moved back on the in-edge and s_out moved forward on the
//! out-edge until the arc fits with tangent entry and exit. Applied at load time after [`crate::smooth`].
//!
//! The JS builder (and `smooth`) already fit the largest arc between the connector's fixed lane points: for a turn
//! Θ with tangent legs t0, t1 (lane point to the intersection of the lane headings), no curve has a smaller maximum
//! curvature than R = min(t0, t1) / tan(Θ / 2). So a wider turn needs longer legs, i.e. other s_in / s_out.
//!
//! Kept consistent: the movement's s_in / s_out / length (`mF`), its points (`pts`, `mInfo` firstPt / ptCount, s =
//! true metres along), the conflict positions (`confS`, recomputed on the new paths as the builder does, else scaled
//! by the length ratio). Unchanged: mouth_s / gore_s (so edge s_exit / s_stop, stop lines and controls), turn
//! angles, the conflict list, hop maps. A movement whose s_in now lies before its in-edge's stop line is a state the
//! builder already produces (its corner fit pulls s_in back up to 40 m); the sim's junction logic uses
//! min(s_stop, s_in) and the controls keep measuring the in-edge s while a car is on the connector.
use crate::junctions::JunctionSet;
use crate::lanegraph::LaneGraph;
use crate::lanes::LaneGeometry;
use rayon::prelude::*;

/// Junction-radius parameters (defaults: DECISIONS.md 34).
#[derive(Clone, Debug)]
pub struct RadiiCfg {
    /// target radius (lane centre, m) by the lower of the two edges' classes: local (unclassified, residential,
    /// living_street), arterial (primary .. tertiary_link), freeway system (motorway, trunk and their links)
    pub r_local: f64, pub r_arterial: f64, pub r_freeway: f64,
    /// a rebuilt connector never goes below this (cars' full-lock radius is 4.3 m sedan / 4.7 m SUV)
    pub r_floor: f64,
    /// largest move of s_in back / s_out forward (m)
    pub max_move: f64,
    /// turns handled: |turn| in [min_turn, max_turn] (U-turns keep their Dubins path)
    pub min_turn: f64, pub max_turn: f64,
    /// straight run kept at each end of the arc (m)
    pub lead: f64,
    pub arc_step: f64,
}
impl Default for RadiiCfg {
    fn default() -> Self { RadiiCfg { r_local: 8.0, r_arterial: 12.0, r_freeway: 15.0, r_floor: 6.0, max_move: 30.0, min_turn: 0.3, max_turn: 2.6, lead: 0.3, arc_step: 0.1 } }
}

/// Statistics of one run.
#[derive(Clone, Debug, Default)]
pub struct RadiiStats {
    /// movements turning within [min_turn, max_turn]
    pub turning: usize,
    /// of those: already at or above the target (left as they were)
    pub already: usize,
    /// rebuilt with moved s_in / s_out and the full target radius
    pub at_target: usize,
    /// rebuilt below the target (largest radius that fits), and why the move stopped: another connector's s_in /
    /// s_out on the same lane, the edge's other end (its mouth or its length), the max_move cap
    pub fit_limited: usize, pub by_lane_order: usize, pub by_edge: usize, pub by_cap: usize,
    /// not rebuilt: no radius larger than the current one fits (or degenerate geometry)
    pub not_improved: usize,
    /// turning connectors whose arc radius (of the leg fit) is below 6 m / 11.5 m (heavy vehicles' full lock) after
    pub below_floor: usize, pub below_heavy: usize,
    /// ... and before
    pub below_floor_before: usize, pub below_heavy_before: usize,
    /// conflict positions recomputed on the new paths / scaled by the length ratio
    pub conf_recomputed: usize, pub conf_scaled: usize,
    /// raw state the sim already handles: movements whose s_in lies before their in-edge's stop line (before / after)
    pub s_in_before_stop: (usize, usize),
    pub max_move_in: f64, pub max_move_out: f64,
}

fn class_target(cfg: &RadiiCfg, cls: u8) -> f64 {
    match cls { 0..=3 => cfg.r_freeway, 4..=9 => cfg.r_arterial, _ => cfg.r_local }
}

#[derive(Clone, Copy)]
struct End { x: f64, y: f64, z: f64, hx: f64, hz: f64 }
fn lane_end(lg: &LaneGeometry, e: usize, s: f64, lane: f64) -> End {
    let p = lg.lat_point(e, s, (lane + 0.5) * lg.lane_width(e, s));
    End { x: p.x, y: p.y, z: p.z, hx: p.hx, hz: p.hz }
}
/// signed turn and tangent legs (t0 from a to the heading intersection, t1 from it to b); None when parallel
fn legs(a: &End, b: &End) -> Option<(f64, f64, f64)> {
    let cr = a.hx * b.hz - a.hz * b.hx; let dt = a.hx * b.hx + a.hz * b.hz;
    if cr.abs() < 1e-6 { return None; }
    let t0 = ((b.x - a.x) * b.hz - (b.z - a.z) * b.hx) / cr;
    let t1 = -((b.x - a.x) * a.hz - (b.z - a.z) * a.hx) / cr;
    Some((cr.atan2(dt), t0, t1))
}

/// Re-fit the turning connectors of `j` on lane geometry `lg` (normally the smoothed one).
pub fn widen_junctions(lg: &LaneGeometry, g: &LaneGraph, j: &JunctionSet, cfg: &RadiiCfg) -> (JunctionSet, RadiiStats) {
    let mc = j.move_count();
    let mut st = RadiiStats::default();
    let fit_r = |t0: f64, t1: f64, turn: f64| (t0.min(t1) - cfg.lead).max(0.0) / (0.5 * turn.abs()).tan();
    // per (in-edge, in-lane): the movements' s_in; per (out-edge, out-lane): their s_out
    use std::collections::HashMap;
    let mut by_in: HashMap<(u32, u32), Vec<f64>> = HashMap::new();
    let mut by_out: HashMap<(u32, u32), Vec<f64>> = HashMap::new();
    for m in 0..mc {
        let d = j.describe(m);
        by_in.entry((d.in_edge, d.in_lane)).or_default().push(d.s_in as f64);
        by_out.entry((d.out_edge, d.out_lane)).or_default().push(d.s_out as f64);
    }
    for m in 0..mc { let d = j.describe(m); if d.s_in + 0.5 < j.mouth_s[2 * d.in_edge as usize + 1] { st.s_in_before_stop.0 += 1; } }
    // result per movement: None = unchanged, Some((s_in, s_out, pts, why)); why: 0 target, 1 lane order, 2 edge, 3 cap
    #[derive(Clone)] enum R { Skip, Already(f64), NotImproved(f64), Done { s_in: f64, s_out: f64, pts: Vec<[f32; 4]>, r: f64, r0: f64, why: u8 } }
    let res: Vec<R> = (0..mc).into_par_iter().map(|m| {
        let d = j.describe(m);
        let (a, b, la, lb) = (d.in_edge as usize, d.out_edge as usize, d.in_lane as f64, d.out_lane as f64);
        let (s_in0, s_out0) = (d.s_in as f64, d.s_out as f64);
        let (pa, pb) = (lane_end(lg, a, s_in0, la), lane_end(lg, b, s_out0, lb));
        let Some((turn, t0, t1)) = legs(&pa, &pb) else { return R::Skip };
        if turn.abs() < cfg.min_turn || turn.abs() > cfg.max_turn || t0 < 0.0 || t1 < 0.0 { return R::Skip; }
        let target = class_target(cfg, g.cls[a].min(g.cls[b])).max(cfg.r_floor);
        let r0 = fit_r(t0, t1, turn);
        if r0 >= target - 1e-6 { return R::Already(r0); }
        // how far the ends may move
        let (len_a, len_b) = (g.length[a] as f64, g.length[b] as f64);
        let sx_a = j.mouth_s[2 * a] as f64; let ss_b = j.mouth_s[2 * b + 1] as f64;
        let edge_lo = if sx_a > 0.0 { sx_a + 1.0 } else { 0.5 }.min(s_in0);
        let edge_hi = if ss_b < len_b { ss_b - 1.0 } else { len_b - 0.5 }.max(s_out0);
        let order_lo = by_in[&(d.in_edge, d.in_lane)].iter().copied().filter(|&s| s < s_in0 - 1e-6).fold(f64::NEG_INFINITY, f64::max);
        let order_hi = by_out[&(d.out_edge, d.out_lane)].iter().copied().filter(|&s| s > s_out0 + 1e-6).fold(f64::INFINITY, f64::min);
        let cap_lo = s_in0 - cfg.max_move; let cap_hi = s_out0 + cfg.max_move;
        let lo = edge_lo.max(order_lo).max(cap_lo); let hi = edge_hi.min(order_hi).min(cap_hi);
        let why_lo = if lo == order_lo { 1u8 } else if lo == edge_lo { 2 } else { 3 };
        let why_hi = if hi == order_hi { 1u8 } else if hi == edge_hi { 2 } else { 3 };
        let (mut s_in, mut s_out) = (s_in0, s_out0);
        let (mut ea, mut eb, mut tr, mut l0, mut l1) = (pa, pb, turn, t0, t1);
        let (mut stuck_in, mut stuck_out) = (false, false);
        for _ in 0..6 {
            let need = target * (0.5 * tr.abs()).tan() + cfg.lead;
            let (din, dout) = ((need - l0).max(0.0), (need - l1).max(0.0));
            if din < 1e-3 && dout < 1e-3 { break; }
            let (ni, no) = ((s_in - 1.1 * din).max(lo), (s_out + 1.1 * dout).min(hi));
            stuck_in = din > 1e-3 && ni <= lo + 1e-9; stuck_out = dout > 1e-3 && no >= hi - 1e-9;
            if (ni - s_in).abs() < 1e-4 && (no - s_out).abs() < 1e-4 { break; }
            let (na, nb) = (lane_end(lg, a, ni, la), lane_end(lg, b, no, lb));
            let Some((t, q0, q1)) = legs(&na, &nb) else { break };
            if q0 < 0.0 || q1 < 0.0 || t.signum() != turn.signum() { break; }
            s_in = ni; s_out = no; ea = na; eb = nb; tr = t; l0 = q0; l1 = q1;
        }
        let r = fit_r(l0, l1, tr).min(target);
        if r <= r0 + 1e-3 { return R::NotImproved(r0); }
        let why = if r >= target - 1e-6 { 0 } else if stuck_in && (!stuck_out || l0 <= l1) { why_lo } else if stuck_out { why_hi } else { 3 };
        // the path: lead from ea, arc of radius r, lead to eb
        let tn = (0.5 * tr.abs()).tan(); let at = r * tn;
        let (vx, vz) = (ea.x + ea.hx * l0, ea.z + ea.hz * l0);
        let (t1x, t1z) = (vx - ea.hx * at, vz - ea.hz * at);
        let sg = tr.signum();
        let (cx, cz) = (t1x - sg * r * ea.hz, t1z + sg * r * ea.hx);
        let (rx, rz) = (t1x - cx, t1z - cz);
        let mut xz: Vec<(f64, f64)> = vec![(ea.x, ea.z)];
        let line = |xz: &mut Vec<(f64, f64)>, x0: f64, z0: f64, x1: f64, z1: f64| { let l = (x1 - x0).hypot(z1 - z0); let n = (l / 2.0).ceil().max(1.0) as usize; for k in 1..=n { let f = k as f64 / n as f64; xz.push((x0 + (x1 - x0) * f, z0 + (z1 - z0) * f)); } };
        if l0 - at > 0.02 { line(&mut xz, ea.x, ea.z, t1x, t1z); }
        let ns = (tr.abs() / cfg.arc_step).ceil().max(2.0) as usize;
        for k in 1..=ns { let ph = sg * tr.abs() * k as f64 / ns as f64; let (c, s) = (ph.cos(), ph.sin()); xz.push((cx + rx * c - rz * s, cz + rx * s + rz * c)); }
        let (t2x, t2z) = *xz.last().unwrap();
        if (eb.x - t2x).hypot(eb.z - t2z) > 0.02 { line(&mut xz, t2x, t2z, eb.x, eb.z); } else { *xz.last_mut().unwrap() = (eb.x, eb.z); }
        let mut cum = vec![0.0f64]; for k in 1..xz.len() { cum.push(cum[k - 1] + (xz[k].0 - xz[k - 1].0).hypot(xz[k].1 - xz[k - 1].1)); }
        let tot = *cum.last().unwrap();
        let pts: Vec<[f32; 4]> = xz.iter().zip(&cum).map(|(&(x, z), &c)| [x as f32, (ea.y + (eb.y - ea.y) * c / tot.max(1e-9)) as f32, z as f32, c as f32]).collect();
        R::Done { s_in, s_out, pts, r, r0, why }
    }).collect();
    // assemble
    let mut m_info = j.m_info.to_vec(); let mut m_f = j.m_f.to_vec();
    let mut pts: Vec<f32> = Vec::with_capacity(j.pts.len() + mc * 8);
    let mut changed = vec![false; mc]; let mut ratio = vec![1.0f64; mc];
    for (m, r) in res.into_iter().enumerate() {
        let (first, n) = (j.m_info[7 * m + 4] as usize, j.m_info[7 * m + 5] as usize);
        let rr = match &r { R::Skip => None, R::Already(x) | R::NotImproved(x) => Some((*x, *x)), R::Done { r, r0, .. } => Some((*r0, *r)) };
        if let Some((before, after)) = rr {
            st.turning += 1;
            if before < cfg.r_floor { st.below_floor_before += 1; } if before < 11.5 { st.below_heavy_before += 1; }
            if after < cfg.r_floor { st.below_floor += 1; } if after < 11.5 { st.below_heavy += 1; }
        }
        match r {
            R::Done { s_in, s_out, pts: p, why, .. } => {
                let old_len = m_f[4 * m + 2] as f64;
                st.max_move_in = st.max_move_in.max(m_f[4 * m] as f64 - s_in); st.max_move_out = st.max_move_out.max(s_out - m_f[4 * m + 1] as f64);
                m_info[7 * m + 4] = (pts.len() / 4) as u32; m_info[7 * m + 5] = p.len() as u32;
                let new_len = p.last().unwrap()[3] as f64;
                m_f[4 * m] = s_in as f32; m_f[4 * m + 1] = s_out as f32; m_f[4 * m + 2] = new_len as f32;
                for q in p { pts.extend_from_slice(&q); }
                changed[m] = true; ratio[m] = new_len / old_len.max(1e-6);
                if why == 0 { st.at_target += 1; } else { st.fit_limited += 1; match why { 1 => st.by_lane_order += 1, 2 => st.by_edge += 1, _ => st.by_cap += 1 } }
            }
            other => {
                match other { R::Already(_) => st.already += 1, R::NotImproved(_) => st.not_improved += 1, _ => {} }
                m_info[7 * m + 4] = (pts.len() / 4) as u32;
                pts.extend_from_slice(&j.pts[4 * first..4 * (first + n)]);
            }
        }
    }
    for m in 0..mc { if m_f[4 * m] + 0.5 < j.mouth_s[2 * m_info[7 * m] as usize + 1] { st.s_in_before_stop.1 += 1; } }
    // conflicts touching a rebuilt movement: recompute on the new paths, else scale
    let mut conf_s = j.conf_s.to_vec();
    let nc = j.conflict_count();
    for c in 0..nc {
        let (a, b) = (j.conf[3 * c] as usize, j.conf[3 * c + 1] as usize);
        if !changed[a] && !changed[b] { continue; }
        match crate::junctions_build::pair_conflict(&pts, &m_info, a, b) {
            Some((_, sa, sb)) => { conf_s[2 * c] = sa as f32; conf_s[2 * c + 1] = sb as f32; st.conf_recomputed += 1; }
            None => { conf_s[2 * c] = (conf_s[2 * c] as f64 * ratio[a]) as f32; conf_s[2 * c + 1] = (conf_s[2 * c + 1] as f64 * ratio[b]) as f32; st.conf_scaled += 1; }
        }
    }
    (JunctionSet { m_info: m_info.into(), m_f: m_f.into(), pts: pts.into(), conf_s: conf_s.into(), ..j.clone() }, st)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legs_of_a_right_angle() {
        let a = End { x: 0.0, y: 0.0, z: 0.0, hx: 1.0, hz: 0.0 };
        let b = End { x: 5.0, y: 0.0, z: 10.0, hx: 0.0, hz: 1.0 };
        let (t, t0, t1) = legs(&a, &b).unwrap();
        assert!((t - std::f64::consts::FRAC_PI_2).abs() < 1e-12 && (t0 - 5.0).abs() < 1e-12 && (t1 - 10.0).abs() < 1e-12);
    }
    #[test]
    fn class_targets() {
        let c = RadiiCfg::default();
        assert_eq!(class_target(&c, 11), 8.0); assert_eq!(class_target(&c, 4), 12.0); assert_eq!(class_target(&c, 0), 15.0);
    }
}
