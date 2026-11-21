//! sim.js leaders: _findLeaders, _blocking, _laneEnd (laneEnds.js laneEndAhead), _offLaneBodies, _sceneLeader,
//! _opposing, _courtesy, _breakDeadlocks.
use crate::jm;
use crate::sim::{f, r32, Response, Sim, MERGE_ROOM};
use crate::vehicle::*;
use crate::world::World;

pub struct LaneEnd { pub dist: f64, pub side: i32, pub merge_cap: f64 }

/// laneEnds.js laneEndAhead
pub fn lane_end_ahead(g: &World, route: &[i32], rp: usize, s: f64, lane: i32, lookahead: f64, stop_index: i32) -> LaneEnd {
    let mut out = LaneEnd { dist: 1e9, side: 0, merge_cap: 1e9 };
    if stop_index == rp as i32 { return out; }
    let e = route[rp] as usize;
    let mut dist = g.e_length[e] - s;
    let mut prev = e;
    let mut end_k = -1i32;
    let mut my_l = lane;
    let n = g.e_lanes[e] as usize;
    let mut alive = [0u8; 16];
    let mut cur = [0i32; 16];
    for l in 0..n { alive[l] = 1; cur[l] = l as i32; }
    let mut k = 1usize;
    while k <= 4 && rp + k < route.len() && dist < lookahead {
        let en = route[rp + k] as usize;
        let mut any = false;
        for l in 0..n { if alive[l] != 0 { let l2 = g.map_lane(prev, en, cur[l]); if l2 < 0 || l2 >= g.e_lanes[en] { alive[l] = 0; } else { cur[l] = l2; any = true; } } }
        let al = |l: i32| -> bool { l >= 0 && (l as usize) < 16 && alive[l as usize] != 0 };
        if !al(lane) && !any && k > 1 {
            let mut need = 1e9f64;
            for l in 0..g.e_lanes[prev] { let l2 = g.map_lane(prev, en, l); if l2 >= 0 && l2 < g.e_lanes[en] { need = jm::min(need, jm::abs((l - my_l) as f64)); } }
            if need > 0.0 && need < 1e9 { out.merge_cap = jm::max(5.0, jm::max(10.0, g.e_length[prev] - 10.0) / (need * g.lane_w(prev, g.e_length[prev]) / 1.3)); }
            return out;
        }
        if !al(lane) { if any { end_k = k as i32; } break; }
        my_l = g.map_lane(prev, en, my_l);
        if stop_index == (rp + k) as i32 { return out; }
        dist += g.e_length[en]; prev = en;
        k += 1;
    }
    if end_k > 0 {
        let mut best = -1i32;
        for l in 0..n as i32 { if alive[l as usize] != 0 && (best < 0 || (l - lane).abs() < (best - lane).abs()) { best = l; } }
        out.side = if best > lane { 1 } else { -1 };
        out.dist = dist;
    }
    out
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Base { pub leader: i32, pub gap: f32, pub leader2: Option<(i32, f32)>, pub cw: Option<usize>, pub le_dist: f32, pub le_side: i8, pub merge_cap: Option<f32> }

impl Sim {
    #[inline]
    pub(crate) fn blocking(&self, mut a: i32, lane: i32) -> i32 {
        let vs = &self.vs;
        while a >= 0 && vs.lane_hi[a as usize] < vs.lane_lo[a as usize] { a = self.lanes.ahead(vs, a as usize, lane); }
        a
    }

    pub(crate) fn find_leaders(&mut self) {
        let g = self.gw();
        let high = self.vs.high;
        if self.empty {
            for i in 0..high { if self.vs.state[i] == STATE_DRIVING {
                let mut b = Base { leader: -1, gap: 1e9, leader2: None, cw: None, le_dist: 1e9, le_side: 0, merge_cap: None };
                self.lane_end_pure(i, &mut b); self.scene_pure(i, &mut b); self.apply_base(i, &b);
            } }
            return;
        }
        if g.j.is_some() { self.index_connectors(); }
        self.off_lane.reset_lists();
        if self.off_lane.len() > 20000 { self.off_lane.clear(); }
        for i in 0..high {
            let st = self.vs.state[i];
            if (st == STATE_DRIVING || st == STATE_CRASHED) && self.vs.lane_hi[i] < self.vs.lane_lo[i] { self.off_lane.push(self.vs.edge[i], i); }
        }
        self.passers.reset_lists();
        for i in 0..high { if self.vs.pass_wreck[i] >= 0 && self.vs.state[i] == STATE_DRIVING { self.passers.push(self.vs.edge[i], i); } }
        if self.passers.len() > 5000 { self.passers.clear(); }
        // phase 1 (no cross-car writes: lane / cross-edge / zipper leaders, leader2, lane ends, off-lane bodies, scenes),
        // optionally in parallel; phase 2 applies them in slot order and runs the order-dependent perception
        let par = self.opts.parallel && high >= 512;
        let bases: Vec<Option<Base>> = if par {
            use rayon::prelude::*;
            let me = &*self;
            (0..high).into_par_iter().with_min_len(64).map(|i| if me.vs.state[i] == STATE_DRIVING { Some(me.lead_base(i)) } else { None }).collect()
        } else { Vec::new() };
        for i in 0..high {
            if self.vs.state[i] != STATE_DRIVING { continue; }
            let b = if par { bases[i].unwrap() } else { self.lead_base(i) };
            self.apply_base(i, &b);
            let e = self.vs.edge[i] as usize;
            let pw = self.vs.pass_wreck[i];
            let _ = pw;
            if g.j.is_some() { self.junction(i); if self.vs.conn[i] < 0 { self.node_bodies(i, e); } }
            if self.vs.conn[i] < 0 { self.converge_bodies(i, e); }
            if g.e_rev[e] >= 0 { self.opposing(i, e); }
            if self.opts.path_sight >= 1.0 { self.path_bodies(i); }
            let vs = &mut self.vs;
            let ll = vs.leader[i];
            if ll >= 0 && vs.state[ll as usize] == STATE_CRASHED && vs.edge[ll as usize] == vs.edge[i] {
                let lu = ll as usize;
                let hi_ = vs.half_extent(i); let h_l = vs.half_extent(lu);
                let ov = jm::min(f(vs.lat[i]) + hi_, f(vs.lat[lu]) + h_l) - jm::max(f(vs.lat[i]) - hi_, f(vs.lat[lu]) - h_l);
                if ov > 0.0 { vs.gap[i] = r32(f(vs.gap[i]) - jm::min(MERGE_ROOM, 2.0 + 4.0 * (ov + 0.3))); }
            }
        }
        self.courtesy();
        if self.opts.deadlock_break >= 1.0 { self.break_deadlocks(); }
    }

    fn apply_base(&mut self, i: usize, b: &Base) {
        if let Some(j) = b.cw { self.conflict_with.insert(i, j); }
        let vs = &mut self.vs;
        vs.leader[i] = b.leader; vs.gap[i] = b.gap;
        if let Some((l2, g2)) = b.leader2 { vs.leader2[i] = l2; vs.gap2[i] = g2; }
        vs.lane_end_dist[i] = b.le_dist; vs.lane_end_side[i] = b.le_side;
        if let Some(m) = b.merge_cap { vs.merge_cap[i] = m; }
    }

    /// Phase-1 perception of slot i (reads only state that the leaders loop does not change).
    fn lead_base(&self, i: usize) -> Base {
        let g = self.gw();
        let look = self.opts.leader_lookahead;
        let zip_look = self.opts.merge_lookahead;
        let i = i;
        let mut leader = -1i32;
            let mut gap = 1e9f64;
            let vs = &self.vs;
            let lanes = &self.lanes;
            let e = vs.edge[i] as usize;
            let route = &vs.routes[i];
            let rp = vs.route_pos[i] as usize;
            let stop_here = vs.refuel_state[i] == 1 && vs.stop_pos[i] == rp as i32;
            let (lo, hi, si, hl) = (vs.lane_lo[i] as i32, vs.lane_hi[i] as i32, f(vs.s[i]), 0.5 * f(vs.length[i]));
            let mut seen = false;
            let pw = vs.pass_wreck[i];
            let mut l = lo;
            while l <= hi {
                let mut a = self.blocking(lanes.ahead(vs, i, l), l);
                while pw >= 0 && a >= 0 && vs.state[a as usize] == STATE_CRASHED && f(vs.s[a as usize]) <= f(vs.s[pw as usize]) + 0.1 { a = self.blocking(lanes.ahead(vs, a as usize, l), l); }
                if a >= 0 { seen = true; let gg = f(vs.s[a as usize]) - si - hl - 0.5 * f(vs.length[a as usize]); if gg < gap { gap = gg; leader = a; } }
                l += 1;
            }
            if !seen && !stop_here {
                let mut dist = g.e_length[e] - si;
                let mut prev = e;
                let (mut plo, mut phi) = (lo, hi);
                let mut s0 = 0.0;
                if let Some(j) = &g.j {
                    if rp + 1 < route.len() {
                        let mut m = vs.conn[i];
                        if m < 0 && g.e_s_stop[e] < g.e_length[e] { m = j.movement(e, vs.lane[i] as i32, route[rp + 1]); }
                        if m >= 0 { let m = m as usize; dist = j.s_in(m) + j.m_len[m] - si; s0 = j.s_out(m); }
                    }
                }
                let mut k = 1usize;
                while k <= 3 && rp + k < route.len() && dist < look {
                    let en = route[rp + k] as usize;
                    let nlo = g.map_lane(prev, en, plo);
                    let nhi = g.map_lane(prev, en, phi);
                    let mut found = false;
                    let mut l2 = 0.max(nlo);
                    while l2 <= (g.e_lanes[en] - 1).min(nhi) {
                        let fr = self.blocking(lanes.first(en as i32, l2), l2);
                        if fr >= 0 { found = true; let gg = dist + f(vs.s[fr as usize]) - (if k == 1 { s0 } else { 0.0 }) - hl - 0.5 * f(vs.length[fr as usize]); if gg < gap { gap = gg; leader = fr; } }
                        l2 += 1;
                    }
                    if found || nhi < 0 || nlo >= g.e_lanes[en] { break; }
                    plo = 0.max(nlo); phi = (g.e_lanes[en] - 1).min(nhi);
                    dist += g.e_length[en] - if k == 1 { s0 } else { 0.0 }; prev = en;
                    k += 1;
                }
            }
            // zipper merge
            let d_node = g.e_length[e] - si;
            let mut cw_set: Option<usize> = None;
            if rp + 1 < route.len() && d_node < jm::max(zip_look, 7.0 * f(vs.speed[i])) && !stop_here {
                let en = route[rp + 1] as usize;
                let my_lane = g.map_lane(e, en, vs.lane[i] as i32);
                let node = g.e_to[e] as usize;
                for q in g.in_start[node]..g.in_start[node + 1] {
                    let e2 = g.in_edges[q as usize] as usize;
                    if e2 == e { continue; }
                    for l in 0..g.e_lanes[e2] {
                        let w = lanes.last(e2 as i32, l);
                        if w >= 0 { let wu = w as usize; if vs.state[wu] == STATE_CRASHED && vs.lane_hi[wu] >= vs.lane_lo[wu] && g.e_length[e2] - f(vs.s[wu]) < 15.0 {
                            let gw = d_node - 0.5 * f(vs.length[i]) - 3.0; if gw < gap { gap = gw; leader = w; } } }
                        let ramp_merge = (g.e_gore[e] >= 0.0 && !(g.e_gore[e2] >= 0.0) && l == 0) || (g.e_gore[e2] >= 0.0 && !(g.e_gore[e] >= 0.0) && vs.lane[i] == 0);
                        if !ramp_merge && g.map_lane(e2, en, l) != my_lane { continue; }
                        let j = lanes.last(e2 as i32, l);
                        if j < 0 || vs.state[j as usize] != STATE_DRIVING { continue; }
                        let ju = j as usize;
                        let rj = &vs.routes[ju];
                        let pj = vs.route_pos[ju] as usize;
                        if pj + 1 >= rj.len() || rj[pj + 1] != en as i32 { continue; }
                        if vs.refuel_state[ju] == 1 && vs.stop_pos[ju] == pj as i32 { continue; }
                        if (self.signed(ju) && !self.signed(i)) || self.must_stop(ju) { continue; }
                        if g.e_gore[e2] >= 0.0 && g.e_gore[e] < 0.0 && f(vs.s[ju]) + f(vs.hl[ju]) < g.e_gore[e2] + 0.5 { continue; }
                        let dj = g.e_length[e2] - f(vs.s[ju]);
                        if self.signed(i) && !self.signed(ju) && vs.ctl_dist[i] > -0.5 && vs.ctl_dist[i] < 40.0 {
                            if dj / jm::max(f(vs.speed[ju]), 2.0) < 5.0 { let gl = f(vs.ctl_dist[i]) - 0.5; if gl < gap { gap = gl; leader = -3; cw_set = Some(ju); } }
                            continue;
                        }
                        if dj < d_node || (dj == d_node && ju < i) {
                            let gz = d_node - dj - hl - 0.5 * f(vs.length[ju]);
                            let gore = g.e_gore[e];
                            if gore >= 0.0 && g.e_lanes[e2] >= g.e_lanes[e] && si + hl < gore + 0.5 && gz < 2.0 + 1.0 * f(vs.speed[i]) {
                                let gg = gore - si - hl - 0.5; if gg < gap { gap = gg; leader = -3; }
                            } else if gz < gap { gap = gz; leader = j; }
                        }
                    }
                }
            }
        let vs = &self.vs;
        let mut l2v = (-1i32, 1e9f32);
        if leader >= 0 { let lu = leader as usize; if vs.edge[lu] == vs.edge[i] && vs.lane_hi[lu] >= vs.lane_lo[lu] {
            let ll = vs.lane[lu] as i32;
            let l2 = self.blocking(self.lanes.ahead(vs, lu, ll), ll);
            if l2 >= 0 && l2 as usize != i { l2v = (l2, r32(f(vs.s[l2 as usize]) - f(vs.s[i]) - hl - 0.5 * f(vs.length[l2 as usize]))); }
        } }
        let mut b = Base { leader, gap: r32(gap), leader2: Some(l2v), cw: cw_set, le_dist: 1e9, le_side: 0, merge_cap: None };
        self.lane_end_pure(i, &mut b);
        self.off_lane_pure(i, &mut b);
        if pw < 0 { self.scene_pure(i, &mut b); }
        b
    }

    fn courtesy(&mut self) {
        let g = self.gw();
        for k in 0..self.vs.high {
            let vs = &self.vs;
            if vs.state[k] != STATE_DRIVING || vs.conn[k] >= 0 || vs.speed[k] > 6.0 { continue; }
            let mut side = 0i32;
            if vs.lane_end_side[k] != 0 && vs.lane_end_dist[k] < 80.0 { side = vs.lane_end_side[k] as i32; }
            else if vs.target_lane[k] >= 0 && vs.target_lane[k] != vs.lane[k] {
                let l = vs.leader[k];
                if (l >= 0 && vs.state[l as usize] == STATE_CRASHED) || l == -2 || vs.gap[k] < 15.0 { side = if vs.target_lane[k] > vs.lane[k] { 1 } else { -1 }; }
            }
            if side == 0 { continue; }
            let e = vs.edge[k];
            let tl = vs.lane[k] as i32 + side;
            if tl < 0 || tl >= g.e_lanes[e as usize] { continue; }
            let (_, b) = self.lanes.neighbors_at(vs, e, tl, f(vs.s[k]), k as i32);
            if b < 0 { continue; }
            let b = b as usize;
            if vs.state[b] != STATE_DRIVING || vs.conn[b] >= 0 { continue; }
            let gg = f(vs.s[k]) - f(vs.s[b]) - f(vs.hl[k]) - f(vs.hl[b]);
            if !(gg >= 0.5) || gg > 40.0 { if gg < 0.5 || gg > 40.0 { continue; } }
            if gg < f(self.vs.gap[b]) { self.vs.gap[b] = r32(gg); self.vs.leader[b] = k as i32; }
        }
    }

    fn break_deadlocks(&mut self) {
        let cap = self.vs.capacity;
        if self.creep_until.len() < cap { self.creep_until.resize(cap, 0.0); self.creep_ignore.resize(cap, -1); }
        let now = self.time;
        let waits = |s: &Sim, k: usize| -> i32 {
            let l = s.vs.leader[k];
            if l >= 0 { return l; }
            if l == -3 { if let Some(&j) = s.conflict_with.get(&k) { return j as i32; } }
            match &s.yield_by { Some(yb) if yb.len() > k && yb[k] >= 0 => yb[k], _ => -1 }
        };
        let still = |s: &Sim, k: usize| -> bool { let vs = &s.vs; vs.state[k] == STATE_DRIVING && vs.speed[k] < 0.3 && vs.stopped_since[k] >= 0.0 && now - f(vs.stopped_since[k]) > 20.0 };
        let mut seen: Vec<i32> = Vec::new();
        for i in 0..self.vs.high {
            if self.vs.state[i] != STATE_DRIVING || self.vs.flags[i] & FLAG_BRAIN != 0 { continue; }
            if self.creep_until[i] > now {
                let w = waits(self, i);
                if w == self.creep_ignore[i] && self.vs.state[w as usize] != STATE_CRASHED { self.vs.leader[i] = -3; self.vs.gap[i] = 3.0; self.conflict_with.remove(&i); }
                continue;
            }
            if !still(self, i) { continue; }
            seen.clear();
            let (mut k, mut lo, mut ok) = (i, i, false);
            for _ in 0..10 {
                let w = waits(self, k);
                if w < 0 || !still(self, w as usize) { break; }
                if w as usize == i { ok = true; break; }
                if seen.contains(&w) { break; }
                seen.push(w); if (w as usize) < lo { lo = w as usize; } k = w as usize;
            }
            if ok && lo == i {
                self.creep_until[i] = now + 10.0; self.creep_ignore[i] = waits(self, i);
                self.vs.leader[i] = -3; self.vs.gap[i] = 3.0; self.conflict_with.remove(&i);
                self.counters.deadlock_breaks += 1;
            }
        }
    }

    pub(crate) fn lane_end_pure(&self, i: usize, b: &mut Base) {
        let g = self.gw();
        let vs = &self.vs;
        let rp = vs.route_pos[i] as usize;
        b.le_dist = 1e9; b.le_side = 0;
        let e = vs.edge[i] as usize;
        let w = g.lane_w(e, f(vs.s[i]));
        let lat = f(vs.lat[i]);
        let mut d_end = -1.0;
        let mut side = 0i32;
        let m = vs.conn[i];
        let stop_index = if vs.refuel_state[i] == 1 { vs.stop_pos[i] } else { -1 };
        if m >= 0 && g.j.is_some() {
            let j = g.j.as_ref().unwrap();
            let m = m as usize;
            let rest = j.s_in(m) + j.m_len[m] - f(vs.s[i]);
            let en = j.info(m, 2) as usize;
            let ol = jm::min((g.e_lanes[en] - 1) as f64, jm::max(0.0, g.map_lane(e, en, vs.lane[i] as i32) as f64)) as i32;
            let o = lane_end_ahead(&g, &vs.routes[i], rp + 1, j.s_out(m), ol, self.opts.lane_end_lookahead, stop_index);
            b.merge_cap = Some(r32(o.merge_cap));
            if o.side == 0 { return; }
            b.le_dist = r32(rest + o.dist); b.le_side = o.side as i8;
            return;
        }
        if vs.taper[i] > 0.0 && (lat < 0.0 || lat > g.e_lanes[e] as f64 * w) { d_end = f(vs.taper[i]) - f(vs.s[i]); side = if lat < 0.0 { 1 } else { -1 }; }
        else {
            let o = lane_end_ahead(&g, &vs.routes[i], rp, f(vs.s[i]), vs.lane[i] as i32, self.opts.lane_end_lookahead, stop_index);
            b.merge_cap = Some(r32(o.merge_cap));
            if o.side != 0 { side = o.side; d_end = if g.lg.is_some() { o.dist } else { o.dist + TAPER }; }
        }
        if d_end < 0.0 { return; }
        b.le_dist = r32(d_end); b.le_side = side as i8;
        let sidef = side as f64;
        let merging = (vs.target_lane[i] >= 0 && vs.target_lane[i] != vs.lane[i] && vs.flags[i] & FLAG_BRAIN == 0) || f(vs.heading_rel[i]) * sidef > 0.04;
        let at_node = d_end; // prevEndNode: + LANE_END_TAPER (0) with lane geometry, + 0 otherwise
        let into = if side > 0 { vs.lane_hi[i] > vs.lane[i] } else { vs.lane_lo[i] < vs.lane[i] };
        let gg = (if merging { if into && g.lg.is_some() { d_end + 30.0 } else { at_node } } else { d_end - MERGE_ROOM }) - 0.5 * f(vs.length[i]) - 1.0;
        if gg < f(b.gap) { b.gap = r32(gg); b.leader = -2; }
    }

    fn off_lane_pure(&self, i: usize, b: &mut Base) {
        let g = &*self.g;
        let vs = &self.vs;
        let e = vs.edge[i];
        let he = vs.half_extent(i);
        let lat = f(vs.lat[i]);
        let w = g.lane_w(e as usize, f(vs.s[i]));
        if lat - he > 0.05 && lat + he < g.e_lanes[e as usize] as f64 * w - 0.05 { return; }
        let Some(list) = self.off_lane.get(&e) else { return };
        let mut best: Option<(f64, i32)> = None;
        let mut cur = f(b.gap);
        for &j in list {
            if j == i || vs.s[j] <= vs.s[i] { continue; }
            if jm::abs(f(vs.lat[j]) - lat) > he + vs.half_extent(j) + 0.3 { continue; }
            let gg = f(vs.s[j]) - f(vs.s[i]) - f(vs.hl[i]) - f(vs.hl[j]);
            if gg < cur { cur = f(r32(gg)); best = Some((gg, j as i32)); }
        }
        if let Some((gg, j)) = best { b.gap = r32(gg); b.leader = j; }
    }

    pub(crate) fn scene_pure(&self, i: usize, b: &mut Base) {
        let e = self.vs.edge[i];
        if self.edge_has_crash[e as usize] == 0 { return; }
        let Some(recs) = self.edge_crashes.get(&e) else { return };
        let vs = &self.vs;
        let mut cur = f(b.gap);
        let mut best: Option<(f64, i32)> = None;
        for &ri in recs {
            let r = &self.crashes[ri];
            if r.response != Response::Active || r.squeeze || r.s <= f(vs.s[i]) { continue; }
            let mut hit = false;
            let mut l = vs.lane_lo[i] as i32;
            while l <= vs.lane_hi[i] as i32 { if (r.lane_mask >> l) & 1 != 0 { hit = true; } l += 1; }
            if !hit { continue; }
            let gg = r.s - f(vs.s[i]) - 0.5 * f(vs.length[i]) - 3.0;
            if gg < cur { cur = f(r32(gg)); best = Some((gg, r.vehicles[0] as i32)); }
        }
        if let Some((gg, j)) = best { b.gap = r32(gg); b.leader = j; }
    }

    fn opposing(&mut self, i: usize, e: usize) {
        let g = self.gw();
        let er = g.e_rev[e] as usize;
        let li = g.e_lanes[er] - 1;
        let hl = f(self.vs.hl[i]);
        let si = f(self.vs.s[i]);
        if let Some(opp) = self.passers.get(&(er as i32)) {
            if !opp.is_empty() && self.vs.lane_hi[i] as i32 == g.e_lanes[e] - 1 {
                for &p in opp {
                    let sp = g.e_length[e] - f(self.vs.s[p]);
                    if sp > si { let gg = sp - si - hl - f(self.vs.hl[p]) - 2.0; if gg < f(self.vs.gap[i]) { self.vs.gap[i] = r32(gg); self.vs.leader[i] = -3; } }
                }
            }
        }
        let pw = self.vs.pass_wreck[i];
        if pw >= 0 {
            let pw = pw as usize;
            let vs = &self.vs;
            let w = g.lane_w(e, si);
            let back = jm::abs(f(vs.lat[i]) - (vs.lane[i] as f64 + 0.5) * w) < 0.6 && vs.lane_hi[i] >= vs.lane_lo[i];
            if vs.state[pw] != STATE_CRASHED || vs.edge[pw] != e as i32 || (si - hl > f(vs.s[pw]) + f(vs.hl[pw]) + 2.0 && back) { self.vs.pass_wreck[i] = -1; return; }
            let (a, b) = self.lanes.range(er as i32, li);
            for k in a..b {
                let j = self.lanes.order[k] as usize;
                let sj = g.e_length[e] - f(self.vs.s[j]);
                if sj > si { let gg = sj - si - hl - f(self.vs.hl[j]) - 2.0; if gg < f(self.vs.gap[i]) { self.vs.gap[i] = r32(gg); self.vs.leader[i] = -3; } }
            }
            return;
        }
        let vs = &self.vs;
        let l = vs.leader[i];
        if !self.opts.pass_wrecks { return; }
        if g.e_lanes[e] != 1 || g.e_lanes[er] != 1 || l < 0 || vs.state[l as usize] != STATE_CRASHED || vs.edge[l as usize] != e as i32 || vs.conn[i] >= 0 { return; }
        if vs.gap[i] > 15.0 || vs.speed[i] > 1.0 || vs.stopped_since[i] < 0.0 || self.time - f(vs.stopped_since[i]) < 5.0 { return; }
        let mut k = l as usize;
        for _ in 0..6 {
            let lane = if vs.lane[k] >= 0 { (vs.lane_hi[k] as i32).min((vs.lane_lo[k] as i32).max(0)) } else { 0 };
            let q = self.lanes.ahead(vs, k, lane);
            if q < 0 || vs.state[q as usize] != STATE_CRASHED || f(vs.s[q as usize]) - f(vs.s[k]) > 15.0 { break; }
            k = q as usize;
        }
        let zone_end = f(vs.s[k]) + f(vs.hl[k]) + 8.0;
        let zone_start = si - hl - 5.0;
        let limit = if g.e_s_stop[e] < g.e_length[e] { g.e_s_stop[e] } else { g.e_length[e] };
        if zone_end + 10.0 > limit { return; }
        let tp = (zone_end - zone_start) / 4.0 + 5.0;
        let (a, b) = self.lanes.range(er as i32, li);
        for q in a..b {
            let j = self.lanes.order[q] as usize;
            let sj = g.e_length[e] - f(vs.s[j]);
            if sj > zone_start && sj - zone_end < f(vs.speed[j]) * tp + 25.0 { return; }
        }
        if let Some(mine) = self.passers.get(&(e as i32)) { for &p in mine { if p != i && f(vs.s[p]) > zone_start - 10.0 { return; } } }
        self.vs.pass_wreck[i] = k as i32; self.counters.wreck_passes += 1;
        self.passers.push(e as i32, i);
    }
}
