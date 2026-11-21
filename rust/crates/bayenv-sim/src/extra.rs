//! sim.js: _controlObs (+ controls.js stateOf / movementOf), _missedExits, _trafficSample, _rerouteBlocked,
//! _sampleDensity, placeWrecks / placeStaticObstacles / placeObstacles (+ scenarios/load.js
//! placeScenarioObstacles), _stuckCause (+ ctx.yieldRequired), fuel / stations.
use crate::jm;
use crate::rng::Rng;
use crate::sim::{f, r32, Driver, Response, Sim, FAIL_STRANDED};
use crate::vehicle::*;
use crate::world::World;
use crate::fx::{HashMap, HashSet};

pub const CTRL_SIGNAL: i8 = 0;
pub const CTRL_STOP: i8 = 1;
pub const CTRL_GIVE_WAY: i8 = 2;
pub const CTRL_RAMP: i8 = 3;
pub const CTRL_CROSSING: i8 = 4;
pub const LIGHT_GREEN: i8 = 0;
pub const LIGHT_YELLOW: i8 = 1;
pub const LIGHT_RED: i8 = 2;
pub const LIGHT_SIGN: i8 = 3;
pub const LIGHT_DARK: i8 = 4;
pub const MOVE_THROUGH: i8 = 0;
pub const MOVE_LEFT: i8 = 1;
pub const MOVE_RIGHT: i8 = 2;

pub fn movement_of(g: &World, e: usize, en: usize) -> i8 {
    let mut d = g.e_head_start[en] - g.e_head_end[e];
    while d > jm::PI { d -= 2.0 * jm::PI; }
    while d < -jm::PI { d += 2.0 * jm::PI; }
    if d < -0.5 { MOVE_LEFT } else if d > 0.5 { MOVE_RIGHT } else { MOVE_THROUGH }
}

/// TrafficControls.stateOf -> (type, state, timeIn, timeLeft)
pub fn state_of(g: &World, a: usize, mv: i8, t: f64) -> (i8, i8, f64, f64) {
    let cc = g.controls.as_ref().unwrap();
    let c = cc.ap_control[a] as usize;
    let ty = cc.ty[c];
    if ty == CTRL_STOP || ty == CTRL_GIVE_WAY { return (ty, LIGHT_SIGN, 0.0, 1e9); }
    let cy = cc.cycle[c];
    let tc = ((t + cc.offset[c]) % cy + cy) % cy;
    if ty == CTRL_RAMP {
        let tod = t % 86400.0;
        if tod < cc.ramp_start || tod > cc.ramp_end { return (ty, LIGHT_DARK, 0.0, 1e9); }
        return if tc < 2.0 { (ty, LIGHT_GREEN, tc, 2.0 - tc) } else { (ty, LIGHT_RED, tc - 2.0, cy - tc) };
    }
    if ty == CTRL_CROSSING {
        return if tc < 75.0 { (ty, LIGHT_GREEN, tc, 75.0 - tc) } else if tc < 78.0 { (ty, LIGHT_YELLOW, tc - 75.0, 78.0 - tc) } else { (ty, LIGHT_RED, tc - 78.0, cy - tc) };
    }
    let ph = &cc.phases[cc.phase_start[c] as usize..cc.phase_start[c + 1] as usize];
    let n = ph.len();
    let grp = cc.ap_group[a] as f64;
    let prot = cc.ap_prot_left[a] == 1;
    let mine = |p: &[f64; 5]| p[0] == grp && (if mv == MOVE_LEFT && prot { p[1] != 0.0 } else { p[1] == 0.0 });
    let len = |p: &[f64; 5]| p[2] + p[3] + p[4];
    let (mut t0, mut cur, mut in_phase) = (0.0, -1i64, 0.0);
    for (k, p) in ph.iter().enumerate() { let l = len(p); if tc < t0 + l { cur = k as i64; in_phase = tc - t0; break; } t0 += l; }
    if cur < 0 { cur = n as i64 - 1; in_phase = 0.0; }
    let cur = cur as usize;
    let p = &ph[cur];
    if mine(p) && in_phase < p[2] { return (ty, LIGHT_GREEN, in_phase, p[2] - in_phase); }
    if mine(p) && in_phase < p[2] + p[3] { return (ty, LIGHT_YELLOW, in_phase - p[2], p[2] + p[3] - in_phase); }
    let mut until = len(p) - in_phase;
    let mut k = (cur + 1) % n;
    let mut guard = 0;
    while !mine(&ph[k]) && { guard += 1; guard - 1 < n } { until += len(&ph[k]); k = (k + 1) % n; }
    let mut since = if mine(p) { in_phase - p[2] - p[3] } else { in_phase };
    k = (cur + n - 1) % n; guard = 0;
    while !mine(&ph[k]) && { guard += 1; guard - 1 < n } { since += len(&ph[k]); k = (k + n - 1) % n; }
    if mine(&ph[k]) && k != cur { since += ph[k][4]; }
    (ty, LIGHT_RED, jm::max(0.0, since), until)
}

#[derive(Clone, Copy)]
struct Ctl { ctl_approach: i32, ctl_type: i8, ctl_state: i8, ctl_dist: f32, ctl_time_in: f32, ctl_time_left: f32, ctl_move: i8, ctl_stopped: u8, ctl_inside: i32, ctl_inside_t: f32, sig: u64, stop: u64 }

impl Sim {
    pub(crate) fn control_obs(&mut self) {
        let high = self.vs.high;
        let par = self.opts.parallel && high >= 512;
        let outs: Vec<Option<Ctl>> = if par {
            use rayon::prelude::*;
            let me = &*self;
            (0..high).into_par_iter().with_min_len(64).map(|i| if me.vs.state[i] == STATE_DRIVING { Some(me.ctl_car(i)) } else { None }).collect()
        } else { (0..high).map(|i| if self.vs.state[i] == STATE_DRIVING { Some(self.ctl_car(i)) } else { None }).collect() };
        for (i, o) in outs.into_iter().enumerate() {
            let Some(o) = o else { continue };
            let vs = &mut self.vs;
            vs.ctl_approach[i] = o.ctl_approach; vs.ctl_type[i] = o.ctl_type; vs.ctl_state[i] = o.ctl_state; vs.ctl_dist[i] = o.ctl_dist; vs.ctl_time_in[i] = o.ctl_time_in;
            vs.ctl_time_left[i] = o.ctl_time_left; vs.ctl_move[i] = o.ctl_move; vs.ctl_stopped[i] = o.ctl_stopped; vs.ctl_inside[i] = o.ctl_inside; vs.ctl_inside_t[i] = o.ctl_inside_t;
            self.counters.signal_violations += o.sig; self.counters.stop_violations += o.stop;
            if o.sig + o.stop > 0 { self.emit(crate::sim::EV_SIGNAL_VIOLATION, i, -1, f32::NAN, f32::NAN, f32::NAN, f32::NAN); }
        }
    }

    /// _controlObs for one car (reads only that car's fields and the static controls).
    fn ctl_car(&self, i: usize) -> Ctl {
        let g = self.gw();
        let cc = g.controls.as_ref().unwrap();
        let look = self.opts.control_lookahead;
        {
            let vs = &self.vs;
            let mut o = Ctl { ctl_approach: vs.ctl_approach[i], ctl_type: vs.ctl_type[i], ctl_state: vs.ctl_state[i], ctl_dist: vs.ctl_dist[i], ctl_time_in: vs.ctl_time_in[i], ctl_time_left: vs.ctl_time_left[i], ctl_move: vs.ctl_move[i], ctl_stopped: vs.ctl_stopped[i], ctl_inside: vs.ctl_inside[i], ctl_inside_t: vs.ctl_inside_t[i], sig: 0, stop: 0 };
            let rp = vs.route_pos[i] as usize;
            let hl = 0.5 * f(vs.length[i]);
            let (mut a, mut dist, mut k) = (-1i32, 0.0f64, rp);
            let e = vs.edge[i] as usize;
            let inside = o.ctl_inside;
            let skip = |x: i32| inside >= 0 && cc.cluster_of[cc.ap_control[x as usize] as usize] == inside;
            let si = f(vs.s[i]);
            let mut ap = -1i32;
            for q in cc.edge_ap_start[e]..cc.edge_ap_start[e + 1] { let x = cc.edge_ap_list[q as usize]; if cc.ap_stop_at[x as usize] - (si + hl) > -1.0 && !skip(x) { ap = x; break; } }
            let route = &vs.routes[i];
            if ap >= 0 { a = ap; dist = cc.ap_stop_at[ap as usize] - (si + hl); }
            else {
                let mut d = g.e_length[e] - si - hl;
                let mut s0 = 0.0;
                if let Some(j) = &g.j { if rp + 1 < route.len() {
                    let mut m = vs.conn[i];
                    if m < 0 && g.e_s_stop[e] < g.e_length[e] { m = j.movement(e, vs.lane[i] as i32, route[rp + 1]); }
                    if m >= 0 { let mu = m as usize; d = j.s_in(mu) + j.m_len[mu] - si - hl; s0 = j.s_out(mu); }
                } }
                k = rp + 1;
                while k < route.len() && d < look && a < 0 {
                    let en = route[k] as usize;
                    let off = if k == rp + 1 { s0 } else { 0.0 };
                    for q in cc.edge_ap_start[en]..cc.edge_ap_start[en + 1] { let x = cc.edge_ap_list[q as usize]; if !skip(x) { a = x; dist = d + cc.ap_stop_at[x as usize] - off; break; } }
                    if a >= 0 { break; }
                    d += g.e_length[en] - off;
                    k += 1;
                }
            }
            let prev_a = o.ctl_approach;
            let prev_dist = f(o.ctl_dist);
            if prev_a >= 0 && prev_dist >= 0.0 && prev_dist < 30.0 && (a != prev_a || dist < 0.0) && !(o.ctl_inside >= 0 && cc.cluster_of[cc.ap_control[prev_a as usize] as usize] == o.ctl_inside) {
                let (st, ty) = (o.ctl_state, o.ctl_type);
                if (ty == CTRL_SIGNAL || ty == CTRL_RAMP || ty == CTRL_CROSSING) && st == LIGHT_RED && o.ctl_time_in > 0.5 && !(o.ctl_move == MOVE_RIGHT && o.ctl_stopped != 0) { o.sig = 1; }
                else if ty == CTRL_STOP && o.ctl_stopped == 0 { o.stop = 1; }
            }
            if a != prev_a {
                o.ctl_stopped = 0;
                if prev_a >= 0 && prev_dist >= -1.0 && prev_dist < 30.0 && cc.ty[cc.ap_control[prev_a as usize] as usize] == CTRL_SIGNAL {
                    o.ctl_inside = cc.cluster_of[cc.ap_control[prev_a as usize] as usize]; o.ctl_inside_t = r32(self.time);
                }
            }
            if o.ctl_inside >= 0 && self.time - f(o.ctl_inside_t) > 15.0 { o.ctl_inside = -1; }
            o.ctl_approach = a;
            if a < 0 { o.ctl_type = -1; o.ctl_dist = 1e9; return o; }
            let app_e = cc.ap_edge[a as usize] as usize;
            let mut mv = MOVE_THROUGH;
            let pos = if app_e as i32 == vs.edge[i] { rp } else { k };
            if pos + 1 < vs.routes[i].len() { mv = movement_of(&g, app_e, vs.routes[i][pos + 1] as usize); }
            let (ty, st, tin, tleft) = state_of(&g, a as usize, mv, self.time);
            o.ctl_type = ty; o.ctl_state = st; o.ctl_time_in = r32(tin); o.ctl_time_left = r32(tleft);
            o.ctl_dist = r32(dist); o.ctl_move = mv;
            if dist < 10.0 && vs.speed[i] < 0.3 { o.ctl_stopped = 1; }
            o
        }
    }

    pub(crate) fn missed_exits(&mut self) {
        let g = self.gw();
        let list = std::mem::take(&mut self.vs.missed);
        let mut keep = Vec::new();
        for (i, m) in list {
            if self.vs.state[i] != STATE_DRIVING { continue; }
            if self.vs.edge[i] != m.took { if self.vs.edge[i] == m.from { keep.push((i, m)); } continue; }
            self.counters.missed_exits += 1;
            self.emit(crate::sim::EV_MISSED_EXIT, i, -1, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
            let took_to = g.e_to[m.took as usize];
            let path = if took_to == m.goal { Some(Vec::new()) } else { self.router.route(took_to, m.goal, None, None) };
            let Some(path) = path else { self.counters.reroute_failed += 1; continue };
            let mut edges = Vec::with_capacity(1 + path.len()); edges.push(m.took); edges.extend_from_slice(&path);
            let s_on = self.vs.s[i];
            self.set_route(i, &edges, 0); self.vs.route_dist[i] = s_on;
            if self.vs.target_lane[i] >= 0 { self.vs.target_lane[i] = self.vs.lane[i]; }
            self.counters.reroutes += 1;
        }
        // entries added while iterating (none: integrate is not called here) keep their order after the kept ones
        let added = std::mem::take(&mut self.vs.missed);
        self.vs.missed = keep;
        for (i, m) in added { if let Some(p) = self.vs.missed.iter_mut().find(|p| p.0 == i) { p.1 = m; } else { self.vs.missed.push((i, m)); } }
    }

    pub(crate) fn traffic_sample(&mut self) {
        let g = self.gw();
        let mut sum: HashMap<i32, f64> = HashMap::default();
        let mut cnt: HashMap<i32, f64> = HashMap::default();
        for i in 0..self.vs.high {
            if self.vs.state[i] != STATE_DRIVING { continue; }
            let e = self.vs.edge[i];
            *sum.entry(e).or_insert(0.0) += f(self.vs.speed[i]);
            *cnt.entry(e).or_insert(0.0) += 1.0;
        }
        let ema = &mut self.edge_speed;
        ema.retain(|e, v| {
            if cnt.contains_key(e) { return true; }
            let lim = g.e_speed[*e as usize];
            let nv = 0.5 * *v + 0.5 * lim;
            if nv > 0.95 * lim { false } else { *v = nv; true }
        });
        for (e, sv) in &sum { let m = sv / cnt[e]; let nv = match ema.get(e) { None => m, Some(&old) => 0.5 * old + 0.5 * m }; ema.insert(*e, nv); }
        let pen = self.traffic_penalty.get_or_insert_with(HashMap::default);
        pen.clear();
        for (&e, &v) in ema.iter() {
            let eu = e as usize;
            if v >= 0.7 * g.e_speed[eu] { continue; }
            pen.insert(e, g.e_length[eu] / jm::max(0.5, v) - g.e_length[eu] / g.e_speed[eu]);
        }
    }

    pub(crate) fn reroute_blocked(&mut self) {
        let g = self.gw();
        let mut closed: HashSet<i32> = HashSet::default();
        for &ri in &self.active {
            let r = &self.crashes[ri];
            if r.response != Response::Active || r.squeeze { continue; }
            let mut all = true;
            for l in 0..g.e_lanes[r.edge as usize] { if (r.lane_mask >> l) & 1 == 0 { all = false; break; } }
            if all { closed.insert(r.edge); }
        }
        let pen = match &self.traffic_penalty { Some(p) if !p.is_empty() => Some(p.clone()), _ => None };
        if closed.is_empty() && pen.is_none() { return; }
        // the Router trait takes std maps (shared with bayenv-gpu); membership tests only, so the hasher is irrelevant
        let closed_std: std::collections::HashSet<i32> = closed.iter().copied().collect();
        let pen_std: Option<std::collections::HashMap<i32, f64>> = pen.as_ref().map(|p| p.iter().map(|(&k, &v)| (k, v)).collect());
        let sight = self.opts.reroute_sight;
        let mut budget = 200;
        let mut i = 0;
        let mut cands: Vec<(usize, i32, i32, bool)> = Vec::new();
        while i < self.vs.high && budget > 0 {
            let vs = &self.vs;
            if vs.state[i] != STATE_DRIVING || vs.conn[i] >= 0 || vs.refuel_state[i] != 0 { i += 1; continue; }
            let rp = vs.route_pos[i] as usize;
            let route = &vs.routes[i];
            if rp + 1 >= route.len() { i += 1; continue; }
            let e = route[rp] as usize;
            let stop_at = if g.e_s_stop[e] < g.e_length[e] { g.e_s_stop[e] } else { g.e_length[e] };
            if f(vs.s[i]) + f(vs.hl[i]) > stop_at - 5.0 { i += 1; continue; }
            let mut d = g.e_length[e] - f(vs.s[i]);
            let mut hit = false;
            let mut k = rp + 1;
            while k < route.len() && d < sight { if closed.contains(&route[k]) { hit = true; break; } d += g.e_length[route[k] as usize]; k += 1; }
            let slow = !hit && pen.is_some() && vs.stopped_since[i] >= 0.0 && self.time - f(vs.stopped_since[i]) > 90.0;
            if !hit && !slow { i += 1; continue; }
            if let Some(&lt) = self.last_reroute.get(&i) { if lt > f(vs.spawn_time[i]) && self.time - lt < (if hit { 120.0 } else { 300.0 }) { i += 1; continue; } }
            self.last_reroute.insert(i, self.time);
            budget -= 1;
            let goal = g.e_to[route[route.len() - 1] as usize];
            if g.e_to[e] == goal { i += 1; continue; }
            cands.push((i, g.e_to[e], goal, slow));
            i += 1;
        }
        // all queries of this pass at once (no query depends on an earlier car's new route: candidacy only reads the
        // car's own state, and avoid / penalty queries never touch the router's cache), applied in slot order
        let queries: Vec<(i32, i32)> = cands.iter().map(|c| (c.1, c.2)).collect();
        let paths = self.router.route_batch(&queries, if closed_std.is_empty() { None } else { Some(&closed_std) }, pen_std.as_ref());
        for (k, &(i, _, _, slow)) in cands.iter().enumerate() {
            let path = paths[k].clone();
            let rp = self.vs.route_pos[i] as usize;
            let route = self.vs.routes[i].clone();
            if slow {
                if let Some(p) = &path {
                    let pm = pen.as_ref().unwrap();
                    let cost = |pp: &[i32]| { let mut t = 0.0; for &x in pp { let xu = x as usize; t += g.e_length[xu] / g.e_speed[xu] + pm.get(&x).copied().unwrap_or(0.0); } t };
                    if cost(p) > cost(&route[rp + 1..]) - 60.0 { continue; }
                }
            }
            let Some(path) = path.filter(|p| !p.is_empty() && !p.iter().any(|x| closed.contains(x))) else { self.counters.reroute_failed += 1; continue };
            let mut edges = Vec::with_capacity(1 + path.len()); edges.push(route[rp]); edges.extend_from_slice(&path);
            let s_on = self.vs.s[i];
            self.set_route(i, &edges, 0);
            self.vs.route_dist[i] = s_on;
            if self.vs.target_lane[i] >= 0 { self.vs.target_lane[i] = self.vs.lane[i]; }
            self.counters.reroutes += 1;
        }
        if self.last_reroute.len() > 100000 { self.last_reroute.clear(); }
    }

    pub(crate) fn sample_density(&mut self) {
        if self.fwy.is_empty() { return; }
        let g = self.gw();
        let cnt = &mut self.edge_count_scratch;
        for i in 0..self.vs.high { let st = self.vs.state[i]; if st == STATE_DRIVING || st == STATE_CRASHED { let e = self.vs.edge[i] as usize; if cnt[e] < 65535 { cnt[e] += 1; } } }
        let (mut cars, mut df, mut peak) = (0.0, 0.0, 0.0);
        for &ei in &self.fwy {
            let lk = g.e_length[ei] * g.e_lanes[ei] as f64 / 1000.0;
            let c = cnt[ei] as f64;
            cars += c;
            let d = c / lk;
            if d > 16.2 { df += lk; }
            if g.e_length[ei] >= 300.0 && d > peak { peak = d; }
        }
        for i in 0..self.vs.high { let e = self.vs.edge[i]; if e >= 0 { cnt[e as usize] = 0; } }
        let mean = cars / self.fwy_lane_km;
        let share = df / self.fwy_lane_km;
        let dd = &mut self.density;
        dd.samples += 1; dd.mean_density_sum += mean; dd.los_df_share_sum += share;
        if mean > dd.peak_mean_density { dd.peak_mean_density = mean; dd.peak_mean_at = self.time; }
        if share > dd.peak_los_df_share { dd.peak_los_df_share = share; dd.peak_los_df_at = self.time; }
        if peak > dd.peak_segment_density { dd.peak_segment_density = peak; }
    }

    // ---------------------------------------------------------------- obstacles
    fn obstacle_alloc(&mut self, type_idx: u8, route: &[i32], lane: i32) -> usize {
        let g = self.gw();
        let s = self.vs.alloc(&g, AllocArgs { type_idx, traits: DEFAULT_TRAITS, route, commuter: -1, leg: 0, time: self.time, learner: false, dt: self.dt, step: self.step_count, lane });
        self.ensure_slot(s);
        self.emit(crate::sim::EV_SPAWNED, s, -1, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        s
    }

    pub fn place_wrecks(&mut self, n: usize) {
        let g = self.gw();
        let cand: Vec<usize> = (0..g.edge_count).filter(|&e| g.e_rank[e] >= 3 && g.e_length[e] > 150.0).collect();
        for _ in 0..n {
            if cand.is_empty() { break; }
            let ei = cand[self.rng.int(cand.len())];
            let slot = self.obstacle_alloc(0, &[ei as i32], 0);
            let vs = &mut self.vs;
            vs.s[slot] = r32(g.e_length[ei] * (0.3 + 0.4 * self.rng.random()));
            let lane = self.rng.int(g.e_lanes[ei] as usize) as f64;
            vs.lat[slot] = r32((lane + 0.5) * g.lane_w(ei, f(vs.s[slot])));
            vs.update_lanes(slot, &g, ei); vs.update_pose(slot, &g); vs.save_pose(slot);
            self.open_scene(&[slot], "wreck", "wreck", 1e9, 1e9, ei, false, None);
        }
    }

    fn static_car(&mut self, type_idx: u8, e: usize, s: f64, lat: f64, conn: i32) -> usize {
        let g = self.gw();
        let slot = self.obstacle_alloc(type_idx, &[e as i32], 0);
        let vs = &mut self.vs;
        vs.s[slot] = r32(s); vs.lat[slot] = r32(lat); vs.conn[slot] = conn; vs.update_lanes(slot, &g, e);
        vs.update_pose(slot, &g); vs.save_pose(slot);
        self.open_scene(&[slot], "obstacle", "wreck", 1e9, 1e9, e, false, None);
        self.vs.flags[slot] |= FLAG_DISABLED;
        slot
    }

    /// placeStaticObstacles([{edge, at, lane | lat, type}]) and scenarios/load.js placeScenarioObstacles (scenario = true:
    /// s clamped to [1, length - 1], lane width from the lane geometry at s).
    pub fn place_static_obstacles(&mut self, list: &[(i32, f64, Option<i32>, Option<f64>, u8)], scenario: bool) -> Vec<usize> {
        let g = self.gw();
        let mut out = Vec::new();
        for &(edge, at, lane, lat, ty) in list {
            if edge < 0 || edge as usize >= g.edge_count { continue; }
            let e = edge as usize;
            let s = if scenario { jm::min(g.e_length[e] - 1.0, jm::max(1.0, at)) } else { jm::max(0.0, jm::min(g.e_length[e], at)) };
            let lw = if scenario { match &g.lg { Some(lg) => lg.lane_width(e, s), None => 3.7 } } else { g.lane_w(e, s) };
            let lat = lat.unwrap_or((lane.unwrap_or(0) as f64 + 0.5) * lw);
            out.push(self.static_car(ty, e, s, lat, -1));
        }
        if !scenario { self.static_obstacles = out.clone(); }
        out
    }

    /// placeObstacles({count, kinds, seed, slowKmh, brakeEvery, allowSingleLane})
    pub fn place_obstacles(&mut self, count: usize, kinds: &[&'static str], seed: f64, slow_kmh: (f64, f64), brake_every: (f64, f64), allow1: bool) {
        let g = self.gw();
        let mut r = Rng::new(seed);
        self.brake_every = brake_every;
        let mut pools: HashMap<&'static str, Vec<usize>> = HashMap::default();
        let mut pool = |k: &'static str| -> Vec<usize> {
            if let Some(p) = pools.get(k) { return p.clone(); }
            let mut out = Vec::new();
            for e in 0..g.edge_count {
                let multi = allow1 || g.e_lanes[e] > 1;
                match k {
                    "slow" | "braking" => { if g.e_rank[e] >= 1 && g.e_length[e] >= 150.0 { out.push(e); } }
                    "midblock" | "partial" => { if g.e_rank[e] >= 1 && g.e_length[e] >= 150.0 && multi { out.push(e); } }
                    "rampEnd" => { if g.e_gore[e] >= 0.0 && g.e_length[e] > 30.0 { out.push(e); } }
                    "afterCurve" | "afterMerge" => {
                        if g.e_length[e] < 90.0 { continue; }
                        let fr = g.e_from[e] as usize;
                        let ins: Vec<usize> = (g.in_start[fr]..g.in_start[fr + 1]).map(|q| g.in_edges[q as usize]).filter(|&x| x != g.e_rev[e]).map(|x| x as usize).collect();
                        if k == "afterMerge" && ins.len() >= 2 && ins.iter().all(|&a| g.hop(a, e).strict) { out.push(e); }
                        if k == "afterCurve" && ins.len() == 1 && multi {
                            let mut d = g.e_head_start[e] - g.e_head_end[ins[0]]; d = jm::abs(jm::atan2(jm::sin(d), jm::cos(d)));
                            if d > 0.3 && g.hop(ins[0], e).strict { out.push(e); }
                        }
                    }
                    _ => {}
                }
            }
            pools.insert(k, out.clone());
            out
        };
        for n in 0..count {
            let kind = kinds[n % kinds.len()];
            let mut slot = -1i64;
            let mut e = 0usize;
            let mut s = 0.0;
            if kind == "junction" {
                let Some(j) = &g.j else { continue };
                if j.move_count == 0 { continue; }
                let mut m = r.int(j.move_count);
                let mut t;
                let _ = m;
                t = 0.5 * j.m_len[m];
                let _ = t;
                for tries in 0..20 {
                    m = r.int(j.move_count);
                    e = j.info(m, 0) as usize;
                    if !allow1 && g.e_lanes[e] == 1 && g.e_lanes[j.info(m, 2) as usize] == 1 && tries < 19 { continue; }
                    let (mut bt, mut bd) = (-1.0, 1e9);
                    let len = j.m_len[m];
                    let mut u = 0.0;
                    while u <= len { let q = j.arc_point(m, u); let dd = jm::hypot(q.x - g.nx[g.e_to[e] as usize], q.z - g.nz[g.e_to[e] as usize]); if dd < bd { bd = dd; bt = u; } u += 0.5; }
                    t = bt; if bd < 4.0 { break; }
                }
                e = j.info(m, 0) as usize; s = j.s_in(m) + t;
                slot = self.static_car(0, e, s, (j.info(m, 1) as f64 + 0.5) * g.lane_w(e, j.s_in(m)), m as i32) as i64;
            } else {
                let p = pool(kind);
                if p.is_empty() { continue; }
                e = p[r.int(p.len())];
                let lane = r.int(g.e_lanes[e] as usize) as i32;
                let w = g.lane_w(e, g.e_length[e] / 2.0);
                let len = g.e_length[e];
                match kind {
                    "midblock" => { s = len * r.range(0.3, 0.7); slot = self.static_car(0, e, s, (lane as f64 + 0.5) * w, -1) as i64; }
                    "afterCurve" => { s = jm::min(len - 10.0, r.range(5.0, 30.0)); slot = self.static_car(0, e, s, (lane as f64 + 0.5) * g.lane_w(e, s), -1) as i64; }
                    "afterMerge" => { s = jm::min(len - 10.0, r.range(30.0, 80.0)); slot = self.static_car(0, e, s, (lane as f64 + 0.5) * g.lane_w(e, s), -1) as i64; }
                    "rampEnd" => { s = jm::max(5.0, len - r.range(3.0, 12.0)); slot = self.static_car(0, e, s, 0.5 * g.lane_w(e, s), -1) as i64; }
                    "partial" => { s = len * r.range(0.3, 0.7); let k = if g.e_lanes[e] > 1 { 1 + r.int(g.e_lanes[e] as usize - 1) } else { r.int(2) }; let lat = k as f64 * g.lane_w(e, s) + r.range(-0.4, 0.4); slot = self.static_car(0, e, s, lat, -1) as i64; }
                    _ => {
                        s = len * r.range(0.1, 0.5);
                        // walk: a 30-edge random route
                        let mut route = vec![e as i32];
                        let mut ee = e;
                        for _ in 0..30 {
                            let to = g.e_to[ee] as usize;
                            let outs: Vec<i32> = (g.out_start[to]..g.out_start[to + 1]).map(|q| g.out_edges[q as usize]).filter(|&x| x != g.e_rev[ee]).collect();
                            if outs.is_empty() { break; }
                            ee = outs[r.int(outs.len())] as usize; route.push(ee as i32);
                        }
                        let sl = self.obstacle_alloc(0, &route, lane);
                        let vs = &mut self.vs;
                        vs.s[sl] = r32(s); vs.lat[sl] = r32((lane as f64 + 0.5) * g.lane_w(e, s)); vs.update_lanes(sl, &g, e); vs.update_pose(sl, &g); vs.save_pose(sl);
                        if kind == "slow" {
                            vs.max_speed[sl] = r32(r.range(slow_kmh.0, slow_kmh.1) / 3.6); vs.speed[sl] = vs.max_speed[sl];
                            self.drivers[sl] = Driver::PropSlow;
                        } else {
                            vs.speed[sl] = r32(jm::min(g.e_speed[e], 15.0));
                            let next = self.time + r.range(brake_every.0, brake_every.1);
                            self.drivers[sl] = Driver::PropBraking { next, until: -1.0 };
                        }
                        slot = sl as i64;
                    }
                }
            }
            if slot >= 0 { self.obstacles.push((slot as usize, kind, e as i32, s)); }
        }
        self.obstacle_rng = Some(r);
    }

    // ---------------------------------------------------------------- stuck labelling
    pub(crate) fn stuck_cause(&mut self, i: usize) -> &'static str {
        let g = self.gw();
        let vs = &self.vs;
        let mut seen: HashSet<usize> = HashSet::default();
        let (mut k, mut n) = (i, 0);
        while vs.state[k] == STATE_DRIVING && vs.leader[k] >= 0 && !seen.contains(&k) && { n += 1; n - 1 < 200 } {
            let l = vs.leader[k] as usize;
            if vs.state[l] == STATE_DRIVING && (vs.speed[l] > 1.0 || vs.gap[k] > 30.0) { break; }
            seen.insert(k); k = l;
        }
        if vs.state[k] == STATE_DRIVING && f(vs.pull_until[k]) > self.time { return "pulled-over-blocking"; }
        if vs.leader[k] >= 0 && seen.contains(&k) { return "deadlock-loop"; }
        if vs.state[k] == STATE_CRASHED { return if vs.flags[k] & FLAG_STRANDED != 0 { "behind-stranded" } else { "behind-wreck" }; }
        if vs.state[k] != STATE_DRIVING { return "other"; }
        if vs.leader[k] == -2 { return "lane-end"; }
        if vs.leader[k] == -3 { return "junction-conflict"; }
        let e = vs.edge[k] as usize;
        if vs.ctl_type[k] >= 0 && vs.ctl_dist[k] < 30.0 {
            let (t, st) = (vs.ctl_type[k], vs.ctl_state[k]);
            if t == CTRL_STOP { let cc = g.controls.as_ref().unwrap(); return if cc.flags[cc.ap_control[vs.ctl_approach[k] as usize] as usize] & 1 != 0 { "all-way-stop" } else { "stop-sign" }; }
            if t == CTRL_GIVE_WAY { return "give-way"; }
            if st == LIGHT_GREEN && vs.ctl_move[k] == MOVE_LEFT { return "signal-permissive-left"; }
            return "signal";
        }
        let dist = g.e_length[e] - f(vs.s[k]);
        if self.yield_required(k, e, dist) { return "junction-yield"; }
        if self.edge_has_crash[self.vs.edge[k] as usize] != 0 { return "scene-on-edge"; }
        "head-stopped-other"
    }

    /// ctx.yieldRequired(i, e, dist)
    pub fn yield_required(&mut self, i: usize, e: usize, dist: f64) -> bool {
        if self.empty { return false; }
        let g = self.gw();
        if g.j.is_some() {
            let r = self.yield_j(i, e);
            if r && self.creep_until.len() > i && self.creep_until[i] > self.time && self.yield_by.as_ref().map(|y| y[i]) == Some(self.creep_ignore[i]) { return false; }
            return r;
        }
        if dist < 1.5 { return false; }
        let to = g.e_to[e] as usize;
        let inc: Vec<usize> = (g.in_start[to]..g.in_start[to + 1]).map(|k| g.in_edges[k as usize]).filter(|&x| x != e as i32 && x != g.e_rev[e]).map(|x| x as usize).collect();
        let out: Vec<usize> = (g.out_start[to]..g.out_start[to + 1]).map(|k| g.out_edges[k as usize]).filter(|&x| x != g.e_rev[e]).map(|x| x as usize).collect();
        let mut hw: f64 = 0.0;
        for &o in inc.iter().chain(out.iter()) { hw = jm::max(hw, (if g.e_rev[o] >= 0 { 2.0 } else { 1.0 }) * g.e_lanes[o] as f64 * 3.7 * 0.5); }
        let stop_back = if !inc.is_empty() { jm::min(12.0, jm::max(1.5, hw + 1.0)) } else { 1.5 };
        if inc.is_empty() { return false; }
        let vs = &self.vs;
        let l0 = vs.leader[i];
        if l0 >= 0 && vs.speed[l0 as usize] < 1.0 && vs.state[l0 as usize] == STATE_DRIVING && f(vs.gap[i]) - dist < f(vs.length[i]) + 3.0 { return true; }
        let route = &vs.routes[i]; let rp = vs.route_pos[i] as usize;
        let next_edge = if rp + 1 < route.len() { route[rp + 1] } else { -1 };
        let next_rev = if next_edge >= 0 { g.e_rev[next_edge as usize] } else { -1 };
        let waited = if vs.stopped_since[i] >= 0.0 { self.time - f(vs.stopped_since[i]) } else { 0.0 };
        let patience = jm::max(0.3, 1.0 - waited / 90.0);
        let v = f(vs.speed[i]);
        let t_me = dist / jm::max(v, 2.0);
        let win = jm::max(2.0 * patience, f(vs.reaction_time[i]) + 1.5);
        for &eo in &out {
            for l in 0..g.e_lanes[eo] {
                let j = self.lanes.first(eo as i32, l);
                if j < 0 || vs.state[j as usize] != STATE_DRIVING { continue; }
                let j = j as usize;
                if vs.speed[j] < 0.5 && waited > 20.0 { continue; }
                if f(vs.s[j]) < f(vs.length[j]) + stop_back + 3.0 && !(eo as i32 == next_edge && l == g.map_lane(e, eo, vs.lane[i] as i32)) { return true; }
            }
        }
        for &e2 in &inc {
            if e2 as i32 == next_rev { continue; }
            for l in 0..g.e_lanes[e2] {
                let j = self.lanes.last(e2 as i32, l);
                if j < 0 || vs.state[j as usize] != STATE_DRIVING { continue; }
                let j = j as usize;
                let rj = &vs.routes[j]; let pj = vs.route_pos[j] as usize;
                if pj + 1 >= rj.len() { continue; }
                if vs.refuel_state[j] == 1 && vs.stop_pos[j] == pj as i32 { continue; }
                let d2 = g.e_length[e2] - f(vs.s[j]);
                let v2 = f(vs.speed[j]);
                if v2 < 0.3 && d2 > 3.0 && waited > 5.0 { continue; }
                let t2 = d2 / jm::max(v2, 2.0);
                if g.e_rank[e2] > g.e_rank[e] { if t2 < 4.0 * patience || d2 < 6.0 { return true; } }
                else if g.e_rank[e2] == g.e_rank[e] {
                    if d2 < 60.0 || t2 < 6.0 { if t2 < t_me - win || (jm::abs(t2 - t_me) <= win && (t2 < t_me || (t2 == t_me && j < i))) || d2 < 3.0 { return true; } }
                } else if d2 < 5.0 && waited < 30.0 { return true; }
            }
        }
        false
    }

    // ---------------------------------------------------------------- fuel / stations
    pub(crate) fn check_fuel(&mut self, i: usize, at_spawn: bool) -> bool {
        self.vs.last_fuel_check[i] = r32(self.time);
        if self.stations.is_none() || self.vs.refuel_state[i] != 0 { return false; }
        let remaining = f(self.vs.route_total[i]) - f(self.vs.route_dist[i]);
        if !self.stations.as_ref().unwrap().needs_fuel(self.vs.type_idx[i] as usize, f(self.vs.fuel[i]), f(self.vs.tank[i]), remaining) { return false; }
        let g = self.gw();
        let route = self.vs.routes[i].clone();
        let rp = self.vs.route_pos[i] as usize;
        let goal = g.e_to[route[route.len() - 1] as usize];
        let (from, prefix): (i32, Vec<i32>) = if at_spawn { (g.e_from[route[0] as usize], Vec::new()) } else {
            let fr = g.e_to[route[rp] as usize];
            if fr == goal { return false; }
            (fr, vec![route[rp]])
        };
        let plan = { let st = self.stations.as_mut().unwrap(); st.plan(&*g, &mut *self.router, from, goal) };
        let Some(plan) = plan else { self.counters.refuel_plan_failed += 1; return false };
        let mut edges = prefix.clone(); edges.extend_from_slice(&plan.edges);
        let s_on = if at_spawn { 0.0 } else { f(self.vs.s[i]) };
        self.set_route(i, &edges, 0);
        self.vs.route_dist[i] = r32(s_on);
        self.vs.refuel_state[i] = 1; self.vs.station[i] = plan.station as i32; self.vs.stop_pos[i] = (plan.stop_index + prefix.len()) as i32;
        self.vs.flags[i] |= FLAG_REFUEL;
        self.counters.refuel_planned += 1;
        true
    }

    pub(crate) fn enter_station(&mut self, i: usize) {
        let k = self.vs.station[i] as usize;
        let vs = &mut self.vs;
        vs.state[i] = STATE_REFUELING; vs.speed[i] = 0.0; vs.accel[i] = 0.0; vs.refuel_state[i] = 2;
        vs.clear_controls(i);
        vs.pump_until[i] = r32(self.time);
        let st = self.stations.as_mut().unwrap();
        st.queue[k].push(i);
        if st.queue[k].len() > st.max_queue { st.max_queue = st.queue[k].len(); }
        self.lot.push(i);
    }

    pub(crate) fn pumps(&mut self) {
        let cfg = self.opts.fuel_cfg.clone();
        let st = self.stations.as_mut().unwrap();
        for k in 0..st.pumps.len() {
            while !st.queue[k].is_empty() && st.busy[k] < st.pumps[k] {
                let i = st.queue[k].remove(0); st.busy[k] += 1;
                self.accum.pump_wait_sum += self.time - f(self.vs.pump_until[i]); self.accum.pump_wait_n += 1.0;
                self.vs.refuel_state[i] = 3; self.vs.pump_until[i] = r32(self.time + cfg.pump_min + (cfg.pump_max - cfg.pump_min) * self.rng.random());
            }
        }
    }

    pub(crate) fn leave_stations(&mut self) {
        let g = self.gw();
        let lot = std::mem::take(&mut self.lot);
        let mut keep = Vec::new();
        for i in lot {
            if self.vs.state[i] != STATE_REFUELING { continue; }
            if self.vs.refuel_state[i] == 3 && self.time >= f(self.vs.pump_until[i]) {
                let k = self.vs.station[i] as usize;
                self.accum.fuel_l += f(self.vs.tank[i]) - f(self.vs.fuel[i]);
                self.vs.fuel[i] = self.vs.tank[i];
                let st = self.stations.as_mut().unwrap();
                st.busy[k] -= 1; st.served[k] += 1; self.vs.refuel_state[i] = 4;
                self.counters.refuelled += 1;
            }
            if self.vs.refuel_state[i] == 4 {
                let next = (self.vs.stop_pos[i] + 1) as usize;
                if next >= self.vs.routes[i].len() { self.arrive(i); continue; }
                let en = self.vs.routes[i][next] as usize;
                let lane = self.entry_lane(en, 0, f(self.vs.length[i]), -1);
                if lane < 0 && self.time - f(self.vs.pump_until[i]) < self.opts.spawn_blocked_timeout { keep.push(i); continue; }
                self.spawned_keys.insert(en as i64 * 8 + lane.max(0) as i64);
                let vs = &mut self.vs;
                vs.state[i] = STATE_DRIVING; vs.refuel_state[i] = 0; vs.route_pos[i] = next as i32; vs.edge[i] = en as i32; vs.s[i] = 0.0; vs.seg[i] = 0;
                vs.lat[i] = r32((lane.max(0) as f64 + 0.5) * g.lane_w(en, 0.0)); vs.heading_rel[i] = 0.0; vs.steer_angle[i] = 0.0; vs.steer_sl[i] = 0.0; vs.target_lane[i] = -1; vs.clear_controls(i);
                vs.update_lanes(i, &g, en); vs.stop_pos[i] = -1; vs.station[i] = -1;
                vs.flags[i] &= !FLAG_REFUEL; vs.stopped_since[i] = -1.0;
                vs.speed[i] = 0.0; vs.v_lat[i] = 0.0; vs.yaw_rate[i] = 0.0;
                vs.update_pose(i, &g); vs.save_pose(i);
                continue;
            }
            keep.push(i);
        }
        self.lot = keep;
    }

    pub(crate) fn strand(&mut self, i: usize) {
        self.events.stranded.push((i, self.vs.commuter[i]));
        self.emit(crate::sim::EV_STRANDED, i, -1, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
        self.vs.flags[i] |= FLAG_STRANDED;
        self.counters.stranded += 1;
        let tow = self.opts.fuel_cfg.tow_median * jm::exp(self.opts.fuel_cfg.tow_sigma * self.rng.gaussian());
        let e = self.vs.edge[i] as usize;
        self.open_scene(&[i], "stranded", "stranded", tow, tow, e, false, None);
        self.fail_trip(i, FAIL_STRANDED, false);
    }
}

/// sim.ctx: the junction / scene queries a built-in driver (IDM, src/sim/driver.js in the training repo) uses.
/// Stable API for external drivers run through `Sim::batch_policy`.
impl Sim {
    /// ctx.rubberneck(i, e): desired-speed factor near an active crash scene (1 = none).
    pub fn rubberneck(&self, i: usize, e: usize) -> f64 {
        const DIST: f64 = 120.0; const FACTOR: f64 = 0.8;
        if self.edge_has_crash[e] == 0 { return 1.0; }
        let vs = &self.vs;
        let (s, lane) = (f(vs.s[i]), vs.lane[i] as i64);
        if let Some(own) = self.edge_crashes.get(&(e as i32)) { for &ri in own { let r = &self.crashes[ri]; if r.is_crash && ((r.lane_mask >> lane) & 1 == 0 || r.response == Response::Shoulder) && jm::abs(r.s - s) < DIST { return FACTOR; } } }
        let rv = self.g.e_rev[e];
        if rv >= 0 { if let Some(opp) = self.edge_crashes.get(&rv) { for &ri in opp { let r = &self.crashes[ri]; if r.is_crash && jm::abs((self.g.e_length[e] - r.s) - s) < DIST { return FACTOR; } } } }
        1.0
    }
    /// ctx.speedCap(i, e): squeeze speed past a full closure (Infinity = none).
    pub fn speed_cap(&self, i: usize, e: usize) -> f64 {
        if self.edge_has_crash[e] == 0 { return f64::INFINITY; }
        let s = f(self.vs.s[i]);
        if let Some(own) = self.edge_crashes.get(&(e as i32)) { for &ri in own { let r = &self.crashes[ri]; if r.squeeze && r.response == Response::Active && r.s - s < 60.0 && s - r.s < 20.0 { return self.opts.squeeze_speed; } } }
        f64::INFINITY
    }
    /// ctx.laneOpen(i, e, lane, within)
    pub fn lane_open(&self, i: usize, e: usize, lane: i32, within: f64) -> bool {
        if lane < 0 || lane >= self.g.e_lanes[e] { return false; }
        if self.edge_has_crash[e] == 0 { return true; }
        let s = f(self.vs.s[i]);
        if let Some(recs) = self.edge_crashes.get(&(e as i32)) { for &ri in recs { let r = &self.crashes[ri]; if r.response == Response::Active && !r.squeeze && (r.lane_mask >> lane) & 1 != 0 && r.s > s && r.s - s < within { return false; } } }
        true
    }
    /// ctx.blockedAhead(i, e): +1 / -1 escape side when my lane is blocked by a stopped obstacle / scene, 0 otherwise.
    pub fn blocked_ahead(&self, i: usize, e: usize) -> i32 {
        let vs = &self.vs;
        let l = vs.leader[i];
        let direct = l >= 0 && vs.state[l as usize] == STATE_CRASHED && vs.lane_hi[l as usize] >= vs.lane_lo[l as usize] && vs.gap[i] < 150.0;
        if !direct && !(self.edge_has_crash[e] != 0 && !self.lane_open(i, e, vs.lane[i] as i32, 150.0)) { return 0; }
        let lane = vs.lane[i] as i32;
        for d in 1..self.g.e_lanes[e] {
            if lane + d < self.g.e_lanes[e] && self.lane_open(i, e, lane + d, 1e9) { return 1; }
            if lane - d >= 0 && self.lane_open(i, e, lane - d, 1e9) { return -1; }
        }
        0
    }
    /// ctx.boxBusy(i, e): a car in the junction box ahead whose path I would cross.
    pub fn box_busy(&mut self, i: usize, e: usize) -> bool {
        let g = self.gw();
        if g.j.is_some() { return self.box_busy_j(i, e); }
        let vs = &self.vs;
        let rp = vs.route_pos[i] as usize;
        let next = if rp + 1 < vs.routes[i].len() { vs.routes[i][rp + 1] } else { -1 };
        let to = g.e_to[e] as usize;
        for q in g.out_start[to]..g.out_start[to + 1] {
            let eo = g.out_edges[q as usize];
            if eo == g.e_rev[e] { continue; }
            for l in 0..g.e_lanes[eo as usize] {
                let j = self.lanes.first(eo, l); if j < 0 || vs.state[j as usize] == STATE_FREE { continue; }
                if eo == next { continue; }
                if f(vs.s[j as usize]) < f(vs.length[j as usize]) + 4.0 { return true; }
            }
        }
        for q in g.in_start[to]..g.in_start[to + 1] {
            let e2 = g.in_edges[q as usize];
            if e2 as usize == e { continue; }
            for l in 0..g.e_lanes[e2 as usize] {
                let j = self.lanes.last(e2, l); if j < 0 || vs.state[j as usize] != STATE_DRIVING { continue; }
                if g.e_length[e2 as usize] - f(vs.s[j as usize]) < 3.0 && vs.speed[j as usize] > 1.0 { return true; }
            }
        }
        false
    }
    /// ctx.laneGapOk(i, e, target, relaxed, transit): delay-aware gap acceptance for a lane change into `target`.
    pub fn lane_gap_ok(&self, i: usize, e: usize, target: i32, relaxed: bool, transit: bool) -> bool {
        let g = self.gw();
        let vs = &self.vs;
        if target < 0 || target >= g.e_lanes[e] { return false; }
        let si = f(vs.s[i]);
        if !transit && self.edge_has_crash[e] != 0 {
            if let Some(recs) = self.edge_crashes.get(&(e as i32)) { for &ri in recs { let r = &self.crashes[ri]; if r.response == Response::Active && !r.squeeze && (r.lane_mask >> target) & 1 != 0 && r.s > si && r.s - si < 60.0 { return false; } } }
        }
        let (a0, bh) = self.lanes.neighbors_at(vs, e as i32, target, si, i as i32);
        let v = f(vs.speed[i]); let half = 0.5 * f(vs.length[i]);
        let ah = self.blocking(a0, target);
        let t_lc = 3.0;
        if ah >= 0 {
            let a = ah as usize;
            let ga = f(vs.s[a]) - si - half - 0.5 * f(vs.length[a]);
            let dv = v - if vs.state[a] == STATE_DRIVING { f(vs.speed[a]) } else { 0.0 };
            let need = if relaxed && v < 2.0 { 1.0 } else { jm::max(3.0, 2.0 + v * 0.5 + jm::max(0.0, dv) * (f(vs.reaction_time[i]) + t_lc)) };
            if ga < need { return false; }
        }
        if bh >= 0 && vs.state[bh as usize] == STATE_DRIVING {
            let b = bh as usize;
            let gb = si - f(vs.s[b]) - half - 0.5 * f(vs.length[b]); let vb = f(vs.speed[b]);
            let need = if relaxed && vb < 2.0 { 1.0 } else { jm::max(3.0, 2.0 + vb * 0.5 + jm::max(0.0, vb - v) * (f(vs.reaction_time[b]) + t_lc)) };
            if gb < need { return false; }
        }
        let far = target + (target - vs.lane[i] as i32);
        if far >= 0 && far < g.e_lanes[e] && far != vs.lane[i] as i32 {
            let dir = (target - far) as f64;
            let (p0, p1) = self.lanes.neighbors_at(vs, e as i32, far, si, i as i32);
            for j in [p0, p1] {
                if j < 0 || vs.state[j as usize] != STATE_DRIVING { continue; }
                let j = j as usize;
                let hr = f(vs.heading_rel[j]);
                let lat_vel = f(vs.speed[j]) * jm::sin(hr) + f(vs.v_lat[j]) * jm::cos(hr);
                let heading = lat_vel * dir > 0.2 || (vs.target_lane[j] as i32 == target && vs.flags[j] & FLAG_BRAIN == 0);
                if !heading { continue; }
                let gap = jm::abs(f(vs.s[j]) - si) - half - 0.5 * f(vs.length[j]);
                if gap < 5.0 + jm::abs(f(vs.speed[j]) - v) * 2.0 { return false; }
            }
        }
        true
    }
}
