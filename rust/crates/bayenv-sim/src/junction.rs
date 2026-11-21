//! sim.js junction perception (src/geom junctions): _indexConnectors, _junction, _nodeBodies, _convergeBodies,
//! _conflicts, _signed, _mustStop, _yieldJ, _seeApproach, _movesInto, _pathBodies, _boxBusyJ.
use crate::jm;
use crate::sim::{f, r32, Sim};
use crate::vehicle::*;
use crate::world::Conflict;

/// lane-index walks stopped by walk_limit (see Sim::walk_limit)
pub static WALK_CYCLES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WALK_DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Sim {
    pub(crate) fn index_connectors(&mut self) {
        let g = self.gw();
        let j = g.j.as_ref().unwrap();
        if self.conn_cars.len() > 50000 { self.conn_cars.clear(); } else { self.conn_cars.reset_lists(); }
        if self.node_conn.len() > 50000 { self.node_conn.clear(); } else { self.node_conn.reset_lists(); }
        let vs = &self.vs;
        for i in 0..vs.high {
            let m = vs.conn[i];
            if m < 0 || (vs.state[i] != STATE_DRIVING && vs.state[i] != STATE_CRASHED) { continue; }
            self.conn_cars.push(m as usize, i);
            let node = g.e_to[j.info(m as usize, 0) as usize] as usize;
            self.node_conn.push(node, i);
        }
    }

    /// Lane-index walks (`behind` / `ahead` chains) take at most this many steps: a walk on a consistent index visits
    /// each index entry once. The index is built before integrate; walks after it (the observation, as in the JS) can
    /// follow entries of cars that changed lane / edge since, and such a walk can cycle (the JS would never return).
    /// A walk that reaches the limit stops as if no further car were found; a walk that ends never reaches it.
    #[inline]
    pub(crate) fn walk_limit(&self) -> usize { self.vs.high * crate::vehicle::MAX_OCC as usize + 1 }
    /// A walk hit walk_limit: counted (WALK_CYCLES), state dumped once to $BAYENV_WALK_DUMP when set.
    #[cold]
    pub(crate) fn walk_cycle(&self, what: &str, i: usize, edge: usize, lane: i32) {
        WALK_CYCLES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let Ok(path) = std::env::var("BAYENV_WALK_DUMP") else { return };
        if WALK_DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) { return; }
        let vs = &self.vs;
        // the walk again, recording its slots (bounded)
        let mut seq = Vec::new();
        let mut jj = self.lanes.last(edge as i32, lane);
        while jj >= 0 && seq.len() < 64 { seq.push(jj as usize); jj = self.lanes.behind(vs, jj as usize, lane); }
        let car = |k: usize| serde_json::json!({ "slot": k, "edge": vs.edge[k], "lane": vs.lane[k], "lane_lo": vs.lane_lo[k], "lane_hi": vs.lane_hi[k],
            "conn": vs.conn[k], "state": vs.state[k], "s": vs.s[k], "indexed_edge": self.indexed.get(k).copied(),
            "ent_pos": &self.lanes.ent_pos[k * crate::vehicle::MAX_OCC as usize..(k + 1) * crate::vehicle::MAX_OCC as usize] });
        let (a, b) = self.lanes.range(edge as i32, lane);
        let d = serde_json::json!({ "what": what, "step": self.step_count, "time": self.time, "car": car(i), "walk_edge": edge, "walk_lane": lane,
            "range": [a, b], "order": &self.lanes.order[a..b.max(a)], "walk": seq.iter().map(|&k| car(k)).collect::<Vec<_>>() });
        let _ = std::fs::write(&path, serde_json::to_string_pretty(&d).unwrap());
        eprintln!("bayenv-sim: lane-index walk cycle ({what}, slot {i}, edge {edge} lane {lane}, step {}); state dumped to {path}", self.step_count);
    }
    #[inline]
    pub(crate) fn signed(&self, j: usize) -> bool { let vs = &self.vs; (vs.ctl_type[j] == 1 || vs.ctl_type[j] == 2) && vs.ctl_dist[j] < 60.0 }
    #[inline]
    pub(crate) fn must_stop(&self, j: usize) -> bool {
        let vs = &self.vs;
        if vs.ctl_type[j] < 0 || vs.ctl_dist[j] > 60.0 { return false; }
        vs.ctl_state[j] == 2 || (vs.ctl_type[j] == 1 && vs.ctl_stopped[j] == 0)
    }

    /// _conflicts(m) without filling the cache (cache hit, else computed): for read-only (parallel) perception.
    pub(crate) fn conflicts_ro(&self, m: usize) -> &'static [Conflict] {
        let g = self.gw();
        g.conf_ext[m].get_or_init(|| self.compute_conflicts(m).into_boxed_slice())
    }

    /// _conflicts(m): world conflicts + same-junction movements whose paths pass within 4.2 m (cached).
    pub(crate) fn compute_conflicts(&self, m: usize) -> Vec<Conflict> {
        let g = self.gw();
        let j = g.j.as_ref().unwrap();
        let base = j.conflicts_of(m);
        let mut c: Vec<Conflict> = base.to_vec();
        let node = g.e_to[j.info(m, 0) as usize] as usize;
        let (in_e, in_l) = (j.info(m, 0), j.info(m, 1));
        let p = &j.pts;
        let (f1, n1) = (j.info(m, 4) as usize, j.info(m, 5) as usize);
        for o in j.movements_at(node) {
            if o == m || base.iter().any(|x| x.other as usize == o) || (j.info(o, 0) == in_e && j.info(o, 1) == in_l) { continue; }
            let (f2, n2) = (j.info(o, 4) as usize, j.info(o, 5) as usize);
            let (mut best, mut sa, mut sb) = (1e9, 0.0, 0.0);
            for a in 0..n1 { for b in 0..n2 {
                let d = jm::hypot(p[4 * (f1 + a)] - p[4 * (f2 + b)], p[4 * (f1 + a) + 2] - p[4 * (f2 + b) + 2]);
                if d < best { best = d; sa = p[4 * (f1 + a) + 3]; sb = p[4 * (f2 + b) + 3]; }
            } }
            if best < 4.2 { c.push(Conflict { other: o as u32, ty: 0, s: sa, s_other: sb }); }
        }
        c
    }

    /// _conflicts(m), memoised per world (a pure function of the junction data; shared by every env of the world).
    pub(crate) fn conflicts(&self, m: usize) -> &'static [Conflict] { self.conflicts_ro(m) }

    fn stuck_in(&self, k: usize) -> bool { let vs = &self.vs; vs.speed[k] < 0.3 && vs.stopped_since[k] >= 0.0 && self.time - f(vs.stopped_since[k]) > 20.0 }

    pub(crate) fn junction_p(&self, p: &P2) {
        let i = p.i;
        let g = self.gw();
        let j = g.j.as_ref().unwrap();
        let e = self.vs.edge[i] as usize;
        let hl = f(self.vs.hl[i]);
        let v = f(self.vs.speed[i]);
        let mut m = self.vs.conn[i];
        if m < 0 {
            if !(g.e_s_stop[e] < g.e_length[e]) { return; }
            let vs = &self.vs;
            let rp = vs.route_pos[i] as usize;
            if rp + 1 >= vs.routes[i].len() || (vs.refuel_state[i] == 1 && vs.stop_pos[i] == rp as i32) { return; }
            if g.e_s_stop[e] - f(vs.s[i]) - hl > jm::max(30.0, v * v / 4.0 + 15.0) { return; }
            m = j.movement(e, vs.lane[i] as i32, vs.routes[i][rp + 1]);
            if m < 0 { return; }
        }
        let m = m as usize;
        let s_in = j.s_in(m);
        let len = j.m_len[m];
        let ti = f(self.vs.s[i]) - s_in;
        let my_node = g.e_to[j.info(m, 0) as usize];
        let to_mouth = -ti - hl - 0.5;
        let mut best = f(p.gap.get());
        let mut who = p.leader.get();
        let mut who_j = -1i32;
        let t0 = jm::max(0.0, ti + hl);
        if t0 < len {
            let pc = j.arc_point(m, 0.5 * (t0 + len));
            let mut cand: Vec<usize> = Vec::new();
            self.hash.near_into(&self.vs.x, &self.vs.z, pc.x, pc.z, 0.5 * (len - t0) + 8.0, &mut cand);
            let vs = &self.vs;
            cand.retain(|&jj| {
                let other_box = vs.conn[jj] >= 0 && g.e_to[j.info(vs.conn[jj] as usize, 0) as usize] != my_node;
                if jj == i || !(vs.state[jj] == STATE_CRASHED || vs.speed[jj] < 2.0 || (other_box && vs.speed[jj] < 7.0)) { return false; }
                if vs.edge[jj] == vs.edge[i] && vs.s[jj] < vs.s[i] && vs.state[jj] == STATE_DRIVING { return false; }
                if vs.state[jj] == STATE_DRIVING && p.leader_of(self, jj) == i as i32 { let i_in = vs.conn[i] >= 0; let j_in = vs.conn[jj] >= 0; if (i_in && !j_in) || (i_in == j_in && i < jj) { return false; } }
                if vs.state[jj] == STATE_DRIVING && vs.conn[i] >= 0 && vs.conn[jj] >= 0 && i < jj && self.stuck_in(i) && self.stuck_in(jj) { return false; }
                true
            });
            if !cand.is_empty() {
                let clear = f(vs.hw[i]) + 0.4;
                let off = f(vs.lat[i]) - (j.info(m, 1) as f64 + 0.5) * g.lane_w(e, s_in);
                let mut t = t0;
                while t <= len + 0.01 {
                    let p = j.arc_point(m, t);
                    let px = p.x + off * p.hz;
                    let pz = p.z - off * p.hx;
                    let mut hit = -1i32;
                    for &jj in &cand {
                        let c = jm::cos(f(vs.heading[jj])); let sn = jm::sin(f(vs.heading[jj]));
                        let dx = px - f(vs.x[jj]); let dz = pz - f(vs.z[jj]);
                        let lx = jm::abs(dx * c + dz * sn) - f(vs.hl[jj]); let lz = jm::abs(dz * c - dx * sn) - f(vs.hw[jj]);
                        if lx < clear && lz < clear && jm::hypot(jm::max(lx, 0.0), jm::max(lz, 0.0)) < clear { hit = jj as i32; break; }
                    }
                    if hit >= 0 { let gg = if ti + hl < 0.0 { jm::min(to_mouth, t - ti - hl - 0.5) } else { t - ti - hl - 0.5 }; if gg < best { best = gg; who = hit; } break; }
                    t += 1.0;
                }
            }
        }
        let confs = self.conflicts(m);
        let vs = &self.vs;
        for c in confs.iter() {
            let Some(list) = self.conn_cars.get(&(c.other as usize)) else { continue };
            if list.is_empty() { continue; }
            let di = c.s - ti;
            if di < -hl - 1.0 { continue; }
            let o = c.other as usize;
            for &jj in list {
                if jj == i { continue; }
                if vs.speed[jj] < 0.5 && (p.leader_of(self, jj) == i as i32 || (p.leader_of(self, jj) == -3 && p.cw_of(self, jj) == Some(i))) { continue; }
                if vs.state[jj] == STATE_DRIVING && vs.conn[i] >= 0 && i < jj && self.stuck_in(i) && self.stuck_in(jj) { continue; }
                let tj = f(vs.s[jj]) - j.s_in(o);
                let dj = c.s_other - tj;
                let hlj = f(vs.hl[jj]);
                if c.ty == 1 {
                    if dj < di || (dj == di && jj < i) { let gg = di - dj - hl - hlj - 1.0; if gg < best { best = gg; who = jj as i32; } }
                    continue;
                }
                if dj < -hlj - f(vs.hw[i]) - 0.5 { continue; }
                if j.info(m, 0) == j.info(o, 0) {
                    let my_pref = j.info(m, 6) != 0; let his_pref = j.info(o, 6) != 0;
                    if my_pref && !his_pref { continue; }
                    if !my_pref && his_pref { let g2 = jm::min(di - hl - f(vs.hw[jj]) - 1.5, if ti + hl < 0.0 { to_mouth } else { 1e9 }); if g2 < best { best = g2; who = -3; who_j = jj as i32; } continue; }
                }
                let mut g_stop = di - hl - f(vs.hw[jj]) - 1.5;
                if ti + hl < 0.0 { g_stop = jm::min(g_stop, to_mouth); }
                if dj < hlj + f(vs.hw[i]) + 0.5 {
                    if di < hl + f(vs.hw[jj]) + 0.5 { let clear_i = di + hl + f(vs.hw[jj]); let clear_j = dj + hlj + f(vs.hw[i]); if clear_i < clear_j || (clear_i == clear_j && i < jj) { continue; } }
                    if g_stop < best { best = g_stop; who = -3; who_j = jj as i32; }
                    continue;
                }
                let t_i = di / jm::max(v, 1.0); let t_j = dj / jm::max(f(vs.speed[jj]), 1.0);
                if (t_j < t_i || (t_j == t_i && jj < i)) && g_stop > v * v / 8.0 { if g_stop < best { best = g_stop; who = -3; who_j = jj as i32; } }
            }
        }
        if best < f(p.gap.get()) {
            p.gap.set(r32(best)); p.leader.set(who);
            if who == -3 { p.cw.set(Some(if who_j >= 0 { who_j as usize } else { usize::MAX })); }
        }
    }

    pub(crate) fn node_bodies_p(&self, p: &P2, e: usize) {
        let i = p.i;
        let g = self.gw();
        let j = g.j.as_ref().unwrap();
        let (hl, hw) = (f(self.vs.hl[i]), f(self.vs.hw[i]));
        let check = |node: usize| {
            let Some(list) = self.node_conn.get(&node) else { return };
            if list.is_empty() { return; }
            let vs = &self.vs;
            let yaw = f(vs.heading[i]); let fx = jm::cos(yaw); let fz = jm::sin(yaw);
            for &jj in list {
                if jj == i { continue; }
                if vs.state[jj] == STATE_DRIVING && vs.speed[jj] < 0.5 && (p.leader_of(self, jj) == i as i32 || (p.leader_of(self, jj) == -3 && p.cw_of(self, jj) == Some(i))) { continue; }
                let dx = f(vs.x[jj]) - f(vs.x[i]); let dz = f(vs.z[jj]) - f(vs.z[i]);
                let along = dx * fx + dz * fz; if along <= 0.0 || along > 60.0 { continue; }
                let lat = dx * fz - dz * fx;
                let mut dpsi = f(vs.heading[jj]) - yaw; dpsi = jm::atan2(jm::sin(dpsi), jm::cos(dpsi));
                let sa = jm::abs(jm::sin(dpsi)); let ca = jm::abs(jm::cos(dpsi));
                if jm::abs(lat) > hw + sa * f(vs.hl[jj]) + ca * f(vs.hw[jj]) + 0.3 { continue; }
                let gg = along - hl - (ca * f(vs.hl[jj]) + sa * f(vs.hw[jj])) - 0.5;
                if gg < f(p.gap.get()) {
                    let ld = if jm::abs(dpsi) < 0.5 && vs.state[jj] == STATE_DRIVING { jj as i32 } else { -3 };
                    p.gap.set(r32(gg)); p.leader.set(ld);
                    if ld == -3 { p.cw.set(Some(jj)); }
                }
            }
        };
        if f(self.vs.s[i]) - hl < 40.0 && g.e_s_exit[e] > 0.0 { check(g.e_from[e] as usize); }
        if g.e_length[e] - f(self.vs.s[i]) < 60.0 && g.e_s_stop[e] < g.e_length[e] {
            let vs = &self.vs;
            let rp = vs.route_pos[i] as usize;
            let r = &vs.routes[i];
            if r.is_empty() || rp + 1 >= r.len() || j.movement(e, vs.lane[i] as i32, r[rp + 1]) < 0 { check(g.e_to[e] as usize); }
        }
    }

    pub(crate) fn converge_p(&self, p: &P2, e: usize) {
        let i = p.i;
        let g = self.gw();
        let d_node = g.e_length[e] - f(self.vs.s[i]);
        if d_node > 150.0 { return; }
        let rp = self.vs.route_pos[i] as usize;
        if self.vs.routes[i].is_empty() || rp + 1 >= self.vs.routes[i].len() { return; }
        let next = self.vs.routes[i][rp + 1] as usize;
        let node = g.e_to[e] as usize;
        if g.in_start[node + 1] - g.in_start[node] < 2 || !g.hop(e, next).strict { return; }
        let yaw = f(self.vs.heading[i]); let fx = jm::cos(yaw); let fz = jm::sin(yaw);
        let (hl, hw) = (f(self.vs.hl[i]), f(self.vs.hw[i]));
        for q in g.in_start[node]..g.in_start[node + 1] {
            let e2 = g.in_edges[q as usize] as usize;
            if e2 == e || e2 as i32 == g.e_rev[next] { continue; }
            for l in 0..g.e_lanes[e2] {
                let mut jj = self.lanes.last(e2 as i32, l);
                let mut n = 0;
                while jj >= 0 && { n += 1; n - 1 < 4 } {
                    let ju = jj as usize;
                    let vs = &self.vs;
                    if g.e_length[e2] - f(vs.s[ju]) > d_node + 60.0 { break; }
                    let rj = &vs.routes[ju]; let pj = vs.route_pos[ju] as usize;
                    let waiting = vs.state[ju] == STATE_DRIVING && vs.speed[ju] < 1.0 && vs.ctl_type[ju] >= 0 && vs.ctl_dist[ju] > -0.5 && vs.ctl_dist[ju] < 15.0 && (vs.ctl_state[ju] == 2 || vs.ctl_state[ju] == 3);
                    if !waiting && (vs.state[ju] == STATE_CRASHED || (!rj.is_empty() && pj + 1 < rj.len() && rj[pj + 1] == next as i32 && g.hop(e2, next).strict)) {
                        let dx = f(vs.x[ju]) - f(vs.x[i]); let dz = f(vs.z[ju]) - f(vs.z[i]);
                        let along = dx * fx + dz * fz;
                        if along > 0.0 && along < 80.0 {
                            let lat = dx * fz - dz * fx;
                            let mut dpsi = f(vs.heading[ju]) - yaw; dpsi = jm::atan2(jm::sin(dpsi), jm::cos(dpsi));
                            let sa = jm::abs(jm::sin(dpsi)); let ca = jm::abs(jm::cos(dpsi));
                            if jm::abs(dpsi) < 0.5 && jm::abs(lat) < hw + sa * f(vs.hl[ju]) + ca * f(vs.hw[ju]) + 0.4 {
                                let gg = along - hl - (ca * f(vs.hl[ju]) + sa * f(vs.hw[ju]));
                                if gg < f(p.gap.get()) { p.gap.set(r32(gg)); p.leader.set(jj); }
                            }
                        }
                    }
                    jj = self.lanes.behind(&self.vs, ju, l);
                }
            }
        }
    }

    /// _yieldJ(i, e): must slot i wait at the mouth of e's junction?
    /// _yieldJ(i, e) with its write to the yield-blocker array as a value (None = not written).
    pub(crate) fn yield_j_core(&self, i: usize, e: usize) -> (bool, Option<i32>) {
        let g = self.gw();
        let j = g.j.as_ref().unwrap();
        if self.vs.conn[i] >= 0 || !(g.e_s_stop[e] < g.e_length[e]) { return (false, None); }
        let rp = self.vs.route_pos[i] as usize;
        if rp + 1 >= self.vs.routes[i].len() { return (false, None); }
        let m = j.movement(e, self.vs.lane[i] as i32, self.vs.routes[i][rp + 1]);
        if m < 0 { return (false, None); }
        let m = m as usize;
        let hl = f(self.vs.hl[i]);
        let d_s = jm::max(0.0, jm::min(g.e_s_stop[e], j.s_in(m)) - f(self.vs.s[i]) - hl);
        let v = f(self.vs.speed[i]);
        let l0 = self.vs.leader[i];
        macro_rules! wait { ($j:expr) => {{ return (true, Some($j as i32)); }}; }
        {
            let vs = &self.vs;
            if l0 >= 0 { let lu = l0 as usize; if vs.state[lu] == STATE_DRIVING && vs.speed[lu] < 1.0 && (vs.edge[lu] != e as i32 || vs.conn[lu] >= 0) && f(vs.gap[i]) - d_s - j.m_len[m] < f(vs.length[i]) + 1.0 { wait!(lu); } }
        }
        let left = j.turn(m) < -0.5;
        let my_stop = self.signed(i);
        let sight = self.opts.junction_sight >= 1.0;
        let confs = self.conflicts_ro(m);
        for c in confs.iter() {
            let o = c.other as usize;
            if c.ty == 0 { if let Some(list) = self.conn_cars.get(&o) { for &jj in list { if f(self.vs.s[jj]) - j.s_in(o) - f(self.vs.hl[jj]) < c.s_other + f(self.vs.hw[i]) + 1.0 { wait!(jj); } } } }
            let (o_in, o_lane, o_out) = (j.info(o, 0) as usize, j.info(o, 1) as i32, j.info(o, 2) as i32);
            let first = |s: &Sim, jj: usize, d2: f64| -> bool {
                let vs = &s.vs;
                let v2 = f(vs.speed[jj]);
                if d2 > 80.0 || s.must_stop(jj) { return false; }
                if v2 < 0.3 && vs.leader[jj] != -1 && f(vs.gap[jj]) < d2 + 3.0 { return false; }
                let t_i = (d_s + c.s) / jm::max(v, 2.0); let t_j = (d2 + c.s_other) / jm::max(v2, 2.0);
                let j_stop = s.signed(jj);
                let j_first = if my_stop != j_stop { my_stop }
                    else if g.e_rank[o_in] != g.e_rank[e] { g.e_rank[o_in] > g.e_rank[e] }
                    else if c.ty == 0 && left != (j.turn(o) < -0.5) { left }
                    else { t_j < t_i || (t_j == t_i && jj < i) };
                j_first && (t_j < 8.0 || d2 < 10.0)
            };
            if !sight {
                let vs = &self.vs;
                let mut jj = self.lanes.last(o_in as i32, o_lane);
                let mut steps = 0usize;
                while jj >= 0 && vs.conn[jj as usize] >= 0 {
                    jj = self.lanes.behind(vs, jj as usize, o_lane);
                    steps += 1; if steps > self.walk_limit() { self.walk_cycle("yield_j", i, o_in, o_lane); jj = -1; }
                }
                if jj < 0 || jj as usize == i || vs.state[jj as usize] != STATE_DRIVING { continue; }
                let ju = jj as usize;
                let rj = &vs.routes[ju]; let pj = vs.route_pos[ju] as usize;
                if pj + 1 >= rj.len() || rj[pj + 1] != o_out || (vs.refuel_state[ju] == 1 && vs.stop_pos[ju] == pj as i32) { continue; }
                if first(self, ju, jm::max(0.0, g.e_s_stop[o_in] - f(vs.s[ju]) - f(vs.hl[ju]))) { wait!(ju); }
                continue;
            }
            if let Some(sj) = self.see_approach(i, o_in, o_lane, o_out, &first) { wait!(sj); }
        }
        let node = g.e_to[e] as usize;
        let my_out = self.vs.routes[i][rp + 1];
        for q in g.in_start[node]..g.in_start[node + 1] {
            let o = g.in_edges[q as usize] as usize;
            if o == e || o as i32 == g.e_rev[my_out as usize] { continue; }
            for l in 0..g.e_lanes[o] {
                let mut jj = self.lanes.last(o as i32, l);
                let mut n = 0;
                while jj >= 0 && { n += 1; n - 1 < 3 } {
                    let ju = jj as usize;
                    let vs = &self.vs;
                    if vs.state[ju] == STATE_DRIVING && vs.conn[ju] < 0 {
                        let rj = &vs.routes[ju]; let pj = vs.route_pos[ju] as usize;
                        if !rj.is_empty() && pj + 1 < rj.len() && rj[pj + 1] != my_out && j.movement(o, l, rj[pj + 1]) < 0 {
                            let dj = g.e_length[o] - f(vs.s[ju]) - f(vs.hl[ju]);
                            if f(vs.s[ju]) + f(vs.hl[ju]) > g.e_s_stop[o] && dj > -f(vs.length[ju]) { wait!(ju); }
                            if g.e_rank[o] >= g.e_rank[e] && !self.signed(ju) && !self.must_stop(ju) && !my_stop {
                                let t_j = (g.e_s_stop[o] - f(vs.s[ju]) - f(vs.hl[ju])) / jm::max(f(vs.speed[ju]), 2.0);
                                if t_j < 6.0 { wait!(ju); }
                            }
                        }
                        break;
                    }
                    jj = self.lanes.behind(vs, ju, l);
                }
            }
        }
        (false, Some(-1))
    }

    #[allow(dead_code)]
    pub(crate) fn yield_j(&mut self, i: usize, e: usize) -> bool {
        let (r, w) = self.yield_j_core(i, e);
        if let Some(v) = w { self.set_yield_by(i, v); }
        r
    }
    pub fn set_yield_by(&mut self, i: usize, v: i32) {
        let cap = self.vs.capacity;
        match &mut self.yield_by { Some(yb) if yb.len() >= cap => {}, Some(yb) => yb.resize(cap, -1), None => self.yield_by = Some(vec![-1; cap]) }
        self.yield_by.as_mut().unwrap()[i] = v;
    }

    pub(crate) fn build_moves_into(g: &crate::world::World) -> Option<crate::fx::HashMap<u32, Vec<usize>>> {
        let j = g.j.as_ref()?;
        let mut m2: crate::fx::HashMap<u32, Vec<usize>> = Default::default();
        for m in 0..j.move_count { m2.entry(j.info(m, 2)).or_default().push(m); }
        Some(m2)
    }
    fn moves_into(&self, e: usize) -> Vec<usize> {
        // world-derived, built once per world (shared by every env on it)
        match self.g.mv_into.get_or_init(|| Self::build_moves_into(&self.g)) { Some(m) => m.get(&(e as u32)).cloned().unwrap_or_default(), None => Vec::new() }
    }

    fn see_approach(&self, i: usize, o_in: usize, o_lane: i32, o_out: i32, first: &dyn Fn(&Sim, usize, f64) -> bool) -> Option<usize> {
        let g = self.gw();
        let j = g.j.as_ref().unwrap();
        let r = 80.0;
        let ok = |s: &Sim, jj: usize, rp1: usize| -> bool {
            let vs = &s.vs;
            jj != i && vs.state[jj] == STATE_DRIVING && !(vs.refuel_state[jj] == 1 && vs.stop_pos[jj] == vs.route_pos[jj]) && !vs.routes[jj].is_empty()
                && vs.routes[jj].get(vs.route_pos[jj] as usize + rp1).copied() == Some(o_out)
        };
        let mut jj = self.lanes.last(o_in as i32, o_lane);
        let mut n = 0;
        let mut steps = 0usize;
        while jj >= 0 && n < 3 {
            steps += 1; if steps > self.walk_limit() { self.walk_cycle("see_approach", i, o_in, o_lane); break; }
            let ju = jj as usize;
            if self.vs.conn[ju] < 0 {
                let d2 = jm::max(0.0, g.e_s_stop[o_in] - f(self.vs.s[ju]) - f(self.vs.hl[ju]));
                if d2 > r { return None; }
                n += 1;
                if ok(self, ju, 1) && first(self, ju, d2) { return Some(ju); }
            }
            jj = self.lanes.behind(&self.vs, ju, o_lane);
        }
        let s_e = g.e_s_stop[o_in];
        if !(s_e < r) { return None; }
        let ups = self.moves_into(o_in);
        for m2 in ups {
            let (len2, s_in2, s_out2) = (j.m_len[m2], j.s_in(m2), j.s_out(m2));
            let base = jm::max(0.0, s_e - s_out2);
            if let Some(list) = self.conn_cars.get(&m2).cloned() {
                for k in list {
                    let d2 = base + jm::max(0.0, s_in2 + len2 - f(self.vs.s[k])) - f(self.vs.hl[k]);
                    if d2 <= r && ok(self, k, 2) && first(self, k, jm::max(0.0, d2)) { return Some(k); }
                }
            }
            let (in_e, in_l) = (j.info(m2, 0) as i32, j.info(m2, 1) as i32);
            let mut q = self.lanes.last(in_e, in_l);
            let mut c = 0;
            let mut steps = 0usize;
            while q >= 0 && c < 2 {
                steps += 1; if steps > self.walk_limit() { self.walk_cycle("see_approach_up", i, in_e as usize, in_l); break; }
                let qu = q as usize;
                if self.vs.conn[qu] < 0 {
                    let d2 = base + len2 + jm::max(0.0, s_in2 - f(self.vs.s[qu])) - f(self.vs.hl[qu]);
                    if d2 > r { break; }
                    c += 1;
                    let vs = &self.vs;
                    let nxt = vs.routes[qu].get(vs.route_pos[qu] as usize + 1).copied();
                    if !vs.routes[qu].is_empty() && nxt == Some(o_in as i32) && ok(self, qu, 2) && !self.must_stop(qu) && first(self, qu, d2) { return Some(qu); }
                }
                q = self.lanes.behind(&self.vs, qu, in_l);
            }
        }
        None
    }

    pub(crate) fn path_bodies_p(&self, p: &P2) {
        let i = p.i;
        let g = self.gw();
        let jo = g.j.as_ref();
        let v = f(self.vs.speed[i]);
        let d_max = jm::min(80.0, 10.0 + 1.5 * v + v * v / 8.0);
        let (hl, hw) = (f(self.vs.hl[i]), f(self.vs.hw[i]));
        let yaw = f(self.vs.heading[i]); let fx = jm::cos(yaw); let fz = jm::sin(yaw);
        let mut cand: Vec<usize> = Vec::new();
        self.hash.near_into(&self.vs.x, &self.vs.z, f(self.vs.x[i]) + fx * 0.5 * d_max, f(self.vs.z[i]) + fz * 0.5 * d_max, 0.5 * d_max + 9.0, &mut cand);
        {
            let vs = &self.vs;
            cand.retain(|&jj| {
                if jj == i { return false; }
                let st = vs.state[jj]; if st != STATE_DRIVING && st != STATE_CRASHED { return false; }
                let dx = f(vs.x[jj]) - f(vs.x[i]); let dz = f(vs.z[jj]) - f(vs.z[i]);
                if dx * fx + dz * fz < hl { return false; }
                if st == STATE_DRIVING && vs.speed[jj] < 0.5 && (p.leader_of(self, jj) == i as i32 || (p.leader_of(self, jj) == -3 && p.cw_of(self, jj) == Some(i))) { return false; }
                if st == STATE_DRIVING && i < jj && self.stuck_in(i) && self.stuck_in(jj) { return false; }
                true
            });
        }
        if cand.is_empty() { return; }
        let route = self.vs.routes[i].clone();
        let mut rp = self.vs.route_pos[i] as usize;
        let mut e = self.vs.edge[i] as usize;
        let mut m = self.vs.conn[i];
        let (mut pos, mut end, mut off);
        let mut lat = f(self.vs.lat[i]);
        let mut lane = self.vs.lane[i] as i32;
        off = 0.0;
        if m >= 0 { let j = jo.unwrap(); let mu = m as usize; pos = f(self.vs.s[i]) - j.s_in(mu); end = j.m_len[mu]; off = lat - (j.info(mu, 1) as f64 + 0.5) * g.lane_w(e, j.s_in(mu)); }
        else {
            pos = f(self.vs.s[i]); end = g.e_length[e];
            if let Some(j) = jo { if g.e_s_stop[e] < g.e_length[e] && !route.is_empty() && rp + 1 < route.len() { let mm = j.movement(e, lane, route[rp + 1]); if mm >= 0 { end = j.s_in(mm as usize); } } }
        }
        let clear = hw + 0.3;
        let (mut d, mut t, mut frames) = (0.0f64, hl + 0.5, 0);
        let result: Option<(f64, usize, f64, f64)>;
        'outer: loop {
            if t > d_max { result = None; break; }
            let rest = end - pos;
            if d + rest < t {
                d += jm::max(0.0, rest);
                frames += 1;
                if frames > 6 { result = None; break; }
                if m >= 0 {
                    let j = jo.unwrap(); let mu = m as usize;
                    let (oe, ol) = (j.info(mu, 2) as usize, j.info(mu, 3) as i32);
                    rp += 1; e = oe; pos = j.s_out(mu); m = -1;
                    lane = (g.e_lanes[e] - 1).min(ol); lat = (lane as f64 + 0.5) * g.lane_w(e, pos); end = g.e_length[e];
                    if g.e_s_stop[e] < g.e_length[e] && rp + 1 < route.len() { let mm = j.movement(e, lane, route[rp + 1]); if mm >= 0 { end = j.s_in(mm as usize); } }
                    continue;
                }
                if rp + 1 >= route.len() { result = None; break; }
                let mm = match jo { Some(j) if g.e_s_stop[e] < g.e_length[e] => j.movement(e, lane, route[rp + 1]), _ => -1 };
                if mm >= 0 { let j = jo.unwrap(); m = mm; let mu = mm as usize; pos = 0.0; end = j.m_len[mu]; off = lat - (j.info(mu, 1) as f64 + 0.5) * g.lane_w(e, j.s_in(mu)); continue; }
                let en = route[rp + 1] as usize; rp += 1;
                let nl = g.map_lane(e, en, lane); lane = (g.e_lanes[en] - 1).min(0.max(nl)); e = en; pos = 0.0; end = g.e_length[e]; lat = (lane as f64 + 0.5) * g.lane_w(e, 0.0);
                if let Some(j) = jo { if g.e_s_stop[e] < g.e_length[e] && rp + 1 < route.len() { let m2 = j.movement(e, lane, route[rp + 1]); if m2 >= 0 { end = j.s_in(m2 as usize); } } }
                continue;
            }
            let u = pos + (t - d);
            let pt = if m >= 0 { let mut q = jo.unwrap().connector_point(m as usize, u); q.x += off * q.hz; q.z -= off * q.hx; q }
                else if let Some(lg) = &g.lg { lg.lat_point(e, jm::min(u, g.e_length[e]), lat) } else { result = None; break };
            let vs = &self.vs;
            for &jj in &cand {
                let c = jm::cos(f(vs.heading[jj])); let sn = jm::sin(f(vs.heading[jj]));
                let dx = pt.x - f(vs.x[jj]); let dz = pt.z - f(vs.z[jj]);
                let lx = jm::abs(dx * c + dz * sn) - f(vs.hl[jj]); let lz = jm::abs(dz * c - dx * sn) - f(vs.hw[jj]);
                if lx < clear && lz < clear && jm::hypot(jm::max(lx, 0.0), jm::max(lz, 0.0)) < clear {
                    result = Some((t - hl - 0.5, jj, pt.hx, pt.hz));
                    break 'outer;
                }
            }
            t += 2.0;
        }
        if let Some((gg, jj, phx, phz)) = result {
            if gg < f(p.gap.get()) {
                let mut dpsi = f(self.vs.heading[jj]) - jm::atan2(phz, phx); dpsi = jm::atan2(jm::sin(dpsi), jm::cos(dpsi));
                p.gap.set(r32(gg));
                if self.vs.state[jj] == STATE_DRIVING && jm::abs(dpsi) < 0.6 { p.leader.set(jj as i32); }
                else { p.leader.set(-3); p.cw.set(Some(jj)); }
            }
        }
    }

    pub fn box_busy_j(&mut self, i: usize, e: usize) -> bool {
        let g = self.gw();
        let j = g.j.as_ref().unwrap();
        if self.vs.conn[i] >= 0 || !(g.e_s_stop[e] < g.e_length[e]) { return false; }
        let rp = self.vs.route_pos[i] as usize;
        if rp + 1 >= self.vs.routes[i].len() { return false; }
        let m = j.movement(e, self.vs.lane[i] as i32, self.vs.routes[i][rp + 1]);
        if m < 0 { return false; }
        let confs = self.conflicts(m as usize);
        for c in confs.iter() {
            if c.ty != 0 { continue; }
            let Some(list) = self.conn_cars.get(&(c.other as usize)) else { continue };
            for &jj in list { if f(self.vs.s[jj]) - j.s_in(c.other as usize) - f(self.vs.hl[jj]) < c.s_other + f(self.vs.hw[i]) + 1.0 { return true; } }
        }
        false
    }
}

/// Per-car state of the order-dependent leader perception (phase 2): own leader / gap / conflict write, and the reads
/// of other cars' leader / conflict that the slot-order semantics make order dependent. In speculative mode (parallel
/// pass) a car j < i is read at its value before its own phase 2 (its phase-1 base), and the read is recorded; the
/// ordered commit re-runs a car whose recorded reads differ from the final values.
pub(crate) struct P2<'a> {
    pub i: usize,
    pub gap: std::cell::Cell<f32>, pub leader: std::cell::Cell<i32>, pub cw: std::cell::Cell<Option<usize>>,
    pub spec: Option<&'a [Option<crate::leaders::Base>]>,
    pub reads: std::cell::RefCell<Vec<(usize, i32, Option<usize>)>>,
}
impl<'a> P2<'a> {
    pub fn new(i: usize, gap: f32, leader: i32, spec: Option<&'a [Option<crate::leaders::Base>]>) -> P2<'a> {
        P2 { i, gap: std::cell::Cell::new(gap), leader: std::cell::Cell::new(leader), cw: std::cell::Cell::new(None), spec, reads: std::cell::RefCell::new(Vec::new()) }
    }
    #[inline]
    fn early(&self, j: usize) -> Option<&crate::leaders::Base> { match self.spec { Some(b) if j < self.i => b.get(j).and_then(|x| x.as_ref()), _ => None } }
    #[inline]
    pub fn leader_of(&self, s: &Sim, j: usize) -> i32 {
        match self.early(j) { Some(b) => { let cw = self.cw_start(s, j, b); self.reads.borrow_mut().push((j, b.leader, cw)); b.leader } None => s.vs.leader[j] }
    }
    #[inline]
    pub fn cw_of(&self, s: &Sim, j: usize) -> Option<usize> {
        match self.early(j) { Some(b) => { let cw = self.cw_start(s, j, b); self.reads.borrow_mut().push((j, b.leader, cw)); cw } None => s.conflict_with.get(&j).copied() }
    }
    #[inline]
    fn cw_start(&self, s: &Sim, j: usize, b: &crate::leaders::Base) -> Option<usize> { b.cw.or_else(|| s.conflict_with.get(&j).copied()) }
}
