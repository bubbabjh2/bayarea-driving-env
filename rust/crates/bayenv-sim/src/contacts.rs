//! sim.js contacts and scenes: _contacts, _adjacentOrFreewayFree, _body, _writeBack, _barrier, _crashEvent,
//! _classifyCrash, _openScene, _moveOffLanes, _scenes; crashModel.js incidentTimeline / severityOf.
use crate::jm;
use crate::physics::{self, Body, Contact};
use crate::rng::Rng;
use crate::sim::{f, r32, Driver, Response, Scene, Sim, FAIL_CRASH};
use crate::vehicle::*;
use crate::world::{World, HW_PULLOVER};

pub const DISABLE_KMH: f64 = 10.0;
/// largest barrier penetration correction per step with SimOpts::barrier_soft_push (m)
pub const BARRIER_PUSH_MAX: f64 = 0.25;
/// SimOpts::mountable_kerbs: closing speed into a kerb below which it is mounted and scraped (m/s)
pub const KERB_MOUNT_SPEED: f64 = 2.0;
/// how far past the kerb line a mounting car may ride before the kerb pushes back (m)
pub const KERB_MOUNT_DEPTH: f64 = 0.3;
/// friction coefficient of a kerb scrape (tyre side wall and body on the kerb)
pub const KERB_FRICTION: f64 = 0.6;
/// a kerb scrape's reported delta-v is capped below the disable threshold (m/s)
pub const KERB_SCRAPE_MAX_DV: f64 = (DISABLE_KMH - 0.5) / 3.6;

/// reward.js crashShare: share of the closing speed along the normal that car A brought.
pub fn crash_share(speed_a: f64, yaw_a: f64, speed_b: f64, yaw_b: f64, nx: f64, nz: f64) -> f64 {
    let mut ca = speed_a * (jm::cos(yaw_a) * nx + jm::sin(yaw_a) * nz);
    let mut cb = -jm::or(speed_b, 0.0) * (jm::cos(jm::or(yaw_b, 0.0)) * nx + jm::sin(jm::or(yaw_b, 0.0)) * nz);
    if ca + cb < 0.0 { ca = -ca; cb = -cb; }
    ca = jm::max(0.0, ca); cb = jm::max(0.0, cb);
    let s = ca + cb;
    if s < 1e-3 { 0.5 } else { ca / s }
}

pub fn severity_of(dv: f64) -> &'static str { if dv < 10.0 { "minor" } else if dv < 25.0 { "moderate" } else if dv < 50.0 { "severe" } else { "very severe" } }

pub struct Timeline { pub notify: f64, pub arrive: f64, pub block: f64, pub clear: f64 }

/// crashModel.js incidentTimeline
pub fn incident_timeline(severity: &str, rng: &mut Rng) -> Timeline {
    // (median, sigma, range lo, hi, blockFrac (None = towAfterArrival 5 min))
    let (median, sigma, lo, hi, bf): (f64, f64, f64, f64, Option<f64>) = match severity {
        "severe" => (50.0 * 60.0, 0.25, 45.0, 75.0, Some(0.8)),
        "very severe" => (150.0 * 60.0, 0.3, 120.0, 240.0, Some(0.9)),
        "minor" => (20.0 * 60.0, 0.4, 15.0, 30.0, Some(0.0)),
        _ => (20.0 * 60.0, 0.4, 10.0, 30.0, None),
    };
    let notify = 3.4 * 60.0 * jm::exp(0.3 * rng.gaussian());
    let arrive = notify + 6.0 * 60.0 * jm::exp(0.3 * rng.gaussian());
    let clear = jm::min(hi * 60.0, jm::max(lo * 60.0, median * jm::exp(sigma * rng.gaussian())));
    let mut block = match bf { None => arrive + 5.0 * 60.0 * jm::exp(0.3 * rng.gaussian()), Some(b) => b * clear };
    block = jm::min(block, clear);
    Timeline { notify, arrive, block, clear: jm::max(clear, block + 60.0) }
}

/// Pre-impact state of a contact (speeds, headings, normal).
#[derive(Clone, Copy, Default)]
pub struct Pre { pub speed: [f64; 2], pub heading: [f64; 2], pub nx: f64, pub nz: f64 }

/// frameDir: road-frame direction at slot i (connector or edge).
pub(crate) fn frame_dir(g: &World, vs: &VehicleStore, i: usize) -> (f64, f64) {
    let m = vs.conn[i];
    if m >= 0 { if let Some(j) = &g.j { let p = j.connector_point(m as usize, f(vs.s[i]) - j.s_in(m as usize)); return (p.hx, p.hz); } }
    let p = g.edge_point_dir(vs.edge[i] as usize, f(vs.s[i]), vs.seg[i]);
    (p.hx, p.hz)
}
pub(crate) fn frame_end(g: &World, vs: &VehicleStore, i: usize) -> f64 {
    let m = vs.conn[i];
    if m >= 0 { if let Some(j) = &g.j { return j.s_in(m as usize) + j.m_len[m as usize]; } }
    g.e_length[vs.edge[i] as usize]
}

impl Sim {
    pub(crate) fn contacts(&mut self) {
        self.reverse_stash.clear();
        let g = self.gw();
        let high = self.vs.high;
        let mut now: crate::fx::HashSet<i64> = std::collections::HashSet::default();
        let layer = self.opts.layer_gap;
        let mut cand = std::mem::take(&mut self.near_buf);
        // parallel pre-scan: which cars would touch a barrier or have a swept contact with a later car, against the
        // state at the start of the pass. The ordered pass below only runs cars the scan flags, cars changed earlier in
        // the pass, and cars with a changed car near them (exact: everything else would find nothing).
        let par = self.opts.parallel && high >= 512;
        let tscan = std::time::Instant::now();
        let scan: Vec<bool> = if par {
            use rayon::prelude::*;
            let me = &*self;
            (0..high).into_par_iter().with_min_len(64).map(|i| me.contact_scan(i)).collect()
        } else { Vec::new() };
        if self.profile { self.sub_ns[8] += tscan.elapsed().as_nanos() as u64; }
        let mut touched = vec![false; if par { high } else { 0 }];
        let mut touched_list: Vec<usize> = Vec::new();
        for i in 0..high {
            let si = self.vs.state[i];
            if si != STATE_DRIVING && si != STATE_CRASHED { continue; }
            if par && !scan[i] && !touched[i] {
                let vs = &self.vs;
                let r = jm::hypot(f(vs.hl[i]), f(vs.hw[i])) + 6.5 + jm::hypot(f(vs.x[i]) - f(vs.px[i]), f(vs.z[i]) - f(vs.pz[i])) + 5.0 + 1.0;
                let (xi, zi) = (f(vs.x[i]), f(vs.z[i]));
                if !touched_list.iter().any(|&k| k > i && jm::hypot(f(vs.x[k]) - xi, f(vs.z[k]) - zi) <= r) { continue; }
            }
            let mov_i = si == STATE_DRIVING || self.vs.speed[i].abs() > 0.01 || jm::abs(f(self.vs.v_lat[i])) > 0.01;
            if si == STATE_DRIVING && !self.empty && self.vs.conn[i] < 0 {
                let eb = self.vs.edge[i] as usize;
                let s = f(self.vs.s[i]);
                if !(g.j.is_some() && (s < g.e_s_exit[eb] + 1.0 || s > g.e_s_stop[eb])) && self.barrier(i) && par && !touched[i] { touched[i] = true; touched_list.push(i); }
            } else if si == STATE_DRIVING && !self.empty && self.opts.connector_barrier && g.j.is_some() {
                if self.connector_barrier(i, true).is_some() && par && !touched[i] { touched[i] = true; touched_list.push(i); }
            }
            if self.empty && si == STATE_DRIVING { continue; }
            let vs = &self.vs;
            let sweep_i = jm::hypot(f(vs.x[i]) - f(vs.px[i]), f(vs.z[i]) - f(vs.pz[i]));
            let hd_i = jm::hypot(f(vs.hl[i]), f(vs.hw[i]));
            let (xi, zi, yi) = (f(vs.x[i]), f(vs.z[i]), f(vs.y[i]));
            self.hash.near_after_into(&vs.x, &vs.z, xi, zi, hd_i + 6.5 + sweep_i + 5.0, i, &mut cand);
            for &j in cand.iter() {
                let vs = &self.vs;
                let sj = vs.state[j];
                if sj != STATE_DRIVING && sj != STATE_CRASHED { continue; }
                let mov_j = sj == STATE_DRIVING || vs.speed[j].abs() > 0.01 || jm::abs(f(vs.v_lat[j])) > 0.01;
                if !mov_i && !mov_j { continue; }
                if jm::abs(f(vs.y[j]) - yi) > layer { continue; }
                if vs.edge[j] == vs.edge[i] && jm::abs(f(vs.s[j]) - f(vs.s[i])) > f(vs.hl[i]) + f(vs.hl[j]) + 1.0 { continue; }
                if vs.edge[j] != vs.edge[i] && !adjacent_or_freeway_free(&g, vs.edge[i] as usize, vs.edge[j] as usize) { continue; }
                let sweep_j = jm::hypot(f(vs.x[j]) - f(vs.px[j]), f(vs.z[j]) - f(vs.pz[j]));
                let hd_j = jm::hypot(f(vs.hl[j]), f(vs.hw[j]));
                if jm::hypot(f(vs.x[j]) - xi, f(vs.z[j]) - zi) > hd_i + hd_j + sweep_i + sweep_j { continue; }
                let (mut a, a0) = body(vs, i, mov_i, self.opts.wreck_real_mass);
                let (mut b, b0) = body(vs, j, mov_j, self.opts.wreck_real_mass);
                let Some((t, c, ta, tb)) = physics::swept_contact(a0, &a, b0, &b, 0.5) else { continue };
                if t < 1.0 { a.x = ta.0; a.z = ta.1; b.x = tb.0; b.z = tb.1; }
                let key = i as i64 * 1048576 + j as i64;
                now.insert(key);
                let pre = Pre { speed: [jm::hypot(a.vx, a.vz), jm::hypot(b.vx, b.vz)], heading: [a.yaw, b.yaw], nx: c.nx, nz: c.nz };
                let res = physics::resolve_contact(&mut a, &mut b, &c, physics::RESTITUTION, physics::FRICTION);
                physics::separate(&mut a, &mut b, &c);
                self.write_back(i, &a); self.write_back(j, &b);
                if par { for q in [i, j] { if !touched[q] { touched[q] = true; touched_list.push(q); } } }
                if self.opts.contact_episodes {
                    // contact episode: delta-v accumulated while the pair keeps touching; every step whose own
                    // delta-v reaches touch_kmh reports, with the episode total as the severity
                    let ongoing = self.touch.contains(&key);
                    let impact = !ongoing || jm::max(res.dva, res.dvb) * 3.6 >= self.opts.touch_kmh;
                    let (ta, tb) = if ongoing { self.touch_ep.get(&key).copied().unwrap_or((0.0, 0.0)) } else { (0.0, 0.0) };
                    let (ta, tb) = if impact { (ta + res.dva, tb + res.dvb) } else { (ta, tb) };
                    self.touch_ep_now.insert(key, (ta, tb));
                    if res.jn > 0.0 && (si == STATE_DRIVING || sj == STATE_DRIVING) && impact {
                        self.crash_event_ep(&[i, j], &[ta, tb], &c, res.closing, "collision", Some(pre), ongoing);
                    }
                } else if !self.touch.contains(&key) && res.jn > 0.0 && (si == STATE_DRIVING || sj == STATE_DRIVING) {
                    self.crash_event(&[i, j], &[res.dva, res.dvb], &c, res.closing, "collision", Some(pre));
                }
            }
        }
        self.near_buf = cand;
        self.touch = now;
        self.touch_ep = std::mem::take(&mut self.touch_ep_now);
    }

    /// Lateral limits of a car on a junction connector (SimOpts::connector_barrier): the kerbs of its in-edge at the
    /// connector's start, in the in-edge lateral coordinate the connector keeps (`lat`); the wall normal from the
    /// connector's tangent at the car. Returns (side, depth, normal) when the car's footprint is past one.
    fn connector_limits(&self, i: usize) -> Option<(f64, f64, f64, f64)> {
        let g = self.gw();
        let j = g.j.as_ref()?;
        let vs = &self.vs;
        let m = vs.conn[i]; if m < 0 { return None; }
        let m = m as usize;
        let e = vs.edge[i] as usize;
        let s_in = j.s_in(m);
        let w = g.lane_w(e, s_in);
        let (mut pad_r, mut pad_l) = (0.0, g.e_lanes[e] as f64 * w);
        if let Some(lg) = &g.lg { let bd = lg.bounds(e, jm::min(s_in, g.e_length[e])); pad_r = bd.r - bd.right; pad_l = bd.r - bd.left; }
        let right = pad_r - if g.is_shoulder_road(e) { self.opts.freeway_shoulder } else { self.opts.shoulder };
        let left = pad_l + if g.e_rev[e] >= 0 { jm::or(g.e_opp[e] as f64, g.e_lanes[e] as f64) * w + self.opts.opposite_margin } else { self.opts.shoulder };
        let hr = f(vs.heading_rel[i]);
        let ext = jm::abs(jm::sin(hr)) * f(vs.hl[i]) + jm::abs(jm::cos(hr)) * f(vs.hw[i]);
        let lat = f(vs.lat[i]);
        let (side, depth) = if lat - ext < right { (-1.0, right - (lat - ext)) } else if lat + ext > left { (1.0, lat + ext - left) } else { return None };
        let p = j.connector_point(m, f(vs.s[i]) - s_in);
        let (nx, nz) = if side > 0.0 { (p.hz, -p.hx) } else { (-p.hz, p.hx) };
        Some((side, depth, nx, nz))
    }
    /// slot i touched a barrier (edge or connector wall) in the last contacts pass
    pub fn touching_barrier(&self, i: usize) -> bool { self.barrier_touch.contains(&i) }
    fn connector_barrier_hit(&self, i: usize) -> bool { self.connector_limits(i).is_some() }
    /// barrier() for a car on a junction connector (SimOpts::connector_barrier; same wall impulse, events and
    /// position correction as an edge barrier)
    pub(crate) fn connector_barrier(&mut self, i: usize, _from_contacts: bool) -> Option<()> {
        let g = self.gw();
        let (side, depth, nx, nz) = self.connector_limits(i)?;
        let vs = &self.vs;
        let (hl, hw) = (f(vs.hl[i]), f(vs.hw[i]));
        let yaw = f(vs.heading[i]);
        let (fx, fz) = (jm::cos(yaw), jm::sin(yaw));
        // the footprint corner deepest past the wall (the normal points from the road toward the wall, as barrier())
        let mut best = f64::NEG_INFINITY; let (mut cx, mut cz) = (f(vs.x[i]), f(vs.z[i]));
        for a in [-1.0f64, 1.0] { for b2 in [-1.0f64, 1.0] {
            let (px, pz) = (f(vs.x[i]) + a * hl * fx + b2 * hw * fz, f(vs.z[i]) + a * hl * fz - b2 * hw * fx);
            let d = px * nx + pz * nz;
            if d > best { best = d; cx = px; cz = pz; }
        } }
        let (mut a, _) = body(vs, i, true, self.opts.wreck_real_mass);
        let mut wb = Body { x: cx, z: cz, ..Default::default() };
        let c = Contact { depth, nx, nz, px: cx, pz: cz };
        let pre = Pre { speed: [jm::hypot(a.vx, a.vz), 0.0], heading: [a.yaw, 0.0], nx, nz };
        // mountable_kerbs: a kerb (not a rigid barrier: freeway system, bridges) met at a low closing speed is mounted
        // and scraped: no rebound, kerb friction, the car may ride up to KERB_MOUNT_DEPTH past the kerb line, a minor
        // contact that never disables; faster strikes hit it like the wall it was before
        // closing speed at the contact point (body velocity + yaw rate x lever arm), as resolve_contact sees it
        let into = (a.vx - a.w * (c.pz - a.z)) * nx + (a.vz + a.w * (c.px - a.x)) * nz;
        let vs_edge_i = self.vs.edge[i] as usize;
        let scrape = self.opts.mountable_kerbs && !g.rigid_edge(vs_edge_i) && into < KERB_MOUNT_SPEED;
        let res = if scrape { physics::resolve_contact(&mut a, &mut wb, &c, 0.0, KERB_FRICTION) } else { physics::resolve_contact(&mut a, &mut wb, &c, physics::WALL_RESTITUTION, physics::WALL_FRICTION) };
        self.write_back(i, &a);
        let corr = if scrape { jm::min(jm::max(0.0, depth - KERB_MOUNT_DEPTH), BARRIER_PUSH_MAX) } else if self.opts.barrier_soft_push { jm::min(depth, BARRIER_PUSH_MAX) } else { depth };
        let vs = &mut self.vs;
        vs.lat[i] = r32(f(vs.lat[i]) - side * corr);
        let e = vs.edge[i] as usize;
        vs.update_lanes(i, &g, e); vs.update_pose(i, &g);
        let was = self.barrier_touch.contains(&i);
        self.barrier_now.insert(i);
        if self.opts.contact_episodes {
            // impacts of the episode (the first touch, then steps with delta-v >= touch_kmh) add up; a resting push
            // (small per-step impulses) neither adds nor reports
            let impact = !was || res.dva * 3.6 >= self.opts.touch_kmh;
            let prev = if was { self.barrier_ep.get(&i).copied().unwrap_or(0.0) } else { 0.0 };
            if scrape {
                // a kerb scrape: reported once (first touch, delta-v below the disable threshold), adds nothing
                self.barrier_ep_now.insert(i, prev);
                if res.jn > 0.0 && !was { self.crash_event_ep(&[i], &[jm::min(res.dva, KERB_SCRAPE_MAX_DV)], &c, res.closing, "barrier", Some(pre), false); }
            } else {
                // on a kerb (not rigid), only real strikes (own delta-v >= KERB_MOUNT_SPEED) add to the episode: a grind of
                // small pushes never sums to a crash
                let counts = impact && (!self.opts.mountable_kerbs || g.rigid_edge(vs_edge_i) || res.dva >= KERB_MOUNT_SPEED);
                let tot = prev + if counts { res.dva } else { 0.0 };
                self.barrier_ep_now.insert(i, tot);
                if res.jn > 0.0 && (counts || !was) { self.crash_event_ep(&[i], &[tot], &c, res.closing, "barrier", Some(pre), was); }
            }
        } else if !was && res.jn > 0.0 { self.crash_event(&[i], &[if scrape { jm::min(res.dva, KERB_SCRAPE_MAX_DV) } else { res.dva }], &c, res.closing, "barrier", Some(pre)); }
        Some(())
    }

    pub(crate) fn write_back(&mut self, i: usize, b: &Body) {
        if b.inv_mass == 0.0 { return; }
        let g = self.gw();
        let vs = &mut self.vs;
        let e = vs.edge[i] as usize;
        let yaw = f(vs.heading[i]);
        let (fx, fz) = (jm::cos(yaw), jm::sin(yaw));
        let mut u = b.vx * fx + b.vz * fz;
        let v = b.vx * fz - b.vz * fx;
        // impact_reverse: a wreck keeps its backward speed; a driving car (no reverse gear) keeps it only if this
        // contact disables it (open_scene applies the stash)
        let rev = self.opts.impact_reverse && u < 0.0;
        let keep = rev && vs.state[i] == STATE_CRASHED;
        if u < 0.0 && !keep { u = 0.0; }
        vs.speed[i] = r32(u);
        if rev && !keep { self.reverse_stash.insert(i, b.vx * fx + b.vz * fz); }
        let vs = &mut self.vs;
        vs.v_lat[i] = r32(v); vs.yaw_rate[i] = r32(-b.w);
        let dx = b.x - b.x0; let dz = b.z - b.z0;
        if dx != 0.0 || dz != 0.0 {
            let (hx, hz) = frame_dir(&g, vs, i);
            let ds = dx * hx + dz * hz; let dl = dx * hz - dz * hx;
            vs.s[i] = r32(jm::min(frame_end(&g, vs, i), jm::max(0.0, f(vs.s[i]) + ds)));
            vs.lat[i] = r32(f(vs.lat[i]) + dl);
            vs.update_lanes(i, &g, e); vs.update_pose(i, &g);
        }
    }

    /// Pre-scan of the contact pass for slot i (pure): would its barrier act, or would a swept contact with a later
    /// car j > i be found, at the current state?
    fn contact_scan(&self, i: usize) -> bool {
        let g = self.gw();
        let vs = &self.vs;
        let si = vs.state[i];
        if si != STATE_DRIVING && si != STATE_CRASHED { return false; }
        if si == STATE_DRIVING && !self.empty && vs.conn[i] < 0 {
            let eb = vs.edge[i] as usize;
            let s = f(vs.s[i]);
            if !(g.j.is_some() && (s < g.e_s_exit[eb] + 1.0 || s > g.e_s_stop[eb])) && self.barrier_hit(i) { return true; }
        } else if si == STATE_DRIVING && !self.empty && self.opts.connector_barrier && g.j.is_some() && self.connector_barrier_hit(i) { return true; }
        if self.empty && si == STATE_DRIVING { return false; }
        let mov_i = si == STATE_DRIVING || vs.speed[i].abs() > 0.01 || jm::abs(f(vs.v_lat[i])) > 0.01;
        let layer = self.opts.layer_gap;
        let sweep_i = jm::hypot(f(vs.x[i]) - f(vs.px[i]), f(vs.z[i]) - f(vs.pz[i]));
        let hd_i = jm::hypot(f(vs.hl[i]), f(vs.hw[i]));
        let (xi, zi, yi) = (f(vs.x[i]), f(vs.z[i]), f(vs.y[i]));
        let mut found = false;
        self.hash.for_each_near(&vs.x, &vs.z, xi, zi, hd_i + 6.5 + sweep_i + 5.0, |j| {
            if j <= i { return true; }
            let sj = vs.state[j];
            if sj != STATE_DRIVING && sj != STATE_CRASHED { return true; }
            let mov_j = sj == STATE_DRIVING || vs.speed[j].abs() > 0.01 || jm::abs(f(vs.v_lat[j])) > 0.01;
            if !mov_i && !mov_j { return true; }
            if jm::abs(f(vs.y[j]) - yi) > layer { return true; }
            if vs.edge[j] == vs.edge[i] && jm::abs(f(vs.s[j]) - f(vs.s[i])) > f(vs.hl[i]) + f(vs.hl[j]) + 1.0 { return true; }
            if vs.edge[j] != vs.edge[i] && !adjacent_or_freeway_free(g, vs.edge[i] as usize, vs.edge[j] as usize) { return true; }
            let sweep_j = jm::hypot(f(vs.x[j]) - f(vs.px[j]), f(vs.z[j]) - f(vs.pz[j]));
            let hd_j = jm::hypot(f(vs.hl[j]), f(vs.hw[j]));
            if jm::hypot(f(vs.x[j]) - xi, f(vs.z[j]) - zi) > hd_i + hd_j + sweep_i + sweep_j { return true; }
            let (a, a0) = body(vs, i, mov_i, self.opts.wreck_real_mass);
            let (b, b0) = body(vs, j, mov_j, self.opts.wreck_real_mass);
            if physics::swept_contact(a0, &a, b0, &b, 0.5).is_some() { found = true; return false; }
            true
        });
        found
    }

    /// Returns whether slot i touched a barrier (state changed).
    pub(crate) fn barrier(&mut self, i: usize) -> bool {
        let g = self.gw();
        let vs = &self.vs;
        let e = vs.edge[i] as usize;
        let s = f(vs.s[i]);
        let w = g.lane_w(e, s);
        let (mut pad_r, mut pad_l) = (0.0, g.e_lanes[e] as f64 * w);
        if let Some(lg) = &g.lg { let bd = lg.bounds(e, s); pad_r = bd.r - bd.right; pad_l = bd.r - bd.left; }
        else { let taper = if f(vs.taper[i]) > s { w } else { 0.0 }; if vs.lat[i] < 0.0 { pad_r -= taper; } if f(vs.lat[i]) > pad_l { pad_l += taper; } }
        let right = pad_r - if g.is_shoulder_road(e) { self.opts.freeway_shoulder } else { self.opts.shoulder };
        let left = pad_l + if g.e_rev[e] >= 0 { jm::or(g.e_opp[e] as f64, g.e_lanes[e] as f64) * w + self.opts.opposite_margin } else { self.opts.shoulder };
        let hr = f(vs.heading_rel[i]);
        let (sh, ch) = (jm::sin(hr), jm::cos(hr));
        let (hl, hw, lat) = (f(vs.hl[i]), f(vs.hw[i]), f(vs.lat[i]));
        let ext = jm::abs(sh) * hl + jm::abs(ch) * hw;
        let (mut side, mut depth) = (0.0f64, 0.0f64);
        if lat - ext < right { side = -1.0; depth = right - (lat - ext); }
        else if lat + ext > left { side = 1.0; depth = lat + ext - left; }
        if side == 0.0 { return false; }
        let mut best = 1e9;
        let (mut cx, mut cz) = (0.0, 0.0);
        let yaw = f(vs.heading[i]);
        let (fx, fz) = (jm::cos(yaw), jm::sin(yaw));
        for a in [-1.0f64, 1.0] { for b2 in [-1.0f64, 1.0] {
            let l = -side * (a * hl * sh + b2 * hw * ch);
            if l < best { best = l; cx = f(vs.x[i]) + a * hl * fx + b2 * hw * fz; cz = f(vs.z[i]) + a * hl * fz - b2 * hw * fx; }
        } }
        let p = g.edge_point_dir(e, s, vs.seg[i]);
        let nx = if side > 0.0 { p.hz } else { -p.hz };
        let nz = if side > 0.0 { -p.hx } else { p.hx };
        let (mut a, _) = body(vs, i, true, self.opts.wreck_real_mass);
        let mut wb = Body { x: cx, z: cz, ..Default::default() };
        let c = Contact { depth, nx, nz, px: cx, pz: cz };
        let pre = Pre { speed: [jm::hypot(a.vx, a.vz), 0.0], heading: [a.yaw, 0.0], nx, nz };
        // mountable_kerbs: a kerb (not a rigid barrier: freeway system, bridges) met at a low closing speed is mounted
        // and scraped: no rebound, kerb friction, the car may ride up to KERB_MOUNT_DEPTH past the kerb line, a minor
        // contact that never disables; faster strikes hit it like the wall it was before
        // closing speed at the contact point (body velocity + yaw rate x lever arm), as resolve_contact sees it
        let into = (a.vx - a.w * (c.pz - a.z)) * nx + (a.vz + a.w * (c.px - a.x)) * nz;
        let vs_edge_i = self.vs.edge[i] as usize;
        let scrape = self.opts.mountable_kerbs && !g.rigid_edge(vs_edge_i) && into < KERB_MOUNT_SPEED;
        let res = if scrape { physics::resolve_contact(&mut a, &mut wb, &c, 0.0, KERB_FRICTION) } else { physics::resolve_contact(&mut a, &mut wb, &c, physics::WALL_RESTITUTION, physics::WALL_FRICTION) };
        self.write_back(i, &a);
        let corr = if scrape { jm::min(jm::max(0.0, depth - KERB_MOUNT_DEPTH), BARRIER_PUSH_MAX) } else if self.opts.barrier_soft_push { jm::min(depth, BARRIER_PUSH_MAX) } else { depth };
        let vs = &mut self.vs;
        vs.lat[i] = r32(f(vs.lat[i]) - side * corr);
        vs.update_lanes(i, &g, e); vs.update_pose(i, &g);
        let was = self.barrier_touch.contains(&i);
        self.barrier_now.insert(i);
        if self.opts.contact_episodes {
            // impacts of the episode (the first touch, then steps with delta-v >= touch_kmh) add up; a resting push
            // (small per-step impulses) neither adds nor reports
            let impact = !was || res.dva * 3.6 >= self.opts.touch_kmh;
            let prev = if was { self.barrier_ep.get(&i).copied().unwrap_or(0.0) } else { 0.0 };
            if scrape {
                // a kerb scrape: reported once (first touch, delta-v below the disable threshold), adds nothing
                self.barrier_ep_now.insert(i, prev);
                if res.jn > 0.0 && !was { self.crash_event_ep(&[i], &[jm::min(res.dva, KERB_SCRAPE_MAX_DV)], &c, res.closing, "barrier", Some(pre), false); }
            } else {
                // on a kerb (not rigid), only real strikes (own delta-v >= KERB_MOUNT_SPEED) add to the episode: a grind of
                // small pushes never sums to a crash
                let counts = impact && (!self.opts.mountable_kerbs || g.rigid_edge(vs_edge_i) || res.dva >= KERB_MOUNT_SPEED);
                let tot = prev + if counts { res.dva } else { 0.0 };
                self.barrier_ep_now.insert(i, tot);
                if res.jn > 0.0 && (counts || !was) { self.crash_event_ep(&[i], &[tot], &c, res.closing, "barrier", Some(pre), was); }
            }
        } else if !was && res.jn > 0.0 { self.crash_event(&[i], &[if scrape { jm::min(res.dva, KERB_SCRAPE_MAX_DV) } else { res.dva }], &c, res.closing, "barrier", Some(pre)); }
        true
    }

    /// Whether _barrier(i) would act (pure; same expressions).
    fn barrier_hit(&self, i: usize) -> bool {
        let g = self.gw();
        let vs = &self.vs;
        let e = vs.edge[i] as usize;
        let s = f(vs.s[i]);
        let w = g.lane_w(e, s);
        let (mut pad_r, mut pad_l) = (0.0, g.e_lanes[e] as f64 * w);
        if let Some(lg) = &g.lg { let bd = lg.bounds(e, s); pad_r = bd.r - bd.right; pad_l = bd.r - bd.left; }
        else { let taper = if f(vs.taper[i]) > s { w } else { 0.0 }; if vs.lat[i] < 0.0 { pad_r -= taper; } if f(vs.lat[i]) > pad_l { pad_l += taper; } }
        let right = pad_r - if g.is_shoulder_road(e) { self.opts.freeway_shoulder } else { self.opts.shoulder };
        let left = pad_l + if g.e_rev[e] >= 0 { jm::or(g.e_opp[e] as f64, g.e_lanes[e] as f64) * w + self.opts.opposite_margin } else { self.opts.shoulder };
        let hr = f(vs.heading_rel[i]);
        let (sh, ch) = (jm::sin(hr), jm::cos(hr));
        let (hl, hw, lat) = (f(vs.hl[i]), f(vs.hw[i]), f(vs.lat[i]));
        let ext = jm::abs(sh) * hl + jm::abs(ch) * hw;
        let (mut side, mut depth) = (0.0f64, 0.0f64);
        if lat - ext < right { side = -1.0; depth = right - (lat - ext); }
        else if lat + ext > left { side = 1.0; depth = lat + ext - left; }
        let _ = depth;
        side != 0.0
    }

    /// _crashEvent: a new physical contact.
    pub(crate) fn crash_event(&mut self, slots: &[usize], dvs: &[f64], c: &Contact, closing: f64, cause: &'static str, pre: Option<Pre>) {
        self.crash_event_ep(slots, dvs, c, closing, cause, pre, false)
    }
    /// crash_event; `ongoing` (contact_episodes): a later step of a contact episode, reported without the pair
    /// repeat filter (dvs = the episode totals)
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn crash_event_ep(&mut self, slots: &[usize], dvs: &[f64], c: &Contact, closing: f64, cause: &'static str, pre: Option<Pre>, ongoing: bool) {
        let kmh: Vec<f64> = dvs.iter().map(|d| d * 3.6).collect();
        let max_dv = kmh.iter().fold(f64::NEG_INFINITY, |a, &b| jm::max(a, b));
        if max_dv < self.opts.touch_kmh { self.counters.touches += 1; return; }
        let mut repeat = false;
        if slots.len() == 2 && !ongoing {
            let (a, b) = (slots[0].min(slots[1]), slots[0].max(slots[1]));
            let key = a as i64 * 1048576 + b as i64;
            let last = self.pair_event.get(&key).copied();
            repeat = matches!(last, Some(t) if t >= f(self.vs.spawn_time[a]) && t >= f(self.vs.spawn_time[b]));
            if !repeat { self.pair_event.insert(key, self.time); }
        }
        if repeat {
            let mut dis = false;
            for (k, &s) in slots.iter().enumerate() { if self.vs.state[s] == STATE_DRIVING && kmh[k] >= DISABLE_KMH { dis = true; } }
            if !dis { self.counters.repeat_contacts += 1; return; }
        }
        let severity = severity_of(max_dv);
        let mut disabled = Vec::new();
        for (k, &s) in slots.iter().enumerate() {
            self.vs.flags[s] |= FLAG_COLLIDED;
            if self.vs.state[s] == STATE_DRIVING && kmh[k] >= DISABLE_KMH { disabled.push(s); }
        }
        if slots.iter().any(|&s| self.vs.flags[s] & FLAG_SPEEDING != 0) { self.counters.crashes_while_speeding += 1; }
        self.counters.crashes += 1;
        *self.counters.by_severity.entry(severity).or_insert(0) += 1;
        let secondary = slots.iter().any(|&s| self.vs.state[s] == STATE_CRASHED);
        if secondary { self.counters.secondary_crashes += 1; }
        let e = self.vs.edge[slots[0]] as usize;
        let tl = incident_timeline(severity, &mut self.rng);
        let (crash_type, maneuver) = self.classify_crash(slots, c, pre.as_ref());
        for (k, &sl) in slots.iter().enumerate() {
            let share = match (slots.len(), pre) {
                (2, Some(p)) => crash_share(p.speed[k], p.heading[k], p.speed[1 - k], p.heading[1 - k], p.nx, p.nz),
                (_, Some(p)) => crash_share(p.speed[0], p.heading[0], 0.0, 0.0, p.nx, p.nz),
                _ => 0.5,
            };
            let driving = self.vs.state[sl] == STATE_DRIVING;
            self.collided_last.push(sl);
            if driving { self.events.crashed.push(crate::sim::CrashedEv { slot: sl, agent: self.vs.commuter[sl], dv_kmh: kmh[k], share, disabled: kmh[k] >= DISABLE_KMH }); }
            // one row per car of the event: CRASHED = disabled by it (driving, own delta-v >= 10 km/h), else CONTACT
            let (other, odv) = if slots.len() == 2 { (slots[1 - k] as i32, kmh[1 - k] as f32) } else { (-1, f32::NAN) };
            self.emit(if driving && kmh[k] >= DISABLE_KMH { crate::sim::EV_CRASHED } else { crate::sim::EV_CONTACT }, sl, other, kmh[k] as f32, odv, share as f32, f32::NAN);
        }
        // the involved drivers step once (collision signal); only the braking prop has state
        for &s in slots { if self.vs.state[s] == STATE_DRIVING && matches!(self.drivers[s], Driver::PropBraking { .. }) { let _ = self.driver_cmd(s); } }
        if !disabled.is_empty() {
            self.counters.disabling_crashes += 1;
            self.counters.crashed_vehicles += disabled.len() as u64;
            for &s in &disabled { self.vs.state[s] = STATE_CRASHED; self.vs.flags[s] |= FLAG_DISABLED; self.vs.clear_controls(s); }
            let ri = self.open_scene(&disabled, cause, severity, tl.block, tl.clear, e, true, None);
            let r = &mut self.crashes[ri];
            r.delta_v_kmh = kmh; r.closing = closing; r.disabled = disabled.clone(); r.crash_type = crash_type; r.maneuver = maneuver;
            for &s in &disabled { self.fail_trip(s, FAIL_CRASH, false); }
            return;
        }
        self.counters.minor_contacts += 1;
        if max_dv >= 3.0 {
            for &sl in slots {
                let ee = self.vs.edge[sl] as usize;
                if self.vs.state[sl] == STATE_DRIVING && self.g.e_hw[ee] & HW_PULLOVER != 0 { self.vs.pull_until[sl] = r32(self.time + 300.0 + 600.0 * self.rng.random()); self.counters.pull_overs += 1; }
            }
        }
        let id = self.crashes.len();
        self.crashes.push(Scene { id, time: self.time, cause, severity, is_crash: true, scene: false, vehicles: slots.to_vec(), commuters: slots.iter().map(|&s| self.vs.commuter[s]).collect(),
            edge: e as i32, s: 0.0, lane_mask: 0, lanes_blocked: 0, x: c.px, z: c.pz, y: f(self.vs.y[slots[0]]), block_start: self.time, block_end: self.time, clear_end: self.time + tl.clear,
            squeeze_at: 0.0, squeeze: false, response: Response::Cleared, permanent: false, gone: Vec::new(), delta_v_kmh: kmh, closing, disabled: Vec::new(), crash_type, maneuver });
    }

    fn classify_crash(&self, slots: &[usize], c: &Contact, pre: Option<&Pre>) -> (String, &'static str) {
        let g = &*self.g;
        let vs = &self.vs;
        let a = slots[0];
        let ha = match pre { Some(p) => p.heading[0], None => f(vs.heading[a]) };
        let mut ty = String::from("barrier");
        if slots.len() > 1 {
            let b = slots[1];
            let hb = match pre { Some(p) => p.heading[1], None => f(vs.heading[b]) };
            let mut d = jm::abs(ha - hb) % (2.0 * jm::PI); if d > jm::PI { d = 2.0 * jm::PI - d; }
            let along = jm::abs(jm::cos(ha) * c.nx + jm::sin(ha) * c.nz);
            ty = (if d > 2.6 { "head-on" } else if d > 0.52 { "angle" } else if along > 0.7 { "rear-end" } else { "sideswipe" }).to_string();
            if vs.state[b] == STATE_CRASHED || vs.state[a] == STATE_CRASHED { ty.push_str("-wreck"); }
        }
        let mut st = a;
        if slots.len() > 1 && (match pre { Some(p) => p.speed[1] > p.speed[0], None => vs.speed[slots[1]] > vs.speed[a] }) { st = slots[1]; }
        let e = vs.edge[st] as usize;
        let d_node = jm::min(f(vs.s[st]), g.e_length[e] - f(vs.s[st]));
        let changing = vs.lane_hi[st] > vs.lane_lo[st] && vs.conn[st] < 0 && ((vs.target_lane[st] >= 0 && vs.target_lane[st] != vs.lane[st]) || jm::abs(f(vs.heading_rel[st])) > 0.035 || jm::abs(f(vs.v_lat[st])) > 0.3);
        let m = if slots.len() > 1 && vs.state[slots[0]] == STATE_CRASHED && vs.state[slots[1]] == STATE_CRASHED { "wreck" }
            else if changing { "lane-change" }
            else if slots.len() > 1 && vs.edge[slots[0]] != vs.edge[slots[1]] && d_node < 40.0 { if g.is_shoulder_road(e) { "merge" } else { "junction" } }
            else if vs.leader[st] >= 0 && (vs.state[vs.leader[st] as usize] != STATE_DRIVING || vs.speed[vs.leader[st] as usize] < 2.0) && vs.speed[st] > 5.0 { "queue-approach" }
            else if d_node < 30.0 && g.in_start[g.e_to[e] as usize + 1] - g.in_start[g.e_to[e] as usize] > 1 { "junction" }
            else { "cruise" };
        (ty, m)
    }

    /// _openScene(fresh, cls, e, isCrash, all): returns the record index.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn open_scene(&mut self, fresh: &[usize], cause: &'static str, severity: &'static str, block: f64, clear: f64, e: usize, is_crash: bool, all: Option<&[usize]>) -> usize {
        let all = all.unwrap_or(fresh);
        let vs = &mut self.vs;
        let mut lane_mask: i64 = 0;
        let mut s_max = 0.0;
        for &s in all { if vs.edge[s] == e as i32 { let mut l = vs.lane_lo[s] as i32; while l <= vs.lane_hi[s] as i32 { lane_mask |= 1i64 << l; l += 1; } if f(vs.s[s]) > s_max { s_max = f(vs.s[s]); } } }
        if lane_mask == 0 { lane_mask = 1i64 << vs.lane[fresh[0]]; s_max = f(vs.s[fresh[0]]); }
        let lanes = self.g.e_lanes[e];
        let mut blocked = 0; for l in 0..lanes { if (lane_mask >> l) & 1 != 0 { blocked += 1; } }
        let any_off = all.iter().any(|&q| vs.edge[q] != e as i32);
        let f0 = fresh[0];
        let id = self.crashes.len();
        let rec = Scene { id, time: self.time, cause, severity, is_crash, scene: true, vehicles: fresh.to_vec(), commuters: fresh.iter().map(|&s| vs.commuter[s]).collect(),
            edge: e as i32, s: s_max, lane_mask, lanes_blocked: blocked, x: f(vs.x[f0]), z: f(vs.z[f0]), y: f(vs.y[f0]),
            block_start: self.time, block_end: self.time + block, clear_end: self.time + clear,
            squeeze_at: self.time + jm::min(block, (if blocked >= lanes || any_off { 1.0 } else { 2.0 }) * self.opts.squeeze_delay),
            squeeze: false, response: Response::Active, permanent: block >= 1e8, gone: Vec::new(), delta_v_kmh: Vec::new(), closing: 0.0, disabled: Vec::new(), crash_type: String::new(), maneuver: "" };
        for &s in fresh { vs.state[s] = STATE_CRASHED; vs.clear_controls(s); vs.crashed_until[s] = r32(rec.clear_end); }
        if self.opts.impact_reverse { for &s in fresh { if let Some(u) = self.reverse_stash.remove(&s) { self.vs.speed[s] = r32(u); } } }
        self.crashes.push(rec);
        self.active.push(id);
        self.edge_crashes.entry(e as i32).or_default().push(id);
        self.edge_has_crash[e] = 1;
        if self.g.e_rev[e] >= 0 { self.edge_has_crash[self.g.e_rev[e] as usize] = 1; }
        id
    }

    fn move_off_lanes(&mut self, ri: usize) {
        let g = self.gw();
        let vehicles = self.crashes[ri].vehicles.clone();
        for s in vehicles {
            if self.vs.state[s] != STATE_CRASHED || self.crashes[ri].gone.contains(&s) { continue; }
            self.vs.flags[s] |= FLAG_SHOULDER;
            let ee = self.vs.edge[s] as usize;
            let sh = if g.is_shoulder_road(ee) { self.opts.freeway_shoulder } else { self.opts.shoulder };
            if sh < 2.0 * f(self.vs.hw[s]) + 0.4 || g.e_length[ee] < 30.0 {
                if self.vs.flags[s] & FLAG_STRANDED != 0 { self.counters.towed += 1; } else { self.counters.cleared += 1; }
                self.counters.hauled_away += 1;
                self.emit(crate::sim::EV_CLEARED, s, -1, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
                self.crashes[ri].gone.push(s); self.release(s); continue;
            }
            let vs = &mut self.vs;
            vs.conn[s] = -1;
            vs.s[s] = r32(jm::min(g.e_length[ee] - 15.0, jm::max(15.0, f(vs.s[s]))));
            vs.lat[s] = r32(-(f(vs.hw[s]) + 0.3)); vs.heading_rel[s] = 0.0; vs.speed[s] = 0.0; vs.v_lat[s] = 0.0; vs.yaw_rate[s] = 0.0;
            vs.update_lanes(s, &g, ee); vs.update_pose(s, &g); vs.save_pose(s); self.counters.towed_to_shoulder += 1;
        }
    }

    pub(crate) fn scenes(&mut self) {
        if self.stations.is_some() && !self.lot.is_empty() { self.pumps(); }
        self.barrier_touch = std::mem::take(&mut self.barrier_now);
        self.barrier_ep = std::mem::take(&mut self.barrier_ep_now);
        if self.pair_event.len() > 200000 { let t = self.time; self.pair_event.retain(|_, v| !(t - *v > 7200.0)); }
        if self.active.is_empty() { return; }
        let list = std::mem::take(&mut self.active);
        let mut keep = Vec::new();
        for ri in list {
            {
                let r = &self.crashes[ri];
                if r.response == Response::Active && r.is_crash && !r.squeeze {
                    let vs = &self.vs;
                    let (mut m, mut s_max) = (0i64, 0.0);
                    for &q in &r.vehicles { if vs.edge[q] == r.edge && vs.state[q] == STATE_CRASHED { let mut l = vs.lane_lo[q] as i32; while l <= vs.lane_hi[q] as i32 { m |= 1i64 << l; l += 1; } if f(vs.s[q]) > s_max { s_max = f(vs.s[q]); } } }
                    if m != 0 { let r = &mut self.crashes[ri]; r.lane_mask = m; r.s = s_max; }
                }
            }
            let t = self.time;
            let r = &self.crashes[ri];
            if r.response == Response::Active && !r.squeeze && !r.permanent && t >= r.squeeze_at && t < r.block_end {
                self.crashes[ri].squeeze = true;
                self.move_off_lanes(ri);
                self.crashes[ri].lane_mask = 0;
            }
            let r = &self.crashes[ri];
            if r.response == Response::Active && t >= r.block_end && t < r.clear_end {
                self.crashes[ri].response = Response::Shoulder;
                self.move_off_lanes(ri);
            }
            if t >= self.crashes[ri].clear_end {
                self.crashes[ri].response = Response::Cleared;
                let vehicles = self.crashes[ri].vehicles.clone();
                for s in vehicles {
                    if self.vs.state[s] == STATE_CRASHED && !self.crashes[ri].gone.contains(&s) {
                        if self.vs.flags[s] & FLAG_STRANDED != 0 { self.counters.towed += 1; } else { self.counters.cleared += 1; }
                        self.emit(crate::sim::EV_CLEARED, s, -1, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
                        self.release(s);
                    }
                }
                let edge = self.crashes[ri].edge;
                let mut emptied = false;
                if let Some(l) = self.edge_crashes.get_mut(&edge) { if let Some(k) = l.iter().position(|&x| x == ri) { l.remove(k); } emptied = l.is_empty(); }
                if emptied {
                    self.edge_crashes.remove(&edge); self.edge_has_crash[edge as usize] = 0;
                    let rv = self.g.e_rev[edge as usize];
                    if rv >= 0 && !self.edge_crashes.contains_key(&rv) { self.edge_has_crash[rv as usize] = 0; }
                }
            } else { keep.push(ri); }
        }
        self.active = keep;
    }
}

/// _body(i, b, b0, moving): body + start-of-step pose (x, z, yaw).
/// collide.js body of slot i. `real_mass` (SimOpts::wreck_real_mass): a resting wreck / parked obstacle keeps its
/// mass; off = the JS: a crashed car at rest is an immovable body (inverse mass and inertia 0).
pub(crate) fn body(vs: &VehicleStore, i: usize, moving: bool, real_mass: bool) -> (Body, (f64, f64, f64)) {
    let yaw = f(vs.heading[i]);
    let (fx, fz) = (jm::cos(yaw), jm::sin(yaw));
    let (u, v) = (f(vs.speed[i]), f(vs.v_lat[i]));
    let fixed = !real_mass && !moving && vs.state[i] == STATE_CRASHED;
    let (x, z) = (f(vs.x[i]), f(vs.z[i]));
    let b = Body { x, z, yaw, hl: f(vs.hl[i]), hw: f(vs.hw[i]), vx: u * fx + v * fz, vz: u * fz - v * fx, w: -f(vs.yaw_rate[i]),
        inv_mass: if fixed { 0.0 } else { 1.0 / f(vs.mass[i]) }, inv_inertia: if fixed { 0.0 } else { 1.0 / f(vs.inertia[i]) }, x0: x, z0: z };
    (b, (f(vs.px[i]), f(vs.pz[i]), f(vs.pyaw[i])))
}

pub(crate) fn adjacent_or_freeway_free(g: &World, a: usize, b: usize) -> bool {
    let (af, at, bf, bt) = (g.e_from[a], g.e_to[a], g.e_from[b], g.e_to[b]);
    if af == bf || af == bt || at == bf || at == bt { return true; }
    if !g.is_shoulder_road(a) && !g.is_shoulder_road(b) { return true; }
    match &g.contact_pairs { Some(set) => set.contains(&World::contact_pair_key(a as i64, b as i64)), None => false }
}
