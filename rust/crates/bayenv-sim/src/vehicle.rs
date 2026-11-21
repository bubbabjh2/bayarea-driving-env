//! src/sim/vehicle.js VehicleStore: struct of arrays indexed by slot, with the JS typed-array element types
//! (Float32Array fields are f32: every store rounds, every read widens exactly).
use crate::jm;
use crate::physics::{self, DynState, Surf, VEHICLE_PARAMS};
use crate::world::{World, Pt, HW_SHOULDER};
use std::collections::HashMap;

pub const STATE_FREE: u8 = 0;
pub const STATE_DRIVING: u8 = 1;
pub const STATE_CRASHED: u8 = 2;
pub const STATE_REFUELING: u8 = 5;
pub const FLAG_LEARNER: u8 = 1;
pub const FLAG_COLLIDED: u8 = 2;
pub const FLAG_BRAIN: u8 = 4;
pub const FLAG_SHOULDER: u8 = 8;
pub const FLAG_SPEEDING: u8 = 16;
pub const FLAG_DISABLED: u8 = 32;
pub const FLAG_STRANDED: u8 = 64;
pub const FLAG_REFUEL: u8 = 128;
pub const MAX_LANES: i32 = 8;
pub const MAX_OCC: i32 = 4;
pub const CTRL_RING: usize = 40;
pub const CTRL_STRIDE: usize = 3;
pub const TAPER: f64 = 40.0;

/// Driver traits read by alloc (genome traits; DEFAULT_TRAITS of genome.js for wrecks / obstacles).
#[derive(Clone, Copy, Debug)]
pub struct Traits { pub aggression: f64, pub patience: f64, pub reaction_time: f64, pub preferred_headway: f64, pub max_accel: f64, pub max_decel: f64 }
pub const DEFAULT_TRAITS: Traits = Traits { aggression: 0.5, patience: 0.5, reaction_time: 1.0, preferred_headway: 1.5, max_accel: 2.5, max_decel: 5.0 };

/// vs.missed entry (a lane that led onto another out-edge than planned).
#[derive(Clone, Copy, Debug)]
pub struct Missed { pub from: i32, pub planned: i32, pub took: i32, pub goal: i32 }

macro_rules! store {
    ($($name:ident : $t:ty = $init:expr, $stride:expr;)*) => {
        pub struct VehicleStore {
            pub capacity: usize, pub count: usize, pub high: usize, pub free: Vec<usize>,
            pub position_mode: bool,
            pub routes: Vec<Vec<i32>>,
            /// insertion-ordered (JS Map) missed exits: slot -> record
            pub missed: Vec<(usize, Missed)>,
            /// vs.missedT: slot -> (time, spawnTime)
            pub missed_t: HashMap<usize, (f64, f64)>,
            $(pub $name: Vec<$t>,)*
        }
        impl VehicleStore {
            pub fn new(capacity: usize) -> Self {
                let c = capacity;
                VehicleStore { capacity: c, count: 0, high: 0, free: Vec::new(), position_mode: true, routes: vec![Vec::new(); c], missed: Vec::new(), missed_t: HashMap::new(),
                    $($name: vec![$init; c * $stride],)* }
            }
            fn grow(&mut self) {
                let c = self.capacity * 2;
                $(self.$name.resize(c * $stride, 0 as $t);)*
                self.routes.resize(c, Vec::new());
                self.capacity = c;
            }
            /// (name, element type, stride, little-endian bytes of slots [0, high)) for every field (traces).
            pub fn fields(&self) -> Vec<(&'static str, &'static str, usize, Vec<u8>)> {
                let h = self.high;
                vec![$((stringify!($name), <$t as crate::pack::PackElem>::DT, $stride, self.$name[..h * $stride].iter().flat_map(|x| x.to_le_bytes()).collect()),)*]
            }
        }
    };
}

store! {
    state: u8 = 0, 1; type_idx: u8 = 0, 1; flags: u8 = 0, 1;
    edge: i32 = 0, 1; s: f32 = 0.0, 1; lat: f32 = 0.0, 1; heading_rel: f32 = 0.0, 1; steer_angle: f32 = 0.0, 1; steer_sl: f32 = 0.0, 1; steer_target: f32 = 0.0, 1;
    lane: i8 = 0, 1; lane_lo: i8 = 0, 1; lane_hi: i8 = 0, 1; seg: i32 = 0, 1;
    speed: f32 = 0.0, 1; v_lat: f32 = 0.0, 1; yaw_rate: f32 = 0.0, 1; ax_prev: f32 = 0.0, 1; accel: f32 = 0.0, 1;
    mass: f32 = 0.0, 1; inertia: f32 = 0.0, 1; hl: f32 = 0.0, 1; hw: f32 = 0.0, 1; px: f32 = 0.0, 1; pz: f32 = 0.0, 1; pyaw: f32 = 0.0, 1; power: f32 = 0.0, 1;
    surf_r: u8 = 0, 1; surf_l: u8 = 0, 1;
    x: f32 = 0.0, 1; y: f32 = 0.0, 1; z: f32 = 0.0, 1; heading: f32 = 0.0, 1;
    length: f32 = 0.0, 1; width: f32 = 0.0, 1; wheelbase: f32 = 0.0, 1; max_steer: f32 = 0.0, 1; max_steer_rate: f32 = 0.0, 1; max_speed: f32 = 0.0, 1;
    max_accel: f32 = 0.0, 1; max_decel: f32 = 0.0, 1; reaction_time: f32 = 0.0, 1; headway_pref: f32 = 0.0, 1; aggression: f32 = 0.0, 1; patience: f32 = 0.0, 1;
    steer_rate: f32 = 0.0, 1; gas: f32 = 0.0, 1; brake: f32 = 0.0, 1;
    ctrl_ring: f32 = 0.0, 120; delay_steps: u32 = 0, 1; cmd_step: i32 = 0, 1; last_decision: f32 = 0.0, 1; target_lane: i8 = 0, 1;
    lane_end_dist: f32 = 0.0, 1; lane_end_side: i8 = 0, 1; merge_cap: f32 = 0.0, 1;
    ctl_approach: i32 = 0, 1; ctl_type: i8 = 0, 1; ctl_state: i8 = 0, 1; ctl_dist: f32 = 0.0, 1; ctl_time_in: f32 = 0.0, 1; ctl_time_left: f32 = 0.0, 1;
    ctl_move: i8 = 0, 1; ctl_stopped: u8 = 0, 1; ctl_inside: i32 = 0, 1; ctl_inside_t: f32 = 0.0, 1;
    conn: i32 = -1, 1; pass_wreck: i32 = -1, 1; taper: f32 = 0.0, 1;
    route_pos: i32 = 0, 1; commuter: i32 = 0, 1; leg: u8 = 0, 1; spawn_time: f32 = 0.0, 1; odometer: f32 = 0.0, 1;
    stopped_since: f32 = 0.0, 1; crashed_until: f32 = 0.0, 1; leader: i32 = 0, 1; gap: f32 = 0.0, 1; leader2: i32 = 0, 1; gap2: f32 = 0.0, 1; pull_until: f32 = 0.0, 1;
    route_off: i32 = 0, 1; route_len: i32 = 0, 1; route_dist: f32 = 0.0, 1; route_total: f32 = 0.0, 1;
    fuel: f32 = 0.0, 1; tank: f32 = 0.0, 1; refuel_state: u8 = 0, 1; station: i32 = 0, 1; stop_pos: i32 = 0, 1; pump_until: f32 = 0.0, 1; last_fuel_check: f32 = 0.0, 1;
}

/// JS field name of a store field (LAYOUT.md / vehicle.js), for trace comparison.
pub fn js_name(rust: &str) -> String {
    let mut out = String::new();
    let mut up = false;
    for ch in rust.chars() { if ch == '_' { up = true; } else if up { out.push(ch.to_ascii_uppercase()); up = false; } else { out.push(ch); } }
    match out.as_str() { "typeIdx" | "headingRel" => out, _ => out }
}

pub struct AllocArgs<'a> { pub type_idx: u8, pub traits: Traits, pub route: &'a [i32], pub commuter: i32, pub leg: u8, pub time: f64, pub learner: bool, pub dt: f64, pub step: i64, pub lane: i32 }

#[inline(always)] fn f(x: f32) -> f64 { x as f64 }

impl VehicleStore {
    pub fn alloc(&mut self, g: &World, a: AllocArgs) -> usize {
        if self.free.is_empty() && self.high >= self.capacity { self.grow(); }
        let i = if let Some(i) = self.free.pop() { i } else { self.high += 1; self.high - 1 };
        self.count += 1;
        let t = &VEHICLE_PARAMS[if (a.type_idx as usize) < 4 { a.type_idx as usize } else { 0 }];
        let lane = a.lane;
        self.state[i] = STATE_DRIVING; self.type_idx[i] = a.type_idx;
        self.length[i] = t.length as f32; self.width[i] = t.width as f32;
        self.wheelbase[i] = t.l as f32; self.max_steer[i] = t.max_steer as f32; self.max_steer_rate[i] = t.max_steer_rate as f32;
        self.max_speed[i] = t.vmax as f32;
        self.max_accel[i] = jm::min(a.traits.max_accel * jm::min(1.0, 1500.0 / t.m * 1.6), 0.9 * t.f0 / t.m) as f32;
        self.max_decel[i] = jm::min(a.traits.max_decel * (t.emergency_decel / 8.0) * 1.6, t.emergency_decel) as f32;
        self.mass[i] = t.m as f32; self.inertia[i] = t.iz as f32; self.hl[i] = (0.5 * t.length) as f32; self.hw[i] = (0.5 * t.width) as f32;
        self.reaction_time[i] = a.traits.reaction_time as f32;
        self.headway_pref[i] = a.traits.preferred_headway as f32;
        self.aggression[i] = a.traits.aggression as f32;
        self.patience[i] = a.traits.patience as f32;
        let r = &mut self.routes[i]; r.clear(); r.extend_from_slice(a.route);
        self.route_pos[i] = 0;
        self.edge[i] = if !a.route.is_empty() { a.route[0] } else { -1 };
        self.s[i] = 0.0;
        self.lat[i] = ((lane as f64 + 0.5) * if self.edge[i] >= 0 { g.lane_w(self.edge[i] as usize, 0.0) } else { crate::world::LANE_WIDTH }) as f32;
        self.heading_rel[i] = 0.0; self.steer_angle[i] = 0.0; self.steer_sl[i] = 0.0; self.steer_target[i] = 0.0; self.seg[i] = 0;
        self.lane[i] = lane as i8; self.lane_lo[i] = lane as i8; self.lane_hi[i] = lane as i8; self.target_lane[i] = -1; self.taper[i] = 0.0; self.conn[i] = -1; self.pass_wreck[i] = -1;
        if self.edge[i] >= 0 {
            let e0 = self.edge[i] as usize;
            let sx = g.e_s_exit[e0];
            if sx > 0.0 { self.s[i] = sx as f32; self.lat[i] = ((lane as f64 + 0.5) * g.lane_w(e0, sx)) as f32; }
        }
        self.speed[i] = 0.0; self.v_lat[i] = 0.0; self.yaw_rate[i] = 0.0; self.ax_prev[i] = 0.0; self.power[i] = 0.0; self.accel[i] = 0.0;
        self.steer_rate[i] = 0.0; self.gas[i] = 0.0; self.brake[i] = 0.0;
        self.ctrl_ring[i * 120..(i + 1) * 120].fill(0.0);
        self.delay_steps[i] = jm::min((CTRL_RING - 1) as f64, jm::max(0.0, jm::round(f(self.reaction_time[i]) / a.dt))) as u32;
        self.cmd_step[i] = (a.step - 1) as i32;
        self.last_decision[i] = -1e9;
        self.commuter[i] = a.commuter; self.leg[i] = a.leg; self.spawn_time[i] = a.time as f32; self.odometer[i] = 0.0;
        self.stopped_since[i] = -1.0; self.crashed_until[i] = 0.0; self.flags[i] = if a.learner { FLAG_LEARNER } else { 0 };
        self.leader[i] = -1; self.gap[i] = 1e9; self.leader2[i] = -1; self.gap2[i] = 1e9; self.pull_until[i] = 0.0;
        self.ctl_approach[i] = -1; self.ctl_type[i] = -1; self.ctl_state[i] = 0; self.ctl_dist[i] = 1e9; self.ctl_stopped[i] = 0; self.ctl_inside[i] = -1; self.merge_cap[i] = 1e9;
        self.route_off[i] = -1; self.route_len[i] = a.route.len() as i32; self.route_dist[i] = 0.0; self.route_total[i] = 0.0;
        for &e in a.route { self.route_total[i] = (f(self.route_total[i]) + g.e_length[e as usize]) as f32; }
        self.tank[i] = t.tank_l as f32; self.fuel[i] = t.tank_l as f32; self.refuel_state[i] = 0; self.station[i] = -1; self.stop_pos[i] = -1; self.pump_until[i] = 0.0;
        self.last_fuel_check[i] = a.time as f32;
        if self.edge[i] >= 0 { self.update_lanes(i, g, self.edge[i] as usize); }
        i
    }

    pub fn release(&mut self, i: usize) {
        if self.state[i] == STATE_FREE { return; }
        self.state[i] = STATE_FREE; self.routes[i].clear(); self.count -= 1; self.conn[i] = -1;
        self.free.push(i);
    }

    #[inline]
    pub fn command(&mut self, i: usize, steer: f64, gas: f64, brake: f64, step: i64) {
        let o = (i * CTRL_RING + (step as usize % CTRL_RING)) * CTRL_STRIDE;
        let r = &mut self.ctrl_ring;
        r[o] = (if steer > 1.0 { 1.0 } else if steer < -1.0 { -1.0 } else { jm::or(steer, 0.0) }) as f32;
        r[o + 1] = (if gas > 1.0 { 1.0 } else if gas > 0.0 { gas } else { 0.0 }) as f32;
        r[o + 2] = (if brake > 1.0 { 1.0 } else if brake > 0.0 { brake } else { 0.0 }) as f32;
        self.cmd_step[i] = step as i32;
    }
    pub fn set_reaction_time(&mut self, i: usize, seconds: f64, dt: f64) {
        self.reaction_time[i] = seconds as f32;
        self.delay_steps[i] = jm::min((CTRL_RING - 1) as f64, jm::max(0.0, jm::round(seconds / dt))) as u32;
    }
    pub fn clear_controls(&mut self, i: usize) {
        self.ctrl_ring[i * 120..(i + 1) * 120].fill(0.0);
        self.steer_rate[i] = 0.0; self.gas[i] = 0.0; self.brake[i] = 0.0;
    }
    #[inline]
    pub fn half_extent(&self, i: usize) -> f64 {
        let h = f(self.heading_rel[i]);
        0.5 * (f(self.width[i]) * jm::abs(jm::cos(h)) + f(self.length[i]) * jm::abs(jm::sin(h)))
    }
    pub fn update_lanes(&mut self, i: usize, g: &World, e: usize) {
        let he = self.half_extent(i);
        let lat = f(self.lat[i]);
        let n = g.e_lanes[e];
        let w = g.lane_w(e, f(self.s[i]));
        let mut lo = jm::floor((lat - he) / w) as i64;
        let mut hi = jm::floor((lat + he - 1e-4) / w) as i64;
        let mut c = jm::floor(lat / w) as i64;
        let n = n as i64;
        if c < 0 { c = 0; }
        if c > n - 1 { c = n - 1; }
        if hi < 0 || lo > n - 1 { self.lane[i] = c as i8; self.lane_lo[i] = c as i8; self.lane_hi[i] = (c - 1) as i8; return; }
        if lo < 0 { lo = 0; }
        if hi > n - 1 { hi = n - 1; }
        if lo > hi { lo = c; hi = c; }
        if hi - lo >= MAX_OCC as i64 { hi = lo + MAX_OCC as i64 - 1; }
        self.lane[i] = c as i8; self.lane_lo[i] = lo as i8; self.lane_hi[i] = hi as i8;
    }
    #[inline]
    pub fn save_pose(&mut self, i: usize) { self.px[i] = self.x[i]; self.pz[i] = self.z[i]; self.pyaw[i] = self.heading[i]; }

    /// updatePose: world x, y, z, heading from (edge, s, lat, headingRel) or the junction connector.
    pub fn update_pose(&mut self, i: usize, g: &World) {
        let e = self.edge[i] as usize;
        let m = self.conn[i];
        let out: Pt;
        if m >= 0 && g.j.is_some() {
            let j = g.j.as_ref().unwrap();
            let m = m as usize;
            let mut p = j.connector_point(m, f(self.s[i]) - j.s_in(m));
            let off = f(self.lat[i]) - (j.info(m, 1) as f64 + 0.5) * g.lane_w(e, j.s_in(m));
            p.x += off * p.hz; p.z -= off * p.hx;
            out = p;
        } else if let Some(lg) = &g.lg {
            out = lg.lat_point(e, f(self.s[i]), f(self.lat[i]));
        } else {
            out = g.edge_point(e, f(self.s[i]), g.e_right[e] - f(self.lat[i]), self.seg[i]);
            self.seg[i] = out.seg;
        }
        self.x[i] = out.x as f32; self.z[i] = out.z as f32; self.y[i] = out.y as f32;
        self.heading[i] = (jm::atan2(out.hz, out.hx) - f(self.heading_rel[i])) as f32;
    }

    fn set_missed(&mut self, i: usize, m: Missed) {
        if let Some(p) = self.missed.iter_mut().find(|p| p.0 == i) { p.1 = m; } else { self.missed.push((i, m)); }
    }

    /// integrate(i, dt, now, graph, step, locked): 0 driving, 1 end of route, 2 refuel stop.
    pub fn integrate(&mut self, i: usize, dt: f64, now: f64, g: &World, step: i64, locked: bool) -> i32 {
        let base = i * CTRL_RING;
        let (mut sr, mut gas, mut brake) = (0.0f64, 0.0f64, 1.0f64);
        if !locked {
            let cur = (base + (step as usize % CTRL_RING)) * CTRL_STRIDE;
            if self.cmd_step[i] as i64 != step {
                let prev = (base + ((((step - 1) % CTRL_RING as i64) + CTRL_RING as i64) % CTRL_RING as i64) as usize) * CTRL_STRIDE;
                for k in 0..3 { self.ctrl_ring[cur + k] = self.ctrl_ring[prev + k]; }
            }
            let o = (base + ((((step - self.delay_steps[i] as i64) % CTRL_RING as i64) + CTRL_RING as i64) % CTRL_RING as i64) as usize) * CTRL_STRIDE;
            sr = f(self.ctrl_ring[o]); gas = f(self.ctrl_ring[o + 1]); brake = f(self.ctrl_ring[o + 2]);
            if self.fuel[i] <= 0.0 { gas = 0.0; if brake < 0.3 { brake = 0.3; } }
        }
        self.steer_rate[i] = sr as f32; self.gas[i] = gas as f32; self.brake[i] = brake as f32;
        let mut e = self.edge[i] as usize;
        let p = &VEHICLE_PARAMS[self.type_idx[i] as usize];
        let mut st = DynState { u: f(self.speed[i]), v: f(self.v_lat[i]), r: f(self.yaw_rate[i]), steer: f(self.steer_angle[i]), sl: f(self.steer_sl[i]), ax: f(self.ax_prev[i]),
            hr: f(self.heading_rel[i]), lat: f(self.lat[i]), s: f(self.s[i]), fuel_l: 0.0, power: 0.0 };
        let u0 = st.u;
        let s0 = st.s;
        let sf = if g.surfaces {
            let (pk, sk, crr, cr, cl) = crate::surface::car_surface(g, e, f(self.s[i]), f(self.lat[i]), f(self.hw[i]), self.conn[i], f(self.taper[i]));
            self.surf_r[i] = cr; self.surf_l[i] = cl;
            Some(Surf { pk, sk, crr })
        } else { None };
        let grade = if self.conn[i] >= 0 { 0.0 } else { g.edge_grade(e, f(self.s[i]), self.seg[i]) };
        physics::step_dynamics(&mut st, sr, gas, brake, p, dt, grade, locked, self.position_mode, sf);
        self.steer_target[i] = sr as f32; self.steer_sl[i] = st.sl as f32;
        self.speed[i] = st.u as f32; self.v_lat[i] = st.v as f32; self.yaw_rate[i] = st.r as f32; self.steer_angle[i] = st.steer as f32; self.ax_prev[i] = st.ax as f32;
        self.heading_rel[i] = st.hr as f32; self.lat[i] = st.lat as f32; self.power[i] = st.power as f32;
        self.accel[i] = ((st.u - u0) / dt) as f32;
        if self.fuel[i] > 0.0 { self.fuel[i] = (f(self.fuel[i]) - st.fuel_l) as f32; if self.fuel[i] < 0.0 { self.fuel[i] = 0.0; } }
        let mut s = st.s;
        let ds = s - s0;
        self.odometer[i] = (f(self.odometer[i]) + jm::abs(ds)) as f32;
        self.route_dist[i] = (f(self.route_dist[i]) + ds) as f32;
        if st.u < 0.2 && jm::abs(st.v) < 0.2 { if self.stopped_since[i] < 0.0 { self.stopped_since[i] = now as f32; } } else { self.stopped_since[i] = -1.0; }
        if s < 0.0 { s = 0.0; }
        let mut ret = 0;
        let mut s_arrive = s0;
        loop {
            if let Some(j) = &g.j {
                let m = self.conn[i];
                if m >= 0 {
                    let m = m as usize;
                    let s_in = j.s_in(m);
                    let end = s_in + j.m_len[m];
                    if s < end { break; }
                    if locked { s = end; break; }
                    let en = j.info(m, 2) as usize;
                    let off = f(self.lat[i]) - (j.info(m, 1) as f64 + 0.5) * g.lane_w(e, s_in);
                    let lo = j.info(m, 3) as f64;
                    self.route_dist[i] = (f(self.route_dist[i]) + (g.e_length[e] - end + j.s_out(m))) as f32;
                    s = j.s_out(m) + (s - end);
                    self.lat[i] = ((lo + 0.5) * g.lane_w(en, j.s_out(m)) + off) as f32;
                    if self.target_lane[i] >= 0 { self.target_lane[i] = lo as i8; }
                    self.route_pos[i] += 1; self.edge[i] = en as i32; self.seg[i] = 0; self.taper[i] = 0.0; self.conn[i] = -1;
                    e = en; s_arrive = j.s_out(m);
                    continue;
                }
                if g.e_s_stop[e] < g.e_length[e] && !locked && !(self.refuel_state[i] == 1 && self.route_pos[i] == self.stop_pos[i]) {
                    let next = self.route_pos[i] as usize + 1;
                    if next < self.routes[i].len() {
                        let la = jm::min((g.e_lanes[e] - 1) as f64, jm::max(0.0, jm::floor(f(self.lat[i]) / g.lane_w(e, s)))) as i32;
                        let rnext = self.routes[i][next];
                        let mut mv = j.movement(e, la, rnext);
                        if mv >= 0 && s < j.s_in(mv as usize) { mv = -2; }
                        let last_m = self.missed_t.get(&i).copied();
                        if mv >= 0 && j.info(mv as usize, 6) == 0 && g.e_lanes[e] > 1 && g.e_length[e] >= 80.0
                            && !(last_m.is_some() && last_m.unwrap().1 == f(self.spawn_time[i]) && now - last_m.unwrap().0 < 300.0) {
                            let mut best = -1i32;
                            let mut bt = 1e9;
                            let to = g.e_to[e] as usize;
                            for q in g.out_start[to]..g.out_start[to + 1] {
                                let o = g.out_edges[q as usize];
                                if o == g.e_rev[e] { continue; }
                                let m2 = j.movement(e, la, o);
                                if m2 >= 0 && j.info(m2 as usize, 6) != 0 && jm::abs(j.turn(m2 as usize)) < bt { bt = jm::abs(j.turn(m2 as usize)); best = m2; }
                            }
                            if best >= 0 {
                                let took = j.info(best as usize, 2) as i32;
                                let route = &self.routes[i];
                                let goal = g.e_to[route[route.len() - 1] as usize];
                                self.set_missed(i, Missed { from: e as i32, planned: rnext, took, goal });
                                self.routes[i][next] = took; self.route_off[i] = -1;
                                mv = best;
                                self.missed_t.insert(i, (now, f(self.spawn_time[i])));
                                if self.missed_t.len() > 100000 { self.missed_t.clear(); }
                            }
                        }
                        if mv >= 0 {
                            let mvu = mv as usize;
                            if s < j.s_in(mvu) { /* not there yet */ } else {
                                let s_in = j.s_in(mvu);
                                if s_arrive > s_in + 0.3 && g.lg.is_some() {
                                    let lg = g.lg.as_ref().unwrap();
                                    let p = lg.lat_point(e, jm::min(s, g.e_length[e]), f(self.lat[i]));
                                    let f_old = jm::atan2(p.hz, p.hx);
                                    let len = j.m_len[mvu];
                                    let mut bt = 0.0;
                                    let mut bd = 1e18;
                                    let h_old = f_old - f(self.heading_rel[i]);
                                    let lim = jm::min(len, s - s_in + 2.0);
                                    let mut t = 0.0;
                                    while t <= lim {
                                        let cp = j.connector_point(mvu, t);
                                        let mut dpsi = jm::atan2(cp.hz, cp.hx) - h_old;
                                        dpsi = jm::atan2(jm::sin(dpsi), jm::cos(dpsi));
                                        let d = (cp.x - p.x) * (cp.x - p.x) + (cp.z - p.z) * (cp.z - p.z) + 36.0 * dpsi * dpsi;
                                        if d < bd { bd = d; bt = t; }
                                        t += 0.5;
                                    }
                                    let cp = j.connector_point(mvu, bt);
                                    let off = (p.x - cp.x) * cp.hz - (p.z - cp.z) * cp.hx;
                                    s = s_in + bt;
                                    self.lat[i] = ((j.info(mvu, 1) as f64 + 0.5) * g.lane_w(e, s_in) + off) as f32;
                                    let mut dh = jm::atan2(cp.hz, cp.hx) - f_old;
                                    dh = jm::atan2(jm::sin(dh), jm::cos(dh));
                                    self.heading_rel[i] = (f(self.heading_rel[i]) + dh) as f32;
                                }
                                self.conn[i] = mv;
                                if self.target_lane[i] >= 0 { self.target_lane[i] = la as i8; }
                                continue;
                            }
                        }
                    }
                }
            }
            if s < g.e_length[e] { break; }
            if locked { s = g.e_length[e]; break; }
            if self.refuel_state[i] == 1 && self.route_pos[i] == self.stop_pos[i] { self.route_dist[i] = (f(self.route_dist[i]) - (s - g.e_length[e])) as f32; s = g.e_length[e]; ret = 2; break; }
            let next = self.route_pos[i] as usize + 1;
            if next >= self.routes[i].len() { self.route_dist[i] = (f(self.route_dist[i]) - (s - g.e_length[e])) as f32; s = g.e_length[e]; ret = 1; break; }
            s -= g.e_length[e];
            let mut en = self.routes[i][next] as usize;
            if g.hop(e, en).strict {
                let lane = jm::min((g.e_lanes[e] - 1) as f64, jm::max(0.0, jm::floor(f(self.lat[i]) / g.lane_w(e, g.e_length[e])))) as i32;
                let ml = g.map_lane(e, en, lane);
                if ml < 0 || ml >= g.e_lanes[en] {
                    let to = g.e_to[e] as usize;
                    for q in g.out_start[to]..g.out_start[to + 1] {
                        let o = g.out_edges[q as usize] as usize;
                        if o == en || o as i32 == g.e_rev[e] || !g.hop(e, o).strict { continue; }
                        let l2 = g.map_lane(e, o, lane);
                        if l2 >= 0 && l2 < g.e_lanes[o] {
                            let route = &self.routes[i];
                            let goal = g.e_to[route[route.len() - 1] as usize];
                            self.set_missed(i, Missed { from: e as i32, planned: en as i32, took: o as i32, goal });
                            self.routes[i][next] = o as i32; self.route_off[i] = -1;
                            en = o; break;
                        }
                    }
                }
            }
            let he = self.half_extent(i);
            let hop = g.hop(e, en);
            let mut l2 = g.map_lat(e, en, f(self.lat[i]), he);
            let w = g.lane_w(en, s);
            let n = g.e_lanes[en] as f64;
            if l2 >= 0.0 && l2 <= n * w && (l2 - he < 0.3 || l2 + he > n * w - 0.3) && (!hop.strict || g.e_lanes[e] != g.e_lanes[en] || g.e_hw[en] & HW_SHOULDER != 0) {
                let ln = jm::min(n - 1.0, jm::max(0.0, jm::floor(l2 / w)));
                l2 = (ln + 0.5) * w;
            }
            self.lat[i] = l2 as f32;
            if self.target_lane[i] >= 0 { let t = g.map_lane(e, en, self.target_lane[i] as i32); self.target_lane[i] = if t < 0 { 0 } else if t >= g.e_lanes[en] { (g.e_lanes[en] - 1) as i8 } else { t as i8 }; }
            self.taper[i] = if g.lg.is_none() && (self.lat[i] < 0.0 || f(self.lat[i]) > g.e_lanes[en] as f64 * w) { TAPER as f32 } else { 0.0 };
            self.route_pos[i] = next as i32;
            self.edge[i] = en as i32; self.seg[i] = 0;
            e = en; s_arrive = 0.0;
        }
        self.s[i] = s as f32;
        self.update_lanes(i, g, e);
        ret
    }
}
