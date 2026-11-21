//! Per-agent observations: port of src/sim/obs.js (fillObs, fillObs3, fillNeighbors), src/sim/perceive.js
//! (senseBodies, senseControls3, senseCommands, movementAhead, indicatorOf), src/sim/obsLayout.js (OBS3 layout,
//! sector selection, bodyFeatures, LeaderHistory), src/sim/gradeObs.js and the sim.js observation helpers
//! (oncomingObs, junctionObs). Bit-exact against env.observe({v3: true}) (harness::check on `--obs 1` traces).
//!
//! State effects: like the JS, `fill_obs` calls ctx.yieldRequired (-> `_yieldJ`), which writes `yield_by[i]`
//! (read by the next step's deadlock breaker before drive resets it) and fills the `_conflicts` cache. So the
//! observe functions take `&mut Sim` and observing can (as in JS) influence the next step; call them at the same
//! point of the loop as the JS env (after env.step, before the actions).
//!
//! # Flat layout (RL)
//! `observe_flat(sim, slots, out, v3)` writes one row of `flat_stride(v3)` f32 per slot:
//! `[0, OBS_N)` = the scalars below (`OBS_FIELDS`, same order), then (v3) `[OBS_N, OBS_N + OBS3)` = the input-
//! version-3 vector. Booleans are 0/1; "none" distances are +Infinity (as in JS); ids are exact up to 2^24.
//! ```text
//!  0 slot              1 agent (commuter)  2 x             3 z              4 heading        5 speed
//!  6 laneIndex         7 laneCount         8 laneWidth     9 laneOffset    10 onConnector   11 laneHeading
//! 12 speedLimit       13 roadId           14 routeDistance 15 steerAngle  16 steerTarget   17 gas
//! 18 brake            19 actionMode (1 position, 0 rate)  20 maxSteerAngle 21 headingRel    22 lat
//! 23 laneEndDistance  24 laneEndSide      25 maxSteer     26 maxSteerRate  27 wheelbase     28 width
//! 29 length           30 mass             31 vehicleType (0 sedan 1 suv 2 truck 3 bus)        32 vLat
//! 33 yawRate          34 control.type     35 control.state 36 control.distance 37 control.timeInState
//! 38 control.timeToChange 39 control.movement 40 control.stopped 41 control.yieldRequired
//! 42..53 neighbors[0..12] ([ownAhead, ownBehind, leftAhead, leftBehind, rightAhead, rightBehind] x {gap, rel})
//! 54 lead2.gap        55 lead2.rel        56 lead2.stopped 57 pullOver
//! 58 oncoming.gap     59 oncoming.closing 60 junction.mouthDistance 61 junction.inBox 62 junction.distance
//! 63 junction.timeToConflict 64 junction.stopped 65 passingWreck 66..68 grade[0..3]  69 leaderId
//! 70 signals.collision 71 signals.stranded 72 signals.harshBraking 73 signals.timeToWorkStart
//! 74 signals.remainingRouteDistance 75 signals.fuelFraction 76 signals.refuelPlanned
//! ```
//! v3 (obsLayout.js): 0..69 left 0 (policy-side encoding), 70..78 mirror, 79..86 controls, 87..93 own commands,
//! 94..101 leader history (policy-side: `LeaderHistory::step`, left 0 by fill_obs3), 102..357 16 bodies x 16.
use crate::jm;
use crate::physics::VEHICLE_PARAMS;
use crate::extra::{movement_of, state_of, CTRL_GIVE_WAY, CTRL_SIGNAL, CTRL_STOP, LIGHT_GREEN, LIGHT_RED, LIGHT_YELLOW, MOVE_LEFT, MOVE_THROUGH};
use crate::sim::{f, Signals, Sim};
use crate::vehicle::*;
use crate::world::{Controls, World};

const INF: f64 = f64::INFINITY;

// ---------------------------------------------------------------- obsLayout.js
pub const RAW3: usize = 102;
pub const NB_K: usize = 16;
pub const NB_D: usize = 16;
pub const NB_OFF: usize = RAW3;
pub const OBS3: usize = RAW3 + NB_K * NB_D; // 358
pub const MIRROR_OFF: usize = 70;
pub const CTL3_OFF: usize = 79;
pub const HIST_OFF: usize = 87;
pub const HIST_VISIBLE: usize = 93;
pub const LHIST_OFF: usize = 94;
pub const SECTORS: usize = 8;
pub const PER_SECTOR: usize = NB_K / SECTORS;
pub const NB_RANGE: f64 = 100.0;
pub const LAYER_GAP: f64 = 2.5;
pub const LHIST_N: usize = 20;
pub const LHIST_AGO: [usize; 3] = [5, 10, 20];
pub const FEATURES: [&str; 16] = ["fwd", "left", "gap", "vFwd", "vLeft", "cosDh", "sinDh", "length", "width", "brake", "indLeft", "indRight", "crashed", "crossesMyPath", "closestApproach", "valid"];
pub const INDICATOR_DIST: f64 = 60.0;
pub const CONFLICT_DIST: f64 = 150.0;
// gradeObs.js
pub const GRADE_AHEAD1: f64 = 40.0;
pub const GRADE_AHEAD2: f64 = 100.0;
pub const GRADE_MAX_HOPS: usize = 12;
pub const VEHICLE_TYPE_NAMES: [&str; 4] = ["sedan", "suv", "truck", "bus"];

/// ego-frame sector 0..7 of a point (fwd, left)
#[inline]
pub fn sector_of(fwd: f64, left: f64) -> usize {
    let a = jm::atan2(left, fwd);
    let s = jm::floor((a + jm::PI) / (jm::PI / 4.0));
    if s < 0.0 { 0 } else if s > 7.0 { 7 } else { s as usize }
}

/// selection of the PER_SECTOR nearest bodies per sector (obsLayout.js resetSelection / offerBody / forceBody)
#[derive(Clone, Copy, Debug)]
pub struct Selection { pub d2: [f64; NB_K], pub j: [i32; NB_K] }
impl Default for Selection { fn default() -> Self { Selection { d2: [INF; NB_K], j: [-1; NB_K] } } }
impl Selection {
    pub fn reset(&mut self) { self.d2 = [INF; NB_K]; self.j = [-1; NB_K]; }
    #[inline]
    pub fn offer(&mut self, sector: usize, d2: f64, j: i32) {
        let b = sector * PER_SECTOR;
        let better = |s: &Selection, k: usize| d2 < s.d2[k] || (d2 == s.d2[k] && j < s.j[k]);
        if !better(self, b + 1) { return; }
        if better(self, b) { self.d2[b + 1] = self.d2[b]; self.j[b + 1] = self.j[b]; self.d2[b] = d2; self.j[b] = j; } else { self.d2[b + 1] = d2; self.j[b + 1] = j; }
    }
    pub fn force(&mut self, sector: usize, d2: f64, j: i32) {
        if self.j.contains(&j) { return; }
        let b = sector * PER_SECTOR;
        self.d2[b + 1] = d2; self.j[b + 1] = j;
        if self.j[b] < 0 || d2 < self.d2[b] || (d2 == self.d2[b] && j < self.j[b]) {
            let (t, tj) = (self.d2[b], self.j[b]);
            self.d2[b] = self.d2[b + 1]; self.j[b] = self.j[b + 1]; self.d2[b + 1] = t; self.j[b + 1] = tj;
        }
    }
}

/// bodyFeatures `me` / `o` records
#[derive(Clone, Copy, Debug, Default)]
pub struct Body { pub x: f64, pub z: f64, pub h: f64, pub v: f64, pub hl: f64, pub hw: f64, pub length: f64, pub width: f64,
    pub brake: f64, pub ind_l: bool, pub ind_r: bool, pub crashed: bool, pub conflict: bool }

/// The 16 features of body `o` in the frame of `me` into out[at .. at + 16).
pub fn body_features(me: &Body, o: &Body, out: &mut [f32], at: usize) {
    let c = jm::cos(me.h); let s = jm::sin(me.h);
    let dx = o.x - me.x; let dz = o.z - me.z;
    let fwd = dx * c + dz * s; let left = dx * s - dz * c;
    let co = jm::cos(o.h); let so = jm::sin(o.h);
    let rvx = o.v * co - me.v * c; let rvz = o.v * so - me.v * s;
    let vf = rvx * c + rvz * s; let vl = rvx * s - rvz * c;
    let d = jm::sqrt(dx * dx + dz * dz);
    let (mut ux, mut uz) = (1.0, 0.0);
    if d > 1e-6 { ux = dx / d; uz = dz / d; }
    let e_me = me.hl * jm::abs(ux * c + uz * s) + me.hw * jm::abs(ux * s - uz * c);
    let e_o = o.hl * jm::abs(ux * co + uz * so) + o.hw * jm::abs(ux * so - uz * co);
    let gap = jm::max(0.0, d - e_me - e_o);
    let mut dh = o.h - me.h; dh = jm::atan2(jm::sin(dh), jm::cos(dh));
    let vv = vf * vf + vl * vl;
    let mut ca = 1.0;
    if vv > 1e-4 {
        let t = -(fwd * vf + left * vl) / vv;
        if t > 0.0 { let mx = fwd + vf * t; let ml = left + vl * t; if mx * mx + ml * ml <= 9.0 { ca = jm::min(1.0, t / 10.0); } }
    }
    out[at] = (fwd / NB_RANGE) as f32; out[at + 1] = (left / NB_RANGE) as f32; out[at + 2] = jm::min(gap / NB_RANGE, 1.0) as f32;
    out[at + 3] = jm::max(-1.0, jm::min(1.0, vf / 20.0)) as f32; out[at + 4] = jm::max(-1.0, jm::min(1.0, vl / 20.0)) as f32;
    out[at + 5] = jm::cos(dh) as f32; out[at + 6] = jm::sin(dh) as f32;
    out[at + 7] = (o.length / 20.0) as f32; out[at + 8] = (o.width / 3.0) as f32;
    out[at + 9] = if o.crashed { 0.0 } else if o.brake > 0.05 { 1.0 } else { 0.0 };
    out[at + 10] = if o.ind_l { 1.0 } else { 0.0 }; out[at + 11] = if o.ind_r { 1.0 } else { 0.0 };
    out[at + 12] = if o.crashed { 1.0 } else { 0.0 }; out[at + 13] = if o.conflict { 1.0 } else { 0.0 };
    out[at + 14] = ca as f32; out[at + 15] = 1.0;
}

/// Leader history (v3 inputs 94..100), kept per car by its driver: call once per decision with the final
/// inputs 8 / 9 and the leader id (obsLayout.js LeaderHistory).
#[derive(Clone, Debug)]
pub struct LeaderHistory { pub g: [f32; LHIST_N], pub r: [f32; LHIST_N], pub n: usize, pub head: usize, pub last_leader: f64, pub since_change: f64 }
impl Default for LeaderHistory { fn default() -> Self { LeaderHistory { g: [0.0; LHIST_N], r: [0.0; LHIST_N], n: 0, head: 0, last_leader: -1e9, since_change: 1e6 } } }
impl LeaderHistory {
    pub fn step(&mut self, input: &mut [f32], leader_id: i32) {
        for q in 0..3 {
            let ago = LHIST_AGO[q]; let o = LHIST_OFF + 2 * q;
            if self.n == 0 { input[o] = input[8]; input[o + 1] = input[9]; continue; }
            let k = ago.min(self.n); let idx = (self.head + LHIST_N - k) % LHIST_N;
            input[o] = self.g[idx]; input[o + 1] = self.r[idx];
        }
        if self.n > 0 && leader_id as f64 != self.last_leader { self.since_change = 0.0; } else { self.since_change = jm::min(self.since_change + 1.0, 1e6); }
        input[LHIST_OFF + 6] = if self.since_change < LHIST_N as f64 { 1.0 } else { 0.0 };
        self.last_leader = leader_id as f64;
        self.g[self.head] = input[8]; self.r[self.head] = input[9];
        self.head = (self.head + 1) % LHIST_N; if self.n < LHIST_N { self.n += 1; }
        input[101] = 0.0;
    }
}

// ---------------------------------------------------------------- the obs object
#[derive(Clone, Copy, Debug, Default)]
pub struct ControlObs { pub ty: i32, pub state: i32, pub distance: f64, pub time_in_state: f64, pub time_to_change: f64, pub movement: i32, pub stopped: bool, pub yield_required: bool }
#[derive(Clone, Copy, Debug, Default)]
pub struct Lead2Obs { pub gap: f64, pub rel: f64, pub stopped: bool }
#[derive(Clone, Copy, Debug, Default)]
pub struct OncomingObs { pub gap: f64, pub closing: f64 }
#[derive(Clone, Copy, Debug, Default)]
pub struct JunctionObs { pub mouth_distance: f64, pub in_box: bool, pub distance: f64, pub time_to_conflict: f64, pub stopped: bool }

/// env.observe() obs for one slot (JS numbers are f64, the Float32Array `neighbors` is f32). `vehicle_type`
/// is the VEHICLE_PARAMS index (JS: its name, VEHICLE_TYPE_NAMES); `action_mode_position` is JS actionMode
/// === 'position'. `leader_id` (JS: set by fillObs3) and `signals` (env.observe) are filled by `observe_one`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Obs {
    pub x: f64, pub z: f64, pub heading: f64, pub speed: f64,
    pub lane_index: i32, pub lane_count: i32, pub lane_width: f64, pub lane_offset: f64, pub on_connector: bool,
    pub lane_heading: f64, pub speed_limit: f64, pub road_id: i32, pub route_distance: f64,
    pub steer_angle: f64, pub steer_target: f64, pub gas: f64, pub brake: f64, pub action_mode_position: bool, pub max_steer_angle: f64,
    pub heading_rel: f64, pub lat: f64, pub lane_end_distance: f64, pub lane_end_side: i32,
    pub max_steer: f64, pub max_steer_rate: f64, pub wheelbase: f64, pub width: f64, pub length: f64, pub mass: f64, pub vehicle_type: usize,
    pub v_lat: f64, pub yaw_rate: f64,
    pub control: ControlObs,
    pub neighbors: [f32; 12],
    pub lead2: Lead2Obs,
    pub pull_over: bool,
    pub oncoming: OncomingObs,
    pub junction: JunctionObs,
    pub passing_wreck: bool,
    pub grade: [f64; 3],
    pub leader_id: i32,
    pub signals: Signals,
    /// junction-yield blocker write of this observation (ctx.yieldRequired -> _yieldBy[slot]), applied by apply_writes
    pub yield_write: Option<i32>,
}

pub const OBS_N: usize = 77;
pub const OBS_FIELDS: [&str; OBS_N] = [
    "slot", "agent", "x", "z", "heading", "speed", "laneIndex", "laneCount", "laneWidth", "laneOffset", "onConnector", "laneHeading",
    "speedLimit", "roadId", "routeDistance", "steerAngle", "steerTarget", "gas", "brake", "actionMode", "maxSteerAngle", "headingRel", "lat",
    "laneEndDistance", "laneEndSide", "maxSteer", "maxSteerRate", "wheelbase", "width", "length", "mass", "vehicleType", "vLat", "yawRate",
    "control.type", "control.state", "control.distance", "control.timeInState", "control.timeToChange", "control.movement", "control.stopped", "control.yieldRequired",
    "neighbors.0", "neighbors.1", "neighbors.2", "neighbors.3", "neighbors.4", "neighbors.5", "neighbors.6", "neighbors.7", "neighbors.8", "neighbors.9", "neighbors.10", "neighbors.11",
    "lead2.gap", "lead2.rel", "lead2.stopped", "pullOver", "oncoming.gap", "oncoming.closing",
    "junction.mouthDistance", "junction.inBox", "junction.distance", "junction.timeToConflict", "junction.stopped", "passingWreck",
    "grade.0", "grade.1", "grade.2", "leaderId",
    "signals.collision", "signals.stranded", "signals.harshBraking", "signals.timeToWorkStart", "signals.remainingRouteDistance", "signals.fuelFraction", "signals.refuelPlanned",
];
#[inline] fn b(x: bool) -> f64 { if x { 1.0 } else { 0.0 } }

impl Obs {
    /// The scalar row in OBS_FIELDS order (f64, exact).
    pub fn scalars(&self, slot: usize, agent: i32) -> [f64; OBS_N] {
        let (c, l, j, s) = (&self.control, &self.lead2, &self.junction, &self.signals);
        let n = |k: usize| self.neighbors[k] as f64;
        [slot as f64, agent as f64, self.x, self.z, self.heading, self.speed, self.lane_index as f64, self.lane_count as f64, self.lane_width, self.lane_offset,
         b(self.on_connector), self.lane_heading, self.speed_limit, self.road_id as f64, self.route_distance, self.steer_angle, self.steer_target, self.gas, self.brake,
         b(self.action_mode_position), self.max_steer_angle, self.heading_rel, self.lat, self.lane_end_distance, self.lane_end_side as f64, self.max_steer,
         self.max_steer_rate, self.wheelbase, self.width, self.length, self.mass, self.vehicle_type as f64, self.v_lat, self.yaw_rate,
         c.ty as f64, c.state as f64, c.distance, c.time_in_state, c.time_to_change, c.movement as f64, b(c.stopped), b(c.yield_required),
         n(0), n(1), n(2), n(3), n(4), n(5), n(6), n(7), n(8), n(9), n(10), n(11),
         l.gap, l.rel, b(l.stopped), b(self.pull_over), self.oncoming.gap, self.oncoming.closing,
         j.mouth_distance, b(j.in_box), j.distance, j.time_to_conflict, b(j.stopped), b(self.passing_wreck),
         self.grade[0], self.grade[1], self.grade[2], self.leader_id as f64,
         b(s.collision), b(s.stranded), b(s.harsh_braking), s.time_to_work_start, s.remaining_route_distance, s.fuel_fraction, b(s.refuel_planned)]
    }
}

// ---------------------------------------------------------------- obs.js fillObs
/// fillObs(sim, i, obs). Mutates sim like the JS (ctx.yieldRequired -> yield_by[i], conflict cache).
pub fn fill_obs(sim: &mut Sim, i: usize, o: &mut Obs) { fill_obs_ro(sim, i, o); apply_writes(sim, i, o); }

/// fill_obs without the state effect (see observe_one_ro).
pub fn fill_obs_ro(sim: &Sim, i: usize, o: &mut Obs) {
    let g = sim.gw();
    let e = sim.vs.edge[i] as usize;
    {
        let vs = &sim.vs;
        o.x = f(vs.x[i]); o.z = f(vs.z[i]); o.heading = f(vs.heading[i]); o.speed = f(vs.speed[i]);
        let m = vs.conn[i];
        if let (true, Some(jj)) = (m >= 0, g.j.as_ref()) {
            let m = m as usize;
            let lw = g.lane_w(e, jj.m_f[4 * m]);
            o.lane_index = 0; o.lane_count = 1; o.lane_width = lw;
            o.lane_offset = f(vs.lat[i]) - (jj.m_info[7 * m + 1] as f64 + 0.5) * lw;
        } else {
            let lw = g.lane_w(e, f(vs.s[i]));
            o.lane_index = vs.lane[i] as i32; o.lane_count = g.e_lanes[e]; o.lane_width = lw;
            o.lane_offset = f(vs.lat[i]) - (vs.lane[i] as f64 + 0.5) * lw;
        }
        o.on_connector = m >= 0;
        o.lane_heading = f(vs.heading[i]) + f(vs.heading_rel[i]);
        o.speed_limit = g.e_speed[e]; o.road_id = e as i32;
        o.route_distance = f(vs.route_dist[i]);
        o.steer_angle = f(vs.steer_angle[i]); o.gas = f(vs.gas[i]); o.brake = f(vs.brake[i]);
        o.steer_target = f(vs.steer_target[i]); o.action_mode_position = vs.position_mode; o.max_steer_angle = f(vs.max_steer[i]);
        o.heading_rel = f(vs.heading_rel[i]); o.lat = f(vs.lat[i]);
        o.lane_end_distance = if vs.lane_end_dist[i] < 1e8 { f(vs.lane_end_dist[i]) } else { INF }; o.lane_end_side = vs.lane_end_side[i] as i32;
        if o.on_connector { o.lane_end_distance = INF; o.lane_end_side = 0; }
        let ti = vs.type_idx[i] as usize;
        let p = &VEHICLE_PARAMS[ti];
        o.max_steer = p.max_steer; o.max_steer_rate = p.max_steer_rate; o.wheelbase = p.l; o.width = p.width; o.length = p.length;
        o.mass = p.m; o.vehicle_type = ti;
        o.v_lat = f(vs.v_lat[i]); o.yaw_rate = f(vs.yaw_rate[i]);
        let c = &mut o.control;
        c.ty = vs.ctl_type[i] as i32; c.state = vs.ctl_state[i] as i32; c.distance = if vs.ctl_type[i] >= 0 { f(vs.ctl_dist[i]) } else { INF };
        c.time_in_state = f(vs.ctl_time_in[i]); c.time_to_change = f(vs.ctl_time_left[i]); c.movement = vs.ctl_move[i] as i32; c.stopped = vs.ctl_stopped[i] == 1;
    }
    let dist = g.e_length[e] - f(sim.vs.s[i]);
    let (yr, yw) = sim.yield_required_core(i, e, dist);
    o.control.yield_required = yr; o.yield_write = yw;
    fill_neighbors(sim, i, e, &mut o.neighbors);
    {
        let vs = &sim.vs;
        let l2 = vs.leader2[i];
        o.lead2.gap = if l2 >= 0 { jm::max(0.0, f(vs.gap2[i])) } else { INF };
        o.lead2.rel = if l2 >= 0 { (if vs.state[l2 as usize] == 1 { f(vs.speed[l2 as usize]) } else { 0.0 }) - f(vs.speed[i]) } else { 0.0 };
        o.lead2.stopped = l2 >= 0 && vs.state[l2 as usize] != 1;
        o.pull_over = f(vs.pull_until[i]) > sim.time;
    }
    oncoming_obs(sim, i, &mut o.oncoming);
    junction_obs(sim, i, &mut o.junction);
    o.passing_wreck = sim.vs.pass_wreck[i] >= 0;
    grade_obs(sim, i, &mut o.grade);
    o.leader_id = sim.vs.leader[i];
}

/// fillNeighbors: neighbour summary from the footprint lane lists.
pub fn fill_neighbors(sim: &Sim, i: usize, e: usize, out: &mut [f32; 12]) {
    let (vs, lanes, g) = (&sim.vs, &sim.lanes, &*sim.g);
    let lane = vs.lane[i] as i32; let s = f(vs.s[i]); let v = f(vs.speed[i]); let half = 0.5 * f(vs.length[i]);
    out.fill(f32::INFINITY);
    for k in (1..12).step_by(2) { out[k] = 0.0; }
    let l = vs.leader[i];
    if l >= 0 || l <= -2 {
        out[0] = jm::max(0.0, f(vs.gap[i])) as f32;
        out[1] = ((if l >= 0 && vs.state[l as usize] == STATE_DRIVING { f(vs.speed[l as usize]) } else { 0.0 }) - v) as f32;
    }
    let bh = lanes.behind(vs, i, lane);
    if bh >= 0 { let bh = bh as usize; out[2] = jm::max(0.0, s - f(vs.s[bh]) - half - 0.5 * f(vs.length[bh])) as f32; out[3] = (f(vs.speed[bh]) - v) as f32; }
    let mut side = |l: i32, o: usize| {
        if l < 0 || l >= g.e_lanes[e] { return; }
        let (p0, p1) = lanes.neighbors_at(vs, e as i32, l, s, i as i32);
        if p0 >= 0 { let p = p0 as usize; out[o] = jm::max(0.0, f(vs.s[p]) - s - half - 0.5 * f(vs.length[p])) as f32; out[o + 1] = ((if vs.state[p] == STATE_DRIVING { f(vs.speed[p]) } else { 0.0 }) - v) as f32; }
        if p1 >= 0 { let p = p1 as usize; out[o + 2] = jm::max(0.0, s - f(vs.s[p]) - half - 0.5 * f(vs.length[p])) as f32; out[o + 3] = ((if vs.state[p] == STATE_DRIVING { f(vs.speed[p]) } else { 0.0 }) - v) as f32; }
    };
    side(lane + 1, 4);
    side(lane - 1, 8);
}

/// sim.oncomingObs: nearest car ahead in the opposing (reverse edge's inner) lane.
pub fn oncoming_obs(sim: &Sim, i: usize, out: &mut OncomingObs) {
    let (vs, g) = (&sim.vs, &*sim.g);
    let e = vs.edge[i] as usize;
    out.gap = INF; out.closing = 0.0;
    if g.e_rev[e] < 0 { return; }
    let er = g.e_rev[e];
    let (a, bnd) = sim.lanes.range(er, g.e_lanes[er as usize] - 1);
    let si = f(vs.s[i]); let hl = f(vs.hl[i]);
    for k in a..bnd {
        let j = sim.lanes.order[k] as usize; let sj = g.e_length[e] - f(vs.s[j]);
        if sj <= si { continue; }
        let gg = jm::max(0.0, sj - si - hl - f(vs.hl[j]));
        if gg < out.gap { out.gap = gg; out.closing = f(vs.speed[i]) + (if vs.state[j] == STATE_DRIVING { f(vs.speed[j]) } else { 0.0 }); }
    }
}

/// sim.junctionObs: junction conflicts ahead for slot i.
pub fn junction_obs(sim: &Sim, i: usize, out: &mut JunctionObs) {
    let g = sim.gw();
    let e = sim.vs.edge[i] as usize;
    out.mouth_distance = INF; out.in_box = sim.vs.conn[i] >= 0; out.distance = INF; out.time_to_conflict = INF; out.stopped = false;
    let Some(jj) = g.j.as_ref() else { return };
    let mut m = sim.vs.conn[i];
    if m < 0 {
        let route = &sim.vs.routes[i]; let rp = sim.vs.route_pos[i] as i64;
        if !(g.e_s_stop[e] < g.e_length[e]) || rp + 1 >= route.len() as i64 { return; }
        m = jj.movement(e, sim.vs.lane[i] as i32, route[(rp + 1) as usize]); if m < 0 { return; }
    }
    let m = m as usize;
    let ti = f(sim.vs.s[i]) - jj.m_f[4 * m]; let hl = f(sim.vs.hl[i]);
    out.mouth_distance = jm::max(0.0, -ti - hl);
    let confs = sim.conflicts_ro(m);
    let vs = &sim.vs;
    for c in confs.iter() {
        let Some(list) = sim.conn_cars.get(&(c.other as usize)) else { continue };
        let di = c.s - ti - hl; if di < -2.0 * hl { continue; }
        for &j in list {
            let dj = c.s_other - (f(vs.s[j]) - jj.m_f[4 * c.other as usize]);
            if dj < -f(vs.hl[j]) - f(vs.hw[i]) - 0.5 { continue; }
            let on_it = jm::abs(dj) < f(vs.hl[j]) + f(vs.hw[i]) + 0.5;
            if di < out.distance {
                out.distance = jm::max(0.0, di);
                out.time_to_conflict = if on_it { 0.0 } else { dj / jm::max(0.1, f(vs.speed[j])) };
                out.stopped = vs.state[j] != STATE_DRIVING || vs.speed[j] < 0.5;
            }
        }
    }
}

// ---------------------------------------------------------------- gradeObs.js
/// height of edge e at s (polyline ys, linear; e.y without ys)
pub fn edge_y(g: &World, e: usize, s: f64) -> f64 {
    if !g.e_has_ys[e] { return jm::or(g.e_y[e], 0.0); }
    let v0 = g.v_start[e] as usize; let n = g.v_start[e + 1] as usize - v0;
    let cum = &g.cum[v0..v0 + n]; let ys = &g.ys[v0..v0 + n];
    let mut k = 0usize; while k + 2 < n && cum[k + 1] < s { k += 1; }
    let d = cum[k + 1] - cum[k];
    let mut t = if d > 0.0 { (s - cum[k]) / d } else { 0.0 };
    t = if t < 0.0 { 0.0 } else if t > 1.0 { 1.0 } else { t };
    ys[k] + (ys[k + 1] - ys[k]) * t
}
/// height d m ahead of (route[pos], s) along the route
pub fn route_y(g: &World, route: &[i32], pos: usize, route_len: usize, s: f64, d: f64) -> f64 {
    let mut q = pos; let mut rem = (if s > 0.0 { s } else { 0.0 }) + d; let mut e = route[q] as usize; let mut hops = 0;
    while rem > g.e_length[e] && q + 1 < route_len && hops < GRADE_MAX_HOPS { rem -= g.e_length[e]; q += 1; hops += 1; e = route[q] as usize; }
    edge_y(g, e, if rem < g.e_length[e] { rem } else { g.e_length[e] })
}
/// [g0, g1, g2] raw grades for slot i.
pub fn grade_obs(sim: &Sim, i: usize, out: &mut [f64; 3]) {
    let (vs, g) = (&sim.vs, &*sim.g);
    let ei = vs.edge[i]; let e = ei as usize; let s = f(vs.s[i]);
    out[0] = if vs.conn[i] >= 0 { 0.0 } else { g.edge_grade(e, s, vs.seg[i]) };
    let route = &vs.routes[i]; let len = vs.route_len[i]; let pos = vs.route_pos[i];
    if !(len > 0) || pos < 0 || pos >= len || route.get(pos as usize).copied() != Some(ei) {
        let el = g.e_length[e];
        let y0 = edge_y(g, e, if s > 0.0 { s } else { 0.0 }); let y1 = edge_y(g, e, jm::min(el, s + GRADE_AHEAD1)); let y2 = edge_y(g, e, jm::min(el, s + GRADE_AHEAD2));
        out[1] = (y1 - y0) / GRADE_AHEAD1; out[2] = (y2 - y1) / (GRADE_AHEAD2 - GRADE_AHEAD1); return;
    }
    let (pos, len) = (pos as usize, len as usize);
    let y0 = route_y(g, route, pos, len, s, 0.0); let y1 = route_y(g, route, pos, len, s, GRADE_AHEAD1); let y2 = route_y(g, route, pos, len, s, GRADE_AHEAD2);
    out[1] = (y1 - y0) / GRADE_AHEAD1; out[2] = (y2 - y1) / (GRADE_AHEAD2 - GRADE_AHEAD1);
}

// ---------------------------------------------------------------- perceive.js
/// movement car j is on, or will take at the end of its edge if within `within` m; -1 none
pub fn movement_ahead(sim: &Sim, j: usize, within: f64) -> i32 {
    let (vs, g) = (&sim.vs, &*sim.g);
    let Some(jj) = g.j.as_ref() else { return -1 };
    let m = vs.conn[j]; if m >= 0 { return m; }
    let e = vs.edge[j] as usize; let route = &vs.routes[j]; let rp = vs.route_pos[j] as i64;
    if rp + 1 >= route.len() as i64 { return -1; }
    if g.e_length[e] - f(vs.s[j]) - f(vs.hl[j]) > within { return -1; }
    jj.movement(e, vs.lane[j] as i32, route[(rp + 1) as usize])
}
/// turn indicator of car j: (left, right)
pub fn indicator_of(sim: &Sim, j: usize) -> (bool, bool) {
    let vs = &sim.vs;
    if vs.state[j] != STATE_DRIVING { return (false, false); }
    let m = movement_ahead(sim, j, jm::max(INDICATOR_DIST, 4.0 * f(vs.speed[j])));
    if m >= 0 { let th = sim.g.j.as_ref().unwrap().m_f[4 * m as usize + 3]; return (th < -0.5, th > 0.5); }
    if vs.lane_end_dist[j] < 100.0 && vs.lane_end_side[j] != 0 { return (vs.lane_end_side[j] > 0, vs.lane_end_side[j] < 0); }
    (false, false)
}

/// senseBodies: NB block + mirror inputs 70..78 for slot i into out (an OBS3 vector).
pub fn sense_bodies(sim: &Sim, i: usize, out: &mut [f32], lane_w: f64, lane_off: f64) {
    let r = NB_RANGE;
    let mut sel = Selection::default();
    let me;
    {
        let vs = &sim.vs;
        let x = f(vs.x[i]); let z = f(vs.z[i]); let y = f(vs.y[i]); let h = f(vs.heading[i]);
        let c = jm::cos(h); let s = jm::sin(h);
        let hl_i = f(vs.hl[i]); let hw_i = f(vs.hw[i]); let v = f(vs.speed[i]);
        let (mut alt_l, mut alt_r) = (0.0f64, 0.0f64);
        let (mut ttc_rl, mut ttc_rr, mut gap_rl, mut gap_rr, mut ttc_own, mut clr_l, mut clr_r) = (INF, INF, INF, INF, INF, INF, INF);
        let hw = 0.5 * lane_w;
        sim.hash.for_each_near(&vs.x, &vs.z, x, z, r, |j| {
            if j == i { return true; }
            let st = vs.state[j]; if st != STATE_DRIVING && st != STATE_CRASHED { return true; }
            if jm::abs(f(vs.y[j]) - y) >= LAYER_GAP { return true; }
            let dx = f(vs.x[j]) - x; let dz = f(vs.z[j]) - z;
            let d2 = dx * dx + dz * dz;
            let fwd = dx * c + dz * s; let left = dx * s - dz * c;
            sel.offer(sector_of(fwd, left), d2, j as i32);
            let hl_j = f(vs.hl[j]); let hw_j = f(vs.hw[j]); let lat = left + lane_off; let lo = lat - hw_j; let hi = lat + hw_j;
            let vj = if st == STATE_DRIVING { f(vs.speed[j]) } else { 0.0 };
            let vf = vj * jm::cos(f(vs.heading[j]) - h) - v;
            let in_l = hi > hw && lo < 3.0 * hw; let in_r = lo < -hw && hi > -3.0 * hw;
            let in_own = hi > jm::min(-hw, lane_off - hw_i) && lo < jm::max(hw, lane_off + hw_i);
            let along = jm::abs(fwd) < hl_i + hl_j + 1.0;
            if along && in_l { alt_l = 1.0; }
            if along && in_r { alt_r = 1.0; }
            if fwd < -(hl_i + hl_j) {
                let gg = -fwd - hl_i - hl_j;
                if gg <= 60.0 {
                    let t = if vf > 0.1 { gg / vf } else { INF };
                    if in_l { if gg < gap_rl { gap_rl = gg; } if t < ttc_rl { ttc_rl = t; } }
                    if in_r { if gg < gap_rr { gap_rr = gg; } if t < ttc_rr { ttc_rr = t; } }
                    if in_own && t < ttc_own { ttc_own = t; }
                }
            }
            if jm::abs(fwd) < hl_i + hl_j {
                if left > 0.0 { let cl = jm::max(0.0, left - hw_i - hw_j); if cl < clr_l { clr_l = cl; } }
                else { let cl = jm::max(0.0, -left - hw_i - hw_j); if cl < clr_r { clr_r = cl; } }
            }
            true
        });
        let l = vs.leader[i];
        if l >= 0 {
            let lu = l as usize;
            if (vs.state[lu] == STATE_DRIVING || vs.state[lu] == STATE_CRASHED) && jm::abs(f(vs.y[lu]) - y) < LAYER_GAP {
                let dx = f(vs.x[lu]) - x; let dz = f(vs.z[lu]) - z; let d2 = dx * dx + dz * dz;
                if d2 <= r * r { sel.force(sector_of(dx * c + dz * s, dx * s - dz * c), d2, l); }
            }
        }
        let t1 = |t: f64| if t == INF { 1.0 } else { jm::min(1.0, t / 6.0) };
        let g1 = |g: f64, n: f64| if g == INF { 1.0 } else { jm::min(1.0, g / n) };
        let mo = MIRROR_OFF;
        out[mo] = alt_l as f32; out[mo + 1] = alt_r as f32; out[mo + 2] = t1(ttc_rl) as f32; out[mo + 3] = t1(ttc_rr) as f32;
        out[mo + 4] = g1(gap_rl, 60.0) as f32; out[mo + 5] = g1(gap_rr, 60.0) as f32; out[mo + 6] = t1(ttc_own) as f32;
        out[mo + 7] = g1(clr_l, 3.7) as f32; out[mo + 8] = g1(clr_r, 3.7) as f32;
        me = Body { x, z, h, v, hl: hl_i, hw: hw_i, ..Default::default() };
    }
    let m_me = movement_ahead(sim, i, CONFLICT_DIST);
    let conf = if m_me >= 0 { Some(sim.conflicts_ro(m_me as usize)) } else { None };
    let vs = &sim.vs;
    for k in 0..NB_K {
        let at = NB_OFF + k * NB_D; let j = sel.j[k];
        if j < 0 { out[at..at + NB_D].fill(0.0); continue; }
        let j = j as usize;
        let crashed = vs.state[j] != STATE_DRIVING;
        let (ind_l, ind_r) = indicator_of(sim, j);
        let mut conflict = false;
        if let Some(cf) = conf.as_ref() {
            if !cf.is_empty() { let mj = movement_ahead(sim, j, CONFLICT_DIST); if mj >= 0 { conflict = cf.iter().any(|q| q.other as i32 == mj); } }
        }
        let o = Body { x: f(vs.x[j]), z: f(vs.z[j]), h: f(vs.heading[j]), v: if crashed { 0.0 } else { f(vs.speed[j]) }, hl: f(vs.hl[j]), hw: f(vs.hw[j]),
            length: f(vs.length[j]), width: f(vs.width[j]), brake: f(vs.brake[j]), ind_l, ind_r, crashed, conflict };
        body_features(&me, &o, out, at);
    }
}

/// type of the stop line at the end of edge e (its last approach), or -1
fn end_sign_type(cc: &Controls, e: usize) -> i8 {
    let a = cc.edge_ap_start[e + 1] - 1;
    if a >= cc.edge_ap_start[e] { cc.ty[cc.ap_control[cc.edge_ap_list[a as usize] as usize] as usize] } else { -1 }
}

/// senseControls3: controls extras 79..86 for slot i.
pub fn sense_controls3(sim: &Sim, i: usize, out: &mut [f32]) {
    let g = sim.gw();
    let o = CTL3_OFF;
    for k in 0..8 { out[o + k] = 0.0; }
    out[o + 3] = 1.0;
    let Some(cc) = g.controls.as_ref() else { return };
    let a = sim.vs.ctl_approach[i];
    {
        let vs = &sim.vs;
        if a >= 0 && vs.ctl_type[i] == CTRL_SIGNAL && vs.ctl_state[i] == LIGHT_GREEN && vs.ctl_move[i] == MOVE_LEFT && cc.ap_prot_left[a as usize] != 1 { out[o] = 1.0; }
    }
    let m = movement_ahead(sim, i, CONFLICT_DIST);
    if m >= 0 {
        let jj = g.j.as_ref().unwrap();
        let in_e = jj.m_info[7 * m as usize] as usize; let my = end_sign_type(cc, in_e);
        let (mut n, mut signed, mut stops) = (0, 0, 0);
        for q in sim.conflicts_ro(m as usize).iter() {
            let e2 = jj.m_info[7 * q.other as usize] as usize; if e2 == in_e { continue; }
            let t = end_sign_type(cc, e2); n += 1;
            if t == CTRL_STOP || t == CTRL_GIVE_WAY { signed += 1; }
            if t == CTRL_STOP { stops += 1; }
        }
        if n > 0 && my != CTRL_STOP && my != CTRL_GIVE_WAY && signed == n { out[o + 1] = 1.0; }
        if n > 0 && my == CTRL_STOP && stops == n { out[o + 2] = 1.0; }
    }
    let vs = &sim.vs;
    if a >= 0 && vs.ctl_type[i] >= 0 {
        let a = a as usize;
        let route = &vs.routes[i]; let rp = vs.route_pos[i] as i64; let e_a = cc.ap_edge[a];
        let rl = route.len() as i64;
        let mut pos: i64 = -1;
        let mut k = rp; while k < rl && k < rp + 12 { if route[k as usize] == e_a { pos = k; break; } k += 1; }
        if pos >= 0 {
            let ea = e_a as usize;
            let (mut a2, mut d) = (-1i32, f(vs.ctl_dist[i])); let mut k: i64;
            for q in cc.edge_ap_start[ea]..cc.edge_ap_start[ea + 1] { let x = cc.edge_ap_list[q as usize]; if cc.ap_stop_at[x as usize] > cc.ap_stop_at[a] { a2 = x; d += cc.ap_stop_at[x as usize] - cc.ap_stop_at[a]; break; } }
            if a2 < 0 {
                d += g.e_length[ea] - cc.ap_stop_at[a];
                k = pos + 1;
                while k < rl && k < pos + 9 && d < 250.0 && a2 < 0 {
                    let en = route[k as usize] as usize;
                    if cc.edge_ap_start[en + 1] > cc.edge_ap_start[en] { a2 = cc.edge_ap_list[cc.edge_ap_start[en] as usize]; d += cc.ap_stop_at[a2 as usize]; break; }
                    d += g.e_length[en];
                    k += 1;
                }
            } else { k = pos; }
            if a2 >= 0 && d < 250.0 {
                let e2 = cc.ap_edge[a2 as usize]; let p2 = if e2 == e_a { pos } else { k };
                let mut mv = MOVE_THROUGH; if p2 + 1 < rl { mv = movement_of(&g, e2 as usize, route[(p2 + 1) as usize] as usize); }
                let (ty, st, _, _) = state_of(&g, a2 as usize, mv, sim.time);
                out[o + 3] = jm::min(1.0, jm::max(0.0, d) / 250.0) as f32;
                if ty == CTRL_SIGNAL { out[o + 4] = 1.0; } else if ty == CTRL_STOP || ty == CTRL_GIVE_WAY { out[o + 5] = 1.0; } else { out[o + 6] = 1.0; }
                out[o + 7] = if st == LIGHT_RED || st == LIGHT_YELLOW { 1.0 } else { 0.0 };
            }
        }
    }
}

/// senseCommands: own command history 87..93 from the reaction-delay ring.
pub fn sense_commands(sim: &Sim, i: usize, out: &mut [f32]) {
    let vs = &sim.vs; let r = &vs.ctrl_ring; let s = sim.step_count; let d = vs.delay_steps[i] as i64;
    let ring = CTRL_RING as i64;
    let slot = |st: i64| (i * CTRL_RING + (((st % ring) + ring) % ring) as usize) * CTRL_STRIDE;
    let now = slot(if d >= 1 { s - d } else { s - 1 }); let last = slot(s - 1);
    out[HIST_OFF] = r[now]; out[HIST_OFF + 1] = r[now + 1]; out[HIST_OFF + 2] = r[now + 2];
    out[HIST_OFF + 3] = r[last]; out[HIST_OFF + 4] = r[last + 1]; out[HIST_OFF + 5] = r[last + 2];
    out[HIST_VISIBLE] = 1.0;
}

/// fillObs3: input version 3 into out (len >= OBS3), for an `obs` just filled by `fill_obs` for the same slot.
/// The whole vector is zeroed first (JS: a fresh Float32Array per observe), so 0..69 and 94..101 are 0.
pub fn fill_obs3(sim: &Sim, i: usize, obs: &Obs, out: &mut [f32]) {
    let out = &mut out[..OBS3];
    out.fill(0.0);
    sense_bodies(sim, i, out, obs.lane_width, obs.lane_offset);
    sense_controls3(sim, i, out);
    sense_commands(sim, i, out);
}

/// One env.observe() entry: fill_obs (+ fill_obs3 into v3 when given) + signals.
pub fn observe_one(sim: &mut Sim, i: usize, obs: &mut Obs, v3: Option<&mut [f32]>) {
    observe_one_ro(sim, i, obs, v3);
    apply_writes(sim, i, obs);
}

/// observe_one without its side effect (the junction-yield blocker write, kept in `obs.yield_write`): for parallel
/// observation; apply it afterwards (`apply_writes`, any order: each write is to the car's own entry).
pub fn observe_one_ro(sim: &Sim, i: usize, obs: &mut Obs, v3: Option<&mut [f32]>) {
    fill_obs_ro(sim, i, obs);
    if let Some(v) = v3 { fill_obs3(sim, i, obs, v); }
    obs.leader_id = sim.vs.leader[i];
    obs.signals = sim.signals_for(i);
}

/// The state effect of an observation of slot i (ctx.yieldRequired -> _yieldBy[i]).
pub fn apply_writes(sim: &mut Sim, i: usize, obs: &Obs) { if let Some(v) = obs.yield_write { sim.set_yield_by(i, v); } }

/// The slots env.observe() covers (every driving car, ascending slot).
pub fn driving_slots(sim: &Sim) -> Vec<usize> { (0..sim.vs.high).filter(|&i| sim.vs.state[i] == STATE_DRIVING).collect() }

/// f32 per row of observe_flat.
pub const fn flat_stride(v3: bool) -> usize { OBS_N + if v3 { OBS3 } else { 0 } }

/// Flat observations of `slots` (in that order) into out (rows of flat_stride(v3) f32; layout in the module doc).
/// Slots must be driving cars (as env.observe). Same state effects as calling env.observe on those slots.
pub fn observe_flat(sim: &mut Sim, slots: &[usize], out: &mut [f32], v3: bool) {
    let st = flat_stride(v3);
    assert!(out.len() >= slots.len() * st, "observe_flat: out too small");
    let row_fn = |sim: &Sim, i: usize, row: &mut [f32]| -> Option<i32> {
        let mut obs = Obs::default();
        let (head, tail) = row.split_at_mut(OBS_N);
        observe_one_ro(sim, i, &mut obs, if v3 { Some(tail) } else { None });
        let sc = obs.scalars(i, sim.vs.commuter[i]);
        for (d, s) in head.iter_mut().zip(sc.iter()) { *d = *s as f32; }
        obs.yield_write
    };
    let writes: Vec<Option<i32>> = if sim.opts.parallel && slots.len() >= 256 {
        use rayon::prelude::*;
        let me: &Sim = sim;
        out[..slots.len() * st].par_chunks_mut(st).zip(slots.par_iter()).with_min_len(32).map(|(row, &i)| row_fn(me, i, row)).collect()
    } else { out[..slots.len() * st].chunks_mut(st).zip(slots.iter()).map(|(row, &i)| row_fn(sim, i, row)).collect() };
    for (k, w) in writes.into_iter().enumerate() { if let Some(v) = w { sim.set_yield_by(slots[k], v); } }
}
