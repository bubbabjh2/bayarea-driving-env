//! Bridge-rail breach and free fall (`SimOpts::barrier_breach`, default OFF; DECISIONS.md 46). CPU only, no JS
//! counterpart: with the flag off nothing here runs and the sim is unchanged.
//!
//! When a driving car strikes the outer (right-hand) barrier of a bridge edge with an impact severity above the rail's
//! capacity, it goes through the rail. Impact severity is the crash-test measure of AASHTO MASH (2016) and NCHRP 350:
//! IS = 1/2 m (v sin theta)^2, the kinetic energy of the velocity component normal to the rail. The default capacity is
//! the MASH Test Level 3 strength test (test 3-11: 2270 kg pickup, 100 km/h, 25 degrees: IS = 156 kJ), the level a
//! bridge rail on a high-speed road is certified to contain. A rail certified at TL-3 is not guaranteed past that, so
//! the sim lets it fail above it (`SimOpts::breach_is_kj`).
//!
//! The rail absorbs its capacity from the normal motion, and rail friction (Coulomb, WALL_FRICTION) takes the matching
//! share of the tangential speed. The car is then a free rigid body: it leaves the vehicle store (the trip fails as a
//! crash, the slot is freed, so it can never arrive and no lane sees it) and lives on in `Sim::falls`. It slides over
//! the deck edge until its centre is past the rail line, then falls under gravity with a constant tumble rate, lands
//! on the ground below (`Sim::ground`: the DEM, or sea level 0 without one), and slides to rest, settling onto its
//! wheels or its roof. On water it sinks to `SINK_DEPTH`. A fall only moves itself: it does not hit other cars or a
//! lower deck.
use crate::jm;
use crate::physics::{Body, Contact, WALL_FRICTION};
use crate::sim::{f, Sim, FAIL_CRASH};
use crate::vehicle::*;
use std::sync::Arc;

pub const GRAVITY: f64 = 9.81;
/// MASH TL-3 strength test (2270P pickup, 100 km/h, 25 deg): IS = 1/2 * 2270 * (27.78 * sin 25)^2 = 156 kJ
pub const MASH_TL3_IS_KJ: f64 = 156.0;
/// smallest outward speed through the broken rail (m/s): a car that only just breaks the rail still tips over it
pub const BREACH_MIN_EXIT: f64 = 1.0;
/// integration step of the fall (s)
pub const FALL_SUBSTEP: f64 = 0.01;
/// tumble given at the rail: roll about the direction of travel, toward the outside (rad/s per m/s of normal impact
/// speed, clamped), and a nose-down pitch. Visual: not derived from the contact.
pub const ROLL_PER_MS: f64 = 0.25;
pub const ROLL_MIN: f64 = 0.8;
pub const ROLL_MAX: f64 = 3.0;
pub const PITCH_RATE: f64 = 0.6;
/// sliding friction on the ground after landing (g)
pub const LAND_MU: f64 = 0.7;
/// how deep a car in the water settles (m below the surface), and how fast it sinks (m/s)
pub const SINK_DEPTH: f64 = 1.2;
pub const SINK_SPEED: f64 = 0.4;

/// The surface under a falling car: (height m, is water).
pub trait Ground: Send + Sync {
    fn height(&self, x: f64, z: f64) -> (f64, bool);
}
/// A flat surface at height y (tests; sea level without a DEM).
pub struct FlatGround { pub y: f64, pub water: bool }
impl Ground for FlatGround { fn height(&self, _x: f64, _z: f64) -> (f64, bool) { (self.y, self.water) } }
/// The committed 20 m ground grid (`data/dem/compact`): bare earth, road corridors conformed to the road surface. Water
/// cells (mask bit 16) and cells outside the grid are the bay at sea level 0.
pub struct DemGround(pub Arc<bayenv_world::extras::Dem>);
pub const DEM_WATER_BIT: u8 = 16;
impl Ground for DemGround {
    fn height(&self, x: f64, z: f64) -> (f64, bool) {
        let h = self.0.height_at(x, z);
        if !h.is_finite() || self.0.mask_at(x, z) & DEM_WATER_BIT != 0 { (0.0, true) } else { (h, false) }
    }
}
impl DemGround {
    /// Load `<data_dir>/dem/compact/ground20.*`.
    pub fn load(data_dir: &std::path::Path) -> std::io::Result<Arc<dyn Ground>> { Ok(Arc::new(DemGround(Arc::new(bayenv_world::extras::Dem::load(data_dir)?)))) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FallPhase {
    /// still over the deck: sliding outward until the centre passes the rail line
    Edge,
    /// ballistic
    Air,
    /// on the ground or in the water, sliding / settling / sinking
    Landed,
    /// at rest (a wreck below the bridge)
    Rest,
}

/// One car that went through a bridge rail.
#[derive(Clone, Debug)]
pub struct Fall {
    /// the slot it had (freed at the breach; may be reused by a new car), its commuter and vehicle type
    pub slot: usize, pub commuter: i32, pub type_idx: u8,
    /// half length / half width (m)
    pub hl: f64, pub hw: f64,
    pub edge: i32, pub time: f64, pub step: i64,
    /// impact: speed normal to the rail (m/s), impact severity (kJ), the speed along it (m/s), own delta-v (km/h)
    pub v_normal: f64, pub is_kj: f64, pub v_along: f64, pub dv_kmh: f64,
    /// rail line point and outward normal (x, z)
    pub rail: (f64, f64), pub normal: (f64, f64),
    /// height of the deck at the breach (m)
    pub deck_y: f64,
    /// centre position (y = the bottom of the car, as the sim's pose y = the road surface), velocity (m/s)
    pub pos: [f64; 3], pub vel: [f64; 3],
    /// orientation quaternion (x, y, z, w) from the car's frame (x forward, y up, z to the right) into the world
    /// (three.js convention: heading h alone = rotation -h about +y); world angular velocity (rad/s)
    pub q: [f64; 4], pub w: [f64; 3],
    pub phase: FallPhase,
    /// landing: time, height of the surface, vertical speed at impact (m/s), drop from the deck (m), water
    pub land_time: f64, pub land_y: f64, pub land_vy: f64, pub drop: f64, pub water: bool,
}

#[inline] fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1], a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
     a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3], a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]]
}
#[inline] fn qnorm(q: [f64; 4]) -> [f64; 4] { let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt(); [q[0] / n, q[1] / n, q[2] / n, q[3] / n] }
/// rotate v by q
#[inline] pub fn qrot(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let p = qmul(qmul(q, [v[0], v[1], v[2], 0.0]), [-q[0], -q[1], -q[2], q[3]]);
    [p[0], p[1], p[2]]
}
/// q advanced by the world angular velocity w over h (exact for constant w)
#[inline] fn qstep(q: [f64; 4], w: [f64; 3], h: f64) -> [f64; 4] {
    let a = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt() * h;
    if a < 1e-12 { return q; }
    let s = (0.5 * a).sin() / (a / h);
    qnorm(qmul([w[0] * s, w[1] * s, w[2] * s, (0.5 * a).cos()], q))
}
#[inline] fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }

impl Fall {
    /// sim heading convention (forward = (cos h, sin h) in x, z)
    pub fn heading(&self) -> f64 { let fw = qrot(self.q, [1.0, 0.0, 0.0]); fw[2].atan2(fw[0]) }
    pub fn speed(&self) -> f64 { (self.vel[0] * self.vel[0] + self.vel[1] * self.vel[1] + self.vel[2] * self.vel[2]).sqrt() }
    /// [x, y, z, heading, speed, state (2 = crashed), qx, qy, qz, qw]: the film replay frame (src/film/replayTraffic.js)
    pub fn frame(&self) -> [f64; 10] {
        [self.pos[0], self.pos[1], self.pos[2], self.heading(), self.speed(), STATE_CRASHED as f64, self.q[0], self.q[1], self.q[2], self.q[3]]
    }
    /// advance by h (one sub-step)
    fn advance(&mut self, ground: &dyn Ground, time: f64, h: f64) {
        match self.phase {
            FallPhase::Edge => {
                self.pos[0] += self.vel[0] * h; self.pos[2] += self.vel[2] * h;
                self.q = qstep(self.q, self.w, h);
                let past = (self.pos[0] - self.rail.0) * self.normal.0 + (self.pos[2] - self.rail.1) * self.normal.1;
                if past > 0.0 {
                    // over the edge: tumble outward (the top toward the normal) and nose-down along the travel direction
                    self.phase = FallPhase::Air;
                    let up = [0.0, 1.0, 0.0];
                    let n = [self.normal.0, 0.0, self.normal.1];
                    let hv = jm::hypot(self.vel[0], self.vel[2]);
                    let fw = if hv > 1e-6 { [self.vel[0] / hv, 0.0, self.vel[2] / hv] } else { n };
                    let roll = jm::min(ROLL_MAX, jm::max(ROLL_MIN, ROLL_PER_MS * self.v_normal));
                    let r = cross(up, n); let p = cross(up, fw);
                    for k in 0..3 { self.w[k] += roll * r[k] + PITCH_RATE * p[k]; }
                }
            }
            FallPhase::Air => {
                self.vel[1] -= GRAVITY * h;
                for k in 0..3 { self.pos[k] += self.vel[k] * h; }
                self.q = qstep(self.q, self.w, h);
                let (gy, water) = ground.height(self.pos[0], self.pos[2]);
                if self.pos[1] <= gy {
                    self.phase = FallPhase::Landed;
                    self.land_time = time; self.land_y = gy; self.land_vy = self.vel[1]; self.drop = self.deck_y - gy; self.water = water;
                    self.pos[1] = gy; self.vel[1] = 0.0; self.w = [0.0; 3];
                }
            }
            FallPhase::Landed => {
                let (gy, water) = ground.height(self.pos[0], self.pos[2]);
                // horizontal: sliding friction on land, strong drag in the water
                let hv = jm::hypot(self.vel[0], self.vel[2]);
                let dec = if water || self.water { 3.0 } else { LAND_MU * GRAVITY } * h;
                let k = if hv > dec { (hv - dec) / hv } else { 0.0 };
                self.vel[0] *= k; self.vel[2] *= k;
                self.pos[0] += self.vel[0] * h; self.pos[2] += self.vel[2] * h;
                // settle onto the wheels or the roof, whichever is nearer
                let up = qrot(self.q, [0.0, 1.0, 0.0]);
                let target = [0.0, if up[1] >= 0.0 { 1.0 } else { -1.0 }, 0.0];
                let ax = cross(up, target); let s = (ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]).sqrt();
                let ang = s.atan2(up[0] * target[0] + up[1] * target[1] + up[2] * target[2]);
                let mut settled = ang < 1e-3;
                if !settled && s > 1e-9 {
                    let step = ang * jm::min(1.0, 6.0 * h);
                    let wv = [ax[0] / s * step / h, ax[1] / s * step / h, ax[2] / s * step / h];
                    self.q = qstep(self.q, wv, h);
                } else if s <= 1e-9 && ang > 1e-3 {
                    // exactly upside-down about a degenerate axis: roll about the forward axis
                    let fw = qrot(self.q, [1.0, 0.0, 0.0]);
                    self.q = qstep(self.q, [fw[0] * 3.0, fw[1] * 3.0, fw[2] * 3.0], h);
                } else { settled = true; }
                if self.water {
                    let floor = self.land_y - SINK_DEPTH;
                    self.pos[1] = jm::max(floor, self.pos[1] - SINK_SPEED * h);
                    if self.pos[1] <= floor && k == 0.0 && settled { self.phase = FallPhase::Rest; }
                } else {
                    self.pos[1] = gy;
                    if k == 0.0 && settled { self.phase = FallPhase::Rest; }
                }
            }
            FallPhase::Rest => {}
        }
    }
}

impl Sim {
    /// Surface height under (x, z): `ground`, or sea level 0 (water) without one.
    pub fn ground_at(&self, x: f64, z: f64) -> (f64, bool) { match &self.ground { Some(g) => g.height(x, z), None => (0.0, true) } }

    /// barrier(): would this strike breach the rail (SimOpts::barrier_breach)? `side` < 0: the right-hand (outer)
    /// barrier; `a` the car body before the impulse, (nx, nz) the outward wall normal. Returns (normal speed, IS in J).
    pub(crate) fn breach_test(&self, e: usize, side: f64, in_box: bool, a: &Body, nx: f64, nz: f64) -> Option<(f64, f64)> {
        if !self.opts.barrier_breach || side >= 0.0 || in_box || !self.g.e_is_bridge.get(e).copied().unwrap_or(false) || a.inv_mass <= 0.0 { return None; }
        let vn = a.vx * nx + a.vz * nz;
        let is = 0.5 / a.inv_mass * vn * vn;
        if vn > 0.0 && is > self.opts.breach_is_kj * 1000.0 { Some((vn, is)) } else { None }
    }

    /// Car i goes through the rail: crash bookkeeping (disabled, trip failed as a crash, CRASHED event, a crash
    /// record with cause "breach" that blocks no lane), then the car leaves the store and becomes a `Fall`.
    pub(crate) fn breach(&mut self, i: usize, a: &Body, c: &Contact, vn: f64, is: f64) {
        let m = 1.0 / a.inv_mass;
        let (nx, nz) = (c.nx, c.nz);
        let vn_out = jm::max(BREACH_MIN_EXIT, (jm::max(0.0, vn * vn - 2.0 * self.opts.breach_is_kj * 1000.0 / m)).sqrt());
        let (tx, tz) = (a.vx - vn * nx, a.vz - vn * nz);
        let vt = jm::hypot(tx, tz);
        let vt_out = jm::max(0.0, vt - WALL_FRICTION * jm::max(0.0, vn - vn_out));
        let k = if vt > 1e-9 { vt_out / vt } else { 0.0 };
        let (vx, vz) = (tx * k + vn_out * nx, tz * k + vn_out * nz);
        let dv_kmh = jm::max(jm::hypot(a.vx - vx, a.vz - vz) * 3.6, crate::contacts::DISABLE_KMH);
        // bookkeeping, as a disabling barrier crash (crash_event) but with no lane-blocking scene
        let vs = &mut self.vs;
        vs.flags[i] |= FLAG_COLLIDED | FLAG_DISABLED;
        let speeding = vs.flags[i] & FLAG_SPEEDING != 0;
        let severity = crate::contacts::severity_of(dv_kmh);
        if speeding { self.counters.crashes_while_speeding += 1; }
        self.counters.crashes += 1; self.counters.disabling_crashes += 1; self.counters.crashed_vehicles += 1;
        *self.counters.by_severity.entry(severity).or_insert(0) += 1;
        let share = crate::contacts::crash_share(jm::hypot(a.vx, a.vz), a.yaw, 0.0, 0.0, nx, nz);
        self.collided_last.push(i);
        self.events.crashed.push(crate::sim::CrashedEv { slot: i, agent: self.vs.commuter[i], dv_kmh, share, disabled: true });
        self.emit(crate::sim::EV_CRASHED, i, -1, dv_kmh as f32, f32::NAN, share as f32, f32::NAN);
        let e = self.vs.edge[i];
        let (deck_y, yaw) = (f(self.vs.y[i]), f(self.vs.heading[i]));
        let id = self.crashes.len();
        self.crashes.push(crate::sim::Scene { id, time: self.time, cause: "breach", severity, is_crash: true, scene: false, vehicles: vec![i], commuters: vec![self.vs.commuter[i]],
            edge: e, s: f(self.vs.s[i]), lane_mask: 0, lanes_blocked: 0, x: c.px, z: c.pz, y: deck_y, block_start: self.time, block_end: self.time, clear_end: self.time,
            squeeze_at: 0.0, squeeze: false, response: crate::sim::Response::Cleared, permanent: false, gone: vec![i], delta_v_kmh: vec![dv_kmh], closing: vn, disabled: vec![i],
            crash_type: "barrier breach".into(), maneuver: "" });
        let fall = Fall {
            slot: i, commuter: self.vs.commuter[i], type_idx: self.vs.type_idx[i], hl: f(self.vs.hl[i]), hw: f(self.vs.hw[i]), edge: e, time: self.time, step: self.step_count,
            v_normal: vn, is_kj: is / 1000.0, v_along: vt, dv_kmh, rail: (c.px - nx * c.depth, c.pz - nz * c.depth), normal: (nx, nz), deck_y,
            pos: [a.x, deck_y, a.z], vel: [vx, 0.0, vz], q: [0.0, (-0.5 * yaw).sin(), 0.0, (-0.5 * yaw).cos()], w: [0.0, a.w, 0.0], phase: FallPhase::Edge,
            land_time: f64::NAN, land_y: f64::NAN, land_vy: f64::NAN, drop: f64::NAN, water: false,
        };
        self.falls.push(fall);
        self.vs.state[i] = STATE_CRASHED; self.vs.clear_controls(i);
        self.fail_trip(i, FAIL_CRASH, true);
    }

    /// advance every fall by dt (after integrate_all)
    pub(crate) fn step_falls(&mut self, dt: f64) {
        let ground: Arc<dyn Ground> = match &self.ground { Some(g) => g.clone(), None => Arc::new(FlatGround { y: 0.0, water: true }) };
        let n = jm::max(1.0, (dt / FALL_SUBSTEP).ceil()) as usize;
        let h = dt / n as f64;
        let t0 = self.time - dt;
        for fl in self.falls.iter_mut() {
            if fl.phase == FallPhase::Rest { continue; }
            for k in 0..n { fl.advance(&*ground, t0 + (k + 1) as f64 * h, h); }
        }
    }
}
