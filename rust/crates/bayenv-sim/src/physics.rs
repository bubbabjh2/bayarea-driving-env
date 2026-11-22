//! src/sim/physics: vehicle parameters (params.js), dynamic bicycle model (dynamics.js) and planar
//! rigid-body contacts (collide.js). Same expressions in the same order as the JS.
use crate::jm;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Drive { Front, Rear, All }

#[derive(Clone, Copy, Debug)]
pub struct VehicleParams {
    pub length: f64, pub width: f64, pub l: f64, pub a: f64, pub b: f64, pub hcg: f64, pub m: f64, pub iz: f64,
    pub cda: f64, pub crr: f64, pub pmax: f64, pub f0: f64, pub drive: Drive, pub vmax: f64,
    pub brake_decel: f64, pub brake_front: f64, pub cf: f64, pub cr: f64, pub mu: f64,
    pub max_steer: f64, pub max_steer_rate: f64, pub steer_slew: f64, pub steer_lag: f64,
    pub diesel: bool, pub bsfc_min: f64, pub idle_lph: f64, pub accessory_w: f64, pub tank_l: f64,
    /// vehicleTypes.js emergencyDecel = min(brakeDecel, mu * 9.81)
    pub emergency_decel: f64,
}

const fn vp(length: f64, width: f64, l: f64, a: f64, b: f64, hcg: f64, m: f64, iz: f64, cda: f64, crr: f64, pmax: f64, f0: f64, drive: Drive, vmax: f64,
    brake_decel: f64, brake_front: f64, cf: f64, cr: f64, mu: f64, max_steer: f64, max_steer_rate: f64, steer_slew: f64, steer_lag: f64,
    diesel: bool, bsfc_min: f64, idle_lph: f64, accessory_w: f64, tank_l: f64, emergency_decel: f64) -> VehicleParams {
    VehicleParams { length, width, l, a, b, hcg, m, iz, cda, crr, pmax, f0, drive, vmax, brake_decel, brake_front, cf, cr, mu, max_steer, max_steer_rate,
        steer_slew, steer_lag, diesel, bsfc_min, idle_lph, accessory_w, tank_l, emergency_decel }
}

/// VEHICLE_PARAMS (sedan, suv, truck, bus). emergencyDecel: min(brakeDecel, mu * 9.81) = 9.81, 9.81, 6.5, 6.5.
pub static VEHICLE_PARAMS: [VehicleParams; 4] = [
    vp(4.5, 1.85, 2.75, 1.10, 1.65, 0.55, 1500.0, 2400.0, 0.66, 0.011, 100e3, 5600.0, Drive::Front, 55.0, 11.0, 0.68, 90e3, 100e3, 1.0, 0.60, 0.70, 0.80, 0.10, false, 265.0, 0.8, 1500.0, 50.0, 9.81),
    vp(4.9, 1.95, 2.90, 1.30, 1.60, 0.70, 2100.0, 4100.0, 0.95, 0.012, 150e3, 8500.0, Drive::All, 50.0, 10.5, 0.65, 115e3, 125e3, 1.0, 0.58, 0.65, 0.73, 0.10, false, 250.0, 1.1, 1100.0, 70.0, 9.81),
    vp(9.0, 2.5, 5.5, 2.4, 3.1, 1.25, 15000.0, 1.1e5, 6.0, 0.0065, 260e3, 55e3, Drive::Rear, 31.0, 6.5, 0.45, 2.6e5, 5.2e5, 0.8, 0.50, 0.45, 0.53, 0.12, true, 200.0, 2.5, 3000.0, 300.0, 6.5),
    vp(12.0, 2.55, 6.0, 3.6, 2.4, 1.15, 13000.0, 1.6e5, 7.0, 0.008, 200e3, 48e3, Drive::Rear, 30.0, 6.5, 0.45, 2.4e5, 4.4e5, 0.8, 0.50, 0.45, 0.53, 0.12, true, 205.0, 2.2, 3000.0, 250.0, 6.5),
];
pub const VEHICLE_TYPE_WEIGHTS: [f64; 4] = [0.62, 0.30, 0.06, 0.02];

pub const RHO_AIR: f64 = 1.225;
pub const G: f64 = 9.81;
pub const PHYS_DT: f64 = 0.01;
/// brake deceleration of trucks and buses with SimOpts::heavy_brakes (the JS: 6.5 m/s^2, 100-0 in 58 m; real loaded
/// heavy vehicles need about 80 m, FMVSS 121)
pub const HEAVY_BRAKE_DECEL: f64 = 4.7;
/// Model fixes inside step_dynamics (all false = the JS)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fixes { pub hold: bool, pub heavy_brakes: bool, pub reverse: bool, pub static_hold: bool, pub idle_creep: bool, pub rollback: bool }
/// idle creep (SimOpts::idle_creep): an automatic's idle drive force, IDLE_CREEP_ROLL x the rolling resistance at rest,
/// fading linearly to 0 at IDLE_CREEP_U; on the flat a car with no pedals settles at
/// IDLE_CREEP_U x (1 - 1 / IDLE_CREEP_ROLL) = 1.5 m/s
pub const IDLE_CREEP_ROLL: f64 = 4.0;
pub const IDLE_CREEP_U: f64 = 2.0;
pub const KIN_BLEND_LO: f64 = 0.7;
pub const KIN_BLEND_HI: f64 = 1.4;
pub const ABS_LONG: f64 = 0.95;
pub const EBD_MARGIN: f64 = 0.05;

#[inline]
pub fn resistance(p: &VehicleParams, u: f64, grade: f64, crr_mul: f64) -> f64 {
    let roll = if u > 0.05 { p.crr * crr_mul * p.m * G } else { 0.0 };
    0.5 * RHO_AIR * p.cda * u * jm::abs(u) + roll + p.m * G * grade / jm::sqrt(1.0 + grade * grade)
}
#[inline]
pub fn bsfc(p: &VehicleParams, load: f64) -> f64 { let q = 1.0 - jm::min(1.0, jm::max(0.0, load)); p.bsfc_min * (1.0 + 0.9 * q * q * q) }
#[inline]
pub fn fuel_flow(p: &VehicleParams, pw: f64) -> f64 {
    let idle = p.idle_lph / 3600.0;
    if pw <= 0.0 { return idle; }
    let gps = (pw / 1000.0) * bsfc(p, pw / p.pmax) / 3600.0;
    jm::max(idle, gps / if p.diesel { 832.0 } else { 745.0 })
}

/// Physics state (dynamics.js createState).
#[derive(Clone, Copy, Debug, Default)]
pub struct DynState { pub u: f64, pub v: f64, pub r: f64, pub steer: f64, pub sl: f64, pub ax: f64, pub hr: f64, pub lat: f64, pub s: f64, pub fuel_l: f64, pub power: f64 }

/// Road-surface factors under the wheels (surface.js carSurface).
#[derive(Clone, Copy, Debug)]
pub struct Surf { pub pk: f64, pub sk: f64, pub crr: f64 }

/// dynamics.js stepDynamics (h0 = PHYS_DT). `position` = actionMode 'position'.
pub fn step_dynamics(st: &mut DynState, steer_rate: f64, gas: f64, brake: f64, p: &VehicleParams, dt: f64, grade: f64, locked: bool, position: bool, surf: Option<Surf>) {
    step_dynamics_h(st, steer_rate, gas, brake, p, dt, grade, locked, position, surf, Fixes::default())
}
/// step_dynamics with the brake-hold model fix selectable (`hold`)
#[allow(clippy::too_many_arguments)]
pub fn step_dynamics_h(st: &mut DynState, steer_rate: f64, gas: f64, brake: f64, p: &VehicleParams, dt: f64, grade: f64, locked: bool, position: bool, surf: Option<Surf>, fx: Fixes) {
    let (mut d, n) = Dyn::new(st, steer_rate, gas, brake, p, dt, grade, locked, position, surf, fx);
    for _ in 0..n { d.substep(); }
    d.finish(st);
}

/// step_dynamics for two cars with the same dt (same substep count), their substeps interleaved: the two dependency
/// chains are independent, so the core overlaps them. Each car's operations and their order are step_dynamics' own.
#[allow(clippy::type_complexity)]
pub fn step_dynamics2(a: (&mut DynState, f64, f64, f64, &VehicleParams, f64, bool, Option<Surf>), b: (&mut DynState, f64, f64, f64, &VehicleParams, f64, bool, Option<Surf>), dt: f64, position: bool, fx: Fixes) {
    let (sa, sra, ga, ba, pa, gra, la, sfa) = a;
    let (sb, srb, gb, bb, pb, grb, lb, sfb) = b;
    let (mut da, n) = Dyn::new(sa, sra, ga, ba, pa, dt, gra, la, position, sfa, fx);
    let (mut db, _) = Dyn::new(sb, srb, gb, bb, pb, dt, grb, lb, position, sfb, fx);
    for _ in 0..n { da.substep(); db.substep(); }
    da.finish(sa); db.finish(sb);
}

/// step_dynamics for four cars (same dt), substeps interleaved as in step_dynamics2.
#[allow(clippy::type_complexity)]
pub fn step_dynamics4(c: [(&mut DynState, f64, f64, f64, &VehicleParams, f64, bool, Option<Surf>); 4], dt: f64, position: bool, fx: Fixes) {
    let [(s0, r0, g0, b0, p0, gr0, l0, f0), (s1, r1, g1, b1, p1, gr1, l1, f1), (s2, r2, g2, b2, p2, gr2, l2, f2), (s3, r3, g3, b3, p3, gr3, l3, f3)] = c;
    let (mut d0, n) = Dyn::new(s0, r0, g0, b0, p0, dt, gr0, l0, position, f0, fx);
    let (mut d1, _) = Dyn::new(s1, r1, g1, b1, p1, dt, gr1, l1, position, f1, fx);
    let (mut d2, _) = Dyn::new(s2, r2, g2, b2, p2, dt, gr2, l2, position, f2, fx);
    let (mut d3, _) = Dyn::new(s3, r3, g3, b3, p3, dt, gr3, l3, position, f3, fx);
    for _ in 0..n { d0.substep(); d1.substep(); d2.substep(); d3.substep(); }
    d0.finish(s0); d1.finish(s1); d2.finish(s2); d3.finish(s3);
}

/// step_dynamics' per-call constants and loop state (the substep body is the JS loop body verbatim)
#[derive(Clone, Copy)]
struct Dyn<'a> {
    h: f64, m: f64, iz: f64, a: f64, b: f64, l: f64, hcg: f64, cf: f64, cr: f64, mu: f64, k_slide: f64, crr_mul: f64,
    sr: f64, g_: f64, b_: f64, cos_g: f64, sin_g: f64, wg: f64, tgt: f64, k_lag: f64, slew: f64, locked: bool, position: bool, hold: bool, rev: bool, shold: bool, creep: bool, rollb: bool, bd: f64,
    p: &'a VehicleParams, grade: f64,
    u: f64, v: f64, r: f64, steer: f64, ax: f64, hr: f64, lat: f64, s: f64, sl: f64, fuel: f64, power: f64,
    m_steer: f64, m_cd: f64, m_sd: f64, m_hr: f64, m_ch: f64, m_sh: f64,
}

impl<'a> Dyn<'a> {
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn new(st: &DynState, steer_rate: f64, gas: f64, brake: f64, p: &'a VehicleParams, dt: f64, grade: f64, locked: bool, position: bool, surf: Option<Surf>, fx: Fixes) -> (Dyn<'a>, i64) {
    let (hold, rev) = (fx.hold, fx.reverse);
    let (shold, creep, rollb) = (fx.static_hold, fx.idle_creep, fx.rollback);
    // heavy-vehicle brake calibration (SimOpts::heavy_brakes): trucks / buses 100-0 km/h in about 80 m (FMVSS 121)
    let bd = if fx.heavy_brakes && p.m > 5000.0 { HEAVY_BRAKE_DECEL } else { p.brake_decel };
    let n = jm::max(1.0, jm::ceil(dt / PHYS_DT - 1e-9));
    let h = dt / n;
    let n = n as i64;
    let (m, iz, a, b, l, hcg, cf, cr) = (p.m, p.iz, p.a, p.b, p.l, p.hcg, p.cf, p.cr);
    let (mu, k_slide, crr_mul) = match surf { Some(s) => (p.mu * s.pk, s.sk / s.pk, s.crr), None => (p.mu, 0.8, 1.0) };
    let sr = if steer_rate < -1.0 { -1.0 } else if steer_rate > 1.0 { 1.0 } else { jm::or(steer_rate, 0.0) };
    let g_ = if gas < 0.0 { 0.0 } else if gas > 1.0 { 1.0 } else { jm::or(gas, 0.0) };
    let b_ = if brake < 0.0 { 0.0 } else if brake > 1.0 { 1.0 } else { jm::or(brake, 0.0) };
    let cos_g = 1.0 / jm::sqrt(1.0 + grade * grade);
    let sin_g = grade * cos_g;
    let wg = p.m * G * sin_g;
    let (mut u, mut v, mut r, mut steer, mut ax, mut hr, mut lat, mut s) = (st.u, st.v, st.r, st.steer, st.ax, st.hr, st.lat, st.s);
    let tgt = sr * p.max_steer;
    let k_lag = jm::min(1.0, (dt / n as f64) / jm::or(p.steer_lag, 0.1));
    let slew = jm::or(p.steer_slew, p.max_steer_rate) * (dt / n as f64);
    let mut sl = st.sl;
    let mut fuel = 0.0;
    let mut power = 0.0;
    // memo of cos/sin(steer): the same input gives the same bits, so reuse while the wheel angle is unchanged
    let (mut m_steer, mut m_cd, mut m_sd) = (f64::NAN, 0.0, 0.0);
    let (mut m_hr, mut m_ch, mut m_sh) = (f64::NAN, 0.0, 0.0);
        (Dyn { h, m, iz, a, b, l, hcg, cf, cr, mu, k_slide, crr_mul, sr, g_, b_, cos_g, sin_g, wg, tgt, k_lag, slew, locked, position, hold, rev, shold, creep, rollb, bd, p, grade, u, v, r, steer, ax, hr, lat, s, sl, fuel, power, m_steer, m_cd, m_sd, m_hr, m_ch, m_sh }, n)
    }

    #[inline(always)]
    #[allow(unused_assignments)]
    fn substep(&mut self) {
        let Dyn { h, m, iz, a, b, l, hcg, cf, cr, mu, k_slide, crr_mul, sr, g_, b_, cos_g, sin_g, wg, tgt, k_lag, slew, locked, position, hold, rev, shold, creep, rollb, bd, p, grade, .. } = *self;
        let Dyn { mut u, mut v, mut r, mut steer, mut ax, mut hr, mut lat, mut s, mut sl, mut fuel, mut power, mut m_steer, mut m_cd, mut m_sd, mut m_hr, mut m_ch, mut m_sh, .. } = *self;
        if !locked {
            if position { sl += (tgt - sl) * k_lag; let d = sl - steer; steer += if d > slew { slew } else if d < -slew { -slew } else { d }; }
            else { steer += sr * p.max_steer_rate * h; sl = steer; }
            if steer > p.max_steer { steer = p.max_steer; } else if steer < -p.max_steer { steer = -p.max_steer; }
        }
        let mut fzf = m * G * cos_g * b / l - (m * ax + wg) * hcg / l;
        let mut fzr = m * G * cos_g * a / l + (m * ax + wg) * hcg / l;
        let fl = 0.1 * m * G;
        if fzf < fl { fzf = fl; }
        if fzr < fl { fzr = fl; }
        let fd = if locked || u >= p.vmax { 0.0 } else { g_ * jm::min(p.f0, p.pmax / jm::max(u, 0.1)) };
        // idle creep (SimOpts::idle_creep): the drive force never drops below the automatic's idle creep
        let fd = if creep && !locked && u < IDLE_CREEP_U {
            let fi = IDLE_CREEP_ROLL * p.crr * crr_mul * m * G * cos_g * (1.0 - jm::max(0.0, u) / IDLE_CREEP_U);
            if fi > fd { fi } else { fd }
        } else { fd };
        let moving = u > 0.02;
        let fb = (if locked { 1.0 } else { b_ }) * m * bd;
        let (mut fxf, mut fxr) = match p.drive { Drive::Front => (fd, 0.0), Drive::Rear => (0.0, fd), Drive::All => { let f = fd * fzf / (fzf + fzr); (f, fd - f) } };
        let mut share_f = p.brake_front;
        if !locked { let ideal = fzf / (fzf + fzr) + EBD_MARGIN; if ideal > share_f { share_f = if ideal > 0.95 { 0.95 } else { ideal }; } }
        let bf = fb * share_f;
        let br = fb - bf;
        // hill_rollback: a driving car rolling backwards is braked forwards; at rest the brake also holds it against the
        // grade (the JS: brakes only ever act against forward motion)
        let back = rollb && !locked && u < -0.02;
        if back { fxf += bf; fxr += br; }
        else if moving { fxf -= bf; fxr -= br; }
        else {
            let hold = jm::min(fb, jm::max(0.0, fd - wg)); let hf = jm::min(bf, hold * share_f); fxf -= hf; fxr -= jm::min(br, hold - hf);
            if rollb && !locked { let hb = jm::min(fb, jm::max(0.0, wg - fd)); let hf = jm::min(bf, hb * share_f); fxf += hf; fxr += jm::min(br, hb - hf); }
        }
        let ff = mu * fzf;
        let fr = mu * fzr;
        let xf = if locked { 1.0 } else { 0.98 };
        let xr = xf;
        let xb = if locked { 1.0 } else { ABS_LONG };
        if fxf > xf * ff { fxf = xf * ff; } else if fxf < -xb * ff { fxf = -xb * ff; }
        if fxr > xr * fr { fxr = xr * fr; } else if fxr < -xb * fr { fxr = -xb * fr; }
        if !locked && fxr < 0.0 {
            let uf0 = if u > 1.0 { u } else { 1.0 };
            let ar0 = -jm::atan((v - b * r) / uf0);
            let lam_r = if u >= KIN_BLEND_HI { 1.0 } else if u <= KIN_BLEND_LO { 0.0 } else { (u - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO) };
            let yr0 = lam_r * fr * jm::tanh(cr * ar0 / fr);
            let lim = jm::sqrt(jm::max(0.0, xb * xb * fr * fr - yr0 * yr0));
            if fxr < -lim { fxr = -lim; }
        }
        let (mut fyf, mut fyr);
        let uf = if u > 1.0 { u } else { 1.0 };
        let yf_max = jm::sqrt(jm::max(ff * ff - fxf * fxf, 0.0)) + if locked { 0.0 } else { 1e-3 };
        let yr_max = jm::sqrt(jm::max(fr * fr - fxr * fxr, 0.0)) + if locked { 0.0 } else { 1e-3 };
        if locked {
            let (vfx, vfy, vrx, vry) = (u, v + a * r, u, v - b * r);
            let sf = jm::hypot(vfx, vfy);
            let sr2 = jm::hypot(vrx, vry);
            let k = k_slide;
            let push_f = jm::max(0.0, -wg) * fzf / (fzf + fzr);
            let push_r = jm::max(0.0, -wg) - push_f;
            if sf > 0.01 { fxf = -k * ff * vfx / sf; fyf = -k * ff * vfy / sf; } else { fxf = -jm::min(ff, push_f); fyf = 0.0; }
            if sr2 > 0.01 { fxr = -k * fr * vrx / sr2; fyr = -k * fr * vry / sr2; } else { fxr = -jm::min(fr, push_r); fyr = 0.0; }
        } else {
            let af = steer - jm::atan((v + a * r) / uf);
            let ar = -jm::atan((v - b * r) / uf);
            let lam_f = if u >= KIN_BLEND_HI { 1.0 } else if u <= KIN_BLEND_LO { 0.0 } else { (u - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO) };
            // lam_f == 0 (crawling): 0 * yMax * tanh(x) is a signed zero with the sign of x (tanh is finite, odd and
            // keeps the sign of zero), so the tanh can be skipped without changing a bit
            if lam_f == 0.0 && yf_max > 0.0 && yr_max > 0.0 {
                let zf = cf * af / yf_max; let zr = cr * ar / yr_max;
                fyf = if zf.is_sign_negative() { -0.0 } else { 0.0 };
                fyr = if zr.is_sign_negative() { -0.0 } else { 0.0 };
                if zf.is_nan() || zr.is_nan() { fyf = lam_f * yf_max * jm::tanh(zf); fyr = lam_f * yr_max * jm::tanh(zr); }
            } else {
                fyf = lam_f * yf_max * jm::tanh(cf * af / yf_max);
                fyr = lam_f * yr_max * jm::tanh(cr * ar / yr_max);
            }
        }
        let res = resistance(p, u, grade, crr_mul);
        // hill_rollback: rolling resistance opposes backward rolling too (resistance() has it for u > 0.05 only)
        let res = if back && u < -0.05 { res - p.crr * crr_mul * m * G } else { res };
        if steer.to_bits() != m_steer.to_bits() { m_steer = steer; m_cd = jm::cos(steer); m_sd = jm::sin(steer); }
        let (cd, sd) = (m_cd, m_sd);
        let fx_body = fxf * cd - fyf * sd + fxr - res;
        let fy_body = fxf * sd + fyf * cd + fyr;
        // impact_reverse wrecks (locked, u may be < 0): the body-frame rotation terms (v r, -u r) are applied as an exact
        // rotation. Explicit Euler grows |(u, v)| by sqrt(1 + (r h)^2) per sub-step, which a spinning wreck (r ~ 15 rad/s)
        // turns into a runaway; the JS's u >= 0 clamp used to hide it.
        let exact = rev && locked;
        let (ur, vr) = if exact { let (c, sn) = (jm::cos(r * h), jm::sin(r * h)); (u * c + v * sn, -u * sn + v * c) } else { (u, v) };
        let du = if exact { 0.0 } else { (fx_body / m + v * r) * h };
        let mut un = if exact { ur + fx_body / m * h } else { u + du };
        // impact_reverse: a wreck (locked wheels) may slide backwards after an impact; else no reverse (the JS)
        if rollb && !locked {
            // hill_rollback: a driving car rolls back only when gravity beats drive + brake + rolling resistance at
            // rest; a backward-rolling car that the forces bring to a stop stops (no reverse drive)
            let roll0 = p.crr * crr_mul * m * G * cos_g;
            if u >= -0.02 && un < 0.0 && wg <= fd + fb + roll0 { un = 0.0; }
            if u < -0.02 && un > 0.0 { un = 0.0; }
        } else if un < 0.0 && !(rev && locked) { un = 0.0; }
        // brake hold (SimOpts::brake_hold): JS brakes only cancel the drive below 0.02 m/s, so a residual speed creeps
        if hold && !locked && u <= 0.05 {
            // the brake plus rolling resistance (which the JS switches off below 0.05 m/s) hold the car at 0 whenever
            // they exceed the drive minus grade force (static friction)
            let roll = p.crr * crr_mul * m * G * cos_g;
            if fb + roll >= jm::abs(fd - wg) && un <= 0.05 { un = 0.0; }
        }
        // static hold (SimOpts::static_hold): below 0.05 m/s the brake holds the car at 0 when the brake force alone
        // beats drive minus grade; with no drive and no brake, rolling resistance holds it against a grade it exceeds
        if shold && !locked && u <= 0.05 && u >= -0.05 && un <= 0.05 && un >= -0.05 {
            let roll = p.crr * crr_mul * m * G * cos_g;
            if (b_ > 0.0 && fb >= jm::abs(fd - wg)) || (fd == 0.0 && b_ == 0.0 && jm::abs(wg) <= roll) { un = 0.0; }
        }
        let mut vn = if exact { vr + fy_body / m * h } else { v + (fy_body / m - u * r) * h };
        let mut rn = r + ((a * (fxf * sd + fyf * cd) - b * fyr) / iz) * h;
        if un < KIN_BLEND_HI && !locked {
            let lam = if un <= KIN_BLEND_LO { 0.0 } else { (un - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO) };
            let rk = un * jm::tan(steer) / l;
            let vk = rk * b;
            rn = lam * rn + (1.0 - lam) * rk;
            vn = lam * vn + (1.0 - lam) * vk;
        }
        if locked && (if rev { jm::abs(un) } else { un }) < 0.05 && jm::abs(vn) < 0.05 { un = 0.0; vn = 0.0; rn *= 0.5; if jm::abs(rn) < 0.01 { rn = 0.0; } }
        ax = (un - u) / h;
        u = un; v = vn; r = rn;
        hr += r * h;
        if hr > jm::PI { hr -= 2.0 * jm::PI; } else if hr < -jm::PI { hr += 2.0 * jm::PI; }
        if hr.to_bits() != m_hr.to_bits() { m_hr = hr; m_ch = jm::cos(hr); m_sh = jm::sin(hr); }
        let (ch, sh) = (m_ch, m_sh);
        lat += (u * sh + v * ch) * h;
        s += (u * ch - v * sh) * h * cos_g;
        let tract = jm::max(0.0, match p.drive { Drive::Front => jm::max(0.0, fxf), Drive::Rear => jm::max(0.0, fxr), Drive::All => jm::max(0.0, fxf) + jm::max(0.0, fxr) }) * u;
        power = if locked { 0.0 } else { (if g_ > 0.0 { tract / 0.88 } else { 0.0 }) + p.accessory_w };
        fuel += (if locked { 0.0 } else { fuel_flow(p, power) }) * h;
            self.u = u; self.v = v; self.r = r; self.steer = steer; self.ax = ax; self.hr = hr; self.lat = lat; self.s = s; self.sl = sl; self.fuel = fuel; self.power = power; self.m_steer = m_steer; self.m_cd = m_cd; self.m_sd = m_sd; self.m_hr = m_hr; self.m_ch = m_ch; self.m_sh = m_sh;
    }

    #[inline(always)]
    fn finish(&self, st: &mut DynState) {
        let Dyn { u, v, r, steer, sl, ax, hr, lat, s, fuel, power, .. } = *self;
        st.u = u; st.v = v; st.r = r; st.steer = steer; st.sl = sl; st.ax = ax; st.hr = hr; st.lat = lat; st.s = s;
        st.fuel_l += fuel; st.power = power;
    }
}

// ------------------------------------------------------------------ collide.js
pub const RESTITUTION: f64 = 0.2;
pub const FRICTION: f64 = 0.5;
pub const WALL_RESTITUTION: f64 = 0.15;
pub const WALL_FRICTION: f64 = 0.4;
pub const SLOP: f64 = 0.01;
pub const CORRECTION: f64 = 0.8;

/// Rigid body {x, z, yaw, vx, vz, w, hl, hw, invMass, invInertia} (+ _x/_z: position before the contact).
#[derive(Clone, Copy, Debug, Default)]
pub struct Body { pub x: f64, pub z: f64, pub yaw: f64, pub vx: f64, pub vz: f64, pub w: f64, pub hl: f64, pub hw: f64, pub inv_mass: f64, pub inv_inertia: f64, pub x0: f64, pub z0: f64 }

#[derive(Clone, Copy, Debug, Default)]
pub struct Contact { pub depth: f64, pub nx: f64, pub nz: f64, pub px: f64, pub pz: f64 }

#[derive(Clone, Copy, Debug, Default)]
pub struct Impulse { pub jn: f64, pub jt: f64, pub dva: f64, pub dvb: f64, pub closing: f64 }

#[inline]
fn axes(yaw: f64) -> [f64; 4] { let c = jm::cos(yaw); let s = jm::sin(yaw); [c, s, -s, c] }
#[inline]
fn radius(hl: f64, hw: f64, ax: &[f64; 4], ux: f64, uz: f64) -> f64 { jm::abs(ax[0] * ux + ax[1] * uz) * hl + jm::abs(ax[2] * ux + ax[3] * uz) * hw }

/// obbContact on poses (x, z, yaw, hl, hw).
pub fn obb_contact(a: (f64, f64, f64, f64, f64), b: (f64, f64, f64, f64, f64)) -> Option<Contact> {
    let aa = axes(a.2);
    let bb = axes(b.2);
    let dx = b.0 - a.0;
    let dz = b.1 - a.1;
    let mut best = f64::INFINITY;
    let (mut bnx, mut bnz) = (0.0, 0.0);
    let mut from_a = true;
    for k in 0..4 {
        let src = if k < 2 { &aa } else { &bb };
        let ux = src[(k & 1) * 2];
        let uz = src[(k & 1) * 2 + 1];
        let d = dx * ux + dz * uz;
        let o = radius(a.3, a.4, &aa, ux, uz) + radius(b.3, b.4, &bb, ux, uz) - jm::abs(d);
        if o <= 0.0 { return None; }
        if o < best { best = o; let sg = if d < 0.0 { -1.0 } else { 1.0 }; bnx = ux * sg; bnz = uz * sg; from_a = k < 2; }
    }
    let (inc, ax) = if from_a { (b, &bb) } else { (a, &aa) };
    let sgn = if from_a { -1.0 } else { 1.0 };
    let mut best_p = f64::INFINITY;
    let (mut px, mut pz) = (0.0, 0.0);
    let mut second_p = f64::INFINITY;
    let (mut qx, mut qz) = (0.0, 0.0);
    for sx in [-1.0f64, 1.0] {
        for sz in [-1.0f64, 1.0] {
            let vx = inc.0 + ax[0] * sx * inc.3 + ax[2] * sz * inc.4;
            let vz = inc.1 + ax[1] * sx * inc.3 + ax[3] * sz * inc.4;
            let p = sgn * (vx * bnx + vz * bnz) * -1.0;
            if p < best_p { second_p = best_p; qx = px; qz = pz; best_p = p; px = vx; pz = vz; }
            else if p < second_p { second_p = p; qx = vx; qz = vz; }
        }
    }
    if second_p - best_p < 0.05 { px = 0.5 * (px + qx); pz = 0.5 * (pz + qz); }
    Some(Contact { depth: best, nx: bnx, nz: bnz, px, pz })
}

pub fn resolve_contact(a: &mut Body, b: &mut Body, c: &Contact, e: f64, mu: f64) -> Impulse {
    let rax = c.px - a.x; let raz = c.pz - a.z; let rbx = c.px - b.x; let rbz = c.pz - b.z;
    let vax = a.vx - a.w * raz; let vaz = a.vz + a.w * rax;
    let vbx = b.vx - b.w * rbz; let vbz = b.vz + b.w * rbx;
    let rvx = vbx - vax; let rvz = vbz - vaz;
    let vn = rvx * c.nx + rvz * c.nz;
    let mut out = Impulse { closing: -vn, ..Default::default() };
    if vn >= 0.0 { return out; }
    let ra_n = rax * c.nz - raz * c.nx; let rb_n = rbx * c.nz - rbz * c.nx;
    let k_n = a.inv_mass + b.inv_mass + ra_n * ra_n * a.inv_inertia + rb_n * rb_n * b.inv_inertia;
    if k_n <= 0.0 { return out; }
    let jn = -(1.0 + e) * vn / k_n;
    let mut tx = rvx - vn * c.nx; let mut tz = rvz - vn * c.nz;
    let tl = jm::hypot(tx, tz);
    let mut jt = 0.0;
    if tl > 1e-6 {
        tx /= tl; tz /= tl;
        let ra_t = rax * tz - raz * tx; let rb_t = rbx * tz - rbz * tx;
        let k_t = a.inv_mass + b.inv_mass + ra_t * ra_t * a.inv_inertia + rb_t * rb_t * b.inv_inertia;
        jt = -(rvx * tx + rvz * tz) / k_t;
        let lim = mu * jn; if jt > lim { jt = lim; } else if jt < -lim { jt = -lim; }
    } else { tx = 0.0; tz = 0.0; }
    let pxi = jn * c.nx + jt * tx; let pzi = jn * c.nz + jt * tz;
    let (ax0, az0, bx0, bz0) = (a.vx, a.vz, b.vx, b.vz);
    a.vx -= pxi * a.inv_mass; a.vz -= pzi * a.inv_mass; a.w -= (rax * pzi - raz * pxi) * a.inv_inertia;
    b.vx += pxi * b.inv_mass; b.vz += pzi * b.inv_mass; b.w += (rbx * pzi - rbz * pxi) * b.inv_inertia;
    out.jn = jn; out.jt = jt;
    out.dva = jm::hypot(a.vx - ax0, a.vz - az0); out.dvb = jm::hypot(b.vx - bx0, b.vz - bz0);
    out
}

pub fn separate(a: &mut Body, b: &mut Body, c: &Contact) {
    let k = a.inv_mass + b.inv_mass;
    if k <= 0.0 { return; }
    let corr = jm::max(0.0, c.depth - SLOP) * CORRECTION / k;
    a.x += -c.nx * corr * a.inv_mass; a.z += -c.nz * corr * a.inv_mass;
    b.x += c.nx * corr * b.inv_mass; b.z += c.nz * corr * b.inv_mass;
}

#[inline]
fn lerp_angle(a: f64, b: f64, t: f64) -> f64 { let mut d = b - a; while d > jm::PI { d -= 2.0 * jm::PI; } while d < -jm::PI { d += 2.0 * jm::PI; } a + d * t }

/// sweptContact(A0, A, B0, B, out, 0.5, ta, tb): returns (t, contact, pose of A at t, pose of B at t) or None.
/// Poses are (x, z, yaw).
pub fn swept_contact(a0: (f64, f64, f64), a: &Body, b0: (f64, f64, f64), b: &Body, max_step: f64) -> Option<(f64, Contact, (f64, f64), (f64, f64))> {
    let rel = jm::hypot((a.x - a0.0) - (b.x - b0.0), (a.z - a0.1) - (b.z - b0.1));
    let n = jm::max(1.0, jm::ceil(rel / max_step));
    let ni = n as i64;
    for k in 1..=ni {
        let t = k as f64 / n;
        let tax = a0.0 + (a.x - a0.0) * t; let taz = a0.1 + (a.z - a0.1) * t; let tay = lerp_angle(a0.2, a.yaw, t);
        let tbx = b0.0 + (b.x - b0.0) * t; let tbz = b0.1 + (b.z - b0.1) * t; let tby = lerp_angle(b0.2, b.yaw, t);
        if let Some(c) = obb_contact((tax, taz, tay, a.hl, a.hw), (tbx, tbz, tby, b.hl, b.hw)) { return Some((t, c, (tax, taz), (tbx, tbz))); }
    }
    None
}
