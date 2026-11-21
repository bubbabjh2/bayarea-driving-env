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
    for _ in 0..n {
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
        let moving = u > 0.02;
        let fb = (if locked { 1.0 } else { b_ }) * m * p.brake_decel;
        let (mut fxf, mut fxr) = match p.drive { Drive::Front => (fd, 0.0), Drive::Rear => (0.0, fd), Drive::All => { let f = fd * fzf / (fzf + fzr); (f, fd - f) } };
        let mut share_f = p.brake_front;
        if !locked { let ideal = fzf / (fzf + fzr) + EBD_MARGIN; if ideal > share_f { share_f = if ideal > 0.95 { 0.95 } else { ideal }; } }
        let bf = fb * share_f;
        let br = fb - bf;
        if moving { fxf -= bf; fxr -= br; }
        else { let hold = jm::min(fb, jm::max(0.0, fd - wg)); let hf = jm::min(bf, hold * share_f); fxf -= hf; fxr -= jm::min(br, hold - hf); }
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
            fyf = lam_f * yf_max * jm::tanh(cf * af / yf_max);
            fyr = lam_f * yr_max * jm::tanh(cr * ar / yr_max);
        }
        let res = resistance(p, u, grade, crr_mul);
        let cd = jm::cos(steer);
        let sd = jm::sin(steer);
        let fx_body = fxf * cd - fyf * sd + fxr - res;
        let fy_body = fxf * sd + fyf * cd + fyr;
        let du = (fx_body / m + v * r) * h;
        let mut un = u + du;
        if un < 0.0 { un = 0.0; }
        let mut vn = v + (fy_body / m - u * r) * h;
        let mut rn = r + ((a * (fxf * sd + fyf * cd) - b * fyr) / iz) * h;
        if un < KIN_BLEND_HI && !locked {
            let lam = if un <= KIN_BLEND_LO { 0.0 } else { (un - KIN_BLEND_LO) / (KIN_BLEND_HI - KIN_BLEND_LO) };
            let rk = un * jm::tan(steer) / l;
            let vk = rk * b;
            rn = lam * rn + (1.0 - lam) * rk;
            vn = lam * vn + (1.0 - lam) * vk;
        }
        if locked && un < 0.05 && jm::abs(vn) < 0.05 { un = 0.0; vn = 0.0; rn *= 0.5; if jm::abs(rn) < 0.01 { rn = 0.0; } }
        ax = (un - u) / h;
        u = un; v = vn; r = rn;
        hr += r * h;
        if hr > jm::PI { hr -= 2.0 * jm::PI; } else if hr < -jm::PI { hr += 2.0 * jm::PI; }
        let ch = jm::cos(hr);
        let sh = jm::sin(hr);
        lat += (u * sh + v * ch) * h;
        s += (u * ch - v * sh) * h * cos_g;
        let tract = jm::max(0.0, match p.drive { Drive::Front => jm::max(0.0, fxf), Drive::Rear => jm::max(0.0, fxr), Drive::All => jm::max(0.0, fxf) + jm::max(0.0, fxr) }) * u;
        power = if locked { 0.0 } else { (if g_ > 0.0 { tract / 0.88 } else { 0.0 }) + p.accessory_w };
        fuel += (if locked { 0.0 } else { fuel_flow(p, power) }) * h;
    }
    st.u = u; st.v = v; st.r = r; st.steer = steer; st.sl = sl; st.ax = ax; st.hr = hr; st.lat = lat; st.s = s;
    st.fuel_l += fuel; st.power = power;
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
